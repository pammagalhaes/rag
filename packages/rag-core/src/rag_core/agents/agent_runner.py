import json
from rag_core.llm.base import ModelClient



class AgentRunner:
    """
    ReAct Agent: Thought → Action → Observation → (loop) → Final Answer.
    """
    def __init__(self, model_client: ModelClient, retriever, prompt_templates, max_steps: int = 5):
        self.model = model_client
        self.retriever = retriever
        self.max_steps = max_steps
        self.qa_prompt = prompt_templates["qa_prompt"]
        self.rerank_prompt = prompt_templates["rerank_prompt"]

    # ----------------------------
    # Tools used by ReAct
    # ----------------------------

    def tool_search(self, query: str, k: int = 5):
        return self.retriever.hybrid(query, k=k)

    def tool_rerank(self, question: str, candidates: list, top_n: int = 5):
        formatted = "\n\n".join(
            f"[{i+1}] {c['text']}" for i, c in enumerate(candidates)
        )
        prompt = self.rerank_prompt.format(
            question=question,
            candidates=formatted
        )
        out = self.model.generate(prompt)
        # modelo retorna os top N textos — simples
        return out

    def tool_finish(self, answer: str):
        return answer

    # -----------------------------
    # Main ReAct loop
    # -----------------------------
    def run(self, question: str) -> str:

        history = ""  # conversation trace

        for step in range(self.max_steps):

            react_prompt = f"""
You are an AI assistant that uses ReAct (Reason + Action).
You have the following tools:

- search: retrieves documents. Input: {{"query": "...", "k": number}}
- rerank: sorts retrieved docs by relevance.
- finish: ends the task.

Follow this loop:
Thought:
Action: <tool>
Action Input: <json>
Observation: <result>

When you know the final answer:
Action: finish
Action Input: {{"answer": "..."}}

Conversation so far:
{history}

User question: {question}

Continue with the next Thought.
"""

            response = self.model.generate(react_prompt, max_tokens=300)

            history += f"\n{response}\n"

            # -----------------------------------------
            # Parse Action + Action Input
            # -----------------------------------------
            lines = [l.strip() for l in response.splitlines()]
            action = None
            action_input = None

            for i, line in enumerate(lines):
                if line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
                if line.startswith("Action Input:"):
                    try:
                        json_str = line.replace("Action Input:", "").strip()
                        action_input = json.loads(json_str)
                    except:
                        action_input = {}

            if action is None:
                return "Agent error: no Action detected."

            # -----------------------------
            # Execute tool
            # -----------------------------
            if action == "search":
                res = self.tool_search(
                    query=action_input.get("query", question),
                    k=action_input.get("k", 5)
                )
                observation = json.dumps(res, ensure_ascii=False)

            elif action == "rerank":
                res = self.tool_rerank(
                    question=question,
                    candidates=action_input.get("candidates", []),
                    top_n=action_input.get("top_n", 5)
                )
                observation = res

            elif action == "finish":
                return action_input.get("answer", "No answer returned.")

            else:
                observation = f"Unknown action: {action}"

            # Add the observation to the history
            history += f"\nObservation: {observation}\n"

        return "Agent stopped: max steps reached without finishing."
