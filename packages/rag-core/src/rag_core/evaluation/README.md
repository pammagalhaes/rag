# RAGAS Evaluation Module

Este módulo fornece ferramentas para avaliar a qualidade das respostas geradas pelo sistema RAG usando a biblioteca **RAGAS** (Retrieval-Augmented Generation Assessment).

## Métricas Avaliadas

- **Faithfulness**: Mede quão fielmente a resposta é baseada no contexto fornecido (0-1, quanto maior melhor)
- **Answer Relevancy**: Mede a relevância da resposta em relação à pergunta feita (0-1, quanto maior melhor)
- **Context Precision**: Mede a precisão do contexto recuperado — quanto dos documentos recuperados é realmente relevante (0-1)
- **Context Recall**: Mede a completude do contexto — quanto de informação relevante foi capturada nos documentos recuperados (0-1)

## Instalação

```bash
pip install ragas
```

## Uso

### 1. Avaliar uma resposta individual

```python
from rag_core.evaluation import RAGEvaluator

evaluator = RAGEvaluator()

result = evaluator.evaluate_response(
    question="O que é um modelo de classificação?",
    answer="Um modelo de classificação é um algoritmo de aprendizado de máquina que...",
    contexts=[
        "Texto do documento 1 com informação relevante...",
        "Texto do documento 2 com informação relevante...",
    ]
)

print(f"Faithfulness: {result.faithfulness}")
print(f"Answer Relevancy: {result.answer_relevancy}")
```

### 2. Avaliar múltiplas respostas

```python
qa_pairs = [
    {
        "question": "O que é classificação?",
        "answer": "Classificação é um tipo de...",
        "contexts": ["contexto 1", "contexto 2"],
    },
    {
        "question": "Quais são os tipos de aprendizado?",
        "answer": "Os tipos de aprendizado são...",
        "contexts": ["contexto 3", "contexto 4"],
    },
]

results = evaluator.evaluate_batch(qa_pairs)
evaluator.print_results(results)
evaluator.export_results(results, "results.json")
```

### 3. Script de linha de comando

Execute o script `evaluate.py` para avaliar o RAG com suas perguntas pré-definidas:

```bash
# Com perguntas padrão
python packages/rag-core/scripts/evaluate.py \
    --faiss-index data/index/faiss.index \
    --faiss-meta data/index/faiss_meta.pkl \
    --whoosh-dir data/index/whoosh

# Com perguntas customizadas
python packages/rag-core/scripts/evaluate.py \
    --questions "O que é classificação?" "O que é regressão?" \
    --output data/my_evaluation_results.json
```

## Exemplo de Saída

```
================================================================================
RAGAS Evaluation Results
================================================================================

[Question 1]
Q: O que é um modelo de classificação?
A: Um modelo de classificação é um algoritmo de aprendizado de máquina que...

Metrics:
  - Faithfulness:       0.8234
  - Answer Relevancy:   0.9102
  - Context Precision:  0.8567
  - Context Recall:     0.7234

...
================================================================================
```

## Integração com RAG Service

Para integrar avaliação contínua no serviço RAG (opcional):

```python
from rag_core.evaluation import RAGEvaluator

class RAGService:
    def __init__(self, cfg, enable_evaluation=False):
        # ... existing code ...
        self.evaluator = RAGEvaluator() if enable_evaluation else None
    
    def answer(self, question: str, evaluate=False) -> Dict:
        # Generate answer
        docs = self.retriever.hybrid(question, k=5)
        answer = self._generate_answer(question, docs)
        
        # Optional: Evaluate
        if evaluate and self.evaluator:
            contexts = [doc["text"] for doc in docs]
            eval_result = self.evaluator.evaluate_response(
                question=question,
                answer=answer,
                contexts=contexts,
            )
            return {
                "answer": answer,
                "evaluation": eval_result.to_dict(),
            }
        
        return {"answer": answer}
```

## Limitações e Notas

- RAGAS requer integração com um modelo LLM (OpenAI, etc.) para calcular algumas métricas
- Ground truth é simulado usando a resposta gerada (em produção, use respostas validadas manualmente)
- A avaliação pode ser lenta dependendo do número de respostas e do provedor LLM
