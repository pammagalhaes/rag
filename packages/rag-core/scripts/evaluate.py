#!/usr/bin/env python
"""
Example script: Evaluate RAG responses using RAGAS.
"""

import os
import sys
from typing import List, Dict

# Configure paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from rag_core.llm.transformers_client import TransformersClient
from rag_core.vectorstore.faiss_store import FaissStore
from rag_core.retrieval.hybrid_retriever import HybridRetriever
from rag_core.prompt_engineering.templates import load_templates
from rag_core.evaluation import RAGEvaluator


def generate_rag_responses(
    questions: List[str],
    faiss_index_path: str,
    faiss_meta_path: str,
    whoosh_index_dir: str,
) -> List[Dict]:
    """Generate RAG responses for a list of questions."""
    
    model = TransformersClient()
    faiss = FaissStore(
        dim=1536,
        index_path=faiss_index_path,
        meta_path=faiss_meta_path,
    )
    retriever = HybridRetriever(
        model_client=model,
        faiss_store=faiss,
        whoosh_index_dir=whoosh_index_dir,
    )
    templates = load_templates()
    
    qa_pairs = []
    
    for question in questions:
        # Retrieve context
        docs = retriever.hybrid(question, k=5)
        contexts = [doc.get("text", "") for doc in docs]
        
        # Generate answer
        context_text = "\n\n".join(
            f"Source: {doc.get('source', 'unknown')}\nText: {doc.get('text', '')}"
            for doc in docs
        )
        
        prompt = templates["qa_prompt"].format(
            context=context_text,
            question=question
        )
        answer = model.generate(prompt)
        
        qa_pairs.append({
            "question": question,
            "answer": answer.strip(),
            "contexts": contexts,
        })
        
        print(f"Generated response for: {question}")
    
    return qa_pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG using RAGAS")
    parser.add_argument(
        "--questions",
        nargs="+",
        default=[
            "O que é um modelo de classificação?",
            "Quais são os tipos de aprendizado de máquina?",
        ],
        help="Questions to evaluate",
    )
    parser.add_argument(
        "--faiss-index",
        default="data/index/faiss.index",
        help="Path to FAISS index",
    )
    parser.add_argument(
        "--faiss-meta",
        default="data/index/faiss_meta.pkl",
        help="Path to FAISS metadata",
    )
    parser.add_argument(
        "--whoosh-dir",
        default="data/index/whoosh",
        help="Path to Whoosh index",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation_results.json",
        help="Output file for evaluation results",
    )
    
    args = parser.parse_args()
    
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Generate responses
    print("Generating RAG responses...")
    qa_pairs = generate_rag_responses(
        questions=args.questions,
        faiss_index_path=args.faiss_index,
        faiss_meta_path=args.faiss_meta,
        whoosh_index_dir=args.whoosh_dir,
    )
    
    # Evaluate
    print("\nEvaluating responses with RAGAS...")
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(qa_pairs)
    
    # Print and export
    evaluator.print_results(results)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    evaluator.export_results(results, args.output)


if __name__ == "__main__":
    main()
