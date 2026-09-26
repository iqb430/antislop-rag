import os
import json
from datasets import Dataset
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
except ImportError:
    print("Ragas is not installed. Please install it using: pip install ragas datasets")
    exit(1)

def run_evaluation(eval_dataset_path="eval_data.json"):
    """
    Automated LLM evaluation pipeline using Ragas.
    This evaluates the Retrieval-Augmented Generation (RAG) pipeline
    for faithfulness, answer relevancy, context precision, and context recall.
    """
    print("[*] Starting Automated RAG Evaluation with Ragas...")
    
    # Load evaluation dataset
    if not os.path.exists(eval_dataset_path):
        print(f"[!] Evaluation dataset '{eval_dataset_path}' not found.")
        print("[*] Creating a dummy evaluation dataset for demonstration...")
        dummy_data = {
            "question": ["What is the core principle of Antislop RAG?"],
            "answer": ["The core principle is Zero LangChain, Semantic Chunking, Local First, and Anti-Halucination."],
            "contexts": [["1. Zero LangChain: Everything is written in raw Python. 2. Semantic Chunking. 3. Local First. 4. Anti-Halucination Protocol."]],
            "ground_truth": ["Antislop RAG relies on Zero LangChain, Semantic Chunking, Local First operation, and an Anti-Halucination Protocol."]
        }
        with open(eval_dataset_path, "w") as f:
            json.dump(dummy_data, f)
            
    with open(eval_dataset_path, "r") as f:
        data = json.load(f)
        
    dataset = Dataset.from_dict(data)
    
    print("[*] Running evaluation metrics...")
    # NOTE: In a real environment, you need OPENAI_API_KEY or a local LLM configured for Ragas.
    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        print("\n[+] Evaluation Results:")
        print(result)
        
        # Save results
        result.to_pandas().to_csv("eval_results.csv", index=False)
        print("[+] Results saved to eval_results.csv")
    except Exception as e:
        print(f"[!] Evaluation failed (possibly missing API keys or local LLM setup for Ragas): {e}")
        print("[*] Ensure OPENAI_API_KEY is set or configure Ragas to use a local LLM.")

if __name__ == "__main__":
    run_evaluation()
