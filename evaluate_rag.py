import os
import json
import warnings
from datasets import Dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    RAGAS_AVAILABLE = True
except Exception as err:
    print(f"Ragas is not available or failed to load: {err}")
    RAGAS_AVAILABLE = False


def run_evaluation(eval_dataset_path="eval_data.json"):
    print("[*] Starting Automated RAG Evaluation with Ragas...")

    if not RAGAS_AVAILABLE:
        print("[!] Ragas evaluation skipped due to missing or incompatible dependencies.")
        with open("eval_results.csv", "w", encoding="utf-8") as f:
            f.write("question,status,error\n")
            f.write('"N/A",skipped,"Ragas dependency not loaded"\n')
        return

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

        result.to_pandas().to_csv("eval_results.csv", index=False)
        print("[+] Results saved to eval_results.csv")
    except Exception as e:
        print(f"[!] Evaluation failed (possibly missing API keys or local LLM setup for Ragas): {e}")
        with open("eval_results.csv", "w", encoding="utf-8") as f:
            f.write("question,status,error\n")
            q = data.get("question", [""])[0].replace('"', "'")
            err_msg = str(e).replace('"', "'")
            f.write(f'\"{q}\",skipped,\"{err_msg}\"\n')


if __name__ == "__main__":
    run_evaluation()
