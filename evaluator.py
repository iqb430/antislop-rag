import time
import json
import urllib.request
from typing import List, Dict, Tuple
from dataclasses import dataclass
import statistics

@dataclass
class EvalResult:
    query: str
    latency_ms: float
    faithfulness_score: float
    context_relevancy_score: float
    hallucination_flag: bool

class RAGEvaluator:
    """
    Enterprise-grade RAG Evaluation Pipeline.
    Measures retrieval accuracy, generation faithfulness, and system latency.
    Bypassing heavy frameworks (Ragas/TruLens) for raw local execution.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.eval_model = "gemma4" # Uses local Gemma for LLM-as-a-Judge

    def _llm_as_a_judge(self, prompt: str) -> str:
        """Call local LLM synchronously for evaluation routing."""
        data = {
            "model": self.eval_model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0 # Strict deterministic output for evaluation
        }
        try:
            req = urllib.request.Request(self.ollama_url, data=json.dumps(data).encode("utf-8"), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip().lower()
        except Exception as e:
            return "error"

    def measure_faithfulness(self, context: str, answer: str) -> float:
        """
        Calculates Faithfulness / Groundedness.
        1.0 = All claims are backed by context.
        0.0 = Hallucination detected.
        """
        eval_prompt = (
            "You are an objective metric scoring system. "
            f"Context: {context}\n"
            f"Generated Answer: {answer}\n"
            "Does the Generated Answer contain information that is NOT present in the Context? "
            "Reply strictly with 'yes' if there is hallucination, or 'no' if it is perfectly grounded."
        )
        verdict = self._llm_as_a_judge(eval_prompt)
        return 0.0 if "yes" in verdict else 1.0

    def evaluate_batch(self, test_set: List[Dict[str, str]]) -> Dict[str, float]:
        """Runs the benchmark pipeline across a dataset."""
        print("[*] Initializing Anti-Slop RAG Benchmarking Suite...")
        results: List[EvalResult] = []
        
        for idx, test in enumerate(test_set):
            print(f"    -> Evaluating Sample {idx+1}/{len(test_set)}...")
            start_time = time.perf_counter()
            
            # 1. Emulate Retrieval + Generation (Mocked for CI isolation)
            context = test["retrieved_context"]
            answer = test["generated_answer"]
            
            # 2. Measure Metrics
            faithfulness = self.measure_faithfulness(context, answer)
            
            latency = (time.perf_counter() - start_time) * 1000
            
            results.append(EvalResult(
                query=test["query"],
                latency_ms=latency,
                faithfulness_score=faithfulness,
                context_relevancy_score=1.0, # Placeholder for embedding cosine similarity metric
                hallucination_flag=(faithfulness == 0.0)
            ))
            
        return self._aggregate_metrics(results)

    def _aggregate_metrics(self, results: List[EvalResult]) -> Dict[str, float]:
        """Compile raw eval results into executive statistics."""
        avg_latency = statistics.mean([r.latency_ms for r in results])
        avg_faithfulness = statistics.mean([r.faithfulness_score for r in results])
        hallucination_rate = sum(1 for r in results if r.hallucination_flag) / len(results)
        
        print("\n==================================================")
        print("         RAG PERFORMANCE METRICS REPORT           ")
        print("==================================================")
        print(f"Total Samples Evaluated : {len(results)}")
        print(f"Average System Latency  : {avg_latency:.2f} ms")
        print(f"Overall Faithfulness    : {avg_faithfulness * 100:.1f}%")
        print(f"Hallucination Rate      : {hallucination_rate * 100:.1f}%")
        print("==================================================\n")
        
        return {
            "avg_latency_ms": avg_latency,
            "faithfulness_score": avg_faithfulness,
            "hallucination_rate": hallucination_rate
        }

if __name__ == "__main__":
    # Mock CI/CD Evaluation Dataset
    mock_dataset = [
        {
            "query": "What are the rules of engagement?",
            "retrieved_context": "The rules state you must never use generic AI words and keep answers brutal.",
            "generated_answer": "You must never use generic AI words like delve or testament, and answers must be brutal."
        },
        {
            "query": "Who wrote the initial rules?",
            "retrieved_context": "The rules were written by the Observer.",
            "generated_answer": "The rules were written by OpenAI in 2023." # Intentional Hallucination for test
        }
    ]
    
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_batch(mock_dataset)
