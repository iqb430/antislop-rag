# Anti-Slop RAG Pipeline 💀

A zero-framework, deterministic Retrieval-Augmented Generation (RAG) system engineered for brutal efficiency and extreme logical constraint. Built natively in Python, this architecture completely bypasses heavy orchestration layers (like LangChain) to retain absolute granular control over token generation, semantic chunking algorithms, and inference latency.

## Why "Anti-Slop"?
Modern AI outputs are plagued with "slop"—fluffy, ungrounded, and hallucinated prose ("delve", "testament to", "seamless"). This project acts as a strict cognitive cage. The RAG pipeline employs a multi-agent methodology where retrieved context acts as an unyielding boundary. The LLM is structurally forbidden from answering using its latent knowledge, enforcing a 100% faithfulness rate to the inserted vector space.

## Enterprise-Grade Architecture
Unlike typical wrapper scripts, this pipeline is structured for Data Science validation and production deployment:

- **Algorithm-First Ingestion (`ingestor.py`):** Custom semantic boundaries (Natural Language sentence parsing) rather than blind N-character chunking, preventing context fracturing.
- **Local Embedded Vector Space (`search.py`):** Utilizes `ChromaDB` for high-dimensional cosine similarity matching, operating entirely in-memory or on local disk (Zero API egress fees).
- **Deterministic Interrogation (`interrogator.py`):** Synchronous REST bindings directly to a local engine (Ollama). Generation temperature is aggressively minimized to enforce factual rigidity.
- **LLM-as-a-Judge Evaluation (`evaluator.py`):** Automated quantitative benchmarking. We treat RAG evaluation as a CI/CD metric, dynamically measuring *Faithfulness Score* and *Hallucination Rate* across an evaluation dataset.

## The Metrics That Matter
A RAG system is useless if it cannot be objectively measured. We implement an internal evaluation suite focusing on:
1. **Context Relevancy (Precision@k):** Are the vectors we retrieve actually mapped to the query?
2. **Generation Faithfulness:** Did the LLM fabricate data outside the injected context window? (Target: 1.0)
3. **Inference Latency:** Raw compute time from query embedding to first-token generation.

## Usage
*Dependencies: Python 3.10+, PyMuPDF, ChromaDB, Local Ollama Instance.*

```bash
# 1. Ingest Unstructured Data
python3 ingestor.py ./docs/source_material.pdf

# 2. Run Synthetic Benchmarks
python3 evaluator.py

# 3. Interrogate the System
python3 interrogator.py "Extract the core thesis."
```
