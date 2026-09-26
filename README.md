# Anti-Slop RAG Pipeline 💀

A completely local, bloated-framework-free Retrieval-Augmented Generation (RAG) pipeline. Built to extract raw facts from PDFs and brutally compress them into actionable philosophy, bypassing modern conversational filler.

## Why "Anti-Slop"?
Modern AI outputs are plagued with "slop"—fluffy, ungrounded, and hallucinated prose ("delve", "testament to", "seamless"). This project acts as a strict cognitive cage. The RAG pipeline employs a multi-agent methodology where retrieved context acts as an unyielding boundary. The LLM is structurally forbidden from answering using its latent knowledge, enforcing a 100% faithfulness rate to the inserted vector space.

## Enterprise-Grade Architecture
Unlike typical wrapper scripts, this pipeline is structured for Data Science validation and production deployment:

- `ingestor.py` - Parses PDFs cleanly and outputs semantic mental chunks.
- `builder.py` - Embeds chunks into `chroma_db` using standard distance matrices.
- `search.py` - Direct CLI script to retrieve Cosine Similarity scores.
- `interrogator.py` - The final terminal UI. Injects retrieved contexts into a locked-down LLaMA prompt.

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
