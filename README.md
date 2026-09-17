# ANTISLOP RAG (Existential Document Parser)

A completely local, bloated-framework-free Retrieval-Augmented Generation (RAG) pipeline. Built to extract raw facts from PDFs and brutally compress them into actionable philosophy, bypassing modern AI conversational "slop".

## Core Principles
1. **Zero LangChain:** Everything is written in raw Python (`urllib`, `chromadb`, `PyMuPDF`). Total transparency of the vector and context injection flow.
2. **Semantic Chunking:** Documents are split intelligently at the end of thoughts (paragraphs/periods), not blindly sliced at exact character limits.
3. **Local First:** 
   - Embedding relies on `sentence-transformers` locally.
   - Vector Storage relies on local `ChromaDB`.
   - Inference relies on local `Ollama` (LLaMA3/Mistral) restricted to a `0.2` temperature.
4. **Anti-Halucination Protocol:** The system prompt forces the model to insult the user if the answer is not present in the ingested document, refusing to guess.

## Architecture

- `ingestor.py` - Parses PDFs cleanly and outputs semantic mental chunks.
- `builder.py` - Embeds chunks into `chroma_db` using robust distance matrices.
- `search.py` - Direct CLI script to retrieve Cosine Similarity scores.
- `interrogator.py` - The final terminal UI. Injects retrieved contexts into a locked-down LLaMA prompt.

## Setup Instructions

1. Activate your virtual environment: 
```bash
source venv/bin/activate
pip install -r requirements.txt
```
2. Feed your document: 
```bash
python builder.py "path/to/your/document.pdf"
```
3. Interrogate the system: 
```bash
python interrogator.py "Your existential question here"
```

---
*Built for the IQBALOG SYS // 01 Ecosystem.*
