import chromadb
import hashlib
from ingestor import extract_pdf_clean, semantic_chunking
import sys

def build_vector_db(pdf_path):
    print("[*] Tahap 1: Ekstrak PDF Mentah...")
    raw_text = extract_pdf_clean(pdf_path)
    
    print("[*] Tahap 2: Semantic Chunking (Menghindari Pemotongan Paksa)...")
    chunks = semantic_chunking(raw_text)
    
    print(f"[*] Berhasil mengekstrak {len(chunks)} dokumen pikiran.")
    
    print("[*] Tahap 3: Inisialisasi ChromaDB Local dan Sentence-Transformers...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    collection = client.get_or_create_collection(
        name="philosophy_docs",
        metadata={"description": "Antislop RAG knowledge base"}
    )
    
    print("[*] Tahap 4: Menginjeksi Matrix ke Database (Idempotent Hash)...")
    
    def generate_id(text_chunk):
        return hashlib.md5(text_chunk.encode('utf-8')).hexdigest()
        
    ids = [generate_id(chunk) for chunk in chunks]
    metadatas = [{"source": pdf_path, "chunk_hash": generate_id(chunk)} for chunk in chunks]
    
    collection.upsert(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

    
    print(f"[+] SELESAI. {len(chunks)} vektor pikiran udah dikunci ke ./chroma_db/")
    print(f"[*] Coba cari pakai query: python search.py 'makna hidup'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python builder.py <path_to_pdf>")
        sys.exit(1)
        
    build_vector_db(sys.argv[1])
