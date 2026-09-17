import chromadb
import sys

def search_db(query, n_results=3):
    print(f"[*] Mencari '{query}' di dalam Vector Database...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="philosophy_docs")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    print("\n==================================")
    print("      HASIL RETRIEVAL RAG          ")
    print("==================================\n")
    
    docs = results['documents'][0]
    distances = results['distances'][0]
    metadata = results['metadatas'][0]
    
    for i in range(len(docs)):
        print(f"[RELEVANSI: {distances[i]:.4f} | SOURCE: {metadata[i]['chunk_index']}]")
        print(f"{docs[i]}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search.py \"query lo disini\"")
        sys.exit(1)
        
    search_db(sys.argv[1])
