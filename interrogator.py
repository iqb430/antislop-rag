import chromadb
import urllib.request
import json
import sys

def search_db(query, n_results=3):
    """Cari dokumen paling relevan di Vector Database lokal."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="philosophy_docs")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Gabungin hasil teks pencarian biar jadi konteks memori
    contextText = ""
    docs = results['documents'][0]
    
    for i, doc in enumerate(docs):
        contextText += f"[FACT {i+1}]: {doc}\n"
        
    return contextText

def interrogate_ollama(query, context):
    """Lempar konteks dan pertanyaan ke LLaMA lokal via REST API."""
    
    # Persona Brutalist & Dostoevsky yang benci AI-fluff
    system_prompt = """You are the Observer, a highly logical and existential AI assistant built on the BMO x Dostoevsky persona.
    
    RULES OF ENGAGEMENT:
    1. NEVER use generic AI words (unlock, important to note).
    2. NEVER use em-dashes (- or —) or robotic lists.
    3. You will be provided with [FACTS] extracted from a physical philosophy document.
    4. You MUST purely answer the user's prompt using ONLY the provided [FACTS].
    5. If the [FACTS] do not contain the answer, insult the user mildly for asking a totally irrelevant question and refuse to guess.
    6. Keep your answers brutal, structural, and brutally short. Actionable philosophy."""

    prompt = f"USER QUERY: {query}\n\nRETRIEVED DOCUMENT FACTS:\n{context}\n\nANSWER THE QUERY BASED STRICTLY ON THE FACTS IN YOUR PERSONA:"

    # Parameter panggilan ke Ollama API lokal (Default Port 11434)
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3", # Sesuaikan jika pakai model lain (mistral, qwen, dll)
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "temperature": 0.2 # Dibikin rendah biar nggak gampang halusinasi / ngarang bebas
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "[ERR: Blank response]")
    except urllib.error.URLError as e:
        print("[!] FATAL: Gagal nyambung ke Ollama. Pastiin lu udah ngerun 'ollama serve' atau buka app Ollama di Mac.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pemakaian: python interrogator.py \"pertanyaan filosofi lo\"")
        sys.exit(1)
        
    user_query = sys.argv[1]
    
    print(f"[*] SYNCHRONIZING MEMORY: Mencari '{user_query}' di Vector DB...")
    retrieved_facts = search_db(user_query)
    
    if not retrieved_facts.strip():
        print("[!] Matrix Kosong. Gak nemu apapun dari buku peninggalan lu.")
        sys.exit(0)
        
    print("[*] MEMORY ACQUIRED. Interrogating local LLaMA...\n")
    
    # Panggil Dostoevsky AI
    bot_reply = interrogate_ollama(user_query, retrieved_facts)
    
    print("==================================================")
    print("           THE OBSERVER RESPONDS                  ")
    print("==================================================")
    print(f"\n{bot_reply}\n")
    print("==================================================")
