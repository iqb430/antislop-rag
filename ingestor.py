import fitz  # PyMuPDF
import re
import sys

def extract_pdf_clean(pdf_path):
    """
    Extract mentahan PDF. 
    Rule: Jangan memanipulasi informasi, ambil mentahan teks visual.
    """
    print(f"[*] Extracting {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        for page in doc:
            text_content += page.get_text("text") + "\n"
        return text_content
    except Exception as e:
        print(f"[!] Gagal ngebaca PDF: {e}")
        sys.exit(1)

def semantic_chunking(text, min_length=200, max_length=800):
    """
    Praktek RAG 01: Semantic Chunking.
    Aturan: Potong kalimat di akhir pikiran (titik/paragraf), bukan di karakter ke N-buta.
    """
    print("[*] Menjalankan Semantic Chunking...")
    # Pisahkan berdasarkan double newline (block paragraf)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Hapus spasi berlebih peninggalan formatting PDF
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue
            
        current_chunk += para + " "
        
        if len(current_chunk) >= max_length:
            last_period = current_chunk.rfind('. ')
            if last_period != -1 and last_period > min_length:
                chunk_str = current_chunk[:last_period+1].strip()
                chunks.append(chunk_str)
                
                # SLIDING WINDOW LOGIC (OVERLAP)
                # Kita ambil 1 kalimat terakhir dari chunk_str untuk dijadikan lem jembatan (overlap)
                # rfind dilakukan setelah menghilangkan titik terakhir [:-1] agar tidak menangkap dirinya sendiri
                prev_period = chunk_str[:-1].rfind('. ')
                if prev_period != -1:
                    overlap_sentence = chunk_str[prev_period+2:].strip()
                else:
                    overlap_sentence = chunk_str  # Jika cuma 1 kalimat super panjang, pakai semua
                    
                # Chunk selanjutnya dimulai dari Overlap + Sisa kalimat yang terputus
                current_chunk = overlap_sentence + " " + current_chunk[last_period+1:].strip()
            else:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pemakaian: python ingestor.py <path_ke_file_pdf>")
        sys.exit(1)
        
    target_pdf = sys.argv[1]
    
    # 1. Ekstrak
    raw_text = extract_pdf_clean(target_pdf)
    
    # 2. Chunking 
    chunks = semantic_chunking(raw_text)
    
    print(f"[+] Selesai! Mengubah dokumen menjadi {len(chunks)} blok pemikiran (chunks).")
    
    # Preview 3 Chunk Pertama
    print("\n--- PREVIEW 3 CHUNKS (DATA MENTAH) ---")
    for i, c in enumerate(chunks[:3]):
        print(f"\n[Chunk {i+1}] (Length: {len(c)} chars)\n{c}")
