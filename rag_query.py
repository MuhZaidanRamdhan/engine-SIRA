from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import re
import gc
from groq import Groq
from dotenv import load_dotenv
import os

# =========================================================
# CONFIG
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
THRESHOLD = 0.04


STOPWORDS = {
    "saya", "tertarik", "dengan", "mata",
    "kuliah", "tentang", "pada", "dan", "membahas","yang"
}

# =========================================================
# LOAD EMBEDDING
# =========================================================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectordb = Chroma(
    persist_directory="embeddings",
    embedding_function=embedding
)

_vectordb = None

def get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = Chroma(
            persist_directory="embeddings",
            embedding_function=embedding
        )
    return _vectordb

def reset_vectordb():
    global _vectordb
    _vectordb = None
    gc.collect()
# =========================================================
# CALL LLM
# =========================================================
# def call_llm(query, context):
#     try:
#         response = requests.post(
#             f"{NGROK_URL}/generate",
#             headers={"Content-Type": "application/json"},
#             json={"query": query, "context": context},
#             timeout=120
#         )

#         print("\n=== DEBUG LLM ===")
#         print("STATUS:", response.status_code)

#         if response.status_code != 200:
#             return None

#         return response.json().get("answer", "")

#     except Exception as e:
#         print("❌ ERROR LLM:", str(e))
#         return None

def call_llm(query, context):
    try:
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah asisten akademik yang membantu menjelaskan mata kuliah. Jawab dalam Bahasa Indonesia."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return answer

    except Exception as e:
        return None


# =========================================================
# BUILD CONTEXT
# =========================================================
def build_context(courses):
    context = ""
    for i, c in enumerate(courses, 1):
        context += f"{i}. {c['nama_mk']} - {c['deskripsi'][:50]}\n"
    return context


# =========================================================
# BUILD REASONS (LLM ONLY)
# =========================================================
def build_reasons(courses):
    context = build_context(courses)
    jumlah = len(courses)

    prompt = f"""
{context}

Tugas:
Berikan tepat {jumlah} alasan sesuai urutan.

Format WAJIB:
- 1 baris 1 alasan
- Setiap baris diawali: Mata kuliah ini membahas
- Panjang 10–18 kata
- Harus kalimat lengkap
- Gunakan informasi dari deskripsi
- Jangan pakai nomor
- Jangan paragraf

Jawab:
"""

    response = call_llm(prompt, "")

    if not response:
        return []

    lines = re.split(r'\n|\.\s', response)

    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # hanya ambil yang valid
        if not line.lower().startswith("mata kuliah ini membahas"):
            continue

        words = line.split()

        # skip terlalu pendek
        if len(words) < 8:
            continue

        # batasi panjang (tapi tidak potong brutal)
        if len(words) > 25:
            line = " ".join(words[:25])

        line = line.rstrip(".")

        cleaned.append(line)

    if len(cleaned) == 0:
        return []

    return cleaned[:jumlah]

QUERY_EXPANSION = {
    "ai": "kecerdasan buatan kecerdasan artifisial machine learning",
    "ml": "machine learning kecerdasan buatan model prediktif",
    "nlp": "natural language processing pemrosesan bahasa",
    "oop": "pemrograman berorientasi objek kelas enkapsulasi",
    "ui": "antarmuka pengguna desain interface figma prototype wireframe",
    "ux": "pengalaman pengguna desain usability figma prototype wireframe",
    "uiux": "antarmuka pengguna desain interface pengalaman pengguna usability figma",
    "db": "basis data database sql",
    "antarmuka": "desain antarmuka ui ux interaksi penggunaan figma",
    "web": "pemrograman web aplikasi web frontend backend keamanan web",
    "jaringan": "jaringan komputer network routing switching lan wan terdistribusi"
}

KEYWORD_EXPANSION = {
    "ai": ["ai", "kecerdasan buatan", "kecerdasan artifisial", "machine learning"],
    "ml": ["ml", "machine learning", "kecerdasan buatan"],
    "nlp": ["nlp", "natural language processing", "bahasa alami"],
    "oop": ["oop", "berorientasi objek", "objek"],
    "ui": ["ui", "antarmuka", "interface", "desain", "figma"],
    "ux": ["ux", "pengalaman pengguna", "usability", "figma"],
    "uiux": ["ui", "ux", "antarmuka", "desain", "figma", "usability"],
    "db": ["db", "basis data", "database", "sql"],
    "antarmuka": ["antarmuka", "ui", "ux", "desain", "interaksi", "pengguna"],
    "web": ["web", "aplikasi web", "frontend", "backend", "html", "javascript"],
    "jaringan": ["jaringan", "network", "routing", "switching", "lan", "wan", "terdistribusi"]
}

def expand_query(query: str) -> str:
    words = query.lower().split()
    
    if len(words) == 1 and words[0] not in QUERY_EXPANSION:
        query = f"saya tertarik pada {query}"
        words = query.lower().split()
    
    expanded = []
    for word in words:
        if word in QUERY_EXPANSION:
            expanded.append(QUERY_EXPANSION[word])
        else:
            expanded.append(word)
    return " ".join(expanded)

def get_recommendation(query: str):
    print("\n=== QUERY ===", query)
    original_query = query
    query = expand_query(query)
    print("=== EXPANDED QUERY ===", query)

    vectordb = get_vectordb()
    results = vectordb.similarity_search_with_score(query, k=10)

    unique_results = []
    seen_codes = set()
    for doc, score in results:
        kode = doc.metadata.get("kode_mk")
        if kode not in seen_codes:
            seen_codes.add(kode)
            unique_results.append((doc, score))

    print("RAW RESULTS:")
    for doc, score in results:
        print(f"  - {doc.metadata.get('nama_mk')} | score: {score:.4f}")

    # keyword dari query asli
    query_keywords = [
        w for w in original_query.lower().split()
        if w not in STOPWORDS
    ]

    # expand keyword untuk filter
    expanded_keywords = []
    for word in query_keywords:
        if word in KEYWORD_EXPANSION:
            expanded_keywords.extend(KEYWORD_EXPANSION[word])
        else:
            expanded_keywords.append(word)

    final_results = []

    for doc, score in unique_results:
        semester = int(doc.metadata.get("semester", 0))
        if semester > 4:
            continue

        relevance_score = 1 / (1 + score)
        print(f"{doc.metadata.get('nama_mk')} | relevance: {relevance_score:.4f}")

        nama_mk = doc.metadata.get("nama_mk", "").lower().strip()
        description = doc.page_content.lower().strip()
        description = re.sub(r'\s+', ' ', description) 

        combined_text = nama_mk + " " + description

        for word in query_keywords:
            if word in nama_mk:
                relevance_score += 0.08
                

        # filter pakai expanded_keywords
        if not any(re.search(r'\b' + re.escape(word) + r'\b', combined_text) for word in expanded_keywords):
            continue

        if relevance_score < THRESHOLD:
            continue

        final_results.append((doc, relevance_score))

    final_results = sorted(
        final_results,
        key=lambda x: x[1],
        reverse=True
    )[:8]

    print("JUMLAH HASIL:", len(final_results))
    for doc, score in final_results:
        print(f"  - {doc.metadata.get('nama_mk')}")

    courses = []

    for doc, _ in final_results:
        courses.append({
            "nama_mk": doc.metadata.get("nama_mk"),
            "kode_mk": doc.metadata.get("kode_mk"),
            "peminatan": doc.metadata.get("peminatan"),
            "sks": doc.metadata.get("sks"),
            "semester": doc.metadata.get("semester"),
            "deskripsi": doc.page_content[:100]
        })

    reasons = build_reasons(courses)

    formatted_results = []

    for i, course in enumerate(courses):

        if i < len(reasons) and len(reasons[i]) > 10:
            reason = reasons[i]
        else:
           reason = (
                    f"Mata kuliah {course['nama_mk']} pada peminatan "
                    f"{course['peminatan'].strip()} relevan dengan minat yang Anda miliki"
                )

        formatted_results.append({
            "nama_mk": course["nama_mk"],
            "kode_mk": course["kode_mk"],
            "peminatan": course["peminatan"],
            "sks": course["sks"],
            "semester": course["semester"],
            "alasan": reason
        })

    return formatted_results

if __name__ == "__main__":
    query = input("Masukkan minat: ")
    result = get_recommendation(query)

    for r in result:
        print(r)