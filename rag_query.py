from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import re

# =========================================================
# CONFIG
# =========================================================
NGROK_URL = "https://onstage-ramrod-swear.ngrok-free.dev"
THRESHOLD = 0.6



STOPWORDS = {
    "saya", "tertarik", "dengan", "mata",
    "kuliah", "tentang", "pada", "dan"
}

# =========================================================
# LOAD EMBEDDING
# =========================================================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vectordb = Chroma(
#     persist_directory="embeddings",
#     embedding_function=embedding
# )


def get_vectordb():
    return Chroma(
        persist_directory="embeddings",
        embedding_function=embedding
    )
# =========================================================
# CALL LLM
# =========================================================
def call_llm(query, context):
    try:
        response = requests.post(
            f"{NGROK_URL}/generate",
            headers={"Content-Type": "application/json"},
            json={"query": query, "context": context},
            timeout=120
        )

        print("\n=== DEBUG LLM ===")
        print("STATUS:", response.status_code)

        if response.status_code != 200:
            return None

        return response.json().get("answer", "")

    except Exception as e:
        print("❌ ERROR LLM:", str(e))
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
        print("⚠️ LLM OFFLINE → pakai fallback di main")
        return []

    print("\n=== DEBUG PARSING ===")
    print("RAW:\n", response)

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

    # 🔥 kalau parsing gagal total → return kosong
    if len(cleaned) == 0:
        print("⚠️ PARSING GAGAL → pakai fallback di main")
        return []

    return cleaned[:jumlah]


# =========================================================
# MAIN RAG
# =========================================================
def get_recommendation(query: str):

    print("\n=== QUERY ===", query)
    vectordb = get_vectordb()

    results = vectordb.similarity_search_with_score(query, k=10)

    unique_results = []
    seen_codes = set()

    for doc, score in results:
        kode = doc.metadata.get("kode_mk")
        if kode not in seen_codes:
            seen_codes.add(kode)
            unique_results.append((doc, score))

    query_keywords = [
        w for w in query.lower().split()
        if w not in STOPWORDS
    ]

    final_results = []

    for doc, score in unique_results:
        relevance_score = 1 / (1 + score)

        nama_mk = doc.metadata.get("nama_mk").lower()
        description = doc.page_content.lower()

        for word in query_keywords:
            if word in nama_mk:
                relevance_score += 0.08

        if not any(word in (nama_mk + " " + description)
                   for word in query_keywords):
            continue

        if relevance_score < THRESHOLD:
            continue

        final_results.append((doc, relevance_score))

    final_results = sorted(
        final_results,
        key=lambda x: x[1],
        reverse=True
    )[:5]

    print("JUMLAH HASIL:", len(final_results))

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

    print("JUMLAH REASONS:", len(reasons))

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


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    query = input("Masukkan minat: ")
    result = get_recommendation(query)

    for r in result:
        print(r)