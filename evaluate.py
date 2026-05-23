import json
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# 1. LOAD VECTOR DB
# =========================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="embeddings",
    embedding_function=embedding
)

# 🔥 lebih ketat
THRESHOLD = 0.6
TOP_K = 8

# =========================
# 2. STOPWORDS
# =========================
STOPWORDS = {
    "saya", "tertarik", "pada", "mata", "kuliah",
    "tentang", "dan", "yang", "di", "ke"
}

# =========================
# 3. RETRIEVAL (IMPROVED)
# =========================
def retrieve(query):
    results = vectordb.similarity_search_with_score(query, k=TOP_K)

    query_words = [
        w for w in re.findall(r"\b\w+\b", query.lower())
        if w not in STOPWORDS
    ]

    contexts = []

    for doc, score in results:
        text = doc.page_content.lower()

        # 🔥 relevance dasar
        relevance_score = 1 / (1 + score)

        # 🔥 hitung jumlah keyword match
        match_count = sum(1 for w in query_words if w in text)

        # 🔥 boosting
        relevance_score += match_count * 0.1

        # 🔥 filter: minimal ada 1 keyword
        if match_count == 0:
            continue

        # 🔥 threshold
        if relevance_score >= THRESHOLD:
            contexts.append(doc.page_content)

    return contexts


# =========================
# 4. METRICS
# =========================
def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def context_precision(contexts, expected_keywords):
    if not contexts:
        return 0.0

    hits = 0
    for ctx in contexts:
        if any(k in ctx.lower() for k in expected_keywords):
            hits += 1

    return round(hits / len(contexts), 2)


def context_recall(contexts, expected_keywords):
    if not expected_keywords:
        return 0.0

    joined = " ".join(contexts).lower()

    found = 0
    for k in expected_keywords:
        if k in joined:
            found += 1

    return round(found / len(expected_keywords), 2)


def faithfulness(answer, contexts):
    if not contexts:
        return 1.0 if "tidak ditemukan" in answer.lower() else 0.0

    ctx_words = tokenize(" ".join(contexts))
    ans_words = tokenize(answer)

    if not ans_words:
        return 0.0

    supported = len(ans_words & ctx_words)
    return round(supported / len(ans_words), 2)


# =========================
# 5. LOAD DATASET
# =========================
with open("ragas_test.json", "r", encoding="utf-8") as f:
    tests = json.load(f)


# =========================
# 6. RUN EVALUATION
# =========================
results = []

print("\n==============================")
print(" HASIL EVALUASI KUALITAS RAG ")
print("==============================\n")

for t in tests:
    query = t["query"]
    expected = t["expected_keywords"]

    contexts = retrieve(query)

    # 🔥 jawaban sederhana (untuk faithfulness)
    answer = (
        "Tidak ditemukan mata kuliah yang relevan."
        if not contexts else
        " ".join(contexts[:1])
    )

    cp = context_precision(contexts, expected)
    cr = context_recall(contexts, expected)
    fa = faithfulness(answer, contexts)

    results.append((cp, cr, fa))

    print(f"Query                : {query}")
    print(f"Context Precision    : {cp:.2f}")
    print(f"Context Recall       : {cr:.2f}")
    print(f"Faithfulness         : {fa:.2f}")
    print("-" * 50)


# =========================
# 7. AVERAGE
# =========================
avg_cp = sum(r[0] for r in results) / len(results)
avg_cr = sum(r[1] for r in results) / len(results)
avg_fa = sum(r[2] for r in results) / len(results)

print("\n==============================")
print(" RATA-RATA METRIK EVALUASI ")
print("==============================\n")

print(f"Rata-rata Context Precision : {avg_cp:.2f}")
print(f"Rata-rata Context Recall    : {avg_cr:.2f}")
print(f"Rata-rata Faithfulness      : {avg_fa:.2f}")