from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="embeddings",
    embedding_function=embedding
)

query = "Saya tertarik pada kecerdasan buatan dan pengolahan data"

print("🔍 Query:", query)
print("=" * 50)

results = vectordb.similarity_search(query, k=10)

for i, doc in enumerate(results, start=1):
    print(f"\n📘 REKOMENDASI {i}")
    print("-" * 40)
    print("Kode MK   :", doc.metadata.get("kode_mk"))
    print("Nama MK   :", doc.metadata.get("nama_mk"))
    print("Peminatan :", doc.metadata.get("peminatan"))
    print("SKS       :", doc.metadata.get("sks"))
    print("Semester  :", doc.metadata.get("semester"))
    print("Alasan    :", doc.page_content)