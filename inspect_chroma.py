from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="embeddings",
    embedding_function=embedding
)

collection = vectordb._collection

print("📦 Nama collection :", collection.name)
print("📊 Jumlah chunk     :", collection.count())

data = collection.get(limit=5)

for i in range(len(data["documents"])):
    print(f"\n=== DATA {i+1} ===")
    print("Document :", data["documents"][i])
    print("Metadata :", data["metadatas"][i])