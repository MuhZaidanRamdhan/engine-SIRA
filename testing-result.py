from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectordb = Chroma(persist_directory="embeddings", embedding_function=embedding)

results = vectordb.similarity_search("Machine Learning", k=1)
for r in results:
    print(r.page_content)