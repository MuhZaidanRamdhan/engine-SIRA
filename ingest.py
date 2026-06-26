import gc
import time
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rag_query import reset_vectordb

def sync_courses_to_chroma(courses):
    print("Menerima data dari Express API...")

    documents = []

    for course in courses:
        combined_text = f"""Nama Mata Kuliah: {course.nama_mk}
        Peminatan: {course.peminatan}
        Deskripsi: {course.deskripsi}"""

        doc = Document(
            page_content=combined_text.strip(),
            metadata={
                "kode_mk": course.kode_mk,
                "nama_mk": course.nama_mk,
                "peminatan": course.peminatan,
                "sks": course.sks,
                "semester": course.semester
            }
        )
        documents.append(doc)

    print(f"Total mata kuliah diterima: {len(documents)}")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # reset koneksi RAG dulu
    reset_vectordb()
    gc.collect()
    time.sleep(1)

    print("Menghapus collection lama...")
    try:
        old_db = Chroma(
            persist_directory="embeddings",
            embedding_function=embedding
        )
        old_db.delete_collection()
        print("Collection lama dihapus")
    except Exception as e:
        print(f"Collection lama tidak ditemukan: {e}")

    print("Menyimpan vector baru...")
    Chroma.from_documents(
        documents,
        embedding=embedding,
        persist_directory="embeddings"
    )

    print("Sinkronisasi selesai")
    return len(documents)