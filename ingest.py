from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def sync_courses_to_chroma(courses):
    print("Menerima data dari Express API...")

    documents = []

    for course in courses:
        combined_text = f"""
        Nama Mata Kuliah: {course.nama_mk}
        Peminatan: {course.peminatan}
        Deskripsi: {course.deskripsi}
        """

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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    print(f"Total chunk setelah split: {len(docs)}")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Menghapus vector lama...")

    try:
        old_db = Chroma(
            persist_directory="embeddings",
            embedding_function=embedding
        )

        old_db.delete_collection()

    except:
        print("Collection lama tidak ditemukan")

    print("Menyimpan vector baru...")

    Chroma.from_documents(
        docs,
        embedding=embedding,
        persist_directory="embeddings"
    )

    print("Sinkronisasi selesai")

    return len(documents)