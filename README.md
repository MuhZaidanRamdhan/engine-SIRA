# RAG Engine SIRA  
### Retrieval-Augmented Generation Engine for Sistem Informasi Rekomendasi Akademik

RAG Engine untuk **SIRA (Sistem Informasi Rekomendasi Akademik)** yang bertugas memproses rekomendasi mata kuliah berbasis **Retrieval-Augmented Generation (RAG)**.

Engine ini dibangun menggunakan **FastAPI**, **ChromaDB**, **Sentence Transformers**, dan **LLaMA 3**, yang digunakan untuk menghasilkan rekomendasi mata kuliah berdasarkan minat pengguna.

---

## 📌 Tentang Proyek

RAG Engine berfungsi sebagai layanan AI utama pada sistem SIRA.

Engine ini menerima request dari backend Express untuk:

- menghasilkan rekomendasi mata kuliah
- melakukan retrieval data dari vector database
- menghasilkan alasan rekomendasi menggunakan LLM
- melakukan sinkronisasi embedding dari data silabus terbaru

---

## 🛠️ Tech Stack

### API Framework
- FastAPI
- Uvicorn

### Vector Database
- ChromaDB

### Embedding Model
- sentence-transformers/all-MiniLM-L6-v2

### LLM
- LLaMA 3 (GGUF)
- llama-cpp-python

### Retrieval Framework
- LangChain

### Data Processing
- Pandas

---

## 📂 Struktur Project

```bash
rag-engine/
│
├── api/
│   └── main.py
│
├── ingest.py
├── rag_query.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Fitur Utama

## 1. Recommendation API

Menghasilkan rekomendasi mata kuliah berdasarkan input minat mahasiswa.

### Endpoint

```http
POST /recommend
```

Contoh request:

```json
{
  "query": "Saya tertarik pada keamanan data"
}
```

---

## 2. Sync Embedding

Sinkronisasi data mata kuliah dari backend Express ke vector database.

### Endpoint

```http
POST /sync-embedding
```

Proses:
1. menerima data mata kuliah
2. membentuk dokumen
3. chunking
4. generate embedding
5. simpan ke ChromaDB

---

## 3. Retrieval Process

Sistem melakukan semantic search pada ChromaDB untuk mengambil mata kuliah paling relevan berdasarkan query pengguna.

---

## 4. LLM Generation

LLaMA 3 digunakan untuk menghasilkan alasan rekomendasi mata kuliah yang kontekstual.

---

## ⚙️ Instalasi

## 1. Clone Repository

```bash
git clone https://github.com/username/rag-engine.git
cd rag-engine
```

---

## 2. Buat Virtual Environment

```bash
python -m venv venv
```

---

## 3. Aktifkan Virtual Environment

### Windows CMD

```bash
venv\Scripts\activate
```

### Windows PowerShell

```bash
.\venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Siapkan Model LLaMA

Letakkan file model `.gguf` pada folder model yang sesuai.

Contoh:

```bash
models/llama-3.gguf
```

Pastikan path model sesuai pada konfigurasi project.

---

## 6. Jalankan Server

```bash
uvicorn main:app --reload
```

Server berjalan di:

```bash
http://127.0.0.1:8000
```

---

## 📦 Generate requirements.txt

Jika melakukan update dependency:

```bash
pip freeze > requirements.txt
```

---

## 🔄 Setup Setelah Clone

Karena folder berikut tidak disimpan ke GitHub:

- `venv/`
- `embeddings/`

Maka setelah clone wajib:

### 1. Rebuild environment

```bash
python -m venv venv
```

### 2. Install package

```bash
pip install -r requirements.txt
```

### 3. Sinkronisasi embedding

Jalankan endpoint:

```http
POST /sync-embedding
```

Untuk membangun ulang vector database.

---

## 📊 Alur Kerja Sistem

```text
Backend Express
    ↓
FastAPI
    ↓
Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Retrieval
    ↓
LLaMA 3 Generation
    ↓
Recommendation Response
```

---

## 🧠 Cara Kerja RAG

### 1. Input Query
Mahasiswa memasukkan minat.

Contoh:

```text
Saya tertarik pada keamanan web
```

---

### 2. Retrieval
Query diubah menjadi embedding dan dicocokkan dengan vector pada ChromaDB.

---

### 3. Context Selection
Dokumen paling relevan dipilih.

---

### 4. Generation
Context dikirim ke LLaMA 3 untuk menghasilkan rekomendasi dan alasan.

---

### 5. Response
Sistem mengembalikan daftar rekomendasi mata kuliah.

---

## 📈 Evaluasi

Engine diuji menggunakan **RAGAS** dengan metrik:

- Context Precision
- Context Recall
- Faithfulness

---

## 🎯 Tujuan

Engine ini dikembangkan untuk:

- mendukung sistem rekomendasi akademik
- memanfaatkan Retrieval-Augmented Generation
- meningkatkan relevansi rekomendasi mata kuliah

---

## 👨‍💻 Developer

**Muhammad Zaidan Ramdhan**  
Program Studi Teknik Informatika

---

## 📚 Penelitian

Dikembangkan sebagai bagian dari skripsi:

**Perancangan dan Implementasi Engine Rekomendasi Mata Kuliah Berbasis Retrieval-Augmented Generation (RAG) dengan Integrasi REST API di Lingkungan Teknik Informatika STT Terpadu Nurul Fikri**
