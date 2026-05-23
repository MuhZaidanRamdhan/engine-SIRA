from fastapi import FastAPI
from pydantic import BaseModel
from rag_query import get_recommendation
from typing import List
from ingest import sync_courses_to_chroma


app = FastAPI(title="RAG Recommendation API")


class QueryRequest(BaseModel):
    query: str

class Course(BaseModel):
    kode_mk: str
    nama_mk: str
    peminatan: str
    sks: int
    semester: int
    deskripsi: str

class CourseRequest(BaseModel):
    courses: List[Course]


@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}


@app.post("/recommend")
async def recommend(request: QueryRequest):

    results = get_recommendation(request.query)

    if not results:
        return {
            "success": False,
            "query": request.query,
            "total_recommendation": 0,
            "message": "Tidak ditemukan mata kuliah relevan"
        }

    return {
        "success": True,
        "query": request.query,
        "total_recommendation": len(results),
        "recommendations": results
    }


@app.post("/sync-embedding")
async def sync_embedding(request: CourseRequest):
    try:
        total = sync_courses_to_chroma(request.courses)

        return {
            "success": True,
            "message": "Sinkronisasi embedding berhasil",
            "total_data": total
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }