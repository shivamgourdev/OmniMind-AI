from fastapi import APIRouter

from app.core.config import settings
from app.services.vector_store import get_collection_stats, get_uploaded_files

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("/")
def health_check():

    stats = get_collection_stats()

    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_database": "ChromaDB",
        "indexed_documents": stats["documents"],
        "indexed_files": stats["files"],
        "files": get_uploaded_files(),
    }
