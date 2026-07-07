from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.upload import router as upload_router
from app.api.ask import router as ask_router
from app.api.health import router as health_router

from app.middleware.logging import LoggingMiddleware

from app.core.exceptions import register_exception_handlers
from app.core.config import settings
from app.core.logger import logger


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Industry Level Hybrid RAG System powered by Gemini + ChromaDB + BM25 + Cross Encoder"
)

register_exception_handlers(app)

# ==========================
# Middleware
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

# ==========================
# Routers
# ==========================

app.include_router(upload_router)

app.include_router(ask_router)

app.include_router(health_router)

# ==========================
# Startup Event
# ==========================

@app.on_event("startup")
async def startup():

    logger.info("=" * 60)
    logger.info(f"{settings.APP_NAME} Started Successfully")
    logger.info(f"Version : {settings.APP_VERSION}")
    logger.info("=" * 60)

# ==========================
# Home
# ==========================

@app.get("/", tags=["Home"])
def home():

    return {
        "message": f"{settings.APP_NAME} Running Successfully 🚀",
        "version": settings.APP_VERSION
    }
