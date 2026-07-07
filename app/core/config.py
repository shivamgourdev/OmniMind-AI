from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    
    APP_NAME: str = "OmniMind AI"
    APP_VERSION: str = "2.1.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    
    
    
    GEMINI_API_KEY: str

    OPENROUTER_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "OmniMind_AI"
    LANGCHAIN_TRACING_V2: bool = False

    
    CHROMA_DB_PATH: str = "chromadb_data"

    
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 20
    MAX_PAGES_PER_PDF: int = 500
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf",)

    
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 250

    
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


    TOP_K: int = 5
    VECTOR_CANDIDATES: int = 50
    BM25_CANDIDATES: int = 20
    RERANK_KEEP: int = 15

    
    LLM_MODEL: str = "gemini-2.5-flash"

    
    CORS_ORIGINS: list[str] = ["*"]

    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
