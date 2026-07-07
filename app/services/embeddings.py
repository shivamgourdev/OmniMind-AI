from google import genai

from app.core.config import settings
from app.core.logger import logger

client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


def get_embedding(text: str) -> list[float]:
    """
    Generate embedding using Gemini Embedding API.
    """

    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )

        embedding = response.embeddings[0].values

        logger.info("Embedding generated successfully.")

        return embedding

    except Exception as e:

        logger.exception(f"Embedding generation failed: {e}")

        raise RuntimeError(
            "Failed to generate embedding."
        ) from e