from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=settings.CHUNK_SIZE,

    chunk_overlap=settings.CHUNK_OVERLAP,

    separators=[
        "\n\n",
        "\n",
        ". ",
        " "
    ]
)


def chunk_text(text: str) -> list[str]:

    if not text or not text.strip():
        return []

    chunks = text_splitter.split_text(text)

    # Drop whitespace-only fragments that can appear at chunk boundaries.
    return [chunk for chunk in chunks if chunk.strip()]
