import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import logger
from app.services.embeddings import get_embedding
from app.services.vector_store import file_exists, search_vector
from app.services.generator import generate_answer

router = APIRouter()



class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    question: str
    requested_files: list[str]
    missing_files: list[str]
    searched_files: list[str]
    answer: str
    sources: list[str]



_FILENAME_TOKEN_RE = re.compile(r'([^\s,;:]+\.pdf)', re.IGNORECASE)


def extract_filenames(question: str) -> list[str]:

    raw_matches = _FILENAME_TOKEN_RE.findall(question)

    cleaned = []
    seen = set()

    for match in raw_matches:
        name = match.strip(" \t\"'()[]{}")

        key = name.lower()

        if name and key not in seen:
            seen.add(key)
            cleaned.append(name)

    return cleaned



@router.post(
    "/ask",
    response_model=AskResponse
)
def ask_question(request: AskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    logger.info(f"Question : {question}")

    requested_files = extract_filenames(question)

    logger.info(f"Requested files: {requested_files}")

    missing_files = [
        name for name in requested_files
        if not file_exists(name)
    ]

    if requested_files and len(missing_files) == len(requested_files):
    
        logger.info(f"None of the requested files exist: {requested_files}")

        return {
            "question": question,
            "requested_files": requested_files,
            "missing_files": missing_files,
            "searched_files": [],
            "answer": (
                "I couldn't find "
                + (", ".join(requested_files) if len(requested_files) > 1 else requested_files[0])
                + " in the uploaded documents. Please check the filename or upload it first."
            ),
            "sources": []
        }

    files_to_search = [f for f in requested_files if f not in missing_files] or None

    query_embedding = get_embedding(question)

    results = search_vector(
        query_embedding=query_embedding,
        question=question,
        filenames=files_to_search
    )

    if len(results) == 0:

        return {
            "question": question,
            "requested_files": requested_files,
            "missing_files": missing_files,
            "searched_files": [],
            "answer": "I couldn't find this information in the uploaded documents.",
            "sources": []
        }

    top_results = results[:settings.TOP_K]

    context = "\n\n".join(
        f"[Source {i + 1} | {result['filename']} | Page {result['page_number']}]\n{result['chunk']}"
        for i, result in enumerate(top_results)
    )

    answer = generate_answer(
        question,
        context,
        multi_document=len(requested_files) > 1
    )


    searched_files = []
    seen_files = set()

    sources = []
    seen_sources = set()

    for result in top_results:

        if result["filename"] not in seen_files:
            seen_files.add(result["filename"])
            searched_files.append(result["filename"])

        source = f"{result['filename']} - Page {result['page_number']}"

        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)

    logger.info(f"Retrieved {len(results)} chunks, used top {len(top_results)}.")
    logger.info(f"Generated answer using {len(sources)} source(s).")

    return {
        "question": question,
        "requested_files": requested_files,
        "missing_files": missing_files,
        "searched_files": searched_files,
        "answer": answer,
        "sources": sources
    }
