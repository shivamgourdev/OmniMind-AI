from typing import List

import fitz
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import logger
from app.services.embeddings import get_embedding
from app.services.vector_store import add_to_vector_store, delete_by_filename
from app.utils.chunking import chunk_text

router = APIRouter()


class UploadedFileResult(BaseModel):
    filename: str
    status: str  
    chunks: int = 0
    reason: str | None = None


class UploadResponse(BaseModel):
    success: bool
    uploaded_files: list[UploadedFileResult]
    total_files: int
    total_chunks: int
    message: str


def _validate_file(file: UploadFile, size_bytes: int) -> str | None:
    """Returns an error reason string, or None if the file is acceptable."""

    if not any(file.filename.lower().endswith(ext) for ext in settings.ALLOWED_EXTENSIONS):
        return f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if size_bytes > max_bytes:
        return f"File exceeds the {settings.MAX_FILE_SIZE_MB}MB size limit."

    if size_bytes == 0:
        return "File is empty."

    return None


@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):

    logger.info(f"Received {len(files)} file(s)")

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded."
        )

    results: list[UploadedFileResult] = []
    total_chunks = 0

    for file in files:

        file_bytes = await file.read()

        validation_error = _validate_file(file, len(file_bytes))

        if validation_error:
            logger.warning(f"Rejected '{file.filename}': {validation_error}")

            results.append(UploadedFileResult(
                filename=file.filename,
                status="failed",
                reason=validation_error
            ))
            continue

        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")

        except Exception as e:
            logger.exception(f"Failed to open '{file.filename}': {e}")

            results.append(UploadedFileResult(
                filename=file.filename,
                status="failed",
                reason="Could not read file - it may be corrupted or not a valid PDF."
            ))
            continue

        if pdf.page_count == 0:
            pdf.close()

            results.append(UploadedFileResult(
                filename=file.filename,
                status="failed",
                reason="PDF has no pages."
            ))
            continue

        if pdf.page_count > settings.MAX_PAGES_PER_PDF:
            pdf.close()

            results.append(UploadedFileResult(
                filename=file.filename,
                status="failed",
                reason=f"PDF exceeds the {settings.MAX_PAGES_PER_PDF}-page limit "
                       f"({pdf.page_count} pages)."
            ))
            continue

        try:
        
            removed = delete_by_filename(file.filename)

            if removed:
                logger.info(f"Replacing {removed} existing chunk(s) for '{file.filename}'.")

            file_chunks = 0
            pages_with_text = 0

            for page_number, page in enumerate(pdf):

                text = page.get_text()

                if not text.strip():
                    continue

                pages_with_text += 1

                chunks = chunk_text(text)

                for chunk in chunks:
                    embedding = get_embedding(chunk)

                    add_to_vector_store(
                        embedding=embedding,
                        chunk=chunk,
                        filename=file.filename,
                        page_number=page_number + 1
                    )

                file_chunks += len(chunks)

            pdf.close()

            if pages_with_text == 0:
                results.append(UploadedFileResult(
                    filename=file.filename,
                    status="failed",
                    reason="No extractable text found (the PDF may be scanned images without OCR)."
                ))
                continue

            results.append(UploadedFileResult(
                filename=file.filename,
                status="success",
                chunks=file_chunks
            ))

            total_chunks += file_chunks

            logger.info(f"'{file.filename}' uploaded successfully ({file_chunks} chunks).")

        except Exception as e:

            logger.exception(f"Upload Error ({file.filename}): {e}")

            results.append(UploadedFileResult(
                filename=file.filename,
                status="failed",
                reason="Internal error while processing this file."
            ))

    succeeded = [r for r in results if r.status == "success"]

    logger.info(
        f"Upload completed. Succeeded={len(succeeded)}/{len(files)} Chunks={total_chunks}"
    )

    return UploadResponse(
        success=len(succeeded) > 0,
        uploaded_files=results,
        total_files=len(succeeded),
        total_chunks=total_chunks,
        message=(
            "All files uploaded successfully" if len(succeeded) == len(files)
            else f"{len(succeeded)}/{len(files)} file(s) uploaded successfully"
        )
    )
