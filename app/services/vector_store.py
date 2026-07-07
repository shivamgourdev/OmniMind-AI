import hashlib
import os

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logger import logger



client = chromadb.PersistentClient(
    path=settings.CHROMA_DB_PATH
)

collection = client.get_or_create_collection(
    name="documents"
)



existing_data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

documents = existing_data.get(
    "documents",
    []
)

metadatas = existing_data.get(
    "metadatas",
    []
)

ids = existing_data.get(
    "ids",
    []
)

all_chunks_by_id: dict[str, dict] = {}

for chunk_id, document, metadata in zip(ids, documents, metadatas):

    all_chunks_by_id[chunk_id] = {
        "id": chunk_id,
        "chunk": document,
        "filename": metadata["filename"],
        "page_number": metadata["page_number"]
    }

logger.info(
    f"Loaded {len(all_chunks_by_id)} chunks from ChromaDB."
)


bm25 = None
bm25_ids: list[str] = []


def rebuild_bm25():
    """
    Rebuilds the in-memory BM25 index from the current chunk set.
    Called after any add/delete/clear so BM25 never drifts from ChromaDB.
    """

    global bm25, bm25_ids

    if len(all_chunks_by_id) == 0:
        bm25 = None
        bm25_ids = []
        return

    bm25_ids = list(all_chunks_by_id.keys())

    tokenized_chunks = [
        all_chunks_by_id[chunk_id]["chunk"].split()
        for chunk_id in bm25_ids
    ]

    bm25 = BM25Okapi(tokenized_chunks)


rebuild_bm25()



reranker = CrossEncoder(
    settings.CROSS_ENCODER_MODEL
)



def generate_chunk_id(filename, page_number, chunk):
    """
    Deterministic ID from (filename, page, chunk content).
    Re-uploading identical content produces the same ID, so `upsert`
    naturally deduplicates instead of erroring or creating copies.
    """

    unique = f"{filename}_{page_number}_{chunk}"

    return hashlib.md5(unique.encode()).hexdigest()


def normalize_filename(filename: str) -> str:
    return os.path.basename(filename).strip().lower()



def add_to_vector_store(embedding, chunk, filename, page_number):
    """
    Upserts a single chunk. Using upsert (not add) means re-processing
    the same file/page/chunk overwrites in place instead of raising a
    duplicate-ID error or silently piling up duplicate rows.
    """

    try:
        chunk_id = generate_chunk_id(filename, page_number, chunk)

        collection.upsert(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[
                {
                    "filename": filename,
                    "page_number": page_number
                }
            ],
            ids=[chunk_id]
        )

        all_chunks_by_id[chunk_id] = {
            "id": chunk_id,
            "chunk": chunk,
            "filename": filename,
            "page_number": page_number
        }

        rebuild_bm25()

        logger.info(
            f"Stored chunk | {filename} | Page {page_number}"
        )

    except Exception as e:

        logger.exception(
            f"Vector Store Error : {e}"
        )

        raise



def delete_by_filename(filename: str) -> int:

    try:
        target = normalize_filename(filename)

        ids_to_delete = [
            chunk_id
            for chunk_id, item in all_chunks_by_id.items()
            if normalize_filename(item["filename"]) == target
        ]

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

            for chunk_id in ids_to_delete:
                all_chunks_by_id.pop(chunk_id, None)

            rebuild_bm25()

        logger.info(
            f"Deleted {len(ids_to_delete)} existing chunk(s) for '{filename}'."
        )

        return len(ids_to_delete)

    except Exception as e:

        logger.exception(
            f"Delete By Filename Error ({filename}) : {e}"
        )

        raise




def search_vector(query_embedding, question, filenames: list[str] | None = None):
    """
    Hybrid retrieval: dense vector search + BM25, merged, optionally
    restricted to a set of requested files, then cross-encoder reranked.

    filenames: optional list of filenames the user explicitly referenced
    (e.g. "Compare A.pdf and B.pdf" -> ["A.pdf", "B.pdf"]). When provided,
    retrieval is scoped ONLY to chunks whose stored filename matches one
    of these (case-insensitive, exact or substring match), so unrelated
    documents can never leak into the answer or citations.
    """

    try:
        normalized_targets = (
            [normalize_filename(f) for f in filenames]
            if filenames
            else None
        )

    
        where_filter = None

        if normalized_targets:
            exact_stored_names = sorted({
                item["filename"]
                for item in all_chunks_by_id.values()
                if normalize_filename(item["filename"]) in normalized_targets
            })

            if exact_stored_names:
                where_filter = {"filename": {"$in": exact_stored_names}}

        

        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.VECTOR_CANDIDATES,
            where=where_filter,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_chunks = []
        seen_chunks = set()

        documents = vector_results.get("documents", [[]])[0]
        metadatas = vector_results.get("metadatas", [[]])[0]

        for document, metadata in zip(documents, metadatas):

            if document in seen_chunks:
                continue

            seen_chunks.add(document)

            retrieved_chunks.append({
                "chunk": document,
                "filename": metadata["filename"],
                "page_number": metadata["page_number"]
            })

    

        if bm25 is not None:

            scores = bm25.get_scores(question.split())

            scored_ids = sorted(
                zip(bm25_ids, scores),
                key=lambda pair: pair[1],
                reverse=True
            )

            added = 0

            for chunk_id, _score in scored_ids:

                if added >= settings.BM25_CANDIDATES:
                    break

                item = all_chunks_by_id.get(chunk_id)

                if item is None:
                    continue

                if normalized_targets is not None and normalize_filename(item["filename"]) not in normalized_targets:
                    continue

                if item["chunk"] in seen_chunks:
                    continue

                seen_chunks.add(item["chunk"])
                retrieved_chunks.append(item)
                added += 1


        if normalized_targets:

            retrieved_chunks = [
                item for item in retrieved_chunks
                if normalize_filename(item["filename"]) in normalized_targets
            ]

        if len(retrieved_chunks) == 0:

            logger.info("No chunks found.")

            return []

        

        pairs = [
            (question, item["chunk"])
            for item in retrieved_chunks
        ]

        scores = reranker.predict(pairs)

        for item, score in zip(retrieved_chunks, scores):
            item["score"] = float(score)

        retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")

        return retrieved_chunks[:settings.RERANK_KEEP]

    except Exception as e:

        logger.exception(f"Search Error : {e}")

        return []




def clear_vector_store():

    try:
        existing = collection.get()
        ids = existing.get("ids", [])

        if ids:
            collection.delete(ids=ids)

        all_chunks_by_id.clear()

        rebuild_bm25()

        logger.info("Vector store cleared successfully.")

    except Exception as e:

        logger.exception(f"Clear Vector Store Error : {e}")

        raise



def get_total_chunks():
    return len(all_chunks_by_id)


def get_uploaded_files():
    return sorted({
        item["filename"]
        for item in all_chunks_by_id.values()
    })


def file_exists(filename: str) -> bool:
    target = normalize_filename(filename)
    return any(
        normalize_filename(item["filename"]) == target
        for item in all_chunks_by_id.values()
    )


def get_collection_stats():
    return {
        "documents": len(all_chunks_by_id),
        "files": len(get_uploaded_files())
    }
