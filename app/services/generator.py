from google import genai

from app.core.config import settings
from app.core.logger import logger


client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


BASE_SYSTEM_PROMPT = """You are OmniMind AI, a document question-answering assistant.

Rules:
1. Answer ONLY using the information in the numbered context blocks below.
2. If the answer is not present in the context, reply exactly:
   "I couldn't find this information in the uploaded documents."
3. Never invent facts, numbers, or details that are not in the context.
4. When you use information from a specific block, refer to it by its
   source number, e.g. "(Source 2)".
5. Structure longer answers with short paragraphs or bullet points.
6. Be concise and direct - do not pad the answer with filler."""

COMPARISON_ADDENDUM = """
7. This question asks you to compare multiple documents. Organize the
   answer by document or by point of comparison (whichever is clearer),
   and explicitly note if one of the documents does not contain
   information relevant to a given point."""


def generate_answer(question: str, context: str, multi_document: bool = False) -> str:
    """
    Generate an answer using Gemini, grounded strictly in the supplied
    context. `multi_document` adds comparison-specific instructions when
    the question referenced more than one file.
    """

    if not context.strip():
        return "I couldn't find this information in the uploaded documents."

    system_prompt = BASE_SYSTEM_PROMPT + (COMPARISON_ADDENDUM if multi_document else "")

    prompt = f"""{system_prompt}

Context:
{context}

Question:
{question}

Answer:"""

    try:

        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt
        )

        answer = (response.text or "").strip()

        if not answer:
            logger.warning("Generator returned an empty answer.")
            return "I couldn't find this information in the uploaded documents."

        logger.info("Answer generated successfully.")

        return answer

    except Exception as e:

        logger.exception(f"Generator Error: {e}")

        return "Sorry, an internal error occurred while generating the answer."
