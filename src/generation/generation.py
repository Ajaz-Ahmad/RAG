import logging

from openai import OpenAI

from config import Settings
from src.ingestion.dataClasses import Chunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer using ONLY the information \
provided in the numbered context passages below. If the answer is not contained \
in the context, say "I don't have enough information to answer that." \
Do not speculate or draw on outside knowledge.\
"""


def generate_answer(query: str, chunks: list[Chunk], settings: Settings) -> str:
    """Generate a grounded answer from retrieved chunks.

    Falls back to returning the highest-scoring chunk verbatim when no
    OpenAI API key is configured — useful for local/offline testing.
    """
    if not chunks:
        return "No relevant passages were found for your question."

    context = "\n\n".join(
        f"[{i + 1}] {c.text}" for i, c in enumerate(chunks)
    )

    if settings.openai_api_key in ("", "EMPTY"):
        logger.warning("No OpenAI API key set — returning best passage as answer.")
        return chunks[0].text

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    answer = response.choices[0].message.content or ""
    logger.info("Generated answer (%d chars) via %s", len(answer), settings.openai_model)
    return answer
