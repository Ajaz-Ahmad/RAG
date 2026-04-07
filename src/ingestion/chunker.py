import logging

from src.ingestion.dataClasses import WikiChunk, Chunk

logger = logging.getLogger(__name__)


def merge_wiki_chunks(wiki_chunks: list[WikiChunk], max_words: int = 200) -> list[Chunk]:
    """Merge consecutive paragraphs that share a section into a single Chunk.

    Keeps chunks under max_words to stay within embedding token limits.
    When adding a paragraph would exceed the limit, the current chunk is
    finalised and a new one is started — still under the same section key
    so context metadata is preserved.
    """
    merged: list[Chunk] = []
    current_section: tuple[str, str] | None = None
    current_text = ""

    def flush(section_key: tuple[str, str]) -> None:
        if current_text:
            merged.append(Chunk(
                text=current_text.strip(),
                metadata={"section": section_key[0], "subsection": section_key[1]},
            ))

    for wc in wiki_chunks:
        section_key = (wc.section, wc.subsection)
        incoming_words = len(wc.text.split())

        if section_key != current_section:
            # Section boundary — flush whatever was accumulating
            if current_section is not None:
                flush(current_section)
            current_section = section_key
            current_text = wc.text
        else:
            # Same section — check whether appending would exceed the limit
            if len(current_text.split()) + incoming_words > max_words:
                flush(current_section)
                current_text = wc.text
            else:
                current_text += " " + wc.text

    if current_section is not None:
        flush(current_section)

    logger.info("Merged into %d chunks (max_words=%d)", len(merged), max_words)
    return merged
