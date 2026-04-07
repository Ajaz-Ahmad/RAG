import logging

import requests
from bs4 import BeautifulSoup

from src.ingestion.dataClasses import WikiChunk

logger = logging.getLogger(__name__)


def load_data_from_url(url: str) -> list[WikiChunk]:
    """Scrape a Wikipedia article and return paragraphs tagged with their section hierarchy.

    Tries multiple known Wikipedia container selectors so the scraper degrades
    gracefully if MediaWiki changes its HTML structure.
    """
    logger.info("Fetching %s", url)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    page_title_el = soup.find("h1")
    if page_title_el is None:
        raise ValueError(f"Could not find page title at {url}")
    page_title = page_title_el.get_text(strip=True)

    # Fallback chain — MediaWiki has changed container IDs across versions
    container = (
        soup.find("div", {"id": "mw-content-container"})
        or soup.find("div", class_="mw-content-container")
        or soup.find("div", {"id": "bodyContent"})
        or soup.find("main")
    )
    if container is None:
        raise ValueError(f"Could not locate article body at {url}. Wikipedia may have changed its HTML structure.")

    chunks: list[WikiChunk] = []
    current = WikiChunk(
        page_title=page_title, level="h1",
        context="", section="", subsection="", text=""
    )

    for el in container.find_all(["h1", "h2", "h3", "h4", "p"]):
        level = el.name
        text = el.get_text(strip=True)

        if not text:
            continue

        if level != "p":
            # Heading — advance the section hierarchy
            if current.section == "":
                current.section = text
            elif current.subsection == "":
                current.subsection = text
            else:
                current.context = current.section
                current.section = current.subsection
                current.subsection = text
            current.level = level
        else:
            # Paragraph — emit a chunk
            current.text = text

            if current.section == "" and chunks:
                # Inherit section from previous chunk if still at article preamble
                current.section = chunks[-1].section
                current.subsection = chunks[-1].subsection

            chunks.append(current)
            current = WikiChunk(
                page_title=page_title, level="h1",
                context="", section="", subsection="", text=""
            )

    logger.info("Extracted %d raw chunks from '%s'", len(chunks), page_title)
    return chunks