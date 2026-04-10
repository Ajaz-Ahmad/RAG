import logging
import re
import urllib.parse

import requests

from src.ingestion.dataClasses import WikiChunk

logger = logging.getLogger(__name__)


def load_data_from_url(url: str) -> list[WikiChunk]:
    """Fetch a Wikipedia article via the MediaWiki REST API and return section-aware chunks.

    Uses the official API instead of HTML scraping so it works reliably
    regardless of Wikipedia's front-end HTML changes or IP-based scraping blocks.
    """
    match = re.search(r"en\.wikipedia\.org/wiki/([^#?]+)", url)
    if not match:
        raise ValueError(f"Not a valid English Wikipedia URL: {url}")

    title = urllib.parse.unquote(match.group(1).replace("_", " "))
    logger.info("Fetching '%s' via Wikipedia API", title)

    api_url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&prop=extracts&explaintext=true"
        f"&titles={urllib.parse.quote(title)}&format=json"
    )

    response = requests.get(
        api_url,
        headers={"User-Agent": "WikiRAGDemo/1.0 (portfolio project)"},
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia article '{title}' not found.")

    page_title = page["title"]
    text = page.get("extract", "")

    if not text:
        raise ValueError(f"Wikipedia returned an empty article for '{title}'.")

    chunks = _parse_extract(text, page_title)
    logger.info("Extracted %d raw chunks from '%s'", len(chunks), page_title)
    return chunks


def _parse_extract(text: str, page_title: str) -> list[WikiChunk]:
    """Parse MediaWiki plain-text extract into section-aware WikiChunk objects.

    The API encodes headings as:
      == Section ==
      === Subsection ===
    Everything else is a paragraph.
    """
    lines = text.split("\n")
    chunks: list[WikiChunk] = []
    section = ""
    subsection = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        h2 = re.match(r"^==\s+(.+?)\s+==$", stripped)
        h3 = re.match(r"^===\s+(.+?)\s+===$", stripped)
        h4 = re.match(r"^====\s+(.+?)\s+====$", stripped)

        if h4:
            subsection = h4.group(1)
        elif h3:
            subsection = h3.group(1)
        elif h2:
            section = h2.group(1)
            subsection = ""
        else:
            chunks.append(WikiChunk(
                page_title=page_title,
                level="p",
                context=section,
                section=section,
                subsection=subsection,
                text=stripped,
            ))

    return chunks
