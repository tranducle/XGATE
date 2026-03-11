"""
Writing and Synthesis Tools
Tools for academic writing, BibTeX management, and document synthesis.
"""

import os
import re
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class BibEntry:
    """Represents a BibTeX entry."""

    key: str
    entry_type: str  # article, inproceedings, book, etc.
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    journal: Optional[str] = None
    booktitle: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None

    def to_bibtex(self) -> str:
        """Convert to BibTeX format."""
        lines = [f"@{self.entry_type}{{{self.key},"]
        lines.append(f"  title = {{{self.title}}},")
        lines.append(f"  author = {{{' and '.join(self.authors)}}},")
        if self.year:
            lines.append(f"  year = {{{self.year}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.journal:
            lines.append(f"  journal = {{{self.journal}}},")
        if self.booktitle:
            lines.append(f"  booktitle = {{{self.booktitle}}},")
        if self.volume:
            lines.append(f"  volume = {{{self.volume}}},")
        if self.pages:
            lines.append(f"  pages = {{{self.pages}}},")
        if self.publisher:
            lines.append(f"  publisher = {{{self.publisher}}},")
        lines.append("}")
        return "\n".join(lines)


async def _lookup_doi_async(doi: str) -> Optional[Dict[str, Any]]:
    """
    Look up DOI metadata from CrossRef API (async implementation).

    Args:
        doi: DOI string (with or without https://doi.org/ prefix)

    Returns:
        Metadata dictionary or None if not found
    """
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required")

    # Clean DOI
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    url = f"https://api.crossref.org/works/{doi}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("message", {})
            return None


def lookup_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Look up DOI metadata from CrossRef API.

    Args:
        doi: DOI string (with or without https://doi.org/ prefix)

    Returns:
        Metadata dictionary or None if not found
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _lookup_doi_async(doi)).result()
    return asyncio.run(_lookup_doi_async(doi))


async def _doi_to_bibtex_async(doi: str) -> Optional[str]:
    """
    Convert DOI to BibTeX using doi2bib.org API (async implementation).

    Args:
        doi: DOI string

    Returns:
        BibTeX string or None if not found
    """
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required")

    # Clean DOI
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    url = f"https://doi2bib.org/bib/{doi}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                bibtex = await response.text()
                if bibtex.startswith("@"):
                    return bibtex
            return None


def doi_to_bibtex(doi: str) -> Optional[str]:
    """
    Convert DOI to BibTeX using doi2bib.org API.

    Args:
        doi: DOI string

    Returns:
        BibTeX string or None if not found
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _doi_to_bibtex_async(doi)).result()
    return asyncio.run(_doi_to_bibtex_async(doi))


def parse_bibtex_file(filepath: str) -> List[BibEntry]:
    """
    Parse a BibTeX file and return list of entries.

    Args:
        filepath: Path to .bib file

    Returns:
        List of BibEntry objects
    """
    entries = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple regex-based parser (for basic cases)
    pattern = r"@(\w+)\s*\{\s*([^,]+),([^@]+)"
    matches = re.findall(pattern, content, re.DOTALL)

    for entry_type, key, fields_str in matches:
        # Parse fields
        field_pattern = r'(\w+)\s*=\s*[{"]([^}"]+)[}"]'
        fields = dict(re.findall(field_pattern, fields_str))

        authors = []
        if "author" in fields:
            authors = [a.strip() for a in fields["author"].split(" and ")]

        entry = BibEntry(
            key=key.strip(),
            entry_type=entry_type.lower(),
            title=fields.get("title", ""),
            authors=authors,
            year=int(fields["year"]) if "year" in fields else None,
            doi=fields.get("doi"),
            journal=fields.get("journal"),
            booktitle=fields.get("booktitle"),
            volume=fields.get("volume"),
            pages=fields.get("pages"),
            publisher=fields.get("publisher"),
        )
        entries.append(entry)

    return entries


def find_missing_dois(entries: List[Any]) -> List[BibEntry]:
    """
    Find entries that are missing DOIs.

    Accepts both BibEntry objects and plain dicts. Dicts are automatically
    coerced to BibEntry for compatibility with host AI callers.
    """
    coerced: List[BibEntry] = []
    for e in entries:
        if isinstance(e, BibEntry):
            coerced.append(e)
        elif isinstance(e, dict):
            coerced.append(
                BibEntry(
                    key=e.get("key", "unknown"),
                    entry_type=e.get("entry_type", "article"),
                    title=e.get("title", ""),
                    authors=e.get("authors", []),
                    year=e.get("year"),
                    doi=e.get("doi"),
                    journal=e.get("journal"),
                    booktitle=e.get("booktitle"),
                    volume=e.get("volume"),
                    pages=e.get("pages"),
                    publisher=e.get("publisher"),
                )
            )
        else:
            continue
    return [e for e in coerced if not e.doi]


def generate_citation_key(authors: List[str], year: Optional[int], title: str) -> str:
    """
    Generate a citation key from metadata.
    Format: FirstAuthorLastName_Year_FirstWord
    """
    # Get first author's last name
    first_author = authors[0] if authors else "Unknown"
    last_name = first_author.split()[-1] if " " in first_author else first_author
    last_name = re.sub(r"[^a-zA-Z]", "", last_name)

    # Get first significant word from title
    stop_words = {"a", "an", "the", "on", "in", "for", "of", "to", "and"}
    words = title.lower().split()
    first_word = "paper"
    for w in words:
        clean = re.sub(r"[^a-zA-Z]", "", w)
        if clean and clean not in stop_words:
            first_word = clean.capitalize()
            break

    year_str = str(year) if year else "XXXX"

    return f"{last_name}{year_str}{first_word}"


def write_markdown_section(
    content: str,
    filepath: str,
    section_title: str,
    mode: str = "append",
) -> str:
    """
    Write a section to a markdown file.

    Args:
        content: Content to write
        filepath: Path to markdown file
        section_title: Title for the section
        mode: "append", "replace", or "create"

    Returns:
        Path to the file
    """
    path = Path(filepath)

    section = f"\n## {section_title}\n\n{content}\n"

    if mode == "create" or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Document\n{section}")
    elif mode == "append":
        with open(path, "a", encoding="utf-8") as f:
            f.write(section)
    elif mode == "replace":
        # Read existing content and replace section
        existing = path.read_text(encoding="utf-8")
        pattern = rf"## {re.escape(section_title)}\n.*?(?=\n## |\Z)"
        if re.search(pattern, existing, re.DOTALL):
            new_content = re.sub(pattern, section.strip(), existing, flags=re.DOTALL)
        else:
            new_content = existing + section
        path.write_text(new_content, encoding="utf-8")

    return str(path)


# Tool metadata for agent integration
WRITING_TOOLS = [
    {
        "name": "lookup_doi",
        "description": "Look up metadata for a DOI from CrossRef. Synchronous — safe for direct call.",
        "function": lookup_doi,
        "parameters": [
            {
                "name": "doi",
                "type": "string",
                "description": "DOI to look up",
                "required": True,
            },
        ],
    },
    {
        "name": "doi_to_bibtex",
        "description": "Convert a DOI to BibTeX format using doi2bib.org. Synchronous — safe for direct call.",
        "function": doi_to_bibtex,
        "parameters": [
            {
                "name": "doi",
                "type": "string",
                "description": "DOI to convert",
                "required": True,
            },
        ],
    },
    {
        "name": "parse_bibtex_file",
        "description": "Parse a BibTeX file and extract entries.",
        "function": parse_bibtex_file,
        "parameters": [
            {
                "name": "filepath",
                "type": "string",
                "description": "Path to .bib file",
                "required": True,
            },
        ],
    },
    {
        "name": "write_markdown_section",
        "description": "Write a section to a markdown file.",
        "function": write_markdown_section,
        "parameters": [
            {
                "name": "content",
                "type": "string",
                "description": "Content to write",
                "required": True,
            },
            {
                "name": "filepath",
                "type": "string",
                "description": "Target file path",
                "required": True,
            },
            {
                "name": "section_title",
                "type": "string",
                "description": "Section heading",
                "required": True,
            },
            {
                "name": "mode",
                "type": "string",
                "description": "append, replace, or create",
                "required": False,
                "default": "append",
            },
        ],
    },
]
