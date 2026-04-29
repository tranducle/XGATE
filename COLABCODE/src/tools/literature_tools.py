"""
Literature Research Tools
Callable tools for literature search and paper discovery.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars


# Try to import API clients, gracefully handle missing dependencies
try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

try:
    from serpapi import GoogleSearch

    HAS_SERPAPI = True
except ImportError:
    GoogleSearch = None
    HAS_SERPAPI = False


@dataclass
class Paper:
    """Represents a research paper."""

    title: str
    authors: List[str]
    year: Optional[int]
    abstract: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    citation_count: Optional[int]
    source: str  # "semantic_scholar", "google_scholar", "scopus"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "doi": self.doi,
            "url": self.url,
            "citation_count": self.citation_count,
            "source": self.source,
        }

    def to_markdown(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        citation = f" (Citations: {self.citation_count})" if self.citation_count else ""
        doi_link = f" [DOI]({self.doi})" if self.doi else ""
        return f"**{self.title}** ({self.year}){citation}\n   {authors_str}{doi_link}"


# ============================================================================
# OPENALEX SEARCH (PRIORITY 1)
# Open catalog of scholarly works - 100,000 credits/day with free API key
# Docs: https://docs.openalex.org/api-guide-for-llms
# ============================================================================


async def search_openalex(
    query: str,
    limit: int = 30,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Paper]:
    """
    Search OpenAlex API for academic papers.

    OpenAlex is a fully open catalog of scholarly works with:
    - 100,000 credits/day (free API key)
    - 10 credits per list request
    - Max 100 requests/second

    Args:
        query: Search query string (searches title/abstract)
        limit: Maximum number of results (max 200 per page)
        year_from: Filter papers from this year (default: current year - 5)
        year_to: Filter papers until this year

    Returns:
        List of Paper objects
    """
    import time

    # Default to last 5 years if not specified
    if year_from is None and year_to is None:
        year_from = datetime.now().year - 5

    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required. Install with: pip install aiohttp")

    api_key = os.getenv("OPENALEX_API_KEY", "")
    base_url = "https://api.openalex.org/works"

    # Build filter string
    filters = []
    if year_from and year_to:
        filters.append(f"publication_year:{year_from}-{year_to}")
    elif year_from:
        filters.append(f"publication_year:>{year_from - 1}")
    elif year_to:
        filters.append(f"publication_year:<{year_to + 1}")

    params = {
        "search": query,
        "per_page": min(limit, 200),
        "select": "id,title,authorships,publication_year,cited_by_count,doi,abstract_inverted_index",
    }

    if filters:
        params["filter"] = ",".join(filters)

    # Add API key if available (recommended for higher limits)
    if api_key:
        params["api_key"] = api_key

    headers = {"User-Agent": "ResearchAgentSystem/1.0 (mailto:research@example.com)"}

    papers = []
    try:
        # Rate limit: wait 0.5s before request
        await asyncio.sleep(0.5)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                base_url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])

                    for item in results:
                        # Extract authors from authorships
                        authors = []
                        for authorship in item.get("authorships", []):
                            author_info = authorship.get("author", {})
                            name = author_info.get("display_name", "")
                            if name:
                                authors.append(name)

                        # Reconstruct abstract from inverted index
                        abstract = None
                        abstract_idx = item.get("abstract_inverted_index")
                        if abstract_idx:
                            try:
                                # Convert inverted index to text
                                word_positions = []
                                for word, positions in abstract_idx.items():
                                    for pos in positions:
                                        word_positions.append((pos, word))
                                word_positions.sort()
                                abstract = " ".join([w for _, w in word_positions])
                            except (KeyError, TypeError):
                                abstract = None

                        # Extract DOI (remove https://doi.org/ prefix if present)
                        doi = item.get("doi", "")
                        if doi and doi.startswith("https://doi.org/"):
                            doi = doi.replace("https://doi.org/", "")

                        paper = Paper(
                            title=item.get("title", "Unknown") or "Unknown",
                            authors=authors,
                            year=item.get("publication_year"),
                            abstract=abstract,
                            doi=doi if doi else None,
                            url=item.get("id"),  # OpenAlex URL
                            citation_count=item.get("cited_by_count"),
                            source="openalex",
                        )
                        papers.append(paper)

                elif response.status == 429:
                    logger.warning("OpenAlex rate limit exceeded")
                else:
                    error_text = await response.text()
                    logger.error(
                        "OpenAlex API error %d: %s", response.status, error_text[:200]
                    )

    except Exception as e:
        logger.error("OpenAlex request failed: %s", e)

    return papers


def search_openalex_sync(
    query: str,
    limit: int = 30,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Paper]:
    """Synchronous wrapper for search_openalex."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, search_openalex(query, limit, year_from, year_to)
            ).result()
    return asyncio.run(search_openalex(query, limit, year_from, year_to))


def _simplify_query_for_semantic_scholar(query: str) -> str:
    """Reduce a natural-language query to keyword-style for Semantic Scholar's API,
    which returns 0 results on sentence-style queries longer than ~8 terms."""
    import re

    _FILLER = {
        "find",
        "search",
        "look",
        "get",
        "retrieve",
        "fetch",
        "show",
        "list",
        "give",
        "provide",
        "about",
        "related",
        "relating",
        "regarding",
        "concerning",
        "from",
        "last",
        "past",
        "recent",
        "years",
        "year",
        "the",
        "for",
        "and",
        "with",
        "that",
        "this",
        "papers",
        "paper",
        "articles",
        "article",
        "publications",
        "academic",
        "scholarly",
        "research",
        "literature",
        "review",
        "studies",
        "study",
        "on",
        "in",
        "of",
        "to",
        "a",
        "an",
        "how",
        "what",
        "which",
        "does",
        "can",
        "could",
        "would",
        "should",
        "me",
        "all",
        "any",
        "some",
        "each",
        "every",
        "between",
        "within",
        "during",
        "across",
        "through",
        "into",
        "onto",
        "their",
        "its",
        "our",
        "your",
        "my",
        "these",
        "those",
    }

    cleaned = re.sub(r"\b\d{1,4}\b", " ", query.lower())
    tokens = cleaned.split()
    keywords = [t for t in tokens if t not in _FILLER and len(t) > 1]

    simplified = " ".join(keywords[:10])
    return simplified if simplified else query


async def search_semantic_scholar(
    query: str,
    limit: int = 30,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    fields_of_study: Optional[List[str]] = None,
) -> List[Paper]:
    """
    Search Semantic Scholar API for academic papers.

    Args:
        query: Search query string
        limit: Maximum number of results (max 100)
        year_from: Filter papers from this year (default: current year - 5)
        year_to: Filter papers until this year
        fields_of_study: Filter by fields like "Computer Science", "Medicine"

    Returns:
        List of Paper objects
    """
    # Default to last 5 years if not specified
    if year_from is None and year_to is None:
        year_from = datetime.now().year - 5

    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required. Install with: pip install aiohttp")

    api_key = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    simplified_query = _simplify_query_for_semantic_scholar(query)
    logger.debug("Semantic Scholar query simplified: %r -> %r", query, simplified_query)

    params = {
        "query": simplified_query,
        "limit": min(limit, 100),
        "fields": "title,authors,year,abstract,citationCount,externalIds,url",
    }

    if year_from or year_to:
        year_filter = f"{year_from or ''}-{year_to or ''}"
        params["year"] = year_filter

    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    papers = []
    max_retries = 5

    for attempt in range(max_retries + 1):
        try:
            wait = 1.0 if attempt == 0 else 2**attempt
            await asyncio.sleep(wait)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    base_url, params=params, headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for item in data.get("data", []):
                            # Extract DOI from externalIds
                            doi = None
                            ext_ids = item.get("externalIds", {})
                            if ext_ids:
                                doi = ext_ids.get("DOI")

                            paper = Paper(
                                title=item.get("title", "Unknown"),
                                authors=[
                                    a.get("name", "") for a in item.get("authors", [])
                                ],
                                year=item.get("year"),
                                abstract=item.get("abstract"),
                                doi=doi,
                                url=item.get("url"),
                                citation_count=item.get("citationCount"),
                                source="semantic_scholar",
                            )
                            papers.append(paper)
                        break  # Success, exit retry loop
                    elif response.status == 429:
                        delay = 2 ** (attempt + 1)
                        logger.warning(
                            "Semantic Scholar rate limit (429), retry %d/%d in %ds",
                            attempt + 1,
                            max_retries,
                            delay,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                "Semantic Scholar: max retries reached for rate limit"
                            )
                    elif response.status == 500:
                        delay = 2 ** (attempt + 1)
                        logger.warning(
                            "Semantic Scholar server error (500), retry %d/%d in %ds",
                            attempt + 1,
                            max_retries,
                            delay,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                "Semantic Scholar: max retries reached for server error"
                            )
                    else:
                        error_text = await response.text()
                        logger.error(
                            "Semantic Scholar API error %d: %s",
                            response.status,
                            error_text[:200],
                        )
                        break  # Non-retryable error
        except Exception as e:
            logger.error("Semantic Scholar request failed: %s", e)
            if attempt < max_retries:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                break

    return papers


def search_google_scholar(
    query: str,
    limit: int = 30,
    year_from: Optional[int] = None,
) -> List[Paper]:
    """
    Search Google Scholar via SerpAPI with pagination.

    Args:
        query: Search query string
        limit: Maximum number of results (supports pagination beyond 20)
        year_from: Filter papers from this year (default: current year - 5)

    Returns:
        List of Paper objects
    """
    import re

    # Default to last 5 years if not specified
    if year_from is None:
        year_from = datetime.now().year - 5

    if not HAS_SERPAPI:
        raise ImportError(
            "serpapi is required. Install with: pip install google-search-results"
        )

    api_key = os.getenv("SERPAPI_KEY", "")
    if not api_key:
        raise ValueError("SERPAPI_KEY environment variable is required")

    papers = []
    per_page = 20  # Google Scholar max per page
    offset = 0

    while len(papers) < limit:
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": api_key,
            "num": min(per_page, limit - len(papers)),
            "start": offset,
        }

        if year_from:
            params["as_ylo"] = year_from

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as e:
            logger.error("Google Scholar request failed: %s", e)
            break

        organic = results.get("organic_results", [])
        if not organic:
            break  # No more results

        for item in organic:
            # Parse authors from publication_info
            authors = []
            pub_info = item.get("publication_info", {})
            if "authors" in pub_info:
                authors = [a.get("name", "") for a in pub_info.get("authors", [])]

            # Extract year from publication info
            year = None
            summary = pub_info.get("summary", "")
            year_match = re.search(r"\b(19|20)\d{2}\b", summary)
            if year_match:
                year = int(year_match.group())

            paper = Paper(
                title=item.get("title", "Unknown"),
                authors=authors,
                year=year,
                abstract=item.get("snippet"),
                doi=None,  # Google Scholar doesn't directly provide DOI
                url=item.get("link"),
                citation_count=item.get("inline_links", {})
                .get("cited_by", {})
                .get("total"),
                source="google_scholar",
            )
            papers.append(paper)

        offset += per_page

        # Safety: max 3 pages to avoid burning SerpAPI credits
        if offset >= 60:
            break

    return papers[:limit]


def _build_scopus_query(query: str) -> str:
    """Convert a natural-language query to Scopus TITLE-ABS-KEY() syntax.

    Scopus returns 0 results on free-text sentences.  This strips filler words,
    expands common abbreviations that Scopus doesn't index, and limits to the
    top-5 keywords joined with AND to avoid over-constraining.
    """
    import re

    if "TITLE-ABS-KEY" in query.upper():
        return query

    _ABBREVIATIONS = {
        "genai": "generative artificial intelligence",
        "llm": "large language model",
        "llms": "large language models",
        "nlp": "natural language processing",
        "ml": "machine learning",
        "dl": "deep learning",
        "iot": "internet of things",
        "smes": "small medium enterprises",
        "sme": "small medium enterprise",
        "ai": "artificial intelligence",
        "xai": "explainable artificial intelligence",
        "rag": "retrieval augmented generation",
    }

    _FILLER = {
        "find",
        "search",
        "look",
        "get",
        "retrieve",
        "fetch",
        "show",
        "list",
        "give",
        "provide",
        "about",
        "related",
        "relating",
        "regarding",
        "concerning",
        "from",
        "last",
        "past",
        "recent",
        "years",
        "year",
        "the",
        "for",
        "and",
        "with",
        "that",
        "this",
        "papers",
        "paper",
        "articles",
        "article",
        "publications",
        "academic",
        "scholarly",
        "research",
        "literature",
        "review",
        "studies",
        "study",
        "on",
        "in",
        "of",
        "to",
        "a",
        "an",
        "how",
        "what",
        "which",
        "does",
        "can",
        "could",
        "would",
        "should",
        "me",
        "all",
        "any",
        "some",
        "each",
        "every",
        "between",
        "within",
        "during",
        "across",
        "through",
        "into",
        "onto",
        "their",
        "its",
        "our",
        "your",
        "my",
        "these",
        "those",
        "deploying",
        "implementing",
        "using",
        "solutions",
        "approaches",
        "methods",
        "techniques",
        "based",
        "use",
        "used",
    }

    cleaned = re.sub(r"\b\d{1,4}\b", " ", query.lower())
    tokens = cleaned.split()

    expanded_tokens = []
    for t in tokens:
        if t in _ABBREVIATIONS:
            expanded_tokens.extend(_ABBREVIATIONS[t].split())
        elif t not in _FILLER and len(t) > 1:
            expanded_tokens.append(t)

    seen = set()
    keywords = []
    for t in expanded_tokens:
        if t not in seen:
            seen.add(t)
            keywords.append(t)
        if len(keywords) >= 5:
            break

    if not keywords:
        return f"TITLE-ABS-KEY({query})"

    return "TITLE-ABS-KEY(" + " AND ".join(keywords) + ")"


async def search_scopus(
    query: str,
    limit: int = 30,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Paper]:
    """
    Search Scopus API for academic papers.
    """
    # Default to last 5 years if not specified
    if year_from is None and year_to is None:
        year_from = datetime.now().year - 5

    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required")

    api_key = os.getenv("SCOPUS_API_KEY", "")
    if not api_key:
        logger.warning("SCOPUS_API_KEY not found, skipping Scopus search")
        return []

    base_url = "https://api.elsevier.com/content/search/scopus"

    structured_query = _build_scopus_query(query)
    logger.debug("Scopus query built: %r -> %r", query, structured_query)

    date_query = ""
    if year_from and year_to:
        date_query = f" AND PUBYEAR > {year_from - 1} AND PUBYEAR < {year_to + 1}"
    elif year_from:
        date_query = f" AND PUBYEAR > {year_from - 1}"

    final_query = f"{structured_query}{date_query}"

    params = {
        "query": final_query,
        "count": min(limit, 25),  # Scopus free tier hard-caps at 25
        "sort": "-citedby-count",
    }

    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}

    papers = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                base_url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    entries = data.get("search-results", {}).get("entry", [])

                    for item in entries:
                        # Skip error entries (Scopus returns these for empty
                        # result sets instead of an empty list)
                        if "error" in item:
                            logger.debug(
                                "Scopus returned error entry: %s",
                                item.get("error"),
                            )
                            continue

                        title = item.get("dc:title", "Unknown")
                        # Skip entries that still lack a title after parsing
                        if title == "Unknown":
                            continue

                        pub_date = item.get("prism:coverDate", "")
                        year = int(pub_date[:4]) if pub_date else None

                        author_name = item.get("dc:creator", "Unknown")
                        authors = [author_name] if author_name else []

                        doi = item.get("prism:doi")
                        url = None
                        for link in item.get("link", []):
                            if link.get("@ref") == "scopus":
                                url = link.get("@href")
                                break

                        citations = item.get("citedby-count")
                        if citations:
                            citations = int(citations)

                        paper = Paper(
                            title=title,
                            authors=authors,
                            year=year,
                            abstract=None,
                            doi=doi,
                            url=url,
                            citation_count=citations,
                            source="scopus",
                        )
                        papers.append(paper)
                else:
                    logger.error("Scopus API error %d", response.status)
    except Exception as e:
        logger.error("Scopus request failed: %s", e)

    return papers


def search_scopus_sync(
    query: str,
    limit: int = 30,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Paper]:
    """Synchronous wrapper for search_scopus."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, search_scopus(query, limit, year_from, year_to)
            ).result()
    return asyncio.run(search_scopus(query, limit, year_from, year_to))


def format_papers_as_markdown(papers: List[Any], title: str = "Search Results") -> str:
    """Format a list of papers (objects or dicts) as markdown."""
    if not papers:
        return f"## {title}\n\nNo papers found."

    output = [f"## {title}\n", f"Found {len(papers)} papers:\n"]

    for i, paper in enumerate(papers, 1):
        if hasattr(paper, "to_markdown"):
            output.append(f"{i}. {paper.to_markdown()}\n")
        elif isinstance(paper, dict):
            # Handle dictionary case manually
            title = paper.get("title", "Unknown")
            year = paper.get("year", "Unknown")
            try:
                authors = paper.get("authors", [])
                if isinstance(authors, list):
                    authors_str = ", ".join(authors[:3])
                    if len(authors) > 3:
                        authors_str += " et al."
                else:
                    authors_str = str(authors)
            except (TypeError, AttributeError):
                authors_str = "Unknown"

            output.append(f"{i}. **{title}** ({year})\n   {authors_str}\n")
        else:
            output.append(f"{i}. {str(paper)}\n")

    return "\n".join(output)


def save_papers_to_json(papers: List[Paper], filepath: str) -> None:
    """Save papers to a JSON file."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "count": len(papers),
        "papers": [p.to_dict() for p in papers],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Synchronous wrapper for use without async
def search_semantic_scholar_sync(
    query: str,
    limit: int = 30,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Paper]:
    """Synchronous wrapper for search_semantic_scholar."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, search_semantic_scholar(query, limit, year_from, year_to)
            ).result()
    return asyncio.run(search_semantic_scholar(query, limit, year_from, year_to))


# Tool metadata for agent integration
# Priority Order: OpenAlex -> GoogleScholar -> Scopus -> Semantic
LITERATURE_TOOLS = [
    {
        "name": "search_openalex",
        "description": "Search OpenAlex (PRIORITY 1) - Open catalog of 250M+ scholarly works. Returns title, authors, abstract, citations, DOI.",
        "function": search_openalex_sync,
        "parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Max results (1-200)",
                "required": False,
                "default": 30,
            },
            {
                "name": "year_from",
                "type": "integer",
                "description": "Filter papers from year",
                "required": False,
            },
            {
                "name": "year_to",
                "type": "integer",
                "description": "Filter papers until year",
                "required": False,
            },
        ],
    },
    {
        "name": "search_google_scholar",
        "description": "Search Google Scholar (PRIORITY 2) via SerpAPI. Good for discovering papers across all disciplines.",
        "function": search_google_scholar,
        "parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Max results (1-20)",
                "required": False,
                "default": 20,
            },
            {
                "name": "year_from",
                "type": "integer",
                "description": "Filter papers from year",
                "required": False,
            },
        ],
    },
    {
        "name": "search_scopus",
        "description": "Search Scopus (PRIORITY 3) - Elsevier's abstract and citation database. High-impact Q1/Q2 papers.",
        "function": search_scopus_sync,
        "parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Max results (1-25)",
                "required": False,
                "default": 25,
            },
            {
                "name": "year_from",
                "type": "integer",
                "description": "Filter papers from year",
                "required": False,
            },
            {
                "name": "year_to",
                "type": "integer",
                "description": "Filter papers until year",
                "required": False,
            },
        ],
    },
    {
        "name": "search_semantic_scholar",
        "description": "Search Semantic Scholar (PRIORITY 4) for academic papers. Returns title, authors, abstract, citation count, and DOI.",
        "function": search_semantic_scholar_sync,
        "parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Max results (1-100)",
                "required": False,
                "default": 30,
            },
            {
                "name": "year_from",
                "type": "integer",
                "description": "Filter papers from year",
                "required": False,
            },
            {
                "name": "year_to",
                "type": "integer",
                "description": "Filter papers until year",
                "required": False,
            },
        ],
    },
]
