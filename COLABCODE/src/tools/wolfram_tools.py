"""
Wolfram Alpha API Tools for Mathematical Model Auditing

This module provides comprehensive functions to interact with Wolfram Alpha Full Results API
for auditing mathematical models, verifying proofs, validating methodology, and performing
advanced symbolic computation.

PERFORMANCE OPTIMIZATIONS:
=========================
- In-memory caching for repeated queries (hash-based key)
- Rate limiting (<=1 req/s, safe for research)
- Exponential backoff retry on failures
- Optimized query parameters (includepodid=Result, format=plaintext)

CAPABILITIES:
============
1. SYMBOLIC COMPUTATION
   - Equation verification (LHS = RHS)
   - Expression simplification
   - Symbolic solving

2. CALCULUS
   - Derivatives (including nth order)
   - Integrals (definite and indefinite)
   - Limits
   - Series expansions

3. LINEAR ALGEBRA
   - Matrix operations (determinant, inverse, eigenvalues)
   - System of linear equations
   - Matrix decomposition

4. DIFFERENTIAL EQUATIONS
   - ODE solving
   - System of ODEs
   - Initial value problems

5. PROBABILITY & STATISTICS
   - Distribution properties (mean, variance, PDF, CDF)
   - Probability calculations
   - Monte Carlo estimation

6. OPTIMIZATION
   - Minimize/Maximize functions
   - Constrained optimization
   - Linear programming

7. MODEL VALIDATION
   - Dimensional analysis
   - Boundary condition checking
   - Parameter sensitivity

8. LLM API (Natural Language)
   - Natural language questions
   - Quick fact lookups
   - LLM-optimized text responses
   - Automatic image URL extraction
"""

import os
import json
import hashlib
import time
import logging
import urllib.parse
import subprocess
import sys
import importlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import datetime
from pathlib import Path
import requests as _requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

WOLFRAM_APPID = os.getenv("WOLFRAM_APPID")
WOLFRAM_APPID_BACKUP = os.getenv("WOLFRAM_APPID_BACKUP")
WOLFRAM_API_URL = "http://api.wolframalpha.com/v2/query"
WOLFRAM_LLM_API_URL = "https://www.wolframalpha.com/api/v1/llm-api"
WOLFRAM_SHORT_ANSWER_URL = "http://api.wolframalpha.com/v1/result"


# =============================================================================
# PERFORMANCE: CACHING & RATE LIMITING & FAILOVER
# =============================================================================

# In-memory cache for query results
_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 3600  # 1 hour cache TTL
_CACHE_FILE = Path(__file__).parent.parent.parent / ".cache" / "wolfram_cache.json"

# Rate limiting state
_RATE_LIMIT_STATE = {
    "last_call": 0.0,
    "min_interval": 1.0,  # 1 second between requests (safe for research)
}

# AppID failover state
_APPID_STATE = {
    "current": WOLFRAM_APPID,
    "backup": WOLFRAM_APPID_BACKUP,
    "using_backup": False,
    "primary_quota_exceeded": False,
}


def _get_current_appid() -> Optional[str]:
    """Get current AppID, switching to backup if primary quota exceeded."""
    if _APPID_STATE["primary_quota_exceeded"] and _APPID_STATE["backup"]:
        return _APPID_STATE["backup"]
    return _APPID_STATE["current"]


def _switch_to_backup_appid() -> bool:
    """Switch to backup AppID. Returns True if switch successful."""
    if _APPID_STATE["backup"] and not _APPID_STATE["using_backup"]:
        _APPID_STATE["primary_quota_exceeded"] = True
        _APPID_STATE["using_backup"] = True
        return True
    return False


def reset_appid_to_primary() -> None:
    """Reset to use primary AppID (call when quota refreshes)."""
    _APPID_STATE["primary_quota_exceeded"] = False
    _APPID_STATE["using_backup"] = False


def get_appid_status() -> Dict[str, Any]:
    """Get current AppID status."""
    return {
        "using_backup": _APPID_STATE["using_backup"],
        "primary_available": bool(_APPID_STATE["current"]),
        "backup_available": bool(_APPID_STATE["backup"]),
    }


def _get_cache_key(
    input_query: str,
    output: str,
    format: str,
    includepodid: Optional[str],
    assumptions: Optional[str] = None,
) -> str:
    """Generate cache key from query parameters."""
    key_data = (
        f"{input_query}|{output}|{format}|{includepodid or ''}|{assumptions or ''}"
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get result from cache if exists and not expired."""
    if cache_key in _QUERY_CACHE:
        cached = _QUERY_CACHE[cache_key]
        if time.time() - cached.get("timestamp", 0) < _CACHE_TTL:
            return cached.get("result")
    return None


def _set_cached_result(cache_key: str, result: Any) -> None:
    """Store result in cache."""
    _QUERY_CACHE[cache_key] = {"result": result, "timestamp": time.time()}
    # Limit cache size (keep most recent 500 entries)
    if len(_QUERY_CACHE) > 500:
        oldest_key = min(
            _QUERY_CACHE.keys(), key=lambda k: _QUERY_CACHE[k].get("timestamp", 0)
        )
        del _QUERY_CACHE[oldest_key]
    _save_cache_to_disk()


def _rate_limit_wait() -> None:
    """Wait if needed to respect rate limit."""
    now = time.time()
    elapsed = now - _RATE_LIMIT_STATE["last_call"]
    wait_time = _RATE_LIMIT_STATE["min_interval"] - elapsed

    if wait_time > 0:
        time.sleep(wait_time)

    _RATE_LIMIT_STATE["last_call"] = time.time()


def _exponential_backoff_retry(func, max_retries: int = 3, initial_delay: float = 1.0):
    """Execute function with exponential backoff retry on failure."""
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff

    raise last_error  # type: ignore


def clear_wolfram_cache() -> int:
    """Clear the in-memory cache. Returns number of entries cleared."""
    count = len(_QUERY_CACHE)
    _QUERY_CACHE.clear()
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
    logger.info("Cleared %d Wolfram cache entries (memory + disk)", count)
    return count


def _load_cache_from_disk() -> None:
    global _QUERY_CACHE
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            now = time.time()
            _QUERY_CACHE = {
                k: v for k, v in raw.items() if now - v.get("timestamp", 0) < _CACHE_TTL
            }
            logger.info("Loaded %d cached Wolfram results from disk", len(_QUERY_CACHE))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load Wolfram cache: %s", e)
            _QUERY_CACHE = {}


def _save_cache_to_disk() -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        valid = {
            k: v
            for k, v in _QUERY_CACHE.items()
            if now - v.get("timestamp", 0) < _CACHE_TTL
        }
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(valid, f, default=str)
        logger.debug("Saved %d Wolfram cache entries to disk", len(valid))
    except (IOError, TypeError) as e:
        logger.warning("Failed to persist Wolfram cache: %s", e)


def _ensure_sympy_installed():
    """
    Check if SymPy is installed, and if not, install it automatically.
    This ensures local math verification is always available.
    """
    try:
        import sympy

        return True
    except ImportError:
        logger.info("SymPy not found, auto-installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sympy"])
            logger.info("SymPy installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to auto-install SymPy: %s", e)
            return False
        except Exception as e:
            logger.warning("Unexpected error installing SymPy: %s", e)
            return False


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "size": len(_QUERY_CACHE),
        "ttl_seconds": _CACHE_TTL,
        "rate_limit_interval": _RATE_LIMIT_STATE["min_interval"],
    }


# =============================================================================
# DATA CLASSES
# =============================================================================

_load_cache_from_disk()


@dataclass
class WolframResult:
    """Represents a Wolfram Alpha query result."""

    success: bool
    input_interpretation: Optional[str] = None
    result: Optional[str] = None
    pods: Optional[List[Dict[str, Any]]] = None
    assumptions: Optional[List[Dict[str, Any]]] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    step_by_step: Optional[str] = None


@dataclass
class LLMResult:
    """Represents a Wolfram LLM API result (plain text optimized for LLMs)."""

    success: bool
    query: Optional[str] = None
    response: Optional[str] = None  # Plain text response
    interpretation: Optional[str] = None
    wolfram_url: Optional[str] = None  # Link to full results
    image_urls: Optional[List[str]] = None
    error: Optional[str] = None


@dataclass
class AuditResult:
    """Represents a mathematical audit result."""

    passed: bool
    category: str  # equation, derivative, integral, matrix, ode, optimization, etc.
    input_expression: str
    expected: Optional[str] = None
    computed: Optional[str] = None
    difference: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# CORE API FUNCTIONS
# =============================================================================


def wolfram_query(
    input_query: str,
    output: str = "json",
    format: str = "plaintext",
    includepodid: Optional[str] = None,
    excludepodid: Optional[str] = None,
    reinterpret: bool = True,
    timeout: int = 30,
    podstate: Optional[str] = None,
    use_cache: bool = True,
    max_retries: int = 3,
) -> WolframResult:
    """
    Execute a query against Wolfram Alpha Full Results API.

    PERFORMANCE FEATURES:
    - Caching: Repeated queries return cached results (1hr TTL)
    - Rate limiting: Max 1 request/second to avoid throttling
    - Retry: Exponential backoff on failures (default 3 retries)

    Args:
        input_query: The mathematical expression or question to evaluate
        output: Response format ('json' or 'xml')
        format: Content format in pods ('plaintext', 'mathml', 'image', 'minput')
        includepodid: Specific pod IDs to include (comma-separated)
        excludepodid: Pod IDs to exclude
        reinterpret: Allow Wolfram to reinterpret ambiguous queries
        timeout: Request timeout in seconds
        podstate: Request additional pod states (e.g., 'Step-by-step solution')
        use_cache: Whether to use caching (default True)
        max_retries: Number of retry attempts with exponential backoff

    Returns:
        WolframResult with parsed response data
    """
    current_appid = _get_current_appid()
    if not current_appid:
        return WolframResult(
            success=False,
            error="WOLFRAM_APPID not found in environment variables. Add WOLFRAM_APPID to .env file.",
        )

    # Check cache first
    cache_key = _get_cache_key(input_query, output, format, includepodid)
    if use_cache:
        cached = _get_cached_result(cache_key)
        if cached is not None:
            return cached  # type: ignore

    # Build query parameters (use current AppID with failover support)
    params = {
        "appid": current_appid,
        "input": input_query,
        "output": output,
        "format": format,
    }

    if reinterpret:
        params["reinterpret"] = "true"
    if includepodid:
        params["includepodid"] = includepodid
    if excludepodid:
        params["excludepodid"] = excludepodid
    if podstate:
        params["podstate"] = podstate

    # Build URL
    query_string = urllib.parse.urlencode(params)
    url = f"{WOLFRAM_API_URL}?{query_string}"

    def make_request() -> WolframResult:
        """Inner function for retry logic."""
        nonlocal url  # Fix: allow reassignment of outer url variable
        # Rate limiting
        _rate_limit_wait()

        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.text

        if output == "json":
            result_data = json.loads(data)
            parsed = _parse_json_response(result_data)

            # Check for quota exceeded error and try backup
            if (
                not parsed.success
                and parsed.error
                and "Invalid appid" in str(parsed.error)
            ):
                if _switch_to_backup_appid():
                    # Retry with backup AppID
                    params["appid"] = _get_current_appid()
                    query_string = urllib.parse.urlencode(params)
                    url = f"{WOLFRAM_API_URL}?{query_string}"
                    return make_request()  # Recursive retry with new AppID

            return parsed
        else:
            return WolframResult(success=True, result=data, raw_response={"xml": data})

    try:
        # Execute with retry
        result = _exponential_backoff_retry(make_request, max_retries=max_retries)

        # Cache successful results
        if use_cache and result.success:
            _set_cached_result(cache_key, result)

        return result

    except _requests.exceptions.HTTPError as e:
        # Check for 403 (quota exceeded) and try backup
        status_code = e.response.status_code if e.response else 0
        reason = e.response.reason if e.response else "unknown"
        if status_code == 403 and _switch_to_backup_appid():
            return wolfram_query(
                input_query,
                output,
                format,
                includepodid,
                excludepodid,
                reinterpret,
                timeout,
                podstate,
                use_cache,
                max_retries,
            )
        return WolframResult(success=False, error=f"HTTP error {status_code}: {reason}")
    except _requests.exceptions.ConnectionError as e:
        return WolframResult(success=False, error=f"Network error: {e}")
    except json.JSONDecodeError as e:
        return WolframResult(success=False, error=f"JSON parse error: {e}")
    except Exception as e:
        return WolframResult(success=False, error=f"Unexpected error: {e}")


def _parse_json_response(data: Dict[str, Any]) -> WolframResult:
    """Parse Wolfram Alpha JSON response into WolframResult."""
    queryresult = data.get("queryresult", {})

    success = queryresult.get("success", False)
    error = queryresult.get("error", False)

    if error and isinstance(error, dict):
        return WolframResult(
            success=False, error=error.get("msg", "Unknown error"), raw_response=data
        )

    pods = queryresult.get("pods", [])

    # Extract key information from pods
    input_interpretation = None
    result = None
    step_by_step = None
    warnings = []

    for pod in pods:
        pod_id = pod.get("id", "")
        subpods = pod.get("subpods", [])

        if pod_id == "Input" and subpods:
            input_interpretation = subpods[0].get("plaintext", "")
        elif pod_id == "Result" and subpods:
            result = subpods[0].get("plaintext", "")
        elif "Result" in pod_id and subpods and not result:
            result = subpods[0].get("plaintext", "")
        elif "Step-by-step" in pod.get("title", "") and subpods:
            step_by_step = subpods[0].get("plaintext", "")

    # Fallback: get first non-input pod result
    if not result and len(pods) > 1:
        for pod in pods:
            if pod.get("id") != "Input":
                subpods = pod.get("subpods", [])
                if subpods:
                    result = subpods[0].get("plaintext", "")
                    break

    # Extract warnings
    if "warnings" in queryresult:
        for w in queryresult.get("warnings", {}).get("warning", []):
            warnings.append(w.get("text", ""))

    return WolframResult(
        success=success,
        input_interpretation=input_interpretation,
        result=result,
        pods=pods,
        assumptions=queryresult.get("assumptions"),
        warnings=warnings if warnings else None,
        step_by_step=step_by_step,
        raw_response=data,
    )


# =============================================================================
# LLM API FUNCTIONS (Natural Language Interface)
# =============================================================================


def preprocess_query_for_wolfram(query: str) -> str:
    """
    Convert natural language to keyword query for better Wolfram Alpha results.

    Wolfram Alpha's LLM API works best with keyword-style queries rather than
    full natural language sentences. This function transforms common patterns.

    Transformations:
    - "What is the X of Y" -> "Y X"
    - "How many X in Y" -> "Y X"
    - "Explain X" -> "X definition"
    - "Calculate X" -> "X"
    - Removes filler words

    Args:
        query: Natural language query

    Returns:
        Optimized keyword query

    Example:
        >>> preprocess_query_for_wolfram("What is the population of France?")
        'France population'
        >>> preprocess_query_for_wolfram("How far is Earth from the Moon?")
        'Earth Moon distance'
    """
    import re

    q = query.strip()

    # Remove common filler phrases (do this FIRST)
    fillers = [
        r"^(?:please\s+)?",
        r"^(?:i\s+want\s+to\s+know\s+)?",
        r"^(?:i\s+need\s+to\s+find\s+)?",
        r"^(?:help\s+me\s+)?",
        r"^(?:use\s+wolfram\s+(?:alpha\s+)?to\s+)?",
        r"^(?:ask\s+wolfram\s+(?:alpha\s+)?to\s+)?",
    ]
    for filler in fillers:
        q = re.sub(filler, "", q, flags=re.IGNORECASE).strip()

    # Remove trailing punctuation
    q = re.sub(r"[?!.]+$", "", q).strip()

    # Pattern: "What is the X of Y" -> "Y X"
    match = re.match(
        r"(?:what|what's|whats)\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+(?:of|for|in)\s+(.+)",
        q,
        re.IGNORECASE,
    )
    if match:
        return f"{match.group(2)} {match.group(1)}"

    # Pattern: "What is X" -> "X"
    match = re.match(
        r"(?:what|what's|whats)\s+(?:is|are)\s+(?:the\s+)?(.+)", q, re.IGNORECASE
    )
    if match:
        return match.group(1)

    # Pattern: "How many X in/of Y" -> "Y X"
    match = re.match(
        r"how\s+(?:many|much)\s+(.+?)\s+(?:in|of|are\s+in|are\s+there\s+in)\s+(.+)",
        q,
        re.IGNORECASE,
    )
    if match:
        return f"{match.group(2)} {match.group(1)}"

    # Pattern: "How far is X from Y" -> "X Y distance"
    match = re.match(
        r"how\s+far\s+is\s+(.+?)\s+from\s+(?:the\s+)?(.+)", q, re.IGNORECASE
    )
    if match:
        return f"{match.group(1)} {match.group(2)} distance"

    # Pattern: "How X is Y" -> "Y X" (e.g., "how tall is Mount Everest")
    match = re.match(r"how\s+(\w+)\s+is\s+(.+)", q, re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"

    # Pattern: "Explain X" or "Tell me about X" -> "X definition"
    match = re.match(
        r"(?:explain|tell\s+me\s+about|describe)\s+(?:the\s+)?(.+)", q, re.IGNORECASE
    )
    if match:
        content = match.group(1)
        # For math concepts, just return as-is
        if any(
            word in content.lower()
            for word in ["derivative", "integral", "equation", "formula", "function"]
        ):
            return content
        return f"{content} definition"

    # Pattern: "Calculate X" or "Compute X" -> just X
    match = re.match(
        r"(?:calculate|compute|find|solve|evaluate)\s+(?:the\s+)?(.+)", q, re.IGNORECASE
    )
    if match:
        return match.group(1)

    # Pattern: "Can you X" or "Could you X" -> just X
    match = re.match(
        r"(?:can|could|would|will)\s+you\s+(?:please\s+)?(.+)", q, re.IGNORECASE
    )
    if match:
        return preprocess_query_for_wolfram(match.group(1))  # Recursively process

    return q


def _parse_llm_response(response_text: str) -> LLMResult:
    """Parse LLM API plain text response."""
    import re

    lines = response_text.strip().split("\n")

    # Extract query (first line usually starts with "Query:")
    query = None
    interpretation = None
    wolfram_url = None
    image_urls = []

    for line in lines:
        if line.startswith("Query:"):
            query = line.replace("Query:", "").strip().strip('"')
        elif line.startswith("Input interpretation:"):
            interpretation = line.replace("Input interpretation:", "").strip()
        elif "wolframalpha.com/input" in line:
            # Extract Wolfram URL
            url_match = re.search(
                r"https://www\.wolframalpha\.com/input\?i=[^\s]+", line
            )
            if url_match:
                wolfram_url = url_match.group(0)
        elif "image:" in line.lower():
            # Extract image URLs
            img_matches = re.findall(r"https://[^\s]+\.(?:png|jpg|gif)[^\s]*", line)
            image_urls.extend(img_matches)

    return LLMResult(
        success=True,
        query=query,
        response=response_text,
        interpretation=interpretation,
        wolfram_url=wolfram_url,
        image_urls=image_urls if image_urls else None,
    )


def wolfram_llm_query(
    input_query: str,
    maxchars: int = 6800,
    timeout: int = 30,
    use_cache: bool = True,
    use_bearer_token: bool = False,
    max_retries: int = 3,
    preprocess: bool = False,
) -> LLMResult:
    """
    Query Wolfram Alpha LLM API for keyword-based queries.

    IMPORTANT: This is NOT a conversational AI. It's Wolfram Alpha's
    knowledge engine with LLM-optimized output format.

    WORKS WELL FOR:
    - Keyword queries: "France population", "speed of light"
    - Math expressions: "derivative of sin(x)", "integrate x^2"
    - Factual lookups: "Tokyo population", "Einstein birth date"

    DOES NOT WORK FOR:
    - Open-ended questions: "Explain how CNN works"
    - Conversational requests: "Tell me about deep learning"

    PERFORMANCE: Uses same caching/rate limiting as Full Results API.

    Args:
        input_query: Keyword query or math expression
        maxchars: Max response length (default 6800)
        timeout: Request timeout in seconds
        use_cache: Enable caching (1hr TTL)
        use_bearer_token: Use Authorization header instead of query param
        max_retries: Retry attempts with exponential backoff
        preprocess: Convert natural language to keywords (default False)

    Returns:
        LLMResult with plain text response, extracted URLs, etc.

    Example:
        >>> result = wolfram_llm_query("France population")
        >>> result = wolfram_llm_query("What is the population of France?", preprocess=True)
    """
    # Apply preprocessing if enabled
    query = preprocess_query_for_wolfram(input_query) if preprocess else input_query

    current_appid = _get_current_appid()
    if not current_appid:
        return LLMResult(
            success=False,
            query=query,
            error="WOLFRAM_APPID not found. Add WOLFRAM_APPID to .env file.",
        )

    # Generate cache key for LLM queries (use preprocessed query)
    cache_key = f"llm_{hashlib.sha256(f'{query}|{maxchars}'.encode()).hexdigest()[:32]}"
    if use_cache:
        cached = _get_cached_result(cache_key)
        if cached is not None:
            return cached  # type: ignore

    # Build query parameters
    params = {
        "input": query,
        "maxchars": str(maxchars),
    }

    # Add appid as param or use bearer token
    headers = {}
    if use_bearer_token:
        headers["Authorization"] = f"Bearer {current_appid}"
    else:
        params["appid"] = current_appid

    query_string = urllib.parse.urlencode(params)
    url = f"{WOLFRAM_LLM_API_URL}?{query_string}"

    def make_request() -> LLMResult:
        """Inner function for retry logic."""
        nonlocal url, headers

        # Rate limiting
        _rate_limit_wait()

        try:
            resp = _requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.text

            # Parse the plain text response
            result = _parse_llm_response(data)
            result.query = query
            return result

        except _requests.exceptions.HTTPError as e:
            error_body = e.response.text if e.response else ""
            status_code = e.response.status_code if e.response else 0
            reason = e.response.reason if e.response else "unknown"

            # Check for 501 (query not understood)
            if status_code == 501:
                return LLMResult(
                    success=False,
                    query=query,
                    response=error_body,
                    error=f"Query not understood (HTTP 501). Suggestions: {error_body}",
                )
            # Check for 403 (invalid/missing appid)
            elif status_code == 403:
                if _switch_to_backup_appid():
                    # Retry with backup
                    if not use_bearer_token:
                        params["appid"] = _get_current_appid()
                        query_string = urllib.parse.urlencode(params)
                        url = f"{WOLFRAM_LLM_API_URL}?{query_string}"
                    else:
                        headers["Authorization"] = f"Bearer {_get_current_appid()}"
                    return make_request()
                return LLMResult(
                    success=False,
                    query=query,
                    error=f"Invalid or missing AppID (HTTP 403)",
                )
            else:
                return LLMResult(
                    success=False,
                    query=query,
                    error=f"HTTP error {status_code}: {reason}",
                )

    try:
        result = _exponential_backoff_retry(make_request, max_retries=max_retries)

        # Cache successful results
        if use_cache and result.success:
            _set_cached_result(cache_key, result)

        return result

    except _requests.exceptions.ConnectionError as e:
        return LLMResult(success=False, query=query, error=f"Network error: {e}")
    except Exception as e:
        return LLMResult(success=False, query=query, error=f"Unexpected error: {e}")


def llm_natural_query(question: str, maxchars: int = 4000) -> Dict[str, Any]:
    """
    Ask a natural language question to Wolfram Alpha via LLM API.

    Automatically preprocesses questions to keyword format.
    Best for:
    - "What is the X of Y?" questions
    - "How many X in Y?" questions
    - General knowledge queries

    Args:
        question: Natural language question (will be preprocessed)
        maxchars: Max response length

    Returns:
        Dict with 'success', 'answer', 'url', 'error'
    """
    result = wolfram_llm_query(question, maxchars=maxchars, preprocess=True)

    return {
        "success": result.success,
        "question": question,
        "answer": result.response,
        "interpretation": result.interpretation,
        "url": result.wolfram_url,
        "images": result.image_urls,
        "error": result.error,
    }


def llm_quick_compute(expression: str, maxchars: int = 2000) -> Dict[str, Any]:
    """
    Quick mathematical computation via LLM API.

    Use this for simple computations when you just need the answer
    without full structured data. For complex auditing, use the
    Full Results API functions instead.

    Args:
        expression: Math expression or computation request
        maxchars: Max response length

    Returns:
        Dict with 'success', 'result', 'error'

    Example:
        >>> llm_quick_compute("derivative of x^3")
        >>> llm_quick_compute("integral of sin(x)")
        >>> llm_quick_compute("solve x^2 - 4 = 0")
    """
    result = wolfram_llm_query(expression, maxchars=maxchars)

    return {
        "success": result.success,
        "expression": expression,
        "result": result.response,
        "interpretation": result.interpretation,
        "url": result.wolfram_url,
        "error": result.error,
    }


def llm_fact_lookup(topic: str, maxchars: int = 3000) -> Dict[str, Any]:
    """
    Quick factual information lookup via LLM API.

    Optimized for looking up facts about entities:
    - Countries, cities, people
    - Scientific constants, elements
    - Dates, events, statistics

    Args:
        topic: Entity or fact to look up
        maxchars: Max response length

    Returns:
        Dict with 'success', 'facts', 'images', 'error'

    Example:
        >>> llm_fact_lookup("speed of light")
        >>> llm_fact_lookup("Tokyo population")
        >>> llm_fact_lookup("Einstein birth date")
    """
    result = wolfram_llm_query(topic, maxchars=maxchars)

    return {
        "success": result.success,
        "topic": topic,
        "facts": result.response,
        "interpretation": result.interpretation,
        "images": result.image_urls,
        "url": result.wolfram_url,
        "error": result.error,
    }


# =============================================================================
# SHORT ANSWERS API (Quick Single-Line Results)
# =============================================================================


def wolfram_short_answer(
    query: str,
    units: str = "metric",
    timeout: int = 5,
    use_cache: bool = True,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Get a single-line text answer from Wolfram Alpha.

    The Short Answers API returns just the result - no pods, no images,
    just plain text. Perfect for quick math verification.

    Args:
        query: Math expression or factual query
        units: "metric" or "imperial" for measurements
        timeout: Request timeout in seconds (default 5)
        use_cache: Enable caching
        max_retries: Retry attempts

    Returns:
        Dict with 'success', 'answer', 'error'

    Example:
        >>> wolfram_short_answer("derivative of x^3")
        {'success': True, 'answer': '3 x^2', 'error': None}
    """
    current_appid = _get_current_appid()
    if not current_appid:
        return {
            "success": False,
            "query": query,
            "answer": None,
            "error": "WOLFRAM_APPID not found. Add WOLFRAM_APPID to .env file.",
        }

    # Generate cache key
    cache_key = f"short_{hashlib.sha256(f'{query}|{units}'.encode()).hexdigest()[:32]}"
    if use_cache:
        cached = _get_cached_result(cache_key)
        if cached is not None:
            return cached  # type: ignore

    # Build query parameters
    params = {
        "appid": current_appid,
        "i": query,
        "units": units,
        "timeout": str(timeout),
    }

    query_string = urllib.parse.urlencode(params)
    url = f"{WOLFRAM_SHORT_ANSWER_URL}?{query_string}"

    def make_request() -> Dict[str, Any]:
        nonlocal url
        _rate_limit_wait()

        try:
            resp = _requests.get(url, timeout=timeout)
            resp.raise_for_status()
            answer = resp.text.strip()

            return {"success": True, "query": query, "answer": answer, "error": None}

        except _requests.exceptions.HTTPError as e:
            error_body = e.response.text if e.response else ""
            status_code = e.response.status_code if e.response else 0
            reason = e.response.reason if e.response else "unknown"

            # 501 = query not understood
            if status_code == 501:
                return {
                    "success": False,
                    "query": query,
                    "answer": None,
                    "error": f"Query not understood: {error_body}",
                }
            # 403 = invalid appid
            elif status_code == 403:
                if _switch_to_backup_appid():
                    params["appid"] = _get_current_appid()
                    query_string = urllib.parse.urlencode(params)
                    url = f"{WOLFRAM_SHORT_ANSWER_URL}?{query_string}"
                    return make_request()
                return {
                    "success": False,
                    "query": query,
                    "answer": None,
                    "error": "Invalid or missing AppID",
                }
            else:
                return {
                    "success": False,
                    "query": query,
                    "answer": None,
                    "error": f"HTTP error {status_code}: {reason}",
                }

    try:
        result = _exponential_backoff_retry(make_request, max_retries=max_retries)

        if use_cache and result.get("success"):
            _set_cached_result(cache_key, result)

        return result

    except _requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "query": query,
            "answer": None,
            "error": f"Network error: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "answer": None,
            "error": f"Unexpected error: {e}",
        }


def quick_math_check(expression: str) -> str:
    """
    Fastest way to verify a math expression.

    Returns just the result text, or error message if failed.
    This is the simplest interface for quick math auditing.

    Args:
        expression: Math expression to evaluate

    Returns:
        Result string or error message

    Example:
        >>> quick_math_check("derivative of x^3")
        '3 x^2'
        >>> quick_math_check("integrate sin(x)")
        '-cos(x) + constant'
        >>> quick_math_check("2 + 2")
        '4'
    """
    result = wolfram_short_answer(expression)
    if result["success"]:
        return result["answer"]
    else:
        return f"[Error: {result['error']}]"


# =============================================================================
# 1. SYMBOLIC COMPUTATION
# =============================================================================


def verify_equation(
    lhs: str, rhs: str, assumptions: Optional[str] = None, domain: str = "Reals"
) -> AuditResult:
    """
    Verify if two mathematical expressions are equivalent.
    Uses FullSimplify[(LHS) - (RHS)] to check if difference is zero.

    Args:
        lhs: Left-hand side expression
        rhs: Right-hand side expression
        assumptions: Conditions (e.g., "a>0 && b>0")
        domain: Domain for simplification ("Reals", "Complexes", "Integers")

    Returns:
        AuditResult with verification status
    """
    # 1. Try Local SymPy Verification First (Free & Fast)
    # Only try if no complex assumptions since our basic SymPy tool handles pure algebra best
    if not assumptions and domain == "Reals":
        try:
            # Ensure SymPy is installed (auto-install if needed)
            if _ensure_sympy_installed():
                import src.tools.sympy_tools

                # Reload if it was previously loaded as unavailable
                if not getattr(src.tools.sympy_tools, "SYMPY_AVAILABLE", False):
                    importlib.reload(src.tools.sympy_tools)

                from src.tools.sympy_tools import verify_equation_sympy, SYMPY_AVAILABLE

                if SYMPY_AVAILABLE:
                    local_res = verify_equation_sympy(lhs, rhs)
                    if local_res.get("verified") is True:
                        return AuditResult(
                            passed=True,
                            category="Equation Verification (SymPy)",
                            input_expression=f"{lhs} == {rhs}",
                            expected="0",
                            computed="0",
                            difference="0",
                            notes=[
                                f"Verified locally using SymPy. Diff: {local_res.get('difference')}"
                            ],
                        )
        except Exception as e:
            logger.debug("Local SymPy verification skipped: %s", e)
            pass  # Fallback to Wolfram

    # 2. Fallback to Wolfram Alpha
    diff_expr = f"({lhs}) - ({rhs})"

    if domain != "Reals":
        base_query = f"FullSimplify[{diff_expr}, Element[_, {domain}]]"
    else:
        base_query = f"FullSimplify[{diff_expr}]"

    if assumptions:
        query = f"Assuming[{assumptions}, {base_query}]"
    else:
        query = base_query

    result = wolfram_query(query, format="plaintext")

    if not result.success:
        return AuditResult(
            passed=False,
            category="equation",
            input_expression=f"{lhs} = {rhs}",
            error=result.error,
        )

    difference = result.result or ""
    is_zero = difference.strip() in ("0", "0.", "0.0", "True")

    return AuditResult(
        passed=is_zero,
        category="equation",
        input_expression=f"{lhs} = {rhs}",
        expected="0",
        computed=difference,
        difference=difference if not is_zero else None,
        notes=[f"Query: {query}"] if not is_zero else [],
    )


def simplify_expression(
    expression: str, assumptions: Optional[str] = None, method: str = "FullSimplify"
) -> Dict[str, Any]:
    """
    Simplify a mathematical expression.

    Args:
        expression: Expression to simplify
        assumptions: Conditions
        method: 'Simplify', 'FullSimplify', 'Expand', 'Factor', 'TrigReduce'

    Returns:
        Dict with simplified result
    """
    if method in [
        "Simplify",
        "FullSimplify",
        "Expand",
        "Factor",
        "TrigReduce",
        "TrigExpand",
    ]:
        base_query = f"{method}[{expression}]"
    else:
        base_query = f"FullSimplify[{expression}]"

    query = f"Assuming[{assumptions}, {base_query}]" if assumptions else base_query
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "original": expression,
        "simplified": result.result,
        "method": method,
        "error": result.error,
    }


def solve_equation(
    equation: str,
    variable: Optional[str] = None,
    domain: str = "Reals",
    assumptions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Solve equations symbolically.

    Args:
        equation: Equation(s) to solve (e.g., "x^2 - 4 = 0", "x^2 == 4")
        variable: Variable(s) to solve for
        domain: Domain ("Reals", "Complexes", "Integers", "PositiveReals")
        assumptions: Additional conditions
    """
    if variable:
        query = f"Solve[{equation}, {variable}, {domain}]"
    else:
        query = f"Solve[{equation}, {domain}]"

    if assumptions:
        query = f"Assuming[{assumptions}, {query}]"

    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "equation": equation,
        "solutions": result.result,
        "domain": domain,
        "pods": result.pods,
        "error": result.error,
    }


# =============================================================================
# 2. CALCULUS
# =============================================================================


def compute_derivative(
    function: str, variable: str = "x", order: int = 1
) -> Dict[str, Any]:
    """Compute derivative of a function."""
    if order == 1:
        query = f"D[{function}, {variable}]"
    else:
        query = f"D[{function}, {{{variable}, {order}}}]"

    result = wolfram_query(query, format="plaintext")
    return {
        "success": result.success,
        "function": function,
        "derivative": result.result,
        "variable": variable,
        "order": order,
        "error": result.error,
    }


def verify_derivative(
    function: str,
    claimed_derivative: str,
    variable: str = "x",
    order: int = 1,
    assumptions: Optional[str] = None,
) -> AuditResult:
    """Verify a derivative computation."""
    computed = compute_derivative(function, variable, order)

    if not computed["success"]:
        return AuditResult(
            passed=False,
            category="derivative",
            input_expression=f"d/d{variable}[{function}]",
            error=computed["error"],
        )

    # Verify equivalence
    verify = verify_equation(computed["derivative"], claimed_derivative, assumptions)

    return AuditResult(
        passed=verify.passed,
        category="derivative",
        input_expression=f"d^{order}/d{variable}^{order}[{function}]",
        expected=claimed_derivative,
        computed=computed["derivative"],
        difference=verify.difference,
        notes=[f"Order: {order}"] if order > 1 else [],
    )


def compute_integral(
    integrand: str,
    variable: str = "x",
    lower: Optional[str] = None,
    upper: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute integral (definite or indefinite).

    Args:
        integrand: Function to integrate
        variable: Integration variable
        lower: Lower bound (if definite)
        upper: Upper bound (if definite)
    """
    if lower is not None and upper is not None:
        query = f"Integrate[{integrand}, {{{variable}, {lower}, {upper}}}]"
        integral_type = "definite"
    else:
        query = f"Integrate[{integrand}, {variable}]"
        integral_type = "indefinite"

    result = wolfram_query(query, format="plaintext")
    return {
        "success": result.success,
        "integrand": integrand,
        "integral": result.result,
        "type": integral_type,
        "bounds": (lower, upper) if lower else None,
        "error": result.error,
    }


def verify_integral(
    integrand: str,
    claimed_result: str,
    variable: str = "x",
    assumptions: Optional[str] = None,
) -> AuditResult:
    """Verify an indefinite integral by differentiating the result."""
    # Differentiate the claimed result
    diff = compute_derivative(claimed_result, variable, 1)

    if not diff["success"]:
        return AuditResult(
            passed=False,
            category="integral",
            input_expression=f"integral{integrand} d{variable}",
            error=diff["error"],
        )

    # Verify derivative equals integrand
    verify = verify_equation(diff["derivative"], integrand, assumptions)

    return AuditResult(
        passed=verify.passed,
        category="integral",
        input_expression=f"integral{integrand} d{variable}",
        expected=integrand,
        computed=diff["derivative"],
        difference=verify.difference,
        notes=["Verified via differentiation"],
    )


def compute_limit(
    expression: str, variable: str, point: str, direction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute a limit.

    Args:
        expression: Expression to evaluate
        variable: Variable approaching limit
        point: Point being approached (can be "Infinity", "-Infinity")
        direction: "Left", "Right", or None for both sides
    """
    if direction:
        query = f"Limit[{expression}, {variable} -> {point}, Direction -> {direction}]"
    else:
        query = f"Limit[{expression}, {variable} -> {point}]"

    result = wolfram_query(query, format="plaintext")
    return {
        "success": result.success,
        "expression": expression,
        "limit": result.result,
        "point": point,
        "direction": direction,
        "error": result.error,
    }


def compute_series(
    function: str, variable: str = "x", point: str = "0", order: int = 5
) -> Dict[str, Any]:
    """Compute Taylor/Maclaurin series expansion."""
    query = f"Series[{function}, {{{variable}, {point}, {order}}}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "function": function,
        "series": result.result,
        "expansion_point": point,
        "order": order,
        "error": result.error,
    }


# =============================================================================
# 3. LINEAR ALGEBRA
# =============================================================================


def matrix_operation(matrix: str, operation: str) -> Dict[str, Any]:
    """
    Perform matrix operations.

    Args:
        matrix: Matrix in Wolfram format, e.g., "{{1,2},{3,4}}"
        operation: One of:
            - "Determinant", "Inverse", "Transpose"
            - "Eigenvalues", "Eigenvectors", "Eigensystem"
            - "MatrixRank", "Trace", "Norm"
            - "SingularValueDecomposition", "QRDecomposition", "LUDecomposition"
            - "CharacteristicPolynomial", "MatrixExp", "MatrixPower"
            - "PositiveDefiniteMatrixQ", "SymmetricMatrixQ"
    """
    query = f"{operation}[{matrix}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "matrix": matrix,
        "operation": operation,
        "result": result.result,
        "error": result.error,
    }


def verify_matrix_equation(
    lhs_matrix: str, rhs_matrix: str, operation: Optional[str] = None
) -> AuditResult:
    """
    Verify matrix equation (e.g., A * A^(-1) = I).

    Args:
        lhs_matrix: Left side matrix expression
        rhs_matrix: Right side matrix expression
        operation: Optional operation to apply to lhs first
    """
    if operation:
        lhs_expr = f"{operation}[{lhs_matrix}]"
    else:
        lhs_expr = lhs_matrix

    query = f"FullSimplify[{lhs_expr} - {rhs_matrix}]"
    result = wolfram_query(query, format="plaintext")

    # Check if result is zero matrix
    is_zero = result.result and (
        "0" in result.result.replace(" ", "") or "{{0" in result.result
    )

    return AuditResult(
        passed=is_zero,
        category="matrix",
        input_expression=f"{lhs_expr} = {rhs_matrix}",
        expected="Zero matrix",
        computed=result.result,
        error=result.error,
    )


def solve_linear_system(equations: str, variables: str) -> Dict[str, Any]:
    """
    Solve system of linear equations.

    Args:
        equations: System in format "{2x + 3y == 5, x - y == 1}"
        variables: Variables in format "{x, y}"
    """
    query = f"Solve[{equations}, {variables}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "equations": equations,
        "variables": variables,
        "solutions": result.result,
        "error": result.error,
    }


# =============================================================================
# 4. DIFFERENTIAL EQUATIONS
# =============================================================================


def solve_ode(
    equation: str,
    function: str = "y[x]",
    variable: str = "x",
    initial_conditions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Solve ordinary differential equation.

    Args:
        equation: ODE in format "y''[x] + y[x] == 0" or "y'' + y = 0"
        function: Dependent function, e.g., "y[x]"
        variable: Independent variable
        initial_conditions: ICs in format "{y[0] == 1, y'[0] == 0}"
    """
    if initial_conditions:
        query = f"DSolve[{{{equation}, {initial_conditions}}}, {function}, {variable}]"
    else:
        query = f"DSolve[{equation}, {function}, {variable}]"

    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "equation": equation,
        "solution": result.result,
        "function": function,
        "initial_conditions": initial_conditions,
        "error": result.error,
    }


def verify_ode_solution(
    ode: str, claimed_solution: str, function: str = "y", variable: str = "x"
) -> AuditResult:
    """
    Verify that a function is a solution to an ODE by substitution.
    """
    # Replace function with claimed solution
    verification_query = f"FullSimplify[({ode}) /. {function} -> Function[{variable}, {claimed_solution}]]"
    result = wolfram_query(verification_query, format="plaintext")

    is_valid = result.result and result.result.strip() in ("True", "0", "0 == 0")

    return AuditResult(
        passed=is_valid,
        category="ode",
        input_expression=ode,
        expected="True (solution satisfies ODE)",
        computed=result.result,
        notes=[f"Claimed solution: {claimed_solution}"],
    )


# =============================================================================
# 5. PROBABILITY & STATISTICS
# =============================================================================


def distribution_property(distribution: str, property: str) -> Dict[str, Any]:
    """
    Get property of a probability distribution.

    Args:
        distribution: Distribution name and params, e.g., "NormalDistribution[0, 1]"
        property: One of:
            - "Mean", "Variance", "StandardDeviation", "Skewness", "Kurtosis"
            - "PDF", "CDF", "InverseCDF", "SurvivalFunction", "HazardFunction"
            - "CharacteristicFunction", "MomentGeneratingFunction"
            - "Quantile", "Median", "Mode"
            - "Entropy", "InformationMatrix"
    """
    query = f"{property}[{distribution}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "distribution": distribution,
        "property": property,
        "value": result.result,
        "error": result.error,
    }


def compute_probability(event: str, distribution: str) -> Dict[str, Any]:
    """
    Compute probability of an event.

    Args:
        event: Event condition, e.g., "x > 1" or "1 < x < 2"
        distribution: Distribution, e.g., "x \\[Distributed] NormalDistribution[0, 1]"
    """
    query = f"Probability[{event}, {distribution}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "event": event,
        "distribution": distribution,
        "probability": result.result,
        "error": result.error,
    }


def compute_expectation(function: str, distribution: str) -> Dict[str, Any]:
    """Compute expected value E[f(X)]."""
    query = f"Expectation[{function}, {distribution}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "function": function,
        "distribution": distribution,
        "expectation": result.result,
        "error": result.error,
    }


def statistical_test(test_type: str, data: str, **kwargs) -> Dict[str, Any]:
    """
    Perform statistical hypothesis test.

    Args:
        test_type: "TTest", "FTest", "ChiSquareTest", "KolmogorovSmirnovTest", etc.
        data: Data in list format
        **kwargs: Additional parameters
    """
    params = ", ".join([f'"{k}" -> {v}' for k, v in kwargs.items()])
    if params:
        query = f"{test_type}[{data}, {{{params}}}]"
    else:
        query = f"{test_type}[{data}]"

    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "test": test_type,
        "data": data,
        "result": result.result,
        "pods": result.pods,
        "error": result.error,
    }


# =============================================================================
# 6. OPTIMIZATION
# =============================================================================


def minimize(
    objective: str,
    variables: str,
    constraints: Optional[str] = None,
    domain: str = "Reals",
) -> Dict[str, Any]:
    """
    Find minimum of a function.

    Args:
        objective: Function to minimize
        variables: Variables, e.g., "x" or "{x, y}"
        constraints: Constraints, e.g., "x > 0 && y > 0"
        domain: "Reals", "Integers", etc.
    """
    if constraints:
        query = f"NMinimize[{{{objective}, {constraints}}}, {variables}]"
    else:
        query = f"NMinimize[{objective}, {variables}]"

    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "objective": objective,
        "minimum": result.result,
        "constraints": constraints,
        "domain": domain,
        "pods": result.pods,
        "error": result.error,
    }


def maximize(
    objective: str, variables: str, constraints: Optional[str] = None
) -> Dict[str, Any]:
    """Find maximum of a function."""
    if constraints:
        query = f"NMaximize[{{{objective}, {constraints}}}, {variables}]"
    else:
        query = f"NMaximize[{objective}, {variables}]"

    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "objective": objective,
        "maximum": result.result,
        "constraints": constraints,
        "pods": result.pods,
        "error": result.error,
    }


def linear_optimization(
    objective: str, constraints: str, variables: str
) -> Dict[str, Any]:
    """
    Solve linear programming problem.

    Args:
        objective: Linear objective, e.g., "2x + 3y"
        constraints: Linear constraints, e.g., "{x + y <= 10, x >= 0, y >= 0}"
        variables: Variables, e.g., "{x, y}"
    """
    query = f"LinearOptimization[{objective}, {constraints}, {variables}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "objective": objective,
        "constraints": constraints,
        "optimal": result.result,
        "error": result.error,
    }


def find_critical_points(function: str, variables: str) -> Dict[str, Any]:
    """Find critical points (where gradient = 0)."""
    query = f"Solve[D[{function}, {variables}] == 0, {variables}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "function": function,
        "critical_points": result.result,
        "error": result.error,
    }


# =============================================================================
# 7. MODEL VALIDATION UTILITIES
# =============================================================================


def check_dimensions(
    expression: str, expected_unit: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform dimensional analysis on an expression.

    Args:
        expression: Physical expression with units
        expected_unit: Expected resulting unit
    """
    query = f"UnitDimensions[{expression}]"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "expression": expression,
        "dimensions": result.result,
        "expected": expected_unit,
        "error": result.error,
    }


def check_boundary_conditions(
    function: str, variable: str, boundaries: List[Dict[str, str]]
) -> List[AuditResult]:
    """
    Verify boundary conditions for a function.

    Args:
        function: The function to check
        variable: Variable name
        boundaries: List of {"point": "0", "expected": "1", "type": "value|derivative"}
    """
    results = []

    for bc in boundaries:
        point = bc["point"]
        expected = bc["expected"]
        bc_type = bc.get("type", "value")

        if bc_type == "derivative":
            query = f"D[{function}, {variable}] /. {variable} -> {point}"
        else:
            query = f"{function} /. {variable} -> {point}"

        result = wolfram_query(query, format="plaintext")

        if result.success:
            verify = verify_equation(result.result, expected)
            results.append(
                AuditResult(
                    passed=verify.passed,
                    category="boundary_condition",
                    input_expression=f"f({point}) = {expected}"
                    if bc_type == "value"
                    else f"f'({point}) = {expected}",
                    expected=expected,
                    computed=result.result,
                    difference=verify.difference,
                )
            )
        else:
            results.append(
                AuditResult(
                    passed=False,
                    category="boundary_condition",
                    input_expression=f"{bc_type} at {point}",
                    error=result.error,
                )
            )

    return results


def sensitivity_analysis(function: str, parameter: str, point: str) -> Dict[str, Any]:
    """
    Compute sensitivity of function to parameter (partial derivative at point).
    """
    # Compute partial derivative
    deriv = compute_derivative(function, parameter, 1)

    if not deriv["success"]:
        return {"success": False, "error": deriv["error"]}

    # Evaluate at point
    query = f"({deriv['derivative']}) /. {parameter} -> {point}"
    result = wolfram_query(query, format="plaintext")

    return {
        "success": result.success,
        "function": function,
        "parameter": parameter,
        "sensitivity_formula": deriv["derivative"],
        "sensitivity_at_point": result.result,
        "point": point,
        "error": result.error,
    }


def audit_mathematical_model(model_equations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Comprehensive audit of a mathematical model.

    Args:
        model_equations: List of equations to verify, each with:
            - "type": "equation"|"derivative"|"integral"|"ode"|"constraint"
            - Other parameters specific to type

    Returns:
        Comprehensive audit report
    """
    results = {
        "total": len(model_equations),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": [],
    }

    for eq in model_equations:
        eq_type = eq.get("type", "equation")

        try:
            if eq_type == "equation":
                audit = verify_equation(eq["lhs"], eq["rhs"], eq.get("assumptions"))
            elif eq_type == "derivative":
                audit = verify_derivative(
                    eq["function"],
                    eq["derivative"],
                    eq.get("variable", "x"),
                    eq.get("order", 1),
                )
            elif eq_type == "integral":
                audit = verify_integral(
                    eq["integrand"], eq["result"], eq.get("variable", "x")
                )
            elif eq_type == "ode":
                audit = verify_ode_solution(
                    eq["ode"],
                    eq["solution"],
                    eq.get("function", "y"),
                    eq.get("variable", "x"),
                )
            else:
                audit = AuditResult(
                    passed=False,
                    category=eq_type,
                    input_expression=str(eq),
                    error=f"Unknown equation type: {eq_type}",
                )

            results["details"].append({"input": eq, "result": audit})

            if audit.error:
                results["errors"] += 1
            elif audit.passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

        except Exception as e:
            results["errors"] += 1
            results["details"].append({"input": eq, "error": str(e)})

    results["summary"] = (
        f"Passed: {results['passed']}/{results['total']}, Failed: {results['failed']}, Errors: {results['errors']}"
    )

    return results


# =============================================================================
# TOOLS REGISTRY
# =============================================================================

WOLFRAM_TOOLS = [
    # Core
    {
        "name": "wolfram_query",
        "description": "Raw query to Wolfram Alpha API",
        "function": wolfram_query,
    },
    # Symbolic
    {
        "name": "verify_equation",
        "description": "Verify equation LHS = RHS",
        "function": verify_equation,
    },
    {
        "name": "simplify_expression",
        "description": "Simplify mathematical expression",
        "function": simplify_expression,
    },
    {
        "name": "solve_equation",
        "description": "Solve equations symbolically",
        "function": solve_equation,
    },
    # Calculus
    {
        "name": "compute_derivative",
        "description": "Compute derivative",
        "function": compute_derivative,
    },
    {
        "name": "verify_derivative",
        "description": "Verify derivative computation",
        "function": verify_derivative,
    },
    {
        "name": "compute_integral",
        "description": "Compute integral",
        "function": compute_integral,
    },
    {
        "name": "verify_integral",
        "description": "Verify integral computation",
        "function": verify_integral,
    },
    {
        "name": "compute_limit",
        "description": "Compute limit",
        "function": compute_limit,
    },
    {
        "name": "compute_series",
        "description": "Compute Taylor series",
        "function": compute_series,
    },
    # Linear Algebra
    {
        "name": "matrix_operation",
        "description": "Matrix operations (det, inv, eigen, etc.)",
        "function": matrix_operation,
    },
    {
        "name": "verify_matrix_equation",
        "description": "Verify matrix equation",
        "function": verify_matrix_equation,
    },
    {
        "name": "solve_linear_system",
        "description": "Solve linear equation system",
        "function": solve_linear_system,
    },
    # ODEs
    {
        "name": "solve_ode",
        "description": "Solve ordinary differential equation",
        "function": solve_ode,
    },
    {
        "name": "verify_ode_solution",
        "description": "Verify ODE solution",
        "function": verify_ode_solution,
    },
    # Probability & Statistics
    {
        "name": "distribution_property",
        "description": "Get distribution property",
        "function": distribution_property,
    },
    {
        "name": "compute_probability",
        "description": "Compute probability",
        "function": compute_probability,
    },
    {
        "name": "compute_expectation",
        "description": "Compute expected value",
        "function": compute_expectation,
    },
    {
        "name": "statistical_test",
        "description": "Perform statistical test",
        "function": statistical_test,
    },
    # Optimization
    {"name": "minimize", "description": "Minimize function", "function": minimize},
    {"name": "maximize", "description": "Maximize function", "function": maximize},
    {
        "name": "linear_optimization",
        "description": "Linear programming",
        "function": linear_optimization,
    },
    {
        "name": "find_critical_points",
        "description": "Find critical points",
        "function": find_critical_points,
    },
    # Model Validation
    {
        "name": "check_dimensions",
        "description": "Dimensional analysis",
        "function": check_dimensions,
    },
    {
        "name": "check_boundary_conditions",
        "description": "Verify boundary conditions",
        "function": check_boundary_conditions,
    },
    {
        "name": "sensitivity_analysis",
        "description": "Parameter sensitivity",
        "function": sensitivity_analysis,
    },
    {
        "name": "audit_mathematical_model",
        "description": "Comprehensive model audit",
        "function": audit_mathematical_model,
    },
    # LLM API (Keyword Queries)
    {
        "name": "wolfram_llm_query",
        "description": "LLM-optimized keyword query",
        "function": wolfram_llm_query,
    },
    {
        "name": "llm_natural_query",
        "description": "Keyword question to Wolfram",
        "function": llm_natural_query,
    },
    {
        "name": "llm_quick_compute",
        "description": "Quick math via LLM API",
        "function": llm_quick_compute,
    },
    {
        "name": "llm_fact_lookup",
        "description": "Fact lookup via LLM API",
        "function": llm_fact_lookup,
    },
    # Short Answers API (Quick Single-Line Results)
    {
        "name": "wolfram_short_answer",
        "description": "Single-line text answer",
        "function": wolfram_short_answer,
    },
    {
        "name": "quick_math_check",
        "description": "Fastest math verification",
        "function": quick_math_check,
    },
    # Utilities
    {
        "name": "preprocess_query_for_wolfram",
        "description": "Convert NL to keyword query",
        "function": preprocess_query_for_wolfram,
    },
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM ALPHA TOOLS - COMPREHENSIVE TEST")
    print("=" * 60)

    tests = [
        ("Basic Query", lambda: wolfram_query("1 + 1")),
        (
            "Equation Verify (True)",
            lambda: verify_equation("(a+b)^2", "a^2 + 2*a*b + b^2"),
        ),
        ("Equation Verify (False)", lambda: verify_equation("(a+b)^2", "a^2 + b^2")),
        ("Simplify", lambda: simplify_expression("sin(x)^2 + cos(x)^2")),
        ("Derivative", lambda: compute_derivative("x^3 + 2*x^2 - 5*x + 3", "x")),
        ("Integral", lambda: compute_integral("x^2", "x")),
        ("Limit", lambda: compute_limit("sin(x)/x", "x", "0")),
        ("Matrix Det", lambda: matrix_operation("{{1,2},{3,4}}", "Determinant")),
        ("Solve ODE", lambda: solve_ode("y''[x] + y[x] == 0", "y[x]", "x")),
        (
            "Distribution",
            lambda: distribution_property("NormalDistribution[0, 1]", "Variance"),
        ),
        ("Minimize", lambda: minimize("x^2 + y^2", "{x, y}", "x + y == 1")),
    ]

    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            result = test_fn()
            if hasattr(result, "result"):
                print(f"  Result: {result.result}")
            elif isinstance(result, dict):
                key = next(
                    (
                        k
                        for k in [
                            "result",
                            "simplified",
                            "derivative",
                            "integral",
                            "limit",
                            "solutions",
                            "value",
                            "minimum",
                        ]
                        if k in result
                    ),
                    None,
                )
                if key:
                    print(f"  {key.title()}: {result[key]}")
            elif hasattr(result, "passed"):
                print(f"  Passed: {result.passed}, Computed: {result.computed}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    # LLM API Tests
    print("\n" + "=" * 60)
    print("LLM API TESTS")
    print("=" * 60)

    llm_tests = [
        ("LLM Basic Query", lambda: wolfram_llm_query("What is 2+2?")),
        ("LLM Natural Question", lambda: llm_natural_query("population of Japan")),
        ("LLM Quick Compute", lambda: llm_quick_compute("derivative of sin(x)")),
        ("LLM Fact Lookup", lambda: llm_fact_lookup("speed of light")),
    ]

    for name, test_fn in llm_tests:
        print(f"\n{name}:")
        try:
            result = test_fn()
            if hasattr(result, "response"):
                # Truncate long responses
                resp = (
                    result.response[:200] + "..."
                    if result.response and len(result.response) > 200
                    else result.response
                )
                print(f"  Success: {result.success}")
                print(f"  Response: {resp}")
            elif isinstance(result, dict):
                print(f"  Success: {result.get('success')}")
                key = next(
                    (
                        k
                        for k in ["answer", "result", "facts"]
                        if k in result and result[k]
                    ),
                    None,
                )
                if key:
                    val = (
                        result[key][:200] + "..."
                        if result[key] and len(str(result[key])) > 200
                        else result[key]
                    )
                    print(f"  {key.title()}: {val}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)

    # Short Answers API Tests
    print("\n" + "=" * 60)
    print("SHORT ANSWERS API TESTS")
    print("=" * 60)

    short_tests = [
        ("Quick Math: 2+2", lambda: quick_math_check("2+2")),
        ("Quick Math: derivative", lambda: quick_math_check("derivative of x^3")),
        ("Quick Math: integrate", lambda: quick_math_check("integrate sin(x)")),
        (
            "Short Answer: distance",
            lambda: wolfram_short_answer("distance from Earth to Moon"),
        ),
    ]

    for name, test_fn in short_tests:
        print(f"\n{name}:")
        try:
            result = test_fn()
            if isinstance(result, str):
                print(f"  Result: {result}")
            elif isinstance(result, dict):
                print(f"  Success: {result.get('success')}")
                print(f"  Answer: {result.get('answer')}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("ALL API TESTS COMPLETE")
    print("=" * 60)
