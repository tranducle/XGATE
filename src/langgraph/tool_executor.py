"""
Tool Execution Engine for LangGraph Research System.

Executes tools with timeout, retry, parallel support, and structured error envelopes.
Never crashes — always returns ToolResult with status.
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from src.utils.config import get_execution_config

logger = logging.getLogger(__name__)

_thread_pool = ThreadPoolExecutor(max_workers=8)

_RESULT_CACHE: Dict[str, Any] = {}

NON_CACHEABLE_TOOLS = frozenset(
    {
        "save_papers_to_json",
        "write_markdown_section",
        "format_papers_as_markdown",
        "convert_file_to_markdown",
        "convert_multiple_files",
    }
)


def _make_cache_key(tool_name: str, kwargs: Dict[str, Any]) -> Optional[str]:
    if tool_name in NON_CACHEABLE_TOOLS:
        return None
    try:
        kwarg_str = json.dumps(kwargs, sort_keys=True, default=str)
        return f"{tool_name}_{hashlib.md5(kwarg_str.encode()).hexdigest()}"
    except Exception:
        return None


def execute_tool_sync(
    tool_name: str,
    func: Callable,
    kwargs: Dict[str, Any],
    timeout: int = 30,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """
    Execute a single tool synchronously with retry, timeout, and caching.

    Returns a ToolResult dict — never raises.
    """
    start_ms = time.time() * 1000

    cache_key = _make_cache_key(tool_name, kwargs)
    if cache_key and cache_key in _RESULT_CACHE:
        logger.info("Cache HIT: %s", tool_name)
        return {
            "tool_name": tool_name,
            "args": kwargs,
            "result": _RESULT_CACHE[cache_key],
            "status": "success",
            "duration_ms": round(time.time() * 1000 - start_ms, 2),
            "error": None,
        }

    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                logger.warning(
                    "Async function '%s' in sync executor — wrapping with asyncio.run()",
                    tool_name,
                )
                result = asyncio.run(func(**kwargs))
            else:
                result = func(**kwargs)

            if isinstance(result, dict) and result.get("error"):
                error_msg = str(result["error"])
                if _is_retryable(error_msg) and attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "Rate limit for %s, retry %d/%d in %.1fs",
                        tool_name,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

            if cache_key:
                _RESULT_CACHE[cache_key] = result

            return {
                "tool_name": tool_name,
                "args": kwargs,
                "result": result,
                "status": "success",
                "duration_ms": round(time.time() * 1000 - start_ms, 2),
                "error": None,
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if _is_retryable(str(exc)) and attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    "Retryable error for %s: %s — retry %d/%d in %.1fs",
                    tool_name,
                    exc,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
            else:
                break

    return {
        "tool_name": tool_name,
        "args": kwargs,
        "result": None,
        "status": "error",
        "duration_ms": round(time.time() * 1000 - start_ms, 2),
        "error": last_error or "Unknown error",
    }


async def execute_tool_async(
    tool_name: str,
    func: Callable,
    kwargs: Dict[str, Any],
    timeout: int = 30,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """
    Execute a single tool asynchronously. Wraps sync functions in a thread pool.

    Returns a ToolResult dict — never raises.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _thread_pool,
                lambda: execute_tool_sync(
                    tool_name, func, kwargs, timeout, max_retries, retry_delay
                ),
            ),
            timeout=timeout + 5,  # outer timeout slightly larger than inner
        )
        return result
    except asyncio.TimeoutError:
        return {
            "tool_name": tool_name,
            "args": kwargs,
            "result": None,
            "status": "timeout",
            "duration_ms": timeout * 1000,
            "error": f"Tool execution timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "tool_name": tool_name,
            "args": kwargs,
            "result": None,
            "status": "error",
            "duration_ms": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def execute_tools_parallel(
    tools: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute multiple tools in parallel (thread-based, no event loop required).

    Args:
        tools: List of dicts with keys: tool_name, func, kwargs
        config: Execution config (uses defaults if None)

    Returns:
        List of ToolResult dicts (same order as input)
    """
    if not tools:
        return []

    exec_cfg = config or get_execution_config()
    timeout = exec_cfg.get("tool_timeout", 30)
    retries = exec_cfg.get("tool_retries", 2)
    parallel = exec_cfg.get("parallel_tools", True)

    if not parallel or len(tools) == 1:
        return [
            execute_tool_sync(
                t["tool_name"], t["func"], t.get("kwargs", {}), timeout, retries
            )
            for t in tools
        ]

    futures = []
    for t in tools:
        future = _thread_pool.submit(
            execute_tool_sync,
            t["tool_name"],
            t["func"],
            t.get("kwargs", {}),
            timeout,
            retries,
        )
        futures.append(future)

    results = []
    for future in futures:
        try:
            results.append(future.result(timeout=timeout + 10))
        except Exception as exc:
            results.append(
                {
                    "tool_name": "unknown",
                    "args": {},
                    "result": None,
                    "status": "error",
                    "duration_ms": 0,
                    "error": f"Future failed: {type(exc).__name__}: {exc}",
                }
            )

    return results


def _is_retryable(error_msg: str) -> bool:
    lower = error_msg.lower()
    return any(
        term in lower
        for term in (
            "429",
            "rate limit",
            "timeout",
            "connection reset",
            "connection refused",
            "temporary",
            "too many requests",
        )
    )


def clear_cache():
    _RESULT_CACHE.clear()
    logger.info("Tool result cache cleared")
