"""
Feedback Handler for LangGraph Research System — Self-Improvement Mechanism.

Two triggers:
  1. **User-triggered**: Detects negative feedback / complaints in user messages
     and loads the previous execution so the pipeline can re-run with lessons.
  2. **Self-triggered**: Called by output_builder when quality self-assessment
     detects issues (high error ratio, empty results, etc.).

Manages a persistent *lessons learned* database stored at
`.memory/lessons/lessons_learned.json`.  Each lesson records:
  - What went wrong (issue)
  - What the resolution/avoidance strategy is
  - Which agent was involved
  - Searchable keywords for future matching

The host AI (Antigravity / OpenCode / Claude Code / Gemini CLI) receives
injected lessons in the optimized prompt so it can avoid repeating mistakes.
No LLM API key is required — all detection is rule-based.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LESSONS_DIR = PROJECT_ROOT / ".memory" / "lessons"
LESSONS_FILE = LESSONS_DIR / "lessons_learned.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MAX_LESSONS = 200  # Prune oldest / least-used beyond this limit


# ============================================================================
# FEEDBACK KEYWORD PATTERNS  (Vietnamese + English)
# ============================================================================

# Negative-feedback indicators — if ≥1 pattern matches, treat as feedback.
_NEGATIVE_PATTERNS_EN = [
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bbad\b",
    r"\bfix\b",
    r"\berror\b",
    r"\bfail(?:ed|ure|s)?\b",
    r"\bnot\s+(?:good|correct|right|working|accurate)\b",
    r"\bredo\b",
    r"\bretry\b",
    r"\btry\s+again\b",
    r"\bimprove\b",
    r"\bworse\b",
    r"\bmissing\b",
    r"\bincomplete\b",
    r"\binaccurate\b",
    r"\bpoor(?:ly)?\b",
    r"\bshould(?:n't| not)\b",
    r"\bdon'?t\s+like\b",
    r"\bchange\s+(?:this|that|it)\b",
    r"\bnot\s+what\s+i\b",
    r"\bplease\s+(?:fix|redo|change|update|correct)\b",
]

_NEGATIVE_PATTERNS_VI = [
    r"sai",
    r"lỗi",
    r"loi",
    r"không đúng",
    r"khong dung",
    r"chưa tốt",
    r"chua tot",
    r"chưa đúng",
    r"chua dung",
    r"không tốt",
    r"khong tot",
    r"làm lại",
    r"lam lai",
    r"sửa",
    r"sua",
    r"thiếu",
    r"thieu",
    r"chưa đủ",
    r"chua du",
    r"tệ",
    r"te ",
    r"kém",
    r"kem",
    r"cải thiện",
    r"cai thien",
]

# Compile once
_COMPILED_EN = [re.compile(p, re.IGNORECASE) for p in _NEGATIVE_PATTERNS_EN]
_COMPILED_VI = [re.compile(p, re.IGNORECASE) for p in _NEGATIVE_PATTERNS_VI]


# ============================================================================
# FEEDBACK DETECTION
# ============================================================================


def detect_feedback(message: str) -> Tuple[bool, List[str]]:
    """Detect if a user message contains negative feedback.

    Args:
        message: Raw user message (may be Vietnamese or English).

    Returns:
        Tuple of (is_feedback: bool, matched_patterns: List[str]).
    """
    if not message or not message.strip():
        return False, []

    matched: List[str] = []

    for pattern in _COMPILED_EN:
        if pattern.search(message):
            matched.append(pattern.pattern)

    for pattern in _COMPILED_VI:
        if pattern.search(message):
            matched.append(pattern.pattern)

    return len(matched) >= 1, matched


# ============================================================================
# PREVIOUS EXECUTION LOADING
# ============================================================================


def load_previous_execution(limit: int = 1) -> Optional[Dict[str, Any]]:
    """Load the most recent execution record from outputs/ directory.

    Scans outputs/{date}/{time}_{agent}/structured_output.json and returns
    the most recent one.

    Args:
        limit: Number of recent executions to load (default 1).

    Returns:
        Dict with previous execution data, or None if not found.
    """
    if not OUTPUTS_DIR.exists():
        return None

    # Collect all structured_output.json files
    candidates: List[Tuple[float, Path]] = []
    try:
        for date_dir in OUTPUTS_DIR.iterdir():
            if not date_dir.is_dir():
                continue
            for run_dir in date_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                output_file = run_dir / "structured_output.json"
                if output_file.exists():
                    mtime = output_file.stat().st_mtime
                    candidates.append((mtime, output_file))
    except OSError as exc:
        logger.warning("Error scanning outputs directory: %s", exc)
        return None

    if not candidates:
        return None

    # Sort by modification time descending (most recent first)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Load the most recent execution
    _, latest_file = candidates[0]
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_output_dir"] = str(latest_file.parent)
        data["_output_file"] = str(latest_file)
        return data
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Could not load previous execution: %s", exc)
        return None


# ============================================================================
# LESSONS LEARNED DATABASE
# ============================================================================


def _ensure_lessons_dir() -> None:
    """Ensure the lessons directory exists."""
    LESSONS_DIR.mkdir(parents=True, exist_ok=True)


def _load_lessons_db() -> Dict[str, Any]:
    """Load the lessons learned database from disk.

    Returns:
        Dict with structure: {"lessons": [...], "created": ..., "updated": ...}
    """
    _ensure_lessons_dir()
    if LESSONS_FILE.exists():
        try:
            with open(LESSONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "lessons": [],
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
    }


def _save_lessons_db(db: Dict[str, Any]) -> bool:
    """Save the lessons learned database to disk.

    Prunes to MAX_LESSONS by removing oldest entries with lowest applied_count.

    Returns:
        True if saved successfully.
    """
    _ensure_lessons_dir()
    lessons = db.get("lessons", [])

    # Prune if over limit: remove lowest applied_count first, then oldest
    if len(lessons) > MAX_LESSONS:
        lessons.sort(key=lambda x: (x.get("applied_count", 0), x.get("timestamp", "")))
        db["lessons"] = lessons[-MAX_LESSONS:]

    db["updated"] = datetime.now().isoformat()

    try:
        with open(LESSONS_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        return True
    except IOError as exc:
        logger.warning("Could not save lessons DB: %s", exc)
        return False


def save_lesson(
    issue: str,
    resolution: str,
    agent: str = "",
    trigger: str = "auto",
    keywords: Optional[List[str]] = None,
) -> bool:
    """Save a new lesson learned.

    Args:
        issue: Description of what went wrong.
        resolution: How to avoid or fix this in the future.
        agent: Agent name involved (if applicable).
        trigger: "user" (from user feedback) or "auto" (self-detected).
        keywords: Searchable keywords for future matching.

    Returns:
        True if saved successfully.
    """
    db = _load_lessons_db()

    lesson = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "trigger": trigger,
        "agent": agent,
        "issue": issue,
        "resolution": resolution,
        "keywords": keywords or [],
        "applied_count": 0,
    }

    db["lessons"].append(lesson)
    logger.info(
        "Saved lesson [%s]: %s -> %s", lesson["id"], issue[:60], resolution[:60]
    )
    return _save_lessons_db(db)


def search_lessons(
    query: str,
    agent: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search lessons learned by keyword relevance.

    Args:
        query: Search query (keywords from current request).
        agent: Optional agent name filter.
        limit: Maximum results.

    Returns:
        List of matching lessons sorted by relevance.
    """
    db = _load_lessons_db()
    lessons = db.get("lessons", [])

    if agent:
        lessons = [
            le for le in lessons if le.get("agent", "") == agent or not le.get("agent")
        ]

    if not lessons or not query.strip():
        return []

    query_words = set(query.lower().split())
    query_words = {w for w in query_words if len(w) > 2}

    if not query_words:
        return lessons[-limit:]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for le in lessons:
        # Build searchable text from issue + resolution + keywords
        searchable = (
            f"{le.get('issue', '')} {le.get('resolution', '')} "
            f"{' '.join(le.get('keywords', []))}"
        ).lower()
        searchable_words = set(searchable.split())

        matches = query_words & searchable_words
        if matches:
            score = len(matches) / len(query_words)
            # Bonus for keyword-list matches
            kw_set = set(k.lower() for k in le.get("keywords", []))
            kw_matches = query_words & kw_set
            if kw_matches:
                score = min(1.0, score + 0.2 * len(kw_matches))
            scored.append((score, le))

    scored.sort(key=lambda x: -x[0])
    return [le for _, le in scored[:limit]]


def increment_lesson_applied(lesson_id: str) -> None:
    """Increment the applied_count for a lesson (tracks usefulness)."""
    db = _load_lessons_db()
    for le in db.get("lessons", []):
        if le.get("id") == lesson_id:
            le["applied_count"] = le.get("applied_count", 0) + 1
            break
    _save_lessons_db(db)


def get_all_lessons(limit: int = 50) -> List[Dict[str, Any]]:
    """Get all lessons, most recent first."""
    db = _load_lessons_db()
    lessons = db.get("lessons", [])
    return lessons[-limit:][::-1]


def get_lessons_summary() -> str:
    """Get a human-readable summary of the lessons database."""
    db = _load_lessons_db()
    lessons = db.get("lessons", [])
    if not lessons:
        return "[LESSONS] No lessons learned yet."

    by_trigger = {"user": 0, "auto": 0}
    by_agent: Dict[str, int] = {}
    for le in lessons:
        trigger = le.get("trigger", "auto")
        by_trigger[trigger] = by_trigger.get(trigger, 0) + 1
        agent = le.get("agent", "unknown")
        if agent:
            by_agent[agent] = by_agent.get(agent, 0) + 1

    lines = [
        "[LESSONS] Self-Improvement Database Summary",
        f"  Total lessons: {len(lessons)}",
        f"  User-triggered: {by_trigger.get('user', 0)}",
        f"  Auto-detected: {by_trigger.get('auto', 0)}",
        f"  Agents involved: {len(by_agent)}",
        f"  Last updated: {db.get('updated', 'N/A')[:19]}",
    ]

    # Top 3 agents with most lessons
    top_agents = sorted(by_agent.items(), key=lambda x: -x[1])[:3]
    if top_agents:
        lines.append("  Top agents:")
        for agent_name, count in top_agents:
            lines.append(f"    - {agent_name}: {count} lessons")

    return "\n".join(lines)


# ============================================================================
# QUALITY SELF-ASSESSMENT (called by output_builder)
# ============================================================================


def _extract_error_msg(entry: Any) -> str:
    """Extract error message from a tool_error entry (dict or str)."""
    if isinstance(entry, dict):
        return str(entry.get("error", "unknown"))
    return str(entry) if entry else "unknown"


def assess_execution_quality(
    tool_results: List[Dict[str, Any]],
    tool_errors: List[Dict[str, Any]],
    agent_name: str = "",
) -> Dict[str, Any]:
    """Assess the quality of a pipeline execution.

    Checks for common failure patterns and returns quality flags.

    Args:
        tool_results: List of tool results from pipeline.
        tool_errors: List of tool errors from pipeline.
        agent_name: The agent that ran.

    Returns:
        Dict with quality assessment:
        {
            "quality_score": float (0.0-1.0),
            "issues": List[str],
            "auto_lessons": List[Dict]  -- lessons to auto-save
        }
    """
    issues: List[str] = []
    auto_lessons: List[Dict[str, Any]] = []

    n_success = sum(1 for r in tool_results if r.get("status") == "success")
    n_skipped = sum(1 for r in tool_results if r.get("status") == "skipped")
    n_failed = len(tool_errors)
    n_total = n_success + n_failed

    # Issue 1: All tools failed
    if n_total > 0 and n_success == 0 and n_failed > 0:
        error_msgs = [_extract_error_msg(e) for e in tool_errors[:3]]
        issue_desc = f"All {n_failed} tools failed for agent {agent_name}"
        issues.append(issue_desc)
        auto_lessons.append(
            {
                "issue": issue_desc,
                "resolution": (
                    f"When using agent {agent_name}, check tool prerequisites. "
                    f"Common errors: {'; '.join(error_msgs)}"
                ),
                "agent": agent_name,
                "keywords": [agent_name.lower(), "tool_failure", "all_failed"],
            }
        )

    # Issue 2: High error ratio (>50% tools failed)
    if n_total > 1 and n_failed > n_success:
        issue_desc = (
            f"High tool failure ratio for {agent_name}: {n_failed}/{n_total} failed"
        )
        issues.append(issue_desc)
        # Extract unique error types
        error_types = set()
        for e in tool_errors:
            err = _extract_error_msg(e)
            if "rate" in err.lower() or "limit" in err.lower():
                error_types.add("rate_limit")
            elif "timeout" in err.lower():
                error_types.add("timeout")
            elif "auth" in err.lower() or "key" in err.lower():
                error_types.add("auth_missing")
            else:
                error_types.add("other")

        if error_types:
            auto_lessons.append(
                {
                    "issue": issue_desc,
                    "resolution": (
                        f"Agent {agent_name} has reliability issues. "
                        f"Error types: {', '.join(error_types)}. "
                        f"Consider using alternative tools or adding delays."
                    ),
                    "agent": agent_name,
                    "keywords": [
                        agent_name.lower(),
                        "high_error_rate",
                        *list(error_types),
                    ],
                }
            )

    # Issue 3: Tools returned empty/null results
    empty_tools: List[str] = []
    for r in tool_results:
        if r.get("status") == "success":
            result = r.get("result")
            if result is None or result == "" or result == [] or result == {}:
                empty_tools.append(r.get("tool_name", "?"))

    if empty_tools and n_success > 0:
        ratio = len(empty_tools) / n_success
        if ratio >= 0.5:
            issue_desc = (
                f"Tools returned empty results for {agent_name}: "
                f"{', '.join(empty_tools)}"
            )
            issues.append(issue_desc)
            auto_lessons.append(
                {
                    "issue": issue_desc,
                    "resolution": (
                        f"When {agent_name} gets empty results from "
                        f"{', '.join(empty_tools)}, try: "
                        f"(1) simplify the search query, "
                        f"(2) use broader keywords, "
                        f"(3) check if the tool requires specific parameters."
                    ),
                    "agent": agent_name,
                    "keywords": [
                        agent_name.lower(),
                        "empty_results",
                        *[t.lower() for t in empty_tools],
                    ],
                }
            )

    # Issue 4: All tools were skipped (nothing executed)
    if n_total == 0 and n_skipped > 0:
        issue_desc = (
            f"No tools executed for {agent_name} — all {n_skipped} were skipped "
            f"(require user-provided parameters)"
        )
        issues.append(issue_desc)

    # Calculate quality score
    if n_total == 0:
        quality_score = 0.5 if n_skipped > 0 else 1.0  # No tools = neutral
    else:
        success_ratio = n_success / n_total
        empty_penalty = len(empty_tools) / max(n_success, 1) * 0.3
        quality_score = max(0.0, min(1.0, success_ratio - empty_penalty))

    return {
        "quality_score": round(quality_score, 2),
        "issues": issues,
        "auto_lessons": auto_lessons,
    }


def save_auto_lessons(auto_lessons: List[Dict[str, Any]]) -> int:
    """Save auto-detected lessons from quality assessment.

    Args:
        auto_lessons: List of lesson dicts from assess_execution_quality.

    Returns:
        Number of lessons saved.
    """
    saved = 0
    for lesson_data in auto_lessons:
        success = save_lesson(
            issue=lesson_data["issue"],
            resolution=lesson_data["resolution"],
            agent=lesson_data.get("agent", ""),
            trigger="auto",
            keywords=lesson_data.get("keywords"),
        )
        if success:
            saved += 1
    return saved


# ============================================================================
# FEEDBACK CONTEXT BUILDER (for pipeline injection)
# ============================================================================


def build_feedback_context(
    user_message: str,
    previous_execution: Optional[Dict[str, Any]],
    matched_patterns: List[str],
) -> Dict[str, Any]:
    """Build a feedback context dict for pipeline injection.

    This is added to the state so downstream nodes (prompt_optimizer,
    agent_loader) know the user is providing feedback on a previous run.

    Args:
        user_message: The feedback message.
        previous_execution: Previous execution data (from load_previous_execution).
        matched_patterns: Which feedback patterns matched.

    Returns:
        Feedback context dict.
    """
    context: Dict[str, Any] = {
        "feedback_message": user_message,
        "matched_patterns": matched_patterns,
        "timestamp": datetime.now().isoformat(),
    }

    if previous_execution:
        prev_agent = previous_execution.get("agent_context", {})
        prev_meta = previous_execution.get("execution_metadata", {})
        prev_errors = previous_execution.get("tool_errors", [])

        context["previous_request"] = previous_execution.get("user_request", "")
        context["previous_agent"] = prev_agent.get("name", "")
        context["previous_domain"] = prev_agent.get("domain", "")
        context["previous_tools_failed"] = prev_meta.get("tools_failed", 0)
        context["previous_tools_success"] = prev_meta.get("tools_success", 0)
        context["previous_error_summary"] = [
            _extract_error_msg(e) for e in prev_errors[:5]
        ]
        context["previous_output_dir"] = previous_execution.get("_output_dir", "")

    return context


def format_lessons_for_prompt(lessons: List[Dict[str, Any]]) -> str:
    """Format lessons into a text block suitable for prompt injection.

    Args:
        lessons: List of lesson dicts from search_lessons.

    Returns:
        Formatted string to append to promized prompt.
    """
    if not lessons:
        return ""

    lines = [
        "",
        "## SYSTEM LESSONS LEARNED (from previous executions)",
        "The following issues were previously encountered. AVOID repeating them:",
        "",
    ]

    for i, le in enumerate(lessons, 1):
        lines.append(f"{i}. **Issue**: {le.get('issue', 'N/A')}")
        lines.append(f"   **Avoidance**: {le.get('resolution', 'N/A')}")
        if le.get("agent"):
            lines.append(f"   **Agent**: {le['agent']}")
        lines.append("")

    return "\n".join(lines)


def format_feedback_for_prompt(feedback_context: Dict[str, Any]) -> str:
    """Format feedback context into a text block for prompt injection.

    Args:
        feedback_context: Dict from build_feedback_context.

    Returns:
        Formatted string to prepend to optimized prompt.
    """
    if not feedback_context:
        return ""

    lines = [
        "",
        "## USER FEEDBACK ON PREVIOUS EXECUTION",
        f'The user is providing feedback: "{feedback_context.get("feedback_message", "")}"',
    ]

    prev_request = feedback_context.get("previous_request", "")
    if prev_request:
        lines.append(f'Previous request was: "{prev_request}"')
        lines.append(
            f"Previous agent: {feedback_context.get('previous_agent', 'N/A')} "
            f"(domain: {feedback_context.get('previous_domain', 'N/A')})"
        )

    prev_failed = feedback_context.get("previous_tools_failed", 0)
    prev_success = feedback_context.get("previous_tools_success", 0)
    if prev_failed > 0:
        lines.append(
            f"Previous execution: {prev_success} tools succeeded, {prev_failed} failed"
        )
        errors = feedback_context.get("previous_error_summary", [])
        if errors:
            lines.append(f"Previous errors: {'; '.join(errors[:3])}")

    lines.extend(
        [
            "",
            "INSTRUCTION: Address the user's feedback. Improve upon the previous execution.",
            "If tools failed previously, try alternative approaches or different parameters.",
            "",
        ]
    )

    return "\n".join(lines)
