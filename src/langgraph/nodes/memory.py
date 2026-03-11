"""
Memory Node for LangGraph Research System
Handles persistent memory storage and retrieval.

Integrates three context modules when available:
- ProjectState: operation history, metadata, SDP directories
- SessionManager: conversation history across sessions
- ResearchDiary: intellectual journey (ideas, insights, decisions)

All three are OPTIONAL — the node degrades gracefully to basic
JSON memory if context modules are not initialized.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MEMORY_DIR = PROJECT_ROOT / ".memory"
MEMORY_FILE = MEMORY_DIR / "memory.json"

# ---------------------------------------------------------------------------
# Context module singletons (lazy-loaded, fail-safe)
# ---------------------------------------------------------------------------
_project_state = None  # type: Optional[Any]
_session_manager = None  # type: Optional[Any]
_research_diary = None  # type: Optional[Any]


def init_context_modules(project_root: str = ".") -> Dict[str, bool]:
    """
    Initialise the three context modules.  Call once at pipeline start.

    Returns dict indicating which modules loaded successfully.
    """
    global _project_state, _session_manager, _research_diary
    status: Dict[str, bool] = {
        "project_state": False,
        "session_manager": False,
        "research_diary": False,
    }

    try:
        from ...context.project_state import ProjectState

        _project_state = ProjectState(project_root=project_root, use_sdp=False)
        status["project_state"] = True
        logger.info("ProjectState context loaded")
    except Exception as exc:
        logger.debug("ProjectState unavailable: %s", exc)

    try:
        from ...context.session import SessionManager

        _session_manager = SessionManager(
            session_dir=str(Path(project_root) / ".sessions")
        )
        status["session_manager"] = True
        logger.info(
            "SessionManager context loaded (session_id=%s)", _session_manager.session_id
        )
    except Exception as exc:
        logger.debug("SessionManager unavailable: %s", exc)

    try:
        from ...context.research_diary import ResearchDiary

        _research_diary = ResearchDiary(project_root=project_root)
        status["research_diary"] = True
        logger.info(
            "ResearchDiary context loaded (%d entries)", len(_research_diary.entries)
        )
    except Exception as exc:
        logger.debug("ResearchDiary unavailable: %s", exc)

    return status


def get_context_snapshot(limit: int = 10) -> Dict[str, Any]:
    """
    Collect a lightweight snapshot from every active context module.

    Precedence when merging into state['memory_context']:
      1. ResearchDiary (most recent insights/decisions)
      2. SessionManager (conversation continuity)
      3. ProjectState (operation history)
      4. Basic JSON memory (lowest priority, always present)

    Args:
        limit: max entries per module

    Returns:
        Dict with keys per active module.
    """
    snapshot: Dict[str, Any] = {}

    # --- ProjectState ---
    if _project_state is not None:
        try:
            ops = _project_state.get_operation_history(limit=limit)
            meta = _project_state.get_metadata()
            snapshot["project_state"] = {
                "recent_operations": ops,
                "metadata": meta,
                "statistics": _project_state.get_statistics(),
            }
        except Exception as exc:
            logger.warning("Failed to read ProjectState: %s", exc)

    # --- SessionManager ---
    if _session_manager is not None:
        try:
            history = _session_manager.get_conversation_history(limit=limit)
            snapshot["session"] = {
                "session_id": _session_manager.session_id,
                "messages": history,
                "summary": _session_manager.get_summary(),
            }
        except Exception as exc:
            logger.warning("Failed to read SessionManager: %s", exc)

    # --- ResearchDiary ---
    if _research_diary is not None:
        try:
            entries = _research_diary.get_recent_entries(limit=limit)
            snapshot["research_diary"] = {
                "recent_entries": entries,
                "total_entries": len(_research_diary.entries),
            }
        except Exception as exc:
            logger.warning("Failed to read ResearchDiary: %s", exc)

    # --- Lessons Learned (self-improvement) ---
    try:
        from .feedback_handler import get_lessons_summary

        summary = get_lessons_summary()
        if summary and "No lessons" not in summary:
            snapshot["lessons_learned"] = summary
    except Exception as exc:
        logger.debug("Lessons summary unavailable: %s", exc)

    return snapshot


def ensure_memory_dir():
    """Ensure memory directory exists."""
    MEMORY_DIR.mkdir(exist_ok=True)


def load_memory() -> Dict[str, Any]:
    """
    Load memory from local JSON file.

    Returns:
        Memory dictionary with memories list and metadata
    """
    ensure_memory_dir()

    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return {
        "memories": [],
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
    }


def save_memory(content: str, category: str = "general") -> bool:
    """
    Save a memory entry.

    Args:
        content: Memory content to save
        category: Category (routing, project_state, execution, etc.)

    Returns:
        True if saved successfully
    """
    ensure_memory_dir()

    data = load_memory()
    data["memories"].append(
        {
            "content": content,
            "category": category,
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Prune memory to prevent unbounded growth (keep last 500 entries)
    MAX_MEMORIES = 500
    if len(data["memories"]) > MAX_MEMORIES:
        data["memories"] = data["memories"][-MAX_MEMORIES:]

    data["updated"] = datetime.now().isoformat()

    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False


def get_memories_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all memories of a specific category."""
    data = load_memory()
    return [m for m in data.get("memories", []) if m.get("category") == category]


def log_execution(agent_name: str, action: str, result: str = "") -> bool:
    """
    Log an agent execution to memory.

    Args:
        agent_name: Name of the agent that executed
        action: What action was performed
        result: Result summary
    """
    content = f"[{agent_name}] {action}"
    if result:
        content += f" -> {result}"
    return save_memory(content, category="execution")


def get_recent_executions(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent execution logs."""
    executions = get_memories_by_category("execution")
    return executions[-limit:] if len(executions) > limit else executions


def get_project_state() -> List[Dict[str, Any]]:
    """Get project state memories."""
    return get_memories_by_category("project_state")


def save_project_state(project_name: str, status: str, topic: str = "") -> bool:
    """Save project state to memory."""
    content = f"Project: {project_name}, Topic: {topic}, Status: {status}"
    return save_memory(content, category="project_state")


def clear_memory() -> bool:
    """Clear all memory (use with caution)."""
    ensure_memory_dir()
    try:
        data = {
            "memories": [],
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
        }
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False


def search_memory(
    query: str, category: Optional[str] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search memory entries by keyword relevance.

    Uses word-overlap scoring to find the most relevant memories.

    Args:
        query: Search query string
        category: Optional category filter
        limit: Maximum number of results to return

    Returns:
        List of matching memory entries, sorted by relevance (highest first).
        Each entry includes an added 'relevance_score' field (0.0 - 1.0).
    """
    data = load_memory()
    memories = data.get("memories", [])

    # Filter by category if specified
    if category:
        memories = [m for m in memories if m.get("category") == category]

    if not memories or not query.strip():
        return []

    # Tokenize query into words
    query_words = set(query.lower().split())
    # Remove short/common words
    query_words = {w for w in query_words if len(w) > 2}

    if not query_words:
        return memories[-limit:]

    scored = []
    for m in memories:
        content = m.get("content", "").lower()
        content_words = set(content.split())

        # Calculate word overlap score
        matches = query_words & content_words
        if matches:
            # Score: proportion of query words found, weighted by match count
            score = len(matches) / len(query_words)
            # Bonus for exact phrase match
            if query.lower() in content:
                score = min(1.0, score + 0.3)
            scored.append({**m, "relevance_score": round(score, 3)})

    # Sort by relevance (highest first), then by timestamp (newest first)
    scored.sort(key=lambda x: (-x["relevance_score"], x.get("timestamp", "")))

    return scored[:limit]


def get_memory_summary() -> str:
    """Get a summary of current memory state."""
    data = load_memory()
    memories = data.get("memories", [])

    # Count by category
    categories = {}
    for m in memories:
        cat = m.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1

    lines = [
        "[MEMORY] SUMMARY",
        f"   Total memories: {len(memories)}",
        f"   Created: {data.get('created', 'N/A')[:10]}",
        f"   Updated: {data.get('updated', 'N/A')[:10]}",
        "   Categories:",
    ]
    for cat, count in categories.items():
        lines.append(f"     - {cat}: {count}")

    return "\n".join(lines)
