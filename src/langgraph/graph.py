"""
LangGraph StateGraph Definition for Research Agent System
Defines the main orchestration graph with nodes and edges.
"""

from typing import Dict, Any, Literal
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
from .state import ResearchState, create_initial_state
from .nodes.router import route_request, extract_keywords
from .nodes.agent_loader import load_agent, get_agent_prompt, format_agent_for_injection
from .nodes.memory import load_memory, log_execution, get_memory_summary
from .nodes.prompt_optimizer import optimize_prompt, inject_lessons_into_prompt
from .nodes.tool_execution_node import tool_execution_node
from .nodes.output_builder import output_builder_node
from .tools_integration import get_tools_for_agent
from .nodes.feedback_handler import (
    detect_feedback,
    load_previous_execution,
    build_feedback_context,
    search_lessons,
)

# Check if langgraph is available
try:
    from langgraph.graph import StateGraph, END

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None
    END = "END"


def _strip_injected_context(message: Any) -> str:
    if not message:
        return message

    markers = (
        "\n\n## SYSTEM LESSONS LEARNED",
        "\n\n## USER FEEDBACK ON PREVIOUS EXECUTION",
    )

    cut = len(message)
    for marker in markers:
        idx = message.find(marker)
        if idx != -1 and idx < cut:
            cut = idx

    return message[:cut].strip() if cut < len(message) else message


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


def keyword_extraction_node(state: ResearchState) -> ResearchState:
    """Extract keywords from user message.
    Uses optimized_message (translated/cleaned) if available, so Vietnamese
    queries get proper keyword matching after translation.
    """
    message = state.get("optimized_message") or state["user_message"] or ""
    message = _strip_injected_context(message)
    try:
        keywords = extract_keywords(message)
    except Exception as exc:
        logger.warning("Keyword extraction failed: %s", exc)
        keywords = []
    state["keywords"] = keywords
    state["execution_log"].append(f"Extracted keywords: {keywords}")
    return state


def feedback_detection_node(state: ResearchState) -> ResearchState:
    """Detect user feedback / complaints and load previous execution context.

    This is the FIRST node in the pipeline. If the user message is negative
    feedback, it sets is_feedback=True and populates feedback_context so
    downstream nodes (prompt_optimizer, output_builder) can act on it.
    """
    user_message = state["user_message"]

    try:
        is_feedback, matched_patterns = detect_feedback(user_message)
    except Exception as exc:
        logger.warning("Feedback detection failed: %s", exc)
        is_feedback, matched_patterns = False, []

    state["is_feedback"] = is_feedback

    if is_feedback:
        previous = load_previous_execution(limit=1)
        ctx = build_feedback_context(user_message, previous, matched_patterns)
        state["feedback_context"] = ctx
        state["execution_log"].append(
            f"[FEEDBACK] Detected negative feedback (patterns: {matched_patterns[:3]}). "
            f"Previous execution loaded: {previous is not None}"
        )
    else:
        state["execution_log"].append("[OK] No feedback detected — normal execution")

    return state


def prompt_optimization_node(state: ResearchState) -> ResearchState:
    """
    Optimize the user prompt before routing.
    This is the FIRST step in the pipeline - before keyword extraction.

    Applies rule-based optimization:
      Step 0: Language detection & Vietnamese translation
      Step 1: Intent classification
      Step 2: Domain hint extraction
      Step 3: Prompt restructuring

    Also loads the PromptOptimizer agent config for downstream context.
    """
    # Store original prompt
    state["original_prompt"] = state["user_message"]

    # Run the rule-based optimizer
    try:
        result = optimize_prompt(state["user_message"])
        state["optimized_message"] = result["optimized_message"]
        state["execution_log"].append(
            f"[OK] Prompt optimized: lang={result['original_language']}, "
            f"intent={result['intent']}, "
            f"domains={result['detected_domain_hints'][:3]}"
        )
    except Exception as e:
        # Fallback: use raw message if optimizer fails
        state["optimized_message"] = state["user_message"]
        state["execution_log"].append(f"[WARN] Prompt optimization failed: {e}")

    # Step 4: Inject lessons learned and feedback context into the prompt
    try:
        state["optimized_message"] = inject_lessons_into_prompt(
            state["optimized_message"], state
        )
    except Exception as e:
        logger.warning("Lessons injection failed: %s", e)

    # Load PromptOptimizer agent config (for Antigravity context injection)
    optimizer_config = load_agent("PromptOptimizer")

    if optimizer_config:
        state["optimizer_prompt"] = optimizer_config.get("system_prompt", "")
        state["execution_log"].append("[OK] PromptOptimizer context loaded")
    else:
        state["optimizer_prompt"] = ""
        state["execution_log"].append(
            "[WARN] PromptOptimizer not found, skipping context"
        )

    return state


def routing_node(state: ResearchState) -> ResearchState:
    """Route request to appropriate domain and agent.
    Uses the optimized message if available, otherwise falls back to raw input.
    """
    message = state.get("optimized_message") or state["user_message"] or ""
    message = _strip_injected_context(message)
    try:
        domain, agent, matched = route_request(message)
    except Exception as exc:
        logger.warning("Routing failed: %s — defaulting to MasterOrchestrator", exc)
        domain, agent, matched = "Strategy & Operations", "MasterOrchestrator", []
    state["target_domain"] = domain
    state["target_agent"] = agent
    state["execution_log"].append(f"Routed to {domain}/{agent}")
    return state


def agent_loading_node(state: ResearchState) -> ResearchState:
    """Load the target agent's configuration."""
    agent_name = state.get("target_agent")
    if not agent_name:
        state["agent_prompt"] = "ERROR: No target agent specified"
        return state

    # Handle SOP pipeline routing
    if agent_name.startswith("SOP:"):
        from src.langgraph.nodes.router import SOP_REGISTRY

        sop_name = agent_name.replace("SOP:", "")
        sop_info = SOP_REGISTRY.get(sop_name, {})
        pipeline = sop_info.get("pipeline", [])
        description = sop_info.get("description", "")

        if pipeline:
            # Load the FIRST agent in the pipeline
            first_agent = pipeline[0]
            config = load_agent(first_agent)
            if config:
                # Prepend SOP pipeline info to the system prompt
                sop_header = (
                    f"# SOP PIPELINE: {sop_name}\n"
                    f"# Description: {description}\n"
                    f"# Pipeline: {' -> '.join(pipeline)}\n"
                    f"# You are agent 1/{len(pipeline)}: {first_agent}\n"
                    f"# After completing your task, hand off to: {pipeline[1] if len(pipeline) > 1 else 'DONE'}\n"
                    f"{'=' * 60}\n\n"
                )
                state["agent_config"] = config
                state["agent_prompt"] = sop_header + config.get("system_prompt", "")
                state["execution_log"].append(
                    f"Loaded SOP pipeline: {sop_name} ({len(pipeline)} agents)"
                )
                return state

        state["agent_prompt"] = f"ERROR: SOP '{sop_name}' not found or has no pipeline"
        return state

    config = load_agent(agent_name)
    if config:
        state["agent_config"] = config
        state["agent_prompt"] = config.get("system_prompt", "")
        state["execution_log"].append(f"Loaded agent config: {agent_name}")
    else:
        state["agent_prompt"] = f"ERROR: Agent '{agent_name}' not found"
        state["execution_log"].append(f"ERROR: Agent not found: {agent_name}")

    return state


def memory_loading_node(state: ResearchState) -> ResearchState:
    """Load basic JSON memory and merge context module snapshots."""
    memory = load_memory()

    # Merge context-module snapshot (ProjectState, SessionManager, ResearchDiary)
    try:
        from .nodes.memory import get_context_snapshot

        snapshot = get_context_snapshot(limit=10)
        if snapshot:
            memory["context_modules"] = snapshot
            state["execution_log"].append(
                f"Context modules loaded: {list(snapshot.keys())}"
            )
    except Exception as exc:
        logger.warning("Context snapshot unavailable: %s", exc)

    state["memory_context"] = memory
    state["execution_log"].append("Loaded memory context")
    return state


def tools_loading_node(state: ResearchState) -> ResearchState:
    """Load available tools for the target agent.
    For SOP pipelines (target_agent='SOP:SOP_NAME'), loads tools for the
    first agent in the pipeline from the already-loaded agent config.
    """
    agent_name = state.get("target_agent")
    if agent_name:
        # For SOP pipelines, extract the first agent name from loaded config
        if agent_name.startswith("SOP:"):
            config = state.get("agent_config", {})
            lookup_name = config.get("name", "")
            tools = get_tools_for_agent(lookup_name) if lookup_name else []
        else:
            tools = get_tools_for_agent(agent_name)
        state["tools_available"] = tools
        state["execution_log"].append(
            f"Loaded {len(tools)} tools for {agent_name}: {tools}"
        )
    else:
        state["tools_available"] = []
        state["execution_log"].append("No target agent, skipping tools loading")
    return state


# ============================================================================
# GRAPH CREATION
# ============================================================================

_COMPILED_GRAPH = None
_COMPILED_GRAPH_LOCK = None

try:
    import threading

    _COMPILED_GRAPH_LOCK = threading.Lock()
except ImportError:
    pass


def create_research_graph():
    """
    Create or return the cached LangGraph StateGraph for research orchestration.
    Thread-safe singleton — compiles once per process.
    """
    global _COMPILED_GRAPH

    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH

    if not HAS_LANGGRAPH:
        logger.warning("LangGraph not installed. Run: pip install langgraph")
        return None

    if _COMPILED_GRAPH_LOCK:
        with _COMPILED_GRAPH_LOCK:
            if _COMPILED_GRAPH is not None:
                return _COMPILED_GRAPH
            _COMPILED_GRAPH = _build_graph()
    else:
        _COMPILED_GRAPH = _build_graph()

    return _COMPILED_GRAPH


def _build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("feedback_detector", feedback_detection_node)
    workflow.add_node("optimize_prompt", prompt_optimization_node)
    workflow.add_node("extract_keywords", keyword_extraction_node)
    workflow.add_node("route", routing_node)
    workflow.add_node("load_agent", agent_loading_node)
    workflow.add_node("load_tools", tools_loading_node)
    workflow.add_node("load_memory", memory_loading_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("output_builder", output_builder_node)

    workflow.set_entry_point("feedback_detector")
    workflow.add_edge("feedback_detector", "optimize_prompt")
    workflow.add_edge("optimize_prompt", "extract_keywords")
    workflow.add_edge("extract_keywords", "route")
    workflow.add_edge("route", "load_agent")
    workflow.add_edge("load_agent", "load_tools")
    workflow.add_edge("load_tools", "load_memory")
    workflow.add_edge("load_memory", "tool_execution")
    workflow.add_edge("tool_execution", "output_builder")
    workflow.add_edge("output_builder", END)

    logger.info("Compiled LangGraph research graph with feedback detection (singleton)")
    return workflow.compile()


def invalidate_graph_cache():
    """Force recompilation on next call. Useful after pipeline rewiring."""
    global _COMPILED_GRAPH
    _COMPILED_GRAPH = None


def run_graph(user_message: str) -> Dict[str, Any]:
    """
    Run the research graph with a user message.

    Args:
        user_message: The user's request

    Returns:
        Final state dictionary
    """
    # Initialise context modules once (idempotent)
    try:
        from .nodes.memory import init_context_modules

        init_context_modules(str(PROJECT_ROOT))
    except Exception as exc:
        logger.debug("Context module init skipped: %s", exc)

    graph = create_research_graph()

    if graph is None:
        # Fallback without LangGraph
        return run_without_langgraph(user_message)

    initial_state = create_initial_state(user_message)
    result = graph.invoke(initial_state)
    return result


def run_without_langgraph(user_message: str) -> Dict[str, Any]:
    """
    Run the orchestration pipeline without LangGraph (fallback).

    This allows the system to work even without langgraph installed.
    """
    state = create_initial_state(user_message)

    state = feedback_detection_node(state)
    state = prompt_optimization_node(state)
    state = keyword_extraction_node(state)
    state = routing_node(state)
    state = agent_loading_node(state)
    state = tools_loading_node(state)
    state = memory_loading_node(state)
    state = tool_execution_node(state)
    state = output_builder_node(state)

    return state


# ============================================================================
# QUICK API FOR CLI
# ============================================================================


def route_and_get_prompt(message: str) -> str:
    """
    Quick API: Route message and return formatted prompt.

    Args:
        message: User request

    Returns:
        Formatted string with agent info and system prompt
    """
    result = run_without_langgraph(message)
    return result.get("response", "ERROR: No response generated")


def get_agent_for_task(message: str) -> Dict[str, str]:
    """
    Get the recommended agent for a task.

    Returns:
        Dict with domain, agent, and prompt info
    """
    domain, agent, keywords = route_request(message)
    prompt = get_agent_prompt(agent)

    return {
        "domain": domain,
        "agent": agent,
        "keywords": keywords,
        "prompt_preview": prompt[:500] if prompt else "",
    }
