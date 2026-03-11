"""
Tool Execution Node for LangGraph Research System.

Reads state["tools_available"] (List[ToolInfo]) and auto-executes tools:
  - Auto-fillable tools (have a 'query' parameter): fill from user_message, execute
  - Non-auto-fillable tools: skip with status "skipped", include parameter guide
    so the host AI knows what parameters are needed

Results are stored in state["tool_results"] and state["tool_errors"].
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.langgraph.state import ResearchState, ToolResult
from src.langgraph.tool_executor import execute_tools_parallel, execute_tool_sync
from src.utils.config import get_execution_config

logger = logging.getLogger(__name__)


# ============================================================================
# PARAMETER GUIDES — reused from retired tool_enforcement.py
# Maps parameter names to human-readable instructions for the host AI
# ============================================================================

_PARAM_GUIDES: Dict[str, str] = {
    # Search
    "query": "AUTO-FILLED from user request",
    "limit": "Use default or adjust based on scope",
    "year_from": "Set to current_year - 5 for recent papers, or as specified",
    # Wolfram / SymPy
    "equation": "Extract the equation/formula from user's message",
    "expression": "Extract the mathematical expression from user's message",
    "lhs": "Left-hand side of the equation",
    "rhs": "Right-hand side of the equation",
    "variable": "The variable to solve for (e.g., 'x', 't')",
    "variables": "List of variables involved",
    "ode": "The ODE expression from user's message",
    "solution": "The proposed solution to verify",
    "func": "The function to analyze",
    "point": "The point at which to evaluate",
    "matrix_a": "First matrix from user's message",
    "matrix_b": "Second matrix (if applicable)",
    "matrix": "The matrix from user's message",
    "operation": "Matrix operation: 'determinant', 'inverse', 'eigenvalues', etc.",
    "coefficients": "Coefficient matrix for the linear system",
    "constants": "Constants vector for the linear system",
    "distribution": "Distribution name (e.g., 'normal', 'poisson')",
    "property_name": "Property to compute: 'mean', 'variance', 'pdf', etc.",
    "params": "Distribution parameters as dict",
    "objective": "Objective function to optimize",
    "constraints": "List of constraint expressions",
    "model_description": "Full mathematical model description from user",
    # Writing / BibTeX
    "doi": "Extract DOI from user's message or search results",
    "bibtex_path": "Path to the .bib file in the project",
    "filepath": "File path — ask user or use project outputs/ directory",
    "content": "The text content to write",
    "section_name": "Section name (e.g., 'Introduction', 'Methodology')",
    "citation_key": "BibTeX citation key",
    "title": "Paper/section title",
    # Analysis / Statistics
    "data": "Numeric data array — load from file or extract from context",
    "group1": "First data group for comparison",
    "group2": "Second data group for comparison",
    "x_data": "X-axis data array",
    "y_data": "Y-axis data array",
    "alpha": "Significance level (default 0.05)",
    "test_type": "Statistical test type: 't-test', 'chi-square', 'anova', etc.",
    "hypothesis": "Null hypothesis statement",
    # Convert
    "file_path": "Path to the file to convert",
    "url": "URL to convert to markdown",
    "output_format": "Output format (default: markdown)",
    # General
    "papers": "List of paper dicts from search results above",
}

# Agents that require user approval before synthesis
CHECKPOINT_AGENTS = frozenset(
    {
        "LiteratureHunter",
        "PublicationReadyWriter",
        "MethodologyArchitect",
        "CyberSecurityArchitect",
    }
)


def _build_auto_fill_kwargs(
    tool_info: Dict[str, Any],
    user_message: str,
) -> Optional[Dict[str, Any]]:
    """
    Attempt to build kwargs for a tool by auto-filling from user_message.

    Returns kwargs dict if ALL required params can be filled, else None.

    Auto-fill rules:
      - 'query' param → filled with user_message
      - Optional params with defaults → use defaults
      - Required params without defaults that aren't 'query' → can't auto-fill
    """
    params = tool_info.get("parameters", [])
    if not params:
        # No parameters means the tool takes no args — callable as-is
        return {}

    kwargs: Dict[str, Any] = {}
    for param in params:
        pname = param.get("name", "")
        required = param.get("required", False)
        default = param.get("default")

        if pname == "query":
            kwargs["query"] = user_message
        elif default is not None:
            kwargs[pname] = default
        elif not required:
            # Optional with no default — skip
            pass
        else:
            # Required param without default that isn't 'query' — can't auto-fill
            return None

    return kwargs


def _build_skip_result(tool_info: Dict[str, Any]) -> ToolResult:
    """
    Build a 'skipped' ToolResult with parameter guide for the host AI.
    """
    params = tool_info.get("parameters", [])
    param_guide: List[Dict[str, str]] = []
    for param in params:
        pname = param.get("name", "")
        ptype = param.get("type", "string")
        required = param.get("required", False)
        guide_text = _PARAM_GUIDES.get(pname, f"Provide '{pname}' from user context")
        param_guide.append(
            {
                "name": pname,
                "type": ptype,
                "required": "yes" if required else "no",
                "guide": guide_text,
            }
        )

    return ToolResult(
        tool_name=tool_info.get("name", "unknown"),
        args={},
        result={
            "message": "Tool requires parameters that cannot be auto-filled from the user query.",
            "parameter_guide": param_guide,
            "description": tool_info.get("description", ""),
            "category": tool_info.get("category", ""),
        },
        status="skipped",
        duration_ms=0.0,
        error=None,
    )


def _resolve_on_failure(state: ResearchState, exec_cfg: Dict[str, Any]) -> str:
    """
    Resolve the on_failure strategy.

    Precedence: agent_config["on_failure_default"] > exec_cfg["on_failure_default"] > "skip"
    """
    agent_config = state.get("agent_config") or {}
    agent_level = agent_config.get("on_failure_default")
    if agent_level in ("skip", "stop", "retry"):
        return agent_level
    cfg_level = exec_cfg.get("on_failure_default")
    if cfg_level in ("skip", "stop", "retry"):
        return cfg_level
    return "skip"


def tool_execution_node(state: ResearchState) -> ResearchState:
    """
    LangGraph node: auto-execute tools loaded into state["tools_available"].

    For each ToolInfo in tools_available:
      - If auto-fillable (has 'query' param or all required params have defaults):
        execute the tool with auto-filled kwargs
      - Otherwise: record as "skipped" with parameter guide for host AI

    on_failure strategies (resolved per-agent or from config):
      - "skip":  failed tools go to tool_errors, pipeline continues (default)
      - "stop":  first tool failure halts remaining executions
      - "retry": uses tool_executor retry mechanism, then falls through to "skip"

    Stores results in state["tool_results"] and errors in state["tool_errors"].
    """
    tools_available = state.get("tools_available") or []
    user_message = state.get("user_message", "")
    agent_name = state.get("target_agent", "Unknown")

    if not tools_available:
        state["execution_log"].append("[TOOLS] No tools available — skipping execution")
        return state

    exec_cfg = get_execution_config()
    max_iterations = exec_cfg.get("max_tool_iterations", 6)
    on_failure = _resolve_on_failure(state, exec_cfg)

    start_ms = time.time() * 1000

    to_execute: List[Dict[str, Any]] = []
    skipped_results: List[ToolResult] = []

    for tool_info in tools_available:
        if not isinstance(tool_info, dict):
            continue

        tool_name = tool_info.get("name", "")
        tool_callable = tool_info.get("callable")

        if tool_callable is None:
            logger.warning("Tool '%s' has no callable — skipping", tool_name)
            skipped_results.append(
                ToolResult(
                    tool_name=tool_name,
                    args={},
                    result={"message": "No callable found for this tool"},
                    status="skipped",
                    duration_ms=0.0,
                    error="No callable registered",
                )
            )
            continue

        kwargs = _build_auto_fill_kwargs(tool_info, user_message)
        if kwargs is not None:
            to_execute.append(
                {
                    "tool_name": tool_name,
                    "func": tool_callable,
                    "kwargs": kwargs,
                }
            )
        else:
            skipped_results.append(_build_skip_result(tool_info))

    if len(to_execute) > max_iterations:
        logger.warning(
            "Truncating tool execution: %d tools exceed max_tool_iterations=%d",
            len(to_execute),
            max_iterations,
        )
        for excess in to_execute[max_iterations:]:
            skipped_results.append(
                ToolResult(
                    tool_name=excess["tool_name"],
                    args=excess["kwargs"],
                    result={
                        "message": f"Skipped: exceeds max_tool_iterations ({max_iterations})"
                    },
                    status="skipped",
                    duration_ms=0.0,
                    error=None,
                )
            )
        to_execute = to_execute[:max_iterations]

    executed_results: List[Dict[str, Any]] = []
    pipeline_halted = False

    if to_execute:
        state["execution_log"].append(
            f"[TOOLS] Executing {len(to_execute)} auto-fillable tools "
            f"for {agent_name} (on_failure={on_failure})"
        )

        if on_failure == "stop":
            # Sequential execution — halt on first failure
            timeout = exec_cfg.get("tool_timeout", 30)
            retries = exec_cfg.get("tool_retries", 2)
            for t in to_execute:
                r = execute_tool_sync(
                    t["tool_name"], t["func"], t.get("kwargs", {}), timeout, retries
                )
                executed_results.append(r)
                if r.get("status") != "success":
                    pipeline_halted = True
                    state["execution_log"].append(
                        f"[TOOLS] HALT: on_failure=stop, tool '{t['tool_name']}' failed"
                    )
                    remaining_idx = to_execute.index(t) + 1
                    for remaining in to_execute[remaining_idx:]:
                        skipped_results.append(
                            ToolResult(
                                tool_name=remaining["tool_name"],
                                args=remaining.get("kwargs", {}),
                                result={
                                    "message": "Skipped: pipeline halted by on_failure=stop"
                                },
                                status="skipped",
                                duration_ms=0.0,
                                error=None,
                            )
                        )
                    break
        else:
            # "skip" or "retry" — parallel execution, errors are non-fatal
            executed_results = execute_tools_parallel(to_execute, exec_cfg)

    tool_results: List[ToolResult] = []
    tool_errors: List[ToolResult] = []

    for r in executed_results:
        tr = ToolResult(
            tool_name=r.get("tool_name", "unknown"),
            args=r.get("args", {}),
            result=r.get("result"),
            status=r.get("status", "error"),
            duration_ms=r.get("duration_ms", 0.0),
            error=r.get("error"),
        )
        if tr.get("status") in ("success",):
            tool_results.append(tr)
        else:
            tool_errors.append(tr)

    tool_results.extend(skipped_results)

    total_ms = round(time.time() * 1000 - start_ms, 2)

    state["tool_results"] = tool_results
    state["tool_errors"] = tool_errors

    n_executed = len(executed_results)
    n_success = sum(1 for r in executed_results if r.get("status") == "success")
    n_failed = len(tool_errors)
    n_skipped = len(skipped_results)

    summary = (
        f"[TOOLS] Complete: {n_executed} executed ({n_success} success, "
        f"{n_failed} failed), {n_skipped} skipped, {total_ms:.0f}ms total"
    )
    if pipeline_halted:
        summary += " [HALTED by on_failure=stop]"
    state["execution_log"].append(summary)

    if agent_name in CHECKPOINT_AGENTS:
        state["execution_log"].append(
            f"[CHECKPOINT] Agent '{agent_name}' requires user approval before synthesis"
        )

    return state
