"""
Output Builder Node for LangGraph Research System.

Packages the final pipeline output into two forms:
  1. structured_output — Dict for host AI (Antigravity/OpenCode/Gemini CLI/Claude Code)
     to consume and synthesize a response from.
  2. response — Human-readable CLI display string.

Also persists an execution record to outputs/{date}/{time}_{agent}/.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.langgraph.state import ResearchState, ExecutionMetadata
from src.langgraph.nodes.tool_execution_node import CHECKPOINT_AGENTS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _serialize_tool_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure tool results are JSON-serializable (truncate large payloads)."""
    MAX_RESULT_CHARS = 8000
    serialized = []
    for r in results:
        entry = dict(r)
        # Truncate very large result payloads
        result_val = entry.get("result")
        if result_val is not None:
            try:
                result_str = json.dumps(result_val, default=str)
            except (TypeError, ValueError):
                result_str = str(result_val)
            if len(result_str) > MAX_RESULT_CHARS:
                entry["result"] = result_str[:MAX_RESULT_CHARS]
                entry["_truncated"] = True
        serialized.append(entry)
    return serialized


def _build_structured_output(state: ResearchState) -> Dict[str, Any]:
    """
    Build the structured output dict consumed by the host AI.

    Format:
    {
      "agent_context": { name, domain, system_prompt, role, goal },
      "user_request": "...",
      "tool_results": [ {tool_name, args, result, status, duration_ms}, ... ],
      "tool_errors": [ ... ],
      "memory_context": { ... },
      "execution_metadata": { tools_executed, tools_failed, tools_skipped, total_duration_ms, pipeline_route },
      "requires_approval": bool,
      "synthesis_instructions": "..."
    }
    """
    agent_config = state.get("agent_config") or {}
    agent_name = state.get("target_agent", "Unknown")
    domain = state.get("target_domain", "Unknown")

    tool_results = state.get("tool_results", [])
    tool_errors = state.get("tool_errors", [])

    # Compute execution metadata
    n_success = sum(1 for r in tool_results if r.get("status") == "success")
    n_skipped = sum(1 for r in tool_results if r.get("status") == "skipped")
    n_failed = len(tool_errors)
    total_duration = sum(r.get("duration_ms", 0) for r in tool_results) + sum(
        r.get("duration_ms", 0) for r in tool_errors
    )

    requires_approval = agent_name in CHECKPOINT_AGENTS

    # Build synthesis instructions based on what tools returned
    if requires_approval:
        synthesis = (
            "IMPORTANT: This agent requires user approval before proceeding. "
            "Present the tool results below to the user and ask: "
            "'Do you want to proceed with these results?' "
            "Do NOT synthesize a final response until the user approves."
        )
    elif n_success > 0 and n_skipped > 0:
        synthesis = (
            "Based on the agent context and tool results above, synthesize a response for the user. "
            "Some tools were skipped because they require parameters that must be provided by the user. "
            "Check the 'parameter_guide' in skipped tool results and ask the user for any needed values "
            "if they are critical to the task."
        )
    elif n_success > 0:
        synthesis = (
            "Based on the agent context and tool results above, synthesize a comprehensive response "
            "for the user. Use the real data returned by the tools — do not fabricate or simulate results."
        )
    elif n_skipped > 0 and n_success == 0:
        synthesis = (
            "No tools could be auto-executed because they all require specific parameters. "
            "Review the parameter guides in the skipped results and ask the user to provide "
            "the necessary information, then re-run."
        )
    else:
        synthesis = (
            "No tools were available or executed. Use the agent context and your knowledge "
            "to respond to the user's request."
        )

    return {
        "agent_context": {
            "name": agent_name,
            "domain": domain,
            "system_prompt": state.get("agent_prompt", ""),
            "role": agent_config.get("role", ""),
            "goal": agent_config.get("goal", ""),
        },
        "user_request": state.get("user_message", ""),
        "tool_results": _serialize_tool_results(tool_results),
        "tool_errors": _serialize_tool_results(tool_errors),
        "memory_context": state.get("memory_context") or {},
        "execution_metadata": {
            "tools_executed": n_success + n_failed,
            "tools_success": n_success,
            "tools_failed": n_failed,
            "tools_skipped": n_skipped,
            "total_duration_ms": round(total_duration, 2),
            "pipeline_route": f"{domain}/{agent_name}",
        },
        "requires_approval": requires_approval,
        "synthesis_instructions": synthesis,
    }


def _build_cli_display(state: ResearchState, output_dir: str) -> str:
    """
    Build a human-readable CLI display string.

    Shows routing result, tool execution summary, and output location.
    """
    agent = state.get("target_agent", "Unknown")
    domain = state.get("target_domain", "Unknown")
    prompt = state.get("agent_prompt", "")
    tool_results = state.get("tool_results", [])
    tool_errors = state.get("tool_errors", [])

    n_success = sum(1 for r in tool_results if r.get("status") == "success")
    n_skipped = sum(1 for r in tool_results if r.get("status") == "skipped")
    n_failed = len(tool_errors)

    lines = [
        "=" * 60,
        "[LANGGRAPH] PIPELINE RESULT",
        "=" * 60,
        f"Request: {state.get('user_message', '')}",
        f"Keywords: {state.get('keywords', [])}",
        f"Domain: {domain}",
        f"Agent: {agent}",
        f"Output Dir: {output_dir}",
        "-" * 60,
        f"Tools: {n_success} executed, {n_skipped} skipped, {n_failed} failed",
        "-" * 60,
    ]

    # Show tool results summary
    for r in tool_results:
        status = r.get("status", "unknown")
        name = r.get("tool_name", "?")
        dur = r.get("duration_ms", 0)
        if status == "success":
            lines.append(f"  [OK] {name} ({dur:.0f}ms)")
        elif status == "skipped":
            lines.append(f"  [SKIP] {name} (needs parameters)")
        else:
            lines.append(f"  [{status.upper()}] {name}")

    for r in tool_errors:
        name = r.get("tool_name", "?")
        err = r.get("error", "unknown error")
        lines.append(f"  [FAIL] {name}: {err}")

    # Truncated system prompt preview
    lines.extend(
        [
            "=" * 60,
            "AGENT CONTEXT (system prompt preview):",
            "-" * 60,
            prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
            "=" * 60,
        ]
    )

    return "\n".join(lines)


def _persist_execution_record(
    state: ResearchState,
    structured: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Save execution record to output directory.

    Files written:
      - routing_info.json: routing + metadata
      - tool_results.json: all tool results
      - structured_output.json: full structured output for host AI
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # routing_info.json
        routing_info = {
            "timestamp": datetime.now().isoformat(),
            "user_message": state.get("user_message", ""),
            "domain": state.get("target_domain", ""),
            "agent": state.get("target_agent", ""),
            "keywords": state.get("keywords", []),
            "tools_executed": structured.get("execution_metadata", {}).get(
                "tools_executed", 0
            ),
            "output_dir": output_dir,
        }
        with open(output_path / "routing_info.json", "w", encoding="utf-8") as f:
            json.dump(routing_info, f, indent=2, ensure_ascii=False, default=str)

        # tool_results.json
        tool_data = {
            "tool_results": structured.get("tool_results", []),
            "tool_errors": structured.get("tool_errors", []),
            "execution_metadata": structured.get("execution_metadata", {}),
        }
        with open(output_path / "tool_results.json", "w", encoding="utf-8") as f:
            json.dump(tool_data, f, indent=2, ensure_ascii=False, default=str)

        # structured_output.json — the full package (without system prompt to save space)
        output_for_save = dict(structured)
        if "agent_context" in output_for_save:
            ctx = dict(output_for_save["agent_context"])
            ctx.pop("system_prompt", None)  # Don't persist large prompt
            output_for_save["agent_context"] = ctx
        with open(output_path / "structured_output.json", "w", encoding="utf-8") as f:
            json.dump(output_for_save, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Saved execution record to %s", output_dir)

    except Exception as e:
        logger.warning("Could not save execution record: %s", e)


def output_builder_node(state: ResearchState) -> ResearchState:
    """
    LangGraph node: build structured output and CLI display from pipeline state.

    This replaces the old output_formatting_node + tool_enforcement.
    Also runs quality self-assessment and saves auto-detected lessons.
    """
    agent = state.get("target_agent", "Unknown")

    # Generate timestamped output directory
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    agent_clean = agent.replace("SOP:", "").replace(" ", "_")
    output_dir = str(PROJECT_ROOT / "outputs" / date_str / f"{time_str}_{agent_clean}")

    # Log execution event
    from src.langgraph.nodes.memory import log_execution

    log_execution(agent, f"Processed: {state.get('user_message', '')[:50]}...")

    # Build structured output
    structured = _build_structured_output(state)

    # --- Quality self-assessment (self-improvement) ---
    try:
        from src.langgraph.nodes.feedback_handler import (
            assess_execution_quality,
            save_auto_lessons,
        )

        quality = assess_execution_quality(
            tool_results=state.get("tool_results", []),
            tool_errors=state.get("tool_errors", []),
            agent_name=agent,
        )
        state["quality_assessment"] = quality
        structured["quality_assessment"] = {
            "quality_score": quality["quality_score"],
            "issues": quality["issues"],
        }

        if quality["auto_lessons"]:
            saved_count = save_auto_lessons(quality["auto_lessons"])
            state["execution_log"].append(
                f"[SELF-IMPROVE] Auto-saved {saved_count} lesson(s) from quality assessment"
            )

        if quality["issues"]:
            structured["synthesis_instructions"] += (
                "\n\nQUALITY WARNING: The following issues were detected during execution:\n"
                + "\n".join(f"- {iss}" for iss in quality["issues"])
                + "\nConsider addressing these issues or suggesting the user re-run with different parameters."
            )

    except Exception as exc:
        logger.debug("Quality self-assessment skipped: %s", exc)

    # --- Save feedback-triggered lesson if this was a feedback run ---
    if state.get("is_feedback") and state.get("feedback_context"):
        try:
            from src.langgraph.nodes.feedback_handler import save_lesson

            fb_ctx = state["feedback_context"]
            save_lesson(
                issue=f"User feedback on {fb_ctx.get('previous_agent', 'unknown')}: "
                f"{fb_ctx.get('feedback_message', '')[:200]}",
                resolution=f"Re-ran with improvements. Previous request: "
                f"{fb_ctx.get('previous_request', '')[:200]}",
                agent=fb_ctx.get("previous_agent", ""),
                trigger="user",
                keywords=[
                    fb_ctx.get("previous_agent", "").lower(),
                    "user_feedback",
                ],
            )
            state["execution_log"].append(
                "[SELF-IMPROVE] Saved user feedback as lesson learned"
            )
        except Exception as exc:
            logger.debug("Feedback lesson save skipped: %s", exc)

    state["structured_output"] = structured

    # Build execution metadata for state
    meta = structured.get("execution_metadata", {})
    state["execution_metadata"] = ExecutionMetadata(
        tools_executed=meta.get("tools_executed", 0),
        tools_success=meta.get("tools_success", 0),
        tools_failed=meta.get("tools_failed", 0),
        tools_skipped=meta.get("tools_skipped", 0),
        total_duration_ms=meta.get("total_duration_ms", 0.0),
        pipeline_route=meta.get("pipeline_route", ""),
        pipeline_nodes=[
            "feedback_detector",
            "optimize_prompt",
            "extract_keywords",
            "route",
            "load_agent",
            "load_tools",
            "load_memory",
            "tool_execution",
            "output_builder",
        ],
    )

    # Build CLI display string
    state["response"] = _build_cli_display(state, output_dir)

    # Persist execution record to disk
    _persist_execution_record(state, structured, output_dir)

    state["execution_log"].append(
        f"[OUTPUT] Structured output built, saved to {output_dir}"
    )

    return state
