"""
State Schema for LangGraph Research System
Defines the TypedDict structure for graph state.
"""

from typing import TypedDict, List, Optional, Dict, Any, Tuple


class ToolResult(TypedDict, total=False):
    """Structured result from a single tool execution."""

    tool_name: str
    args: Dict[str, Any]
    result: Any
    status: str  # "success", "error", "timeout", "skipped"
    duration_ms: float
    error: Optional[str]


class ExecutionMetadata(TypedDict, total=False):
    """Metadata about the overall pipeline execution."""

    tools_executed: int
    tools_success: int
    tools_failed: int
    tools_skipped: int
    total_duration_ms: float
    pipeline_route: str  # e.g. "cybersecurity/ThreatModeler"
    pipeline_nodes: List[str]


class ResearchState(TypedDict):
    """
    State schema for the research orchestration graph.

    Routing fields:
        user_message: Original user request
        original_prompt: Raw user input (before optimization)
        optimized_message: Actual optimized query for routing
        optimizer_prompt: PromptOptimizer system prompt for context
        keywords: Extracted keywords from user message
        target_domain: Classified domain (1 of 10)
        target_agent: Agent name to handle the request

    Agent fields:
        agent_prompt: System prompt for the target agent
        agent_config: Full agent configuration (role, goal, etc.)
        tools_available: List of tool descriptors (ToolInfo dicts) the agent can use

    Tool execution fields:
        tool_results: Structured results from auto-executed tools
        tool_errors: Errors from failed tool executions

    Memory fields:
        memory_context: Loaded memory from .memory/memory.json

    Output fields:
        response: Human-readable formatted response (CLI display)
        structured_output: Final packaged output for host AI consumption
        execution_metadata: Timing, tool count, pipeline trace
        execution_log: Step-by-step log of pipeline execution

    Self-improvement fields:
        is_feedback: True if user message is negative feedback on previous run
        feedback_context: Context about the previous execution being criticized
        lessons_applied: Lessons from lessons_learned DB injected into this run
        quality_assessment: Quality flags from output self-assessment
    """

    # Routing
    user_message: str
    original_prompt: Optional[str]
    optimized_message: Optional[str]
    optimizer_prompt: Optional[str]
    keywords: List[str]
    target_domain: Optional[str]
    target_agent: Optional[str]

    # Agent
    agent_prompt: Optional[str]
    agent_config: Optional[Dict[str, Any]]
    tools_available: Optional[List[Any]]  # List[ToolInfo] — Any for TypedDict compat

    # Tool execution
    tool_results: List[ToolResult]
    tool_errors: List[ToolResult]

    # Memory
    memory_context: Optional[Dict[str, Any]]

    # Output
    response: Optional[str]
    structured_output: Optional[Dict[str, Any]]
    execution_metadata: Optional[ExecutionMetadata]
    execution_log: List[str]

    # Self-improvement
    is_feedback: bool
    feedback_context: Optional[Dict[str, Any]]
    lessons_applied: List[Dict[str, Any]]
    quality_assessment: Optional[Dict[str, Any]]


def create_initial_state(user_message: str) -> ResearchState:
    """Create an initial state from user message."""
    return ResearchState(
        user_message=user_message,
        original_prompt=user_message,
        optimized_message=None,
        optimizer_prompt=None,
        keywords=[],
        target_domain=None,
        target_agent=None,
        agent_prompt=None,
        agent_config=None,
        tools_available=None,
        tool_results=[],
        tool_errors=[],
        memory_context=None,
        response=None,
        structured_output=None,
        execution_metadata=None,
        execution_log=[],
        # Self-improvement fields
        is_feedback=False,
        feedback_context=None,
        lessons_applied=[],
        quality_assessment=None,
    )
