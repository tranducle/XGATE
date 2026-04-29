"""
Agent Schema Definitions for Research Agent System v5
Production-ready agent definitions with tools, memory, and state management.
"""

from typing import List, Dict, Optional, Any, Callable, Literal
from pydantic import BaseModel, Field
from enum import Enum


class MemoryType(str, Enum):
    """Memory types available for agents."""

    NONE = "none"
    BUFFER = "buffer"  # Recent conversation context
    SUMMARY = "summary"  # Summarized history
    VECTOR = "vector"  # Semantic search over history


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""

    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    """
    Definition of a callable tool for an agent.
    Tools enable agents to interact with external systems.
    """

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="What this tool does - shown to LLM")
    parameters: List[ToolParameter] = Field(default_factory=list)
    function_path: Optional[str] = Field(
        None,
        description="Python path to actual function, e.g., 'src.tools.literature.search_semantic_scholar'",
    )
    requires_approval: bool = Field(
        False, description="If True, requires human approval before execution"
    )


class TriggerCondition(BaseModel):
    """Conditions that trigger routing to this agent."""

    keywords: List[str] = Field(
        default_factory=list, description="Keywords that trigger this agent"
    )
    intent_patterns: List[str] = Field(
        default_factory=list, description="Regex patterns for intent matching"
    )
    priority: int = Field(
        default=0, description="Higher = more priority when multiple agents match"
    )


class WorkflowStep(BaseModel):
    """A single step in a multi-step workflow."""

    name: str
    description: str
    agent_name: Optional[str] = None  # Can delegate to another agent
    tool_name: Optional[str] = None  # Or use a specific tool
    input_mapping: Dict[str, str] = Field(default_factory=dict)  # Map outputs to inputs
    output_key: str = "result"
    requires_approval: bool = False
    on_failure: Literal["stop", "skip", "retry"] = "stop"
    max_retries: int = 3


class AgentDefinition(BaseModel):
    """
    Complete definition of a research agent.
    This schema replaces the old YAML-only format with proper tool support.
    """

    # Identity
    name: str = Field(..., description="Unique agent name, e.g., 'LiteratureHunter'")
    role: str = Field(..., description="Agent's role description")
    goal: str = Field(..., description="What this agent aims to achieve")
    backstory: str = Field(
        ..., description="Background context for the agent's expertise"
    )

    # Domain classification
    domain: str = Field(..., description="One of the 10 strategic domains")
    is_domain_lead: bool = Field(
        False, description="Whether this agent leads its domain"
    )

    # Capabilities
    tools: List[ToolDefinition] = Field(
        default_factory=list, description="Available tools"
    )
    can_delegate_to: List[str] = Field(
        default_factory=list, description="Agent names this can delegate to"
    )

    # Memory & State
    memory_type: MemoryType = Field(
        MemoryType.BUFFER, description="Type of memory to use"
    )
    max_context_tokens: int = Field(
        128000, description="Max tokens for context window (modern LLMs support 100k+)"
    )

    # Execution
    max_iterations: int = Field(10, description="Max reasoning iterations")
    requires_human_approval: bool = Field(
        False, description="Always require approval before action"
    )

    # Routing
    triggers: TriggerCondition = Field(default_factory=TriggerCondition)

    # Multi-step workflows (optional)
    workflow_steps: List[WorkflowStep] = Field(default_factory=list)

    # Original system prompt (for migration compatibility)
    system_prompt: Optional[str] = Field(
        None, description="Legacy system prompt from YAML"
    )

    def to_system_message(self) -> str:
        """Generate system message for LLM from agent definition."""
        tools_desc = ""
        if self.tools:
            tools_desc = "\n\n## Available Tools\n"
            for tool in self.tools:
                params = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters])
                tools_desc += f"- **{tool.name}**({params}): {tool.description}\n"

        return f"""# Role: {self.role}

## Goal
{self.goal}

## Backstory
{self.backstory}
{tools_desc}

## Instructions
- Use your tools to accomplish tasks
- If you need human input, clearly indicate what you need
- Stay focused on your goal and domain expertise
- If a task is outside your domain, suggest delegation to appropriate agent
"""


class AgentRegistry(BaseModel):
    """Registry of all available agents."""

    agents: Dict[str, AgentDefinition] = Field(default_factory=dict)
    domain_leads: Dict[str, str] = Field(default_factory=dict)  # domain -> agent_name

    def register(self, agent: AgentDefinition) -> None:
        """Register an agent."""
        self.agents[agent.name] = agent
        if agent.is_domain_lead:
            self.domain_leads[agent.domain] = agent.name

    def get_by_domain(self, domain: str) -> List[AgentDefinition]:
        """Get all agents in a domain."""
        return [a for a in self.agents.values() if a.domain == domain]

    def find_by_keywords(self, query: str) -> List[AgentDefinition]:
        """Find agents matching keywords in query."""
        query_lower = query.lower()
        matches = []
        for agent in self.agents.values():
            for keyword in agent.triggers.keywords:
                if keyword.lower() in query_lower:
                    matches.append((agent.triggers.priority, agent))
                    break
        # Sort by priority descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in matches]


# Standard domains for the Research Agent System
DOMAINS = [
    "Strategy & Operations",
    "Research & Discovery",
    "Methodology & Analysis",
    "Security & Risk",
    "Writing & Synthesis",
    "Coding & Engineering",
    "Visualization",
    "Review & Quality",
    "Business & Enterprise",
    "Innovation & Ideation",
]
