"""
Agent Loader Node for LangGraph Research System
Loads agent configurations from agents/*.json files.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from ...schemas.agent_schema import AgentDefinition

logger = logging.getLogger(__name__)


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
AGENTS_DIR = PROJECT_ROOT / "agents"


def load_agent(
    agent_name: str, validate_definition: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Load agent configuration from JSON file.

    Args:
        agent_name: Name of the agent (e.g., "TikZPlotter")
        validate_definition: If True, validate JSON with AgentDefinition schema

    Returns:
        Agent configuration dictionary or None if not found
    """
    agent_file = AGENTS_DIR / f"{agent_name}.json"

    if not agent_file.exists():
        # Try case-insensitive search
        for f in AGENTS_DIR.glob("*.json"):
            if f.stem.lower() == agent_name.lower():
                logger.debug(f"Case-insensitive match: '{agent_name}' -> '{f.stem}'")
                agent_file = f
                break
        else:
            return None

    try:
        with open(agent_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            if validate_definition:
                AgentDefinition.model_validate(config)
            return config
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading agent {agent_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error validating agent {agent_name}: {e}")
        return None


def get_agent_prompt(agent_name: str) -> str:
    """
    Get the system prompt for an agent.

    Args:
        agent_name: Name of the agent

    Returns:
        System prompt string or error message
    """
    config = load_agent(agent_name)
    if config is None:
        return f"ERROR: Agent '{agent_name}' not found in agents/ directory."

    return config.get(
        "system_prompt", f"Agent {agent_name} has no system_prompt defined."
    )


def get_agent_role(agent_name: str) -> str:
    """Get agent's role description."""
    config = load_agent(agent_name)
    if config:
        return config.get("role", agent_name)
    return agent_name


def get_agent_goal(agent_name: str) -> str:
    """Get agent's goal description."""
    config = load_agent(agent_name)
    if config:
        return config.get("goal", "")
    return ""


def get_agent_domain(agent_name: str) -> str:
    """Get agent's domain."""
    config = load_agent(agent_name)
    if config:
        return config.get("domain", "Unknown")
    return "Unknown"


def list_all_agents() -> List[str]:
    """List all available agents."""
    if not AGENTS_DIR.exists():
        return []
    return [f.stem for f in AGENTS_DIR.glob("*.json")]


def get_agent_summary(agent_name: str) -> Dict[str, str]:
    """
    Get a summary of an agent's configuration.

    Returns:
        Dict with name, role, goal, domain
    """
    config = load_agent(agent_name)
    if config is None:
        return {"error": f"Agent '{agent_name}' not found"}

    return {
        "name": config.get("name", agent_name),
        "role": config.get("role", ""),
        "goal": config.get("goal", ""),
        "domain": config.get("domain", ""),
        "description": config.get("description", ""),
    }


def format_agent_for_injection(agent_name: str) -> str:
    """
    Format agent config for prompt injection into Antigravity.

    Returns:
        Formatted string ready to be used as system context
    """
    config = load_agent(agent_name)
    if config is None:
        return f"# ERROR: Agent '{agent_name}' not found"

    lines = [
        f"# AGENT: {config.get('name', agent_name)}",
        f"## Role: {config.get('role', 'N/A')}",
        f"## Domain: {config.get('domain', 'N/A')}",
        f"## Goal: {config.get('goal', 'N/A')}",
        "",
        "## System Prompt:",
        config.get("system_prompt", "No system prompt defined."),
    ]

    return "\n".join(lines)


def print_agent_info(agent_name: str) -> None:
    """Print agent information for debugging."""
    config = load_agent(agent_name)
    if config is None:
        logger.warning(f"Agent '{agent_name}' not found")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info(f"AGENT: {config.get('name', agent_name)}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Role: {config.get('role', 'N/A')}")
    logger.info(f"Domain: {config.get('domain', 'N/A')}")
    logger.info(f"Goal: {config.get('goal', 'N/A')[:100]}...")
    logger.info(f"{'=' * 60}")
    logger.info("System Prompt Preview:")
    prompt = config.get("system_prompt", "")
    logger.info(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    logger.info(f"{'=' * 60}")
