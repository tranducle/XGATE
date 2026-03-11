"""
Configuration management for ResearchAgentSystem.

This module provides configuration loading and management.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load system configuration from file.

    Args:
        config_file: Path to configuration file

    Returns:
        Dict containing configuration
    """
    config_path = Path(config_file)

    default_config = {
        "session_dir": ".sessions",
        "log_level": "INFO",
        "execution": {
            "auto_confirm": False,
            "parallel_agents": True,
            "max_parallel": 3,
            "tool_timeout": 30,
            "tool_retries": 2,
            "max_tool_iterations": 6,
            "parallel_tools": True,
            "output_format": "structured",
        },
    }

    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    default_config.update(loaded_config)
                    logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
    else:
        logger.debug(f"Config file not found: {config_file}, using defaults")

    return default_config


def load_env_vars(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file

    Returns:
        Dict of environment variables
    """
    env_path = Path(env_file)

    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded environment variables from {env_file}")

    # Return relevant API keys
    return {
        "SERPAPI_KEY": os.getenv("SERPAPI_KEY", ""),
        "SEMANTIC_SCHOLAR_KEY": os.getenv("SEMANTIC_SCHOLAR_KEY", ""),
        "SCOPUS_API_KEY": os.getenv("SCOPUS_API_KEY", ""),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    }


_EXECUTION_DEFAULTS = {
    "auto_confirm": False,
    "parallel_agents": True,
    "max_parallel": 3,
    "tool_timeout": 30,
    "tool_retries": 2,
    "max_tool_iterations": 6,
    "parallel_tools": True,
    "output_format": "structured",
    "on_failure_default": "skip",  # "skip" | "stop" | "retry" — per-tool failure strategy
}


def get_execution_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract and validate execution config with defaults for missing keys.
    """
    if config is None:
        config = load_config()

    exec_cfg = config.get("execution", {})
    if not isinstance(exec_cfg, dict):
        exec_cfg = {}

    merged = {**_EXECUTION_DEFAULTS, **exec_cfg}

    merged["tool_timeout"] = max(1, int(merged["tool_timeout"]))
    merged["tool_retries"] = max(0, int(merged["tool_retries"]))
    merged["max_tool_iterations"] = max(1, int(merged["max_tool_iterations"]))
    merged["max_parallel"] = max(1, int(merged["max_parallel"]))
    merged["parallel_tools"] = bool(merged["parallel_tools"])
    merged["parallel_agents"] = bool(merged["parallel_agents"])
    merged["auto_confirm"] = bool(merged["auto_confirm"])

    if merged["output_format"] not in ("structured", "text", "json"):
        merged["output_format"] = "structured"

    if merged.get("on_failure_default") not in ("skip", "stop", "retry"):
        merged["on_failure_default"] = "skip"

    return merged
