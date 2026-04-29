"""
Core module for ResearchAgentSystem.

Exports constants and base exception class used by the pipeline.
"""

from .constants import DOMAINS, DOMAIN_LEADS, SYSTEM_VERSION, SYSTEM_NAME
from .exceptions import AgentError

__all__ = [
    "DOMAINS",
    "DOMAIN_LEADS",
    "SYSTEM_VERSION",
    "SYSTEM_NAME",
    "AgentError",
]
