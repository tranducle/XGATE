"""
Centralized constants for Research Agent System.

Single source of truth for domain names, tier definitions,
and other system-wide constants.
"""

# System version
SYSTEM_VERSION = "v6.0"
SYSTEM_NAME = "ResearchAgentSystem"

# Agent tiers
TIER_1_DIRECT = 1  # Direct report to MasterOrchestrator
TIER_2_COORDINATOR = 2  # Domain lead / coordinator
TIER_3_SPECIALIST = 3  # Specialist agent

# Domain names (must match agent JSON "domain" fields and ROUTING_RULES keys)
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

# Domain leads (tier 2 coordinators)
DOMAIN_LEADS = {
    "Strategy & Operations": "StrategicArchitect",
    "Research & Discovery": "LiteratureHunter",
    "Methodology & Analysis": "MethodologyArchitect",
    "Security & Risk": "CyberSecurityArchitect",
    "Writing & Synthesis": "PublicationReadyWriter",
    "Coding & Engineering": "CoderReproAgent",
    "Visualization": "VisualCommunicationArchitect",
    "Review & Quality": "PeerReviewer",
    "Business & Enterprise": "CostBenefitAnalyst",
    "Innovation & Ideation": "InnovationStrategist",
}

# Agent cache TTL (seconds)
AGENT_CACHE_TTL = 300.0  # 5 minutes

# Memory file location (relative to project root)
MEMORY_DIR_NAME = ".memory"
MEMORY_FILE_NAME = "memory.json"
