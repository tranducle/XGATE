#!/usr/bin/env python
"""
Simple entry point for LangGraph Research Orchestrator.
Run directly from Antigravity terminal.

Usage:
    python src/langgraph/run.py "your request here"
    python src/langgraph/run.py --help
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows Unicode Output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


from src.langgraph.graph import run_graph
from src.langgraph.nodes.router import route_request
from src.langgraph.nodes.agent_loader import format_agent_for_injection


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("""
LangGraph Research Orchestrator - Simple Runner

Usage:
    python src/langgraph/run.py "your request"
    python src/langgraph/run.py --quick "your request"  (agent name only)
    python src/langgraph/run.py --inject "your request" (full prompt injection)

Examples:
    python src/langgraph/run.py "create tikz diagram"
    python src/langgraph/run.py "find literature on cybersecurity"
    python src/langgraph/run.py --quick "threat model"
""")
        return

    if sys.argv[1] == "--quick" and len(sys.argv) > 2:
        request = " ".join(sys.argv[2:])
        domain, agent, keywords = route_request(request)
        print(agent)
        return

    if sys.argv[1] == "--inject" and len(sys.argv) > 2:
        request = " ".join(sys.argv[2:])
        domain, agent, keywords = route_request(request)
        # For SOP pipelines, inject the first agent in the pipeline
        if agent.startswith("SOP:"):
            from src.langgraph.nodes.router import SOP_REGISTRY

            sop_name = agent.replace("SOP:", "")
            pipeline = SOP_REGISTRY.get(sop_name, {}).get("pipeline", [])
            agent = pipeline[0] if pipeline else agent
        print(format_agent_for_injection(agent))
        return

    request = " ".join(sys.argv[1:])
    try:
        result = run_graph(request)
        print(result.get("response", "ERROR: No response generated"))
    except Exception as exc:
        print(f"ERROR: Pipeline failed — {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
