"""
CLI Interface for LangGraph Research Orchestrator
Run from Antigravity terminal to get agent routing.

Usage:
    python -m src.langgraph.cli "your request here"
    python -m src.langgraph.cli --list-agents
    python -m src.langgraph.cli --agent TikZPlotter
    python -m src.langgraph.cli --test
"""

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.langgraph.graph import (
    run_graph,
    get_agent_for_task,
)
from src.langgraph.nodes.router import (
    route_request,
    get_routing_table,
)
from src.langgraph.nodes.agent_loader import (
    list_all_agents,
    print_agent_info,
    format_agent_for_injection,
    get_agent_summary,
)
from src.langgraph.nodes.memory import get_memory_summary
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Research Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.langgraph.cli "create tikz diagram"
  python -m src.langgraph.cli "find literature on cybersecurity"
  python -m src.langgraph.cli --agent TikZPlotter
  python -m src.langgraph.cli --list-agents
  python -m src.langgraph.cli --domains
  python -m src.langgraph.cli --test
        """,
    )

    parser.add_argument("request", nargs="?", help="User request to route")
    parser.add_argument("--agent", "-a", help="Show info for specific agent")
    parser.add_argument(
        "--list-agents", "-l", action="store_true", help="List all available agents"
    )
    parser.add_argument(
        "--domains", "-d", action="store_true", help="Show all domains and their agents"
    )
    parser.add_argument(
        "--memory", "-m", action="store_true", help="Show memory summary"
    )
    parser.add_argument(
        "--full", action="store_true", help="Show full agent prompt (not truncated)"
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Output agent config ready for prompt injection",
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Run test routing examples"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Minimal output (just agent name)"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run system health check (verifies agents, routing, SOPs)",
    )

    args = parser.parse_args()

    # Handle different modes
    if args.list_agents:
        agents = list_all_agents()
        print(f"\n[LIST] AVAILABLE AGENTS ({len(agents)} total):\n")
        for i, agent in enumerate(sorted(agents), 1):
            print(f"  {i:3}. {agent}")
        print()
        return

    if args.domains:
        routing_table = get_routing_table()
        print("\n[DOMAINS] DOMAINS AND AGENTS:\n")
        for domain, agents in routing_table.items():
            print(f"\n{'=' * 50}")
            print(f"[D] {domain}")
            print(f"{'=' * 50}")
            for agent, keywords in agents.items():
                print(f"  [A] {agent}")
                print(f"     Keywords: {', '.join(keywords[:5])}")
        print()
        return

    if args.memory:
        print(get_memory_summary())
        return

    if args.agent:
        # Validate agent name exists
        all_agents = list_all_agents()
        if args.agent not in all_agents:
            # Try case-insensitive match
            match = [a for a in all_agents if a.lower() == args.agent.lower()]
            if match:
                args.agent = match[0]
            else:
                print(
                    f"[ERROR] Agent '{args.agent}' not found. Use --list-agents to see available agents."
                )
                return
        if args.inject:
            print(format_agent_for_injection(args.agent))
        else:
            print_agent_info(args.agent)
        return

    if args.test:
        run_tests()
        return

    if args.health:
        run_health_check()
        return

    if args.request:
        if args.quiet:
            # Minimal output
            domain, agent, keywords = route_request(args.request)
            print(agent)
        elif args.inject:
            # Output ready for injection
            domain, agent, keywords = route_request(args.request)
            print(format_agent_for_injection(agent))
        else:
            # Full routing result
            result = run_graph(args.request)
            print(result.get("response", "ERROR"))
    else:
        parser.print_help()


def run_tests():
    """Run test routing examples."""
    test_cases = [
        "create tikz diagram for FAIR framework",
        "find literature on cybersecurity in SMEs",
        "write methodology section",
        "analyze data with t-test",
        "threat model for zero trust",
        "brainstorm research ideas",
        "fix my bibtex references",
        "design an experiment",
        "cost benefit analysis for cyber insurance",
        "review my paper draft",
    ]

    print("\n" + "=" * 60)
    print("[TEST] ROUTING TESTS")
    print("=" * 60)

    for test in test_cases:
        domain, agent, keywords = route_request(test)
        print(f"\n[Q] '{test}'")
        print(f"   -> Domain: {domain}")
        print(f"   -> Agent: {agent}")
        print(f"   -> Keywords: {keywords}")

    print("\n" + "=" * 60)
    print("[OK] All tests completed")
    print("=" * 60)


def run_health_check():
    """Run comprehensive system health check (LangGraph-only)."""
    from src.langgraph.nodes.router import ROUTING_RULES, SOP_REGISTRY
    from src.langgraph.nodes.agent_loader import load_agent
    from src.core.constants import DOMAINS, DOMAIN_LEADS

    agents_dir = PROJECT_ROOT / "agents"
    all_json = sorted([f.stem for f in agents_dir.glob("*.json")])
    errors = 0
    warnings = 0

    print("\n" + "=" * 60)
    print("[HEALTH] SYSTEM HEALTH CHECK")
    print("=" * 60)

    print(f"\n[1] Agent JSON files: {len(all_json)}")

    load_fails = []
    for name in all_json:
        config = load_agent(name)
        if config is None:
            load_fails.append(name)
    if load_fails:
        print(f"[FAIL] {len(load_fails)} agents failed to load: {load_fails}")
        errors += len(load_fails)
    else:
        print("[2] All agents load successfully: OK")

    routing_agents = set()
    for domain, agents in ROUTING_RULES.items():
        for agent in agents:
            routing_agents.add(agent)
    ghosts = routing_agents - set(all_json)
    orphans = set(all_json) - routing_agents - {"MasterOrchestrator"}
    if ghosts:
        print(f"[FAIL] Ghost agents in routing (no JSON): {sorted(ghosts)}")
        errors += len(ghosts)
    else:
        print(f"[3] All {len(routing_agents)} routing agents have JSON files: OK")
    if orphans:
        print(f"[WARN] {len(orphans)} orphan agents (in JSON, not in routing)")
        warnings += len(orphans)

    domain_map = {}
    for domain, agents in ROUTING_RULES.items():
        for agent_name in agents:
            domain_map[agent_name] = domain
    mismatches = []
    for agent_name, expected in domain_map.items():
        config = load_agent(agent_name)
        if config and config.get("domain", "") != expected:
            mismatches.append(
                f"{agent_name}: JSON='{config.get('domain')}' vs ROUTING='{expected}'"
            )
    if mismatches:
        print(f"[FAIL] {len(mismatches)} domain mismatches:")
        for m in mismatches:
            print(f"       {m}")
        errors += len(mismatches)
    else:
        print("[4] Domain consistency: OK")

    print(f"[5] SOP registry: {len(SOP_REGISTRY)} SOPs loaded")

    keyword_agents = defaultdict(set)
    for domain, agents_dict in ROUTING_RULES.items():
        for agent_name, keywords in agents_dict.items():
            for kw in keywords:
                keyword_agents[kw].add(agent_name)
    conflicts = {kw: a for kw, a in keyword_agents.items() if len(a) > 1}
    print(f"[6] Keyword conflicts: {len(conflicts)} (threshold < 15)")

    print("\n" + "=" * 60)
    if errors == 0:
        print(f"[HEALTH] SYSTEM HEALTHY -- {warnings} warnings")
    else:
        print(f"[HEALTH] {errors} ERRORS, {warnings} WARNINGS")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
