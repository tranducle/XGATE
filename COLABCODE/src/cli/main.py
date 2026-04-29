"""
Command-line interface for Research Agent System.

Routes all requests through the LangGraph pipeline.
No legacy orchestration - LangGraph is the sole execution engine.
"""

import argparse
import logging
import sys
from typing import Optional

from ..utils.logger import setup_logging
from ..utils.config import load_config, load_env_vars


logger = logging.getLogger(__name__)


class CLI:
    """Command-line interface for the agent system (LangGraph-only)."""

    def __init__(self, config: dict = None):
        """
        Initialize the CLI.

        Args:
            config: Optional configuration dict
        """
        self.config = config or load_config()
        self._graph_module = None

    def _get_graph(self):
        """Lazy-import the LangGraph module to avoid circular imports."""
        if self._graph_module is None:
            from ..langgraph.graph import (
                run_graph,
                get_agent_for_task,
            )
            from ..langgraph.nodes.router import route_request
            from ..langgraph.nodes.agent_loader import list_all_agents

            self._graph_module = {
                "run_graph": run_graph,
                "get_agent_for_task": get_agent_for_task,
                "route_request": route_request,
                "list_all_agents": list_all_agents,
            }
        return self._graph_module

    def initialize(self) -> None:
        """Initialize the agent system."""
        log_level = self.config.get("log_level", "INFO")
        setup_logging(level=log_level)
        logger.info("Initializing Research Agent System (LangGraph mode)...")

        env_vars = load_env_vars()
        logger.info(
            f"API keys loaded: {sum(1 for v in env_vars.values() if v)}/{len(env_vars)}"
        )

        # Validate LangGraph pipeline is importable
        self._get_graph()
        logger.info("System initialized successfully")

    def run_query(self, query: str) -> None:
        """
        Run a single query through the LangGraph pipeline.

        Args:
            query: User query
        """
        g = self._get_graph()
        result = g["run_graph"](query)
        self._display_result(result)

    def run_interactive(self) -> None:
        """Run interactive REPL mode."""
        self.initialize()

        print("\n" + "=" * 60)
        print("Research Agent System - Interactive Mode (LangGraph)")
        print("=" * 60)
        print("\nEnter your queries or type 'help' for commands.")
        print("Type 'quit' or 'exit' to leave.\n")

        g = self._get_graph()

        while True:
            try:
                user_input = input("\033[32m>\033[0m ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    print("\nGoodbye!")
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if user_input.lower() == "agents":
                    self._list_agents()
                    continue

                if user_input.lower().startswith("route "):
                    query = user_input[6:].strip()
                    if query:
                        info = g["get_agent_for_task"](query)
                        print(f"\n  Domain: {info['domain']}")
                        print(f"  Agent:  {info['agent']}")
                        print(f"  Keywords: {info['keywords']}\n")
                    continue

                # Process as query through LangGraph
                result = g["run_graph"](user_input)
                self._display_result(result)

            except KeyboardInterrupt:
                print("\n\nUse 'quit' to exit.")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\033[31mError: {e}\033[0m")

    def _display_result(self, result: dict) -> None:
        """Display pipeline result."""
        response = result.get("response", "")
        if response:
            print(response)
        else:
            print("\033[33mNo response generated.\033[0m")

    def _show_help(self) -> None:
        """Show help information."""
        print("\n\033[36mAvailable Commands:\033[0m")
        print("  help           - Show this help message")
        print("  agents         - List all available agents")
        print("  route <query>  - Show routing info without full execution")
        print("  quit/exit      - Exit interactive mode")
        print("\n\033[36mUsage Tips:\033[0m")
        print("  - Use natural language queries")
        print("  - The system routes to appropriate agents automatically")
        print("  - Example: 'Find papers on federated learning from 2023-2024'")

    def _list_agents(self) -> None:
        """List available agents."""
        g = self._get_graph()
        agents = g["list_all_agents"]()
        print(f"\n\033[36mAvailable Agents ({len(agents)}):\033[0m\n")
        for i, agent in enumerate(sorted(agents), 1):
            print(f"  {i:3}. {agent}")
        print()


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Research Agent System - LangGraph Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Find papers on federated learning from 2023-2024"
  %(prog)s -i
  %(prog)s --list-agents
        """,
    )

    parser.add_argument("query", nargs="?", help="Research query or task")

    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Launch interactive mode"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--list-agents", action="store_true", help="List all available agents"
    )

    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    if args.verbose:
        config["log_level"] = "DEBUG"

    cli = CLI(config)

    try:
        if args.list_agents:
            cli.initialize()
            cli._list_agents()
            return 0

        if args.interactive:
            cli.run_interactive()
            return 0

        if args.query:
            cli.initialize()
            cli.run_query(args.query)
            return 0

        parser.print_help()
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\033[31mError: {e}\033[0m")
        return 1


def main_sync() -> int:
    """
    Synchronous wrapper for main().

    Returns:
        Exit code
    """
    # main() is now synchronous (LangGraph pipeline is sync)
    return main()


if __name__ == "__main__":
    sys.exit(main_sync())
