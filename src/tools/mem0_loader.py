"""
Memory Loader for Research Agent System
Auto-loads routing rules and project context from local storage

Usage:
    python src/tools/mem0_loader.py [PROJECT_ID]

Example:
    python src/tools/mem0_loader.py PAPER2
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Local memory storage path
MEMORY_DIR = Path(__file__).parent.parent.parent / ".memory"


def get_memory_file(project_id: str = None) -> Path:
    """Get path to project's memory file"""
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / "memory.json"


def load_local_memory(project_id: str) -> dict:
    """Load memory from local JSON file"""
    memory_file = get_memory_file(project_id)
    if memory_file.exists():
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "memories": [],
        "project_id": project_id,
        "created": datetime.now().isoformat(),
    }


def save_local_memory(project_id: str, content: str, category: str = "general") -> bool:
    """Save memory to local JSON file"""
    memory_file = get_memory_file(project_id)
    data = load_local_memory(project_id)

    data["memories"].append(
        {
            "content": content,
            "category": category,
            "timestamp": datetime.now().isoformat(),
        }
    )
    data["updated"] = datetime.now().isoformat()

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


def extract_routing_keywords() -> dict:
    """Static routing keyword map (no longer tied to a specific agent)."""
    routing = {
        "visualization": [
            "tikz",
            "diagram",
            "figure",
            "chart",
            "plot",
            "latex diagram",
        ],
        "literature": [
            "literature",
            "papers",
            "search",
            "find papers",
            "academic search",
        ],
        "writing": ["write paper", "manuscript", "draft", "abstract"],
        "methodology": ["experiment", "statistics", "methodology", "survey"],
        "security": ["threat model", "security", "cyber", "risk assessment"],
        "coding": ["pytorch", "code", "implement", "debug"],
    }
    return routing


def load_routing_rules() -> dict:
    """Load routing rules from the LangGraph router (canonical source)."""
    try:
        from src.langgraph.nodes.router import ROUTING_RULES

        return {
            "name": "LangGraphRouter",
            "domains": list(ROUTING_RULES.keys()),
            "agent_count": sum(len(agents) for agents in ROUTING_RULES.values()),
        }
    except ImportError:
        return {"error": "LangGraph router not available"}


def init_project_memory(project_id: str, project_name: str, topic: str = ""):
    """Initialize memory for a new project (local storage)"""
    # Save routing rules
    save_local_memory(
        project_id,
        "ROUTING RULES: 'tikz/diagram'->TikZPlotter, 'literature/papers'->LiteratureHunter, "
        "'figure/chart'->VisualCommunicationArchitect, 'write paper'->PublicationReadyWriter, "
        "'methodology'->MethodologyArchitect, 'security/threat'->CyberSecurityArchitect",
        category="routing",
    )

    # Save project context
    save_local_memory(
        project_id,
        f"Project: {project_name}, Topic: {topic}, Status: Initialized",
        category="project_state",
    )

    logger.info("Local memory initialized for project: %s", project_id)
    logger.info("Memory file: %s", get_memory_file(project_id))


def print_context_injection(project_id: str = "PAPER2"):
    """Print context that will be injected into AI prompt"""
    print("=" * 70)
    print("[MEMORY] CONTEXT LOADED - MANDATORY ROUTING RULES")
    print("=" * 70)

    # Load routing rules from files
    rules = load_routing_rules()

    print("\n" + "=" * 70)
    print("[ROUTING] AGENT ROUTING (from MasterOrchestrator Section 6)")
    print("=" * 70)

    print("""
+----------------------------------------------------------------------+
| DOMAIN 1: STRATEGY & OPERATIONS (Lead: StrategicArchitect)           |
+----------------------------------------------------------------------+
| plan, roadmap, strategy, wbs       -> StrategicArchitect             |
| research plan, detailed plan       -> ResearchPlanGenerator          |
| track, status, log, progress       -> ProjectStateKeeper             |
| scope, boundary                    -> Scoper / ResearchScoper        |
+----------------------------------------------------------------------+
| DOMAIN 2: RESEARCH & DISCOVERY (Lead: LiteratureHunter)              |
+----------------------------------------------------------------------+
| find papers, literature, academic  -> LiteratureHunter (ACADEMIC 1ST)|
| systematic review, slr, prisma     -> SLRProtocolDroid               |
| gap, research gap                  -> GapScout                       |
| prior art, novelty check           -> PriorArtNoveltyScanner         |
| research idea, find topic          -> JournalIdeaScout               |
| dataset, find data                 -> DatasetResearchSpecialist      |
+----------------------------------------------------------------------+
| DOMAIN 3: METHODOLOGY & ANALYSIS (Lead: MethodologyArchitect)        |
+----------------------------------------------------------------------+
| methodology, research design       -> MethodologyArchitect           |
| experiment, experiment design      -> MethodologyExperimentDesigner  |
| statistics, p-value, anova         -> StatisticalAnalyst             |
| survey, likert, questionnaire      -> SurveyDesignerAnalyst          |
| qualitative, thematic              -> QualitativeCoder               |
| math model, optimization           -> AppliedMathModeler             |
| proof, theorem, verify math        -> MathProofAuditor               |
| wolfram verify, symbolic verify    -> WolframMathAuditor (FINAL STEP)|
| game theory, nash                  -> GameTheoryStrategist           |
+----------------------------------------------------------------------+
| DOMAIN 4: SECURITY & RISK (Lead: CyberSecurityArchitect)             |
+----------------------------------------------------------------------+
| security architecture, defense     -> CyberSecurityArchitect         |
| threat model, stride               -> ThreatModeler                  |
| attack, red team, kill chain       -> AdversarialAttackSimulator     |
| cyber insurance, risk transfer     -> CyberInsuranceAnalyst          |
| compliance, gdpr, hipaa            -> RegulatoryComplianceAuditor    |
+----------------------------------------------------------------------+
| DOMAIN 5: WRITING & SYNTHESIS (Lead: PublicationReadyWriter)         |
+----------------------------------------------------------------------+
| paper outline, structure paper     -> PaperOutlineArchitect          |
| write paper, manuscript, draft     -> PublicationReadyWriter         |
| polish, refine, style, grammar     -> WritingStylePolisher           |
| latex, bibtex, format paper        -> LatexPaperGenerator            |
| references, bibliography           -> ReferenceManager               |
| optimize bib, fix bibtex, doi      -> BibTeXOptimizer                |
| revise manuscript, reviewer        -> ManuscriptReviser              |
+----------------------------------------------------------------------+
| DOMAIN 6: CODING & ENGINEERING (Lead: CoderReproAgent)               |
+----------------------------------------------------------------------+
| pytorch, neural network, code      -> PyTorchImplementer             |
| debug, error, fix code             -> CodeDebugger                   |
| reproducibility, docker            -> ReproducibilityArtifactEngineer|
+----------------------------------------------------------------------+
| DOMAIN 7: VISUALIZATION (Lead: VisualCommunicationArchitect)         |
+----------------------------------------------------------------------+
| plot, chart, visualize, figure     -> VisualCommunicationArchitect   |
| tikz, latex diagram                -> TikZPlotter                    |
| mermaid, flowchart                 -> DiagramGenerator               |
| presentation, slides               -> PresentationArchitect          |
+----------------------------------------------------------------------+
| DOMAIN 8: REVIEW & QUALITY (Lead: PeerReviewer)                      |
+----------------------------------------------------------------------+
| review, audit, check, verify       -> PeerReviewer                   |
| simulate reviewer, reviewer 2      -> ReviewerSimulator              |
| citation check, verify references  -> CitationVerifier               |
| journal, venue, where to publish   -> JournalSelector                |
+----------------------------------------------------------------------+
| DOMAIN 9: BUSINESS & ENTERPRISE (Lead: CostBenefitAnalyst)           |
+----------------------------------------------------------------------+
| sme, small business, smb           -> SMETypologyArchitect           |
| cost benefit, roi, tco             -> CostBenefitAnalyst             |
| cyber insurance, premium           -> CyberInsuranceActuary          |
+----------------------------------------------------------------------+
| DOMAIN 10: INNOVATION & IDEATION (Lead: InnovationStrategist)        |
+----------------------------------------------------------------------+
| brainstorm, ideation, ideas        -> BrainstormingFacilitator       |
| innovate, novel idea               -> InnovationStrategist           |
| system dynamics, feedback loop     -> SystemDynamicsMapper           |
+----------------------------------------------------------------------+
""")

    # Load from local memory file
    memory_data = load_local_memory(project_id)
    if memory_data.get("memories"):
        print("\n[PROJECT] STATE (from local memory):")
        for mem in memory_data.get("memories", []):
            if isinstance(mem, dict):
                print(
                    f"  - [{mem.get('category', 'general')}] {mem.get('content', mem)}"
                )
    else:
        print("\n[INFO] No project memory found. Run with --init to initialize.")

    print("\n" + "=" * 70)
    print("[WARN] MANDATORY BEFORE EXECUTING ANY TASK:")
    print("=" * 70)
    print("  1. IDENTIFY keywords in user request")
    print("  2. LOOKUP correct agent in routing table above")
    print("  3. READ agent's .json file: agents/[AgentName].json")
    print("  4. ADOPT that agent's system_prompt for execution")
    print("  5. FOLLOW the agent's design standards")
    print("=" * 70)


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "PAPER2"

    if len(sys.argv) > 2 and sys.argv[2] == "--init":
        # Initialize new project memory
        project_name = sys.argv[3] if len(sys.argv) > 3 else "Research Project"
        topic = sys.argv[4] if len(sys.argv) > 4 else ""
        init_project_memory(project, project_name, topic)
    else:
        # Normal context loading
        print_context_injection(project)
