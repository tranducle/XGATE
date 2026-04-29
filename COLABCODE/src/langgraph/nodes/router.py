"""
Router Node for LangGraph Research System
Handles keyword extraction and agent routing based on MasterOrchestrator rules.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# ============================================================================
# LITERATURE SEARCH PRIORITY ORDER (Updated 2026-01-20)
# When LiteratureHunter is invoked, sub-agents are called in this order:
# ============================================================================
LITERATURE_SEARCH_PRIORITY = [
    "OpenAlexSearch",  # PRIORITY 1: Open catalog, 250M+ papers, free API
    "GoogleScholarSearch",  # PRIORITY 2: Broad coverage, recent papers, patents
    "ScopusSearch",  # PRIORITY 3: High-impact Q1/Q2 papers
    "SemanticSearch",  # PRIORITY 4: Citation graphs, deep metadata
    # GeneralWebSearcher: LOW - Only on explicit "web search" or "internet" request
]

# ============================================================================
# CITATION VERIFICATION CONFIG (Updated 2026-01-20)
# Citation audit MUST run passes with all 4 search tools
# ============================================================================
CITATION_VERIFICATION_CONFIG = {
    "min_passes": 3,
    "search_order": [
        "OpenAlexSearch",
        "GoogleScholarSearch",
        "ScopusSearch",
        "SemanticSearch",
    ],
    "per_pass_agents": ["CitationIntegrityAuditor"],
    "final_agents": ["CitationVerifier", "ReferenceManager"],
}


# ============================================================================
# ROUTING RULES - Complete 115 Agents from MasterOrchestrator.json
# ============================================================================

ROUTING_RULES: Dict[str, Dict[str, List[str]]] = {
    # Domain 1: Strategy & Operations (Lead: StrategicArchitect)
    "Strategy & Operations": {
        "StrategicArchitect": ["plan", "roadmap", "strategy", "blueprint", "wbs"],
        "ProjectPlanner": [
            "wbs",
            "gantt",
            "project plan",
            "timeline",
            "agile",
            "sprint",
            "kanban",
            "scrum",
        ],
        "LogicStrategist": ["logic", "mece", "hypothesis", "problem solve"],
        "ResourceConstraintAuditor": ["resource", "budget", "tco", "feasibility"],
        "ProjectStateKeeper": ["track", "status", "log", "progress"],
        "ProgressTracker": ["milestone", "track progress"],
        "Scoper": ["scope", "boundary", "limit"],
        "ResearchScoper": ["research scope", "research boundary"],
        "ResearchPlanGenerator": [
            "research plan",
            "detailed plan",
            "comprehensive plan",
        ],
    },
    # Domain 2: Research & Discovery (Lead: LiteratureHunter)
    # PRIORITY ORDER: OpenAlex -> GoogleScholar -> Scopus -> Semantic (all 4 always run)
    "Research & Discovery": {
        "LiteratureHunter": [
            "find papers",
            "search",
            "literature",
            "academic search",
            "papers",
            "find research",
        ],
        "OpenAlexSearch": ["openalex", "open alex", "open catalog"],  # PRIORITY 1
        "GoogleScholarSearch": ["google scholar", "scholar"],  # PRIORITY 2
        "ScopusSearch": ["scopus", "elsevier", "high impact"],  # PRIORITY 3
        "SemanticSearch": ["semantic scholar", "citation graph"],  # PRIORITY 4
        "GeneralWebSearcher": [
            "web search",
            "general search",
            "internet",
        ],  # LOW - explicit only
        "ResearchLibrarian": ["organize files", "sdp", "file management"],
        "SLRProtocolDroid": ["systematic review", "slr", "prisma"],
        "DeepSearchPlanner": ["deep search", "multi-step search", "search strategy"],
        "GapScout": ["gap", "research gap", "opportunity"],
        "GapMapperResearchOpportunityExtractor": [
            "gap mapping",
            "opportunity extraction",
        ],
        "PriorArtNoveltyScanner": ["prior art", "novelty check", "patent"],
        "JournalIdeaScout": [
            "research idea",
            "find topic",
            "what should i research",
            "trending topics",
            "new project idea",
        ],
        "DatasetResearchSpecialist": ["dataset", "data source", "find data"],
        "FileToMarkdownConverter": [
            "convert to markdown",
            "convert pdf",
            "convert file",
            "convert docx",
            "convert pptx",
            "read pdf",
            "read docx",
            "read pptx",
            "read excel",
            "read document",
            "read this document",
            "extract text",
            "parse document",
            "file to markdown",
            "youtube transcript",
            "ocr image",
            "markitdown",
            "pdf to text",
            "convert to text",
        ],
    },
    # Domain 3: Methodology & Analysis (Lead: MethodologyArchitect)
    "Methodology & Analysis": {
        "MethodologyArchitect": ["methodology", "research design", "rigor"],
        "MethodologyExperimentDesigner": [
            "experiment",
            "experiment design",
            "hardware profile",
        ],
        "ExperimentConductor": ["run experiment", "execute experiment"],
        "StatisticalAnalyst": [
            "statistics",
            "p-value",
            "anova",
            "t-test",
            "regression",
        ],
        "SurveyDesignerAnalyst": ["survey", "likert", "questionnaire", "efa", "cfa"],
        "QualitativeCoder": [
            "qualitative",
            "thematic",
            "coding",
            "interview",
            "grounded theory",
        ],
        "AppliedMathModeler": ["math model", "optimization", "linear program", "ilp"],
        "MathArchitectureAnalyst": ["math structure", "mathematical framework"],
        "MathProofAuditor": ["proof", "theorem", "verify math"],
        "MathSymbolicSolver": ["solve equation", "symbolic", "sympy", "wolfram"],
        "EconometricsModeler": ["econometrics", "did", "iv", "rdd"],
        "CausalAnalyst": ["causal", "causal inference"],
        "CausalIdentificationStrategist": [
            "identification strategy",
            "causal identification",
        ],
        "GameTheoryStrategist": ["game theory", "nash", "attacker defender"],
        "NashEquilibriumStrategist": ["nash equilibrium", "market equilibrium"],
        "DataMetricsAnalyst": ["eda", "metrics", "data analysis"],
        "WolframMathAuditor": [
            "wolfram verify",
            "wolfram check",
            "wolfram audit",
            "symbolic verify",
            "computational verify",
            "audit math",
            "math audit",
            "verify formula",
            "check equation",
            "verify equation",
            "validate math",
            "mathematical verification",
            "computational check",
        ],
    },
    # Domain 4: Security & Risk (Lead: CyberSecurityArchitect)
    "Security & Risk": {
        "CyberSecurityArchitect": [
            "security architecture",
            "defense",
            "nist",
            "zero trust",
        ],
        "AdversarialAttackSimulator": ["attack", "red team", "kill chain", "mitre"],
        "ThreatModeler": ["threat model", "stride", "dread"],
        "SaaSShadowITCartographer": ["shadow it", "saas", "cloud risk"],
        "ProtocolNetworkSemanticsVerifier": ["protocol", "rfc", "packet"],
        "NetworkTrafficModeler": ["network", "traffic", "traffic analysis"],
        "CryptoProtocolVerifier": ["crypto", "encryption", "tls", "certificate"],
        "IncidentReadinessPlaybookGenerator": ["incident", "playbook", "ir"],
        "MinViableSecurityArchitect": ["sme security", "lightweight security"],
        "RedTeamEthicsDualUseGuard": ["dual use", "ethics", "harm"],
        "SecurityStandardsChecker": ["nist", "iso", "security standard"],
        "SupplyChainRiskAnalyst": [
            "supply chain",
            "vendor risk",
            "tprm",
            "third party",
        ],
        "CyberInsuranceAnalyst": ["cyber insurance", "risk transfer", "coverage"],
        "CyberInsuranceActuary": ["premium", "actuarial", "insurance cost"],
        "RegulatoryComplianceAuditor": [
            "compliance",
            "gdpr",
            "hipaa",
            "iso",
            "regulatory",
        ],
    },
    # Domain 5: Writing & Synthesis (Lead: PublicationReadyWriter)
    "Writing & Synthesis": {
        "PaperOutlineArchitect": [
            "paper outline",
            "manuscript outline",
            "structure paper",
        ],
        "PublicationReadyWriter": ["write paper", "manuscript", "draft"],
        "PaperWriter": ["write section", "draft section"],
        "ManuscriptReviser": [
            "revise manuscript",
            "address reviewer",
            "revision",
            "fix paper",
            "rewrite paragraph",
        ],
        "WritingStylePolisher": ["polish", "refine", "style", "grammar"],
        "GrantProposalStrategist": ["grant", "proposal", "funding", "nsf", "nih"],
        "AbstractTitleGenerator": ["abstract", "title", "summary"],
        "LatexPaperGenerator": ["latex", "bibtex", "format paper"],
        "ReferenceManager": ["references", "bibliography", "citation manager"],
        "BibTeXOptimizer": [
            "optimize bib",
            "fix bibtex",
            "find doi",
            "clean references",
            "doi2bib",
        ],
        "DeepSynthesizer": ["synthesize", "summarize", "combine", "deep synthesis"],
        "DocumentSynthesizer": ["document synthesis", "single doc summary"],
        "MultiSourceSynthesizer": ["multi source", "cross source", "integrate sources"],
        "DialecticalSynthesizer": ["thesis antithesis", "dialectical"],
        "SummarizerSynthesizer": ["paper summary", "quick summary"],
        "DailySummarizer": ["daily summary", "session summary"],
        "CaseStudyArchivist": ["case study", "real world example"],
    },
    # Domain 6: Coding & Engineering (Lead: CoderReproAgent)
    "Coding & Engineering": {
        "PyTorchImplementer": ["pytorch", "neural network", "deep learning"],
        "DataPreprocessingEngineer": [
            "preprocess",
            "preprocessing",
            "tiền xử lý",
            "tiền xử lý dữ liệu",
            "data cleaning",
            "data preparation",
            "feature engineering",
            "feature scaling",
            "normalization",
            "standardization",
            "imputation",
            "missing values",
            "class imbalance",
            "SMOTE",
            "augmentation",
            "data augmentation",
            "one-hot",
            "encoding",
            "tokenization",
            "transform data",
            "prepare data for training",
            "prepare data",
            "prepare my data",
            "clean data",
            "clean my data",
        ],
        "CoderReproAgent": [
            "code",
            "implement",
            "programming",
            "debug",
            "error",
            "fix code",
            "refactor",
            "clean code",
        ],
        "ReproducibilityArtifactEngineer": [
            "reproducibility",
            "docker",
            "requirements.txt",
        ],
        "FrameworkArchitect": ["framework", "architecture", "system design"],
        "FrameworkValidationArchitect": ["validate framework", "framework testing"],
        "FileIntegrator": ["merge files", "file integration"],
        "HardwareresourceEstimator": ["gpu", "hardware", "memory estimate", "ram"],
        "ModelCapabilityRouter": ["which model", "model selection", "choose model"],
        "CrossDomainTransferHybridizationAgent": [
            "transfer learning",
            "cross domain",
            "domain adaptation",
        ],
    },
    # Domain 7: Visualization (Lead: VisualCommunicationArchitect)
    "Visualization": {
        "VisualCommunicationArchitect": ["plot", "chart", "visualize", "figure"],
        "TikZPlotter": ["tikz", "latex diagram"],
        "FigureGenerator": ["generate figure", "create chart", "python chart"],
        "ResultVisualizer": ["visualize results", "data visualization"],
        "HybridVisualizer": [
            "ascii diagram",
            "text diagram",
            "unicode diagram",
            "mermaid",
            "flowchart",
            "sequence diagram",
        ],
        "PresentationArchitect": ["presentation", "slides", "deck"],
        "PresentationGenerator": ["beamer", "ppt", "generate slides"],
        "ExplainabilityTranslator": ["explain", "lay summary", "simplify"],
    },
    # Domain 8: Review & Quality (Lead: PeerReviewer)
    "Review & Quality": {
        "PeerReviewer": ["review", "audit", "check", "verify"],
        "ReviewerSimulator": ["simulate reviewer", "reviewer 2"],
        "HarshReviewer": ["harsh", "critical review"],
        "ReviewerStrategist": ["respond to reviewer", "rebuttal"],
        "CitationVerifier": ["citation check", "verify references"],
        "CitationIntegrityAuditor": [
            "citation audit",
            "deep citation check",
            "hallucination",
        ],
        "FeasibilityRigorSoundnessChecker": ["rigor", "soundness", "feasibility"],
        "BaselineBenchmarkNoveltyDefender": [
            "baseline",
            "benchmark",
            "novelty defense",
        ],
        "JournalSelector": ["journal", "venue", "where to publish"],
        "EthicalComplianceGuard": ["ethics", "irb", "consent"],
        "DataPrivacyOfficer": ["privacy", "data protection", "gdpr"],
        "AgentSystemArchitect": ["self repair", "fix agent"],
        "PromptOptimizer": ["optimize prompt", "refine prompt"],
    },
    # Domain 9: Business & Enterprise (Lead: CostBenefitAnalyst)
    "Business & Enterprise": {
        "SMETypologyArchitect": [
            "sme",
            "small business",
            "micro enterprise",
            "smb",
            "naics",
            "nace",
        ],
        "CostBenefitAnalyst": [
            "cost benefit",
            "roi",
            "tco",
            "budget",
            "competitor",
            "competitive analysis",
            "market",
        ],
        "EntrepreneurialPsychProfiler": [
            "entrepreneur",
            "owner bias",
            "decision maker",
            "psychology",
        ],
        "HumanFactorCultureQuantifier": [
            "organizational culture",
            "culture audit",
            "human factor",
        ],
        "FutureScenarioForecaster": ["business scenario", "future planning", "what if"],
    },
    # Domain 10: Innovation & Ideation (Lead: InnovationStrategist)
    "Innovation & Ideation": {
        "InnovationStrategist": ["innovate", "novel idea", "creative"],
        "BrainstormingFacilitator": [
            "brainstorm",
            "ideation",
            "generate ideas",
            "ideation session",
        ],
        "IdeaMutationDesignSpaceExplorer": ["design space", "explore options"],
        "OmniThinker": ["general reasoning", "think", "analyze"],
        "SystemDynamicsMapper": [
            "system dynamics",
            "feedback loop",
            "causal loop",
            "systems thinking",
        ],
        "MissingPartSuggester": [
            "missing",
            "suggest",
            "what's next",
            "what next",
            "what do i do",
            "what should i do",
            "next step",
            "stuck",
        ],
    },
}

# ============================================================================
# INTENT SYNONYM GROUPS - Words expressing the same intent
# ============================================================================

INTENT_SYNONYMS: Dict[str, set] = {
    "verify": {
        "check",
        "verify",
        "validate",
        "assess",
        "audit",
        "evaluate",
        "confirm",
        "test",
        "examine",
    },
    "create": {
        "write",
        "draft",
        "create",
        "generate",
        "compose",
        "build",
        "make",
        "design",
        "prepare",
        "develop",
    },
    "find": {
        "find",
        "search",
        "discover",
        "identify",
        "locate",
        "explore",
        "hunt",
        "survey",
    },
    "analyze": {
        "analyze",
        "analyse",
        "investigate",
        "study",
        "review",
        "inspect",
        "diagnose",
        "compare",
    },
    "fix": {
        "fix",
        "repair",
        "correct",
        "resolve",
        "debug",
        "patch",
        "address",
        "clean",
        "optimize",
    },
    "plan": {
        "plan",
        "organize",
        "schedule",
        "roadmap",
        "strategy",
        "outline",
        "kickoff",
        "start",
        "initialize",
    },
}

# ============================================================================
# SOP CONCEPT FINGERPRINTS - What each SOP handles (concepts + intents)
# ============================================================================

SOP_CONCEPTS: Dict[str, Dict] = {
    # Research & Discovery
    "SOP_IDEA_DISCOVERY": {
        "concepts": ["idea", "topic", "research", "project"],
        "intents": ["find"],
    },
    "SOP_SYSTEMATIC_REVIEW": {
        "concepts": ["systematic", "literature review", "slr", "prisma"],
        "intents": ["find", "analyze"],
    },
    "SOP_GAP_ANALYSIS": {
        "concepts": ["gap", "missing", "opportunity", "unexplored"],
        "intents": ["find", "analyze"],
    },
    "SOP_NOVELTY_DEFENSE": {
        "concepts": ["novelty", "originality", "contribution", "prior art", "baseline"],
        "intents": ["verify", "analyze"],
    },
    "SOP_DATA_ACQUISITION": {
        "concepts": ["dataset", "data", "collect", "source"],
        "intents": ["find"],
    },
    # Methodology & Analysis
    "SOP_QUANT_EXPERIMENT": {
        "concepts": [
            "experiment",
            "hypothesis",
            "variable",
            "control",
            "statistics",
            "anova",
            "regression",
        ],
        "intents": ["create", "plan", "analyze", "verify"],
    },
    "SOP_QUAL_STUDY": {
        "concepts": ["survey", "questionnaire", "likert", "respondent"],
        "intents": ["create", "analyze"],
    },
    "SOP_CAUSAL_ANALYSIS": {
        "concepts": ["causal", "causation", "inference", "treatment"],
        "intents": ["analyze"],
    },
    "SOP_MATH_FORMULATION": {
        "concepts": ["math", "equation", "formula", "model", "theorem"],
        "intents": ["create", "analyze"],
    },
    "SOP_MATH_AUDIT": {
        "concepts": ["math", "equation", "formula", "proof"],
        "intents": ["verify"],
    },
    "SOP_GAME_THEORY_ANALYSIS": {
        "concepts": ["game theory", "nash", "equilibrium", "payoff"],
        "intents": ["analyze"],
    },
    # Security & Risk
    "SOP_CYBER_RISK_SIM": {
        "concepts": ["threat", "attack", "risk", "vulnerability"],
        "intents": ["analyze", "find"],
    },
    "SOP_DEFENSE_ARCH": {
        "concepts": ["security", "defense", "architecture", "protection"],
        "intents": ["create", "plan"],
    },
    "SOP_SME_RISK_ASSESSMENT": {
        "concepts": ["sme", "small business", "risk", "insurance"],
        "intents": ["analyze", "verify"],
    },
    "SOP_PROTOCOL_SECURITY_AUDIT": {
        "concepts": ["protocol", "rfc", "network", "packet"],
        "intents": ["verify"],
    },
    # Writing & Synthesis
    "SOP_PAPER_OUTLINE": {
        "concepts": ["paper outline", "structure paper", "outline paper"],
        "intents": ["create", "plan"],
    },
    "SOP_MANUSCRIPT_PREP": {
        "concepts": ["paper", "manuscript", "article", "publication", "journal"],
        "intents": ["create"],
    },
    "SOP_GRANT_APPLICATION": {
        "concepts": ["grant", "funding", "proposal", "application"],
        "intents": ["create"],
    },
    "SOP_SYNTHESIS_REPORTING": {
        "concepts": ["synthesis", "summary", "report", "consolidate"],
        "intents": ["create", "analyze"],
    },
    "SOP_CASE_STUDY": {
        "concepts": ["case study", "real world", "example", "scenario"],
        "intents": ["create", "analyze"],
    },
    "SOP_BIBTEX_OPTIMIZATION": {
        "concepts": ["bibtex", "reference", "doi", "bibliography"],
        "intents": ["fix"],
    },
    "SOP_MANUSCRIPT_REVISION": {
        "concepts": ["revision", "reviewer", "feedback", "rebuttal"],
        "intents": ["fix", "create"],
    },
    "SOP_DAILY_SUMMARY": {
        "concepts": ["daily", "summary", "progress", "session"],
        "intents": ["create"],
    },
    # Citation
    "SOP_CITATION_AUDIT": {
        "concepts": ["citation", "reference", "bibliography", "doi"],
        "intents": ["verify"],
    },
    # Coding & Engineering
    "SOP_CODE_IMPLEMENTATION": {
        "concepts": ["code", "implement", "program", "pytorch"],
        "intents": ["create"],
    },
    "SOP_FRAMEWORK_DEV": {
        "concepts": ["framework", "architecture", "system"],
        "intents": ["create"],
    },
    "SOP_TRANSFER_LEARNING": {
        "concepts": ["transfer", "domain", "adaptation", "pretrained"],
        "intents": ["create", "analyze"],
    },
    "SOP_DATA_PREPROCESSING": {
        "concepts": ["preprocess", "clean", "normalize", "feature"],
        "intents": ["fix", "create"],
    },
    "SOP_MODEL_SELECTION": {
        "concepts": ["model", "select", "choose", "compare"],
        "intents": ["find", "analyze"],
    },
    # Visualization
    "SOP_PRESENTATION_GEN": {
        "concepts": ["presentation", "slides", "deck", "talk"],
        "intents": ["create"],
    },
    "SOP_FIGURE_GENERATION": {
        "concepts": ["figure", "chart", "plot", "diagram", "visualization"],
        "intents": ["create"],
    },
    # Review & Audit
    "SOP_COMPREHENSIVE_REVIEW": {
        "concepts": ["paper review", "critique", "manuscript", "feedback"],
        "intents": ["analyze", "verify"],
    },
    # Innovation
    "SOP_IDEATION_SESSION": {
        "concepts": ["idea", "brainstorm", "creative", "innovate"],
        "intents": ["create", "find"],
    },
    "SOP_SYSTEMS_ANALYSIS": {
        "concepts": ["system", "feedback loop", "dynamics", "complex"],
        "intents": ["analyze"],
    },
    "SOP_SELF_REPAIR": {
        "concepts": ["agent", "repair", "self", "fix agent"],
        "intents": ["fix"],
    },
    # Project Management
    "SOP_PROJECT_KICKOFF": {
        "concepts": ["project", "kickoff", "initialize", "begin"],
        "intents": ["plan"],
    },
    "SOP_PROGRESS_TRACKING": {
        "concepts": ["progress", "milestone", "track", "status"],
        "intents": ["verify", "analyze"],
    },
}


def _detect_intents(message_lower: str) -> set:
    """Detect user intents from message using synonym groups."""
    words = set(message_lower.split())
    detected = set()
    for intent_group, synonyms in INTENT_SYNONYMS.items():
        if words & synonyms:
            detected.add(intent_group)
    return detected


def score_sop_by_intent(message: str) -> Optional[Tuple[str, float]]:
    """
    Score SOPs by semantic intent + concept matching.

    Decomposes user query into intents (what action) and concepts (what topic),
    then scores each SOP by overlap. Returns best match if above threshold.

    Args:
        message: User message (already lowercased/translated)

    Returns:
        Tuple of (sop_name, score) if score >= 0.6, else None
    """
    detected_intents = _detect_intents(message)

    if not detected_intents:
        return None

    best_sop = None
    best_score = 0.0

    for sop_name, fingerprint in SOP_CONCEPTS.items():
        concepts = fingerprint["concepts"]
        sop_intents = set(fingerprint["intents"])

        # --- Concept score (0 to 1): how many SOP concepts appear in query ---
        concept_hits = 0
        for concept in concepts:
            if concept in message:
                concept_hits += 1

        if concept_hits == 0:
            continue  # Skip SOPs with zero concept overlap

        concept_score = concept_hits / len(concepts)

        # --- Intent score (0 or 0.5): does user intent match SOP intent ---
        intent_match = detected_intents & sop_intents
        intent_score = 0.5 if intent_match else 0.0

        total_score = concept_score + intent_score

        if total_score > best_score:
            best_score = total_score
            best_sop = sop_name

    # Threshold: require at least 0.6 to avoid false positives
    # (e.g. concept_score=0.2 + intent_score=0.5 = 0.7 → valid)
    # (e.g. concept_score=0.1 + intent_score=0.0 = 0.1 → rejected)
    if best_sop and best_score >= 0.6:
        return best_sop, best_score

    return None


# ============================================================================
# SOP ROUTING - 39 Standard Operating Playbooks
# ============================================================================

SOP_REGISTRY: Dict[str, Dict] = {
    # Research & Discovery SOPs
    "SOP_IDEA_DISCOVERY": {
        "triggers": [
            "no idea",
            "new project",
            "find topic",
            "what to research",
            "tìm đề tài",
            "chưa có ý tưởng",
        ],
        "pipeline": ["JournalIdeaScout", "ResearchScoper", "ResearchPlanGenerator"],
        "description": "When user has no research idea - discover topics",
    },
    "SOP_SYSTEMATIC_REVIEW": {
        "triggers": ["systematic review", "slr", "prisma", "meta-analysis"],
        "pipeline": [
            "SLRProtocolDroid",
            "LiteratureHunter",
            "CitationIntegrityAuditor",
            "DeepSynthesizer",
            "PeerReviewer",
        ],
        "description": "PRISMA systematic literature review",
    },
    "SOP_GAP_ANALYSIS": {
        "triggers": ["find gap", "research gap", "gap analysis"],
        "pipeline": [
            "LiteratureHunter",
            "ResearchScoper",
            "GapScout",
            "InnovationStrategist",
            "MissingPartSuggester",
        ],
        "description": "Identify research gaps and opportunities",
    },
    "SOP_DATA_ACQUISITION": {
        "triggers": ["find dataset", "data acquisition", "need data"],
        "pipeline": [
            "DatasetResearchSpecialist",
            "MethodologyArchitect",
            "DataMetricsAnalyst",
            "DataPreprocessingEngineer",
            "ProjectStateKeeper",
        ],
        "description": "Find, evaluate, and preprocess datasets",
    },
    "SOP_NOVELTY_DEFENSE": {
        "triggers": ["novelty defense", "prior art", "defend contribution"],
        "pipeline": [
            "PriorArtNoveltyScanner",
            "BaselineBenchmarkNoveltyDefender",
            "InnovationStrategist",
            "ReviewerStrategist",
        ],
        "description": "Prepare novelty defense against reviewers",
    },
    "SOP_RESEARCH_PLAN_CREATION": {
        "triggers": ["create research plan", "research planning", "detailed plan"],
        "pipeline": ["ResearchPlanGenerator", "ProjectPlanner", "ProjectStateKeeper"],
        "description": "Create comprehensive research plan with WBS",
    },
    # Methodology & Analysis SOPs
    "SOP_QUANT_EXPERIMENT": {
        "triggers": [
            "quantitative experiment",
            "run experiment",
            "experimental design",
        ],
        "pipeline": [
            "MethodologyExperimentDesigner",
            "MethodologyArchitect",
            "DataPreprocessingEngineer",
            "ExperimentConductor",
            "DataMetricsAnalyst",
            "StatisticalAnalyst",
            "ResultVisualizer",
        ],
        "description": "Design, preprocess, and run quantitative experiments",
    },
    "SOP_QUAL_STUDY": {
        "triggers": ["qualitative study", "interview study", "thematic analysis"],
        "pipeline": [
            "MethodologyArchitect",
            "SurveyDesignerAnalyst",
            "QualitativeCoder",
            "DocumentSynthesizer",
        ],
        "description": "Conduct qualitative research",
    },
    "SOP_CAUSAL_ANALYSIS": {
        "triggers": ["causal analysis", "did analysis", "iv analysis", "rdd"],
        "pipeline": [
            "CausalIdentificationStrategist",
            "EconometricsModeler",
            "CausalAnalyst",
            "StatisticalAnalyst",
        ],
        "description": "Causal inference with econometric methods",
    },
    "SOP_MATH_FORMULATION": {
        "triggers": ["math formulation", "mathematical model", "prove theorem"],
        "pipeline": [
            "AppliedMathModeler",
            "MathArchitectureAnalyst",
            "MathProofAuditor",
            "MathSymbolicSolver",
            "WolframMathAuditor",
        ],
        "description": "Formulate and verify mathematical models (with Wolfram verification)",
    },
    "SOP_MATH_AUDIT": {
        "triggers": [
            "audit math",
            "math audit",
            "verify math",
            "mathematical verification",
            "wolfram check",
        ],
        "pipeline": ["MathProofAuditor", "WolframMathAuditor"],
        "description": "Mathematical audit with mandatory Wolfram Alpha verification as final step",
    },
    "SOP_GAME_THEORY_ANALYSIS": {
        "triggers": [
            "game theory analysis",
            "attacker defender",
            "strategic interaction",
        ],
        "pipeline": [
            "GameTheoryStrategist",
            "NashEquilibriumStrategist",
            "CostBenefitAnalyst",
            "FutureScenarioForecaster",
        ],
        "description": "Game-theoretic analysis of strategic interactions",
    },
    # Security & Risk SOPs
    "SOP_CYBER_RISK_SIM": {
        "triggers": ["cyber risk simulation", "attack simulation", "kill chain"],
        "pipeline": [
            "SaaSShadowITCartographer",
            "AdversarialAttackSimulator",
            "ThreatModeler",
            "SecurityStandardsChecker",
        ],
        "description": "Simulate cyber attacks and assess risk",
    },
    "SOP_DEFENSE_ARCH": {
        "triggers": ["defense architecture", "security design", "defense in depth"],
        "pipeline": [
            "CyberSecurityArchitect",
            "MinViableSecurityArchitect",
            "IncidentReadinessPlaybookGenerator",
            "RegulatoryComplianceAuditor",
        ],
        "description": "Design cybersecurity defense architecture",
    },
    "SOP_ETHICAL_AUDIT": {
        "triggers": ["ethics audit", "irb review", "dual use check"],
        "pipeline": [
            "EthicalComplianceGuard",
            "DataPrivacyOfficer",
            "RedTeamEthicsDualUseGuard",
            "RegulatoryComplianceAuditor",
        ],
        "description": "Comprehensive ethics and compliance audit",
    },
    "SOP_PROTOCOL_SECURITY_AUDIT": {
        "triggers": ["protocol audit", "rfc compliance", "crypto audit"],
        "pipeline": [
            "ProtocolNetworkSemanticsVerifier",
            "CryptoProtocolVerifier",
            "NetworkTrafficModeler",
            "ThreatModeler",
        ],
        "description": "Audit protocol security and RFC compliance",
    },
    "SOP_SME_RISK_ASSESSMENT": {
        "triggers": ["sme risk", "small business security", "sme cybersecurity"],
        "pipeline": [
            "SMETypologyArchitect",
            "EntrepreneurialPsychProfiler",
            "HumanFactorCultureQuantifier",
            "SupplyChainRiskAnalyst",
            "CyberInsuranceAnalyst",
            "CyberInsuranceActuary",
        ],
        "description": "SME-specific cybersecurity risk assessment",
    },
    # Writing & Publication SOPs
    "SOP_PAPER_OUTLINE": {
        "triggers": ["paper outline", "manuscript outline", "structure paper"],
        "pipeline": ["PaperOutlineArchitect"],
        "description": "Create hierarchical paper outline",
    },
    "SOP_MANUSCRIPT_PREP": {
        "triggers": [
            "write paper",
            "manuscript preparation",
            "draft paper",
            "viết bài báo",
            "chuẩn bị bản thảo",
        ],
        "pipeline": [
            "PaperOutlineArchitect",
            "JournalSelector",
            "PublicationReadyWriter",
            "PaperWriter",
            "CitationVerifier",
            "WritingStylePolisher",
            "CitationIntegrityAuditor",
            "PeerReviewer",
        ],
        "description": "Full manuscript preparation workflow",
    },
    "SOP_GRANT_APPLICATION": {
        "triggers": [
            "grant application",
            "funding proposal",
            "nsf grant",
            "đề xuất tài trợ",
        ],
        "pipeline": [
            "GrantProposalStrategist",
            "CostBenefitAnalyst",
            "ResourceConstraintAuditor",
            "WritingStylePolisher",
        ],
        "description": "Write grant/funding application",
    },
    "SOP_SYNTHESIS_REPORTING": {
        "triggers": ["synthesis report", "multi-source summary"],
        "pipeline": [
            "MultiSourceSynthesizer",
            "SummarizerSynthesizer",
            "AbstractTitleGenerator",
            "PublicationReadyWriter",
        ],
        "description": "Synthesize multiple sources into report",
    },
    "SOP_CASE_STUDY": {
        "triggers": ["case study", "real world example"],
        "pipeline": [
            "CaseStudyArchivist",
            "QualitativeCoder",
            "DocumentSynthesizer",
            "PublicationReadyWriter",
        ],
        "description": "Write case study research",
    },
    "SOP_BIBTEX_OPTIMIZATION": {
        "triggers": ["optimize bibtex", "fix references", "clean bibtex"],
        "pipeline": ["BibTeXOptimizer"],
        "description": "Find DOIs and optimize BibTeX references",
    },
    "SOP_MANUSCRIPT_REVISION": {
        "triggers": ["manuscript revision", "address feedback", "reviewer response"],
        "pipeline": ["ManuscriptReviser", "WritingStylePolisher", "ReviewerStrategist"],
        "description": "Revise manuscript based on reviewer feedback",
    },
    "SOP_DAILY_SUMMARY": {
        "triggers": ["daily summary", "session summary", "progress report"],
        "pipeline": ["DailySummarizer", "ProjectStateKeeper", "ProgressTracker"],
        "description": "Create daily/session summary",
    },
    # Coding & Engineering SOPs
    "SOP_CODE_IMPLEMENTATION": {
        "triggers": [
            "implement code",
            "code implementation",
            "build model",
            "viết code",
            "lập trình",
        ],
        "pipeline": [
            "MethodologyExperimentDesigner",
            "MethodologyArchitect",
            "PyTorchImplementer",
            "ReproducibilityArtifactEngineer",
        ],
        "description": "Implement research code",
    },
    "SOP_FRAMEWORK_DEV": {
        "triggers": ["build framework", "framework development"],
        "pipeline": [
            "FrameworkArchitect",
            "PyTorchImplementer",
            "FrameworkValidationArchitect",
            "FileIntegrator",
        ],
        "description": "Develop research framework",
    },
    "SOP_TRANSFER_LEARNING": {
        "triggers": ["transfer learning", "domain adaptation", "cross domain"],
        "pipeline": [
            "MethodologyExperimentDesigner",
            "CrossDomainTransferHybridizationAgent",
            "PyTorchImplementer",
            "DataMetricsAnalyst",
        ],
        "description": "Apply transfer learning methods",
    },
    "SOP_DATA_PREPROCESSING": {
        "triggers": [
            "preprocess data",
            "preprocessing pipeline",
            "prepare data for training",
            "tiền xử lý dữ liệu",
        ],
        "pipeline": [
            "DataPreprocessingEngineer",
            "DataMetricsAnalyst",
            "PyTorchImplementer",
        ],
        "description": "Complete data preprocessing pipeline for ML training",
    },
    "SOP_MODEL_SELECTION": {
        "triggers": ["model selection", "which model", "choose model"],
        "pipeline": [
            "ModelCapabilityRouter",
            "MethodologyExperimentDesigner",
            "PyTorchImplementer",
            "ExperimentConductor",
        ],
        "description": "Select appropriate model for task",
    },
    # Visualization SOPs
    "SOP_PRESENTATION_GEN": {
        "triggers": [
            "create presentation",
            "make slides",
            "presentation",
            "tạo bài thuyết trình",
            "làm slide",
        ],
        "pipeline": [
            "PresentationArchitect",
            "ExplainabilityTranslator",
            "VisualCommunicationArchitect",
            "FigureGenerator",
            "TikZPlotter",
            "PresentationGenerator",
        ],
        "description": "Generate presentation slides",
    },
    "SOP_FIGURE_GENERATION": {
        "triggers": [
            "create figures",
            "generate charts",
            "visualize data",
            "vẽ biểu đồ",
            "tạo hình",
        ],
        "pipeline": [
            "ResultVisualizer",
            "FigureGenerator",
            "TikZPlotter",
            "HybridVisualizer",
        ],
        "description": "Generate publication-quality figures",
    },
    # Review & Audit SOPs
    "SOP_COMPREHENSIVE_REVIEW": {
        "triggers": [
            "comprehensive review",
            "full review",
            "paper review",
            "đánh giá bài báo",
            "phản biện",
        ],
        "pipeline": [
            "PeerReviewer",
            "ReviewerSimulator",
            "HarshReviewer",
            "FeasibilityRigorSoundnessChecker",
            "ReviewerStrategist",
        ],
        "description": "Comprehensive manuscript review",
    },
    "SOP_CITATION_AUDIT": {
        "triggers": [
            "citation audit",
            "verify citations",
            "check references",
            "kiểm tra trích dẫn",
        ],
        "pipeline": [
            # Pass 1 (OpenAlex first)
            "CitationIntegrityAuditor",
            "OpenAlexSearch",
            "GoogleScholarSearch",
            "ScopusSearch",
            "SemanticSearch",
            # Pass 2
            "CitationIntegrityAuditor",
            "OpenAlexSearch",
            "GoogleScholarSearch",
            "ScopusSearch",
            "SemanticSearch",
            # Pass 3
            "CitationIntegrityAuditor",
            "OpenAlexSearch",
            "GoogleScholarSearch",
            "ScopusSearch",
            "SemanticSearch",
            # Final
            "CitationVerifier",
            "ReferenceManager",
        ],
        "passes": 3,
        "description": "3-pass deep citation audit with all 4 search sources (OpenAlex->GoogleScholar->Scopus->Semantic)",
    },
    # Innovation SOPs
    "SOP_IDEATION_SESSION": {
        "triggers": [
            "ideation session",
            "brainstorm session",
            "generate ideas",
            "phiên sáng tạo",
            "động não",
        ],
        "pipeline": [
            "BrainstormingFacilitator",
            "IdeaMutationDesignSpaceExplorer",
            "OmniThinker",
            "MissingPartSuggester",
            "InnovationStrategist",
        ],
        "description": "Creative ideation session",
    },
    "SOP_SYSTEMS_ANALYSIS": {
        "triggers": ["systems analysis", "feedback loops", "systems thinking"],
        "pipeline": [
            "SystemDynamicsMapper",
            "CausalAnalyst",
            "FutureScenarioForecaster",
            "ResultVisualizer",
        ],
        "description": "Analyze complex systems with feedback loops",
    },
    "SOP_SELF_REPAIR": {
        "triggers": ["self repair", "fix agent", "agent repair"],
        "pipeline": ["PeerReviewer", "AgentSystemArchitect", "PromptOptimizer"],
        "description": "Find flaws and repair agent systems",
    },
    # Project Management SOPs
    "SOP_PROJECT_KICKOFF": {
        "triggers": [
            "project kickoff",
            "start project",
            "initialize project",
            "khởi động dự án",
        ],
        "pipeline": [
            "StrategicArchitect",
            "ResearchLibrarian",
            "Scoper",
            "ResearchScoper",
            "ProjectPlanner",
            "ProjectStateKeeper",
        ],
        "description": "Initialize new research project",
    },
    "SOP_PROGRESS_TRACKING": {
        "triggers": ["track progress", "check milestones", "progress check"],
        "pipeline": [
            "ProgressTracker",
            "ProjectStateKeeper",
            "ResourceConstraintAuditor",
            "DailySummarizer",
        ],
        "description": "Track project progress and milestones",
    },
}

# Flatten for quick lookup
KEYWORD_TO_AGENT: Dict[str, Tuple[str, str]] = {}  # keyword -> (domain, agent)
for domain, agents in ROUTING_RULES.items():
    for agent, keywords in agents.items():
        for keyword in keywords:
            KEYWORD_TO_AGENT[keyword.lower()] = (domain, agent)


# ============================================================================
# VIETNAMESE -> ENGLISH KEYWORD TRANSLATION
# Lightweight bilingual support: translates common Vietnamese research phrases
# to English keywords before routing. No API needed.
# ============================================================================

VI_TO_EN: Dict[str, str] = {
    # Research & Discovery
    "tim bai bao": "find papers",
    "tim tai lieu": "literature",
    "tim kiem": "search",
    "tim": "find",
    "nghien cuu": "research",
    "bai bao": "papers",
    "tai lieu": "literature",
    "tong quan": "systematic review",
    "khoang trong": "research gap",
    "y tuong": "research idea",
    "du lieu": "dataset",
    "kiem tra moi": "novelty check",
    # Methodology & Analysis
    "phuong phap": "methodology",
    "thiet ke thi nghiem": "experiment design",
    "thi nghiem": "experiment",
    "thong ke": "statistics",
    "phan tich": "analyze",
    "khao sat": "survey",
    "dinh luong": "quantitative",
    "dinh tinh": "qualitative",
    "mo hinh toan": "math model",
    "ly thuyet tro choi": "game theory",
    "nhan qua": "causal",
    # Security & Risk
    "bao mat": "security",
    "rui ro": "risk",
    "de doa": "threat model",
    "tan cong": "attack",
    "phong thu": "defense",
    "ma hoa": "encryption",
    "tuan thu": "compliance",
    # Writing & Synthesis
    "viet bai": "write paper",
    "ban thao": "manuscript",
    "nhap": "draft",
    "de cuong": "paper outline",
    "tong hop": "synthesize",
    "tom tat": "summary",
    "chinh sua": "revise manuscript",
    "tai tro": "grant",
    "de xuat": "proposal",
    "tai lieu tham khao": "references",
    "trich dan": "citation",
    # Coding & Engineering
    "lap trinh": "programming",
    "sua loi": "debug",
    "xay dung": "implement",
    "tien xu ly": "preprocess",
    "tien xu ly du lieu": "preprocess data",
    # Visualization
    "bieu do": "chart",
    "hinh anh": "figure",
    "do thi": "plot",
    "trinh bay": "presentation",
    "slide": "slides",
    "so do": "flowchart",
    # Review & Quality
    "danh gia": "review",
    "kiem tra": "check",
    "xac minh": "verify",
    "phan bien": "critical review",
    "chat luong": "quality",
    "tap chi": "journal",
    # Strategy & Management
    "ke hoach": "plan",
    "lo trinh": "roadmap",
    "chien luoc": "strategy",
    "tien do": "progress",
    "muc tieu": "milestone",
    # Innovation
    "dong nao": "brainstorm",
    "sang tao": "creative",
}

# Pre-sort Vietnamese keys by length (longest first) for best matching
_VI_KEYS_SORTED = sorted(VI_TO_EN.keys(), key=len, reverse=True)


def _normalize_vietnamese(text: str) -> str:
    """Remove Vietnamese diacritical marks for fuzzy matching.
    Converts e.g. 'tìm bài báo' -> 'tim bai bao'.
    Also handles đ/Đ which NFKD doesn't decompose.
    """
    import unicodedata

    # Handle đ/Đ first (NFKD doesn't decompose these)
    text = text.replace("đ", "d").replace("Đ", "D")
    # Decompose into base chars + combining marks, then strip marks
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _translate_vietnamese(message: str) -> str:
    """Translate Vietnamese keywords to English equivalents.
    Returns the original message with Vietnamese phrases replaced by English.
    """
    normalized = _normalize_vietnamese(message)
    translated = normalized

    for vi_key in _VI_KEYS_SORTED:
        if vi_key in translated:
            translated = translated.replace(vi_key, VI_TO_EN[vi_key])

    # If translation changed something, return translated; else return original
    if translated != normalized:
        return translated
    return message.lower()


def extract_keywords(message: str) -> List[str]:
    """Extract potential routing keywords from user message.
    Supports both English and Vietnamese via automatic translation.
    """
    # Translate Vietnamese keywords to English before matching
    message_lower = _translate_vietnamese(message)
    found_keywords = []

    # Check for multi-word keywords first (longer matches take priority)
    sorted_keywords = sorted(KEYWORD_TO_AGENT.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in message_lower:
            found_keywords.append(keyword)
            # Remove matched keyword to avoid sub-matches
            message_lower = message_lower.replace(keyword, " ")

    return found_keywords


def route_request(message: str) -> Tuple[str, str, List[str]]:
    """
    Route user request to appropriate domain and agent.

    Args:
        message: User's request message

    Returns:
        Tuple of (domain, agent_name, matched_keywords)
    """
    message_lower = _translate_vietnamese(message)

    # ========================================================================
    # SOP PRIORITY ROUTING - Check SOP triggers FIRST before direct routing
    # This ensures complex workflows are triggered instead of single agents
    # ========================================================================
    SOP_PRIORITY_KEYWORDS = {
        # Math audit keywords → SOP_MATH_AUDIT pipeline (MathProofAuditor → WolframMathAuditor)
        "audit math": "SOP_MATH_AUDIT",
        "math audit": "SOP_MATH_AUDIT",
        "verify math": "SOP_MATH_AUDIT",
        "check math": "SOP_MATH_AUDIT",
        "validate math": "SOP_MATH_AUDIT",
        "mathematical verification": "SOP_MATH_AUDIT",
        "verify formula": "SOP_MATH_AUDIT",
        "check equation": "SOP_MATH_AUDIT",
        "verify equation": "SOP_MATH_AUDIT",
        # Citation audit → SOP_CITATION_AUDIT (3-pass)
        "citation audit": "SOP_CITATION_AUDIT",
        "verify citations": "SOP_CITATION_AUDIT",
        "check references": "SOP_CITATION_AUDIT",
        # Wolfram-specific → SOP_MATH_AUDIT (WolframMathAuditor is in pipeline)
        "wolfram verify": "SOP_MATH_AUDIT",
        "wolfram check": "SOP_MATH_AUDIT",
        "wolfram audit": "SOP_MATH_AUDIT",
    }

    # Check for SOP-triggering keywords (sorted by length for best match)
    for sop_kw in sorted(SOP_PRIORITY_KEYWORDS.keys(), key=len, reverse=True):
        if sop_kw in message_lower:
            sop_name = SOP_PRIORITY_KEYWORDS[sop_kw]
            sop_info = SOP_REGISTRY.get(sop_name, {})
            pipeline = sop_info.get("pipeline", [])

            # Return the FIRST agent in the pipeline as entry point
            # The agent system will then follow the SOP pipeline
            if pipeline:
                first_agent = pipeline[0]
                # Find domain for this agent
                for domain, agents in ROUTING_RULES.items():
                    if first_agent in agents:
                        keywords = extract_keywords(message)
                        return domain, f"SOP:{sop_name}", keywords
            break

    # ========================================================================
    # PRIORITY KEYWORDS - These override semantic matching
    # Critical for ensuring tikz, bibtex, mermaid go to correct direct agent
    # ========================================================================
    priority_keywords = {
        "tikz": ("Visualization", "TikZPlotter"),
        "latex diagram": ("Visualization", "TikZPlotter"),
        "mermaid": ("Visualization", "HybridVisualizer"),
        "flowchart": ("Visualization", "HybridVisualizer"),
        "threat model": ("Security & Risk", "ThreatModeler"),
        "systematic review": ("Research & Discovery", "SLRProtocolDroid"),
        "prisma": ("Research & Discovery", "SLRProtocolDroid"),
        "bibtex": ("Writing & Synthesis", "BibTeXOptimizer"),
    }

    # Check priority keywords
    for kw, (domain, agent) in priority_keywords.items():
        if kw in message_lower:
            keywords = extract_keywords(message)
            return domain, agent, keywords

    # ========================================================================
    # SEMANTIC SOP MATCHING - Intent + Concept scoring
    # Catches cases like "check novelty" → SOP_NOVELTY_DEFENSE
    # even though triggers only have "novelty defense"
    # ========================================================================
    semantic_result = score_sop_by_intent(message_lower)
    if semantic_result:
        sop_name, score = semantic_result
        sop_info = SOP_REGISTRY.get(sop_name, {})
        pipeline = sop_info.get("pipeline", [])
        if pipeline:
            # Find domain for the first agent in pipeline
            for domain, agents in ROUTING_RULES.items():
                if pipeline[0] in agents:
                    keywords = extract_keywords(message)
                    return domain, f"SOP:{sop_name}", keywords

    # Normal keyword extraction
    keywords = extract_keywords(message)

    if not keywords:
        # Default to MasterOrchestrator if no match
        return "Strategy & Operations", "MasterOrchestrator", []

    # Use the first (longest) matched keyword to determine routing
    best_keyword = keywords[0]
    domain, agent = KEYWORD_TO_AGENT.get(
        best_keyword, ("Strategy & Operations", "MasterOrchestrator")
    )

    return domain, agent, keywords


def get_routing_table() -> Dict[str, Dict[str, List[str]]]:
    """Get the full routing table."""
    return ROUTING_RULES
