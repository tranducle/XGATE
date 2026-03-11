"""
Prompt Optimizer Module for LangGraph Research System

Rule-based prompt optimization that implements the PromptOptimizer.json protocol:
  Step 0: Language detection & Vietnamese translation
  Step 1: Intent classification
  Step 2: Domain hint extraction
  Step 3: Prompt restructuring
  Step 4: Lessons injection (from self-improvement mechanism)

No LLM call needed — the heavy strategies (CoT, CoVe, Meta-CoT) remain in
the PromptOptimizer's system_prompt for when Antigravity adopts it.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .router import (
    _normalize_vietnamese,
    _translate_vietnamese,
    VI_TO_EN,
    KEYWORD_TO_AGENT,
    ROUTING_RULES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STEP 0: LANGUAGE DETECTION
# ============================================================================

# Vietnamese-specific Unicode ranges & common diacritical patterns
_VI_DIACRITICS = re.compile(
    r"[\u00C0-\u00FF\u0100-\u024F\u1E00-\u1EFF\u01A0-\u01B0\u0300-\u036F]"
)

# Common Vietnamese function words (high signal, unlikely in English)
_VI_FUNCTION_WORDS = {
    "cua",
    "va",
    "trong",
    "cho",
    "voi",
    "la",
    "den",
    "tu",
    "mot",
    "nhung",
    "nay",
    "do",
    "co",
    "khong",
    "duoc",
    "se",
    "da",
    "dang",
    "rat",
    "nhu",
    "thi",
    "ma",
    "de",
    "bai",
    "ve",
    "cac",
    "cung",
    "hay",
    "neu",
    "gi",
}


def detect_language(text: str) -> str:
    """Detect if text is Vietnamese or English.

    Returns:
        'vi' if Vietnamese detected, 'en' otherwise.
    """
    # Check for Vietnamese diacritical marks in original text
    if _VI_DIACRITICS.search(text):
        return "vi"

    # Check normalized text against Vietnamese function words
    normalized = _normalize_vietnamese(text)
    words = set(normalized.split())

    vi_word_count = len(words & _VI_FUNCTION_WORDS)
    if vi_word_count >= 2:
        return "vi"

    # Check if any VI_TO_EN keys match in normalized text
    for vi_key in VI_TO_EN:
        if vi_key in normalized:
            return "vi"

    return "en"


# ============================================================================
# STEP 1: INTENT CLASSIFICATION
# ============================================================================

# Intent patterns: (intent_name, keywords_list)
_INTENT_PATTERNS = [
    (
        "fact_retrieval",
        [
            "find",
            "search",
            "look up",
            "what is",
            "define",
            "list",
            "who",
            "when",
            "where",
            "how many",
        ],
    ),
    (
        "creative_generation",
        [
            "write",
            "draft",
            "create",
            "generate",
            "compose",
            "design",
            "brainstorm",
            "ideate",
            "propose",
            "suggest",
        ],
    ),
    (
        "logical_reasoning",
        [
            "analyze",
            "compare",
            "evaluate",
            "prove",
            "verify",
            "audit",
            "check",
            "validate",
            "assess",
            "review",
            "critique",
            "experiment",
            "anova",
            "t-test",
            "regression",
            "statistics",
        ],
    ),
    (
        "transformation",
        [
            "convert",
            "transform",
            "translate",
            "format",
            "refactor",
            "clean",
            "preprocess",
            "optimize",
            "fix",
            "debug",
        ],
    ),
    (
        "multi_step_research",
        [
            "systematic review",
            "research plan",
            "methodology",
            "case study",
            "literature review",
            "gap analysis",
            "comprehensive",
            "full review",
            "deep",
        ],
    ),
]


def classify_intent(message: str) -> str:
    """Classify the user's intent based on keyword matching.

    Args:
        message: Normalized/translated message (English).

    Returns:
        Intent string, one of: fact_retrieval, creative_generation,
        logical_reasoning, transformation, multi_step_research, general.
    """
    message_lower = message.lower()
    scores: Dict[str, int] = {}

    for intent, keywords in _INTENT_PATTERNS:
        score = sum(1 for kw in keywords if kw in message_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "general"

    return max(scores, key=lambda k: scores[k])


# ============================================================================
# STEP 2: DOMAIN HINT EXTRACTION
# ============================================================================


def extract_domain_hints(message: str) -> List[str]:
    """Extract domain hints from the message for routing assistance.

    Args:
        message: Normalized/translated message (English).

    Returns:
        List of domain names that likely match.
    """
    message_lower = message.lower()
    domain_scores: Dict[str, int] = {}

    for domain, agents in ROUTING_RULES.items():
        score = 0
        for agent, keywords in agents.items():
            for kw in keywords:
                if kw in message_lower:
                    score += 1
        if score > 0:
            domain_scores[domain] = score

    # Return domains sorted by score (highest first)
    return sorted(domain_scores, key=lambda k: domain_scores[k], reverse=True)


# ============================================================================
# STEP 3: PROMPT RESTRUCTURING
# ============================================================================


def _restructure_prompt(
    translated_message: str,
    intent: str,
    domain_hints: List[str],
    original_language: str,
    original_message: str,
) -> str:
    """Restructure the prompt with intent and context metadata.

    Produces a clean, routing-optimized query string enriched with
    classification metadata for downstream consumers.

    Args:
        translated_message: The English (translated if needed) message.
        intent: Classified intent type.
        domain_hints: Detected domain hints.
        original_language: 'vi' or 'en'.
        original_message: The raw user input.

    Returns:
        Optimized message string.
    """
    # Start with the translated/cleaned message
    optimized = translated_message.strip()

    # For Vietnamese input, prepend translation note so downstream knows
    if original_language == "vi":
        optimized = f"[Translated from Vietnamese] {optimized}"

    return optimized


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def optimize_prompt(raw_message: str) -> Dict:
    """Optimize a user prompt through the 4-step protocol.

    Steps:
        0. Language detection & Vietnamese translation
        1. Intent classification
        2. Domain hint extraction
        3. Prompt restructuring

    Args:
        raw_message: The raw user input string.

    Returns:
        Dict with keys:
            optimized_message: str   — cleaned/translated query for routing
            original_language: str   — 'vi' or 'en'
            original_message: str    — preserved raw input
            intent: str              — classified task type
            detected_domain_hints: list — domain names found
    """
    if not raw_message or not raw_message.strip():
        return {
            "optimized_message": raw_message or "",
            "original_language": "en",
            "original_message": raw_message or "",
            "intent": "general",
            "detected_domain_hints": [],
        }

    # Step 0: Language detection & translation
    original_language = detect_language(raw_message)

    if original_language == "vi":
        translated = _translate_vietnamese(raw_message)
    else:
        translated = raw_message.strip()

    # Step 1: Intent classification (on translated text)
    intent = classify_intent(translated)

    # Step 2: Domain hint extraction (on translated text)
    domain_hints = extract_domain_hints(translated)

    # Step 3: Restructure prompt
    optimized_message = _restructure_prompt(
        translated_message=translated,
        intent=intent,
        domain_hints=domain_hints,
        original_language=original_language,
        original_message=raw_message,
    )

    return {
        "optimized_message": optimized_message,
        "original_language": original_language,
        "original_message": raw_message,
        "intent": intent,
        "detected_domain_hints": domain_hints,
    }


# ============================================================================
# STEP 4: LESSONS INJECTION
# ============================================================================


def inject_lessons_into_prompt(
    optimized_message: str,
    state_dict: Dict[str, Any],
) -> str:
    """Append relevant lessons and feedback context to the optimized prompt.

    Called by the prompt_optimization_node in graph.py AFTER optimize_prompt()
    finishes Steps 0-3.  This is Step 4.

    Args:
        optimized_message: Output from Steps 0-3.
        state_dict: Current ResearchState dict (needs is_feedback,
                    feedback_context, lessons_applied).

    Returns:
        The optimized_message with lessons/feedback blocks appended.
    """
    try:
        from .feedback_handler import (
            format_feedback_for_prompt,
            format_lessons_for_prompt,
            search_lessons,
        )
    except ImportError:
        logger.warning("feedback_handler not available — skipping lessons injection")
        return optimized_message

    chunks: List[str] = [optimized_message]

    query_keywords = optimized_message[:200]
    relevant_lessons = search_lessons(query=query_keywords, limit=5)
    if relevant_lessons:
        state_dict.setdefault("lessons_applied", []).extend(relevant_lessons)
        chunks.append(format_lessons_for_prompt(relevant_lessons))

    is_feedback = state_dict.get("is_feedback", False)
    feedback_ctx = state_dict.get("feedback_context")
    if is_feedback and feedback_ctx:
        chunks.append(format_feedback_for_prompt(feedback_ctx))

    if len(chunks) == 1:
        return optimized_message

    return "\n\n".join(chunks)
