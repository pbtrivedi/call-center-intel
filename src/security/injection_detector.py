from __future__ import annotations

import re
from typing import NamedTuple

from src.models.schemas import InjectionCheckResult

# ---------------------------------------------------------------------------
# Pattern bank — 22 patterns across 7 attack categories
# ---------------------------------------------------------------------------

class _Pattern(NamedTuple):
    name: str
    regex: re.Pattern[str]
    risk_level: str  # "low" | "medium" | "high" | "critical"


_PATTERNS: list[_Pattern] = [
    # --- Ignore / override instructions ---
    _Pattern("IGNORE_PREVIOUS", re.compile(
        r"ignore\s+(all\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("DISREGARD_INSTRUCTIONS", re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|constraints?)",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("FORGET_INSTRUCTIONS", re.compile(
        r"forget\s+(everything|all|your\s+(previous|prior|earlier|original))\s*(instructions?|rules?|context|prompts?)?",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("NEW_INSTRUCTIONS", re.compile(
        r"(your\s+new\s+instructions?\s+(are|is)|from\s+now\s+on\s+(you\s+)?(must|will|should|are\s+to))",
        re.IGNORECASE,
    ), "high"),

    # --- Role / identity switching ---
    _Pattern("ACT_AS", re.compile(
        r"\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|evil|malicious|hacker|DAN)\b",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("PRETEND_TO_BE", re.compile(
        r"\bpretend\s+(to\s+be|you\s+are)\s+(an?\s+)?(AI|assistant|system|model)\s+(without|with\s+no)\s+(restrictions?|filters?|limits?|guidelines?)",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("YOU_ARE_NOW", re.compile(
        r"you\s+are\s+now\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|DAN|evil\s+AI|hacker)",
        re.IGNORECASE,
    ), "critical"),

    # --- DAN / jailbreak modes ---
    _Pattern("DAN_MODE", re.compile(
        r"\bDAN\s*(mode|prompt|jailbreak)?\b|\bdo\s+anything\s+now\b",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("JAILBREAK", re.compile(
        r"\bjailbreak\b|\buncensored\s+mode\b|\bgrandma\s+(exploit|trick|jailbreak)\b",
        re.IGNORECASE,
    ), "high"),
    _Pattern("DEVELOPER_MODE", re.compile(
        r"\bdeveloper\s+mode\b|\benable\s+(unrestricted|jailbreak|god)\s+mode\b",
        re.IGNORECASE,
    ), "high"),

    # --- System / prompt tag injection ---
    _Pattern("SYSTEM_TAG", re.compile(
        r"<\s*(system|sys|prompt|instruction|INST|SYS)\s*>",
        re.IGNORECASE,
    ), "high"),
    _Pattern("CLOSING_TAG_INJECTION", re.compile(
        r"</\s*(system|human|assistant|context|instruction)\s*>",
        re.IGNORECASE,
    ), "high"),
    _Pattern("PROMPT_DELIMITER", re.compile(
        r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<<SYS>>|<</SYS>>)",
        re.IGNORECASE,
    ), "high"),

    # --- Prompt / system leakage requests ---
    _Pattern("REVEAL_PROMPT", re.compile(
        r"(reveal|show|print|output|repeat|tell\s+me)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions?|context|rules?|guidelines?)",
        re.IGNORECASE,
    ), "medium"),
    _Pattern("WHAT_IS_YOUR_PROMPT", re.compile(
        r"what\s+(is|are|were)\s+your\s+(original\s+)?(instructions?|system\s+prompt|rules?|guidelines?)",
        re.IGNORECASE,
    ), "medium"),
    _Pattern("INITIAL_PROMPT", re.compile(
        r"(output|print|show)\s+(the\s+)?(initial|original|first|starting)\s+(prompt|instruction|message)",
        re.IGNORECASE,
    ), "medium"),

    # --- Conversation / context injection ---
    _Pattern("INJECT_HUMAN", re.compile(
        r"\n\s*(Human|User|Customer|Assistant|AI)\s*:\s*",
        re.IGNORECASE,
    ), "high"),
    _Pattern("INJECT_SEPARATOR", re.compile(
        r"(-{4,}|={4,}|\*{4,})\s*(end\s+of\s+)?(system|prompt|context|instructions?)\s*(-{4,}|={4,}|\*{4,})?",
        re.IGNORECASE,
    ), "medium"),

    # --- Capability / restriction bypass ---
    _Pattern("OVERRIDE_SAFETY", re.compile(
        r"(disable|bypass|override|remove|turn\s+off)\s+(your\s+)?(safety|content|ethical|moral|output)\s+(filters?|guidelines?|restrictions?|checks?|policies?)",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("WITHOUT_RESTRICTIONS", re.compile(
        r"(respond|answer|reply|act)\s+without\s+(any\s+)?(restrictions?|filters?|limitations?|constraints?|guidelines?)",
        re.IGNORECASE,
    ), "critical"),
    _Pattern("HYPOTHETICALLY", re.compile(
        r"hypothetically\s+(speaking\s*,?\s*)?(if\s+you\s+(had\s+no|were\s+without|could\s+ignore)\s+(rules?|restrictions?|filters?|guidelines?))",
        re.IGNORECASE,
    ), "medium"),

    # --- Token / encoding tricks ---
    _Pattern("BASE64_INJECTION", re.compile(
        r"(decode|base64|interpret)\s+(this|the\s+following)\s*\w*\s*(as\s+)?(instructions?|command|prompt)",
        re.IGNORECASE,
    ), "high"),
    _Pattern("TRANSLATION_TRICK", re.compile(
        r"translate\s+(and\s+)?(then\s+)?(execute|run|follow|apply)\s+(the\s+|these\s+)?(instructions?|commands?|prompt)",
        re.IGNORECASE,
    ), "medium"),
]

# Risk level ordering for aggregation
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_RISK_LABELS = {v: k for k, v in _RISK_ORDER.items()}


def detect_injection(text: str) -> InjectionCheckResult:
    """Scan text for prompt injection patterns. Returns InjectionCheckResult."""
    matched_names: list[str] = []
    max_risk = -1
    first_match: str | None = None

    for pattern in _PATTERNS:
        m = pattern.regex.search(text)
        if m:
            matched_names.append(pattern.name)
            level = _RISK_ORDER[pattern.risk_level]
            if level > max_risk:
                max_risk = level
                first_match = m.group(0)

    if not matched_names:
        return InjectionCheckResult(matched=False)

    return InjectionCheckResult(
        matched=True,
        matched_patterns=matched_names,
        risk_level=_RISK_LABELS[max_risk],
        flagged_text=first_match,
    )
