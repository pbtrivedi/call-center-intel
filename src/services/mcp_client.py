from __future__ import annotations

from src.common.logger import get_logger

_logger = get_logger(__name__)


def get_compliance_rules(call_type: str) -> str:
    """
    Fetch compliance rules for the given call type from the MCP Compliance Server.

    Returns a formatted string for injection into the QA scoring prompt.
    Returns empty string if the MCP server is unavailable or the call type is unknown.
    """
    try:
        import yaml
        from pathlib import Path

        rules_path = Path(__file__).parent.parent.parent / "config" / "compliance_rules.yaml"
        if not rules_path.exists():
            return ""

        with rules_path.open() as f:
            rulebook: dict = yaml.safe_load(f) or {}

        call_rules = rulebook.get(call_type)
        if not call_rules:
            return ""

        lines = [f"Call type: {call_type}"]
        disclosures = call_rules.get("required_disclosures", [])
        if disclosures:
            lines.append("Required disclosures:")
            lines.extend(f"  - {d}" for d in disclosures)
        steps = call_rules.get("verification_steps", [])
        if steps:
            lines.append("Verification steps:")
            lines.extend(f"  - {s}" for s in steps)

        return "\n".join(lines)

    except Exception as exc:
        _logger.warning("mcp_client get_compliance_rules call_type=%s error=%s", call_type, exc)
        return ""


def get_agent_benchmarks(call_type: str) -> str:
    """
    Fetch historical agent benchmark scores for the given call type from the MCP Stats Server.

    Returns a formatted string for injection into the QA scoring prompt.
    Returns empty string if the MCP server is unavailable.

    Phase 4: reads from a static YAML fixture.
    Phase 6: will query the SQLite repository via the MCP Stats Server.
    """
    # Static benchmarks — replaced by live DB queries in Phase 6
    _BENCHMARKS: dict[str, dict[str, float]] = {
        "credit_dispute": {
            "professionalism": 3.8,
            "empathy": 3.5,
            "problem_resolution": 3.2,
            "compliance": 3.9,
            "clarity": 3.6,
        },
        "account_inquiry": {
            "professionalism": 4.1,
            "empathy": 3.7,
            "problem_resolution": 3.8,
            "compliance": 4.2,
            "clarity": 3.9,
        },
        "billing_issue": {
            "professionalism": 3.9,
            "empathy": 3.6,
            "problem_resolution": 3.5,
            "compliance": 3.8,
            "clarity": 3.7,
        },
        "password_reset": {
            "professionalism": 4.0,
            "empathy": 3.4,
            "problem_resolution": 4.1,
            "compliance": 4.3,
            "clarity": 4.0,
        },
    }

    benchmarks = _BENCHMARKS.get(call_type)
    if not benchmarks:
        return ""

    lines = [f"Historical avg scores for {call_type}:"]
    for dim, avg in benchmarks.items():
        lines.append(f"  {dim}: {avg:.1f}/5.0")
    return "\n".join(lines)
