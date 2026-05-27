"""
MCP Compliance Rules Server.

Exposes call-type-specific compliance rules from config/compliance_rules.yaml.
Run standalone: python mcp_servers/compliance_rules_server.py

The QA scoring agent fetches rules via src/services/mcp_client.get_compliance_rules()
before every LLM call, injecting them as prompt context.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_RULES_PATH = Path(__file__).parent.parent / "config" / "compliance_rules.yaml"


def _load_rules() -> dict:
    if not _RULES_PATH.exists():
        return {}
    with _RULES_PATH.open() as f:
        return yaml.safe_load(f) or {}


def get_compliance_rules(call_type: str) -> str:
    """Return formatted compliance rules for the given call type."""
    rulebook = _load_rules()
    rules = rulebook.get(call_type)
    if not rules:
        return f"No compliance rules found for call type: {call_type!r}"

    lines = [f"Compliance rules for call type: {call_type}"]
    disclosures = rules.get("required_disclosures", [])
    if disclosures:
        lines.append("Required disclosures:")
        lines.extend(f"  - {d}" for d in disclosures)
    steps = rules.get("verification_steps", [])
    if steps:
        lines.append("Verification steps:")
        lines.extend(f"  - {s}" for s in steps)
    return "\n".join(lines)


def list_call_types() -> list[str]:
    """Return all call types that have rules defined."""
    return sorted(_load_rules().keys())


if __name__ == "__main__":
    # Quick smoke test
    print("Available call types:", list_call_types())
    for ct in list_call_types():
        print(f"\n--- {ct} ---")
        print(get_compliance_rules(ct))
