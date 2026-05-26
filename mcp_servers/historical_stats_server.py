"""
MCP Historical Stats Server.

Exposes average QA dimension scores and recent compliance flags by call type.
Phase 4: backed by static fixtures.
Phase 6: will query the live SQLite repository.

Run standalone: python mcp_servers/historical_stats_server.py
"""
from __future__ import annotations

# Static benchmark data — Phase 6 replaces this with live DB queries
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

_RECENT_FLAGS: dict[str, list[dict]] = {
    "credit_dispute": [
        {"description": "Dispute rights not read before account access", "severity": "high", "frequency": 12},
        {"description": "Failed to confirm dispute timeframe", "severity": "medium", "frequency": 7},
    ],
    "account_inquiry": [
        {"description": "Full account number read back to caller", "severity": "critical", "frequency": 3},
        {"description": "Identity not verified before data access", "severity": "high", "frequency": 8},
    ],
    "billing_issue": [
        {"description": "Dispute rights not communicated", "severity": "medium", "frequency": 15},
    ],
    "password_reset": [
        {"description": "Account existence confirmed to unverified caller", "severity": "high", "frequency": 5},
    ],
}


def get_agent_benchmarks(call_type: str) -> str:
    """Return formatted historical average scores for the given call type."""
    benchmarks = _BENCHMARKS.get(call_type)
    if not benchmarks:
        return f"No benchmark data available for call type: {call_type!r}"

    lines = [f"Historical average QA scores for '{call_type}':"]
    for dim, avg in benchmarks.items():
        lines.append(f"  {dim}: {avg:.1f}/5.0")
    return "\n".join(lines)


def get_recent_flags(call_type: str) -> str:
    """Return the most common recent compliance flags for the given call type."""
    flags = _RECENT_FLAGS.get(call_type)
    if not flags:
        return f"No recent compliance flag data for call type: {call_type!r}"

    lines = [f"Most common compliance flags for '{call_type}':"]
    for flag in flags:
        lines.append(
            f"  [{flag['severity'].upper()}] {flag['description']} "
            f"(seen {flag['frequency']}x recently)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for ct in sorted(_BENCHMARKS):
        print(f"\n=== {ct} ===")
        print(get_agent_benchmarks(ct))
        print(get_recent_flags(ct))
