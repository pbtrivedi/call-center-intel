# Architecture Decision Records

> Log key decisions here as development progresses.

## Template

**Decision:** [Short title]
**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded

**Context:** Why does this decision need to be made?
**Decision:** What was decided?
**Consequences:** What are the trade-offs?

---

## ADR-001: Five-layer src structure
**Date:** 2026-05-22
**Status:** Accepted

**Context:** Capstone requires a clean separation between orchestration, business logic, and infrastructure.
**Decision:** Use `src/{agents,graph,services,database,ui}` as the five required layers, plus `models/`, `security/`, and `config/` as supporting packages.
**Consequences:** Clear boundaries make isolated unit testing possible; each agent is independently callable with typed inputs/outputs.
