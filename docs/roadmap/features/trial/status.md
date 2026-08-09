```toml
name = "Trial Concept Refactoring"
status = "in_progress"
progress_pct = 60
description = """
Aligns the DB schema, CLI, and UI around `trial` (a named time window within a capture) as the \
first-class, self-contained unit of the workflow — session -> capture -> trial -> detection run \
-> tracking run — including trial-scoped segmentation and a reworked trial panel that surfaces \
its detection/tracking runs directly instead of embedding the full detection editor.
"""
categories = ["ui", "data-model", "cli"]
target_release = "TBD"
last_updated = 2026-08-06
```

# Trial Concept Refactoring — Implementation Status

See [design.md](design.md) for the full goal, per-level UI panel design, and gap analysis.

## Current state

**Not independently tracked before this document** — status below is inferred from a source
check against the design doc's own gap list, not from a dedicated implementation log. Treat as
a snapshot, not an exhaustive audit.

| Gap | Design doc's ask | Found in source |
|---|---|---|
| 1 — DB schema | `tracking_runs.trial_id` FK + backfill migration | ✅ Present (`db/session_schema.sql`) |
| 2 — CLI | `trial create`, `trial show` commands (`trial list` already existed) | ✅ Present (`python/posetrak/cli/trial.py`) |
| 3 — UI | Reworked `TrialPanel` (info + segmentation + detection/tracking run lists), breadcrumbs on panels | ✅ `TrialPanel` class and breadcrumb support exist in `content_panels.py` |

The CLI `export`/`import` top-level-vs-under-`trial` question (Gap 2) was explicitly resolved
in the design doc itself: keep them top-level, since they accept multiple anchor types
(`--trial`, `--capture`, `--detection`, `--tracking-run`) and aren't trial-specific — the
original Phase 5 sketch predates the multi-anchor implementation.

## Known issues

- **Not verified against every checklist item** in the design doc's implementation-order table —
  specifically the capture-panel button rename ("Detect pose…" → "New trial…"), the detection
  panel's run-dropdown/segmentation-button removal and label renames ("By detection" → "By
  camera", "Apply" → "Save assignments"), and the "Export BVH…" context-menu item. Presence of
  `TrialPanel` and `trial create`/`show` confirms substantial progress but not full completion.
- **Tracking run panel** is explicitly lower priority / design TBD in the source doc — the
  long-term direction (consolidate with the person/tracking-run page reached via the tree) is
  noted but deferred; current state is usable as-is.
