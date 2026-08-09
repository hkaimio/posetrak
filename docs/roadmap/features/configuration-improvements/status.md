```toml
name = "Tracker Configuration Improvements"
status = "complete"
description = """
Reworks the tracking-run configuration workflow: named/default configs resolved per session, \
capture, and trial (instead of re-entering settings every run), and an organized, tabbed \
configuration dialog with tooltips replacing the old single monolithic parameter list.
"""
categories = ["ui", "data-model"]
target_release = "TBD"
last_updated = 2026-08-06
```

# Tracker Configuration Improvements — Implementation Status

See:
- [confg-improvement-brief.md](confg-improvement-brief.md) — original problem statement
- [config-improvements-design.md](config-improvements-design.md) — full design and phase-by-phase
  implementation log (phases 0-6)

## Current state

Per the design doc's own status header: **implementation started 2026-07-24, phases 0-6 are all
done — the full proposal is implemented**, including three rounds of live-review fixes found
along the way. Highlights from the design doc's phase log:

- **Phase 0**: fixed a real, silent data-loss bug in `manage_config.py` (`edit_config()`/
  `create_config_from_toml()` used a hardcoded column list that had drifted out of sync with
  the schema — rewritten to derive columns from `PRAGMA table_info` at call time).
- **Phase 1**: schema for named configs and per-session/capture/trial default-config resolution
  (`tracker_configs.is_named`, `captures.default_tracker_config_id`,
  `trials.default_tracker_config_id`). Also fixed an adjacent bug: the registry's own migration
  chain had stopped tracking `tracker_configs` columns after schema v6, so a registry DB
  predating that point was silently missing every tuning column added since — fixed generically
  via a schema-diffing migration, not just for the one column this phase needed.
- **Phases 2-6**: tabbed configuration dialog, default-config UI wiring, and the CLI-side model
  — see the design doc for the full breakdown.

## Known issues

- `copy_config_to_session()` auto-invocation gap: a confirmed real issue (referenced configs/
  skeletons aren't always copied into a session DB for self-containment), explicitly tracked as
  a TODO but deliberately left out of this proposal's scope — no phase here depends on it.
- Session-level default config tier (a third tier below capture/trial) was deliberately
  deferred; revisit only if a `SessionPanel` gets built for other reasons.

See the design doc directly for the full list of fixes made along the way (some, like the
registry migration gap, were found opportunistically while implementing this feature rather
than pre-planned).
