# Easing AI-assistant (MCP) setup — design sketch

> **Status (2026-08-23)**: Proposal only, nothing implemented. Written up
> after Harri noted that an AI assistant working in this repo rarely
> reaches for the MCP diagnostic server, and asked what should change to
> make it easy for users to set up — the intended use case being an AI
> assistant helping diagnose tracking (or other pipeline) problems.

## Motivating problem

Two separate things are going on, worth untangling:

1. **Most work in this repo isn't the MCP server's use case.** The
   server's tools (`get_filter_stats`, `get_camera_coverage`,
   `get_observation_gaps`, `get_camera_geometry`, `get_edit_coverage`)
   answer "why does *this tracking run* look wrong" — they need a
   session database with real tracking results in it. Source-level bug
   fixing, CLI/tooling work, and docs don't touch that surface at all,
   so not reaching for it there isn't a setup problem.
2. **When it *would* apply, the setup is genuinely more friction than
   it should be** — this is the real, fixable problem, and it's exactly
   what a packaged release (see the packaging design doc) can't expect
   an end user to do by hand at all.

## Current state, traced concretely

- The server (`python/app/mcp/server.py`) takes `--db-path` as a
  **required, startup-only** argument (`argparse`, stored in a
  module-level global `_db_path`). There is no tool or mechanism to
  point an already-running server at a different database — switching
  which session/capture you're diagnosing means editing `.mcp.json` and
  restarting the MCP client's connection to the server.
- Wiring a project up at all means hand-writing `.mcp.json` at the repo
  root with a literal absolute path to the target `.db` file (the exact
  JSON is in `server.py`'s own docstring and repeated in CLAUDE.md's
  "MCP Diagnostic Server" section) — a manual, per-project,
  per-database-switch step with no tooling support.
- `uv sync --group mcp-server` is a separate, easy-to-forget
  installation step (the MCP server's own dependencies aren't in the
  base install) — reasonable for a dev environment, a real gap for a
  packaged release aimed at less technical users.
- `docs/user-guide/tracking-troubleshooting.md`'s "Diagnosing a run
  (MCP diagnostic server)" section is already a stub pointing at
  CLAUDE.md rather than a real user-facing walkthrough — the natural
  place to write one once there's something simpler to describe than
  "hand-edit this JSON file."
- There's also a `--mcp-allow-write` flag (off by default) enabling
  `run_detection`/`run_tracking` write tools — not part of the
  onboarding friction, but worth remembering when designing any
  GUI-driven config generation below, since a generated config
  shouldn't silently default to write-enabled.

## Target vision

From a session already open in `posetrak-ui`, connecting an AI assistant
to it is a small number of clicks, not a hand-edited JSON file with a
path the user has to go find. Once connected, asking questions about
"the capture/trial I'm currently looking at" doesn't require knowing
that a restart is needed if you switch which one that is.

## Sketch of the changes this implies

Not designed in detail — sizing the work for whoever picks this up:

**1. A "Connect AI assistant…" action in `posetrak-ui`.** Given the
currently-open session's path, generate (or update) the right MCP
client config automatically:
- Detect known client config locations (Claude Desktop's config file,
  a project-local `.mcp.json` for Claude Code) and offer to write to
  whichever is found, or fall back to showing the JSON snippet to copy
  with the path already filled in.
- Should not need `uv sync --group mcp-server` as a separate manual
  step — either the action runs it itself, or the base install stops
  treating the MCP server as a heavyweight optional extra (its own
  dependencies are modest compared to `segmentation`).

**2. Let the server follow the active session instead of a fixed
`--db-path`.** The bigger structural change, and the one that actually
removes "restart when you switch sessions" rather than just making the
initial setup easier. Two directions, not evaluated against each other:
- The GUI writes its currently-open session's path to a small state
  file; the server watches it and reconnects when it changes.
- A `switch_session` tool the assistant itself can call mid-conversation
  (e.g. "now let's look at the other capture") — keeps the server
  process-lifetime simple (still one `--db-path` at startup as a
  default/fallback) but adds a live-switch escape hatch without a GUI
  round-trip.

Either way, `_db_path`'s current module-level-global-set-once shape
needs to become something that can change after `main()` returns.

**3. Write the actual `tracking-troubleshooting.md` walkthrough** once
(1) and/or (2) exist — replacing the current stub with real steps
matching whatever the connect flow ends up being, plus a short "what to
ask it" section (the tool docstrings in `server.py`'s own `FastMCP`
instructions are a reasonable starting point for that framing).

**4. Packaged-release angle** (see packaging-design.md): whatever (1)
produces should work without any dependency on a dev checkout's file
layout — a packaged install won't have `python/app/mcp/server.py` at a
predictable relative path the way a repo clone does, so the generated
config needs to reference wherever the release actually installs the
MCP server entry point (`posetrak-mcp`, already a registered console
script) rather than the `uv run python python/app/mcp/server.py`
invocation the docstring currently shows.

## Open questions (not resolved here)

1. **Multiple concurrent sessions/windows** — if a user has more than
   one `posetrak-ui` window open against different session databases,
   "the active session" for a follow-the-GUI server (sketch item 2)
   isn't well-defined. Needs a decision (last-focused window? one
   server per window, each with its own config? just document it as a
   single-window feature for now?).
2. **Security of a shared state file** — sketch item 2's state-file
   approach means anything that can write to that file can redirect the
   MCP server at an arbitrary database on the local machine. Probably
   fine for a single-user local tool (the server is already local-only
   and read-only by default), but worth a deliberate look rather than
   an accidental default.
3. **Which MCP clients to prioritize wiring config-generation for** —
   Claude Desktop and Claude Code have different config file formats
   and locations; scope for "detect and offer to configure" needs to
   name which clients v1 actually supports rather than trying to cover
   every possible MCP client.
4. **Should the packaged release bundle a default, pre-filled
   `.mcp.json` template** rather than generating one at runtime at all
   — simpler, but only works for the single-session-per-install case a
   template can hardcode.
