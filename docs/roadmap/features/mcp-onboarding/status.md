```toml
name = "MCP Server AI-Assistant Onboarding"
status = "proposed"
progress_pct = 0
description = """
Makes connecting an AI assistant to a Posetrak session (for tracking-run diagnosis) a small \
number of clicks from posetrak-ui instead of hand-editing .mcp.json with an absolute db path, \
and lets the server follow whichever session is currently open instead of needing a restart \
every time that changes.
"""
categories = ["mcp", "ux", "release"]
target_release = "TBD"
last_updated = 2026-08-23
```

# MCP Onboarding — Implementation Status

See [mcp-onboarding-design.md](mcp-onboarding-design.md) for the full
motivating problem, current-state trace, target vision, and sketch of
changes.

## Current state

**2026-08-23: proposal only.** Nothing implemented — `server.py`'s
`--db-path` is still a required, startup-only argument with no live
switching, and there's no GUI action to generate an MCP client config.
Written up after Harri asked why an AI assistant working in this repo
rarely uses the MCP server, and what should ease setup for users, given
the intended use case is an assistant helping diagnose tracking (or
other pipeline) problems.

## Known issues / open questions

See mcp-onboarding-design.md's "Open questions" section: multiple
concurrent sessions/windows, the security shape of a shared
active-session state file, which MCP clients to prioritize, and whether
a packaged release should just bundle a pre-filled config template
instead of generating one at runtime.
