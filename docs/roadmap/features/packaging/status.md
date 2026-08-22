```toml
name = "Release Packaging (Windows/Linux Installer)"
status = "proposed"
progress_pct = 0
description = """
Produce an installable release artifact (Windows installer, Linux AppImage/tarball) that doesn't \
require a compiler or manual `uv sync` -- a thin bootstrapper (uv binary + pre-built C++ tracker + \
pinned lockfile) rather than a fully offline fat bundle, keeping one dependency-resolution path \
for both a release install and a dev checkout.
"""
categories = ["release", "packaging", "build"]
target_release = "TBD"
last_updated = 2026-08-23
```

# Release Packaging — Implementation Status

See [packaging-design.md](packaging-design.md) for the full motivating
problem, current-state trace, target vision, and recommended approach.
See [code-signing-plan.md](code-signing-plan.md) for the Windows
code-signing sub-plan (certificate options, CI wiring, a
prototype-then-small-group-test phasing).

## Current state

**2026-08-23: proposal only.** Nothing implemented — no installer
scripts, no AppImage recipe, no release CI workflow, no `onnxruntime-gpu`
split, no certificate acquired. Written up in response to Harri asking
how to produce a real release artifact, ahead of the first Posetrak
release.

## Known issues / open questions

See packaging-design.md's "Open questions" section: code signing,
auto-update, GPU vendor scope, Linux distro coverage, whether a
fully-offline variant is ever needed, and where the pinned lockfile
snapshot for a given release should live.
