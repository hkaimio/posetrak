```toml
name = "Release Packaging (Windows/Linux Installer)"
status = "in_progress"
progress_pct = 10
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
See [installer-prototype-plan.md](installer-prototype-plan.md) for the
near-term, Windows/CPU-only prototype-then-small-group-test plan this
is actually starting with. See
[code-signing-plan.md](code-signing-plan.md) for the Windows
code-signing sub-plan — deliberately deferred until there's real
evidence of interest in Posetrak beyond today's use.

## Current state

**2026-08-23: installer-prototype-plan.md's Phase 0-1 (manual bootstrap
folder) built and sanity-checked on the dev machine; not yet run through
Windows Sandbox.**

- Built the optimized C++ tracker (`meson setup --reconfigure optbuild
  --buildtype=release`, `meson compile`) and assembled a bootstrap
  folder (`uv.exe`, `tracker/` with the binary + its 2 runtime DLLs, an
  `app/` snapshot of `pyproject.toml`/`uv.lock`/`README.md`/
  `.python-version`/`python/` extracted via `git archive HEAD` — tracked
  files only, no dev-machine build cruft) plus a `launch.ps1` and a
  `posetrak-bootstrap-test.wsb` Windows Sandbox config, at
  `D:\mocap\posetrak-bootstrap-proto\` (outside the repo, disposable per
  the plan).
- `uv sync --project app` succeeds standalone: resolves and installs 38
  base packages (PySide6, OpenCV, numpy, onnxruntime-gpu, etc.) plus
  `posetrak` itself in editable mode from the snapshot, with no
  dependency on the main repo's own `.venv`. `uv run --project app
  posetrak --help` and an `app.ui.main`/`PySide6.QtWidgets` import both
  work from the synced environment.
- **Concrete size data**: shipped bootstrap payload (`uv.exe` + tracker
  + source snapshot, everything that would go in an installer) is
  **~60MB**. What `uv sync` downloads at first run (the resulting
  `.venv`) is **~1.5GB** — the real, now-measured shape of the "first
  launch needs internet and a few minutes" tradeoff
  packaging-design.md's recommendation accepts.
- Validated the tracker-binary "installed location" mechanism needs no
  new code at all: `launch.ps1` copies `tracker/*` to
  `%USERPROFILE%\.posetrak\`, which
  `posetrak.tracker.runner.default_binary_path()` already checks first.
  Confirmed live: copied the binary there and `default_binary_path()`
  resolved to it.
- **Found and fixed a real bug in the process**: `runner.py`'s
  dev-build fallback path was still `optbuild/cli/...`, stale since the
  C++ source tree moved under `cpp/` (commit `dd6afae`); every existing
  test mocks `default_binary_path()` out entirely so nothing caught it.
  Fixed and regression-tested (commit `7ee7a65`).
- **First real Sandbox finding, before the small-group phase even
  started**: Harri opened the Sandbox and it "does not recognize how to
  run .ps1" — a stock Windows install associates `.ps1` with Notepad on
  double-click rather than running it, and even "Run with PowerShell"
  hits the default execution policy (`Restricted`), which refuses to
  run *any* unsigned script with no override. Added `launch.bat`
  (double-click-safe, calls `powershell.exe -ExecutionPolicy Bypass
  -File launch.ps1` — bypasses only for that one invocation, not a
  system-wide change) as the actual entry point; `launch.ps1` is now an
  implementation detail. `README.txt` updated to match. This is exactly
  the class of thing the plan's Phase 3 (small-group test) exists to
  catch — caught even earlier, during the very first run. Not yet
  re-verified inside the Sandbox itself.
- **Not yet done**: confirming `launch.bat` actually resolves the .ps1
  problem inside the Sandbox; validating `uv`'s from-scratch Python
  provisioning and a truly cache-free download (this machine's own
  sanity check reused an already-installed system Python); the rest of
  the small-group test; CI automation; Linux.

## Known issues / open questions

See packaging-design.md's "Open questions" section: code signing
(deferred), auto-update, GPU vendor scope, Linux distro coverage,
whether a fully-offline variant is ever needed, and where the pinned
lockfile snapshot for a given release should live. Plus, newly surfaced
by the prototype: `uv.exe` itself is ~44MB, larger than
packaging-design.md's "a few MB" estimate — still small next to the
~1.5GB dependency download, but worth correcting in that doc.
