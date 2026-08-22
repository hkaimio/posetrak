# Installer prototype — implementation plan

> **Status (2026-08-23)**: Proposal only, nothing implemented. Written
> up after Harri asked for a general plan to prototype the thin-
> bootstrapper installer approach from
> [packaging-design.md](packaging-design.md), with a small user-group
> test before committing further. Code signing is explicitly out of
> scope here — deferred per [code-signing-plan.md](code-signing-plan.md)
> until there's real evidence of interest in Posetrak beyond today's
> use; this prototype ships unsigned.

## What this prototype needs to prove

Not "build the release pipeline" yet — validate the *idea* cheaply
before investing in automation:

1. The thin-bootstrap shape actually works end to end on a genuinely
   clean machine (no Python, no compiler, no `uv`, no dev tools of any
   kind) — this is the load-bearing assumption the whole
   packaging-design.md recommendation rests on, and it's untested.
2. The resulting first-run experience (install → first launch → `uv
   sync` fetching dependencies → app opens) is acceptable to someone
   who isn't the developer and doesn't already know what's supposed to
   happen.
3. Real machines surface real problems a dev machine won't — antivirus
   flagging an unsigned `.exe`, corporate/locked-down machines blocking
   script execution, path or permission issues, GPU-detection edge
   cases. Cheaper to find these with five informed testers than with a
   public release.

## Scope for this prototype (deliberately narrow)

- **Windows only, CPU-only variant.** Both packaging-design.md's target
  platforms (Windows + Linux) and both dependency variants (CPU/GPU)
  multiply the number of things that can go wrong in a first pass. Get
  one path solid before adding axes. Windows first because it's where
  today's setup friction (native MSVC / conda for Pinocchio-Boost) is
  worst and the audience is largest; CPU-only first because it removes
  GPU-detection and driver-version variables entirely, and rtmlib's
  CPU fallback already works today per `docs/setup.md`.
- **No code signing** — ships unsigned; see "Documenting the warning"
  under Phase 2.
- **No CI automation yet** — Phase 1-2 below are done by hand, on a
  local machine. Worth automating (the GitHub Actions release workflow
  from packaging-design.md's sketch) only once the manual process is
  proven to actually work; automating something unvalidated just moves
  the same open questions into YAML.
- **No auto-update** — out of scope per packaging-design.md; the
  version-manifest shape discussed there doesn't need to exist yet for
  a single prototype build.

Linux (AppImage), the GPU variant, CI automation, and auto-update are
all natural fast-follows once this narrow slice is validated — each
reuses the same pinned-lockfile mechanism, just with one more variable
at a time instead of all at once.

## Phase 0 — Pin a version to build against

- Tag a pre-release version (e.g. `v0.1.0-proto1`) so the prototype has
  a fixed, reproducible `uv.lock` snapshot to build from, rather than a
  moving target on `main`. Doesn't require the full versioning-scheme
  decision packaging-design.md leaves open — just something stable
  enough to build one prototype against.
- Build the optimized C++ tracker for Windows from that tag
  (`meson setup optbuild --buildtype=release` per `docs/setup.md`,
  either native MSVC or the MinGW cross-compile path — whichever is
  faster to produce right now; the prototype doesn't need to settle
  which build path the eventual CI pipeline uses).

## Phase 1 — Manual bootstrap validation (no installer yet)

Before wrapping anything in an installer, prove the core mechanism by
hand:

1. Assemble a folder with: the tracker `.exe` (+ its runtime DLLs if
   MinGW-cross-compiled), the `uv` binary, the pinned `pyproject.toml`
   + `uv.lock` from Phase 0, and a minimal launcher (a `.bat`/`.ps1` to
   start, a proper entry point later) that runs `uv sync` then
   `uv run posetrak-ui`.
2. Test this folder on a genuinely clean environment —
   [Windows Sandbox](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/)
   (built into Windows 10/11 Pro, free, disposable, resets on close) is
   a good fit: no separate VM image to maintain, and it starts from a
   guaranteed-clean state every time.
3. Confirm specifically: `uv` can provision its own Python 3.13
   (matching `pyproject.toml`'s `requires-python`) without a system
   Python present at all — this is one of `uv`'s actual selling points
   for this use case and is worth confirming rather than assuming.
   Confirm `uv sync` successfully pulls PySide6/OpenCV/etc. with no
   dev-toolchain dependency, and that `posetrak-ui` actually opens and
   looks functional afterward.
4. Time it. First-launch `uv sync` duration is a real UX number worth
   having before deciding whether it needs a progress indicator in the
   eventual launcher, not just "a few minutes" as a guess.

This phase is deliberately manual and disposable — the goal is a fast,
cheap yes/no on the core assumption, not a polished artifact.

## Phase 2 — Wrap it as an actual installer

Only once Phase 1 works:

1. Write the Inno Setup script wrapping Phase 1's folder contents into
   a proper installer (Start Menu shortcut, uninstaller, the works).
2. Build the `.exe` installer locally (`iscc`).
3. Repeat the Windows Sandbox test, but starting from the installer
   this time — install, launch from the Start Menu shortcut, confirm
   first-run behaves the same as Phase 1's manual folder did.
4. **Documenting the warning**: since this ships unsigned, write the
   short "you'll see a SmartScreen warning, here's why, here's what to
   click" note this needs (in the installer's own first page, and/or
   wherever the download link lives) — this is the concrete,
   low-cost alternative to code signing for a prototype: not hiding
   the warning, explaining it honestly.

## Phase 3 — Small user group test

- Recruit a handful of real external testers — not the developer's own
  machine, ideally varying Windows versions and at least one machine
  with typical consumer antivirus software active (a common source of
  unsigned-`.exe` false positives worth surfacing now rather than at a
  wider release).
- Give them the installer and nothing else — no "here's what should
  happen" walkthrough — to see what actually confuses a first-time
  user versus what only makes sense with prior context.
- Collect structured feedback per tester: did SmartScreen show the
  documented warning or something unexpected; did antivirus flag
  anything; how long did first launch actually take; did
  `posetrak-ui` open and look right; anything that silently failed
  with no visible error.
- Iterate on Phase 2's installer/documentation based on what actually
  came back, rather than guessing what needs fixing.

## Phase 4 — Decide what's next

Based on Phase 3 results, decide (not resolved here):

- Whether the prototype is solid enough to invest in CI automation
  (the GitHub Actions release workflow) so every future release
  doesn't repeat Phases 0-2 by hand.
- Whether to add the Linux AppImage and/or GPU variant next, or gather
  more Windows/CPU feedback first.
- Whether real interest has materialized to justify revisiting
  code-signing-plan.md.

## Open questions (not resolved here)

1. **How many testers, and how to find them** — not specified; "a
   handful" is deliberately vague pending an actual decision on who's
   realistic to ask (existing Posetrak users, a wider call for
   volunteers, etc.).
2. **Windows Sandbox vs. a real second machine** — Sandbox is
   convenient but is still fundamentally a Microsoft-controlled
   virtualized environment; whether it's representative enough of a
   real consumer machine (especially for the antivirus-interaction
   question in Phase 3) isn't validated here.
3. **What "acceptable first-launch time" means** — Phase 1 says to
   measure it, not what threshold would be a problem; that judgment
   call is deferred until there's a real number to react to.
