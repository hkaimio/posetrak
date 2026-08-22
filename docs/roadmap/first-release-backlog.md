# First-release backlog

Captured 2026-07-15 (Harri), item 7 added 2026-08-03, items 8-9 added
2026-08-23. One item expected to meaningfully improve tracking quality, the
rest tech debt / UX. Not sequenced here beyond the note on each; see
individual design docs (linked below) for anything already scoped in more
detail.

## 1. Cross-person relative observations

The one item with high expected impact on tracking quality — assisted
movements, handshakes, two-person contact (ukemi throws are exactly this
case). Design already exists: Phase 5 of
`docs/roadmap/features/error-improvements/implementation-plan.md`. Its
prerequisites (Phases 1, 3, 4 — split pose/calib noise, within-person
`PAIR_DIFF` relative measurements, spatial cross-pairs) are already
implemented and tested; Phase 5 itself (cross-*person* `ANCHORED_RELATIVE`
mode, the `MultiPersonTracker` Gauss-Seidel orchestrator, contact-window
detection) is not yet built. Estimated 3-5 days per the doc. Currently
being planned (2026-07-15).

## 2. Update the keypoint-editing user guide

`docs/roadmap/features/keypoint-editing/` predates hand-detection-refinement
Idea 3. Needs updating for: the "Auto-redetect hands" toggle and what
auto-detect vs. keep-existing-state means in practice, the new
`STATUS_ORANGE` timeline color and what it signals, the "Revert hand
redetection" context-menu action, and how interpolation now interacts with
automated hand redetection (interpolate wrist/elbow → unselected fingers
get redetected, not geometrically interpolated).

## 3. Verify hand-detection-refinement's actual tracking impact, tune if needed

Idea 3 is implemented and confirmed *working* (no crashes, redetection
fires and writes sensible data), but its effect on tracking *quality* at
scale hasn't been measured yet — the hand-detection-refinement design doc's
own validation criteria (before/after garbage-detection comparison at trial
scale, hand-editing completion-time comparison) are still open. Also: the
700ms debounce window is an untuned guess, worth adjusting against how it
actually feels over extended real use.

## 4. CLI tools and MCP server: verify compatibility with new features, extend as needed

The MCP diagnostic server (`python/app/mcp/`) and CLI tooling predate the
Phase 2 multi-source `pose_observations` schema and Idea 3's `.refined`
sources / auto-detect toggle. Check whether `describe_config`-style tools
and any observation-quality tooling account for the new `source` values
correctly, and whether the upcoming Phase 5 multi-person config
(`cross_person_max_world_mm`, iteration count, contact windows) needs its
own MCP/CLI surfacing (the error-improvements plan already calls this out
for `describe_config`).

## 5. Surface "crisis debugging" patterns directly in the app

`docs/roadmap/features/tracking-crisis-debugging-log.md` documents several
real failure patterns diagnosed by hand, ad hoc, over multiple sessions
(swapped shoulder/wrist keypoints, near-origin/garbage edited keypoints,
covariance condition-number blowups, frequent PSD-eigensolver repairs,
edits that get silently gate-rejected). Each of these required a one-off
script or manual cross-referencing to find. Worth adding: timeline
warnings that flag a time range as likely problematic, with a short,
specific explanation of the probable root cause (not just "something's
wrong here") — turning this session's diagnostic experience into a
standing feature rather than one-off investigations repeated per trial.

## 6. Fix segmentation reusability properly (schema change)

Draft design already written:
`docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md`.
Today, a Cutie segmentation is permanently tied to the one detection run
it was created for (`seg_quality_runs.detection_run_id`), so it can't be
reused as the bbox source for a second detection run (e.g. a redo with a
different pose model, or adding a hand-refinement pass to an older
segmentation). The draft resolves this with a time-range-scoped
segmentation (own `time_start_s`/`time_end_s`, reusable by any trial it
fully contains) instead of a capture- or detection-run-level link — a
non-additive schema migration (PK rebuild), plus either a new UI dispatch
point or a real convergence of the YOLO and segmentation-driven pose
pipelines. Explicitly postponed until after hand-detection-refinement;
now back on the table.

## 7. Finish Windows build DX improvements

Native Windows (MSVC) setup is now fully working and documented
(`CONTRIBUTING.md`'s "Windows (native, MSVC)" section, `setup-windows.ps1`), and
a `.wraplock`/wrap-cache mechanism already exists under `subprojects/`. Two
originally-planned items from that effort are still outstanding:

- **Pre-seed the meson wrap cache in the repo.** `meson setup` currently still
  downloads every wrap dependency (Catch2, fmt, nlohmann_json, etc.) fresh on
  first setup, which is the biggest remaining chunk of "clone → first build"
  time on a new machine. Commit the downloaded wrap archives (or point Meson's
  cache dir at a repo-tracked location) so `meson setup` hits a local cache
  instead of the network.
- **Add a `.vsconfig` file at the repo root** listing
  `Microsoft.VisualStudio.Workload.NativeDesktop` so the Visual Studio installer
  can install exactly the right workload in one click, instead of a new
  contributor having to guess which workload includes the C++ toolchain.

(The originally-planned "trim Boost to a curated header subset committed to the
repo" item was superseded by the current approach — fetching full Pinocchio +
Boost via a dedicated conda environment — which turned out simpler to keep in
sync with the Linux/WSL Pinocchio version than a hand-curated subset.)

## 8. Package a real release artifact

No release workflow exists today — installing Posetrak means the full
developer setup (`docs/setup.md`): a C++ toolchain, `uv`, manual `uv sync`.
Design proposal: `docs/roadmap/features/packaging/packaging-design.md` — a
thin bootstrapper (bundled `uv` binary + pre-built C++ tracker + a pinned
lockfile snapshot) rather than a fully offline fat bundle, packaged as a
Windows installer (Inno Setup) and a Linux AppImage, built via a new GitHub
Actions release workflow. Near-term plan:
`docs/roadmap/features/packaging/installer-prototype-plan.md` — a narrow
Windows/CPU-only prototype, validated manually before any CI automation,
then handed to a small group of real testers before deciding what's next.
Ships unsigned; code signing
(`docs/roadmap/features/packaging/code-signing-plan.md`) is deliberately
deferred until there's real evidence of interest in Posetrak beyond
today's use — a hobby project's budget doesn't justify it otherwise.
Proposal only as of 2026-08-23, nothing implemented. Independent, worth
doing regardless: split the base `onnxruntime-gpu` dependency into a plain
`onnxruntime` (CPU) base plus an opt-in GPU variant, matching the
`segmentation` group's existing optional-heavy-dependency pattern.

## 9. Ease AI-assistant (MCP) setup for end users

The MCP diagnostic server (`python/app/mcp/`, see item 4 above for its own
feature-compatibility gaps) is aimed at exactly the kind of problem a
packaged release's users will hit — "why does this tracking run look
wrong" — but connecting to it today means hand-editing `.mcp.json` with an
absolute database path, and switching which session/capture you're asking
about means restarting the server (`--db-path` is a required, startup-only
argument). Design proposal:
`docs/roadmap/features/mcp-onboarding/mcp-onboarding-design.md` — a
"Connect AI assistant…" action in `posetrak-ui` that generates the client
config automatically, and letting the server follow whichever session is
currently open instead of needing a restart per switch. Proposal only as
of 2026-08-23, nothing implemented.
