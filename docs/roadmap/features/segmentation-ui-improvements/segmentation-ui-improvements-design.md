# Segmentation UI improvements — analysis and proposals

> **Status (2026-08-28): analysis only, nothing built.** Six issues Harri
> raised after the [`segmentation-pose-treatment`](../segmentation-pose-treatment/segmentation-pose-treatment-design.md)
> and [`pose-model-finetuning`](../pose-model-finetuning/pose-model-finetuning-design.md)
> work surfaced real friction in the segmentation UI (`CutieInitPanel`,
> `python/app/pose/cutie_init_panel.py`) while producing training data and
> rerunning experiments. Each issue below was checked against the actual
> code, not answered from general impression — several turn out to be
> cheap orchestration fixes on top of machinery that already exists;
> others are genuinely new capability. A suggested build order is at the
> end.
>
> **Three UX questions resolved (2026-08-28)**: where segmentations sit
> in the tree relative to trials, how to keep that understandable, and
> how split points (Issue 4) interact with Mark Start/End — see the
> updated Issue 1 and Issue 4 sections below.

## Issue 1 — segmentations aren't visible in the UI, can't rename/delete

**Confirmed**: today a `seg_quality_runs` row is visible in exactly two
places, both indirect — `content_panels.py`'s `_seg_sources` (a
per-camera heuristic that just picks whichever run has the most `seg_masks`
rows, not something the user chooses or even sees) and the pose-detection
dialog's bbox-source dropdown. There is no list of segmentations anywhere
comparable to how `SessionTreeWidget` (`python/app/ui/session_tree.py`)
already lists `detection_runs` and `tracking_runs` per capture/trial, with
working rename (`QInputDialog`-based, see `_rename_trial`/`_rename_person`)
and delete (`_confirm_delete`) already implemented for those.

**This is also a real, already-demonstrated pain point**, not a
hypothetical one: earlier work this session had to reconstruct which
`seg_quality_run_id` was "the real one" for a capture by cross-referencing
actual `seg_masks` frame coverage against stale recorded time ranges,
because nothing in the UI could just show "here are this capture's
segmentations, here's what each one covers." A tree-visible list would
have made that a five-second lookup instead of DB archaeology.

**Proposal**: add `ItemKind.SEGMENTATION_RUN` to `session_tree.py`,
following the exact existing `DETECTION_RUN` pattern — `_load_segmentation_runs`
(query `seg_quality_runs` by `shot_id`), `_make_segmentation_run_item`
(label from `created_at`/`quality_source`/a mask count), a
`segmentation_run_selected` signal, and a context menu with Rename and
Delete. Two real gaps to close to make that menu work:

- **No `name` column on `seg_quality_runs`** (checked the schema —
  columns are `id, shot_id, trial_id, time_start_s, time_end_s, created_at,
  quality_source, erosion_px, mask_dir, notes, persons_json`; `notes` is a
  separate free-text field, not a display name). Needs a migration (next
  number after `026_hierarchical_solver_stages.sql`) adding
  `name TEXT`, same shape as the existing rename actions elsewhere.
- **Delete needs a "not built upon" guard**, matching the project's
  existing immutability convention (`finalise_to_db`'s refusal to
  re-finalise a sequence with tracking results or edits already on it —
  see CLAUDE.md's data-model-invariants note). Before deleting a
  `seg_quality_runs` row, check whether any `detection_runs` were created
  from it (`PoseExtractionJob.seg_quality_run_id`, not currently a
  queryable FK but derivable) and refuse/warn rather than silently
  orphaning that downstream data.

**Tree placement — capture-level, not trial-level (resolved 2026-08-28)**.
`seg_quality_runs` has a nullable `trial_id`, the same shape
`detection_runs` has, and `session_tree.py` already nests a detection run
under its trial when that's set, under the capture directly when it
isn't (`_add_capture_children`). That conditional rule isn't right for
segmentation, though: a detection run's *normal* case is being scoped to
one trial; a segmentation's normal case is closer to the opposite — it's
routinely created before trial boundaries exist and legitimately spans
several trials at once (`CutieInitPanel`'s own docstring: "segmentation
is capture-scoped and independent of any specific detection run"). Using
the same conditional-nesting rule would make a `seg_quality_runs` row
jump between "under Trial A" and "at the capture level" as the user
extends its range — confusing in a different way than the problem being
solved here. **List segmentations at the capture level, always** — never
conditionally nested under a trial.

To keep that understandable without structural nesting: (1) show the
segmentation's own covered time range in its label, the same way trial
items already do (`name += f"  ({start:.1f}s – {end:.1f}s)"`), and (2)
compute (don't store) which trial(s) it overlaps and surface that as a
tooltip or trailing annotation, e.g. `"covers: Heitot, Cooldown
(partial)"`. Distinguishing a segmentation item from a detection-run item
at the same capture level is then just the existing label-prefix
convention (`"Segmentation  6.5s–67.7s  …"` vs. `"Detection [model]  …"`)
— nothing else in this tree uses icons, no reason to start here.

## Issue 2 — every panel open starts a brand-new segmentation from zero

**Confirmed, and the mechanism is subtler than "always fresh"**: within
one open `CutieInitPanel` session, `_seg_init_run_id` is set once (by the
first `_ensure_seg_run()` call) and correctly reused for every subsequent
mark/track action in that session — many sub-ranges and camera switches
in one sitting *do* accumulate into the same `seg_quality_runs` row. The
gap is specifically **across sessions**: `_seg_init_run_id` always starts
`None` in `__init__` (`cutie_init_panel.py:229`), with no constructor
parameter or picker to resume an existing run's ID — so closing and
reopening the panel (or restarting the app) always creates a new row via
`_ensure_seg_run()`, even though `_load_run()` already finds the most
recent existing run (`self._seg_run_id`) and uses it as a **read-only
fallback** for displaying prior masks (`_load_stored_mask`) — it's just
never promoted to the *write* target.

This exact mechanism is what produced the confusing multi-run situation
this session had to untangle by hand (see Issue 1) — a capture
accumulating several `seg_quality_runs` rows over several editing
sessions, each covering a different sub-range, with no record of which
one is "the" segmentation to keep using.

**Proposal**: once Issue 1's tree list exists, selecting an existing
segmentation run and choosing "Open/Continue" opens `CutieInitPanel` with
that run's ID passed in (new constructor parameter,
e.g. `seg_init_run_id: str | None = None`); when given, skip
`_ensure_seg_run()`'s creation path entirely and set
`self._seg_init_run_id` directly, so new tracking jobs extend that run's
actual coverage instead of fragmenting into a new one. A capture-level
"New segmentation…" action (also from the tree, or the panel's own
toolbar) stays available for when starting over is actually wanted.

## Issue 3 — "queue forward/backward" is confusing; seed + start/end is clearer

**Confirmed exactly**: today, covering a range from a middle seed frame
needs the user to (1) scrub to the seed frame and click to define masks,
(2) press Mark Start, (3) press Mark End, (4) press **Track Forward**
(seed → mark end), (5) press **Track Backward** (mark start → seed) — two
separate manual actions for what's conceptually one operation, easy to
do only one of by mistake. Both existing buttons already call the same
`_queue_tracking(direction)` with `first_frame`/`last_frame` bounds
derived from `mark_start`/`mark_end`/the current frame
(`cutie_init_panel.py:1264-1322`) — nothing about the underlying
`TrackingJob`/`CutieWorker`/`JobQueueRunner` machinery needs to change for
this.

**Proposal**: this is a cheap, pure-orchestration fix. Add one combined
action — "Segment range from seed frame" — that takes the current
mark-start/mark-end range and the current scrubber position as the seed,
and internally issues both `_queue_tracking` calls (skipping whichever
direction is a no-op when the seed coincides with an edge, i.e.
`seed == mark_start` only needs a forward job, `seed == mark_end` only a
backward one). Keep the existing individual Forward/Backward buttons
for the power-user case (resuming just one direction after a failed or
cancelled job) rather than removing them — just make the combined action
the primary, recommended path and relabel the section so "seed frame"
and "range" are the concepts foregrounded, not "forward"/"backward."

## Issue 4 — pre-planning split points at known-difficult moments

The underlying capability already exists — `TrackingJob.first_frame`/
`last_frame` already bounds propagation to an arbitrary sub-range, so
nothing stops a user from manually segmenting a video as several
independent local ranges today, each with its own seed frame(s), never
crossing a hard moment (e.g. two people crossing paths) where a single
continuous propagation would diverge. What's missing is **workflow
support** for planning that up front, and a safety net against
accidentally queueing a job that crosses a planned split.

**Proposal**: a lightweight "split points" list, editable on the
`RangeBar` timeline before any SAM2 clicking starts — mark one or more
candidate hard-transition frames.

**Visual treatment (resolved 2026-08-28)**: `RangeBar` already stacks
four layers (teal mask-coverage band, steel-blue selection fill, amber
trial-range band, bright-blue Mark Start/End ticks, white position tick
— see the class docstring, `cutie_init_panel.py:50`). Split points need
a fifth, visually distinct layer — thin full-height ticks in a color
nothing else uses (red/orange, since blue is already the mark-boundary
color), so a split point never reads as a current-selection edge.

**Interaction with Mark Start/End (resolved 2026-08-28) — snap, don't
hard-lock**: clicking a split-point marker snaps Mark Start/Mark End to
it (the low-friction path for respecting the plan). Manually setting a
range that spans a split point is still *allowed* — don't hard-block,
since sometimes a planned split turns out to be unnecessary — but the
crossed marker renders in a warning state and the queue action confirms
before proceeding, so crossing one is never silent. This also connects
to Issue 3: once split points exist, picking a seed frame should
auto-populate Mark Start/Mark End from the *enclosing* split-point pair
rather than requiring them to be set by hand each time — the marks
become "the current planned sub-segment's bounds, pre-filled, still
freely overridable."

**Persistence (revised 2026-08-28)**: originally proposed storing split
points alongside `seg_quality_runs.persons_json`. On reflection that's
the wrong owner — "two people cross paths here" is a property of the
*footage*, not of one particular attempt at segmenting it, and tying it
to a `seg_quality_runs` row means it vanishes whenever that row is
deleted or superseded (exactly the run-churn Issues 1–2 are trying to
reduce). Use a small **capture-scoped** table instead, e.g.
`capture_segmentation_hints(id, capture_id, time_s, note, created_at)`,
storing `time_s` as global time the same way trial ranges already are —
split points must be global-time, not per-camera frame numbers, since
"the moment two people cross" is one synchronized real-world event that
has to mean the same thing on all cameras of the capture. This survives
across "New segmentation" resets and isn't tied to any one run's
lifecycle. Still pure planning scaffolding otherwise — no change to
`TrackingJob`/`CutieWorker` needed, just a new small table plus the
RangeBar/interaction UI above — medium effort, not trivial.

**Built (2026-08-29), then corrected from real use**: shipped as
described above, plus a fifth RangeBar layer for split points and a
sixth showing queued (not-yet-run) job ranges in the same coverage band.
One assumption above turned out wrong the moment it was actually used:
"the moment two people cross is one synchronized real-world event that
has to mean the same thing on all cameras" is true as a *physical*
statement but not the useful one — whether that crossing is actually
*hard to segment* is camera-angle-dependent (an occlusion in one
camera's 2D projection can be a clean parallax separation in another's).
`capture_segmentation_hints` gained a `shot_video_id` column (migration
v45→v46) and split points are now loaded/created/removed per the
currently-selected camera, not shared capture-wide. Also fixed: "Snap
Marks to Segment" was falling back to the full capture range instead of
the trial's own bounds (matching what marks already default to on load)
when there was no split point on one side.

## Issue 5 — manual mask editing (brush) when SAM2 gets it wrong

**Confirmed**: `ClickController` (`cutie_click_controller.py`) is
point-prompt only — positive/negative clicks (`push_point`), nothing
freehand. This is not a new idea — it's the same "hand-painted mask"
idea already flagged in `segmentation-pose-treatment-design.md`'s Future
Work section (2026-08-26): "the existing interactive workflow only
supports SAM2-point-click correction today... a true freehand paint tool
was the first idea raised, before the study pivoted to the
masking-treatment question." This is the point where it's worth actually
scoping it, now that the broader UI pass makes clear where it plugs in.

**Proposal sketch**: a brush/eraser mode toggle on the canvas, painting
directly into the current frame's mask (add/remove pixels for the
selected person's label within a brush radius). Two use modes worth
supporting, not just one:

- **Single-frame fix**: SAM2 got a frame mostly right; paint out a
  sliver and write directly to `seg_masks` for that one frame, no
  re-propagation needed.
- **Re-seed**: the painted correction becomes the seed mask for a new
  forward/backward job (reuses Issue 3's flow unchanged — the corrected
  mask is just a different source than a fresh set of SAM2 clicks).

This is the largest single item here — real interactive paint-canvas
work (brush cursor, undo, label-aware painting, acceptable performance
on full-resolution frames) — bigger in scope than Issues 1–4 combined.
Sequence it after the others (see "Suggested build order").

**Design finalized (2026-08-29)**, after reading `ClickController` and
`VideoCanvas` in full rather than guessing at the architecture:

- **Four mutually-exclusive tools** (`QButtonGroup` of checkable
  buttons, same pattern the person selector already uses): Select
  (today's SAM2 point-click, unchanged, default), Paint (stamps the
  currently-selected person's label within a brush radius), Erase
  (stamps background/0 — see below), Zoom (click zooms in centered on
  the clicked pixel, Alt-click zooms out, right-click resets to fit).
  Icons: no icon font ships with PySide6 and none of Qt's built-in
  `QStyle.standardIcon` set fits (no paintbrush/magic-wand/zoom
  glyphs); continuing this file's existing plain-Unicode-emoji-as-
  button-text convention (🎯 ✂ ⇤ ▶ ◀ ✓ already in use) rather than
  adding an icon-font dependency for this.
- **Erase = stamp background (0), not "revert to the layer below."**
  Simpler (paint and erase become symmetric — both just stamp a value),
  and directly serves the motivating use case: cleaning up stray
  leftover pixels Cutie/SAM2 mislabeled when two people cross paths,
  which "revert" wouldn't reach if there's no clean layer underneath to
  revert to.
- **The real architecture problem**: `ClickController._run_predictions()`
  is a pure function of `(base_mask, live_clicks)`, re-run and fully
  rebuilt on *every* click — a naive paint/erase writing straight into
  `self._mask` would be silently discarded by the next SAM2 click
  anywhere. Fix: a third compositing layer, applied last, always wins —
  `self._paint_overlay` (H×W, sentinel 255 = untouched), applied after
  `_run_predictions()`'s existing base+SAM2 compositing. A "Clear Manual
  Edits" action (mirroring the existing Clear Person/Clear All) resets
  it. Consequence worth knowing: once a person's pixels are hand-edited,
  a *later* SAM2 click for that same person won't override them (the
  overlay always wins) until manually cleared.
- **Re-seed use case validated for free**: Harri's motivating scenario —
  erase/relabel Cutie's leftover pixels from an occlusion, then use the
  corrected frame as the seed for re-running the affected range — needs
  no new plumbing beyond the overlay itself, since `_queue_tracking()`
  already seeds from `self._controller.get_mask()` when it's non-empty,
  and the overlay is folded into exactly that return value.
- **`VideoCanvas` needs real additions**, not just wiring: it has no
  mouse-move tracking and no zoom/pan state today (always fits the
  whole frame to the widget). Brush cursor: track mouse-move, but repaint
  cheaply by copying the last fully-rendered `QPixmap` and drawing the
  cursor circle on top, rather than redoing the cv2 decode/resize/blend
  pipeline at mouse-move rates. Zoom: a zoom factor + pan center on top
  of today's fit-to-widget scale; brush radius is stored in image
  pixels (not screen pixels) so painting stays consistent across zoom
  levels, only the on-screen cursor circle's radius scales with zoom.
- **Known v1 gap, accepted rather than solved now**: click-only zoom
  means panning without changing zoom level is indirect (zoom out, then
  zoom in elsewhere). Ship it as asked; revisit with drag-to-pan or
  scroll-wheel-zoom only if that proves annoying in practice.

**Implemented (2026-08-29)** exactly as designed above, across
`ClickController` (paint overlay), `VideoCanvas` (zoom/pan, brush
cursor), and `CutieInitPanel` (the four tool buttons, brush slider,
Clear Manual Edits). The motivating use case was verified end to end in
tests: erasing/relabeling a stray Cutie/SAM2 leftover pixel and re-
seeding the affected range from the corrected frame needed no new
plumbing beyond the overlay itself.

## Issue 6 — cross-camera seeding / triangulation error-detection (future)

Flagged explicitly as far-fetched/future — scoped only briefly here.
Two distinct sub-ideas, worth not conflating:

- **Cross-camera seeding**: literal pixel-mask transfer between cameras
  isn't meaningful (different viewpoints), but a *reprojection-guided*
  hint is plausible — given one camera's already-segmented mask/bbox
  centroid, extrinsics, and a rough 3D estimate (from an
  already-triangulated skeleton state, if tracking has run, or
  approximate scene geometry), project an approximate point into another
  camera's image as a positive-click hint rather than a full mask.
  Meaningfully cheaper than it sounds, but depends on extrinsics already
  being solved, which isn't always true this early in a capture's
  pipeline.
- **Triangulation-based segmentation error detection**: the same
  DLT-triangulation-consistency idea already proposed in
  `segmentation-pose-treatment-design.md` for validating keypoint edits
  (reusing `skeleton_scaling_panel.py`'s patterns), just applied one
  level coarser — mask/bbox centroid consistency across cameras instead
  of full keypoint reprojection. Cheaper to compute than the keypoint
  version, and could plausibly run automatically as a "this camera's
  segmentation looks inconsistent with the others" flag.

Neither is recommended to start now — real complexity (extrinsics
dependency, a second triangulation-tooling effort alongside the one
already proposed for keypoints) for a payoff that's speculative until
Issues 1–5 are in place and segmentation work is actually easier to do
well by hand first.

## Suggested build order

1. **Issue 3** (seed+range combined queue action) — cheapest, pure UI
   orchestration on existing job machinery, no schema change.
2. **Issue 1** (tree visibility, rename, delete) — one schema migration
   (`name` column) plus a list view modeled directly on the existing
   `detection_run`/`tracking_run` tree pattern.
3. **Issue 2** (continue existing segmentation) — small change once
   Issue 1's picker exists to select which run to continue; the
   underlying `_seg_init_run_id` mechanism is most of the way there
   already.
4. **Issue 4** (split-point planning) — benefits from 1–3 already making
   "several independent local segments" a visible, manageable concept
   rather than an invisible side effect of how many times the panel
   happened to be reopened.
5. **Issue 5** (manual brush editing) — largest scoped effort; sequence
   last among the concrete items so it isn't competing for attention
   with the cheaper, high-value fixes above.
6. **Issue 6** — revisit later, not scheduled.

## References

- `python/app/pose/cutie_init_panel.py` — the segmentation panel these
  issues are about.
- `python/app/ui/session_tree.py` — the existing tree-list/rename/delete
  pattern Issue 1 proposes extending.
- [`segmentation-reuse`](../segmentation-reuse/segmentation-reuse-design.md) —
  segmentation-as-bbox-source; a different, mostly-built concern from the
  UI-management issues here.
- [`segmentation-pose-treatment`](../segmentation-pose-treatment/segmentation-pose-treatment-design.md) —
  source of the hand-painted-mask idea (Issue 5) and the
  triangulation-consistency proposal (Issue 6).
