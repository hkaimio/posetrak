# Segmentation as a reusable, time-range-scoped bbox source — design sketch

> **Status (2026-08-16)**: All three gaps implemented and unit-tested,
> option (a) chosen for gap 2's pipeline convergence (invoke
> PoseWorker/PoseExtractionJob rather than teaching DetectionPipeline a
> second bbox source) — see status.md for what landed, how, and what's
> still open (notably: not yet live-tested against a real video/session).
>
> Originally written up 2026-07-14 after a real-data test of
> hand-detection-refinement exposed that reusing an existing segmentation
> across multiple detection runs was currently impossible, tracing back to
> this same data-model misunderstanding.

## Motivating misunderstanding

While validating hand-detection-refinement Phase 2 against real session
data, the workflow required creating a second detection run for a shot that
already had a completed, human-reviewed Cutie interactive segmentation, and
copying that run's existing full-body detections into it by hand (a one-off
script) rather than through any supported UI path. Investigating why no such
path exists surfaced that this isn't a small gap — it's a genuine
architecture mismatch with the intended design:

- **User's intent**: a segmentation covers an explicit time range on a
  capture's timeline (typically one trial's worth, see terminology note
  below), independent of any specific detection run. Any number of
  detection runs whose time range it covers (a redo with a different pose
  model, an added hand-refinement pass, etc.) should be able to reuse it as
  their bbox source instead of re-running YOLO.
- **What's actually built**: `seg_quality_runs.detection_run_id` is a
  `NOT NULL REFERENCES detection_runs(id)` (`db/session_schema.sql:338-346`)
  — a segmentation quality run is permanently tied to exactly one detection
  run at the schema level. There is no supported way to point a *second*
  detection run at an existing segmentation.

**Terminology note / correction (Harri, 2026-07-14)**: the schema
distinguishes a **capture** (`captures` table, sometimes called "shot" —
`capture_videos.shot_id` confusingly references `captures`, not a separate
shots table) — one continuous multi-camera recording — from a **trial**
(`trials` table): "a named, bounded time window within a capture: one
technique, one attempt" (the user-facing unit of analysis). The original
draft of this note guessed the user meant "capture" when saying "trial" —
wrong. The intent is literally the trial: **a segmentation should be linked
to an explicit time range on the capture's timeline (not to a specific
trial ID or detection run), and is reusable by any trial whose range it
fully contains** — not a rigid one-segmentation-per-capture link. In
practice segmentations are usually kept as short as a single trial (both to
bound Cutie processing time and because segmentation frequently needs a lot
of manual correction), so the common case is one segmentation per trial —
but a capture can have several segmentations, and a segmentation can
deliberately span more than one trial when that's useful. Reusability is
therefore a **containment check** (segmentation range ⊇ trial range), not a
foreign key to one specific trial.

## Current state, traced concretely

Three separate, independent gaps — none of them small:

**1. Schema: segmentation is 1:1 with a detection run, not time-range-scoped.**
`seg_quality_runs.detection_run_id` is the FK described above, and the table
has no `time_start_s`/`time_end_s` of its own (unlike `trials` and
`detection_runs`, which both do) — there's no way today to ask "which
segmentations cover this trial's time range" at all. `seg_masks` (the actual
mask storage) is keyed by `(seg_quality_run_id, shot_video_id, frame_idx)` —
already capture/camera/frame scoped underneath, so the mask *data* has no
inherent dependency on a detection run. The dependency is purely
`seg_quality_runs`' own FK column plus its missing time-range columns.

**2. Pipeline: segmentation-driven pose extraction is a wholly separate code
path from YOLO-driven detection, not a bbox-source option within one flow.**
- YOLO path: `RunDetectionDialog` → `DetectionPipeline`/`DetectionJob`
  (`posetrak/detection/pipeline.py`). No awareness of segmentation at all
  (confirmed: zero references to `seg_quality_run` anywhere in
  `run_detection_dialog.py`).
- Segmentation path: `CutieInitPanel`'s "Queue Pose" button →
  `PoseWorker`/`PoseExtractionJob` (`app/pose/pose_worker.py`), entirely
  separate from `DetectionPipeline`. This is *already* the better bbox
  source in practice (tighter, more accurate crops than YOLO — see
  `hand-detection-refinement-design.md`'s commit `4b0266e` message), but it's
  only reachable from the Cutie segmentation panel itself, immediately after
  segmenting — never as a later, independent choice against an
  already-existing segmentation.

**3. Auto-assignment from segmentation labels isn't built, even though the
data already supports it.** `PoseExtractionJob.persons_ordered: list[str]`
already gives a stable `track_id (= index+1) → person_name` mapping straight
from the interactive segmentation labeling — no ByteTrack-style identity
drift to stitch, unlike YOLO tracks. But `pose_worker.py` never writes
`detection_track_assignments` or `person_tracks` from it. Today, even a
segmentation-driven run still requires the user to manually assign and
stitch tracks in `stitcher_panel` before `finalise_to_db` can run, exactly
as if it were a YOLO run with unstable track IDs.

## Target vision (user's description, 2026-07-14)

1. A segmentation covers an explicit time range on the capture's timeline —
   typically kept as short as one trial (to bound Cutie processing time and
   the manual-fixing effort segmentation often needs), but not required to
   be. It is not specific to any detection run, and a capture can have
   several segmentations covering different (or overlapping) trials.
2. When running detection against a trial that has a covering segmentation,
   the user can choose to source bboxes from that segmentation instead of
   running YOLO person detection. Running detection directly from the
   segmentation panel (`CutieInitPanel`'s existing "Queue Pose" flow)
   remains available exactly as today.
3. When a detection run is segmentation-sourced: persons are automatically
   created from the segmentation's labels, and track assignments are
   generated automatically (segmentation labels are already stable
   identities — no stitching ambiguity in the common case). The user can
   still open the stitcher to correct segmentation errors (segmentation
   isn't perfect either), but the common path is accept-and-finalise with no
   manual assignment step.
4. When a detection run is YOLO-sourced (no segmentation available, or the
   user chooses not to use it): today's existing behavior is unchanged —
   the user creates and assigns persons manually in the stitcher.

## Sketch of the changes this implies

Not designed in detail — this is a first pass to size the work, for
whoever picks this up later.

**Schema migration** (non-additive, like Phase 2's `pose_observations`
migration in hand-detection-refinement): give `seg_quality_runs` a `shot_id`
(capture) column plus `time_start_s`/`time_end_s` (matching `trials`' and
`detection_runs`' own convention), and drop `detection_run_id`. SQLite can't
alter this in place — rebuild-the-table migration, same shape as the
`pose_observations` source-column migration. "Which segmentations cover
trial X" becomes a containment query: same `shot_id`, and
`seg.time_start_s <= trial.time_start_s AND seg.time_end_s >=
trial.time_end_s`. Existing rows migrate by copying their current
`detection_run_id`'s `shot_id` and `time_start_s`/`time_end_s` (a reasonable
default since today's 1:1 relationship means the owning detection run's
range *is* the segmentation's range in every existing session). Anything
reading `seg_quality_runs.detection_run_id` today (`cutie_init_panel.py`,
`trial_export.py`, tooling under `python/tools/`) needs updating to look up
by `shot_id` + range containment (+ whichever selection UI replaces "the
segmentation for this detection run" with "the segmentation(s) covering this
trial," see below).

**Pipeline convergence**: `RunDetectionDialog` needs a bbox-source choice
(YOLO vs. an existing `seg_quality_runs` row covering this trial, if any
exist — could be more than one, needing a picker if so).
Two ways to get there, not evaluated against each other yet:
- Have `RunDetectionDialog`, when segmentation is chosen, invoke the same
  `PoseWorker`/`PoseExtractionJob` machinery `CutieInitPanel` already uses,
  rather than `DetectionPipeline`/`DetectionJob`.
- Or teach `DetectionPipeline` itself to accept a segmentation-derived bbox
  source alongside YOLO, unifying the two pipelines properly instead of
  dispatching between them.
The first is much less work; the second is probably the more honest
long-term shape (one detection pipeline, pluggable bbox source) given
hand-detection-refinement's own multi-source `pose_observations` direction
is already pushing this codebase toward "many sources feeding one merged
result" as a recurring pattern.

**Auto-assignment**: when finalising a segmentation-sourced run, generate
`TrackAssignment`s directly from `persons_ordered` (one per label, spanning
the full processed frame range) instead of requiring the user to build them
in the stitcher first. The stitcher UI would still open for review/correction
when the user wants it, but stop being a mandatory gate for this bbox
source.

## Open questions (not resolved here)

1. ~~Capture-level vs. trial-level scoping for segmentation~~ — resolved:
   time-range-scoped (own `time_start_s`/`time_end_s`), reusable by any
   trial it fully contains. See terminology note above.
2. ~~Whether to converge the two pose-extraction pipelines...~~ — resolved
   2026-08-16: kept separate, dispatch point added in `RunDetectionDialog`
   (option (a)). `DetectionPipeline`/`DetectionJob` (YOLO) are untouched;
   choosing a segmentation invokes `PoseWorker`/`PoseExtractionJob`/
   `JobQueueRunner` instead. Properly unifying the two into one pipeline
   with a pluggable bbox source remains the more honest long-term shape
   per the reasoning below, just not done.
3. What happens to `keypoint_obs_quality` (per-keypoint segmentation quality
   scores, currently keyed by `seg_run_id` alongside `detection_run_id`'s
   own `detection_keypoints`) once one segmentation can feed multiple
   detection runs — does quality scoring stay keyed to the segmentation
   alone (shared across all runs sourced from it), or does it need to be
   recomputed / rekeyed per consuming detection run?
4. UI for the case where more than one existing segmentation covers a
   trial's range (e.g. a capture-spanning segmentation and a shorter
   trial-specific one both qualify) — needs a picker, not just "the"
   segmentation for a trial. Partially addressed 2026-08-16:
   `RunDetectionDialog`'s bbox-source combo does list every segmentation
   for the capture (most recent first), so a picker exists -- but it
   lists all of them unconditionally rather than filtering/ranking by
   actual range containment against the trial being detected, since
   time_start_s/time_end_s narrowing (open question below) isn't done
   either.
5. Migration path for existing sessions that already have
   `seg_quality_runs` rows tied to now-old detection runs — straightforward
   given the 1:1 relationship today (each existing row's new `shot_id` and
   `time_start_s`/`time_end_s` are just its current `detection_run_id`'s
   own values), but worth confirming no session has more than one
   `seg_quality_runs` row per capture already in a way that would change
   semantics post-migration.
