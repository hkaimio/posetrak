# Cross-person observation diagnostics (Phase 5 follow-on) — design sketch

## Status

Design sketch only, not yet implemented. Written up after Harri asked whether
there's any way to tell, from tracker output, which keypoints fed a
cross-person anchor on a given frame — motivated by observing more jitter
when persons are close together in test runs, and wanting to distinguish
"the cross-person coupling algorithm is doing this" from "there's just more
occlusion/bad data when people are close" (both are plausible and not
distinguishable from existing output).

## Problem

Cross-person anchor observations (`phase5-cross-person-plan.md`) are built
at runtime by `MultiPersonTracker`, consumed by the UKF update, and then
discarded. Nothing downstream records which marker was anchored to which
other person's marker, in which camera, on which frame. Confirmed by reading
the code:

- `Observation::anchor_position` (`observation.hpp:48`) is a bare pixel
  value. The person and marker it was computed from exist only in the
  transient `Candidate` struct inside `build_cross_person_anchors()`
  (`multi_person_tracker.cpp:768-774`) and are gone once the `Observation`
  is built.
- None of the output CSVs (`observations.csv`, `marker_projections.csv`,
  `predicted_observations.csv`, `tracking_stats.csv`) or the session DB
  (`tracking_obs_results.obs_blob`) carry a mode/kind/anchor field at all.
- The MCP tools (`get_filter_stats`, `get_camera_coverage`,
  `get_observation_gaps`) inherit this gap — they decode `obs_blob` with a
  fixed `float32[n_cam, n_mrk, 8]` shape that has no room for a "kind"
  dimension.
- This matches the plan doc's own Stage 4 status note (line 47-48): the
  "Contact-window summary UI" is explicitly flagged as a deferred,
  not-yet-designed data surface, not an oversight.

## A pre-existing bug this surfaced

While tracing this, found that direct detections and PAIR_DIFF observations
for the *same* `(marker, camera)` in the *same* frame already collide in
today's output, and this is not specific to cross-person anchors — it
affects the existing within-person `PAIR_DIFF` relative observations from
Phase 3/4 too.

`step_person_context()` (`multi_person_tracker.cpp:456-457`) concatenates a
person's direct-detection observations with the anchor observations built
for that frame into one vector fed to `track_frame()`. A cross-person
anchor's `marker_id` is deliberately the *same* marker id as the direct
detection it was built from (`multi_person_tracker.cpp:827`:
`anchor.marker_id = c.mine->marker_id`, and `c.mine` itself comes from that
person's own per-frame detections at line 750). So whenever cross-person
gating is active, a direct `POSITION` observation and a `PAIR_DIFF` anchor
observation for the same `(marker_id, camera_id)` coexist in the same
frame's observation list by construction — this is the normal case during
contact, not an edge case.

Downstream, though:

- `tracking_export.cpp:197-201`'s `obs_map` (feeding `observations.csv` and
  `marker_projections.csv`) is keyed by `(marker_id, camera_id)` only.
- `result_writer.cpp:320`'s blob-slot lookup (feeding
  `tracking_obs_results.obs_blob`) is keyed by `marker_name` only.

Both silently let one observation overwrite the other when building
per-frame diagnostic rows. Today this already happens for within-person
`PAIR_DIFF`; Phase 5 makes it worse (three candidates can now compete for
one slot: the direct detection, a within-person relative observation, and a
cross-person anchor).

## Why marker identity must be a name, not an index

Harri's point, confirmed against the code: the anchoring person may use a
different skeleton, where marker indices don't mean the same keypoint. The
anchor-construction code already knows this — it resolves the *other*
person's marker via `other_skeleton.markers()[c.other_marker].name`
(`multi_person_tracker.cpp:816`) specifically to get a cross-skeleton-safe
lookup for FK/projection — but that resolved name is discarded immediately
after use; only the raw index (meaningless outside `other_skeleton`)
survives structurally in the `Candidate`. Any provenance we add must carry
the resolved **name**, not the index.

## Design

### A. `Observation` (`observation.hpp`) — carry anchor identity, not just anchor position

Add two fields alongside `anchor_position`:

```cpp
int anchor_person_idx = -1;       // index into MultiPersonTracker's person list; -1 = not cross-person
std::string anchor_marker_name;   // other person's marker name, in their skeleton's naming
```

Populated at `multi_person_tracker.cpp:825-837` from `other_idx` and the
already-resolved `other_marker_name` (currently computed at line 816 and
discarded). `mode` and `anchor_position.has_value()` already distinguish
direct detection / within-person `PAIR_DIFF` / cross-person anchor — no new
enum needed, just the missing identity fields.

### B. `ObservationResult` (`update_result.hpp`) — mirror that provenance through the UKF

Add:

```cpp
MeasurementMode mode;
std::string ref_marker_name;      // within-person PAIR_DIFF: parent marker name, same skeleton
int anchor_person_id = -1;        // cross-person: other person's DB person_id, else -1
std::string anchor_marker_name;   // cross-person: other person's marker name
```

Populate at the three `ObservationResult` construction sites in `ukf.cpp`
(~lines 2062, 2176, 2199) directly from the source `Observation`. This is
additive only — `predicted`/`actual`/`mahalanobis_distance`/`is_outlier`
are already computed for every observation regardless of mode; the
provenance is discarded on the way out, not missing at the source.

### C. Stop routing `PAIR_DIFF` rows into the detection-shaped outputs

`observations.csv`, `marker_projections.csv`, and
`tracking_obs_results.obs_blob` are structurally a dense
`(camera, marker) → one value` grid. That model is correct for direct
detections and wrong for `PAIR_DIFF`, which is the root cause of the
collision above. Filter `mode != POSITION` out of `tracking_export.cpp`'s
`obs_map` build and `result_writer.cpp`'s blob-fill loop, so only direct
detections land in these outputs.

**This is a behavior change for existing Phase 3/4 output**: within-person
`PAIR_DIFF` rows currently do land in `observations.csv` /
`marker_projections.csv` / `obs_blob` today (silently clobbering the
detection row when both exist for the same slot). After this change they
would stop appearing there entirely, moving instead to the new table below.
Confirm this is acceptable before implementing — it changes the contents of
these outputs for any run using within-person relative observations, not
just multi-person runs.

### D. New table for relative/cross-person observations

`obs_blob`'s dense `(camera, marker)` grid is the wrong shape for `PAIR_DIFF`
observations — they aren't one-value-per-slot, and cross-person ones also
need to reference a second person/marker. A relative/cross-person
observation is also sparse by nature (only exists during contact or
within-person relative-pair windows), so a dense blob would waste space.
Add a normalized table instead:

```sql
CREATE TABLE tracking_relative_obs_results (
    run_id             TEXT    NOT NULL REFERENCES tracking_runs(id),
    person_id          INTEGER NOT NULL,   -- owning person (whose track_frame consumed this)
    tracker_step       INTEGER NOT NULL,
    camera_id          INTEGER NOT NULL,
    marker_name        TEXT    NOT NULL,   -- owning marker, this person's skeleton
    kind               TEXT    NOT NULL,   -- 'within_person_pair_diff' | 'cross_person_anchor'
    ref_marker_name    TEXT,               -- within-person: parent marker name (same skeleton)
    anchor_person_id   INTEGER,            -- cross-person: other person's person_id, else NULL
    anchor_marker_name TEXT,               -- cross-person: other person's marker name (their skeleton)
    dist3d_mm          REAL,               -- cross-person: 3D distance that gated the pair in
    predicted_x REAL, predicted_y REAL,
    actual_x REAL, actual_y REAL,
    mahalanobis_distance REAL,
    is_outlier         INTEGER NOT NULL,
    PRIMARY KEY (run_id, person_id, tracker_step, camera_id, marker_name, kind, anchor_person_id)
);
```

`person_id` / `anchor_person_id` are real DB `person_id`s, not
`MultiPersonTracker` array indices — resolved from each person context's
`person_id` at write time. Written from `ResultWriter`, same place
`write_obs_results` is called, filtering `UpdateResult::observations` for
`mode == PAIR_DIFF` instead of `mode == POSITION`.

### E. MCP surfacing

New tool, e.g. `get_cross_person_observations(run_id, start_s, end_s)`,
reading the new table: per frame, which of this person's markers were
anchored, to which other person, to which of their markers, in which
cameras, with what 3D gating distance, mahalanobis distance, and
inlier/outlier status. This is the backing data the original Phase 5 plan's
deferred "Contact-window summary UI" needed but never had.

## How this answers the jitter question

Join, for a jittery marker/frame: whether a cross-person anchor was active
for it (new table), its mahalanobis distance and inlier/outlier status
(same table), and whether a direct detection also existed simultaneously in
the same camera (join against the now-unpolluted `obs_blob` /
`observations.csv` on `(person_id, camera_id, marker_name, tracker_step)`,
which after change C only contains direct detections).

- Anchor present, inlier, low mahalanobis, but jitter persists → points at
  the coupling algorithm itself (or the noise composition being too tight —
  see the σ_anchor floor/inflation discussion in
  `phase5-cross-person-plan.md`'s measurement-model section).
- Anchor frequently rejected as outlier, or sparse/flickering camera
  coverage on either side → points at bad data / occlusion rather than the
  algorithm.

## Explicitly out of scope for this sketch

- **Per-frame contact-gate logging independent of realized observations**
  (i.e. recording the active-pair set even on frames where the gate
  excluded a pair, or where hysteresis kept a pair active past when it
  should have deactivated). Would help debug gate mis-decisions
  specifically, as opposed to the anchors that were actually built. Smaller
  addition on top of this design (the gate already computes
  `active_contact_pairs_` every frame in `update_contact_gate()`) but not
  needed to answer the immediate jitter question and left for a later pass
  if the per-observation table proves insufficient.
- **UI panel** (`content_panels.py`) surfacing this — MCP-level access is
  the immediate need; a UI surface is a separate, later piece of work, same
  as the original plan's deferred Contact-window summary UI.
- **Fixing the pre-existing within-person `PAIR_DIFF` collision** as an
  isolated bug fix without the rest of this design — possible to do
  narrowly, but change C above (routing `PAIR_DIFF` out of the
  detection-shaped outputs entirely) is the natural fix and is needed for
  Phase 5 regardless, so no separate narrower fix is proposed here.

## Files likely touched

- `include/posetrak/core/observation.hpp` (`anchor_person_idx`,
  `anchor_marker_name`)
- `src/tracking/multi_person_tracker.cpp` (~lines 816, 825-837 — populate
  the new fields instead of discarding `other_marker_name`/`other_idx`)
- `include/posetrak/filters/update_result.hpp` (`mode`, `ref_marker_name`,
  `anchor_person_id`, `anchor_marker_name` on `ObservationResult`)
- `src/filters/ukf.cpp` (~lines 2062, 2176, 2199 — populate the new
  `ObservationResult` fields)
- `src/io/tracking_export.cpp` (~lines 197-202 — filter `obs_map` to
  `mode == POSITION`)
- `src/db/result_writer.cpp` (~line 320 — filter blob-fill to
  `mode == POSITION`; new write path for `tracking_relative_obs_results`)
- `db/session_schema.sql` (+ migration) — new
  `tracking_relative_obs_results` table
- `python/app/mcp/` — new `get_cross_person_observations` tool, plus
  `db.py` query support for the new table
