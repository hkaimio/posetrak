# SOMA/GEM-X productization: architecture and schema summary

The target architecture after all features of
[the productization plan](soma-gemx-productization-plan.md), and every database change merged across
the feature designs, with overlaps resolved here. The feature designs are:

- [detection pipeline](../detection-pipeline/detection-pipeline-design.md)
- [pose-model skeleton binding](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md)
- [keypoint candidates](../keypoint-candidates/keypoint-candidates-design.md)
- [track to person assignment](../track-person-assignment/track-person-assignment-design.md)
- [hand refinement without segmentation](../hand-detection-refinement/hand-refinement-without-segmentation-design.md)
- [segmentation versioning](../segmentation-versioning/segmentation-versioning-design.md)
- [tracklet editing](../keypoint-editing/tracklet-editing-design.md)
- [tracker performance](../tracker-performance/tracker-performance-design.md)
- [session DB concurrency](../session-db-concurrency/session-db-concurrency-design.md)
- [calibration robustness](../extrinsics-improvements/calibration-robustness-design.md)

## 1. Components and data flow

```
 capture videos ─┬─────────────────────────────────────────────────────────────────────────┐
                 │                                                                         │
 segmentation ───┤ seg_quality_runs, seg_masks @ revision                                  │
 (Cutie, editor) │   seg_mask_revisions                                                    │
                 ▼                                                                         │
 ┌─────────────────────── detect run (one frame loop) ──────────────────────┐              │
 │ decode threads ─▶ BoxSource ─▶ batched PoseEstimator ─▶ hand pass ─▶ writer │            │
 │   (per camera)     YOLOX |        RTMPose-133 |          rtmlib.Hand        │            │
 │                    masks@rev |    SOMA-77 (+mask gate,  (any body model)    │            │
 │                    run           + candidates, linked)                      │            │
 └──────────────────────────────────────────────────────────────────────────┘              │
        │ detection_runs (+ seg run/revision, source run), person_detections,               │
        │ detection_keypoints: full_body | hand_l | hand_r | kp_candidates                  │
        ▼                                                                                   │
 track to person assignment (YOLOX only) ─▶ detection_track_assignments                    │
        ▼                                                                                   │
 finalise ─▶ pose_observation_sequences, pose_observations (body | hand_* | kp_candidates), │
             pose_sequence_keypoints (manifest per obs_source)                              │
        ▼                                                                                   │
 posetrak-tracker (C++)                                                                     │
   SessionReader: manifest binding per source ─▶ fixed observations                        │
                  kp_candidates + dots ─▶ UnlabeledCandidate (class, owner, merged)        │
   per step: predict all subjects ─▶ shared assignment (Hungarian; modifiers: class,        │
             ownership, tracklet continuity, reacquisition, multi-view agreement,           │
             segmentation mask @ revision, tracklet rules) ─▶ update all subjects           │
   hierarchical stages (main with candidates; hand stages with refined hands)               │
   RTS smoother (diagonal covariance kept; cache released per stage)                        │
        │ tracking_runs (+ status), tracking_run_persons (+ seg run/label/revision),        │
        │ tracking_results, tracking_obs_results                                            │
        ▼                                                                                   │
 main viewer/editor (MainWindow, PersonPanel)                                               │
   crop grid: candidates, tracklet trails, used pick  ◀──────────────────────────────────────┘
   timeline: tracklet lanes, rules, review windows
   operations ─▶ dot_assignment_edits (assign, forbid, reject, other, force, split; groups)
        └──────▶ re-track the trial with the rules
```

Model weights for SOMA-77 are not in the repository; the pipeline downloads them on first use after
licence acceptance and falls back to RTMPose-133 without them.

## 2. Overlaps between the feature designs, resolved

| Overlap | Resolution |
|---|---|
| Where the segmentation used by a result is recorded (detection pipeline, mask versioning, candidates) | Columns on `detection_runs` and `tracking_run_persons`, added in the first release with `seg_revision` NULL; mask versioning fills the revision later, no rewrite. |
| Manifest layout (skeleton binding, hand refinement) | One change: `pose_sequence_keypoints.obs_source` with the key `(sequence_id, obs_source, keypoint_idx)`; hand refinement writes `hand21` rows under it. |
| Candidate settings (candidates, dots) | The existing `dot_*` settings govern every anonymous candidate; the only new setting is `kp_candidates_enabled`. |
| Tracklet decisions (editing, dots) | One table, `dot_assignment_edits`, for both; tracklet ids are unique per (detection run, camera). Its name stays to avoid churn in the dot review code; its comment says it covers keypoint tracklets. |
| Tracklet table (candidates, editing) | None. Extents and statistics are computed from candidate rows; a summary table is added only if the editor measures the computation as too slow. |
| Run status (concurrency, detection) | `detection_runs.status` exists; `tracking_runs` gets the same three values. Interrupted runs are detected the same way for both. |
| Default skeletons (binding, joint limits) | Generated together from base geometry, the joint-limit table and the SOMA offsets catalog file. |

## 3. Schema changes

Session schema version is 63 and registry schema version 16 today. Every changed table is listed with
all its columns; new columns are marked **new**. Every listed table is added to `DOCUMENTED_TABLES` in
`python/tests/db/test_schema_documentation.py` in the change that touches it, with a comment on every
column in the schema file.

### 3.1 `tracking_runs` (session) — first release

A tracker execution. Written by `posetrak-tracker` when it creates the run and when it ends.

| Column | Type | Meaning |
|---|---|---|
| id | TEXT PK | uuid of the run |
| observation_sequence_id | TEXT | `pose_observation_sequences.id` tracked |
| tracker_config_id | TEXT | registry `tracker_configs.id`, copy kept in the session |
| skeleton_id | TEXT | registry `skeletons.id`; per-subject overrides in `tracking_run_persons` |
| extrinsic_calibration_id | TEXT | `extrinsic_calibrations.id` used |
| sync_config_id | TEXT | `sync_configs.id` used |
| ran_at | TEXT | UTC start time, ISO 8601 with Z |
| posetrak_version | TEXT | tracker version and build time |
| active_camera_ids | TEXT | JSON array of camera labels; order = camera index in result blobs |
| marker_names | TEXT | JSON array of marker names; order = marker index in result blobs |
| notes | TEXT | free text; NULL when none |
| trial_id | TEXT | `trials.id`; NULL outside a trial |
| name | TEXT | display name; NULL until named |
| status | TEXT NOT NULL DEFAULT 'complete' | **new** 'running', 'complete' or 'failed'; rows from before the column are 'complete' |
| completed_at | TEXT | **new** UTC end time; NULL while running or when unknown |
| failure | TEXT | **new** error text of a failed run; NULL otherwise |

### 3.2 `tracking_run_persons` (session) — first release

Who a tracking run tracked, and the skeleton and segmentation used for them. Written by the tracker.

| Column | Type | Meaning |
|---|---|---|
| run_id | TEXT | `tracking_runs.id` |
| person_id | INTEGER | subject index in the run's results |
| skeleton_id | TEXT | registry `skeletons.id` for this subject |
| capture_object_id | TEXT | `capture_objects.id` when the subject is an object; NULL for a person |
| subject_id | TEXT | `capture_subjects.id`; NULL if unresolved |
| seg_quality_run_id | TEXT | **new** `seg_quality_runs.id` whose masks gated this subject's candidates; NULL without a mask gate |
| seg_mask_label | INTEGER | **new** the subject's label in those masks (1-255); NULL with the column above |
| seg_revision | INTEGER | **new** `seg_mask_revisions.revision` read; NULL when unversioned or no gate |

PK `(run_id, person_id)`.

### 3.3 `detection_runs` (session) — first release

One execution of a detection pipeline (pose, marker or dots). Written by `detect run`, the GUI dialogs
and the import commands; append-only.

| Column | Type | Meaning |
|---|---|---|
| id | TEXT PK | uuid of the run |
| shot_id | TEXT | `captures.id` (historical name) |
| sync_config_id | TEXT | `sync_configs.id` mapping frames to time |
| trial_id | TEXT | `trials.id`; NULL outside a trial |
| time_start_s, time_end_s | REAL | covered range, capture timeline, seconds |
| detector_model | TEXT | box source: detector name (`yolox-x`), `segmentation`, or `detection-run` |
| pose_model | TEXT | pose model name (`rtmpose-l-133kp`, `soma77-vitpose-h`); marker runs name their detector |
| detector_version, pose_version | TEXT | model versions; NULL when unknown |
| detector_conf | REAL | person detector confidence threshold |
| pose_conf_threshold | REAL | keypoint confidence threshold recorded with the run |
| pose_input_width, pose_input_height | INTEGER | model input size, pixels; NULL for marker runs |
| status | TEXT | 'running', 'complete' or 'failed' |
| created_at, completed_at | TEXT | UTC times; completed_at NULL while running |
| detector_type | TEXT | 'pose' or 'aruco' (marker and dot runs) |
| config_json | TEXT | detector and estimator options: flip test, crop enlargement, decoding, mask gate, `candidates` block, frame-rate rule, hand pass |
| capture_object_id | TEXT | `capture_objects.id` for an object run; NULL for persons |
| seg_quality_run_id | TEXT | **new** `seg_quality_runs.id` whose masks gave boxes or gated heatmaps; NULL when none |
| seg_revision | INTEGER | **new** mask revision used; NULL when unversioned or none |
| source_detection_run_id | TEXT | **new** `detection_runs.id` whose boxes were reused; NULL otherwise |

### 3.4 `detection_keypoints` (session) — first release (documentation), second release (new region type)

Raw per-crop detector output. Written by the detection pipeline and the hand pass.

| Column | Type | Meaning |
|---|---|---|
| detection_run_id | TEXT | `detection_runs.id` |
| shot_video_id | TEXT | `capture_videos.id` (one camera's video) |
| video_frame | INTEGER | absolute frame in that video |
| track_id | INTEGER | detector track id, or mask label for a mask-sourced run; the fixed dot track id for dot rows |
| region_type | TEXT | 'full_body', 'hand_l', 'hand_r', 'dots', marker corner types, **new** 'kp_candidates' |
| keypoints | BLOB | by region type: float32 [n, 3] (x, y, confidence) distorted px; dots: int32 n + float32 [n, 9]; kp_candidates: int32 n, int32 n_fields, float32 [n, n_fields] with keypoint_idx, x, y, peak, rank, tracklet_id |
| noise_scale | REAL | crop width / model input width; NULL for dots |

PK `(detection_run_id, shot_video_id, video_frame, track_id, region_type)`.

### 3.5 `pose_observations` (session) — second release (new source value)

Tracking input rows. Written by finalisation and by `add_dots_to_sequence`.

| Column | Type | Meaning |
|---|---|---|
| sequence_id | TEXT | `pose_observation_sequences.id` |
| camera_instance_id | TEXT | registry `camera_instances.id` |
| video_frame | INTEGER | absolute frame |
| timestamp_s | REAL | capture time from the sync config |
| person_id | INTEGER | index within the sequence |
| source | TEXT | 'body', 'hand_l', 'hand_r' (and '.refined' variants), 'dots', **new** 'kp_candidates' |
| detection_run_id | TEXT | run the row came from |
| kp_blob | BLOB | same layouts as `detection_keypoints.keypoints` for the corresponding type |
| noise_scale | REAL | measurement noise scale |

PK `(sequence_id, camera_instance_id, video_frame, person_id, source)`.

### 3.6 `pose_sequence_keypoints` (session) — first release (rebuild)

Keypoint manifest: names each blob slot of a sequence's rows of one observation source. Written by
finalisation for every new sequence and by marker finalisation.

| Column | Type | Meaning |
|---|---|---|
| sequence_id | TEXT | `pose_observation_sequences.id` |
| obs_source | TEXT NOT NULL DEFAULT 'body' | **new** `pose_observations.source` whose blob the row indexes ('body', 'hand_l', 'hand_r'); existing rows 'body' |
| keypoint_idx | INTEGER | slot in that blob |
| name | TEXT | landmark matched against `Marker::landmark`: `<model>:<keypoint>` (`soma77:LeftHand`), or a marker corner `<marker>:c<0-3>` |
| source | TEXT | model or detector the slot comes from ('soma77', 'coco133', 'hand21', 'aruco') |

PK changes to `(sequence_id, obs_source, keypoint_idx)`.

### 3.7 `seg_masks` (session) — second release (rebuild)

| Column | Type | Meaning |
|---|---|---|
| seg_quality_run_id | TEXT | `seg_quality_runs.id` |
| shot_video_id | TEXT | `capture_videos.id` |
| frame_idx | INTEGER | absolute frame |
| revision | INTEGER NOT NULL DEFAULT 0 | **new** `seg_mask_revisions.revision` from which the row holds |
| mask_blob | BLOB | indexed PNG, 0 background, 1-255 subject labels; **NULL (new)** = no mask from this revision on |

PK changes to `(seg_quality_run_id, shot_video_id, frame_idx, revision)`. View `seg_masks_latest`
returns the newest row per frame.

### 3.8 `seg_mask_revisions` (session) — second release (new table)

One saved revision of a segmentation run. Written by the segmentation panel and the migration.

| Column | Type | Meaning |
|---|---|---|
| seg_quality_run_id | TEXT | `seg_quality_runs.id` |
| revision | INTEGER | 0 for the state before versioning, then increasing per save |
| created_at | TEXT | UTC time the revision was opened |
| closed_at | TEXT | UTC time the revision was saved; NULL while it is open (being edited) |
| note | TEXT | optional description; NULL when none |

PK `(seg_quality_run_id, revision)`. A partial unique index on `seg_quality_run_id WHERE closed_at IS
NULL` allows at most one open revision per segmentation run. `seg_masks_latest` and every pinned
read consider closed revisions only.

### 3.9 `tracker_configs` (registry and session copy) — first and second release

All existing columns stay as documented in `registry_schema.sql` (id, name, parent_id, created_at,
alpha, beta, kappa, process_noise_std, process_noise_vel_std, velocity_half_life_s,
measurement_noise_std, outlier_threshold, tracker_fps, ik_max_iterations, ik_tolerance,
init_position_std, init_orientation_std, init_joint_std, init_velocity_std, min_cameras_for_init,
velocity_mode_camera_ids, velocity_measurement_noise_std, notes, pose_noise_std,
use_relative_observations, relative_min_confidence, cross_pair_max_px, cross_pair_max_n,
process_noise_vel_gain_joint, process_noise_vel_ref_joint, process_noise_vel_gain_root,
process_noise_vel_ref_root, process_noise_vel_joint_names, pose_reg_joint_names,
pose_reg_equal_split_noise_std, pose_reg_rest_pose_noise_std, nis_feedback_scopes,
nis_feedback_window, nis_feedback_threshold, nis_feedback_max_multiplier, process_noise_vel_scopes,
soft_limit_joint_names, soft_limit_margin_rad, soft_limit_noise_std, near_limit_damping_joint_names,
near_limit_margin_rad, near_limit_spread_sigma, near_limit_damping_factor, edited_kp_noise_std,
cross_person_max_world_mm, cross_person_min_confidence, cross_person_max_n, is_named,
dot_streak_velocity_enabled, dot_streak_k_window, dot_streak_k_min_samples,
dot_streak_min_displacement_px, dot_streak_min_elongation_px, dot_streak_velocity_noise_std,
process_noise_vel_max_multiplier, dot_assignment_gate_mahalanobis, dot_tracklet_gate_multiplier,
confidence_threshold_marker_names, confidence_threshold_override, dot_reacquire_gap_frames,
dot_reacquire_max_px, dot_cross_view_corroboration_px, dot_corroboration_gate_multiplier,
dot_camera_noise_scale, dot_seg_mask_enabled, dot_seg_mask_dilate_px, dot_seg_mask_tracklet_window,
dot_seg_mask_min_tracklet_fraction, dot_seg_mask_cost_scale_at_zero_fraction,
dot_seg_mask_hard_distance_px). The comments of the `dot_*` assignment columns are widened to say they
apply to every anonymous candidate, dots and keypoint candidates.

| Column | Type | Meaning |
|---|---|---|
| active_joint_groups | TEXT | **new, first release** JSON array of the skeleton group names the top-level (parent) stage tracks, e.g. `["main"]`; NULL = the whole skeleton. Replaces the tracker's `--joint-groups` option as the source of the parent's groups, so a run records them |
| kp_candidates_enabled | INTEGER | **new, second release** **new** 0/1: use the sequence's keypoint candidates instead of its fixed body observations for the markers of that model; NULL/0 = off |

### 3.10 `dot_assignment_edits` (session) — third release (rebuild)

Human decisions about dot and keypoint tracklets, per trial. Written by the dot review and the
keypoint editor; a later row overrides an earlier one for the same tracklet, slot and frames.

| Column | Type | Meaning |
|---|---|---|
| id | TEXT PK | id of the edit |
| trial_id | TEXT | `trials.id` |
| detection_run_id | TEXT | run whose tracklet ids the edit refers to |
| subject_id | TEXT | `capture_subjects.id` the slot belongs to; NULL for a single-subject capture |
| camera_instance_id | TEXT | registry `camera_instances.id` |
| tracklet_id | INTEGER | tracklet id, unique per detection run and camera |
| first_frame, last_frame | INTEGER | inclusive video frame range; both NULL for the whole tracklet; for 'split' first_frame is the split frame |
| kind | TEXT | 'assign', 'forbid', 'reject', 'other', **new** 'force', **new** 'split' |
| slot_name | TEXT | marker name; NULL for 'reject', 'other', 'split' |
| edit_group_id | TEXT | **new** id shared by the rows of one group operation; NULL for a single edit |
| created_at | TEXT | UTC time; the latest edit wins |

The `CHECK` on `kind` is the reason for the rebuild. `dot_assignment_proposals` is unchanged.

## 4. Migration order

| Session version | Release | Change | How |
|---|---|---|---|
| 64 | first | `tracking_runs.status`, `completed_at`, `failure`; `tracking_run_persons.seg_quality_run_id`, `seg_mask_label`, `seg_revision` | `ALTER TABLE ADD COLUMN` |
| 65 | first | `detection_runs.seg_quality_run_id`, `seg_revision`, `source_detection_run_id`; existing rows stay NULL (mask-sourced runs did not record their segmentation; the spike's prototype runs name their source run in `config_json`, which is copied into `source_detection_run_id`) | `ALTER TABLE ADD COLUMN` + update |
| 66 | first | `pose_sequence_keypoints.obs_source` and new key; `tracker_configs.active_joint_groups` (registry version 17 in step) | table rebuild (small table); `ALTER TABLE ADD COLUMN` |
| 67 | second | `seg_mask_revisions`; `seg_masks.revision`, nullable blob, new key; `seg_masks_latest` view | table rebuild with copy, revision 0 rows |
| 68 | second | `tracker_configs.kp_candidates_enabled` (registry version 18 in step) | `ALTER TABLE ADD COLUMN` |
| 69 | third | `dot_assignment_edits` kinds 'force', 'split', column `edit_group_id` | table rebuild with copy |

The version numbers are provisional and follow version 63; the order is what matters. A stage
branch takes the next free session and registry version when it merges, after any migration that
other work merged first (plan section 5, branches and schema versions).

Each version is one migration function in `db.py` and, for the rebuilds, one SQL file in
`python/posetrak/db/sql/migrations/`. The order puts every column a first-release component writes in
the first release, so the second and third releases only add.
