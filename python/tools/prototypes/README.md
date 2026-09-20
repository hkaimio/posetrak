# Prototypes

Exploratory scripts written while developing marker-based mocap. Each one
answered a question on real data; the answer was then either absorbed into
the `posetrak` package or superseded by a different design. They are kept
because a few are still the only way to reproduce a result, not because they
are maintained. Do not add new workflows here — a script a user needs to run
belongs in the `posetrak` CLI with tests
(see `docs/roadmap/features/marker-based-mocap/productization-architecture-and-plan.md` §4).

Scripts import each other as `tools.<module>` (`python/` on `sys.path`), so
run them from anywhere with `python python/tools/prototypes/<name>.py`.

Usage examples use `/path/to/...` placeholders; the capture data they were
run against lives outside the repository.

## Still needed

Three prototypes were meant to live here but remain in `python/tools/`
because a tool that is still the only working path to its workflow imports
them as a library. Each moves when the package code that replaces that tool
lands.

| Script (in `python/tools/`) | Imported by | Moves with |
|---|---|---|
| `prototype_multi_camera_fusion.py` | `fit_calibrated_attachment_set.py`, `label_tracklet_groups_gui.py`, several prototypes | `posetrak marker-set` (tracklet grouping and attachment set fitting) |
| `prototype_fk_marker_prediction.py` | `build_tracklet_groups.py`, `label_tracklet_groups_gui.py`, `gt/eval_fk_prediction.py` | `posetrak marker-set` |
| `prototype_person_marker_assignment.py` | `render_dot_detection_overview_video.py` | `posetrak detect render` |

## Absorbed or reference

| Script | What it established | Where it went |
|---|---|---|
| `prototype_dot_blob_detector.py` | Manual tuning tool for reflective-dot detection | `posetrak/detection/dot_blob_detector.py` is the production detector; the script remains a tuning aid |
| `prototype_streak_detector.py` | Long motion-blur streaks can be detected and linked into per-camera tracklets | Ported to `dot_blob_detector.py` (streak mode) and `dot_tracklet.py` |
| `prototype_motion_gated_linker.py` | A per-track Kalman filter with Mahalanobis gating separates identity switches that the fixed-radius linker merged | Ported to `MotionGatedLinker` in `posetrak/detection/dot_tracklet.py` |
| `prototype_ball_tracking.py` | A single reflective ball can be positioned in 3D from per-camera 2D tracks, with a cross-camera consistency check | The consistency check becomes the cross-camera corroboration cost modifier (plan §3.6.2, WS2 item 3) |
| `prototype_shin_rigid_cluster_fit.py` | Marker offsets on one segment can be fitted from that segment's own triangulated rigid cluster, independent of the tracked skeleton's bone lengths, so a weak marker is carried by the stronger ones (skeleton-scaling design §2.1) | Skeleton scaling from rigid clusters (plan WS6 item 4) |
| `prototype_pen_band_tracking.py`, `prototype_track_pen_trajectory.py` | 6-DOF trajectory of a pen from its ArUco tags and reflective bands | Feeds `blender/export_pen_pad_to_blender.py`; no package equivalent planned |
| `prototype_calibrate_pen_and_pad.py` | Rigid-body geometry calibration, hardcoded to one capture's pen and pad | General paths are `posetrak marker-body calibrate` (fixed cameras) and `calibrate-video` (one moving camera) |
| `calibrate_harness_from_orbit.py` | Recalibrating a rigid body from one moving camera orbiting the stationary rig, anchored by static markers | Absorbed by `posetrak marker-body calibrate-video` (`posetrak/calibration/video_marker_body.py`), which needs no old body for its dots; delete once it has been used on a second prop |

## Superseded

`superseded/` holds the person-marker assignment experiments that the marker
catalog and assignment redesign replaced
(`marker-catalog-and-assignment-redesign.md` §0), plus the Cutie-based ball
localization that the imported-2D-track path replaced. They are kept while
their replacement is built, and **each is deleted in the change that
completes the workstream in the last column** — this table is that checklist.

| Script | Retired by |
|---|---|
| `prototype_hybrid_person_marker_assignment.py` | WS1 item 3: `marker-set group-tracklets` / `fit` |
| `prototype_fused_person_marker_assignment.py` | WS1 item 3 |
| `prototype_marker_normal_assignment.py` | WS1 item 3 (backface culling in tracking-time assignment: WS2 item 1) |
| `prototype_marker_person_filter.py` | WS1 item 3 |
| `prototype_tracklet_smoothed_assignment.py` | WS1 item 3 |
| `render_person_marker_assignment_video.py` | WS1 item 3 |
| `render_hybrid_assignment_video.py` | WS1 item 3 |
| `render_marker_normal_debug_frame.py` | WS1 item 3 |
| `prototype_ball_cutie_segmentation.py` | WS1 item 2: `posetrak detect import-2d` |
| `cutie_segment_ball_all_cameras.py` | WS1 item 2 |

The retiring workstream for each row is a reading of the plan, not something
the plan states per file; correct it here if the mapping is wrong.
