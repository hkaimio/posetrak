# Tracking & troubleshooting

*Draft — outline only.*

The tracker rarely gets perfect results on the first run. This page
covers the tools available for figuring out *why* a run looks wrong and
fixing it, without re-capturing.

## Adjusting the skeleton

- The tracker fits a fixed skeleton (bone lengths, joint limits) to the
  observed keypoints. If the skeleton's proportions don't match the
  performer, the fit will be systematically off even with perfect
  detections.
- *(TBD: how to adjust skeleton dimensions in the UI/config — manual
  editing vs. any automatic bone-length calibration pass, see the
  `scale` CLI subcommand mentioned in CLAUDE.md.)*

## Adjusting tracker parameters

- *(TBD — no user-facing guide yet. See
  `docs/skeleton-scaling-calibration-design.md` and
  `docs/per-frame-measurement-noise-design.md` for the underlying design
  if you need to go deeper before this section is written.)*

## Diagnosing a run (MCP diagnostic server)

- For tracking runs with a session database, `python/app/mcp/` exposes a
  read-only diagnostic server (see CLAUDE.md's "MCP Diagnostic Server"
  section) with tools for per-step filter statistics (NIS/DOF,
  covariance condition number), per-camera coverage, observation gaps
  (actual vs. predicted pixel position), and camera geometry/parallax —
  useful for narrowing down *which* camera, marker, or time range is
  responsible for a bad stretch of tracking before diving into keypoint
  edits.

## Editing keypoints

- If a particular camera's detections are wrong or missing for some
  frames (but at least some other cameras got it right), Posetrak has an
  in-app keypoint editor in the main viewer rather than requiring a full
  re-detect.
- *(TBD: walkthrough of edit mode itself — selecting a frame/camera/
  keypoint, dragging vs. marking as outlier, how edits interact with a
  later automated re-detect.)*

## See also

- [Your first capture](first-capture.md)
- [Analyzing poses](analyzing-poses.md)
