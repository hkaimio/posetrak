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

## Choosing among keypoint candidates

A sequence made from a detection run with *Keypoint candidates* (see
[Analyzing poses](analyzing-poses.md)) holds several maxima per keypoint. By default the tracker uses the
strongest one, as always. `posetrak track run ... --candidates` makes it choose, per camera and step,
among the candidates of each body keypoint, by how well they fit its prediction (the same assignment that
chooses among reflective dots, restricted to the keypoint's own candidates). Hands keep their refined-hand
keypoints, and the first frame still starts from the strongest maxima. Two options change the choice, each
independently of the other:

- `--strict-mask` (needs `--seg-run`): a candidate is used only while its tracklet lies on the tracked
  person's own mask; `--mask-min-fraction` and `--mask-window` set how much and over how many steps.
  It helps where one person is often hidden behind another, and costs the correct candidates of joints
  that really are under the other person's mask.
- `--multi-view-agreement`: a candidate is preferred when another camera's candidate for the same keypoint
  lies on its epipolar line (`--agreement-px`, `--agreement-factor`).

A candidate's cost is its distance to the prediction plus the log-determinant of the covariance it implies, so a
weak maximum, which implies a wide covariance, does not win on distance alone. `--kp-peak-weight` adds a reward for
a high peak (default 0, off) and `--kp-tracklet-multiplier` a preference for the candidate that continues the track
(tracklet) chosen for the same keypoint on the previous step (default 1, off). On the evaluation trial neither
changed the result noticeably, so both stay off unless you have measured a gain on your footage.

The settings are recorded with the run (`kp_candidates_enabled`, `kp_candidate_*` and the `dot_*` columns of its tracker
config), so two runs on one sequence differ in their configuration only. **Cost.** The shared assignment takes about 14 ms
per step for 90 candidates per camera and person, a 4-10% longer run, and the memory is the same to 1%; the mask
option adds another 20-25% (it decodes a mask per camera and step).

The choice is made per step from the tracker's prediction. On the multi-person evaluation trial it was about as good
as the strongest keypoint by itself (0.5 cm better for one person and 0.2 cm worse for the other; the offline
choice of the research prototype was 3-4 cm better, and the two options each helped); the measurements are in section 9 of the
keypoint candidates design (`docs/roadmap/features/keypoint-candidates/`), so try it on your footage before relying on it.

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
