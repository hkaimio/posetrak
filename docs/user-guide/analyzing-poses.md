# Analyzing poses (detection & segmentation)

*Draft — outline only.*

Before the tracker can run, Posetrak needs 2D body keypoints detected in
every camera's video, for the trial's frame range. There are two ways to
get there, and picking the right one for a given capture matters more
than almost any other single decision in the pipeline.

## Direct detection

- Runs YOLO (person detection) + RTMPose (keypoint estimation) directly
  on the video, then you assign the detected tracks to persons.
- Simple, fast, no extra setup step.
- Works well when performers don't cross paths in frame — the underlying
  person-tracker can lose or swap identities across an occlusion or a
  crossing, which direct detection has no way to correct after the fact.

## Segmentation-assisted detection

- For scenes where performers do cross paths (multi-person, close
  interaction, one occluding another), run a **segmentation** pass
  first: an interactive, mask-propagation-based tool (Cutie) that tracks
  exactly which pixels belong to which performer frame-to-frame, with
  your correction where it drifts.
- Requires manually initializing a mask for each performer near the
  start of the range you want segmented, then queuing forward/backward
  propagation from that initialization frame.
- More setup work, but immune to the identity-swap problem direct
  detection has — the mask, not a re-detected bounding box, is what's
  being tracked frame to frame.
- *(TBD: step-by-step walkthrough of the segmentation UI — mask
  initialization, Queue Forward/Backward, reviewing and correcting a
  propagated mask, and when a redetect is needed.)*

## Which to use

*(TBD: concrete guidance — "start with direct detection, only reach for
segmentation if X".)*

## See also

- [Your first capture](first-capture.md)
- [Tracking & troubleshooting](tracking-troubleshooting.md) — once poses
  are analyzed and the tracker has run.
