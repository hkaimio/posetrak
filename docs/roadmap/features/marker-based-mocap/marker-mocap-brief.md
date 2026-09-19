# Marker based motion capture in Posetrak

## Use cases

In priority order

- Tracking props
  - In simple cases, props can just be added post-mocap in animation software by
    attaching the prop model to the skeleton, so that the tracked person's mocap
    drives the prop. In more complex cases this is not feasible.
    - E.g. in aikido bokken (sword) practice, trajectory & blade angle are
      important — small tracking errors in hand pose are amplified by the long
      sword, and the grip usually changes during cuts. Better to track the
      sword as a separate object (and use its relative keypoint locations to
      also optimize hand pose)
  - This is likely the simplest case, as the prop can be assumed rigid, so
    only location & orientation need to be produced.

- Improving person tracking
  - Add markers to improve person tracking — more accurate measurements,
    keypoints at body locations that are not well covered by pose detection
    algorithms (spine, shoulder)

- Moving camera
  - Fiducial markers attached to the environment are already supported in
    extrinsics calibration. We could also track those during the actual trial
    to detect camera movement (either accidental or intentional) & recalculate
    extrinsics for the moved camera

## What kind of markers?

Design should be flexible

- Reflective dots — likely the type that can be used in fast movements
- ArUco or similar — identity is a plus, but these work only for
  static/slowly moving objects
- Other types

## Workflow

- Characterizing the prop/other object
  - We already have a file format for defining a calibration rig
    (python\app\setup\fiducial_markers.py)
  - Need a tool to capture geometry & markers from video/still images
  - After doing that, add the prop definition to the database and add it to
    the capture just like persons are added — the tracker then tries to
    detect it (additional tracking algorithms during the detection phase)
- How does this extend to a person with markers attached
  - My mental model: point/detect the markers in a single frame, then run
    automatic calibration during a test sequence where the person moves (the
    tracker follows the pose estimation algorithm and optimizes marker
    locations so that they match).
    - Does the user need to link markers to a specific joint coordinate
      frame, or could this be automated?
    - What about markers on a soft/flexible body part, e.g. the chest? These
      might be affected by multiple joints, so we might need a "weight map"?

## Technical questions

- For actual tracker markers should be very similar to markerless keypoint detections. Likely measurement noise will be smaller. Markers that return orintation might need special handling.
- How to detect — many fiducial marker algorithms run on CPU and are
  relatively slow. Also likely needs to run at full resolution
- How to assign detections without ID to actual markers? Should we have
  e.g. per-marker prediction before running the pose UKF, or combine this
  into the update step (optimize marker assignments to fit predicted
  positions)?
