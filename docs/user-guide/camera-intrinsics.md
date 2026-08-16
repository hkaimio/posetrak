# Camera intrinsics calibration

*Draft — outline only, needs a full walkthrough + screenshots.*

Intrinsics calibration finds each camera's focal length, principal point,
and lens distortion coefficients. Posetrak needs these before extrinsics
calibration or tracking can use a camera at all — 2D keypoint detections
can't be related to 3D positions without them.

Intrinsics are calibrated per **camera mode** (a camera + resolution +
frame rate combination) rather than per physical camera, since changing
resolution or frame rate changes the intrinsics on many cameras. A camera
can have several modes registered, each with its own calibration(s).

## Prerequisites

- A checkerboard or ChArUco calibration target, printed flat and rigid.
- A short video (or set of still images) of that target, filmed with the
  camera in the exact mode (resolution/fps) you'll use for capture —
  moved around to cover different positions, angles, and distances in
  frame.

## Workflow

1. Open **Cameras → Manage cameras…** (the Camera Registry).
2. Select or create the camera mode you're calibrating.
3. **Calibrate from video…** — point it at the checkerboard/ChArUco
   footage; it detects the pattern across sampled frames and solves for
   intrinsics, with an undistortion preview before you save.
   *(TBD: how much coverage is "enough", how to read the reported error,
   what a bad calibration looks like in the preview.)*
4. Alternatively, **Import calibration…** an intrinsics file produced
   outside Posetrak.
5. If a mode has more than one calibration, **Set as default** picks
   which one new captures use.

## Command-line

`python/pipeline/calibration/calibrate_intrinsics.py` can also be run
standalone against a video or a directory of images — useful for
batch/offline calibration. *(TBD: document its CLI flags here.)*

## See also

- [Your first capture](first-capture.md) — where this fits in the
  overall workflow.
- [Extrinsics calibration](extrinsics-calibration.md) — the next step,
  which assumes intrinsics are already done.
