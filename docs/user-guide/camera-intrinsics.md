# Camera intrinsics calibration

Intrinsics calibration finds a camera's **focal length** (`fx`, `fy`),
**principal point** (`cx`, `cy`), and **lens distortion coefficients** —
together, the parameters that describe how a 3D point in front of the
camera projects onto a specific pixel in its image, independent of where
the camera is or which way it's pointed (that part is [extrinsics
calibration](extrinsics-calibration.md)). Posetrak needs intrinsics for
every camera before extrinsics calibration or tracking can use it at
all: triangulating 2D keypoint detections into a 3D estimate, and later
projecting the skeleton's 3D joints back into each camera's image to
compare against detections, both depend on knowing this per-camera
mapping precisely.

If you want the underlying theory rather than just how to run it in
Posetrak, OpenCV's own docs cover it well — see the [camera calibration
tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
and the [`calib3d` module reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
for the standard pinhole + radial-tangential distortion model, or the
[fisheye model reference](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
if you're calibrating a very-wide-FOV lens (see "Distortion model,"
below). Posetrak calls straight into OpenCV's calibration routines under
the hood, so all of that applies directly.

Intrinsics are calibrated per **camera mode** (a camera + resolution +
frame rate combination) rather than per physical camera, since changing
resolution or frame rate changes the intrinsics on many cameras. A
camera can have several modes registered, and each mode can have several
calibrations — one marked as the mode's **default**, used automatically
whenever a video is later assigned that mode (see "Which calibration
gets used," below).

## Prerequisites

- **A checkerboard or ChArUco calibration target**, printed flat and
  rigid — any warp in the target directly corrupts the result. You'll
  need to know its physical square size in meters; that's what fixes
  the real-world scale of the calibration (and, downstream, of your
  whole capture).
- **Footage of that target in the exact camera mode you'll calibrate**
  (same resolution and frame rate you'll actually shoot with) — a video
  where you move the target around the frame, or a set of still images.
  Cover different positions, angles, and distances, and make sure the
  target reaches into the corners of the frame at some point — that's
  where distortion is largest and where the calibration most needs data
  to constrain it. Posetrak's pipeline warns if it finds fewer than 10
  usable detections; more, spread across the frame, is better.
- **Decide the distortion model up front**: the standard
  radial-tangential (pinhole) model fits ordinary lenses; a **fisheye /
  equidistant** model exists for very-wide-FOV lenses (action cameras
  with 150°+ fields of view) where the standard model can't fit the
  distortion well. Pick one when you run the calibration — you can't mix
  models for the same calibration.

## Workflow (UI)

1. Open **Cameras → Manage cameras…** — the Camera Registry. Its left
   pane is a tree of camera models and their registered modes.
2. Select (or **+ Add Mode** to create) the mode you're calibrating,
   then **Edit…** to open it. The mode's dialog lists its resolution,
   fps, and any calibrations it already has (date, RMS error, notes, and
   a `●` marking the current default).
3. Click **Calibrate from video…**. In the dialog that opens:
   - **Input** — browse to your calibration video or an image directory.
   - **Calibration pattern** — Checkerboard or ChArUco board. Note that
     rows/cols mean different things for each: checkerboard wants
     *internal corner* counts (one less than the number of squares per
     side), ChArUco wants the number of *full squares* — the dialog
     relabels the fields when you switch, since getting this wrong is a
     common way to fail detection entirely. Square size is in meters.
     ChArUco additionally needs a marker-size ratio and an ArUco
     dictionary (`DICT_4X4_100` by default).
   - **Camera model** — check **Fisheye** if you decided on that model
     above; otherwise leave the standard model selected. Notes here are
     just stored alongside the saved calibration for your own reference.
   - **Frame selection** (video input only) — Posetrak doesn't run
     detection on every frame; it first scans for sharp ones (a
     Laplacian-variance threshold, checked over a sliding window), to
     avoid wasting time on motion-blurred frames that would fail
     detection anyway. **Frame skip** processes only every Nth frame of
     the scan if you want it faster; the defaults are reasonable
     starting points.
4. **Run Calibration**. This runs in the background — a log pane shows
   progress (frame scan, corner detection, the calibration solve
   itself), and you can **Cancel** a scan/detection in progress. Once
   done, you get the RMS reprojection error (in pixels), resolution,
   `fx`/`fy`/`cx`/`cy`, and the distortion coefficients.
5. **Preview undistortion…** (needs a video input) lets you scrub
   through the footage toggling between the original and
   undistorted view. This is the real sanity check: straight lines in
   the physical world (a straight table edge, a door frame) should look
   straight once undistorted. If they still visibly bow, or the image
   warps in an obviously wrong way, something's off — usually the wrong
   rows/cols for the pattern, or too few/too clustered detections.
6. **Save calibration** writes it to this mode. The *first* calibration
   saved for a mode becomes its default automatically; if you calibrate
   again later (different footage, tried fisheye vs. standard, etc.),
   use **Set as default** on whichever entry in the calibrations list you
   want future videos to use — saving a new one doesn't replace the
   default on its own.

Already have intrinsics from somewhere else? **Import calibration…**
(next to "Calibrate from video…") offers the same two options: import an
HDF5 file (the format the command-line tool below also writes, so you
can calibrate offline and bring the result in later), or **Enter
manually** if you already know `fx`/`fy`/`cx`/`cy` and the distortion
coefficients from some other calibration process.

### Which calibration gets used

A mode's **default** calibration is what actually gets attached to a
video: when you assign a camera mode to a video while setting up a
capture, Posetrak looks up that mode's default calibration and links it
automatically (shown as "calib ✓" next to the mode picker at that point).
Changing a mode's default later does not retroactively change videos
already assigned that mode — it only affects captures you set up
afterward.

## Command-line

The same underlying pipeline is available standalone as
`python/pipeline/calibration/calibrate_intrinsics.py`, useful for
scripting or batch/offline calibration — its output HDF5 file is exactly
what the GUI's "Import calibration… → From HDF5 file" expects.

```
python calibrate_intrinsics.py INPUT_PATH \
    --camera-name "GoPro HERO11" --camera-mode "4K" \
    [--rows 7] [--cols 10] \
    [--output-file calibration.h5] [--output-dir results/] \
    [--window 10] [--threshold 0.8] [--skip 1] \
    [--global-sharpness-metric] [--fisheye]
```

`INPUT_PATH` is a video file or a directory of images, same as the GUI.
`--rows`/`--cols` are internal-corner counts (checkerboard only — this
CLI tool doesn't currently support ChArUco). `--fisheye` selects the
fisheye distortion model. `--output-dir`, if given, also saves the
detected (and undistorted) checkerboard images, useful for eyeballing
detection quality across a whole run. Run with `--help` for the full,
current option list.

## See also

- [Your first capture](first-capture.md) — where this fits in the
  overall workflow.
- [Extrinsics calibration](extrinsics-calibration.md) — the next step,
  which assumes intrinsics are already done.

---

*Screenshots still needed: the Camera Registry tree, the mode dialog's
calibrations list, the "Calibrate from video…" dialog mid-run, and the
undistortion preview toggle.*
