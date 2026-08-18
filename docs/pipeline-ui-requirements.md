# Posetrak Pipeline UI — Requirements and Architecture

**Status note (2026-08-19):** pre-implementation requirements doc — the
five-separate-marimo-notebooks pipeline described below has since been
replaced by the unified, DB-backed GUI apps (`posetrak-setup`,
`posetrak-pose`, `posetrak-ui`) and CLI. "YOLO11" below refers to the
now-removed `ultralytics` dependency; the current detector is
`posetrak.detection.backends_rtmdet.YOLOXDetector` (see
`docs/license-analysis.md`). Kept for historical reference of the
original requirements analysis, not current architecture — see
`docs/architecture/` instead.

## 1. Current Pipeline Overview

The end-to-end pipeline from raw video to tracking output currently consists of five
separate manual steps, each in a different tool with no shared data model:

```
Raw videos  ──► [1] Person detection & pose extraction  (marimo notebook, rtmlib/YOLO/RTMPose)
                [2] Video synchronization                (marimo notebook, LED analysis)
                [3] Extrinsic calibration                (Pose2Sim tool, external)
                [4] posetrak track                       (C++ CLI)
                [5] Result visualization                 (Python scripts, Rerun)
```

Steps 1–3 produce files consumed by step 4, but the connections are implicit (matching
directory paths, manually copied file names).

---

## 2. Functional Requirements by Stage

### Stage 1 — Person Detection and Pose Extraction

**What the current tool does**
- Runs YOLO11 over the full video to produce person bounding-box timelines.
- Interactive "stitcher" UI: user selects which persons to track by name and merges
  fragmented detection timelines (the detector loses tracking across cuts/occlusions).
- Runs RTMPose on each named person's bounding-box timeline to extract keypoints.
- Interactive bounding-box editor: user can manually correct boxes when pose quality is
  poor; re-runs RTMPose on corrected regions.
- Exports OpenPose-format JSON (one file per camera per frame) for consumption by posetrak.

**Requirements**
- R1.1 — Process multiple cameras for the same shot in a single session (currently one
  notebook run per camera).
- R1.2 — Show pose confidence over time so the user can quickly spot frames needing
  correction without scrubbing through all frames.
- R1.3 — Support fisheye/non-pinhole video input: apply undistortion using camera
  intrinsics before or during pose estimation (currently assumes pre-rectified videos).
- R1.4 — Avoid re-running YOLO or RTMPose if the video has not changed; cache results
  keyed to the video file hash or path+mtime.
- R1.5 — Export directly to the session storage format (§2 of data-model-and-storage.md)
  rather than per-frame JSON files.
- R1.6 — Allow tuning confidence threshold and person selection without re-running the
  full pose model.

### Stage 2 — Camera Synchronization

**What the current tools do**

*Automatic (video_sync.py — primary):*
- User specifies a per-camera pixel ROI covering a blinking LED sync device.
- Reads the full video and extracts per-frame mean brightness change in the ROI.
- Detects blink events using z-score + peak detection (SciPy or NumPy fallback),
  subframe refinement by parabolic interpolation.
- Matches events across cameras with DTW + RANSAC affine fit.
- Falls back to cross-correlation if too few events detected.
- Outputs a time map `t_global = f(t_local)` for each camera; exports `sync_data.json`.

*Manual (sync_videos.py — fallback):*
- Shows all video feeds side by side in a PyQt6 window.
- User steps frames to find the sync event (e.g. clap, flash) and marks the sync frame
  per camera.
- Saves sync frame + FPS per camera to a YAML project file.

**Requirements**
- R2.1 — LED ROI selection must be interactive: user clicks/drags on a video frame to
  define the ROI; current approach requires editing hardcoded pixel coordinates.
- R2.2 — Show the brightness change signal plot and detected events so the user can
  verify quality before writing the sync file.
- R2.3 — Manual override: if automatic detection fails or there is no LED in the scene,
  fall back to the manual frame-stepping UI (the current PyQt sync_videos.py).
- R2.4 — Support cameras with different frame rates (handled by current tool — preserve).
- R2.5 — Support non-affine time maps for cameras with variable-rate clocks (handled by
  current PCHIP path — preserve).
- R2.6 — Write sync output to the session storage format.
- R2.7 — **Frame rate verification**: the RANSAC affine fit estimates the true camera fps
  implicitly via the `a` coefficient of `t_global = a × t_local + b`. If `a` deviates from
  1.0 by more than ~0.1%, report a warning and update the stored fps accordingly.  The
  current tool silently accepts a wrong stored fps (e.g. 118.88 Hz entered for a 120 Hz
  camera), which causes the NN event-matching window to be exceeded at late recording times
  and prevents RANSAC from fitting the true slope — producing up to 1.5s of accumulated
  drift over a 3-minute session.

**Known limitation — fps calibration errors (R2.7 background)**

The current sync algorithm assumes the stored fps value is accurate.  If it is wrong by
more than ~0.5 Hz, events at the end of a recording drift beyond the NN matching window
(1.0 s), so RANSAC only sees early-recording pairs and fits slope ≈ 1.0, masking the
error.  Approaches to make the system robust to this:

1. **Multiple rough-sync anchors**: allow the user to mark two or more widely-spaced
   anchor frames (not just one near the start).  A pair of anchors constrains both the
   offset *and* the slope of the affine map, making the rough offset accurate enough for
   NN matching even with a 1 Hz fps error over 3+ minutes.

2. **Structured LED code (PRBS/Manchester)**: replace the free-running blink with a
   pseudo-random binary sequence (PRBS) or Manchester-coded timestamp signal.  The global
   time can then be read directly from the LED pattern at any point in the recording,
   without requiring cross-camera event matching or a user-supplied anchor frame.

3. **Timecode from video metadata**: many cameras embed SMPTE timecode or creation
   timestamps in the MP4 container.  Reading this provides an absolute time reference and
   can resolve rough offsets without any manual anchor, as long as the camera clocks are
   set correctly.

### Stage 3 — Extrinsic Calibration

Currently handled by Pose2Sim's OpenCV-based calibration pipeline (checkerboard or
ChArUco).  This works well enough to defer a custom UI.

**Requirements (future)**
- R3.1 — Accept the calibration output from Pose2Sim and import it into the session storage.
- R3.2 — Visualize the camera positions in 3D so user can spot obvious calibration errors
  before running tracking.
- R3.3 — Support fisheye camera models (kb4 / OpenCV fisheye) in storage and in the
  posetrak projection pipeline.

### Stage 4 — Running the Tracker

The posetrak C++ CLI already works.

**Requirements**
- R4.1 — Accept a session file path and observation sequence ID as primary inputs instead
  of separate TOML path collection.
- R4.2 — Progress reporting: fractional progress + estimated time remaining streamed to
  stdout or a log file.
- R4.3 — Write tracking results into the session file (not a separate CSV).

### Stage 5 — Result Visualization and Export

**Requirements**
- R5.1 — Overlay tracking skeleton on original video frames (for validation).
- R5.2 — Export to BVH and/or FBX.
- R5.3 — Time-aligned plots of joint angles as a quality check.
- R5.4 — Support for multi-person results in the same session.

---

## 3. Cross-Cutting Requirements

- **CC1 — Language-independent storage**: All intermediate and final data must be in a
  format readable from both Python (analysis/UI tools) and C++ (tracker). See
  data-model-and-storage.md; SQLite + BLOB packing is the chosen format.
- **CC2 — Video read performance**: The current OpenCV-based single-threaded frame reader
  is the dominant bottleneck. All video-reading stages (pose extraction, LED sync
  extraction) must be replaced with a faster backend. See §4 below.
- **CC3 — Non-pinhole camera support**: The pipeline must carry fisheye intrinsic
  parameters (kb4 or OpenCV fisheye model) end-to-end. Undistortion must happen before
  pose estimation and before the UKF projection step.
- **CC4 — Reproducibility**: every processing step that writes to the session must record
  what inputs and parameters were used (model name, version, thresholds).

---

## 4. Video Decoding Performance

### Problem

OpenCV's `VideoCapture` is the current bottleneck in both the pose extraction and sync
pipelines. It uses software-only decoding (FFmpeg), is single-threaded, and requires
random-seeks (`CAP_PROP_POS_FRAMES`) that defeat codec GOP structure.

For a 10-minute 4K 120 fps video this means:
- ~72 000 frames to decode
- Sequential software H.264/H.265 decode: ~30–50 fps throughput → **24–40 min per camera**
- LED ROI extraction needs only a 20×20 px crop from each frame — a massive waste

### Option A — PyAV / av (recommended for Python tools)

PyAV wraps libavcodec/libavformat directly and exposes:
- Hardware-accelerated decode (`cuda`, `videotoolbox`, `vaapi` depending on platform)
- Generator-based frame iteration: no random seek needed
- Direct numpy conversion with zero-copy (`frame.to_ndarray(format='bgr24')`)

```python
import av
with av.open("cam1.mp4") as container:
    for frame in container.decode(video=0):
        # frame.to_ndarray() is already a numpy array — no copy
        roi = frame.to_ndarray(format='bgr24')[y1:y2, x1:x2]
        process(roi, frame.index)
```

For the LED sync pipeline this alone gives **8–20× throughput improvement** because the
ROI crop is done on the decoded frame in Python without any OpenCV overhead, and hardware
decode is used when available.

For the pose pipeline, PyAV feeds batches of decoded frames directly to the GPU inference
pipeline without the OpenCV intermediate copy.

**Library:** `av` (PyAV), PyPI, BSD-2-Clause. No system library installation needed on
Linux/macOS; on Windows via conda-forge or pip wheel.

### Option B — NVIDIA Video Codec SDK / NVDEC (for high-throughput pose extraction)

When the processing machine has an NVIDIA GPU already running RTMPose, NVDEC can be used
to decode video entirely on the GPU, passing decoded frames directly to the CUDA inference
pipeline with no CPU memory transfer:

- Python: `PyNvVideoCodec` (NVIDIA), or `torchvision.io.VideoReader` with `backend='cuda'`
- Throughput: typically **5–10× faster than CPU decode**, saturating the GPU inference pipeline

This is the architecture used in production video analytics systems.  Requires CUDA ≥ 11.0
and an NVIDIA driver with NVDEC.

**Recommendation:** use PyAV as the baseline (works everywhere); add NVDEC path as
a conditional fast path when `torch.cuda.is_available()` and the GPU supports NVDEC.

### Option C — FFmpeg subprocess with pipe

```bash
ffmpeg -i cam1.mp4 -vf "crop=20:20:x:y" -f rawvideo -pix_fmt gray pipe:1 \
  | numpy_processing
```

Leverages FFmpeg's native multithreaded decode and built-in crop filter, pushing only the
needed bytes over a pipe. Works anywhere FFmpeg is installed; no Python video library
needed. Throughput matches or exceeds PyAV for sequential reads. Main downside: awkward
for interactive / frame-index-addressable access.

Good fit for the LED brightness extraction pipeline which is purely sequential.

---

## 5. Fisheye / Non-Pinhole Camera Support

### Problem

Action cameras (GoPro, Insta360, etc.) use fisheye lenses with strong radial distortion.
The current pipeline assumes pre-rectified videos, which means a separate preprocessing step
and a second storage of the video data.

### Requirements

The pipeline should work directly on the original fisheye videos:

1. **Intrinsics storage**: store the camera's native (distorted) intrinsics — pinhole +
   Brown-Conrady for mild fisheye, or OpenCV fisheye (kb4) for strong fisheye — in the
   session registry. The current `Observation` struct already stores both distorted and
   undistorted coordinates.

2. **Undistortion on decode**: during pose extraction, apply per-frame undistortion of the
   bounding boxes / 2D keypoints (not the whole frame) using
   `cv2.undistortPoints()` / `cv2.fisheye.undistortPoints()`. This is cheap and does not
   require creating a rectified video.

3. **UKF projection**: the UKF already projects 3D marker positions to 2D undistorted
   pixels. Ensure the fisheye model is passed through and used in the `Camera::project()`
   call.

4. **Calibration**: Pose2Sim's OpenCV fisheye calibration produces `k1,k2,k3,k4` for the
   kb4 model; these are stored in the intrinsics record alongside the pinhole params.

---

## 6. Proposed Application Architecture

### Option A — Marimo Notebook Suite (recommended near-term)

Keep the existing marimo-based workflow; fix the seams between notebooks.

```
Stage 1: pose_extraction.py    (marimo)   ─┐
Stage 2: video_sync.py         (marimo)   ─┤─► session.db  (SQLite)  ─► posetrak
Stage 3: import_calibration.py (marimo)   ─┘
Stage 5: results_viewer.py     (marimo)   ◄── session.db
```

Each notebook reads from and writes to a shared `session.db` file.  A small shared Python
library (`posetrak_session`) provides the SQLite schema, read/write helpers, and the
`SessionFile` context manager used by all notebooks.

**Pros:**
- Minimal refactoring of existing working code
- Marimo's reactive cells handle parameter changes gracefully (change threshold → re-run
  only dependent cells)
- Good fit for exploratory/debugging use where notebook cells need independent control
- No UI framework to maintain

**Cons:**
- Not a unified application; user still opens separate notebooks per stage
- Marimo browser UI is less snappy than a native app for large video scrubbing

**Fast-path addition:** replace `cv2.VideoCapture` with PyAV in the inner loops of both
`pose_extraction.py` and `video_sync.py`.

### Option B — Single Marimo App with Tab Navigation

A single marimo file with tabs:

```
┌─────────────────────────────────────────────────────┐
│ Session: 2026-03-01-gym                [Session DB] │
├────────────┬────────────┬────────────┬──────────────┤
│  1. Sync   │ 2. Pose    │ 3. Run     │  4. Review   │
│            │ Extraction │  Tracker   │              │
└────────────┴────────────┴────────────┴──────────────┘
```

Each tab is a marimo `mo.ui` panel.  State is shared through the session DB; tabs are
independent reactive DAGs.

**Pros:** Single entry point; shared session context is explicit; tabs make the pipeline
linear.

**Cons:** Marimo apps are browser-based with WebSocket round-trips for every UI interaction;
video scrubbing at 30 fps is not feasible over this channel without custom JS.

### Option C — Native Desktop App (Qt + Python)

A PyQt6 / PySide6 application with:
- Video player widget (QMediaPlayer or custom OpenGL/Vulkan texture upload from PyAV)
- Side panels for each pipeline stage
- Direct frame access without browser round-trips

**Pros:** Best performance for interactive video work; can run GPU inference in background
threads; works offline.

**Cons:** Significantly more engineering effort; Qt video stack on Linux can be fragile;
distribution packaging is harder.

**Verdict:** Worth revisiting once the data model and CLI tool are stable.  Not the
right immediate investment.

---

## 7. Recommendation and Phased Plan

### Phase P1 — Pose extraction notebook rewrite (1–2 weeks)

The pose extraction step is the most time-consuming and error-prone part of the current
workflow.  This phase makes it faster and multi-camera-capable.

**P1a — PyAV integration (performance)**
- Replace `cv2.VideoCapture` in the frame-reading loop with PyAV
- Enable hardware-accelerated decode (VAAPI on Linux, VideoToolbox on macOS, CUDA/NVDEC
  when a CUDA GPU is present and the codec supports it)
- Benchmark and document throughput improvement vs. the current OpenCV path
- Add NVDEC conditional fast path behind a `torch.cuda.is_available()` capability check

**P1b — Port to Marimo**
- Rewrite the existing notebook as a marimo app (`notebooks/pose_extraction.py`)
- Reactive parameter cells: YOLO model, RTMPose model, confidence threshold
- Person-stitcher UI using `mo.ui` components (replaces the current IPython widgets)
- Confidence-over-time plot (R1.2): sparkline per camera so bad regions are immediately
  visible without frame scrubbing
- Result caching keyed to video path + mtime (R1.4): skip YOLO/RTMPose if results are
  already on disk

**P1c — Multi-camera simultaneous processing (R1.1)**
- Accept a list of video files (one per camera) for the same shot
- Run YOLO + stitcher + RTMPose for all cameras in the same marimo session
- Person identity is assigned once and propagated across cameras
- Output: per-camera OpenPose JSON directories (existing format, preserving compatibility
  with the current posetrak CLI)

### Phase P2 — video_sync.py improvements (1 week)

- Replace `cv2.VideoCapture` with PyAV in the LED brightness extraction loop (same
  performance motivation as P1a; likely the larger win because it reads every frame)
- Interactive ROI selection (R2.1): clickable region selector on a video thumbnail inside
  marimo; replaces hardcoded pixel coordinates
- Show brightness signal + detected events plot before writing output (R2.2)
- Manual frame-stepping fallback when automatic detection fails (R2.3, preserving current
  sync_videos.py behaviour)
- Port to marimo (`notebooks/video_sync.py`) with reactive parameter cells

### Phase P3 — Session file (1–2 weeks)

Implement `posetrak_session` Python library:
- SQLite schema matching the data model in data-model-and-storage.md
- `SessionFile` class: `open()`, `write_observations()`, `write_sync()`,
  `write_extrinsics()`, `read_observations()`, `read_tracking_results()`
- Importers for existing formats: per-frame JSON → observations BLOB, TOML calibration →
  session cameras, existing sync JSON → sync table
- Update P1 and P2 notebooks to write to `session.db` instead of per-frame files

### Phase P4 — Fisheye support (1–2 weeks)

- Extend intrinsics schema to carry `fisheye_model: kb4 | brown_conrady` + coefficients
- Add `undistort_points()` wrapper that dispatches to the right OpenCV call
- Update `Camera::project()` in C++ to handle kb4 model
