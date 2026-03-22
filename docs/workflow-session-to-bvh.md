# Workflow: Raw videos + HDF5 intrinsics → BVH

This document covers the full pipeline starting from a completed mocap recording
session (videos per camera + pre-existing HDF5 intrinsics files) through to a
BVH animation export.

---

## Prerequisites

| What you have | Where it lives |
|---|---|
| Camera videos | `/mnt/d/mocap/<session>/videos/cam1.mp4`, `cam2.mp4`, … |
| HDF5 intrinsics (one per camera) | `/mnt/d/mocap/calibration/cam1.h5`, `cam2.h5`, … |
| Skeleton YAML | e.g. `tests/harri-skeleton.yaml` |
| Tracker TOML config | e.g. `tests/regress.toml` (copy and edit for each session) |

**Software** (must be accessible):

| Tool | Repo | Used for |
|---|---|---|
| `posetrak-db` | this repo | all DB import/export commands |
| `posetrak track` | this repo (C++ build) | tracker |
| `calibrate_extrinsics.py` | `pose2sim-preprocess` | extrinsics TOML from a calibration frame |
| `video_sync.py` | `rtmlib/harritests` | LED-based sync JSON |
| `pose_extraction.py` | `rtmlib/harritests` | YOLO + RTMpose → OpenPose JSON |
| `export_bvh.py` | this repo (`python/tools/`) | convert tracker output to BVH |

---

## Part 1 — One-time registry setup

Do this once per machine / project. Skip if the registry already has your cameras.

```bash
# Create the registry DB (default: ~/.posetrak/registry.db)
posetrak-db registry init

# Register the camera hardware model
posetrak-db camera-model add \
    --manufacturer "GoPro" \
    --model-name "Hero 12 Black"
# → prints model_id (save it)

# Register the capture mode (resolution + fps you actually record at)
posetrak-db camera-mode add \
    --model-id <model_id> \
    --width 3840 --height 2160 --fps 120
# → prints mode_id (save it)

# Register a second mode if you use multiple resolutions
posetrak-db camera-mode add \
    --model-id <model_id> \
    --width 1920 --height 1080 --fps 120
```

### Register camera instances

Camera *instances* (individual physical units) represent your actual cameras.
Register each one once — the labels you assign here (`cam1`, `cam2`, …) are
matched automatically when importing session YAMLs.

```bash
posetrak-db camera-instance add \
    --model-id <model_id> \
    --label cam1 \
    --serial SN12345
# → camera_instance_id: <inst1_id>  label='cam1'

posetrak-db camera-instance add \
    --model-id <model_id> \
    --label cam2 \
    --serial SN12346
# → camera_instance_id: <inst2_id>  label='cam2'
```

List registered instances at any time:

```bash
posetrak-db camera-instance list
# or with full calibration history:
posetrak-db camera-instance show <inst1_id>
```

---

## Part 2 — Import HDF5 intrinsics

Do this whenever you have a fresh intrinsics calibration run (new HDF5 files).
The session YAML importer will automatically use the most recent intrinsics for
each camera mode.

```bash
posetrak-db calib import-h5 /mnt/d/mocap/calibration/cam1.h5 \
    --camera-mode <mode_id>
# → intrinsics_id: <new_intr1_id>

posetrak-db calib import-h5 /mnt/d/mocap/calibration/cam2.h5 \
    --camera-mode <mode_id>
# → intrinsics_id: <new_intr2_id>
```

Use `--no-maps` if storage space is a concern (skips the ~3 MB undistortion maps).

---

## Part 3 — Extract undistorted video clips

`setup_project.py` reads a project YAML (produced by `sync_videos.py` or written
manually), loads undistortion maps, and writes a trimmed, undistorted clip per
scene per camera.  These clips are the inputs to pose extraction (Part 6).

The project YAML used here has a different format from the session YAML in Part 4
— it is the older format with a flat `cameras` list and `scenes` with
`start_frame`/`end_frame`:

```yaml
# /mnt/d/mocap/2026-03-22-my-session/setup_project.yaml
path: /mnt/d/mocap/2026-03-22-my-session
ref_camera: cam1

cameras:
  - name: cam1
    path: /mnt/d/mocap/2026-03-22-my-session/raw/cam1.mp4
    fps: 120
    sync_frame: 5678
    calib:
      intrinsics: <new_intr1_id>   # intrinsics_calibration_id UUID from Part 2
      extrinsics:
        frame: 6100                # frame to extract for extrinsics calibration

  - name: cam2
    path: /mnt/d/mocap/2026-03-22-my-session/raw/cam2.mp4
    fps: 120
    sync_frame: 5681
    calib:
      intrinsics: <new_intr2_id>
      extrinsics:
        frame: 6100

scenes:
  - name: take1
    start_frame: 6000
    end_frame: 8400
  - name: take2
    start_frame: 10000
    end_frame: 12000
```

When `calib.intrinsics` is a UUID, pass `--registry` so the script can load
the undistortion maps from the DB:

```bash
python python/pipeline/calibration/setup_project.py \
    /mnt/d/mocap/2026-03-22-my-session/setup_project.yaml \
    --registry ~/.posetrak/registry.db

# → extracts per-scene clips with undistortion applied:
#   take1/videos/cam1.mp4  take1/videos/cam2.mp4
#   take2/videos/cam1.mp4  take2/videos/cam2.mp4
# → saves extrinsics calibration frames:
#   calibration/extrinsics/ext_cam1_ext/frame_NNNN.png
#   calibration/extrinsics/ext_cam2_ext/frame_NNNN.png
```

A file path (`.h5` or `.yaml`) is still accepted for `calib.intrinsics` if you
have not imported intrinsics into the DB yet.

---

## Part 4 — Session setup

Each recording session gets its own session database.

### 3a. Write a project YAML

Create `project.yaml` in the session directory.  Use the instance IDs printed
in Part 1 and the actual sync frames from the LED flash (rough values are OK
here — they will be replaced in step 3c).

```yaml
# /mnt/d/mocap/2026-03-22-my-session/project.yaml

name: "2026-03-22-my-session"
location: "gym"
recorded_at: "2026-03-22"

cameras:
  cam1:
    video_path: "/mnt/d/mocap/2026-03-22-my-session/videos/cam1.mp4"
    fps: 120.0
    sync_frame: 5678          # rough estimate — replaced by LED sync in step 3c
    camera_instance_id: <inst1_id>
    camera_mode_id: <mode_id>
    # intrinsics_calibration_id: omit → auto-picks latest for this mode

  cam2:
    video_path: "/mnt/d/mocap/2026-03-22-my-session/videos/cam2.mp4"
    fps: 120.0
    sync_frame: 5681          # rough estimate
    camera_instance_id: <inst2_id>
    camera_mode_id: <mode_id>

scenes:
  - label: "take1"
    cameras:
      cam1:
        first_frame: 6000
        last_frame: 8400      # 20 s at 120 fps
      cam2:
        first_frame: 6003
        last_frame: 8403

  - label: "take2"
    cameras:
      cam1:
        first_frame: 10000
        last_frame: 12000
      cam2:
        first_frame: 10003
        last_frame: 12003
```

### 3b. Import the YAML

```bash
SESSION_DIR=/mnt/d/mocap/2026-03-22-my-session

posetrak-db session import-yaml "$SESSION_DIR/project.yaml" \
    --session-db "$SESSION_DIR/session.db"
# → session_id: <session_id>
#   camera cam1: instance=<inst1_id>
#   camera cam2: instance=<inst2_id>
#   shot "take1": id=<shot1_id>  sync_config=<rough_sync1_id>
#   shot "take2": id=<shot2_id>  sync_config=<rough_sync2_id>
```

Use `--dry-run` first to check that all camera lookups resolve correctly.

---

## Part 5 — Extrinsics calibration

Run `calibrate_extrinsics.py` on a checkerboard frame captured during (or close
to) the recording session, when the camera rig was in place.

```bash
python calibrate_extrinsics.py \
    --input "$SESSION_DIR/calib_frame/" \
    --output "$SESSION_DIR/Calib_scene.toml"
```

Import and link to each shot:

```bash
# Get the session_id from the session DB
SESSION_ID=$(posetrak-db session list --session-db "$SESSION_DIR/session.db" | awk 'NR==3{print $1}')

posetrak-db extrinsics import \
    --session-db "$SESSION_DIR/session.db" \
    --calib "$SESSION_DIR/Calib_scene.toml" \
    --session $SESSION_ID \
    --camera-instance cam1=<inst1_id> --camera-instance cam2=<inst2_id> \
    --shot <shot1_id>
# → extrinsic_calibration_id: <ext_id>
#   take1 is now linked to this extrinsic calibration

# Link the same extrinsics to take2 (if same rig position)
posetrak-db extrinsics import \
    --session-db "$SESSION_DIR/session.db" \
    --calib "$SESSION_DIR/Calib_scene.toml" \
    --session $SESSION_ID \
    --camera-instance cam1=<inst1_id> --camera-instance cam2=<inst2_id> \
    --shot <shot2_id>
```

---

## Part 6 — Video synchronisation

Run `video_sync.py` to detect the LED sync flash and produce `sync_data.json`.

```bash
python video_sync.py \
    "$SESSION_DIR/videos/cam1.mp4" \
    "$SESSION_DIR/videos/cam2.mp4" \
    --output "$SESSION_DIR/sync_data.json"
```

Import into the session DB, replacing the rough sync from the YAML:

```bash
posetrak-db sync import \
    --session-db "$SESSION_DIR/session.db" \
    --shot <shot1_id> \
    --sync-json "$SESSION_DIR/sync_data.json" \
    --camera-instance cam1=<inst1_id> --camera-instance cam2=<inst2_id> \
    --notes "LED detection"
# → sync_config_id: <led_sync1_id>

# Repeat for each shot (same sync file, different shot_id)
posetrak-db sync import \
    --session-db "$SESSION_DIR/session.db" \
    --shot <shot2_id> \
    --sync-json "$SESSION_DIR/sync_data.json" \
    --camera-instance cam1=<inst1_id> --camera-instance cam2=<inst2_id> \
    --notes "LED detection"
# → sync_config_id: <led_sync2_id>
```

---

## Part 7 — Pose extraction

Run the Marimo `pose_extraction.py` app to extract 2D keypoints from all camera
videos. This produces a directory of OpenPose-format JSON files per camera.

```bash
# Launch the Marimo app (interactive)
marimo run python/pipeline/pose/pose_extraction.py

# Or run headless for a single video
python python/pipeline/pose/pose_extraction.py \
    --video "$SESSION_DIR/videos/cam1.mp4" \
    --output "$SESSION_DIR/pose/cam1/"
# Repeat for cam2, cam3, …
```

The output directory should have the structure:
```
$SESSION_DIR/pose/
  cam1/
    cam1_000000.json
    cam1_000001.json
    …
  cam2/
    cam2_000000.json
    …
```

Import into the session DB for each take:

```bash
posetrak-db pose import \
    --session-db "$SESSION_DIR/session.db" \
    --shot <shot1_id> \
    --sync-config <led_sync1_id> \
    --pose-dir "$SESSION_DIR/pose/" \
    --camera-instance cam1=<inst1_id> --camera-instance cam2=<inst2_id> \
    --pose-model "rtmpose-x"
# → sequence_id: <seq1_id>
#   n_observations: 48000

posetrak-db pose import \
    --session-db "$SESSION_DIR/session.db" \
    --shot <shot2_id> \
    --sync-config <led_sync2_id> \
    --pose-dir "$SESSION_DIR/pose/" \
    --camera-instance cam1=<inst1_id> --camera-instance cam2=<inst2_id> \
    --pose-model "rtmpose-x"
```

---

## Part 8 — Tracker config

### 8a. Import skeleton

```bash
posetrak-db skeleton import \
    --file tests/harri-skeleton.yaml \
    --session-db "$SESSION_DIR/session.db"
# → skeleton_id: <skel_id>
```

### 8b. Create tracker TOML

Copy `tests/regress.toml` and edit the `[data]` section to point at the session:

```toml
# /mnt/d/mocap/2026-03-22-my-session/tracker_take1.toml

[data]
skeleton       = "tests/harri-skeleton.yaml"
cameras        = "/mnt/d/mocap/2026-03-22-my-session/Calib_scene.toml"
observations_dir = "/mnt/d/mocap/2026-03-22-my-session/pose/"
sync           = "/mnt/d/mocap/2026-03-22-my-session/sync_data.json"
person_id      = 0

[tracking]
process_noise_std     = 0.15
measurement_noise_std = 20.0
outlier_threshold     = 4.0

[tracking.initialization]
ik_max_iterations  = 1000
ik_tolerance       = 0.02
min_cameras_for_init = 2

[tracking.ukf]
alpha = 0.1
beta  = 2.0
kappa = 0.0

[output]
directory             = "/mnt/d/mocap/2026-03-22-my-session/tracking/take1"
export_tracking_results = true
export_statistics       = true

[processing]
start_time  = 0.0         # seconds in the common timeline
end_time    = 20.0
tracker_fps = 120.0
```

---

## Part 9 — Run the tracker

Use the optimised build for performance (debug build is ~10× slower):

```bash
# Build optimised if not already done
meson setup optbuild --buildtype=release
meson compile -C optbuild

# Run the tracker
optbuild/cli/posetrak track /mnt/d/mocap/2026-03-22-my-session/tracker_take1.toml
```

Output CSVs are written to the `[output] directory`:

| File | Contents |
|---|---|
| `joint_angles.csv` | Per-frame joint angles (axis-angle, one row per frame) |
| `root_pose.csv` | Root position + quaternion per frame |
| `state_vectors.csv` | Full UKF state vector per frame |
| `tracking_stats.csv` | Per-frame n_inliers, covariance condition number |
| `tracking_results.csv` | Combined state + stats |

Check `tracking_stats.csv` for frames where `tracking_lost = 1` — these indicate
dropped tracking. If too many frames are lost, tune `measurement_noise_std` (higher
= more tolerant of noisy observations) or `outlier_threshold` (higher = reject
fewer observations).

### Optional: RTS smoothing

Add to the `[processing]` section:
```toml
[processing]
smooth = true
```
Smoothed results are written alongside the non-smoothed ones (flagged in
`tracking_results.csv` by `is_smoothed = 1`).

---

## Part 10 — Export BVH

```bash
python python/tools/export_bvh.py \
    /mnt/d/mocap/2026-03-22-my-session/tracking/take1 \
    --skeleton tests/harri-skeleton.yaml \
    --output /mnt/d/mocap/2026-03-22-my-session/take1.bvh \
    --fps 120 \
    --units m \
    --coord yup
```

Key options:

| Flag | Default | Notes |
|---|---|---|
| `--fps` | from CSV | Output BVH frame rate |
| `--units` | `m` | `m` = metres, `cm` = centimetres |
| `--coord` | `yup` | `yup` for Blender/MotionBuilder, `zup` for some other tools |
| `--smoothed` | off | Export the RTS-smoothed result |
| `--start-frame` / `--end-frame` | all | Trim the output |
| `--no-rest-frame` | off | Skip frame 0 (rest pose) |

Import into Blender with **File → Import → Motion Capture (.bvh)**.

---

## Quick reference — all IDs

It is useful to keep a scratch file with the IDs for a session. Example:

```
registry:    ~/.posetrak/registry.db
model_id:    <uuid>          GoPro Hero 12 Black
mode_id:     <uuid>          3840×2160 @ 120fps
inst1_id:    <uuid>          cam1  (serial: SN12345)
inst2_id:    <uuid>          cam2  (serial: SN12346)

session:     /mnt/d/mocap/2026-03-22-my-session/session.db
session_id:  <uuid>
shot1_id:    <uuid>          take1
shot2_id:    <uuid>          take2
led_sync1:   <uuid>          sync config for take1
led_sync2:   <uuid>          sync config for take2
```

Run `posetrak-db session list --session-db session.db` and
`posetrak-db shot list --session-db session.db` to retrieve IDs at any time.

---

## Troubleshooting

**Tracker diverges / loses tracking immediately**
: Check that `Calib_scene.toml` cameras match the extrinsics used. The TOML camera
  names (`cam1`, `cam2`) must match those used when importing extrinsics and sync.

**"No shot_videos row found" during sync import**
: The shot videos (`first_frame`/`last_frame`) must be created before importing
  sync. Verify with `posetrak-db shot list --session-db session.db` that the shot
  has video rows.

**"No intrinsics_calibration found for camera"**
: Run `posetrak-db calib import-h5` before `session import-yaml`.

**BVH root position drifts / is in wrong units**
: Use `--units cm` if the Calib_scene.toml was calibrated in centimetres (check
  the `translation` magnitude — values > 100 usually mean centimetres).
