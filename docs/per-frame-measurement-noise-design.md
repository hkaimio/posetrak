# Per-Frame Per-Camera Measurement Noise Design

**Date:** 2026-03-24
**Status:** Proposal

## Problem

The UKF update step uses a single scalar `measurement_noise_std` (pixels) applied identically to
every observation across all cameras and all frames.  This is wrong in practice because the
pixel-space noise from a pose detector depends on how large the person appears in the camera image.

### Why the noise scales with bounding box size

The pose estimator (RTMPose / ViTPose) receives a cropped, resized bounding box of the person as
its input.  It works in detector-space pixels (e.g. 192×256 input resolution).  The resulting
keypoint coordinates are then scaled back to original video coordinates.  The detector's intrinsic
localisation uncertainty — roughly 5–10 px in detector space — inflates by the same scale factor
when mapped back to original image space:

```
scale_factor(cam, frame) = bbox_height_orig / detector_input_height
σ_eff(cam, frame)        = σ_base * scale_factor(cam, frame)
```

Each camera has its own distance to the subject and its own sensor resolution, so scale factors
differ across cameras.  The same camera's scale factor changes frame-by-frame as the person moves
(e.g. crouching in a squat reduces apparent height).

### Observed impact

In session `20260322-teacup-exc2`, run `eace7339`, the person fills ~1447 px vertically in the 4K
GoPro frames.  With RTMPose at 256 px input height the scale factor is ≈ 5.7×.  Effective noise
is therefore ~75–100 px, but `measurement_noise_std` was configured as 20 px.  At
`outlier_threshold: 4.0` this means anything beyond 80 px is rejected as an outlier — which is
almost every valid knee/hip observation from the three GoPros.  Only the Insta360 (different
viewing angle) passed the knee consistently, leaving the left leg under-constrained from frame 1.
When the squat started the prediction diverged by 200–300 px and all leg markers became outliers.

## Proposed design

### 1. Database: store detector input size and YOLO bounding boxes

`pose_observation_sequences` gains one new column:

```sql
ALTER TABLE pose_observation_sequences
    ADD COLUMN detector_input_height INTEGER;   -- e.g. 256 for RTMPose-256, 384 for RTMPose-384
```

The `yolo_detections` table planned in `capture-pipeline-architecture.md` already captures the
per-frame per-camera bounding boxes:

```sql
CREATE TABLE yolo_detections (
    sequence_id  TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_id    INTEGER NOT NULL,
    video_frame  INTEGER NOT NULL,
    track_id     INTEGER NOT NULL,
    bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL,
    confidence   REAL NOT NULL,
    PRIMARY KEY (sequence_id, camera_id, video_frame, track_id)
);
```

No additional schema work is needed once both additions are in place.

`pose_extraction.py` must write `detector_input_height` to the sequence row and write one
`yolo_detections` row per (camera, frame) using the bbox that was fed to RTMPose.

### 2. Observation struct: add per-observation noise std

`Observation` already has a `measurement_noise_std(base_noise)` helper that scales by confidence.
Extend it with a `bbox_scale_factor` field that accounts for the detector scale:

```cpp
struct Observation {
    // ... existing fields ...
    double bbox_scale_factor = 1.0;  ///< bbox_height_orig / detector_input_height for this
                                     ///< camera+frame.  1.0 if bbox data not available.

    /// Effective noise std: base_noise * bbox_scale_factor / max(confidence, 0.1)
    double measurement_noise_std(double base_noise = 5.0) const {
        return base_noise * bbox_scale_factor / std::max(confidence, 0.1);
    }
};
```

`base_noise` in this model is the detector's intrinsic precision in detector-space pixels —
a value around 5–10 px that is stable across sessions and cameras, unlike the current
session-specific tuning.

### 3. Observation loading: populate bbox_scale_factor

The DB session reader (`session_reader.cpp`) and the JSON observation loader
(`observation_loader.cpp`) need a path to supply this value.

**DB mode** (`session_reader.cpp`): at load time, join `yolo_detections` and
`pose_observation_sequences` to compute scale factor per (camera, frame, person).  Store a
`std::unordered_map<(camera_id, frame_idx), double>` and fill `Observation::bbox_scale_factor`
when constructing each observation.  If no YOLO row exists for a frame, fall back to 1.0.

**JSON / TOML mode** (`observation_loader.cpp`): no YOLO data is available.  Leave
`bbox_scale_factor = 1.0` and rely on the user setting an appropriately large `measurement_noise_std`
in the config — as is the case today.

### 4. UKF update: use per-observation noise

Currently `Tracker::track_frame()` calls:

```cpp
ukf_->update(observations, cameras_, *fk_, config_.measurement_noise_std, outlier_threshold_);
```

and `UnscentedKalmanFilter::update()` builds the R matrix as a uniform diagonal
`σ² * I_{2n}`.

Change the signature to accept a per-observation noise vector instead of a single scalar:

```cpp
// New signature — vector length = number of active observations
UpdateResult update(std::vector<Observation> const& observations,
                    std::unordered_map<int, Camera> const& cameras,
                    ForwardKinematics& fk,
                    std::vector<double> const& noise_stds,   // replaces single scalar
                    double outlier_threshold_mahalanobis = 0.0);
```

`Tracker::track_frame()` builds the noise vector just before the call:

```cpp
std::vector<double> noise_stds;
noise_stds.reserve(active_obs.size());
for (auto const& obs : active_obs)
    noise_stds.push_back(obs.measurement_noise_std(config_.measurement_noise_std));
ukf_->update(active_obs, cameras_, *fk_, noise_stds, config_.outlier_threshold_mahalanobis);
```

Inside `UnscentedKalmanFilter::update()` the R matrix becomes a non-uniform diagonal:

```cpp
// R is 2*n × 2*n block-diagonal, 2×2 blocks per observation
Eigen::VectorXd r_diag(2 * n);
for (int i = 0; i < n; ++i) {
    double s = noise_stds[i];
    r_diag(2*i)   = s * s;
    r_diag(2*i+1) = s * s;
}
Eigen::MatrixXd R = r_diag.asDiagonal();
```

The Mahalanobis outlier test (`d² = ν^T S^{-1} ν`) already operates per-observation on the
innovation covariance `S`, so non-uniform R flows through correctly without further changes.

`SubsetUKF::update()` (hierarchical tracker) needs the same signature update.

### 5. Tracker config parameter semantics change

`measurement_noise_std` in `tracker_configs` changes meaning:

| Before | After |
|--------|-------|
| Noise in original video pixels; must be re-tuned per session | Noise in detector-space pixels; ~5–10 px, stable across sessions |

The recommended default moves from 20 px (original) to ~8 px (detector space).  When no YOLO data
is present `bbox_scale_factor = 1.0` so the parameter retains its old meaning and existing
session configs continue to work.

## Fallback when YOLO data is unavailable

When running in JSON/TOML mode or when `yolo_detections` has not been populated for a session,
`bbox_scale_factor = 1.0` for all observations and the behaviour is identical to today.  As a
stop-gap the user can estimate the effective noise from the keypoint bounding box computed at
tracker initialization and set `measurement_noise_std` accordingly in the config.

## Implementation order

1. DB migration: add `detector_input_height` to `pose_observation_sequences`.
2. Populate `yolo_detections` table in `pose_extraction.py` (ties into the existing YOLO tracking
   phase which already computes these bboxes).
3. Add `Observation::bbox_scale_factor` field; update `session_reader.cpp` to join and fill it.
4. Change `UnscentedKalmanFilter::update()` and `SubsetUKF::update()` to accept per-observation
   noise vector; update `Tracker::track_frame()` to build it.
5. Update default `measurement_noise_std` in the default tracker config and documentation.
