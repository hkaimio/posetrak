# Design: Split measurement noise into pose-estimation and calibration terms

**Status:** Ready to implement
**Complexity:** Low — localised changes, no schema migration required

---

## Problem

The UKF measurement noise is a single number `measurement_noise_std` (pixels, original video
space). This conflates two unrelated error sources:

1. **Pose estimation error** — the error of the RTMPose/ViTPose model in its own input image.
   Because pose estimation runs on a crop rescaled to a fixed model input size (e.g. 384×288),
   the same model pixel error maps to a *larger* original-video error when the person is far
   away (small crop) and a *smaller* one when the person fills the frame (large crop).

2. **Calibration error** — systematic residuals from imperfect extrinsic/intrinsic calibration.
   These are roughly constant for a given camera in a given part of the image, regardless of
   how zoomed-in the person is.

A single global constant cannot represent both correctly.

---

## Data already available

`DetectionBatchWriter.add_frame()` in `app/pose/db_cache.py` already computes and stores:

```python
noise_scale = float(det.bbox[2]) / self._pose_input_width   # bbox_w / model_input_w
```

This is the scale factor from model-pixel space to original-video-pixel space.
It is stored in `pose_observations.noise_scale` (one value per detection frame/camera row,
shared by all keypoints in that frame). The C++ `session_reader.cpp` fetches keypoint blobs
from this table but currently ignores `noise_scale`.

---

## Proposed noise formula

Replace the single `measurement_noise_std` with two parameters:

| Symbol | Config key | Unit | Meaning |
|---|---|---|---|
| `ep` | `pose_noise_std` | pixels in model input | RTMPose/ViTPose model accuracy |
| `ec` | `calib_noise_std` | pixels in original video | Extrinsic + intrinsic residual |

Effective noise for a given observation:

```
sigma = (ep * crop_scale + ec) / max(confidence, 0.1)
```

where `crop_scale = bbox_width / pose_input_width` (loaded from DB).

When the person is small in the frame (crop_scale ≈ 0.1), the pose estimation contribution
is negligible and `sigma ≈ ec / confidence`. When the person fills the frame
(crop_scale ≈ 1.0), both terms contribute equally.

### Backward compatibility

Keep `measurement_noise_std` in the TOML parser. If only it is set, map it to
`ec = measurement_noise_std, ep = 0`. If `pose_noise_std` and/or `calib_noise_std` are
set they take precedence. This means all existing configs continue to work unchanged.

---

## Implementation plan

### 1. `Observation` struct  (`include/posetrak/core/observation.hpp`)

Add one field and update the noise formula:

```cpp
double crop_scale = 1.0;   ///< bbox_width / pose_input_width from detection pipeline.
                           ///< 1.0 = unknown (falls back to calibration-only formula).

double measurement_noise_std(double ep, double ec) const {
    double effective_ec = (noise_std_override > 0.0) ? noise_std_override : ec;
    double sigma = ep * crop_scale + effective_ec;
    return sigma / std::max(confidence, 0.1);
}
```

Keep the existing single-argument overload for velocity-mode and existing call sites:

```cpp
double measurement_noise_std(double base_noise = 5.0) const {
    return measurement_noise_std(0.0, base_noise);
}
```

### 2. `TrackerConfig`  (`include/posetrak/core/config.hpp`)

```cpp
// Replace measurement_noise_std (keep for compat):
double calib_noise_std = 5.0;           ///< Calibration error (pixels, original video)
double pose_noise_std  = 0.0;           ///< Pose model error (pixels, model input image)
```

### 3. TOML parser  (`src/core/config.cpp`)

```cpp
tc.calib_noise_std = tracking["calib_noise_std"].value_or(
    tracking["measurement_noise_std"].value_or(5.0));   // backward compat
tc.pose_noise_std  = tracking["pose_noise_std"].value_or(0.0);
```

Validation: both must be ≥ 0, at least one > 0.

### 4. `session_reader.cpp` — load `noise_scale`

The inner SQL query that fetches `pose_observations` rows currently selects:
```sql
camera_instance_id, video_frame, timestamp_s, kp_blob [+ edits]
```

Extend to also fetch `noise_scale`:
```sql
po.camera_instance_id, po.video_frame, po.timestamp_s,
COALESCE(noise_scale, 1.0) AS crop_scale,
[kp_blob logic]
```

`noise_scale` is one value per `(video_frame, camera_instance_id)` row, so read it once
and apply it to all keypoints in that row:

```cpp
double crop_scale = sqlite3_column_double(obs_stmt.ptr, /*col*/);
// ... keypoint loop:
obs.crop_scale = crop_scale;
```

### 5. `UKF::update()`  (`src/filters/ukf.cpp`)

The call site that assembles R currently passes a single scalar:

```cpp
double noise_std = observations[i].measurement_noise_std(measurement_noise_std);
```

Change the UKF `update()` signature to accept both parameters:

```cpp
UpdateResult update(std::vector<Observation> const& observations,
                    std::unordered_map<int, Camera> const& cameras,
                    ForwardKinematics& fk,
                    double pose_noise_std,
                    double calib_noise_std,
                    double outlier_threshold_mahalanobis);
```

Update the R-assembly block:
```cpp
double noise_std = observations[i].measurement_noise_std(pose_noise_std, calib_noise_std);
```

The hierarchical child filter (`SubsetUKF`, `src/filters/subset_ukf.cpp`) has its own
`update()` wrapper — apply the same signature change there.

### 6. `Tracker`  (`src/tracking/tracker.cpp`)

Forward `config_.pose_noise_std` and `config_.calib_noise_std` to every `ukf_->update()`
call.

### 7. Python UI

- `run_tracker.py`: add `_pose_noise` spin box alongside `_meas_noise`; wire to new config keys.
- `content_panels.py`: update the config summary string to show both terms.
- `app/mcp/tools/runs.py`: update `describe_config` text.

---

## Suggested starting values

Run the tracker with `ep` only (`ec = 0`) and with `ec` only (`ep = measurement_noise_std`)
and compare NIS. A well-tuned split should give lower NIS variance and better Mahalanobis
scores when the crop scale varies significantly across cameras (close vs. far cameras).

Typical starting point based on RTMPose-L accuracy on COCO:
- `pose_noise_std ≈ 3–5` px (in model input space, 288×384)
- `calib_noise_std ≈ 2–5` px (original video space, depends on calibration quality)

---

## Hand-estimation extension (future)

Once this formula is in place, a separate hand-estimation pass becomes straightforward:
hand keypoints detected from a tight hand crop will have a smaller `crop_scale` than body
keypoints from the person crop, so their pose-estimation contribution to noise is
automatically smaller — no separate config needed.
