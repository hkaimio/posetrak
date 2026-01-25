# Synchronization Metadata Format Specification

**Version:** 1.0
**Date:** January 2026
**Purpose:** Define frame-to-timestamp synchronization for multi-camera motion capture

---

## Overview

The synchronization metadata format provides explicit frame-to-timestamp mappings for camera recordings. This is essential when:
- Cameras are triggered asynchronously
- Recording start times differ between cameras
- Frame rates vary or are non-uniform
- Precise temporal alignment is required for triangulation and tracking

The synchronization file only provides **sync points**. Camera FPS and start_frame are defined in the camera calibration, not here.

---

## File Format

**Format:** JSON
**Extension:** `.json`
**Encoding:** UTF-8

---

## Schema

### Root Object

The root object is a JSON object with camera names as keys.

```json
{
  "cam1": { <CameraSync> },
  "cam2": { <CameraSync> },
  ...
}
```

### CameraSync Object

Each camera's synchronization data is an **array of SyncPoint objects**.

```json
{
  "cam1": [
    {"frame": 0, "timestamp": 0.0},
    {"frame": 100, "timestamp": 3.333}
  ],
  "cam2": [
    {"frame": 0, "timestamp": 0.083},
    {"frame": 100, "timestamp": 3.383}
  ]
}
```

**Notes:**
- Empty array `[]` is valid (means no sync points, use camera's default FPS)
- Null value is also valid (equivalent to empty array)
- Minimum 1 sync point if array is non-empty

### SyncPoint Object

A synchronization point maps a specific frame index to a timestamp.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `frame` | integer | Yes | Frame index (0-based) |
| `timestamp` | number | Yes | Timestamp in seconds (relative to recording start) |

**Notes:**
- Frame indices must be non-negative
- Timestamps must be non-negative and monotonically increasing
- Sync points should be sorted by frame index (ascending)
- Minimum 1 sync point if array is present (empty arrays are invalid)

---

## Timestamp Calculation Semantics

### With Sync Points

When sync points are provided:

1. **Between Points:** Linear interpolation
   ```
   t = t₁ + (frame - f₁) × (t₂ - t₁) / (f₂ - f₁)
   ```
   where `(f₁, t₁)` and `(f₂, t₂)` are adjacent sync points bracketing `frame`.

2. **Before First Point:** Backward extrapolation using the rate from first two points
   ```
   rate = (t₂ - t₁) / (f₂ - f₁)
   t = t₁ - (f₁ - frame) × rate
   ```
   If only one sync point exists, use camera's FPS: `rate = 1.0 / fps`

3. **After Last Point:** Forward extrapolation using the rate from last two points
   ```
   rate = (t₂ - t₁) / (f₂ - f₁)
   t = t₂ + (frame - f₂) × rate
   ```
   If only one sync point exists, use camera's FPS: `rate = 1.0 / fps`

### Without Sync Points

When no sync points are provided (empty array or camera not in sync file):
```
t = (frame - start_frame) / fps
```

This uses the camera's default FPS and start_frame from calibration.

### Floor Semantics for Frame Lookup

When converting timestamp → frame index, use **floor semantics**:
- Return the last frame that starts **at or before** the given timestamp
- Example: If frame 10 is at t=0.333s and frame 11 is at t=0.366s:
  - `get_frame_at_time(0.350)` returns frame 10
  - `get_frame_at_time(0.366)` returns frame 11

---

## Validation Rules

### Required Validations

1. **Camera Name Match:** Camera names in sync file should match camera names in calibration (warning only)
2. **Non-Empty Sync Points:** If array exists and is not empty, it must have at least 1 element
3. **Monotonic Frames:** Sync point frame indices must be strictly increasing
4. **Monotonic Timestamps:** Sync point timestamps must be strictly increasing
5. **Non-Negative Values:** Frame indices and timestamps must be >= 0

### Optional Validations (Warnings)

1. **Large Time Gaps:** Warn if timestamp gaps between sync points exceed 60 seconds
2. **Frame Rate Consistency:** Warn if computed rate from sync points differs significantly from camera's nominal FPS

---

## Example Files

### Example 1: No Sync File Needed

If cameras have uniform FPS and aligned start times, no sync file is needed. The Camera objects already have FPS and start_frame.

### Example 2: Simple Time Offset

```json
{
  "cam1": [
    {"frame": 0, "timestamp": 0.0},
    {"frame": 300, "timestamp": 10.0}
  ],
  "cam2": [
    {"frame": 0, "timestamp": 0.083},
    {"frame": 300, "timestamp": 10.083}
  ]
}
```

**Behavior:** `cam2` has 83ms offset (2.5 frames at 30fps) relative to `cam1`.

### Example 3: Non-Uniform Frame Timing

```json
{
  "cam1": [
    {"frame": 0, "timestamp": 0.0},
    {"frame": 100, "timestamp": 3.2},
    {"frame": 200, "timestamp": 6.8},
    {"frame": 300, "timestamp": 10.0}
  ]
}
```

**Behavior:** Frames 0-100 run slower (32ms/frame), frames 100-200 run faster (36ms/frame), frames 200-300 run at nominal rate (32ms/frame).

### Example 4: Mixed (Some Cameras with Sync Points)

```json
{
  "cam1": [
    {"frame": 0, "timestamp": 0.0},
    {"frame": 300, "timestamp": 10.0}
  ]
}
```

**Behavior:** Only `cam1` gets sync points. Other cameras use their default FPS timing.

---

## Integration with Camera Objects

The synchronization loader should:

1. **Load** sync metadata from JSON file
2. **Validate** frame and timestamp ordering
3. **Apply** sync points to Camera objects via `Camera::set_sync_points()`

**Pseudo-code:**
```cpp
auto sync_data = load_sync_metadata("sync.json");
for (auto& [cam_name, camera] : cameras) {
    if (sync_data.contains(cam_name) && !sync_data[cam_name].empty()) {
        camera.set_sync_points(sync_data[cam_name]);
    }
    // Otherwise camera uses its existing fps and start_frame
}
```

---

## Future Extensions

Possible additions in future versions:

1. **Global Offset:** Root-level `global_time_offset` field to shift all timestamps
2. **Drift Correction:** Polynomial or spline interpolation for clock drift
3. **Trigger Events:** Named event markers (e.g., "trial_start", "contact")
4. **Confidence Weights:** Per-sync-point confidence for robust interpolation
5. **Frame Rate Changes:** Support for mid-recording FPS changes

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01 | Initial specification |

---

## References

- Camera model: `include/posetrak/core/camera.hpp`
- SyncPoint structure: `Camera::set_sync_points()` method
- Floor semantics: `Camera::get_frame_at_time()` documentation
