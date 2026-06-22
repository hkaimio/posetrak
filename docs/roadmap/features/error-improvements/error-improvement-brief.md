# Posetrak measurement error improvements

Posetrak measurement error is currently expressed as a single configurable number (pixel std deviation). This has a few issues:

- The number is in image pixels, but pose estimation is done from a crop region containing
  the person that is scaled to the size expected by the algorithm. So the pose estimation
  error in original video coordinates depends on the scaling factor (if the subject fills
  the whole screen, the error is larger).
- For the same reason, adding support for a separate pose estimation pass for hands would
  benefit from this: since hand estimation uses a tighter crop, the pose estimation error
  would be smaller for hand keypoints. This is not supported by the current tracker
  (should be straightforward to add once the split error model is in place).
- Another source of measurement error is camera calibration error. These are not Gaussian
  and can be the same order of magnitude as, or larger than, pose estimation errors.

---

## Idea 1 — Split measurement error into two terms (straightforward)

**Background:** `noise_scale = bbox_width / pose_input_width` is already computed and stored
in `pose_observations` by the detection pipeline (`DetectionBatchWriter`, `db_cache.py`).
It equals the scale factor from pose estimator pixel space back to original video pixels.
The C++ `session_reader.cpp` loads pose observations from the DB but currently ignores this
column.

**Proposal:** Replace the single `measurement_noise_std` config parameter with two terms:

| Parameter | Unit | Meaning |
|---|---|---|
| `ep` | pixels in pose estimator input | Pose estimation error (RTMPose / ViTPose model accuracy) |
| `ec` | pixels in original video | Calibration error (extrinsic + intrinsic residual) |

The effective noise for a given observation becomes:

```
sigma = (ep * noise_scale + ec) / max(confidence, 0.1)
```

where `noise_scale = bbox_width / pose_input_width` (already stored in the DB).

This correctly accounts for the crop scale: when the person is small in the frame
(`noise_scale` is small), the pose estimation contribution is small; when the crop
fills the frame, the model's pixel error maps 1:1 to video pixels.

**Implementation sketch:**

1. Add `double crop_scale = 1.0` field to `Observation` (`observation.hpp`).
2. In `session_reader.cpp` `load_observations()`: extend the SQL query to also fetch
   `noise_scale` from `pose_observations` (one value per frame/camera row, not per
   keypoint), and set `obs.crop_scale = noise_scale`.
3. Update `Observation::measurement_noise_std()` to accept `ep` and `ec` and apply
   the formula above.
4. Add `ep` and `ec` to `TrackerConfig` (both in `config.hpp` and the TOML parser).
   Backward compat: if only the legacy `measurement_noise_std` is set, treat it as
   `ec` with `ep = 0`.
5. Propagate the two new parameters through `UKF::update()` → `tracker.cpp`.
6. Update config form in `run_tracker.py` and `content_panels.py` to show both fields.

---

## Idea 2 — Modelling calibration error spatial correlation (research)

**The problem:** Calibration error is not random white noise. A systematic offset for a given
camera typically changes smoothly as the subject moves through the scene (the reprojection
error of a camera is a smooth function of 3D position). The current filter models all
measurement noise as i.i.d. Gaussian, which misrepresents this structure.

### Known approaches

**Per-camera bias state (simplest, worth trying first)**

Augment the UKF state vector with a 2D pixel-offset bias for each camera. Give the bias
a very slow process noise (essentially a random walk with small variance). The filter will
estimate and track the systematic offset online. The `velocity_measurement_noise_std` mode
already uses a differencing trick to cancel constant biases; this is the principled
extension of that idea.

*Risk:* With many cameras the state grows; also, if the bias is pose-dependent rather than
constant, it may not converge or may corrupt the pose estimate.

**Spatial Gaussian Process over calibration residuals**

After a full tracking pass, compute per-camera signed residuals (predicted minus observed)
as a function of 3D marker position. Fit a GP with a Matérn kernel to these residuals.
Use the GP posterior mean as a correction in a second tracking pass, and the GP variance
to inflate `ec` location-dependently. This is offline but could be applied iteratively.

**Outlier-robust measurement likelihood**

Replace the Gaussian likelihood in the UKF update with a Student-*t* distribution (heavy
tails) or a mixture model (inlier Gaussian + uniform outlier component). This does not
model the spatial correlation but is more robust to occasional large calibration errors.
Can be layered on top of the existing Mahalanobis gate without restructuring the filter.
A practical starting point: scale the measurement covariance by `(ν + d) / (ν + DOF)` in
the Joseph update (Huber-style robust UKF).

**Online calibration refinement (most complex)**

Treat extrinsic parameters (e.g. camera position + orientation, or lens distortion
coefficients) as slowly-drifting states in the filter, similar to SLAM back-ends. This
requires computing Jacobians of the projection function with respect to extrinsic
parameters and integrating them into the sigma-point framework. Expensive but gives the
most principled treatment.

### Suggested next steps for exploration

1. **Diagnose first.** After a tracking run, plot per-camera signed residuals
   (predicted − observed pixel position) vs. time and vs. 3D marker position (from the
   smoothed state). If residuals have a clear smooth structure in 3D space or drift slowly
   over time, that is direct evidence of calibration error dominating noise.
2. **Try per-camera bias states** as the lowest-effort structured approach. Use one bias
   per camera (2 DOFs, slow random walk). Compare NIS before and after.
3. **Try the robust likelihood** (Student-*t* or scaled covariance) as a cheap robustness
   improvement that requires no state changes.
4. **GP-based offline correction** as a post-processing step once reliable tracking output
   is available — use it to improve calibration before re-running the tracker.
