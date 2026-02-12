# Rerun Visualization Design for Posetrak (v2)

**Created**: 2026-02-12
**Status**: Design Phase
**Version**: 2.0 - Incorporates feedback: multi-person, transform hierarchy, per-obs stats, unified observations

---

## Changes from v1

1. ✅ **Multi-person support**: Entity hierarchy supports tracking N people simultaneously
2. ✅ **Transform hierarchy**: Skeleton uses Transform3D tree (not just points+lines) for proper rotation visualization
3. ✅ **Unified observations**: Single Points2D series with per-point properties (not separate raw/inlier/outlier)
4. ✅ **Per-observation statistics**: Innovation, Mahalanobis distance, confidence logged as per-point data
5. ✅ **Image coordinate system**: Observations are children of image entity (proper 2D coordinates)
6. ✅ **Posterior markers**: Added `markers/posterior/` (was missing in v1)
7. ✅ **Dual timelines**: Both `frame` (sequence) and `timestamp` (time) timelines

---

## Entity Hierarchy

```
world/
├── cameras/
│   ├── cam_0/
│   │   ├── image                        [Image] Camera frame (optional, future)
│   │   ├── image/observations/person_0  [Points2D] Observations in image coords
│   │   │                                - Color: green (inlier) or red (outlier)
│   │   │                                - Radii: scaled by confidence
│   │   │                                - Labels: marker names
│   │   │                                - Class IDs: marker IDs
│   │   │                                - KeypointIds: for annotation
│   │   ├── image/observations/person_1  [Points2D]
│   │   ├── image/predictions/person_0   [Points2D] UKF predictions (blue)
│   │   ├── image/predictions/person_1   [Points2D]
│   │   ├── image/reprojection_errors/person_0  [LineStrips2D] predicted→actual
│   │   ├── image/reprojection_errors/person_1  [LineStrips2D]
│   │   ├── pinhole                      [Pinhole] Camera intrinsics
│   │   └── transform                    [Transform3D] Camera extrinsics (world pose)
│   ├── cam_1/
│   │   └── ... (same structure)
│   └── cam_{2,3,...N}/
│
├── person_0/
│   ├── skeleton/
│   │   ├── rest_pose/                   *** Transform hierarchy (logged once) ***
│   │   │   ├── root                     [Transform3D] Root frame
│   │   │   ├── root/pelvis              [Transform3D] Pelvis relative to root
│   │   │   ├── root/pelvis/spine_01     [Transform3D] Spine_01 relative to pelvis
│   │   │   ├── root/pelvis/spine_01/spine_02  [Transform3D]
│   │   │   ├── root/pelvis/left_hip     [Transform3D] Left leg chain...
│   │   │   ├── root/pelvis/left_hip/left_knee   [Transform3D]
│   │   │   ├── root/pelvis/right_hip    [Transform3D] Right leg chain...
│   │   │   └── ... (full kinematic tree)
│   │   │
│   │   ├── predicted/                   *** After UKF predict step ***
│   │   │   ├── root                     [Transform3D] Predicted transforms
│   │   │   ├── root/pelvis              [Transform3D] (with predicted rotations)
│   │   │   └── ... (same tree structure, updated poses)
│   │   │
│   │   └── posterior/                   *** After UKF update step ***
│   │       ├── root                     [Transform3D] Updated transforms
│   │       ├── root/pelvis              [Transform3D] (with posterior rotations)
│   │       └── ... (same tree structure, final poses)
│   │
│   ├── markers/
│   │   ├── triangulated                 [Points3D] Initial 3D (from multi-view)
│   │   ├── predicted                    [Points3D] From FK (predicted state)
│   │   ├── posterior                    [Points3D] From FK (posterior state) ***NEW***
│   │   └── labels                       [TextLog] Marker names
│   │
│   ├── trajectory/
│   │   └── root                         [LineStrips3D] Root path over time (trail)
│   │
│   ├── tracking_quality/                *** Aggregate metrics per person ***
│   │   ├── reprojection_error_mean      [Scalar] Mean reprojection error (pixels)
│   │   ├── reprojection_error_max       [Scalar] Max reprojection error (pixels)
│   │   ├── num_observations             [Scalar] Total observations used
│   │   ├── num_inliers                  [Scalar] Inlier count
│   │   ├── num_outliers                 [Scalar] Outlier count
│   │   ├── outlier_rate                 [Scalar] Percentage (0-100)
│   │   ├── nis                          [Scalar] Normalized Innovation Squared
│   │   ├── covariance_condition_number  [Scalar] Condition number
│   │   └── covariance_min_eigenvalue    [Scalar] Min eigenvalue (stability)
│   │
│   ├── state/                           *** State vectors ***
│   │   ├── joint_angles                 [Tensor] (DOF,) vector
│   │   ├── joint_velocities             [Tensor] (DOF,) vector
│   │   ├── root_position                [Vec3D] Position
│   │   └── root_orientation             [Quaternion] Orientation
│   │
│   ├── observation_stats/               *** Per-observation diagnostics (CRITICAL) ***
│   │   │                                All as (N_obs,) tensors or BarChart
│   │   ├── innovation_norms             [Tensor] Innovation magnitude per obs
│   │   ├── innovation_x                 [Tensor] Innovation x component (pixels)
│   │   ├── innovation_y                 [Tensor] Innovation y component (pixels)
│   │   ├── mahalanobis_distances        [Tensor] Mahalanobis distance per obs
│   │   ├── outlier_flags                [Tensor] Boolean outlier status (0/1)
│   │   ├── camera_ids                   [Tensor] Which camera per obs
│   │   ├── marker_ids                   [Tensor] Which marker per obs
│   │   ├── confidences                  [Tensor] OpenPose confidence per obs
│   │   └── predicted_vs_actual          [Tensor] (N_obs, 4) [pred_u, pred_v, act_u, act_v]
│   │
│   └── diagnostics/                     *** Advanced (optional) ***
│       ├── sigma_points                 [Points3D] UKF sigma points (3D cloud)
│       ├── covariance_ellipsoid         [Ellipsoids3D] 3D uncertainty visualization
│       └── joint_limit_violations       [Tensor] Per-joint flags (0=ok, 1=at limit)
│
├── person_1/                            *** Additional tracked people ***
│   └── ... (exact same structure as person_0)
│
├── person_2/
│   └── ...
│
└── scene/                               *** Scene-level metadata ***
    ├── floor_plane                      [Plane3D] Ground plane (if known)
    └── coordinate_axes                  [Arrows3D] World XYZ axes
```

---

## Timelines

### Dual Timeline System

Rerun supports multiple timelines. We use **two** for flexibility:

```cpp
// Frame-based timeline (integer sequence)
rec.set_time_sequence("frame", frame_number);

// Time-based timeline (float seconds)
rec.set_time_seconds("timestamp", timestamp);
```

**Why both?**
- **`frame`**: Easy scrubbing (frame 0, 1, 2, ...), aligns with test data indices
- **`timestamp`**: Actual time for synchronization, analysis of temporal dynamics

**User experience**:
- Scrub by frame in Rerun UI: "Show me frame 272 where tracking fails"
- Or scrub by time: "Show me what happens at t=9.0 seconds"
- Plot metrics over time (seconds) or frames (count)

---

## Key Design Decisions

### 1. Multi-Person Support

**Problem**: Need to track multiple people simultaneously.

**Solution**: Each person gets independent entity subtree under `world/person_{id}/`.

**Benefits**:
- Clean separation (no name conflicts)
- Can toggle visibility per person
- Easy to compare multiple people
- Scalable to N people

**Example**:
```cpp
// Person 0
rec.log("world/person_0/skeleton/posterior/root", ...);
rec.log("world/person_0/markers/posterior", ...);

// Person 1
rec.log("world/person_1/skeleton/posterior/root", ...);
rec.log("world/person_1/markers/posterior", ...);
```

### 2. Transform Hierarchy (not Points + Lines)

**Problem**: Points + lines don't show rotation, hard to add geometry, not semantically correct.

**Solution**: Log skeleton as **Transform3D hierarchy** matching kinematic chain.

**Benefits**:
- **Proper rotation visualization**: Can see ball joints rotating, not just positions
- **Add geometry later**: Attach capsules, meshes to transforms
- **Semantic structure**: Rerun understands parent-child relationships
- **Future-proof**: Can add IK visualizations, collision shapes, etc.

**Example**:
```cpp
// Rest pose (logged once at initialization)
rec.log("world/person_0/skeleton/rest_pose/root",
        Transform3D::from_translation_rotation(root_pos, root_quat));

rec.log("world/person_0/skeleton/rest_pose/root/pelvis",
        Transform3D::from_translation_rotation(pelvis_local_pos, pelvis_local_quat));

rec.log("world/person_0/skeleton/rest_pose/root/pelvis/spine_01",
        Transform3D::from_translation_rotation(spine01_local_pos, spine01_local_quat));

// Predicted pose (after predict step)
rec.log("world/person_0/skeleton/predicted/root",
        Transform3D::from_translation_rotation(predicted_root_pos, predicted_root_quat));
// ... etc for all joints

// Posterior pose (after update step)
rec.log("world/person_0/skeleton/posterior/root",
        Transform3D::from_translation_rotation(posterior_root_pos, posterior_root_quat));
// ... etc for all joints
```

**Note**: Can still overlay points/lines if desired for debugging, but hierarchy is primary.

### 3. Unified Observations (not separate series)

**Problem**: v1 had separate `raw/`, `inliers/`, `outliers/` which is redundant.

**Solution**: Single `Points2D` with per-point properties:
- **Color**: Green (inlier), red (outlier)
- **Radius**: Scaled by confidence (larger = more confident)
- **Labels**: Marker names
- **Class IDs**: Marker IDs (for filtering/selection)

**Benefits**:
- Less cluttered entity tree
- Easier to toggle all obs on/off
- Per-point data attached (innovation, Mahalanobis)
- Natural Rerun pattern

**Example**:
```cpp
std::vector<rerun::Position2D> positions;
std::vector<rerun::Color> colors;
std::vector<float> radii;
std::vector<std::string> labels;
std::vector<uint16_t> class_ids;

for (auto const& obs_result : update_result.observations) {
    positions.push_back({obs_result.actual.x(), obs_result.actual.y()});

    // Color by outlier status
    if (obs_result.is_outlier) {
        colors.push_back({255, 0, 0});  // Red
    } else {
        colors.push_back({0, 255, 0});  // Green
    }

    // Radius by confidence
    radii.push_back(2.0f + 3.0f * obs_result.confidence);  // 2-5 pixels

    labels.push_back(obs_result.marker_name);
    class_ids.push_back(obs_result.marker_id);
}

rec.log("world/cameras/cam_0/image/observations/person_0",
        rerun::Points2D(positions)
            .with_colors(colors)
            .with_radii(radii)
            .with_labels(labels)
            .with_class_ids(class_ids));
```

### 4. Per-Observation Statistics

**Problem**: Need granular data to debug outlier rejection and tune thresholds.

**Solution**: Log per-observation diagnostics as tensors:
- Innovation (x, y components and norm)
- Mahalanobis distance
- Outlier flag
- Camera/marker IDs
- Confidence

**Benefits**:
- **Histogram analysis**: "What's the distribution of Mahalanobis distances?"
- **Identify problematic markers**: "Which markers are consistently outliers?"
- **Correlation analysis**: "Do high innovation_x correspond to specific cameras?"
- **Threshold tuning**: "If I set threshold to 3.0, how many outliers?"

**Example**:
```cpp
// Collect per-observation stats
std::vector<float> innovation_norms;
std::vector<float> mahalanobis_dists;
std::vector<uint8_t> outlier_flags;
// ... etc

for (auto const& obs_result : update_result.observations) {
    innovation_norms.push_back(obs_result.innovation.norm());
    mahalanobis_dists.push_back(obs_result.mahalanobis_distance);
    outlier_flags.push_back(obs_result.is_outlier ? 1 : 0);
}

// Log as tensors
rec.log("world/person_0/observation_stats/innovation_norms",
        rerun::Tensor::from_shape_and_data(
            {static_cast<size_t>(innovation_norms.size())},
            innovation_norms
        ));

rec.log("world/person_0/observation_stats/mahalanobis_distances",
        rerun::Tensor::from_shape_and_data(
            {static_cast<size_t>(mahalanobis_dists.size())},
            mahalanobis_dists
        ));
```

**Analysis in Rerun**:
- Select tensor, view as plot/histogram
- Identify outliers visually
- Correlate with other metrics

### 5. Image Coordinate System

**Problem**: 2D observations need to align with camera images.

**Solution**: Log observations as **children of image entity**.

**Rerun behavior**:
- When logging `image` archetype, Rerun sets up 2D coordinate system
- Child entities (e.g., `image/observations/person_0`) are automatically in pixel coordinates
- 2D and 3D views are synchronized

**Example**:
```cpp
// Log image (optional, future)
rec.log("world/cameras/cam_0/image",
        rerun::Image::from_rgb24(image_data, width, height));

// Log camera intrinsics (required for 2D→3D mapping)
rec.log("world/cameras/cam_0/pinhole",
        rerun::Pinhole::from_focal_length_and_resolution(
            {fx, fy}, {width, height}
        ));

// Log observations as child of image → automatically in pixel coords
rec.log("world/cameras/cam_0/image/observations/person_0",
        rerun::Points2D(positions)...);
```

### 6. Predicted vs Posterior Markers

**Clarification of marker series**:

| Series | When | Purpose |
|--------|------|---------|
| `triangulated` | Initialization only | 3D positions from multi-view triangulation (ground truth-ish) |
| `predicted` | After UKF predict step | FK from predicted state (prior) |
| `posterior` | After UKF update step | FK from updated state (final result) |

**Why separate?**
- Compare prediction (blue) vs posterior (green) → see effect of measurements
- Check if update is correcting in right direction
- Validate FK consistency

---

## Class Interface Updates

### RerunLogger Constructor

```cpp
class RerunLogger {
public:
    /**
     * @brief Construct logger for multiple people
     *
     * @param config Rerun configuration
     * @param skeleton Skeleton model (shared by all people)
     * @param cameras Map of camera_id → Camera
     * @param person_ids List of person IDs to track
     */
    RerunLogger(RerunConfig const& config,
                Skeleton const& skeleton,
                std::unordered_map<int, Camera> const& cameras,
                std::vector<int> const& person_ids = {0});  // Default: single person

    // ... rest of interface ...
};
```

### Log Frame Methods (Multi-Person)

```cpp
/**
 * @brief Log observations for all people
 *
 * @param observations_by_person Map of person_id → observations
 */
void log_observations(
    std::unordered_map<int, std::vector<Observation>> const& observations_by_person);

/**
 * @brief Log prediction for specific person
 */
void log_prediction(int person_id,
                   State const& predicted_state,
                   std::map<std::string, Eigen::Vector3d> const& predicted_markers);

/**
 * @brief Log update for specific person with detailed diagnostics
 */
void log_update(int person_id,
               std::vector<Observation> const& observations,
               UpdateResult const& update_result,
               State const& predicted_state,  // For comparison
               State const& posterior_state,  // Final result
               ForwardKinematics& fk);
```

### Transform Hierarchy Logging

```cpp
private:
    /**
     * @brief Log skeleton as Transform3D hierarchy
     *
     * Recursively logs all joints as transforms in parent-child relationships.
     *
     * @param entity_prefix Base path (e.g., "world/person_0/skeleton/posterior")
     * @param state State to extract transforms from
     */
    void log_skeleton_hierarchy(std::string const& entity_prefix,
                               State const& state);

    /**
     * @brief Recursively log joint transform and children
     */
    void log_joint_recursive(std::string const& entity_path,
                            Joint const& joint,
                            State const& state,
                            Eigen::Isometry3d const& parent_transform);
```

---

## Configuration Updates

```toml
[rerun]
enabled = true
output_path = "tracking_output/tracking.rrd"
live_streaming = false

# Multi-person
person_ids = [0, 1]  # Track person 0 and person 1

# Visualization detail
log_camera_images = false
log_sigma_points = false
log_transform_hierarchy = true      # Use transforms (recommended)
log_skeleton_lines = false          # Also log lines for debugging (optional)
log_reprojection_vectors = true
log_observation_stats = true        # Per-obs diagnostics (IMPORTANT)

# Performance
log_every_n_frames = 1

# Timelines
use_frame_timeline = true           # Integer frame numbers
use_timestamp_timeline = true       # Float seconds
```

---

## Data Flow (Multi-Person Example)

```
Frame 42, t=1.400s, Person 0 and Person 1
──────────────────────────────────────────────────────────────

1. rerun_logger.log_frame_start(42, 1.400)
   • rec.set_time_sequence("frame", 42)
   • rec.set_time_seconds("timestamp", 1.400)

2. rerun_logger.log_observations(obs_by_person)
   for person_id in [0, 1]:
     for camera_id in [0, 1, 2, 3]:
       • world/cameras/cam_X/image/observations/person_Y
         (unified Points2D with colors/radii/labels)

3. Predict step (person 0)
   rerun_logger.log_prediction(0, predicted_state_0, predicted_markers_0)
   • world/person_0/skeleton/predicted/root/... (transform hierarchy)
   • world/person_0/markers/predicted (Points3D)

4. Predict step (person 1)
   rerun_logger.log_prediction(1, predicted_state_1, predicted_markers_1)
   • world/person_1/skeleton/predicted/root/... (transform hierarchy)
   • world/person_1/markers/predicted (Points3D)

5. Update step (person 0)
   rerun_logger.log_update(0, obs_0, update_result_0, pred_state_0, post_state_0, fk)
   • world/cameras/cam_X/image/predictions/person_0 (blue points)
   • world/cameras/cam_X/image/reprojection_errors/person_0 (lines)
   • world/person_0/observation_stats/* (tensors)

6. Update step (person 1)
   rerun_logger.log_update(1, obs_1, update_result_1, pred_state_1, post_state_1, fk)
   • world/cameras/cam_X/image/predictions/person_1 (blue points)
   • world/cameras/cam_X/image/reprojection_errors/person_1 (lines)
   • world/person_1/observation_stats/* (tensors)

7. rerun_logger.log_frame_end(person_id=0, posterior_state_0, ...)
   • world/person_0/skeleton/posterior/root/... (transform hierarchy)
   • world/person_0/markers/posterior (Points3D)
   • world/person_0/tracking_quality/* (scalars)
   • world/person_0/state/* (tensors)

8. rerun_logger.log_frame_end(person_id=1, posterior_state_1, ...)
   • world/person_1/skeleton/posterior/root/... (transform hierarchy)
   • world/person_1/markers/posterior (Points3D)
   • world/person_1/tracking_quality/* (scalars)
   • world/person_1/state/* (tensors)
```

---

## Debugging Workflow Example

### Scenario: Frame 272 Tracking Divergence

**Question**: Why does tracking fail at frame 272?

**Workflow**:

1. **Load recording**: `rerun tracking.rrd`

2. **Scrub to frame 272** using frame timeline

3. **Check 3D view**:
   - Is skeleton still reasonable?
   - Are markers drifting from skeleton?
   - Are predicted vs posterior very different? (prediction bad or update bad?)

4. **Check 2D camera views**:
   - Which cameras have red (outlier) observations?
   - Are reprojection vectors large?
   - Are outliers random or systematic? (e.g., all in one camera → calibration issue)

5. **Check metrics** (time series plots):
   - `outlier_rate`: Does it spike before frame 272?
   - `reprojection_error_mean`: Does it grow over time?
   - `covariance_condition_number`: Does it explode? (filter instability)
   - `nis`: Is innovation too large? (model mismatch)

6. **Check per-observation stats** (frame 272 specifically):
   - Select `world/person_0/observation_stats/mahalanobis_distances`
   - View as histogram: Are distances concentrated near threshold?
   - Select `world/person_0/observation_stats/innovation_norms`
   - Are innovations asymmetric (x vs y)?
   - Correlate with `camera_ids`: Is one camera pathological?

7. **Hypothesis**: "Camera 2 has systematic error causing many outliers"
   - Filter view to show only camera 2 observations
   - Check if predicted projections are systematically off
   - → Action: Re-calibrate camera 2

8. **Hypothesis**: "Outlier threshold too aggressive"
   - Check distribution of `mahalanobis_distances` for inliers
   - Are many distances near threshold (e.g., 3.5 when threshold is 4.0)?
   - → Action: Increase threshold to 5.0 and re-test

---

## Implementation Priority

### Phase 1: Core Infrastructure (3 days)
- [ ] RerunLogger class with dual timelines
- [ ] Multi-person entity structure
- [ ] Transform hierarchy logging (skeleton)
- [ ] Basic camera setup (pinhole, extrinsics)
- [ ] Unified observation logging (single Points2D per person)

**Deliverable**: View animated 3D skeleton(s) with proper rotations

### Phase 2: 2D Observations & Diagnostics (3 days)
- [ ] Image coordinate system (observations as children of image)
- [ ] Prediction projections (blue points)
- [ ] Reprojection error vectors (lines)
- [ ] **Per-observation statistics** (tensors) - CRITICAL

**Deliverable**: See 2D/3D synchronized, analyze per-obs stats to debug outliers

### Phase 3: Metrics & Integration (2 days)
- [ ] Aggregate tracking quality metrics (scalars)
- [ ] State logging (joint angles, velocities)
- [ ] Root trajectory (3D trail)
- [ ] CLI integration (`--rerun output.rrd`)

**Deliverable**: Full working system ready for real data debugging

### Phase 4: Advanced (Optional, 2-3 days)
- [ ] Sigma points visualization
- [ ] Covariance ellipsoids
- [ ] Camera image loading
- [ ] Comparison mode (Python vs C++)

**Total**: 8-11 days for essential features

---

## Summary of Improvements

| Feature | v1 | v2 | Benefit |
|---------|----|----|---------|
| **Multi-person** | ❌ Single | ✅ N people | Track multiple people simultaneously |
| **Skeleton** | Points+lines | ✅ Transform hierarchy | Proper rotation viz, add geometry later |
| **Observations** | 3 separate series | ✅ Unified with properties | Cleaner, per-point data |
| **Markers** | predicted, triangulated | ✅ +posterior | See before/after update |
| **Per-obs stats** | ❌ None | ✅ Tensors | Debug outlier rejection |
| **Coordinates** | World space | ✅ Image space (2D) | Proper alignment with images |
| **Timelines** | ❌ Unclear | ✅ frame + timestamp | Flexible scrubbing |

---

## Open Questions

1. **Skeleton transform hierarchy**:
   - Should we log both transforms AND points+lines?
   - Or just transforms (cleaner but less familiar)?
   - **Suggestion**: Transforms only, add optional lines via config flag

2. **Per-observation stats granularity**:
   - Log every frame or only "interesting" frames (high outlier rate)?
   - **Suggestion**: Every frame (tensors are compact), use subsampling if needed

3. **Multi-person observation assignment**:
   - How do we know which observations belong to which person?
   - **Assumption**: Tracker already does assignment (person_id in observations)

4. **Image loading**:
   - Where do camera images come from? (not part of current pipeline)
   - **Suggestion**: Phase 4 feature, requires adding video loading to tracker

5. **Comparison mode**:
   - Load Python and C++ recordings separately or combined?
   - **Suggestion**: Separate recordings, load both in Rerun viewer, compare visually

---

## Next Steps

1. **Review this design** with stakeholders
2. **Prototype Phase 1** (transform hierarchy logging) to validate Rerun API usage
3. **Test with synthetic data** (10 frames) before real data
4. **Implement Phase 2** (per-obs stats) - this is critical for debugging
5. **Deploy on real data** (kotegaeshi sequence) and analyze divergence

Ready to proceed with implementation?
