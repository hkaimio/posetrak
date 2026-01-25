# Phase 2 Complete: I/O Layer

**Status**: ✅ All Phase 2 components implemented and tested

## Summary

The I/O layer for posetrak is complete. All data loaders have been implemented with comprehensive testing:

### Phase 2.1: Skeleton Loader (YAML) ✅
- Implemented: `skeleton_loader.hpp/cpp`
- Tests: `test_skeleton_loader.cpp`
- Format: Custom YAML format for skeleton hierarchy and bone connections
- Features:
  * Joint hierarchy with parent-child relationships
  * Bone definitions connecting joints
  * Default poses (translations/rotations)
  * Support for null parents (root joints)
  * Comprehensive error handling

### Phase 2.2: Camera Loader (TOML) ✅
- Implemented: `camera_loader.hpp/cpp`
- Tests: `test_camera_loader.cpp`
- Format: Pose2Sim-compatible TOML camera calibration files
- Features:
  * Intrinsics: camera matrix, distortion coefficients
  * Extrinsics: rotation (Rodriguez vector), translation
  * Metadata: camera name, resolution
  * FPS and sync point support
  * Multi-camera loading from single file

### Phase 2.3: Sync Metadata Loader (JSON) ✅
- Implemented: `sync_metadata_loader.hpp/cpp`
- Tests: `test_sync_metadata_loader.cpp`
- Format: Custom JSON format for multi-camera synchronization
- Features:
  * Per-camera sync point arrays
  * Frame number to timestamp mapping
  * Simplified format (removed fps/start_frame)
  * Supports arbitrary number of cameras

### Phase 2.4: OpenPose Observation Loader (JSON) ✅
- Implemented: `observation_loader.hpp/cpp`
- Tests: `test_observation_loader.cpp`
- Format: OpenPose JSON output with COCO-133 keypoints
- Features:
  * Parses people array with pose_keypoints_2d
  * Each keypoint → one Observation struct (marker_id = keypoint index)
  * ObservationSequence accumulates per-camera observations
  * ObservationSet collects multi-camera sequences
  * Automatic coordinate undistortion via Camera
  * Confidence threshold filtering
  * Multi-person support (person_id parameter)
  * Frame range filtering for sequence loading

## Test Coverage

**Total Assertions**: 2127 (across 57 test cases)

### Breakdown by Component:
- **Phase 1 (Core models)**: 292 assertions
- **Phase 2.1 (Skeleton)**: +405 assertions → 697 total
- **Phase 2.2 (Camera)**: +37 assertions → 734 total
- **Phase 2.3 (Sync)**: +29 assertions → 763 total
- **Phase 2.4 (Observation)**: +1364 assertions → 2127 total

All tests pass. Coverage includes:
- Success cases (loading valid data)
- Error handling (missing files, invalid JSON/YAML/TOML, malformed data)
- Edge cases (empty arrays, out-of-range indices, missing fields)
- Round-trip validation where applicable

## Data Flow

The I/O layer supports the following workflow:

1. **Load Skeleton** (`skeleton_loader`) → Skeleton model
2. **Load Cameras** (`camera_loader`) → Map of Camera objects
3. **Load Sync Metadata** (`sync_metadata_loader`) → SyncMetadata for timestamp alignment
4. **Load Observations** (`observation_loader`) → ObservationSet with 2D keypoints
   - Each frame per camera → observations for all keypoints
   - Multiple cameras → multi-view observations for triangulation

These components provide all inputs needed for the kinematics layer (Phase 3).

## Design Decisions

### Observation Loader API

The observation loader maps OpenPose multi-keypoint format to posetrak's single-observation model:
- OpenPose: One person with 133 keypoints
- Posetrak: 133 separate Observation structs (one per keypoint)
- Grouping: ObservationSequence (per-camera) → ObservationSet (multi-camera)

This design allows:
- Uniform handling of observations (whether single markers or keypoints)
- Easy confidence-based filtering at keypoint level
- Flexible marker identification via marker_id

### Camera Name Handling

The load_openpose_frame function takes an explicit `camera_name` parameter rather than using `Camera::name()`:
- TOML file specifies camera internal name ("int_cam1_img")
- Map key specifies external name ("cam1")
- OpenPose data uses external name in directory structure
- Observation sequences use external name for consistency

This separation allows flexible naming without coupling to camera metadata.

## Next Steps: Phase 3 (Kinematics Layer)

With the I/O layer complete, Phase 3 will implement:

1. **Pinocchio Integration**
   - Convert Skeleton to Pinocchio model
   - Build kinematic tree
   - Implement forward kinematics

2. **3D Reconstruction**
   - Triangulation from multi-view observations
   - Camera coordinate transformations
   - Point cloud generation

3. **Inverse Kinematics**
   - Map 3D markers to joint angles
   - Pose optimization
   - Constraint handling

All required input data (skeleton, cameras, observations) is now available through the I/O layer.
