# Camera ID Assignment Refactoring - Design Document

**Date:** February 4, 2026
**Status:** Proposal
**Related Issue:** C++ and Python trackers starting from different frames

## Problem Statement

The current camera and observation loading implementation has critical design flaws that lead to:

1. **Non-deterministic camera ID assignment** - IDs depend on unordered map iteration
2. **Incorrect observation camera IDs** - All observations initially have `camera_id = 0`
3. **CLI workarounds** - Custom mapping logic to fix broken IDs after loading
4. **Python/C++ synchronization issues** - Different starting frames due to ID mismatches

### Root Causes

#### 1. Camera Loader Issues (`src/io/camera_loader.cpp`)

**Current Implementation:**
```cpp
std::unordered_map<std::string, Camera> cameras;
int camera_id = 0;

for (auto const& [key, value] : config) {  // toml::table iteration
    std::string section_name(key.str());    // e.g., "cam1", "cam2", ...
    // ... parse camera ...
    Camera camera(camera_id++, cam_name, intrinsics, extrinsics);
    cameras.emplace(section_name, std::move(camera));
}
```

**Problems:**
- **`toml::table` uses `std::map` internally** (ordered), BUT the result is stored in `std::unordered_map`
- **Later code iterates the `unordered_map`**, which has undefined iteration order
- **Camera IDs become non-deterministic** when the map is iterated later
- **Section names (cam1, cam2, ...) used as keys**, but iteration order doesn't match numeric order

**Evidence from TOML file:**
```toml
[cam1]
name = "int_cam1_img"
...
[cam2]
name = "int_cam2_img"
...
[cam4]
name = "int_cam4_img"
...
```

Expected: cam1 → ID 0, cam2 → ID 1, cam4 → ID 2
Actual: **Unpredictable** due to `unordered_map` iteration in downstream code

#### 2. Observation Loader Issues (`src/io/observation_loader.cpp`)

**Current Implementation:**
```cpp
// In parse_openpose_person():
Observation obs;
obs.camera_id = 0;  // HARDCODED! Will be set by caller if needed
```

**All 5 locations where camera_id is set to 0:**
- Line 52: `obs.camera_id = 0;` (parse_openpose_person)
- Line 103: `seq.camera_id = 0;` (load_openpose_frame - no people)
- Line 112: `seq.camera_id = 0;` (load_openpose_frame - person out of range)
- Line 121: `seq.camera_id = 0;` (load_openpose_frame - return)
- Line 172: `full_seq.camera_id = 0;` (load_openpose_sequence)

**Problems:**
- All observations get `camera_id = 0` regardless of actual camera
- No mechanism to propagate camera ID from Camera object to Observation
- Function signature doesn't accept camera ID: `load_openpose_frame(filepath, camera, camera_name, ...)`
- The `camera` parameter has an `.id()` method but it's never used!

#### 3. CLI Workaround Code (`cli/track.cpp`)

**Current Band-Aid Implementation:**
```cpp
// Helper: Convert camera map with string keys to int keys
std::unordered_map<int, Camera>
convert_camera_map(std::unordered_map<std::string, Camera> const& cameras_by_name,
                   std::unordered_map<std::string, int>& name_to_id) {
    std::unordered_map<int, Camera> cameras_by_id;
    int next_id = 0;
    for (auto const& [name, cam] : cameras_by_name) {  // UNDEFINED ORDER!
        name_to_id[name] = next_id;
        cameras_by_id.emplace(next_id, cam);
        next_id++;
    }
    return cameras_by_id;
}

// Helper: Update observation camera IDs from names
void update_observation_camera_ids(ObservationSet& obs_set,
                                   std::unordered_map<std::string, int> const& name_to_id) {
    for (auto& [cam_name, sequence] : obs_set.sequences()) {
        auto it = name_to_id.find(cam_name);
        if (it != name_to_id.end()) {
            sequence.camera_id = it->second;
            for (auto& obs : sequence.observations) {
                obs.camera_id = it->second;
            }
        }
    }
}
```

**Problems:**
- **Re-assigns camera IDs** after loading, creating different IDs than in Camera objects
- **Still non-deterministic** because it iterates `unordered_map` again
- **Workaround should not be needed** - IDs should be correct from the start
- **Code duplication** - every consumer needs similar logic

#### 4. Test Code Issues (`tests/test_ukf_frame0_comparison.cpp`)

**Current Implementation:**
```cpp
cameras_by_name = load_cameras_from_toml(cameras_path);

// Manual conversion (same pattern as CLI)
for (auto const& [name, camera] : cameras_by_name) {
    cameras.insert({camera.id(), camera});
    camera_name_to_id[name] = camera.id();
}
```

**Problems:**
- **Test uses Camera's original ID** (from loader), which may differ from CLI's reassigned IDs
- **C++ test and Python may use different camera ordering**
- **Each test file needs custom mapping code**

## Impact Analysis

### How This Causes Frame Mismatch

1. **Python tracker** loads cameras in TOML file order: cam1=0, cam2=1, cam3=2, cam4=3
2. **C++ tracker CLI** loads cameras, then:
   - Iterates `unordered_map` → unpredictable order (e.g., cam4, cam1, cam3, cam2)
   - Assigns new IDs: cam4=0, cam1=1, cam3=2, cam2=3
   - Updates observation camera IDs to match new assignment
3. **Result:** Observations that should be from camera 0 (cam1) are now from camera 1
4. **Frame synchronization breaks** because camera start frames differ

### Concrete Example

**TOML file order:**
```
[cam1] -> Should be ID 0
[cam2] -> Should be ID 1
[cam3] -> Should be ID 2
[cam4] -> Should be ID 3
```

**What happens in C++:**
1. Loader creates: `unordered_map<string, Camera>` with IDs: cam1=0, cam2=1, cam3=2, cam4=3 ✓
2. CLI iterates unordered_map (undefined order): cam4, cam1, cam3, cam2
3. CLI reassigns: cam4=0, cam1=1, cam3=2, cam2=3 ✗
4. Observations get reassigned IDs to match ✗

**What Python does:**
1. Parses TOML in order: cam1=0, cam2=1, cam3=2, cam4=3 ✓
2. Observations use correct IDs ✓

**Result:** C++ camera 0 has different frames than Python camera 0!

## Design Requirements

### Functional Requirements

1. **FR-1: Deterministic Camera ID Assignment**
   - Camera IDs MUST be assigned based on TOML file order
   - First camera section in TOML → ID 0, second → ID 1, etc.
   - ID assignment must be stable across runs

2. **FR-2: Correct Observation Camera IDs**
   - Each observation must have correct camera_id from creation
   - No post-load ID fixup should be required
   - Observation camera_id must match the Camera object's id()

3. **FR-3: Preserve Camera-Name Mapping**
   - Must maintain string name (e.g., "cam1", "int_cam1_img") for:
     - Directory traversal in `load_openpose_sequence`
     - ObservationSet organization by camera name
     - Debugging and logging

4. **FR-4: Remove CLI Workarounds**
   - Eliminate `convert_camera_map()` function
   - Eliminate `update_observation_camera_ids()` function
   - CLI should use camera objects as-is from loaders

5. **FR-5: Python/C++ Consistency**
   - C++ and Python must assign identical camera IDs for same TOML file
   - Tests must use same ID assignment as production code

### Non-Functional Requirements

1. **NFR-1: API Compatibility**
   - Minimize breaking changes to public APIs
   - Existing test files should require minimal updates

2. **NFR-2: Performance**
   - No significant performance degradation
   - Prefer ordered containers only where order matters

3. **NFR-3: Type Safety**
   - Use type system to prevent ID mismatches
   - Consider strong typing for camera IDs if beneficial

## Proposed Solution

### Overview

**Key Principle:** Assign camera IDs at parse time based on TOML section order, and propagate those IDs correctly throughout the system.

### Component Changes

#### 1. Camera Loader (`camera_loader.cpp`)

**Change:** Use `std::map` or `std::vector` to preserve TOML order

**Option A: Return std::map (Recommended)**
```cpp
std::map<std::string, Camera> load_cameras_from_toml(std::string const& filepath) {
    // toml::table already provides ordered iteration
    std::map<std::string, Camera> cameras;  // ORDERED, not unordered
    int camera_id = 0;

    for (auto const& [key, value] : config) {
        // ... parse camera ...
        Camera camera(camera_id++, cam_name, intrinsics, extrinsics);
        cameras.emplace(section_name, std::move(camera));
    }
    return cameras;
}
```

**Rationale:**
- `std::map` preserves insertion order (sorted by key)
- TOML section names (cam1, cam2, ...) sort lexicographically correctly
- Minimal API change - still returns map keyed by string
- Downstream code can iterate deterministically

**Option B: Return std::vector + name map**
```cpp
struct CameraSet {
    std::vector<Camera> cameras;  // Indexed by ID
    std::map<std::string, int> name_to_id;  // section_name -> camera_id
};

CameraSet load_cameras_from_toml(std::string const& filepath);
```

**Rationale:**
- More explicit: ID is the vector index
- Clearer separation: cameras by ID, lookup by name
- **Larger API change** - requires updating all consumers

**Recommendation:** **Option A** for minimal disruption, with clear documentation that iteration order is guaranteed.

#### 2. Observation Loader (`observation_loader.cpp`)

**Change:** Set camera_id correctly at observation creation

**Fix 1: Use camera.id() in parse_openpose_person**
```cpp
Observation obs;
obs.camera_id = camera.id();  // Use camera's ID, not hardcoded 0
obs.marker_id = it->second;
// ...
```

**Fix 2: Propagate camera to load_openpose_sequence**
```cpp
ObservationSet load_openpose_sequence(
    std::string const& base_dir,
    std::map<std::string, Camera> const& cameras,  // Changed from unordered_map
    Skeleton const& skeleton,
    std::pair<uint32_t, uint32_t> frame_range,
    double min_confidence,
    int person_id) {

    ObservationSet obs_set(person_id);

    // Iterate cameras in deterministic order
    for (auto const& [cam_name, camera] : cameras) {
        // ... load observations ...
        auto frame_seq = load_openpose_frame(filepath, camera, cam_name, ...);
        // camera_id already set correctly in observations
        full_seq.observations.insert(...);
    }
    return obs_set;
}
```

**Key Changes:**
- Accept `std::map` instead of `std::unordered_map` for deterministic iteration
- Use `camera.id()` when creating observations
- Remove all `camera_id = 0` hardcoded assignments

#### 3. CLI Application (`track.cpp`)

**Change:** Remove workaround functions, use cameras directly

**Delete:**
- `convert_camera_map()` function
- `update_observation_camera_ids()` function

**Simplify:**
```cpp
// Load cameras (now returns std::map, iteration order guaranteed)
auto cameras = load_cameras_from_toml(config.cameras_path.string());

// Apply sync if needed (function signature updated to accept std::map)
if (config.sync_path) {
    auto sync_data = load_sync_metadata(config.sync_path->string());
    apply_sync_metadata(cameras, sync_data, false);
}

// Load observations (accepts std::map, uses camera.id() internally)
auto observations_set = load_openpose_sequence(
    config.observations_dir.string(),
    cameras,  // No conversion needed!
    skeleton,
    {config.start_frame, end_frame},
    0.1,
    config.person_id);

// Camera IDs already correct in observations!

// Create tracker (needs cameras by ID)
std::unordered_map<int, Camera> cameras_by_id;
for (auto const& [name, cam] : cameras) {
    cameras_by_id.emplace(cam.id(), cam);
}
Tracker tracker(skeleton, cameras_by_id, tracker_config);
```

**Key Changes:**
- No more ID reassignment
- Simple conversion to ID-keyed map for Tracker constructor
- Observations already have correct camera IDs

#### 4. Test Files

**Change:** Use same pattern as CLI (no custom mapping needed)

**test_ukf_frame0_comparison.cpp:**
```cpp
// Load cameras (returns std::map with deterministic order)
cameras_by_name = load_cameras_from_toml(cameras_path);

// Convert to ID-keyed map for UKF
for (auto const& [name, camera] : cameras_by_name) {
    cameras.insert({camera.id(), camera});
}

// Camera IDs now match Python's IDs - no custom mapping needed!
```

**test_camera_loader.cpp:**
```cpp
// Update signature
auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");
// cameras is now std::map<string, Camera>

// Tests can verify deterministic ID assignment
REQUIRE(cameras.at("cam1").id() == 0);
REQUIRE(cameras.at("cam2").id() == 1);
REQUIRE(cameras.at("cam3").id() == 2);
REQUIRE(cameras.at("cam4").id() == 3);
```

### API Changes Summary

| Component | Current Return Type | New Return Type | Breaking? |
|-----------|-------------------|-----------------|-----------|
| `load_cameras_from_toml()` | `std::unordered_map<string, Camera>` | `std::map<string, Camera>` | **Yes** - Minor |
| `load_openpose_sequence()` | Accepts `unordered_map` | Accepts `std::map` | **Yes** - Minor |
| `apply_sync_metadata()` | Accepts `unordered_map&` | Accepts `std::map&` | **Yes** - Minor |

**Migration Path:**
1. Update function signatures to use `std::map`
2. Update all call sites (grep for `load_cameras_from_toml`)
3. Remove workaround functions from CLI
4. Update tests to verify ID assignment
5. Add test to verify C++/Python ID consistency

## Verification Plan

### Unit Tests

1. **Camera Loader Order Test**
```cpp
TEST_CASE("Camera IDs assigned in TOML order") {
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");

    // Verify deterministic IDs
    REQUIRE(cameras.at("cam1").id() == 0);
    REQUIRE(cameras.at("cam2").id() == 1);
    REQUIRE(cameras.at("cam3").id() == 2);
    REQUIRE(cameras.at("cam4").id() == 3);

    // Verify iteration order
    int expected_id = 0;
    for (auto const& [name, cam] : cameras) {
        REQUIRE(cam.id() == expected_id++);
    }
}
```

2. **Observation Camera ID Test**
```cpp
TEST_CASE("Observations have correct camera IDs") {
    auto cameras = load_cameras_from_toml("tests/data/cameras.toml");
    auto skeleton = load_skeleton_from_yaml("tests/data/skeleton.yaml");

    auto obs_seq = load_openpose_frame("tests/data/cam1_000000.json",
                                       cameras.at("cam1"),
                                       "cam1", skeleton, 0, 0.1, 0);

    // All observations should have camera_id matching camera.id()
    for (auto const& obs : obs_seq.observations) {
        REQUIRE(obs.camera_id == cameras.at("cam1").id());
    }
}
```

3. **Python Consistency Test**
```cpp
TEST_CASE("C++ and Python assign same camera IDs") {
    // Load same TOML file used by Python
    auto cameras = load_cameras_from_toml("path/to/python_test_cameras.toml");

    // Verify IDs match Python's expected assignment
    // (Can be validated by checking Python debug output)
    REQUIRE(cameras.at("cam1").id() == 0);
    // ... etc
}
```

### Integration Tests

1. **Frame 0 Debug Test**
   - After refactoring, re-run frame 0 comparison test
   - Verify observation counts match Python (should fix the 51 vs 115 inlier discrepancy)
   - Verify camera IDs in C++ debug output match Python debug output

2. **CLI End-to-End Test**
   - Run full tracking on test dataset
   - Compare output with Python tracker results
   - Verify synchronized frames and consistent results

## Implementation Plan

### Phase 1: Camera Loader (Low Risk)
- [ ] Change `load_cameras_from_toml()` return type to `std::map`
- [ ] Update header file
- [ ] Run camera loader tests
- [ ] Add new test for deterministic ID assignment

### Phase 2: Observation Loader (Medium Risk)
- [ ] Update `load_openpose_sequence()` signature to accept `std::map`
- [ ] Fix all `camera_id = 0` assignments to use `camera.id()`
- [ ] Update `load_openpose_frame()` to propagate camera ID
- [ ] Run observation loader tests
- [ ] Add test for correct observation camera IDs

### Phase 3: Remove CLI Workarounds (High Impact)
- [ ] Delete `convert_camera_map()` function
- [ ] Delete `update_observation_camera_ids()` function
- [ ] Simplify camera loading in `main()`
- [ ] Update sync metadata functions to accept `std::map`
- [ ] Build and test CLI application

### Phase 4: Update Tests (Low Risk)
- [ ] Update `test_ukf_frame0_comparison.cpp`
- [ ] Update `test_camera_loader.cpp`
- [ ] Update any other tests using camera loader
- [ ] Run full test suite

### Phase 5: Validation (Critical)
- [ ] Re-run frame 0 comparison test
- [ ] Verify C++ and Python observation counts match
- [ ] Compare C++ debug output with Python debug output
- [ ] Run full tracking comparison

## Risk Assessment

### Low Risk
- Camera loader change (isolated component)
- Test file updates (easy to revert)

### Medium Risk
- Observation loader changes (used in multiple places)
- API signature changes (requires updating multiple call sites)

### High Risk
- CLI changes (main user-facing application)
- Integration test failures (may reveal other issues)

### Mitigation
- Implement in phases with testing between each phase
- Keep git commits small and focused
- Maintain backward compatibility shims if needed
- Have rollback plan for each phase

## Alternative Approaches Considered

### Alternative 1: Add camera_name to Observation struct
**Approach:** Keep IDs as-is, use camera name for matching
**Rejected because:**
- Still need deterministic IDs for Tracker/UKF
- Adds redundant data to every observation
- Doesn't solve root cause of non-deterministic IDs

### Alternative 2: Use camera section name as ID
**Approach:** Use "cam1", "cam2" as integer IDs (parse number from name)
**Rejected because:**
- Fragile: assumes specific naming convention
- Breaks if TOML has non-standard names (e.g., "frontCam", "leftCam")
- Doesn't handle gaps (cam1, cam2, cam4 → what is cam3's ID?)

### Alternative 3: Keep unordered_map, sort before iteration
**Approach:** Sort keys before iterating unordered_map
**Rejected because:**
- Still requires workaround code everywhere
- Doesn't fix observation loader issues
- More error-prone than using correct container

## Success Criteria

### Must Have
- ✅ Camera IDs assigned deterministically based on TOML order
- ✅ All observations have correct camera_id from creation
- ✅ No ID reassignment workarounds in CLI
- ✅ C++ and Python assign identical camera IDs for same TOML file
- ✅ Frame 0 comparison test passes with matching observation counts

### Should Have
- ✅ All tests pass
- ✅ CLI runs without errors on test dataset
- ✅ Tracking results match Python output

### Nice to Have
- 📝 Documentation updated with camera ID assignment guarantees
- 📝 API reference clarifies iteration order
- 📝 Example code showing correct usage patterns

## References

- [toml++ documentation](https://marzer.github.io/tomlplusplus/) - Confirms toml::table iteration order
- [C++ std::map documentation](https://en.cppreference.com/w/cpp/container/map) - Ordered container
- Frame 0 debug status: `docs/frame0-debug-status.md`
- Frame 0 test plan: `docs/FRAME0_TEST_PLAN.md`
