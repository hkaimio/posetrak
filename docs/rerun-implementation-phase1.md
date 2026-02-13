# Rerun Phase 1 Implementation Guide

**Goal**: Get 3D markers visualized in Rerun viewer

**Estimated time**: 2-3 days

---

## Prerequisites

### 1. Add Rerun C++ SDK Dependency

**meson.build** (root):
```meson
# Add Rerun dependency
rerun_dep = dependency('rerun_sdk', required: true)
# Or as subproject if not in system:
# rerun_proj = subproject('rerun_sdk')
# rerun_dep = rerun_proj.get_variable('rerun_dep')
```

**Alternative**: Manual subproject wrap (if not available via pkg-config)
```bash
# Create wrap file: subprojects/rerun_sdk.wrap
[wrap-git]
url = https://github.com/rerun-io/rerun
revision = v0.20.0
depth = 1

[provide]
dependency_names = rerun_sdk
```

### 2. Verify Rerun Installation

```bash
# Check if rerun viewer is installed
rerun --version

# If not, install:
pip install rerun-sdk
# or
cargo install rerun-cli
```

---

## Step 1.1: Basic RerunLogger Class (0.5 days)

### Files to Create

**include/posetrak/visualization/rerun_logger.hpp**:
```cpp
#pragma once

#include <rerun.hpp>
#include <memory>
#include <string>
#include <map>
#include <vector>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/camera.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/state.hpp"

namespace posetrak {

struct RerunConfig {
    bool enabled = false;
    std::string output_path = "tracking.rrd";
    bool live_streaming = false;
    std::string live_address = "127.0.0.1:9876";
    std::string application_id = "posetrak";
    std::string recording_id;
};

class RerunLogger {
public:
    RerunLogger(RerunConfig const& config,
                Skeleton const& skeleton,
                std::unordered_map<int, Camera> const& cameras);

    ~RerunLogger();

    bool enabled() const { return config_.enabled; }

    // Timeline management
    void log_frame_start(int frame, double timestamp);

    // Phase 1 Step 1.2: 3D Markers
    void log_markers_3d(std::string const& entity_path,
                       std::map<std::string, Eigen::Vector3d> const& markers,
                       std::vector<uint8_t> const& color);  // RGB

    void flush();

private:
    RerunConfig config_;
    Skeleton const& skeleton_;
    std::unordered_map<int, Camera> const& cameras_;
    std::unique_ptr<rerun::RecordingStream> rec_;

    int current_frame_ = -1;
    double current_timestamp_ = 0.0;
};

}  // namespace posetrak
```

**src/visualization/rerun_logger.cpp** (minimal):
```cpp
#include "posetrak/visualization/rerun_logger.hpp"
#include <fmt/core.h>

namespace posetrak {

RerunLogger::RerunLogger(RerunConfig const& config,
                         Skeleton const& skeleton,
                         std::unordered_map<int, Camera> const& cameras)
    : config_(config), skeleton_(skeleton), cameras_(cameras) {

    if (!config_.enabled) {
        return;
    }

    // Initialize recording
    rec_ = std::make_unique<rerun::RecordingStream>(config_.application_id);

    if (config_.live_streaming) {
        rec_->connect(config_.live_address).throw_on_failure();
        fmt::print("Rerun: Streaming to {}\n", config_.live_address);
    } else {
        rec_->save(config_.output_path).throw_on_failure();
        fmt::print("Rerun: Saving to {}\n", config_.output_path);
    }

    // Set recording ID if provided
    if (!config_.recording_id.empty()) {
        // TODO: Set recording ID (check Rerun API)
    }
}

RerunLogger::~RerunLogger() {
    if (rec_) {
        flush();
    }
}

void RerunLogger::log_frame_start(int frame, double timestamp) {
    if (!config_.enabled || !rec_) return;

    current_frame_ = frame;
    current_timestamp_ = timestamp;

    // Set dual timelines
    rec_->set_time_sequence("frame", frame);
    rec_->set_time_seconds("timestamp", timestamp);
}

void RerunLogger::log_markers_3d(std::string const& entity_path,
                                std::map<std::string, Eigen::Vector3d> const& markers,
                                std::vector<uint8_t> const& color) {
    if (!config_.enabled || !rec_) return;

    // TODO: Implement in Step 1.2
}

void RerunLogger::flush() {
    if (rec_) {
        // Rerun auto-flushes, but can call explicitly if needed
    }
}

}  // namespace posetrak
```

**src/visualization/meson.build**:
```meson
visualization_sources = files(
    'rerun_logger.cpp',
)
```

**Update src/meson.build**:
```meson
subdir('visualization')

# Add to main library sources
sources += visualization_sources
```

### Add to TrackerConfig

**include/posetrak/core/config.hpp**:
```cpp
#include "posetrak/visualization/rerun_logger.hpp"  // Add this

struct TrackerConfig {
    // ... existing fields ...

    RerunConfig rerun;
};
```

**src/core/config.cpp** (in `load()` function):
```cpp
// Parse [rerun] section
if (toml_config.contains("rerun")) {
    auto rerun_table = toml_config["rerun"];
    config.rerun.enabled = rerun_table["enabled"].value_or(false);
    config.rerun.output_path = rerun_table["output_path"].value_or(std::string("tracking.rrd"));
    config.rerun.live_streaming = rerun_table["live_streaming"].value_or(false);
    config.rerun.live_address = rerun_table["live_address"].value_or(std::string("127.0.0.1:9876"));
    config.rerun.application_id = rerun_table["application_id"].value_or(std::string("posetrak"));
    config.rerun.recording_id = rerun_table["recording_id"].value_or(std::string(""));
}
```

### Test Compilation

```bash
cd /home/harri/projects/posetrak
meson compile -C builddir
```

**Expected**: Compiles successfully (even though `log_markers_3d` is empty)

---

## Step 1.2: 3D Marker Logging (1 day) ⭐ **START HERE**

### Implementation

**src/visualization/rerun_logger.cpp** - Complete `log_markers_3d`:
```cpp
void RerunLogger::log_markers_3d(std::string const& entity_path,
                                std::map<std::string, Eigen::Vector3d> const& markers,
                                std::vector<uint8_t> const& color) {
    if (!config_.enabled || !rec_ || markers.empty()) return;

    // Extract positions
    std::vector<rerun::Position3D> positions;
    std::vector<std::string> labels;
    positions.reserve(markers.size());
    labels.reserve(markers.size());

    for (auto const& [name, pos] : markers) {
        positions.push_back({static_cast<float>(pos.x()),
                            static_cast<float>(pos.y()),
                            static_cast<float>(pos.z())});
        labels.push_back(name);
    }

    // Create Points3D archetype
    auto points = rerun::Points3D(positions)
        .with_colors({rerun::Color(color[0], color[1], color[2])})
        .with_radii({0.015f})  // 15mm radius
        .with_labels(labels);

    // Log to entity path
    rec_->log(entity_path, points);
}
```

### Integration with Tracker

**include/posetrak/tracking/tracker.hpp**:
```cpp
#include "posetrak/visualization/rerun_logger.hpp"  // Add this

class Tracker {
public:
    // ... existing methods ...

    void set_rerun_logger(std::shared_ptr<RerunLogger> logger) {
        rerun_logger_ = logger;
    }

private:
    std::shared_ptr<RerunLogger> rerun_logger_;
    // ... existing fields ...
};
```

**src/tracking/tracker.cpp** - Add logging calls:
```cpp
TrackingResult Tracker::track_frame(std::vector<Observation> const& observations,
                                   double timestamp) {
    // ... existing code ...

    // Frame start
    if (rerun_logger_ && rerun_logger_->enabled()) {
        rerun_logger_->log_frame_start(current_frame_, timestamp);
    }

    // ... predict step ...

    // Log predicted markers (after predict, before update)
    if (rerun_logger_ && rerun_logger_->enabled()) {
        auto predicted_markers = fk_->compute(ukf_->state());
        rerun_logger_->log_markers_3d(
            "world/person_0/markers/predicted",
            predicted_markers,
            {200, 0, 200}  // Purple
        );
    }

    // ... update step ...

    // Log posterior markers (after update)
    if (rerun_logger_ && rerun_logger_->enabled()) {
        auto posterior_markers = fk_->compute(ukf_->state());
        rerun_logger_->log_markers_3d(
            "world/person_0/markers/posterior",
            posterior_markers,
            {0, 255, 0}  // Green
        );
    }

    // ... rest of existing code ...
}
```

**cli/track.cpp** - Create and attach logger:
```cpp
int main(int argc, char* argv[]) {
    // ... existing setup ...

    // Create Rerun logger if enabled
    std::shared_ptr<RerunLogger> rerun_logger;
    if (config.rerun.enabled) {
        rerun_logger = std::make_shared<RerunLogger>(
            config.rerun, skeleton, cameras
        );
        tracker.set_rerun_logger(rerun_logger);
        fmt::print("Rerun visualization enabled\n");
    }

    // ... rest of tracking ...
}
```

### Configuration File

**posetrak_config.toml** (or test config):
```toml
[rerun]
enabled = true
output_path = "tracking_output/tracking.rrd"
live_streaming = false
application_id = "posetrak"
recording_id = "test_run_001"
```

### Test Run

```bash
# Compile
meson compile -C builddir

# Run on test sequence (short)
./builddir/cli/posetrak tests/cpp-python/cpp_test_config.toml

# View results
rerun tracking_tests/full-alpha-0_1/tracking.rrd
```

**Expected Output**:
- Rerun viewer opens
- Timeline shows frames 0-N
- `world/person_0/markers/predicted` shows purple points
- `world/person_0/markers/posterior` shows green points
- Points animate over time as you scrub timeline
- Marker names visible as labels (hover over points)

---

## Troubleshooting

### Issue: "rerun_sdk dependency not found"

**Solution 1**: Install Rerun SDK system-wide
```bash
# Download Rerun C++ SDK from releases
# https://github.com/rerun-io/rerun/releases
```

**Solution 2**: Use Python installation via CMake's rerun-sdk
```bash
pip install rerun-sdk
export CMAKE_PREFIX_PATH=$HOME/.local/lib/python3.11/site-packages/rerun_sdk
```

### Issue: Markers not visible in 3D view

**Checklist**:
- [ ] Are timelines being set? (`log_frame_start` called?)
- [ ] Are marker positions valid? (not NaN/inf)
- [ ] Is color correct? (RGB 0-255)
- [ ] Check entity path: `world/person_0/markers/posterior` (case-sensitive)
- [ ] Try increasing radius: `.with_radii({0.03f})` (30mm)

### Issue: Compilation errors with Rerun types

**Common fixes**:
- Include `<rerun.hpp>` not `<rerun/rerun.hpp>`
- Use `rerun::Position3D` not `Position3D`
- Colors are `rerun::Color(r, g, b)` (uint8)
- Check Rerun C++ SDK version (>= 0.18.0)

---

## Next Steps (After Phase 1 Step 1.2 Works)

Once you can see animated markers in Rerun:

### Step 1.3: Camera Setup (0.5 days)
- Log camera pinhole models
- Log camera transforms (extrinsics)
- See camera frustums in 3D view

### Step 1.4: Initialization Logging
- Log triangulated markers at frame 0
- Log skeleton rest pose (just markers, not full hierarchy yet)

### Validation
- Scrub timeline: markers should move smoothly
- Compare predicted (purple) vs posterior (green): should be close
- Check marker labels: should match skeleton

---

## Success Criteria for Phase 1

✅ **Minimal Success**: Green posterior markers visible and animating
✅ **Good Success**: Both predicted and posterior markers, different colors
✅ **Full Success**: Cameras visible, timelines work, labels correct

**Time to Phase 2**: When you can debug tracking by watching marker motion in 3D
