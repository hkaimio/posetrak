# PoseTrack - High-Performance Motion Capture Tracker

Fast, accurate multi-camera motion capture using Unscented Kalman Filter in joint space.

## Features

- **Joint-space tracking**: Track skeletal motion directly in joint angles (not per-marker)
- **Multi-camera support**: Arbitrary number of synchronized cameras with fisheye support
- **Robust outlier rejection**: Mahalanobis distance-based filtering
- **Configurable skeletons**: Support any skeleton structure (not hardcoded)
- **High performance**: 20-50× faster than Python prototype (30-60 Hz for 120 DOF)
- **Modern C++20/23**: Clean, maintainable codebase

## Building from Source

### Dependencies

All dependencies are managed via [Meson WrapDB](https://mesonbuild.com/Wrapdb-projects.html) for reproducible builds. No system packages required except for future dependencies (Pinocchio, OpenCV).

**Current dependencies** (via wraps):
- Eigen3 5.0.1 - Linear algebra
- fmt 12.0.0 - String formatting
- nlohmann-json 3.11+ - JSON parsing
- yaml-cpp 0.8.0 - YAML config files
- tomlplusplus 3.4.0 - TOML calibration files
- CLI11 2.6.1 - Command-line parsing
- Catch2 3.12.0 - Unit testing

**Future dependencies** (system packages):
- Pinocchio (>=3.9) - for kinematics
- OpenCV (>=4.5) - for camera distortion and video I/O
- libarchive - for ZIP export

### Building

```bash
# Clone repository
git clone <repo-url> posetrak
cd posetrak

# Set up build directory (automatically downloads dependencies via wraps)
meson setup builddir

# Build
meson compile -C builddir

# Run tests
./run_tests.sh   # Use this script to avoid conda library conflicts
# OR
meson test -C builddir  # Direct command (may fail if conda is active)
```

### Known Issues

- **Conda environment conflicts**: If you have conda/miniconda active, the test executable may fail with `GLIBCXX` version errors. Use `./run_tests.sh` which prioritizes system libraries, or deactivate conda before running tests.

### Build Options

```bash
# Use GTest instead of Catch2
meson setup builddir -Dtest_framework=gtest

# Disable tests
meson setup builddir -Denable_tests=false

# Debug build
meson setup builddir --buildtype=debug
```

## Quick Start

*(Coming in Phase 8)*

```bash
posetrak track \
    --skeleton skeleton.yaml \
    --calib calibration.toml \
    --base-dir data/aikido \
    --output-dir results/
```

## Project Status

🚧 **Currently in Phase 0-1: Core implementation** 🚧

- [x] Phase 0: Project setup
- [ ] Phase 1: Core models (in progress)
- [ ] Phase 2: I/O layer
- [ ] Phase 3: Kinematics with Pinocchio
- [ ] Phase 4: Basic UKF & tracking
- [ ] Phase 5+: Advanced features

See [docs/cpp-implementation-plan.md](docs/cpp-implementation-plan.md) for detailed roadmap.

## Documentation

- [Requirements](docs/cpp-requirements.md) - Functional and non-functional requirements
- [Architecture Overview](docs/cpp-architecture-overview.md) - System design
- [Detailed Architecture](docs/cpp-detailed-architecture.md) - Implementation details
- [Implementation Plan](docs/cpp-implementation-plan.md) - Phased development plan

## Development

### Project Structure

```
posetrak/
├── include/posetrak/    # Public headers
│   ├── core/           # State, Skeleton, Camera, Observation
│   ├── kinematics/     # FK, IK, triangulation
│   ├── filters/        # UKF, process models, outlier rejection
│   ├── tracking/       # Tracker orchestration
│   └── io/             # Loaders and exporters
├── src/                # Implementation (.cpp files)
├── tests/              # Unit and integration tests
├── cli/                # Command-line tool
├── examples/           # Example skeletons and data
└── docs/               # Documentation
```

### Coding Standards

- C++20 standard
- Use Eigen for all linear algebra
- Modern C++ features (ranges, concepts, std::optional, etc.)
- Value semantics where possible
- Clear error messages with context
- Comprehensive tests alongside implementation

### Running Tests

```bash
# All tests
meson test -C builddir

# Verbose output
meson test -C builddir -v

# Specific test
meson test -C builddir test_name

# With valgrind (memory leaks)
meson test -C builddir --wrap='valgrind --leak-check=full'
```

## Contributing

This is currently a focused implementation project. Contributions will be welcome once core functionality is stable (Phase 10+).

## License

MIT License (see LICENSE file)

## Acknowledgments

Based on Python prototype with insights from:
- UKF in joint space for motion capture
- Pinocchio for efficient kinematics
- Modern C++ best practices
