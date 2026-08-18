# Contributing to PoseTrack

## Development Setup

1. Install dependencies (see README.md)
2. Set up build: `meson setup builddir`
3. Build: `meson compile -C builddir`
4. Run tests: `meson test -C builddir`

The above is the Linux/WSL path, where Pinocchio is installed system-wide via
robotpkg at `/opt/openrobots` (see `docs/pinocchio-header-only-analysis.md`).
For native Windows, see the next section — the same `meson setup`/`compile`/
`test` commands apply, just with a few extra options pointing at manually
supplied Pinocchio/Boost headers.

### Windows (native, MSVC)

**One-time setup:**

1. Visual Studio 2022 (or later) with the "Desktop development with C++"
   workload.
2. [Meson](https://mesonbuild.com/) and Ninja (`pip install meson ninja`, or
   the standalone Meson MSI installer).
3. Pinocchio 3.9.0 headers + a compiled Boost Serialization — easiest via a
   dedicated conda environment (this project only needs the headers/libs,
   not a Python installation of Pinocchio itself):
   ```powershell
   conda create -y -n posetrak-pinocchio -c conda-forge pinocchio=3.9.0
   ```
   **Match the version to whatever `/opt/openrobots` actually has** — check
   the [openrobots package list](https://robotpkg.openrobots.org/) or ask
   whoever maintains the Linux/WSL dev environment. A header-version
   mismatch shows up as real compile errors (e.g. `Frame::parentJoint` not
   existing — that field was `Frame::parent` before Pinocchio 3.0), not
   something that just happens to link with slightly wrong behaviour.
4. Run `setup-windows.ps1` (repo root). It configures and builds two build
   directories — `builddir/` (debug, for day-to-day unit testing/debugging)
   and `optbuild/` (release, for actual tracking runs — see the performance
   note in the main `CLAUDE.md`/project instructions) — and copies the two
   runtime DLLs described below next to each built `.exe`. The script is
   idempotent; re-run it after a `meson.build`/`meson_options.txt` change
   instead of hand-reconstructing the `-D...` flags below.

**Manually, if you'd rather not use the script** (or need to see exactly
what it does):
```powershell
$pinocchioEnv = "$env:USERPROFILE\miniconda3\envs\posetrak-pinocchio\Library"
meson setup builddir  -Dbuildtype=debug   -Ddefault_library=static `
  -Dpinocchio_includedir=$pinocchioEnv/include -Dboost_includedir=$pinocchioEnv/include -Dboost_libdir=$pinocchioEnv/lib
meson setup optbuild   -Dbuildtype=release -Ddefault_library=static `
  -Dpinocchio_includedir=$pinocchioEnv/include -Dboost_includedir=$pinocchioEnv/include -Dboost_libdir=$pinocchioEnv/lib
meson compile -C builddir
meson test    -C builddir
```

**Before running any built `.exe`**, copy the two runtime DLLs
(`boost_serialization.dll`, `yaml-cpp.dll` — everything else links
statically) into the same directory as the `.exe`:
```powershell
Copy-Item "$pinocchioEnv\bin\boost_serialization.dll", "$env:USERPROFILE\miniconda3\Library\bin\yaml-cpp.dll" -Destination builddir\cli
Copy-Item "$pinocchioEnv\bin\boost_serialization.dll", "$env:USERPROFILE\miniconda3\Library\bin\yaml-cpp.dll" -Destination builddir\tests
```
(and the same into `optbuild\cli` / `optbuild\tests`). **Colocate, don't put
on `PATH`**: Windows always searches an executable's own directory for its
DLL dependencies first, so this is what makes the binary runnable
standalone — including when the Python UI (`python/posetrak/tracker/runner.py`)
launches `posetrak-tracker.exe` as a subprocess, which inherits whatever
environment the UI happens to be running in and has no reason to know
about this conda environment. Forgetting this step doesn't fail loudly:
Windows kills the process at load time with `STATUS_DLL_NOT_FOUND`
(`0xC0000135` / `3221225781`), which turns up as a `libshiboken: Overflow`
warning when the huge exit code is marshalled back into a Qt `int` signal —
if you see that, this is almost always why.

**Why the extra options, if you're wondering what's Windows-specific and
why** (all gated behind `cpp.get_id() == 'msvc'` / a native, non-cross build
in `meson.build` — none of this affects the Linux/WSL or MinGW cross paths):
- `pinocchio_includedir`/`boost_includedir`/`boost_libdir` — Linux/WSL finds
  Pinocchio and Boost at fixed system paths (`/opt/openrobots`, the default
  compiler include path); native Windows has neither, so these point at the
  conda environment instead.
- `default_library=static` — this codebase has no `dllexport`/`dllimport`
  annotations, so as a shared library nothing would be exported and every
  consumer (the CLI, the tests) fails to link. The existing MinGW cross
  build (`cpp/cross/mingw-w64-x86_64.ini`) already works around this the same
  way, for the same reason.
- `NOMINMAX`, `WIN32=1`, `BOOST_ALL_NO_LIB`, `_USE_MATH_DEFINES`, `/bigobj`
  (debug only) — Pinocchio/Eigen/MSVC preprocessor and object-format quirks;
  see the comments beside each in `meson.build` for the specific error each
  one fixes.

**Alternative: MinGW cross-compile from WSL.** If you'd rather not maintain
a native Windows toolchain at all, `cpp/cross/mingw-w64-x86_64.ini` cross-compiles
a `posetrak-tracker.exe` from a Linux/WSL host (`apt install mingw-w64
libboost-dev`, then `meson setup winbuild --cross-file
cpp/cross/mingw-w64-x86_64.ini`) — see the comments at the top of that file for
the exact steps and the four runtime DLLs it needs alongside the `.exe`.

## Coding Standards

### C++ Style

- **Standard**: C++20 minimum (C++23 features where beneficial)
- **Formatting**: Use clang-format (configuration in `.clang-format`)
- **Naming conventions**:
  - `ClassNames`: PascalCase
  - `function_names()`: snake_case
  - `variable_names`: snake_case
  - `CONSTANT_VALUES`: UPPER_SNAKE_CASE
  - `member_variables_`: trailing underscore
  - `namespace posetrak`: lowercase

### Code Quality

- **No warnings**: Code must compile with `-Wall -Wextra` without warnings
- **No raw pointers**: Use smart pointers (unique_ptr, shared_ptr) or references
- **No manual memory management**: Use RAII
- **Const-correct**: Mark all const methods and parameters
  - **Use right const (east const)**: `uint32_t const&` not `const uint32_t&`
  - Consistently place `const` after the type for better readability
- **Error handling**: Use exceptions for errors, std::optional for missing data
- **Documentation**:
  - **Required for ALL functions**: Public, private, and implementation functions
  - Use Doxygen-style comments (`///` or `/** */`)
  - Document parameters with `@param`, return values with `@return`
  - Include brief description of what the function does
  - Explain non-obvious implementation details
- **Code formatting**: Run `clang-format` before committing
  - In project root: `clang-format -i cpp/src/**/*.cpp cpp/include/**/*.hpp cpp/tests/**/*.cpp`
  - Verify with: `git diff` before committing

### Documentation Organization

- **Keep root clean**: Don't put random documents in project root
- **Planning docs**: Use `docs/plans/<feature>/` for GenAI planning documents
  - Example: `docs/plans/phase-0/status.md`, `docs/plans/camera-model/design.md`
- **Architecture docs**: Use `docs/` for high-level architecture and design documents
- **Self-descriptive names**: Avoid short-lived terms in file names, comments, and commits
  - ❌ Bad: `phase0`, `tmp_fix`, `old_version`
  - ✅ Good: `project-setup`, `camera-initialization`, `legacy-distortion-model`
  - Reason: Code should be understandable years later without context

### Testing

- **Write tests first**: Or alongside implementation
- **Test coverage**: Aim for 80%+ coverage
- **Test naming**: `TEST_CASE("Clear description", "[tag]")`
- **Assertions**: Use testing framework assertions (REQUIRE, CHECK)
- **No memory leaks**: All tests should pass valgrind

### Example Code

```cpp
#include <posetrak/core/state.hpp>
#include <Eigen/Core>
#include <memory>

namespace posetrak {

class MyClass {
public:
    explicit MyClass(int value) : value_(value) {}

    int get_value() const { return value_; }
    void set_value(int value) { value_ = value; }

    // Right const examples:
    void process(Eigen::VectorXd const& input);  // const reference
    double const* get_data() const;              // const pointer

private:
    int value_;
};

}  // namespace posetrak
```

## Git Workflow

### Commit Messages

Format:
```
component: Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points are okay
- Use imperative mood ("Add feature" not "Added feature")
```

Component prefixes:
- `core:` - Core models (State, Skeleton, Camera, etc.)
- `kinematics:` - Forward/inverse kinematics
- `filters:` - UKF and related filtering code
- `tracking:` - Tracker implementation
- `io:` - Input/output, serialization
- `cli:` - Command-line interface
- `tests:` - Test code
- `build:` - Build system, dependencies
- `docs:` - Documentation

Examples:
```
core: Add Camera class with fisheye distortion support

filters: Implement UKF prediction step for joint-space state

tests: Add unit tests for Skeleton DOF validation
```

### Branch Strategy

- `main`: Stable code, passes all tests
- Feature branches: Short-lived, merge via PR (when we have team)

### Before Committing

```bash
# Build
meson compile -C builddir

# Run tests
meson test -C builddir

# Check formatting
clang-format -i cpp/src/**/*.cpp cpp/include/**/*.hpp

# Check for memory leaks (on new code)
meson test -C builddir --wrap='valgrind --leak-check=full'
```

## Implementation Workflow

We're following a phased implementation plan (see [docs/cpp-implementation-plan.md](docs/cpp-implementation-plan.md)). Planning documents for each phase are in `docs/plans/`.

**Note**: When referring to work in comments/commits, use descriptive names:
- Write "Setup build system" not "Phase 0 setup"
- Write "Implement camera distortion models" not "Phase 2 task 3"
- This ensures code remains understandable without external planning documents

### Task Checklist

Before moving to next phase:
- [ ] All tasks complete
- [ ] All tests pass
- [ ] No memory leaks
- [ ] Documentation updated
- [ ] Exit criteria met
- [ ] Code reviewed

## Questions?

Check the documentation in `docs/` or ask the team.
