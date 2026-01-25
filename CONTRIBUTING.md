# Contributing to PoseTrack

## Development Setup

1. Install dependencies (see README.md)
2. Set up build: `meson setup builddir`
3. Build: `meson compile -C builddir`
4. Run tests: `meson test -C builddir`

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
- **Documentation**: Doxygen-style comments for public APIs

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
clang-format -i src/**/*.cpp include/**/*.hpp

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
