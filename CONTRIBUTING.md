# Contributing to PoseTrack

## Development Setup

See the **[Setup guide](docs/setup.md)** for prerequisites and build
instructions on both platforms, including the native-MSVC and cross-compiled
paths for Windows. There's no separate contributor setup doc — it's the same
one anyone building Posetrak from source uses.

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
comp: Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points are okay
- Use imperative mood ("Add feature" not "Added feature")
```

`comp` is whatever area the change affects — often a source directory
(`core`, `kinematics`, `filters`, `tracking`, `cli`) or a general area
(`doc`, `test`, `build`, `setup`, `ci`). There's no fixed list; pick
whatever a reader would recognize years from now. Don't use ephemeral
planning names from a task's own planning docs (`phase0`, `step3`) —
commit messages record project history and must stay understandable
without that context.

Examples:
```
core: Add Camera class with fisheye distortion support

filters: Implement UKF prediction step for joint-space state

test: Add unit tests for Skeleton DOF validation
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

Feature and task planning documents live under `docs/plans/<feature>/` and
`docs/roadmap/`. See [docs/roadmap.md](docs/roadmap.md) for where the
project is headed overall.

**Note**: When referring to work in comments/commits, use descriptive names:
- Write "Setup build system" not "Phase 0 setup"
- Write "Implement camera distortion models" not "Phase 2 task 3"
- This ensures code remains understandable without external planning documents

### Task Checklist

Before considering a task done:
- [ ] All tasks complete
- [ ] All tests pass
- [ ] No memory leaks
- [ ] Documentation updated
- [ ] Exit criteria met
- [ ] Code reviewed

## Questions?

Check the documentation in `docs/` or ask the team.
