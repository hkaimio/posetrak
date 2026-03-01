# Pinocchio Header-Only Mode Analysis

## Question

Can posetrak use Pinocchio in header-only mode, eliminating the need for installed shared
libraries?  Would this make a Windows port easier?

## Answer: Yes, Fully Achievable

### How "header-only" works in Pinocchio

Pinocchio uses the **`.hpp` declares, `.hxx` implements** template pattern.  All algorithm
`.hpp` files end with `#include "algorithm/foo.hxx"`, where the `.hxx` contains the full
template function bodies.  All 35 algorithm implementation files follow this pattern.
Nothing is compiled into a library unless you explicitly request it.

The flag that switches modes is `PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION`.  The current
pkg-config entry for the installed package declares:

```
Cflags: -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION
Libs:   -lpinocchio_default -lpinocchio_parsers -lpinocchio_casadi
        -lboost_filesystem -lboost_serialization -lboost_system
```

That macro instructs the headers to omit template bodies (they are pre-compiled in
`libpinocchio_default.so`).  **Drop the macro and the linker flags — everything becomes
header-only.**

---

## Pinocchio APIs Used by Posetrak

Every single Pinocchio header currently included is a template; none require the compiled
`.so`.

| Header | Symbol | Header-only? |
|---|---|---|
| `algorithm/kinematics.hpp` | `forwardKinematics()` | ✅ full template |
| `algorithm/frames.hpp` | `updateFramePlacements()` | ✅ full template |
| `algorithm/joint-configuration.hpp` | `integrate()` | ✅ full template |
| `algorithm/jacobian.hpp` | `computeJointJacobians()` | ✅ full template |
| `multibody/model.hpp` | `pinocchio::Model` | ✅ template class |
| `multibody/data.hpp` | `pinocchio::Data` | ✅ template class |
| `multibody/joint/joint-free-flyer.hpp` | `JointModelFreeFlyer` | ✅ template |
| `multibody/joint/joint-spherical.hpp` | `JointModelSpherical` | ✅ template |
| `multibody/joint/joint-revolute.hpp` | `JointModelRX/RY/RZ` | ✅ template |

---

## Remaining Dependencies in Header-Only Mode

| Dependency | Why needed | Header-only? | Windows (vcpkg)? |
|---|---|---|---|
| **Eigen3** | Matrix math throughout | ✅ header-only | ✅ trivial |
| **boost::math::constants** | `pi<Scalar>()` in `pinocchio/math/fwd.hpp` | ✅ header-only | ✅ trivial |
| **boost::type_traits** | `is_floating_point<T>` in same header | ✅ header-only | ✅ trivial |
| ~~urdfdom~~ | URDF parser (`_parsers` lib only) | eliminated | eliminated |
| ~~casadi~~ | CasADi symbolic math (`_casadi` lib only) | eliminated | eliminated |
| ~~boost_filesystem/serialization~~ | Used by parsers/serialization only | eliminated | eliminated |

The two Boost components actually required (`boost/math/constants/constants.hpp` and
`boost/type_traits/is_floating_point.hpp`) are themselves header-only — they link against
nothing.  On Windows: `vcpkg install boost-math boost-type-traits`.

---

## Windows Portability of Pinocchio Headers

Checked for POSIX-specific includes (`unistd.h`, `sys/`, `linux/`) in:

- `pinocchio/math/fwd.hpp`
- `pinocchio/spatial/fwd.hpp`
- `pinocchio/multibody/model.hpp` / `data.hpp`
- All algorithm `.hxx` files used above

**None found.**  Pinocchio's template core is written as portable C++14/17.  The project
maintains Windows CI in its upstream repo.

---

## Compile-Time Tradeoff

Without `PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION` every translation unit that includes a
Pinocchio algorithm header instantiates all templates locally.  In practice:

- `forward_kinematics.cpp` and `ukf.cpp` become slower to compile (CRTP-heavy templates).
- Mitigation: a single `src/pinocchio_impl.cpp` that includes all the algorithm headers and
  acts as an explicit-instantiation translation unit, combined with a precompiled header
  covering the joint type headers.  This is ~100 lines and recovers most of the compile-time
  advantage of the pre-compiled `.so`.

---

## Required Build System Changes

1. Remove the `pkg-config('pinocchio')` / `dependency('pinocchio')` call from `meson.build`.
2. Replace with a plain `include_directories` pointing at the Pinocchio headers (system
   install, subproject wrap, or vcpkg toolchain file on Windows).
3. Do **not** define `PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION`.
4. Add Eigen3 and the two Boost header-only deps (Meson has a `boost` module; vcpkg handles
   both on Windows).

For a fully self-contained Windows build, ship Pinocchio headers + Eigen3 + Boost headers
as `subprojects/` wraps.  The resulting executable has **zero runtime shared-library
dependencies** beyond the C++ standard library — an extremely clean Windows deployment.

---

## Recommendation

Switch to header-only mode.  The change is small (a few lines in `meson.build`), eliminates
the hardest installation step (building/installing `libpinocchio_default.so` and its
transitive deps), and opens a straightforward path to Windows support via vcpkg or bundled
subproject wraps.
