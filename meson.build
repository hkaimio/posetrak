# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

project('posetrak', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'werror=false',  # Enable once code is stable
    'buildtype=release',
  ],
  license: 'MIT',
  meson_version: '>=0.63.0'
)

# Project information
project_description = 'High-performance motion capture tracker using UKF in joint space'

# Add project-wide compiler arguments
cpp = meson.get_compiler('cpp')
add_project_arguments(
  cpp.get_supported_arguments([
    '-Wno-unused-parameter',  # Common in interface implementations
  ]),
  language: 'cpp'
)

if cpp.get_id() == 'msvc'
  # M_PI and friends are not defined by <cmath> on MSVC unless this is set
  # before the first include -- the MinGW cross build needs the equivalent
  # define too (cross/mingw-w64-x86_64.ini), but that file only applies when
  # actually cross-compiling.
  add_project_arguments('-D_USE_MATH_DEFINES', language: 'cpp')

  # Debug builds only: heavy Eigen/Pinocchio template instantiation (e.g.
  # inverse_kinematics.cpp's use of computeJointJacobians) generates far more
  # COFF sections per object file in an unoptimized debug build than a
  # release build (no inlining/dead-code-elimination collapses them) --
  # observed hitting "fatal error C1128: number of sections exceeded object
  # file format limit" building builddir with -Dbuildtype=debug, the same
  # buildtype CLAUDE.md's own convention uses for day-to-day unit
  # testing/debugging (optbuild/ is the separate release build for actual
  # tracking runs). /bigobj raises that limit; only needed in debug because
  # release's optimizer already keeps section counts well under the default
  # limit for every file in this codebase.
  if get_option('buildtype') == 'debug'
    add_project_arguments('/bigobj', language: 'cpp')
  endif

  # Pinocchio's fwd.hpp does `#ifdef _WIN32: #include <windows.h>` unconditionally.
  # Without NOMINMAX, windows.h #defines min/max as function-like macros, which
  # would corrupt any std::min/std::max call or template `<`/`>` parse throughout
  # Pinocchio and Eigen. Standard, low-risk practice for any Windows C++ project
  # that pulls in windows.h transitively while also using std::min/max or
  # Eigen/Pinocchio-style templates -- kept even though it wasn't the actual
  # cause of a similar-looking cascade hit while debugging this (that one
  # turned out to be a missing PINOCCHIO_DEPRECATED definition instead).
  add_project_arguments('-DNOMINMAX', language: 'cpp')

  # pinocchio/macros.hpp does `#if WIN32` (bare, not _WIN32) to choose between
  # __FUNCSIG__ (MSVC) and __PRETTY_FUNCTION__ (GCC/Clang) for
  # PINOCCHIO_PRETTY_FUNCTION. MSVC does not predefine bare WIN32 (only
  # _WIN32), so without this the macro is completely undefined and `#if WIN32`
  # fails with "invalid integer constant expression" the moment macros.hpp is
  # ever included. The MinGW cross build needs the same define for a different
  # reason (cross/mingw-w64-x86_64.ini: GCC's own headers clear WIN32 after
  # Pinocchio's macros.hpp already needs it) -- __FUNCSIG__ genuinely exists on
  # MSVC, so unlike the cross build this needs no __FUNCSIG__ remap.
  add_project_arguments('-DWIN32=1', language: 'cpp')

  # Some Pinocchio header transitively includes Boost Serialization, which
  # (unlike the header-only boost/math, boost/type_traits this project
  # otherwise relies on) triggers Boost's MSVC-only "auto-link" feature: a
  # `#pragma comment(lib, "boost_serialization.lib")` requesting the linker
  # pull in a compiled library, purely because MSVC honours that pragma and
  # GCC/Clang silently ignore it -- the Linux/WSL build links fine without
  # ever providing that library. Rather than chase down which header pulls
  # this in and hunt a matching compiled Boost lib, disable the auto-link
  # feature outright: nothing this project actually calls into needs Boost
  # Serialization's compiled support, only its header-only pieces.
  add_project_arguments('-DBOOST_ALL_NO_LIB', language: 'cpp')

  # Without an explicit /arch: flag, MSVC only targets the x64-guaranteed
  # SSE2 baseline (128-bit vector registers, 2 doubles/instruction), so
  # Eigen's auto-vectorization -- used throughout the UKF's matrix/vector
  # math -- never gets past that. /arch:AVX2 lets Eigen use 256-bit AVX2/FMA
  # codepaths instead (4 doubles/instruction plus fused multiply-add),
  # purely a compile-time codegen change, no source changes needed. AVX2
  # has been present on essentially all x86_64 CPUs sold since ~2013
  # (Haswell onward); not using AVX-512 since it's still absent on some
  # current CPUs (notably AMD before Zen 4) and support is inconsistent
  # enough to not be a safe default the way AVX2 now is.
  add_project_arguments('/arch:AVX2', language: 'cpp')
endif

# Dependencies
# Strategy: Use meson wraps (wrapdb) when possible, system packages only when necessary
message('Finding dependencies...')

# Eigen (header-only, use wrap)
eigen_dep = dependency('eigen3',
  version: '>=3.4',
  required: false,
  fallback: ['eigen', 'eigen_dep']
)
if not eigen_dep.found()
  eigen_dep = dependency('eigen3', required: true)
endif

# fmt (use wrap)
fmt_dep = dependency('fmt',
  version: '>=10.0',
  required: false,
  fallback: ['fmt', 'fmt_dep']
)
if not fmt_dep.found()
  fmt_dep = dependency('fmt', required: true)
endif

# nlohmann_json (header-only, use wrap)
nlohmann_json_dep = dependency('nlohmann_json',
  version: '>=3.11',
  required: false,
  fallback: ['nlohmann_json', 'nlohmann_json_dep']
)
if not nlohmann_json_dep.found()
  nlohmann_json_dep = dependency('nlohmann_json', required: true)
endif

# yaml-cpp -- always build from the vendored subproject (skip system-package
# lookup entirely). A system/conda-provided yaml-cpp on a Windows dev machine
# was found to have a YAML::Load(std::string const&) that silently returns an
# empty node for nested block sequences (YAML::LoadFile is unaffected),
# breaking load_skeleton_from_yaml_string() -- used by every DB-driven
# tracker run -- on any real skeleton with joints/markers. Discovered via
# tests/test_multi_person_tracker.cpp; see error-improvements Phase 5 history.
# The other dependency() + fallback: [...] blocks below still prefer a system
# package when present -- only yaml-cpp has shown this failure mode so far.
yaml_cpp_proj = subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp_proj.get_variable('yaml_cpp_dep')

# tomlplusplus (use wrap - toml11 not in wrapdb)
# The tomlplusplus subproject defaults to shared_library; override to static so
# the Windows build does not produce a libtomlplusplus-3.dll that must be
# distributed alongside the exe.
toml_dep = dependency('tomlplusplus',
  required: false,
  fallback: ['tomlplusplus', 'tomlplusplus_dep'],
  default_options: ['default_library=static']
)
if not toml_dep.found()
  toml_dep = dependency('tomlplusplus', required: true)
endif

# SQLite3 (system library with wrap fallback for cross-compilation)
sqlite3_dep = dependency('sqlite3',
  required: false,
  fallback: ['sqlite3', 'sqlite3_dep'],
)
if not sqlite3_dep.found()
  sqlite3_dep = dependency('sqlite3', required: true)
endif

# CLI11 (header-only, use wrap)
cli11_dep = dependency('CLI11',
  version: '>=2.3',
  required: false,
  fallback: ['cli11', 'CLI11_dep']
)
if not cli11_dep.found()
  cli11_dep = dependency('CLI11', required: true)
endif

# Pinocchio - header-only mode
# All algorithm headers expose full template bodies when
# PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION is NOT defined, so no libpinocchio_default.so
# linking is required.  The only non-Eigen dependencies (boost/math/constants and
# boost/type_traits) are themselves header-only and present at the system include path.
# See docs/pinocchio-header-only-analysis.md for the full rationale.
#
# One exception found bringing up a native Windows build against Pinocchio
# 3.9.0 (the version /opt/openrobots actually has -- Frame::parentJoint,
# renamed from Frame::parent post-2.7.1, is what confirmed the version):
# something transitively pulls in boost::archive::codecvt_null<wchar_t>,
# which needs actually-compiled Boost Serialization, not just headers. The
# Linux/WSL build links fine without ever providing it explicitly -- Boost's
# MSVC-only auto-link pragma (a `#pragma comment(lib, ...)` GCC/Clang simply
# ignore) would normally request it there by mangled name, but that name
# doesn't reliably match what a given Boost distribution actually calls the
# file, so BOOST_ALL_NO_LIB disables the pragma and boost_libdir below links
# the real file explicitly instead.
#
# Default assumes a robotpkg install at /opt/openrobots (the Linux/WSL dev
# environment), where Boost is also on the compiler's default system include
# path. Native Windows has neither: pinocchio_includedir/boost_includedir let
# a native Windows build point at manually-placed header copies instead.
pinocchio_includedir_opt = get_option('pinocchio_includedir')
if pinocchio_includedir_opt != ''
  pinocchio_inc = include_directories(pinocchio_includedir_opt, is_system: true)
else
  pinocchio_inc = include_directories('/opt/openrobots/include', is_system: true)
endif

pinocchio_extra_incs = []
boost_includedir_opt = get_option('boost_includedir')
if boost_includedir_opt != ''
  pinocchio_extra_incs += include_directories(boost_includedir_opt, is_system: true)
endif

if meson.is_cross_build()
  # Pinocchio's templates pull in Boost (math, type_traits).  The cross-compiler
  # does not search the Linux host's /usr/include, but adding it wholesale would
  # contaminate the build with Linux glibc headers (bits/libc-header-start.h etc).
  # cross/include/ contains only a symlink to /usr/include/boost — the pure-C++
  # Boost headers — so only those are exposed to the MinGW compiler.
  # Requires: apt install libboost-dev  (symlink: cross/include/boost -> /usr/include/boost)
  pinocchio_extra_incs += include_directories('cross/include', is_system: true)
endif

pinocchio_deps = []
boost_libdir_opt = get_option('boost_libdir')
if boost_libdir_opt != ''
  pinocchio_deps += cpp.find_library('boost_serialization', dirs: [boost_libdir_opt])
endif

pinocchio_dep = declare_dependency(
  include_directories: [pinocchio_inc] + pinocchio_extra_incs,
  dependencies: pinocchio_deps,
)

# Will add these as we progress:
# opencv_dep = dependency('opencv4', version: '>=4.5',
#                        modules: ['core', 'imgproc', 'videoio', 'calib3d'],
#                        required: false)
# archive_dep = dependency('libarchive', required: false)
openmp_dep = dependency('openmp', required: true)

# Testing dependency (optional, selected by user preference)
if get_option('enable_tests')
  test_framework = get_option('test_framework')
  if test_framework == 'catch2'
    test_dep = dependency('catch2-with-main',
      version: '>=3.0',
      required: false,
      fallback: ['catch2', 'catch2_with_main_dep']
    )
    if not test_dep.found()
      test_dep = dependency('catch2-with-main', required: true)
    endif
  elif test_framework == 'gtest'
    test_dep = dependency('gtest',
      main: true,
      required: false,
      fallback: ['gtest', 'gtest_main_dep']
    )
    if not test_dep.found()
      test_dep = dependency('gtest', main: true, required: true)
    endif
  endif
endif

# Include directories
posetrak_inc = include_directories('include')

# Subdirectories
subdir('src')
subdir('cli')

if get_option('enable_tests')
  subdir('tests')
endif

# Summary
summary({
  'prefix': get_option('prefix'),
  'bindir': get_option('bindir'),
  'libdir': get_option('libdir'),
  'includedir': get_option('includedir'),
  'cpp_std': get_option('cpp_std'),
  'buildtype': get_option('buildtype'),
  'tests': get_option('enable_tests'),
}, section: 'Configuration')

summary({
  'eigen3': eigen_dep.found(),
  'fmt': fmt_dep.found(),
  'nlohmann_json': nlohmann_json_dep.found(),
  'yaml-cpp': yaml_cpp_dep.found(),
  'tomlplusplus': toml_dep.found(),
  'CLI11': cli11_dep.found(),
  'pinocchio': pinocchio_dep.found(),
  'sqlite3': sqlite3_dep.found(),
}, section: 'Dependencies')
