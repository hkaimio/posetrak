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

# yaml-cpp (use wrap)
yaml_cpp_dep = dependency('yaml-cpp',
  required: false,
  fallback: ['yaml-cpp', 'yaml_cpp_dep']
)
if not yaml_cpp_dep.found()
  yaml_cpp_dep = dependency('yaml-cpp', required: true)
endif

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
# When cross-compiling for Windows the MinGW toolchain does not search the Linux
# host's /usr/include, but Pinocchio's Boost dependencies live there.  They are
# pure C++ templates with no OS-specific API calls, so including them from the
# host is safe and correct for the Windows target.
pinocchio_inc = include_directories('/opt/openrobots/include', is_system: true)
if meson.is_cross_build()
  # Pinocchio's templates pull in Boost (math, type_traits).  The cross-compiler
  # does not search the Linux host's /usr/include, but adding it wholesale would
  # contaminate the build with Linux glibc headers (bits/libc-header-start.h etc).
  # cross/include/ contains only a symlink to /usr/include/boost — the pure-C++
  # Boost headers — so only those are exposed to the MinGW compiler.
  # Requires: apt install libboost-dev  (symlink: cross/include/boost -> /usr/include/boost)
  pinocchio_dep = declare_dependency(
    include_directories: [
      pinocchio_inc,
      include_directories('cross/include', is_system: true),
    ],
  )
else
  pinocchio_dep = declare_dependency(
    include_directories: pinocchio_inc,
  )
endif

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
