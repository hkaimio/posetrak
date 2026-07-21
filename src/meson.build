subdir('db')

# Generated version header — rebuilt on every compile to capture current git state.
# Including it as a source makes Meson add its output dir to the compiler include path,
# so src/version.cpp can #include "posetrak_build_version.hpp" without an explicit -I flag.
build_version_h = custom_target('build_version_h',
  command: [find_program('python3'), meson.project_source_root() / 'scripts/gen_version.py', '@OUTPUT@'],
  output: 'posetrak_build_version.hpp',
  build_always_stale: true,
)

# Core library source files
posetrak_sources = files(
  # Core models (Phase 1)
  'core/state.cpp',
  'core/skeleton.cpp',
  'core/skeleton_layout.cpp',
  'core/camera.cpp',
  'core/observation.cpp',
  'core/config.cpp',

  # Kinematics (Phase 3)
  'kinematics/pinocchio_model_builder.cpp',
  'kinematics/forward_kinematics.cpp',
  'kinematics/triangulation.cpp',
  'kinematics/inverse_kinematics.cpp',

  # Core models (Phase 0c)
  'core/skeleton_state.cpp',

  # Filters (Phase 4)
  'filters/process_model.cpp',
  'filters/sigma_points.cpp',
  'filters/ukf.cpp',
  'filters/rts_smoother.cpp',
  # 'filters/subset_ukf.cpp',  # SubsetUKF scheduled for deletion (Phase 3c)

  # Tracking (Phase 4)
  'tracking/tracker.cpp',
  'tracking/multi_person_tracker.cpp',
  'tracking/trajectory_stream.cpp',
  'tracking/relative_observations.cpp',
  'tracking/hierarchical_solver.cpp',

  # Build metadata
  'version.cpp',

  # Calibration (CP3)
  'calibration/scale_calibration.cpp',

  # I/O (Phase 2)
  'io/skeleton_loader.cpp',
  'io/camera_loader.cpp',
  'io/sync_loader.cpp',
  'io/observation_loader.cpp',
  'io/tracking_export.cpp',
  'io/statistics_tracker.cpp',
)

# Add DB sources (Phase 3 DB)
posetrak_sources += db_sources

# Build core library
libposetrak = library('posetrak',
  posetrak_sources + [build_version_h],
  include_directories: posetrak_inc,
  dependencies: [
    eigen_dep,
    fmt_dep,
    nlohmann_json_dep,
    yaml_cpp_dep,
    toml_dep,
    pinocchio_dep,
    sqlite3_dep,
    openmp_dep,
  ],
  install: true,
  version: meson.project_version(),
)

# Declare dependency for internal use
posetrak_dep = declare_dependency(
  link_with: libposetrak,
  include_directories: posetrak_inc,
  dependencies: [
    eigen_dep,
    fmt_dep,
    nlohmann_json_dep,
  ]
)
