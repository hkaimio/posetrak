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
  # 'filters/subset_ukf.cpp',  # SubsetUKF scheduled for deletion (Phase 3c)

  # Tracking (Phase 4)
  'tracking/tracker.cpp',

  # I/O (Phase 2)
  'io/skeleton_loader.cpp',
  'io/camera_loader.cpp',
  'io/sync_loader.cpp',
  'io/observation_loader.cpp',
  'io/tracking_export.cpp',
  'io/statistics_tracker.cpp',
)

# Build core library
libposetrak = library('posetrak',
  posetrak_sources,
  include_directories: posetrak_inc,
  dependencies: [
    eigen_dep,
    fmt_dep,
    nlohmann_json_dep,
    yaml_cpp_dep,
    toml_dep,
    pinocchio_dep,
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
