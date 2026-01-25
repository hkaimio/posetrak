# Core library source files
posetrak_sources = files(
  # Core models (Phase 1)
  'core/state.cpp',
  'core/skeleton.cpp',
  'core/camera.cpp',
  'core/observation.cpp',

  # Kinematics (Phase 3)
  # 'kinematics/forward_kinematics.cpp',
  # 'kinematics/inverse_kinematics.cpp',
  # 'kinematics/triangulation.cpp',

  # Filters (Phase 4)
  # 'filters/process_model.cpp',
  # 'filters/sigma_points.cpp',
  # 'filters/ukf.cpp',
  # 'filters/outlier_rejection.cpp',

  # Tracking (Phase 4)
  # 'tracking/tracker.cpp',
  # 'tracking/tracker_diagnostics.cpp',

  # I/O (Phase 2)
  'io/skeleton_loader.cpp',
  # 'io/camera_loader.cpp',
  # 'io/observation_loader.cpp',
  # 'io/sync_loader.cpp',
  # 'io/exporters.cpp',
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
