# Tests
# Phase 0: Set up testing framework

test_sources = files(
  # Phase 0: Basic test to verify framework
  'test_basic.cpp',

  # Phase 1: Core models
  'test_state.cpp',
  'test_skeleton.cpp',
  'test_camera.cpp',
  'test_observation.cpp',
  'test_config.cpp',

  # Phase 2: I/O
  'test_skeleton_loader.cpp',
  'test_camera_loader.cpp',
  'test_sync_loader.cpp',
  'test_observation_loader.cpp',

  # Phase 3: Kinematics
  'test_pinocchio_integration.cpp',
  'test_triangulation.cpp',
  'test_inverse_kinematics.cpp',

  # Phase 4: UKF
  'test_process_model.cpp',
  'test_sigma_points.cpp',
  'test_ukf.cpp',
  'test_ukf_update.cpp',
  'test_skeleton_layout.cpp',
  'test_skeleton_state.cpp',

  # Phase 4.6: Integration tests
  'test_tracker_integration.cpp',

  # Phase 7: Export
  'test_tracking_export.cpp',
  'test_statistics_tracker.cpp',

  # Scale calibration
  'calibration/test_scale_calibration.cpp',
  'calibration/test_scale_regression.cpp',

  # DB read path
  'test_session_reader.cpp',

  # DB blob codec
  'test_blob_codec.cpp',
)

test_exe = executable('test_posetrak',
  test_sources,
  include_directories: posetrak_inc,
  dependencies: [posetrak_dep, test_dep, pinocchio_dep, yaml_cpp_dep, sqlite3_dep],
)

test('posetrak_tests', test_exe,
  workdir: meson.project_source_root(),
)
