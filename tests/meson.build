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

  # Phase 2: I/O
  'test_skeleton_loader.cpp',
  'test_camera_loader.cpp',
  'test_sync_loader.cpp',
  # 'test_observation_loader.cpp',

  # Phase 3: Kinematics
  # 'test_forward_kinematics.cpp',
  # 'test_triangulation.cpp',
  # 'test_inverse_kinematics.cpp',

  # Phase 4: UKF
  # 'test_process_model.cpp',
  # 'test_sigma_points.cpp',
  # 'test_ukf.cpp',
)

test_exe = executable('test_posetrak',
  test_sources,
  include_directories: posetrak_inc,
  dependencies: [posetrak_dep, test_dep],
)

test('posetrak_tests', test_exe,
  workdir: meson.project_source_root(),
)
