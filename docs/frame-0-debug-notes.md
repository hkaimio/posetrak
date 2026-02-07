## Goal

C++ tracker run with command

optbuild/cli/posetrak tracking_tests/cpp-python-comparison/cpp_test_config.toml

produces at every step exactly same interim results as Python tracker when run with command
specified in C:\Users\HarriKaimio\projects\posing-notebooks\run_tracker.ps1

Debug output from Python tracker available in /home/harri/projects/posetrak/tracking_tests/cpp-python-comparison/python_results


### Initialization

- C++ tracker initialized with same data as Python tracker from frome 0 in

/home/harri/projects/posetrak/tracking_tests/cpp-python-comparison/python_results/state/person_0/frames.csv

Status: verified
- C++ data in /home/harri/projects/posetrak/tracking_tests/cpp-python-comparison/cpp_results/state_vectors.csv, frame 0

### Predict step

- This should be noop as all velocities are 0

- Verified

### Update

Use same set of observations

- Python and C++ apparently start tracking from different frames. C++ loader uses frame 270 fro cam0 while Python starts from 275.
  - Fixed by redesigned sync & config file formats.
  - Should be OK in 739c9ea Add FPS support to sync metadata JSON format
- Camera names and camera IDs do not match. Apparently the camera IDs are given "randomly" in tracker cli!!!
  - Fixed in f5149c9 Refactor: Use std::map for deterministic camera ID assignment
- Camera 4 positions differ slightly. Python position uses coordinates from pose file, C++ corrects with distortion - camera 4 is the only one vit distortion correction

Marker positioopn  predictions
- Python predictions: tracking_tests/cpp-python-comparison/python_results/forward_kinematics/markers_3d/table.csv
- C++ predictions: tracking_tests/cpp-python-comparison/cpp_results/tracking_results.csv
- TIme stamps differ, C++ timestomans weird (first frame 0.0125)
-
