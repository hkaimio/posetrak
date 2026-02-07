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

### Error state dimension

Python error state has 210 dimensions, C++ has currently 222

Active DOFs in Python:

Joint spine1 active DOF mask: [ True  True  True], total active DOF: 3
Joint spine2 active DOF mask: [ True  True  True], total active DOF: 3
Joint neck1 active DOF mask: [ True  True  True], total active DOF: 3
Joint neck2 active DOF mask: [ True  True  True], total active DOF: 3
Joint head active DOF mask: [ True  True  True], total active DOF: 3
Joint shoulder.L active DOF mask: [ True  True  True], total active DOF: 3
Joint upper_arm.L active DOF mask: [ True  True  True], total active DOF: 3
Joint forearm.L active DOF mask: [ True  True False], total active DOF: 2
Joint hand.L active DOF mask: [ True False  True], total active DOF: 2
Joint palm.01.L active DOF mask: [ True], total active DOF: 1
Joint f_index.01.L active DOF mask: [ True False  True], total active DOF: 2
Joint f_index.02.L active DOF mask: [ True], total active DOF: 1
Joint f_index.03.L active DOF mask: [ True], total active DOF: 1
Joint thumb.01.L active DOF mask: [False  True  True], total active DOF: 2
Joint thumb.02.L active DOF mask: [ True], total active DOF: 1
Joint thumb.03.L active DOF mask: [ True], total active DOF: 1
Joint palm.02.L active DOF mask: [ True], total active DOF: 1
Joint f_middle.01.L active DOF mask: [ True False  True], total active DOF: 2
Joint f_middle.02.L active DOF mask: [ True], total active DOF: 1
Joint f_middle.03.L active DOF mask: [ True], total active DOF: 1
Joint palm.03.L active DOF mask: [ True], total active DOF: 1
Joint f_ring.01.L active DOF mask: [ True False  True], total active DOF: 2
Joint f_ring.02.L active DOF mask: [ True], total active DOF: 1
Joint f_ring.03.L active DOF mask: [ True], total active DOF: 1
Joint palm.04.L active DOF mask: [ True], total active DOF: 1
Joint f_pinky.01.L active DOF mask: [ True False  True], total active DOF: 2
Joint f_pinky.02.L active DOF mask: [ True], total active DOF: 1
Joint f_pinky.03.L active DOF mask: [ True], total active DOF: 1
Joint shoulder.R active DOF mask: [ True  True  True], total active DOF: 3
Joint upper_arm.R active DOF mask: [ True  True  True], total active DOF: 3
Joint forearm.R active DOF mask: [ True  True False], total active DOF: 2
Joint hand.R active DOF mask: [ True False  True], total active DOF: 2
Joint palm.01.R active DOF mask: [ True], total active DOF: 1
Joint f_index.01.R active DOF mask: [ True False  True], total active DOF: 2
Joint f_index.02.R active DOF mask: [ True], total active DOF: 1
Joint f_index.03.R active DOF mask: [ True], total active DOF: 1
Joint thumb.01.R active DOF mask: [False  True  True], total active DOF: 2
Joint thumb.02.R active DOF mask: [ True], total active DOF: 1
Joint thumb.03.R active DOF mask: [ True], total active DOF: 1
Joint palm.02.R active DOF mask: [ True], total active DOF: 1
Joint f_middle.01.R active DOF mask: [ True False  True], total active DOF: 2
Joint f_middle.02.R active DOF mask: [ True], total active DOF: 1
Joint f_middle.03.R active DOF mask: [ True], total active DOF: 1
Joint palm.03.R active DOF mask: [ True], total active DOF: 1
Joint f_ring.01.R active DOF mask: [ True False  True], total active DOF: 2
Joint f_ring.02.R active DOF mask: [ True], total active DOF: 1
Joint f_ring.03.R active DOF mask: [ True], total active DOF: 1
Joint palm.04.R active DOF mask: [ True], total active DOF: 1
Joint f_pinky.01.R active DOF mask: [ True False  True], total active DOF: 2
Joint f_pinky.02.R active DOF mask: [ True], total active DOF: 1
Joint f_pinky.03.R active DOF mask: [ True], total active DOF: 1
Joint thigh.L active DOF mask: [ True  True  True], total active DOF: 3
Joint shin.L active DOF mask: [ True], total active DOF: 1
Joint foot.L active DOF mask: [ True  True  True], total active DOF: 3
Joint toe.L active DOF mask: [ True], total active DOF: 1
Joint thigh.R active DOF mask: [ True  True  True], total active DOF: 3
Joint shin.R active DOF mask: [ True], total active DOF: 1
Joint foot.R active DOF mask: [ True  True  True], total active DOF: 3
Joint toe.R active DOF mask: [ True], total active DOF: 1
Joint hips active DOF mask: [ True  True  True  True  True  True], total active DOF: 6
Error dim 210

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
- TIme stamps differ, C++ timestamps weird (first frame 0.0125)
-
