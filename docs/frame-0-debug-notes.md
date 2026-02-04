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

- Python and C++ apparently start tracking from different frames. C++ loader uses frame 270 fro cam0 while Python starts from 275.
