#!/bin/bash
# Debug script to check error state dimensions

echo "=== Running C++ tracker to check dimensions ==="
cd /home/harri/projects/posetrak
./optbuild/cli/posetrak tests/cpp-python/cpp_test_config.toml 2>&1 | tee debug_output.txt | grep -E "(DEBUG|Skeleton|active_dof|error_dim|Covariance|SIGMA)"

echo ""
echo "=== Checking covariance dimensions ==="
echo "Python covariance:"
wc -l tracking_tests/cpp-python-comparison/python_results/debug/frame_0000/prior_covariance.csv | awk '{print "Lines: " $1}'
head -1 tracking_tests/cpp-python-comparison/python_results/debug/frame_0000/prior_covariance.csv | tr ',' '\n' | wc -l | awk '{print "Columns: " $1}'

echo ""
echo "C++ covariance:"
if [ -f tracking_tests/cpp-python-comparison/cpp_results/debug/frame_0000/prior_covariance.csv ]; then
    wc -l tracking_tests/cpp-python-comparison/cpp_results/debug/frame_0000/prior_covariance.csv | awk '{print "Lines: " $1}'    head -1 tracking_tests/cpp-python-comparison/cpp_results/debug/frame_0000/prior_covariance.csv | tr ',' '\n' | wc -l | awk '{print "Columns: " $1}'
else
    echo "File not found - test may have crashed"
fi

echo ""
echo "=== Python covariance diagonal (first 10 values) ==="
awk -F',' 'NR>1 && NR<=11 {print "Row", NR-1, "Col", NR-1, ":", $NR}' tracking_tests/cpp-python-comparison/python_results/debug/frame_0000/prior_covariance.csv

echo ""
echo "=== Full debug output saved to debug_output.txt ==="
