"""
Debug script to understand what's stored in the HDF5 calibration file.
"""

import h5py
import numpy as np
import sys


def inspect_calibration(h5_path: str):
    """Print all calibration data from HDF5 file."""
    print(f"Inspecting: {h5_path}")
    print("=" * 70)

    with h5py.File(h5_path, 'r') as f:
        print("\n=== ROOT ATTRIBUTES ===")
        for key in f.attrs.keys():
            print(f"{key}: {f.attrs[key]}")

        print("\n=== INTRINSICS GROUP ===")
        if 'intrinsics' in f:
            intr = f['intrinsics']
            print(f"Matrix:\n{intr['matrix'][:]}")
            print(f"\nDistortions: {intr['distortions'][:]}")
            print(f"\nAttributes:")
            for key in intr.attrs.keys():
                print(f"  {key}: {intr.attrs[key]}")

        print("\n=== CALIBRATION_UNDISTORTED GROUP ===")
        if 'calibration_undistorted' in f:
            calib_undist = f['calibration_undistorted']
            print(f"Matrix:\n{calib_undist['matrix'][:]}")
            print(f"\nDistortions: {calib_undist['distortions'][:]}")
            print(f"\nAttributes:")
            for key in calib_undist.attrs.keys():
                print(f"  {key}: {calib_undist.attrs[key]}")

        print("\n=== ANALYSIS ===")
        if 'intrinsics' in f and 'calibration_undistorted' in f:
            intr_mat = f['intrinsics']['matrix'][:]
            undist_mat = f['calibration_undistorted']['matrix'][:]
            intr_dist = f['intrinsics']['distortions'][:]
            undist_dist = f['calibration_undistorted']['distortions'][:]

            print("Matrix difference (intrinsics - undistorted):")
            print(intr_mat - undist_mat)

            print(f"\nIntrinsics distortion magnitude: {np.linalg.norm(intr_dist):.6f}")
            print(f"Undistorted distortion magnitude: {np.linalg.norm(undist_dist):.6f}")

            print("\nConclusion:")
            if np.allclose(intr_mat, undist_mat, atol=10):
                print("  - Matrices are SIMILAR (difference < 10)")
            else:
                print("  - Matrices are DIFFERENT")

            if np.linalg.norm(intr_dist) > 0.05:
                print("  - intrinsics/distortions has SIGNIFICANT distortion")
                print("    → This appears to be the ORIGINAL distortion")
            else:
                print("  - intrinsics/distortions has LOW distortion")

            if np.linalg.norm(undist_dist) > 0.05:
                print("  - calibration_undistorted/distortions has SIGNIFICANT distortion")
            else:
                print("  - calibration_undistorted/distortions has LOW distortion")
                print("    → This is from recalibration on undistorted images")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_calibration.py <path_to_calibration.hdf5>")
        sys.exit(1)

    inspect_calibration(sys.argv[1])
