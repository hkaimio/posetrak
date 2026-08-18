# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # C++ vs Python Tracker Comparison

    This notebook validates the C++ UKF tracker implementation against the Python reference tracker.

    ## Comparison Goals:
    1. Verify state vectors match (root pose + joint angles)
    2. Compare observation statistics (inliers, outliers, reprojection errors)
    3. Analyze trajectory differences
    4. Identify frames with largest discrepancies
    5. Validate covariance matrices
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from scipy.spatial.transform import Rotation

    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    return Path, Rotation, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data
    """)
    return


@app.cell
def _(Path, pd):
    # Paths
    cpp_dir = Path('/home/harri/projects/posetrak/tracking_tests/cpp-python-comparison/cpp_results')
    python_dir = Path('/home/harri/projects/posetrak/tracking_tests/cpp-python-comparison/python_results')

    # Load C++ results
    cpp_states = pd.read_csv(cpp_dir / 'state_vectors.csv')
    cpp_stats = pd.read_csv(cpp_dir / 'tracking_stats.csv')

    # Load Python results
    python_states = pd.read_csv(python_dir / 'state' / 'person_0' / 'frames.csv')

    _min_rows = min(len(cpp_states), len(python_states))

    if len(cpp_states) > _min_rows:
        cpp_states = cpp_states.iloc[:_min_rows].copy()
    if len(python_states) > _min_rows:
        python_states = python_states.iloc[:_min_rows].copy()

    common_frame_count = _min_rows
    print(f"C++ frames: {len(cpp_states)}")
    print(f"Python frames: {len(python_states)}")
    print(f"\nC++ state columns: {len(cpp_states.columns)}")
    print(f"Python state columns: {len(python_states.columns)}")
    return cpp_states, cpp_stats, python_states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filter out heel.02 joints from C++ (not in Python's 'main' group)
    """)
    return


@app.cell
def _(cpp_states, python_states):
    # Filter columns - remove heel.02.L and heel.02.R from C++
    heel_cols = [_col for _col in cpp_states.columns if 'heel.02' in _col]
    print(f'Removing {len(heel_cols)} heel.02 columns from C++: {heel_cols[:5]}...')
    cpp_states_filtered = cpp_states.drop(columns=heel_cols)
    print(f'\nFiltered C++ state columns: {len(cpp_states_filtered.columns)}')
    print(f'Python state columns: {len(python_states.columns)}')
    return (cpp_states_filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Root Pose Comparison
    """)
    return


@app.cell
def _(Rotation, cpp_states, np, python_states):
    # Extract root position and quaternion for both
    cpp_root_pos = cpp_states[['root_position_x', 'root_position_y', 'root_position_z']].values
    python_root_pos = python_states[['root_position_x', 'root_position_y', 'root_position_z']].values

    cpp_root_quat = cpp_states[['root_quaternion_w', 'root_quaternion_x', 'root_quaternion_y', 'root_quaternion_z']].values
    python_root_quat = python_states[['root_quaternion_w', 'root_quaternion_x', 'root_quaternion_y', 'root_quaternion_z']].values

    # Compute position differences
    pos_diff = np.linalg.norm(cpp_root_pos - python_root_pos, axis=1)

    # Compute rotation angle differences (geodesic distance on SO(3))
    rot_diff = []
    for cpp_q, py_q in zip(cpp_root_quat, python_root_quat):
        cpp_rot = Rotation.from_quat([cpp_q[1], cpp_q[2], cpp_q[3], cpp_q[0]])  # xyzw format
        py_rot = Rotation.from_quat([py_q[1], py_q[2], py_q[3], py_q[0]])
        relative = cpp_rot.inv() * py_rot
        angle = np.abs(relative.magnitude())
        rot_diff.append(np.degrees(angle))

    rot_diff = np.array(rot_diff)

    print("Root Position Differences (meters):")
    print(f"  Mean: {pos_diff.mean():.6f}")
    print(f"  Median: {np.median(pos_diff):.6f}")
    print(f"  Max: {pos_diff.max():.6f}")
    print(f"  Std: {pos_diff.std():.6f}")

    print("\nRoot Rotation Differences (degrees):")
    print(f"  Mean: {rot_diff.mean():.4f}")
    print(f"  Median: {np.median(rot_diff):.4f}")
    print(f"  Max: {rot_diff.max():.4f}")
    print(f"  Std: {rot_diff.std():.4f}")
    return pos_diff, rot_diff


@app.cell
def _(cpp_states, plt, pos_diff, rot_diff):
    # Plot root pose differences over time
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 8))
    frames = cpp_states['tracker_frame_idx'].values
    _axes[0].plot(frames, pos_diff, linewidth=2)
    _axes[0].set_ylabel('Position Difference (m)', fontsize=12)
    _axes[0].set_title('Root Position Error Over Time', fontsize=14, fontweight='bold')
    _axes[0].grid(True, alpha=0.3)
    _axes[1].plot(frames, rot_diff, linewidth=2, color='orange')
    _axes[1].set_xlabel('Frame', fontsize=12)
    _axes[1].set_ylabel('Rotation Difference (degrees)', fontsize=12)
    _axes[1].set_title('Root Rotation Error Over Time', fontsize=14, fontweight='bold')
    _axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (frames,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Joint Angle Comparison
    """)
    return


@app.cell
def _(cpp_states_filtered, np, pd, python_states):
    # Get all joint angle columns (excluding velocities, root, and metadata)
    joint_angle_cols = [_col for _col in cpp_states_filtered.columns if _col.startswith('joint_') and 'angle' in _col and ('velocity' not in _col)]
    print(f'Comparing {len(joint_angle_cols)} joint angles')
    print(f'Sample joints: {joint_angle_cols[:5]}')
    joint_diffs = {}
    for _col in joint_angle_cols:
        if _col in python_states.columns:
    # Compute per-joint differences
            _diff = np.abs(cpp_states_filtered[_col].values - python_states[_col].values)
            joint_diffs[_col] = {'mean': _diff.mean(), 'max': _diff.max(), 'std': _diff.std(), 'median': np.median(_diff)}
    joint_diff_df = pd.DataFrame(joint_diffs).T
    joint_diff_df = joint_diff_df.sort_values('mean', ascending=False)
    print('\nTop 10 joints with largest mean angle differences (radians):')
    print(joint_diff_df.head(10))
    return (joint_angle_cols,)


@app.cell
def _(cpp_states_filtered, joint_angle_cols, np, plt, python_states):
    # Plot distribution of joint angle errors
    all_joint_diffs = []
    for _col in joint_angle_cols:
        if _col in python_states.columns:
            _diff = np.abs(cpp_states_filtered[_col].values - python_states[_col].values)
            all_joint_diffs.extend(_diff)
    all_joint_diffs = np.array(all_joint_diffs)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _axes[0].hist(all_joint_diffs, bins=100, edgecolor='black', alpha=0.7)
    _axes[0].set_xlabel('Absolute Difference (radians)', fontsize=12)
    _axes[0].set_ylabel('Frequency', fontsize=12)
    _axes[0].set_title('Distribution of Joint Angle Errors', fontsize=14, fontweight='bold')
    _axes[0].set_yscale('log')
    _axes[0].grid(True, alpha=0.3)
    _axes[1].hist(np.degrees(all_joint_diffs), bins=100, edgecolor='black', alpha=0.7, color='orange')
    _axes[1].set_xlabel('Absolute Difference (degrees)', fontsize=12)
    _axes[1].set_ylabel('Frequency', fontsize=12)
    _axes[1].set_title('Distribution of Joint Angle Errors (degrees)', fontsize=14, fontweight='bold')
    _axes[1].set_yscale('log')
    _axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f'\nJoint angle error statistics (radians):')
    print(f'  Mean: {all_joint_diffs.mean():.6f}')
    print(f'  Median: {np.median(all_joint_diffs):.6f}')
    print(f'  Max: {all_joint_diffs.max():.6f}')
    print(f'  Std: {all_joint_diffs.std():.6f}')
    print(f'\nJoint angle error statistics (degrees):')
    print(f'  Mean: {np.degrees(all_joint_diffs.mean()):.4f}')
    print(f'  Median: {np.degrees(np.median(all_joint_diffs)):.4f}')
    print(f'  Max: {np.degrees(all_joint_diffs.max()):.4f}')
    return (all_joint_diffs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Velocity Comparison
    """)
    return


@app.cell
def _(cpp_states, np, python_states):
    # Compare root velocities
    cpp_root_vel = cpp_states[['root_velocity_x', 'root_velocity_y', 'root_velocity_z']].values
    python_root_vel = python_states[['root_velocity_x', 'root_velocity_y', 'root_velocity_z']].values

    root_vel_diff = np.linalg.norm(cpp_root_vel - python_root_vel, axis=1)

    print("Root Velocity Differences (m/s):")
    print(f"  Mean: {root_vel_diff.mean():.6f}")
    print(f"  Median: {np.median(root_vel_diff):.6f}")
    print(f"  Max: {root_vel_diff.max():.6f}")
    print(f"  Std: {root_vel_diff.std():.6f}")
    return (root_vel_diff,)


@app.cell
def _(cpp_states_filtered, np, python_states):
    # Compare joint velocities
    joint_vel_cols = [_col for _col in cpp_states_filtered.columns if _col.startswith('joint_') and 'velocity' in _col]
    all_joint_vel_diffs = []
    for _col in joint_vel_cols:
        if _col in python_states.columns:
            _diff = np.abs(cpp_states_filtered[_col].values - python_states[_col].values)
            all_joint_vel_diffs.extend(_diff)
    all_joint_vel_diffs = np.array(all_joint_vel_diffs)
    print('\nJoint Velocity Differences (rad/s):')
    print(f'  Mean: {all_joint_vel_diffs.mean():.6f}')
    print(f'  Median: {np.median(all_joint_vel_diffs):.6f}')
    print(f'  Max: {all_joint_vel_diffs.max():.6f}')
    print(f'  Std: {all_joint_vel_diffs.std():.6f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Frame-by-Frame Analysis
    """)
    return


@app.cell
def _(
    cpp_states_filtered,
    frames,
    joint_angle_cols,
    np,
    pd,
    pos_diff,
    python_states,
    root_vel_diff,
    rot_diff,
):
    # Compute total error per frame (position + orientation)
    frame_errors = pd.DataFrame({'frame': frames, 'pos_error': pos_diff, 'rot_error': rot_diff, 'root_vel_error': root_vel_diff})
    frame_joint_errors = []
    for i in range(len(cpp_states_filtered)):
        frame_diff = 0
        count = 0
        for _col in joint_angle_cols:
            if _col in python_states.columns:
    # Add per-frame joint angle error
                frame_diff += np.abs(cpp_states_filtered[_col].iloc[i] - python_states[_col].iloc[i])
                count += 1
        frame_joint_errors.append(frame_diff / count if count > 0 else 0)
    frame_errors['mean_joint_error'] = frame_joint_errors
    print('Frames with largest position errors:')
    print(frame_errors.nlargest(5, 'pos_error')[['frame', 'pos_error', 'rot_error', 'mean_joint_error']])
    print('\nFrames with largest rotation errors:')
    print(frame_errors.nlargest(5, 'rot_error')[['frame', 'pos_error', 'rot_error', 'mean_joint_error']])
    print('\nFrames with largest joint angle errors:')
    # Find frames with largest errors
    print(frame_errors.nlargest(5, 'mean_joint_error')[['frame', 'pos_error', 'rot_error', 'mean_joint_error']])
    return (frame_errors,)


@app.cell
def _(frame_errors, np, plt):
    # Plot frame errors
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 10))
    _axes[0, 0].plot(frame_errors['frame'], frame_errors['pos_error'], linewidth=2)
    _axes[0, 0].set_ylabel('Position Error (m)', fontsize=11)
    _axes[0, 0].set_title('Root Position Error', fontsize=12, fontweight='bold')
    _axes[0, 0].grid(True, alpha=0.3)
    _axes[0, 1].plot(frame_errors['frame'], frame_errors['rot_error'], linewidth=2, color='orange')
    _axes[0, 1].set_ylabel('Rotation Error (deg)', fontsize=11)
    _axes[0, 1].set_title('Root Rotation Error', fontsize=12, fontweight='bold')
    _axes[0, 1].grid(True, alpha=0.3)
    _axes[1, 0].plot(frame_errors['frame'], frame_errors['root_vel_error'], linewidth=2, color='green')
    _axes[1, 0].set_xlabel('Frame', fontsize=11)
    _axes[1, 0].set_ylabel('Velocity Error (m/s)', fontsize=11)
    _axes[1, 0].set_title('Root Velocity Error', fontsize=12, fontweight='bold')
    _axes[1, 0].grid(True, alpha=0.3)
    _axes[1, 1].plot(frame_errors['frame'], np.degrees(frame_errors['mean_joint_error']), linewidth=2, color='red')
    _axes[1, 1].set_xlabel('Frame', fontsize=11)
    _axes[1, 1].set_ylabel('Mean Joint Error (deg)', fontsize=11)
    _axes[1, 1].set_title('Mean Joint Angle Error', fontsize=12, fontweight='bold')
    _axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Observation Statistics Comparison
    """)
    return


@app.cell
def _(cpp_stats):
    # Analyze C++ tracking stats
    print("C++ Tracking Statistics:")
    print(f"  Total frames: {len(cpp_stats)}")
    print(f"  Mean observations per frame: {cpp_stats['num_observations'].mean():.1f}")
    print(f"  Mean inliers: {cpp_stats['num_inliers'].mean():.1f}")
    print(f"  Mean outliers: {cpp_stats['num_outliers'].mean():.1f}")
    print(f"  Inlier rate: {(cpp_stats['num_inliers'].sum() / cpp_stats['num_observations'].sum() * 100):.1f}%")
    print(f"\n  Mean reprojection error: {cpp_stats['mean_reprojection_error'].mean():.2f} px")
    print(f"  Max reprojection error: {cpp_stats['max_reprojection_error'].max():.2f} px")

    # Check frames with non-zero reprojection
    nonzero_reproj = cpp_stats[cpp_stats['mean_reprojection_error'] > 0]
    print(f"\n  Frames with non-zero reprojection: {len(nonzero_reproj)} / {len(cpp_stats)}")
    if len(nonzero_reproj) > 0:
        print(f"  Mean reprojection (non-zero frames): {nonzero_reproj['mean_reprojection_error'].mean():.2f} px")
    return


@app.cell
def _(cpp_stats, plt):
    # Plot observation statistics
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 10))
    _axes[0, 0].plot(cpp_stats['frame'], cpp_stats['num_inliers'], label='Inliers', linewidth=2)
    _axes[0, 0].plot(cpp_stats['frame'], cpp_stats['num_outliers'], label='Outliers', linewidth=2, alpha=0.7)
    _axes[0, 0].set_ylabel('Count', fontsize=11)
    _axes[0, 0].set_title('Inliers vs Outliers', fontsize=12, fontweight='bold')
    _axes[0, 0].legend()
    _axes[0, 0].grid(True, alpha=0.3)
    _axes[0, 1].plot(cpp_stats['frame'], cpp_stats['mean_reprojection_error'], linewidth=2, color='red')
    _axes[0, 1].set_ylabel('Mean Error (px)', fontsize=11)
    _axes[0, 1].set_title('Mean Reprojection Error', fontsize=12, fontweight='bold')
    _axes[0, 1].grid(True, alpha=0.3)
    _axes[1, 0].semilogy(cpp_stats['frame'], cpp_stats['covariance_condition_number'], linewidth=2, color='purple')
    _axes[1, 0].set_xlabel('Frame', fontsize=11)
    _axes[1, 0].set_ylabel('Condition Number', fontsize=11)
    _axes[1, 0].set_title('Covariance Condition Number', fontsize=12, fontweight='bold')
    _axes[1, 0].grid(True, alpha=0.3)
    _axes[1, 1].plot(cpp_stats['frame'], cpp_stats['nis_value'], linewidth=2, color='green')
    _axes[1, 1].set_xlabel('Frame', fontsize=11)
    _axes[1, 1].set_ylabel('NIS Value', fontsize=11)
    _axes[1, 1].set_title('Normalized Innovation Squared', fontsize=12, fontweight='bold')
    _axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Summary and Conclusions
    """)
    return


@app.cell
def _(
    all_joint_diffs,
    cpp_stats,
    frame_errors,
    joint_angle_cols,
    np,
    pos_diff,
    root_vel_diff,
    rot_diff,
):
    print("="*70)
    print("C++ vs Python Tracker Comparison Summary")
    print("="*70)

    print("\n1. ROOT POSE:")
    print(f"   Position error: {pos_diff.mean():.4f} ± {pos_diff.std():.4f} m (mean ± std)")
    print(f"   Rotation error: {rot_diff.mean():.3f} ± {rot_diff.std():.3f} deg")
    print(f"   Velocity error: {root_vel_diff.mean():.4f} ± {root_vel_diff.std():.4f} m/s")

    print("\n2. JOINT ANGLES:")
    print(f"   Mean error: {np.degrees(all_joint_diffs.mean()):.4f} deg")
    print(f"   Median error: {np.degrees(np.median(all_joint_diffs)):.4f} deg")
    print(f"   Max error: {np.degrees(all_joint_diffs.max()):.4f} deg")
    print(f"   Joints compared: {len(joint_angle_cols)}")

    print("\n3. TRACKING QUALITY (C++):")
    print(f"   Total frames: {len(cpp_stats)}")
    print(f"   Inlier rate: {(cpp_stats['num_inliers'].sum() / cpp_stats['num_observations'].sum() * 100):.1f}%")
    print(f"   Mean reprojection error: {cpp_stats['mean_reprojection_error'].mean():.2f} px")
    print(f"   Tracking lost frames: {cpp_stats['tracking_lost'].sum()}")

    print("\n4. WORST FRAMES:")
    worst_pos = frame_errors.nlargest(1, 'pos_error').iloc[0]
    worst_rot = frame_errors.nlargest(1, 'rot_error').iloc[0]
    worst_joint = frame_errors.nlargest(1, 'mean_joint_error').iloc[0]
    print(f"   Worst position error: Frame {worst_pos['frame']:.0f} ({worst_pos['pos_error']:.4f} m)")
    print(f"   Worst rotation error: Frame {worst_rot['frame']:.0f} ({worst_rot['rot_error']:.3f} deg)")
    print(f"   Worst joint error: Frame {worst_joint['frame']:.0f} ({np.degrees(worst_joint['mean_joint_error']):.3f} deg)")

    print("\n5. ISSUES TO INVESTIGATE:")
    if cpp_stats['num_outliers'].mean() > cpp_stats['num_inliers'].mean():
        print("   ⚠ High outlier rate - check outlier detection thresholds")
    if (cpp_stats['mean_reprojection_error'] == 0).sum() > len(cpp_stats) * 0.5:
        print("   ⚠ Many frames with zero reprojection error - check logging")
    if pos_diff.max() > 0.5:
        print(f"   ⚠ Large position discrepancies detected (max: {pos_diff.max():.3f} m)")
    if rot_diff.max() > 10:
        print(f"   ⚠ Large rotation discrepancies detected (max: {rot_diff.max():.1f} deg)")

    print("\n" + "="*70)
    return


if __name__ == "__main__":
    app.run()
