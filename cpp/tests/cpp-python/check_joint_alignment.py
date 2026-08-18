#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Check if C++ skeleton joint order matches Python CSV joint order."""

import pandas as pd
import yaml
import sys

# Load CSV to see Python joint order
csv = pd.read_csv('/home/harri/projects/posetrak/tests/cpp-python/initial_state.csv')

# Extract joint names and DoFs from CSV
csv_joints = []
for col in csv.columns:
    if col.startswith('joint_') and '_angle_' in col:
        parts = col.replace('joint_', '').split('_angle_')
        joint_name = parts[0]
        angle_idx = int(parts[1])

        # Add to list if not already there
        if not csv_joints or csv_joints[-1][0] != joint_name:
            csv_joints.append([joint_name, 1])  # Start with 1 DoF
        else:
            csv_joints[-1][1] = angle_idx + 1  # Update DoF count

print("=" * 70)
print("CSV JOINT STRUCTURE")
print("=" * 70)
print(f"Total joints: {len(csv_joints)}")
print()

# Show joints with DoF counts
for i, (joint_name, dof_count) in enumerate(csv_joints):
    angle_cols = [c for c in csv.columns if c.startswith(f'joint_{joint_name}_angle_')]
    values = [csv.iloc[0][c] for c in angle_cols]
    has_nonzero = any(abs(v) > 1e-10 for v in values)
    mark = " ✱" if has_nonzero else ""
    joint_type = "SPHERICAL" if dof_count == 3 else "REVOLUTE" if dof_count == 1 else f"{dof_count}-DOF"

    # Highlight leg joints
    if any(x in joint_name for x in ['thigh', 'shin', 'foot', 'toe', 'heel']):
        print(f"[{i:3d}] {joint_name:25s} {joint_type:10s}{mark}")

print()
print("=" * 70)
print("C++ SKELETON STRUCTURE")
print("=" * 70)

# Load skeleton YAML to see C++ joint order
skeleton_path = "/mnt/d/mocap/2026-01-11-kotegaesh-joint-space-test/Harri_skeleton-shouldery-rot.yaml"
try:
    with open(skeleton_path, 'r') as f:
        skeleton = yaml.safe_load(f)

    joints = skeleton.get('joints', [])
    print(f"Total joints: {len(joints)}")
    print()

    # Show leg joints
    for i, joint in enumerate(joints):
        joint_name = joint['name']
        joint_type = joint.get('type', 'revolute').lower()

        if any(x in joint_name for x in ['thigh', 'shin', 'foot', 'toe', 'heel', 'hips']):
            dof = 3 if joint_type in ['spherical', 'ball'] else 1 if joint_type == 'revolute' else 0
            print(f"[{i:3d}] {joint_name:25s} {joint_type:10s} ({dof} DoF)")

    print()
    print("=" * 70)
    print("COMPARING CSV vs C++ SKELETON")
    print("=" * 70)

    # Build C++ joint list
    cpp_joints = [(j['name'], 3 if j.get('type', 'revolute').lower() in ['spherical', 'ball'] else 1) for j in joints]

    # Compare
    if len(csv_joints) != len(cpp_joints):
        print(f"⚠️  MISMATCH: CSV has {len(csv_joints)} joints, C++ has {len(cpp_joints)} joints")
        print()

        # Find which are different
        csv_names = [name for name, _ in csv_joints]
        cpp_names = [name for name, _ in cpp_joints]

        csv_only = set(csv_names) - set(cpp_names)
        cpp_only = set(cpp_names) - set(csv_names)

        if csv_only:
            print(f"Joints in CSV but NOT in C++ skeleton ({len(csv_only)}):")
            for name in sorted(csv_only):
                idx = csv_names.index(name)
                print(f"  [{idx:3d}] {name}")
            print()

        if cpp_only:
            print(f"Joints in C++ skeleton but NOT in CSV ({len(cpp_only)}):")
            for name in sorted(cpp_only):
                idx = cpp_names.index(name)
                print(f"  [{idx:3d}] {name}")
            print()
    else:
        print("✓ Same number of joints")
        print()

    # Check order and DoF mismatch
    mismatches = []
    for i in range(min(len(csv_joints), len(cpp_joints))):
        csv_name, csv_dof = csv_joints[i]
        cpp_name, cpp_dof = cpp_joints[i]

        if csv_name != cpp_name or csv_dof != cpp_dof:
            mismatches.append((i, csv_name, csv_dof, cpp_name, cpp_dof))

    if mismatches:
        print(f"⚠️  FOUND {len(mismatches)} MISMATCHES:")
        print()
        for idx, csv_name, csv_dof, cpp_name, cpp_dof in mismatches[:20]:  # Show first 20
            if csv_name != cpp_name:
                print(f"  [{idx:3d}] NAME MISMATCH: CSV='{csv_name}' vs C++='{cpp_name}'")
            if csv_dof != cpp_dof:
                print(f"  [{idx:3d}] DOF MISMATCH: '{csv_name}' CSV={csv_dof} vs C++={cpp_dof}")
    else:
        print("✓ All joints match in order and DoF count!")

except Exception as e:
    print(f"Error loading skeleton: {e}")
    sys.exit(1)
