#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Test rerun API to verify correct usage."""
import rerun as rr
import numpy as np

print(f"Rerun version: {rr.__version__}")

# Initialize
rr.init("test_app")
rr.save("test_output.rrd")

# Test static logging
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Test time setting
rr.set_time("frame", 1)
rr.set_time("timestamp", 0.0)

# Test Points3D
rr.log(
    "test/points",
    rr.Points3D(
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        colors=[255, 0, 0],
        radii=0.1,
    ),
)

print("✅ Test complete! Check test_output.rrd")
