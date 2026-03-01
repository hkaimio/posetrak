#!/usr/bin/env python3
"""Quick test of Rerun API."""
import rerun as rr

print(f"Rerun version: {rr.__version__}")
print("\nTime-related attributes:")
for attr in dir(rr):
    if 'time' in attr.lower() or 'Time' in attr:
        print(f"  {attr}")

print("\nTrying to find set_time signature:")
import inspect
try:
    sig = inspect.signature(rr.set_time)
    print(f"  set_time signature: {sig}")
except Exception as e:
    print(f"  Error: {e}")

print("\nLooking for Time classes:")
for attr in dir(rr):
    obj = getattr(rr, attr)
    if isinstance(obj, type) and 'Time' in attr:
        print(f"  {attr}: {obj}")
