+++
name = "Camera Control"
status = "proposal"
description = """
A system to start/stop and confirm a heterogeneous set of capture cameras (GoPro, Insta360, \
Android phones, eventually iPhone) from a single controller PC over a wireless capture LAN, \
with per-camera start-time accuracy and automatic video offload — replacing today's manual, \
error-prone per-camera button presses.
"""
categories = ["capture-hardware"]
target_release = "TBD"
last_updated = 2026-08-06
+++

# Camera Control — Implementation Status

See:
- [camera-control-brief.md](camera-control-brief.md) — problem statement and requirements
- [camera-control-analysis.md](camera-control-analysis.md) — per-camera control/timing/offload
  options analysis and communication-architecture evaluation
- [camera-control-design.md](camera-control-design.md) — resulting system design proposal

## Current state

Design-only. No hardware built, no firmware or controller-side software written. The design
proposes a star topology on a dedicated capture WLAN: per-camera ESP32 "CamNode" BLE bridges
for GoPro/Insta360 (one firmware image, selectable personality), Android phones driven directly
over wireless ADB (no bridge node needed), an MQTT broker on the controller PC for status/
command delivery, and a sync-beacon node for LED fine-sync. Confidence levels are explicit per
camera type (Confirmed / Commanded / unconfirmed) rather than assuming uniform confirmation
across all camera types — see the design doc's "confirm where possible, verify-on-collect
always" principle.

## Known issues / open questions

- Insta360 Ace Pro 2 control is based on a reverse-engineered protocol (GPS Action Remote
  emulation) and is explicitly flagged as untested against the actual Ace Pro 2 hardware.
- No prototype has validated BLE range/reliability at ~0.5 m (CamNode-to-camera) or WiFi
  reliability at 10 m+ (CamNode-to-controller) in a real capture environment.
- iPhone control path is sketched (BLE HID volume-key remote) but explicitly "not designed
  here" — lowest-confidence camera type in the analysis.
- No estimate yet of build cost/time for the CamNode hardware or firmware.
