```toml
name = "Camera Control"
status = "proposal"
description = """
A system to start/stop and confirm a heterogeneous set of capture cameras (GoPro Hero13 \
first, then Hero11, Insta360, Android phones, eventually iPhone) from a single controller PC \
over a wireless capture LAN, with per-camera start-time accuracy, remote control of the \
marker-lighting LEDs, automatic video offload, and a mobile web UI for field operation.
"""
categories = ["capture-hardware"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Camera Control — Implementation Status

See:
- [camera-control-brief.md](camera-control-brief.md) — problem statement and requirements
- [camera-control-analysis.md](camera-control-analysis.md) — per-camera control/timing/offload
  options analysis and communication-architecture evaluation
- [camera-control-design.md](camera-control-design.md) — resulting system design proposal
- [node-protocol.md](node-protocol.md) — wire contract with the ESP32 node firmware
  (developed in a separate repository)

## Current state

Design-only. No controller software written. The sync beacon and node firmware are being
developed separately in Rust.

The design is a capture daemon (`posetrak-capture`) on the controller PC that serves an HTTP
API and event stream to both the Qt app and a mobile-first web UI. Cameras are controlled
directly where they can join the capture WLAN (Hero13 via COHN and the Open GoPro HTTP API,
Android via wireless ADB). ESP32 nodes are peripherals with roles (BLE bridge for Hero11 and
Insta360, camera-mounted LED ring, sync beacon); their commands are scheduled against a
synchronized node clock so that sub-millisecond LED synchronization can be added later.
Confidence levels are explicit per camera type (Confirmed / Commanded / Manual).

Camera priority: Hero13, then Hero11 and Insta360 Ace Pro 2, then Android, then iPhone.

## Known issues / open questions

- Hero13 COHN behaviour is unverified on hardware: link stability while recording at mocap
  frame rates, battery and heat with WiFi active, wake from off at capture distances,
  aggregate download throughput (spikes S7–S9 in the design doc). Phase 1 is gated on these.
- Insta360 Ace Pro 2 control is based on a reverse-engineered protocol (GPS Action Remote
  emulation) and is untested against the actual hardware (S1/S2).
- Rust BLE on ESP32 needs to cover both central (GoPro) and peripheral (Insta360 remote)
  roles (S11).
- Method for sub-ms node synchronization is open (S10); the protocol only reserves the
  slots for it.
- iPhone control path is sketched (BLE HID volume-key remote) but not designed.
- The LED ring hardware and its power and driver requirements are not yet specified.
- No estimate yet of build cost/time for the nodes or the controller.
