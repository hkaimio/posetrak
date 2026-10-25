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
last_updated = 2026-10-25
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

Implemented on the controller side (`python/pipeline/capture/`, run as `posetrak-capture`):

- **Daemon and API.** Controller state machines with confidence levels, a control lease, an
  event stream, and the HTTP/WebSocket API with a mobile web UI (device wall, arm, start and
  stop with confirmation, checklist, alerts). `run --demo N` serves simulated cameras.
- **Hero13 over COHN.** Driver with link monitoring and fallback between WLANs, mode and
  settings verification at arm time, and start and stop confirmed from the camera's encoding
  flag. Provisioning over Bluetooth stores the credentials in the user configuration
  directory. Exercised against a fake camera and provisioned on one real camera; a capture
  with several real cameras has not been run.
- **Registry integration.** `provision <label>` links a registry camera to the physical
  camera (`control_endpoints`), and the daemon reads the control method and device from the
  registry. Capture modes are named registry modes with driver-specific `capture_settings`;
  configured setups choose a mode per camera or camera model, the operator can override it
  per capture, and the mode and settings each camera was armed with are recorded in
  `capture_events`. Cameras whose settings cannot be set remotely get a checklist item.
  Design: section 3.6 of the design document.

Credentials are kept in a registry table that sessions never get (`control_credentials`).

`posetrak-capture offload <label>...` copies the recordings of Hero13s over the network, with
resume, size checking, a SHA-256 and the camera's metadata in a JSON file beside each copy.
Matching the copied files to captures and registering them in the session (the rest of the
collection step in the design) is not built.

`posetrak-capture detect-mode <file>...` tells which registry camera and capture mode a video
file was recorded in, from what the file states about itself (needs `exiftool`). It is
checked on Hero13 files only, and is not yet wired into the setup wizard or the offload.

Not built yet: matching and registering offloaded files, the other camera drivers (Hero11, Insta360, Android, iPhone), node peripherals (LED rings, sync
beacon) and the Qt integration. The beacon and node firmware are developed separately in Rust.

Camera priority: Hero13, then Hero11 and Insta360 Ace Pro 2, then Android, then iPhone.

## Known issues / open questions

- Hero13 COHN behaviour is only partly verified on hardware: the link was stable over a 38
  minute poll, but a camera with WiFi active reached its thermal limit within about 25 minutes
  of recording. Wake from off at capture distances, behaviour with several cameras and WLANs,
  and aggregate download throughput are untested (spikes S7–S9 in the design doc).
- Whether a Hero13 can hold credentials for several WLANs and pick the one it finds is untested.
- Setups and modes are read when the daemon starts; editing modes in the registry needs a restart.
- Insta360 Ace Pro 2 control is based on a reverse-engineered protocol (GPS Action Remote
  emulation) and is untested against the actual hardware (S1/S2).
- Rust BLE on ESP32 needs to cover both central (GoPro) and peripheral (Insta360 remote)
  roles (S11).
- Method for sub-ms node synchronization is open (S10); the protocol only reserves the
  slots for it.
- iPhone control path is sketched (BLE HID volume-key remote) but not designed.
- The LED ring hardware and its power and driver requirements are not yet specified.
- No estimate yet of build cost/time for the nodes or the controller.
