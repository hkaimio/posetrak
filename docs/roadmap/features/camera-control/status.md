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
last_updated = 2026-11-02
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

After a capture stops the daemon can copy the capture's recordings by itself (`auto_offload`,
or the switch in the web UI). The recordings that appeared on a camera between arming and
stopping are the capture's; they are copied to `<offload dir>/capture-<number>/<camera>/`
and registered in the capture with their capture mode, a video in another mode than the
camera was armed with raises an alert, and the cameras cannot be armed again until copying
has finished or been cancelled. Matching files that were recorded without the daemon to
captures (by time) is not built. Chaptered recordings (a long recording split by the
camera) become one registered video per file.

`posetrak-capture detect-mode <file>...` tells which registry camera and capture mode a video
file was recorded in, from what the file states about itself (needs `exiftool`). It is
checked on Hero13 files only, and is not yet wired into the setup wizard or the offload.

Measured on one real Hero13: a 30 s 4K 120 recording (284 MB, about 76 Mbit/s) copied in a
little over 30 s over the WLAN, i.e. about real time. Copying a 10 minute take therefore takes
about 10 minutes per camera. With the default of two cameras copying at once, whether the
rate holds per camera is not yet measured.

Not built yet: registering files that were recorded without the daemon, the other camera
drivers (Hero11, Insta360, Android, iPhone), node peripherals (LED rings, sync beacon) and the
Qt integration. The beacon and node firmware are developed separately in Rust.

Camera priority: Hero13, then Hero11 and Insta360 Ace Pro 2, then Android, then iPhone.

## Next steps

1. **Multi-camera run on real cameras** with automatic copying on: time from stop to every
   camera done, per-camera rate with two copies at once, cost of reading the media list when
   arming with a full card (it grows with the number of files on the card, not with how full
   it is), behaviour of chaptered recordings.
2. **Fixes from that run**: copy concurrency from the measured rates; join the chapters of a
   long recording or link them to each other; an arm-time warning when a card has less free
   space than the planned take needs.
3. **Setup wizard integration**: make the wizard's video probe call exiftool directly, pre-fill
   the mode of videos added by hand, and a `register` command for files recorded without the
   daemon (needs a tolerance for clock differences and a camera clock check).
4. **Usability**: reload modes and setups without a restart; try the web UI on a phone; keep
   registry credentials in the OS keystore or encrypted.
5. **Minimal desktop integration** (Qt): capture state and starting a capture.
6. **Field setup guide and acceptance run** with 2 or 3 cameras, including more than one WLAN.
7. **More camera drivers**, in the priority order above, once the Hero13 acceptance run passes.

More sample footage in other lenses and resolutions would complete the format detection.

## Known issues / open questions

- Hero13 COHN behaviour is only partly verified on hardware: the link was stable over a 38
  minute poll, but a camera with WiFi active reached its thermal limit within about 25 minutes
  of recording. Wake from off at capture distances, behaviour with several cameras and WLANs,
  and aggregate download throughput are untested (spikes S7–S9 in the design doc).
- Whether a Hero13 can hold credentials for several WLANs and pick the one it finds is untested.
- Setups and modes are read when the daemon starts; editing modes in the registry needs a restart.
- Chaptered recordings (the camera splits files at about 4 GB, roughly 7 minutes at 4K 120)
  register as one video per file.
- Insta360 Ace Pro 2 control is based on a reverse-engineered protocol (GPS Action Remote
  emulation) and is untested against the actual hardware (S1/S2).
- Rust BLE on ESP32 needs to cover both central (GoPro) and peripheral (Insta360 remote)
  roles (S11).
- Method for sub-ms node synchronization is open (S10); the protocol only reserves the
  slots for it.
- iPhone control path is sketched (BLE HID volume-key remote) but not designed.
- The LED ring hardware and its power and driver requirements are not yet specified.
- No estimate yet of build cost/time for the nodes or the controller.
