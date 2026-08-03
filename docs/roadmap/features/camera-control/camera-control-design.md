# Camera Control — System Design Proposal

Solution proposal for the requirements in [camera-control-brief.md](camera-control-brief.md).
Option evaluation and the evidence behind the choices made here is in
[camera-control-analysis.md](camera-control-analysis.md).

## Goals recap

From the brief, the system must:

1. Start and stop **4–10+ heterogeneous cameras** (GoPro Hero11 Mini, Insta360 Ace Pro 2,
   Android phones running their stock camera apps; iPhone path desirable) from a single PC
2. **Confirm** that all cameras actually started/stopped
3. Record per-camera **start times to ~1 s accuracy**
4. Work **wirelessly over 10 m+** with no venue WLAN (own access point allowed; cheap
   per-camera helper hardware allowed)
5. Prevent the classic failure modes: wrong mode, missed start, camera nudged at
   start (invalidating extrinsics)
6. Preferably **collect the recorded files** to the PC automatically, named by camera and
   capture time

Remote triggering alone eliminates failure mode "camera moves when the start button is
pressed" — nobody touches a rigged camera after extrinsics calibration.

---

## 1. Architecture overview

Star topology on a dedicated capture WLAN:

```
                          ┌──────────────────────────────┐
                          │  Controller PC               │
                          │  posetrak app + CaptureCtl   │
                          │  MQTT broker · ADB · NTP ref │
                          └──────────────┬───────────────┘
                                         │ WiFi
                          ┌──────────────┴───────────────┐
                          │  Travel router (capture WLAN)│
                          └──┬────────┬────────┬─────┬───┘
                     WiFi/MQTT   WiFi/MQTT  WiFi/ADB │ WiFi/MQTT
                  ┌──────┴───┐ ┌────┴─────┐ ┌───┴────┐ ┌──┴─────────┐
                  │ CamNode  │ │ CamNode  │ │ Pixel /│ │ Sync beacon│
                  │ (ESP32)  │ │ (ESP32)  │ │ OnePlus│ │ (ESP32+LED)│
                  └────┬─────┘ └────┬─────┘ │ stock  │ └────────────┘
                   BLE │ ~0.5 m BLE │ ~0.5 m│ camera │
                  ┌────┴─────┐ ┌────┴─────┐ │ app    │
                  │ GoPro 11 │ │ Insta360 │ └────────┘
                  │ Mini     │ │ AcePro 2 │
                  └──────────┘ └──────────┘
```

Key decisions (rationale in the analysis doc, §6):

- **Own WLAN via travel router.** WiFi covers the 10 m+ distances reliably; BLE does not.
  The router also isolates the capture network from anything else and gives phones a
  network to join.
- **Per-camera ESP32 bridge nodes ("CamNodes")** for cameras that only speak BLE. Each
  node is velcroed/clamped next to its camera, so the BLE link is ~0.5 m and rock solid;
  the node bridges to the controller over WiFi/MQTT. One firmware image with selectable
  *personalities*: `gopro` (Open GoPro BLE client), `insta360` (GPS-remote emulation),
  `ble-hid` (selfie-remote keyboard, for iPhone and as Android fallback).
- **Android phones need no node** — they join the WLAN and are driven over ADB wireless
  debugging, which also provides state confirmation and file pull.
- **MQTT broker on the PC.** Retained per-device status topics give the UI an
  always-current device wall; MQTT last-will marks a node offline the moment it drops off
  the network; QoS 1 covers command delivery over flaky radio.
- **Sync beacon node** — same ESP32 hardware driving a high-brightness LED, commanded by
  the controller to blink logged patterns. Automates the existing LED fine-sync step and
  anchors it to absolute time (§5).

### Design principle: confirm where possible, verify-on-collect always

Not every camera can confirm "recording started" (Insta360 cannot). Rather than pretend,
the system is explicit about three confidence levels per camera, shown in the UI:

| Level | Meaning | Cameras |
|---|---|---|
| **Confirmed** | positive status readback (encoding flag / camera pipeline active) | GoPro, Android |
| **Commanded** | command delivered to the camera, no readback | Insta360, iPhone |
| **Manual** | operator started it by hand | anything, as fallback |

Independently, the **collection step verifies every file** after the fact — fps,
resolution, duration, and timestamps are probed and compared against the expected camera
mode and capture window. A wrong-mode or short-file problem is caught the same day with
the rig still standing, not weeks later at pose-extraction time.

---

## 2. Hardware

### Bill of materials (10-camera setup)

| Item | Qty | Unit ≈ | Notes |
|---|---|---|---|
| Travel router (GL.iNet GL-A1300 / GL-MT3000 class) | 1 | €50 | dual-band, battery/USB-powerable; 2.4 GHz for range |
| CamNode: ESP32-S3 or C3 dev board (e.g. Seeed XIAO) | up to 1/BLE camera | €7 | WiFi + BLE 5, tiny |
| CamNode battery: 500 mAh LiPo + charger circuit | per node | €4 | days of standby, USB-C charge |
| CamNode case + mounting (printed case, velcro/clamp) | per node | €2 | attach to camera clamp/tripod |
| Sync beacon: ESP32 + 1–3 W LED + driver + battery | 1–2 | €15 | must be visible to all cameras; 2 units for opposing camera arcs |
| Powered USB hub, ≥10 ports ("collection dock") | 1 | €30 | offload + charging after teardown |
| USB cables (USB-C) | ~10 | €3 | |

Total for a 10-camera rig with, say, 6 BLE cameras: **≈ €180**. Phones and the PC need
no extra hardware.

### CamNode hardware notes

- A status RGB LED on the node mirrors its state (offline / connected / armed /
  recording / error) so a glance across the room shows rig health even without the UI.
- One push-button for pairing/identify.
- Nodes are generic and interchangeable: personality + target camera are assigned from
  the controller at registration time and stored in NVS, so a spare node can replace a
  failed one in the field.
- 1 node : 1 camera by default. The GoPro personality could serve 2–3 co-located GoPros
  from one node (NimBLE multi-connection), but the Insta360 personality is a BLE
  *peripheral* (it advertises as a remote) while the GoPro one is a *central* — do not
  plan on mixing personalities in one node.

---

## 3. Software components

### 3.1 Capture control service — `python/pipeline/capture/`

New Python package, used by the posetrak GUI (and exposable as a CLI for headless use):

```
python/pipeline/capture/
├── service.py        # CaptureController: orchestration + state machines
├── devices.py        # Device abstraction + per-type drivers
├── mqtt_link.py      # broker mgmt (embedded mosquitto or amqtt), topic schema
├── adb_driver.py     # Android control: pairing, connect, intents, keyevents, dumpsys
├── collect.py        # collection service + verify-on-collect
├── beacon.py         # sync beacon control + blink-schedule logging
└── protocol.py       # MQTT message dataclasses, versioned
```

**Device model.** Each controllable camera is a `CaptureDevice` bound to a registry
`camera_instances` row, with a driver:

- `GoProNodeDriver` — talks MQTT to a CamNode running the `gopro` personality
- `Insta360NodeDriver` — MQTT to `insta360` personality
- `BleHidNodeDriver` — MQTT to `ble-hid` personality (iPhone / Android fallback)
- `AdbDriver` — direct ADB over the WLAN
- `ManualDriver` — no automation; the UI shows operator instructions and asks for
  a manual tick (safety net so a broken node never blocks a shoot)

**Per-device state machine:**

```
OFFLINE → DISCOVERED → READY → ARMED → RECORDING → STOPPING → STOPPED → COLLECTED
                         │        │
                         └── fault states carry a reason (low battery, SD full,
                             mode mismatch, link lost) and are always recoverable
                             by falling back to ManualDriver
```

- `READY`: link up, battery/storage OK (where readable)
- `ARMED`: pre-flight passed — GoPros have had their preset pushed *and read back*;
  checklist items acknowledged for cameras without remote mode control
- `RECORDING`: per the device's confidence level (confirmed / commanded)

**Orchestration.** `start_capture()` fans out start commands to all armed devices in
parallel, collects confirmations with a timeout (default 5 s), and returns a per-camera
result wall. Cameras are *not* required to start simultaneously (the brief allows offset;
LED sync aligns frames later) — what matters is that every start is known and
timestamped. `stop_capture()` mirrors this. Any camera failing to confirm raises a
prominent UI alert while the shot can still be saved by one person walking to one camera.

### 3.2 CamNode firmware

ESP-IDF (C++) with NimBLE. Responsibilities:

- Join capture WLAN (credentials provisioned over USB serial at setup), connect to
  broker, publish retained status at ~1 Hz and on every state change; LWT → offline.
- SNTP time sync against the controller PC (§5).
- Personality behaviours:
  - **gopro**: maintain BLE connection + keep-alive to the assigned camera (by BLE MAC);
    wake camera; apply preset/settings on `arm` and read back for verification; start/stop
    shutter; poll status (encoding, battery %, SD remaining) into MQTT status.
  - **insta360**: hold the emulated-remote pairing; `wake` via manufacturer advertisement
    payload (captured per camera at registration — payloads are per-model/per-unit);
    `shutter` on start/stop. Status = command-delivery only.
  - **ble-hid**: present as a BLE HID keyboard paired to the phone; send volume-key /
    shutter keycodes on start/stop.
- Timestamp every command execution and BLE acknowledgement locally and include the
  timestamps in the MQTT response — command latency over WiFi then never pollutes the
  event log.

Existing open-source references: Open GoPro's own demos, `pchwalek/insta360_ble_esp32`.

### 3.3 MQTT topic schema (sketch)

```
posetrak/node/<node_id>/status          retained; {state, personality, camera, battery_v,
                                        camera_status?, ip, fw_version, time_offset_ms}
posetrak/node/<node_id>/cmd             {op: wake|arm|start|stop|identify|assign, seq, ts}
posetrak/node/<node_id>/resp            {seq, ok, ts_executed, ts_camera_ack?, error?}
posetrak/beacon/<id>/cmd                {op: blink, pattern, at_ts}
posetrak/beacon/<id>/resp               {pattern, ts_actual[]}
```

### 3.4 UI — extend the main posetrak app

The capture-day UI belongs in the **main viewer/editor** (`python/app/ui/`), next to the
existing `CapturePanel` (which already handles capture metadata and persons). New
elements:

- **Device wall**: one tile per camera — link, battery, storage, mode-verified,
  recording state, confidence level. All-green = safe to start. Colours mirror the
  node status LEDs.
- **Rigging checklist** per capture: auto-generated from the camera list; items that
  cannot be machine-verified (Insta360 mode, phone camera settings) are explicit
  operator check-offs, stored with the capture.
- **ARM / START / STOP** controls with the confirmation wall and per-camera failure
  banners; a manual-override tick per camera.
- **Collection view**: per-file progress, then the verify-on-collect report
  (mode/duration mismatches highlighted).

### 3.5 Collection service and verify-on-collect

Runs after teardown, cameras plugged into the collection dock (Android optionally
wireless before teardown):

1. **Enumerate** per type: GoPro wired USB → Open GoPro HTTP media list; Insta360 →
   mass-storage scan; Android → `adb shell ls` of DCIM, filtered to the capture window.
2. **Match** files to `capture_events` rows by camera + timestamp overlap; unmatched
   files and unmatched captures are both flagged.
3. **Copy** to `{session_dir}/{capture_label}/{camera_label}_{start_iso}.mp4`, SHA-256
   on the fly.
4. **Verify**: ffprobe fps/resolution/codec/duration vs the expected `camera_modes` row
   and the logged start–stop window; creation time vs logged start (±2 s warn).
5. **Register**: create/complete `capture_videos` rows (`file_path`, `actual_fps`,
   frame range) and write provenance (§4). The capture wizard's manual
   "add videos" step becomes a review step.

---

## 4. Data model additions

### Registry DB

```sql
-- How a physical camera is controlled. One row per camera_instance that is
-- capture-controllable; cameras without a row default to 'manual'.
CREATE TABLE control_endpoints (
    camera_instance_id TEXT PRIMARY KEY,        -- references camera_instances(id)
    method             TEXT NOT NULL,           -- 'gopro_ble' | 'insta360_ble' |
                                                -- 'adb' | 'ble_hid' | 'manual'
    ble_mac            TEXT,                    -- gopro_ble: camera BLE address
    wake_payload       BLOB,                    -- insta360_ble: per-unit wake adv payload
    adb_serial         TEXT,                    -- adb: persistent device serial
    notes              TEXT
);
```

(Node-to-camera assignment is runtime state held by the controller/node NVS, not
registry data — nodes are interchangeable.)

### Session DB

```sql
-- One row per camera per capture: the capture-control event log.
-- Timestamps are ISO-8601 UTC on the controller clock unless noted.
CREATE TABLE capture_events (
    capture_id          TEXT NOT NULL REFERENCES captures(id),
    camera_instance_id  TEXT NOT NULL,          -- references registry: camera_instances(id)
    start_cmd_ts        TEXT,                   -- start command sent
    start_ack_ts        TEXT,                   -- node/ADB executed the command
    start_confirm_ts    TEXT,                   -- positive recording confirmation (NULL if none)
    stop_cmd_ts         TEXT,
    stop_confirm_ts     TEXT,
    confidence          TEXT NOT NULL,          -- 'confirmed' | 'commanded' | 'manual'
    status              TEXT NOT NULL,          -- 'ok' | 'failed' | 'operator_override'
    notes               TEXT,
    PRIMARY KEY (capture_id, camera_instance_id)
);

-- Collection provenance + verification result on the video itself.
ALTER TABLE capture_videos ADD COLUMN collected_at   TEXT;
ALTER TABLE capture_videos ADD COLUMN source_path    TEXT;   -- path/URI on the camera
ALTER TABLE capture_videos ADD COLUMN sha256         TEXT;
ALTER TABLE capture_videos ADD COLUMN verify_status  TEXT;   -- 'ok' | 'mode_mismatch' |
                                                             -- 'duration_mismatch' | ...
ALTER TABLE capture_videos ADD COLUMN verify_notes   TEXT;
```

Sync-beacon blink schedules go into the existing sync machinery: each commanded blink
produces `sync_anchors` rows with absolute controller-clock timestamps, which the fine-sync
step then matches against detected LED transitions.

---

## 5. Timing design

Requirement: know each recording's start time to ~1 s. Three layers, none of which
requires camera-clock trust:

1. **Controller event log** (`capture_events`) — every start is timestamped at command,
   at node execution, and (where possible) at confirmation. Node clocks are SNTP-synced
   to the PC over the WLAN (≪100 ms), so `start_ack_ts` bounds the true recording start
   to a few hundred ms even for unconfirmed cameras. This alone meets the requirement.
2. **Camera RTC discipline** — makes in-file timestamps agree with the log, which the
   verify-on-collect step exploits:
   - GoPro: **Labs precision-time QR** displayed full-screen by the posetrak app at the
     start of the capture day; walk the phone/laptop past each GoPro (or each GoPro past
     the screen) before rigging.
   - Android: phones NTP-sync themselves; ensure they were online recently
     (checklist item).
   - Insta360: RTC syncs when the camera last connected to its phone app —
     checklist item on capture day.
3. **Sync beacon** — at capture start and end the controller commands a distinctive blink
   pattern and logs the per-flash timestamps. The existing LED fine-sync pipeline gains
   (a) a guaranteed, well-formed sync event in every capture without anyone running
   around with a lamp, and (b) absolute-time anchoring of the common timeline. Sub-frame
   *relative* sync stays the job of the existing cross-correlation step; nothing about
   the tracker's inputs changes.

---

## 6. Capture-day workflow

1. **Base setup** — router on, PC on WLAN, controller app started. Nodes power on and
   appear on the device wall as they connect.
2. **Time-set** — Labs QR shown on screen, GoPros synced (once per day).
3. **Rig** — mount cameras + their nodes, aim, then run extrinsics calibration as today.
   From here on, no one touches a camera.
4. **Pre-flight (ARM)** — controller wakes all cameras; pushes and read-back-verifies
   GoPro presets; checks battery/storage where readable; operator ticks the checklist
   items for Insta360 modes and phone camera settings. Device wall must be all green.
5. **START** — fan-out start; confirmation wall within ~5 s; sync beacon fires its
   start pattern. Any red tile: fix or override before the action starts.
6. **Record** — device wall live-monitors (GoPro encoding flag, phone camera pipeline,
   node link health). A camera dying mid-capture raises an alert *during* the take,
   not after.
7. **STOP** — fan-out stop + beacon end pattern; confirmation wall; capture row and
   `capture_events` finalised.
8. **Collect** — Android pulls can start over WLAN immediately; action cameras go on the
   dock after teardown. Files are renamed, hashed, verified, and registered as
   `capture_videos`; the verification report is reviewed before leaving the venue.

---

## 7. Implementation phases

Ordered so that each phase delivers standalone value and the riskiest external
dependency (Insta360) is isolated:

**Phase 1 — Orchestration core + Android + collection (no custom hardware).**
Capture WLAN, `pipeline/capture` service, ADB driver, device wall UI, `capture_events`
+ provenance schema, collection service with verify-on-collect (all camera types —
verification works on manually-started cameras too). Deliverable: phones fully
automated; every camera verified-on-collect. This alone removes most of the file
handling pain.

**Phase 2 — GoPro CamNodes.** Node firmware with `gopro` personality, MQTT schema,
registration flow, preset push + read-back, confirmed start/stop, live encoding/battery/
SD status. Deliverable: GoPros fully automated with confirmation.

**Phase 3 — Insta360 support (risk-gated).** Run spikes S1/S2 first (below). If the
protocol works on the Ace Pro 2: `insta360` personality (wake + shutter), per-unit wake
payload capture at registration, `commanded` confidence handling in UI. If not: cameras
stay on `ManualDriver` with checklist support — the architecture is unchanged. In
parallel, apply for the official Insta360 SDK.

**Phase 4 — Collection dock automation.** Open GoPro wired-USB driver, Insta360
mass-storage driver, multi-device parallel offload, optional post-copy card cleanup.

**Phase 5 — Sync beacon + polish.** Beacon hardware/firmware, `sync_anchors` integration,
Labs QR time-set screen in the app, `ble-hid` personality (iPhone path, Android fallback).

## 8. Risks and verification spikes

| # | Risk / unknown | Impact | Spike / mitigation |
|---|---|---|---|
| S1 | Ace Pro 2 may not accept the emulated GPS-remote protocol | Phase 3 blocked | Buy a €15 compatible BLE remote, confirm behaviour on the actual camera, sniff its traffic (nRF52 dongle + Wireshark) before writing any firmware |
| S2 | No known Insta360 recording-state feedback | Insta360 stays at `commanded` confidence | Sniff for state in advertisement data / notifications during S1; otherwise rely on verify-on-collect + operator glance |
| S3 | OEM differences in stock-app key-event handling (esp. OnePlus); settings like "volume = shutter" may reset | Phone start/stop unreliable | Per-model smoke-test script in Phase 1; KeyMapper as per-device hardening; `ble-hid` node as fallback |
| S4 | GoPro BLE link stability over multi-hour sessions (sleep, keep-alive) | Missed starts | Node auto-reconnect + wake logic; soak test in Phase 2; device wall makes link loss visible before it matters |
| S5 | Open GoPro wired-USB media download quirks on Hero11 Mini | Phase 4 offload | Verify early with one camera; MTP fallback |
| S6 | Required capture presets (e.g. high-fps modes) not all settable via BLE on the Mini | Mode enforcement partial | Enumerate needed modes against the Open GoPro settings matrix during Phase 2; anything unsettable becomes a checklist item + verify-on-collect |
| — | ADB wireless-debugging port changes / pairing loss after reboot | Phone control friction | mDNS re-discovery in `adb_driver`; re-pair procedure documented in checklist |
| — | 2.4 GHz congestion at venues | Command latency | Router on least-congested channel; commands are tiny and QoS 1 retries; start fan-out tolerates seconds of skew by design |

## 9. Explicitly out of scope

- Frame-accurate hardware genlock / timecode (LED fine sync already provides sub-frame
  alignment; see analysis §4)
- Live video preview/monitoring on the controller (bandwidth-prohibitive; the device
  wall monitors *state*, not pixels)
- Wireless bulk offload for action cameras (dock is faster and simpler; can be added
  per-type later without changing the collection model)
- Controlling exposure/white-balance mid-capture
