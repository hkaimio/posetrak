# Camera Control — System Design Proposal

Solution proposal for the requirements in [camera-control-brief.md](camera-control-brief.md).
Option evaluation and the evidence behind the choices made here is in
[camera-control-analysis.md](camera-control-analysis.md). The wire protocol between the
controller and ESP32 nodes is specified in [node-protocol.md](node-protocol.md).

## Goals recap

From the brief, the system must:

1. Start and stop **4–10+ heterogeneous cameras** from a single PC: GoPro Hero13 (bulk of
   the fleet), Hero11 Mini, Insta360 Ace Pro 2, Android phones running their stock camera
   apps; iPhone path desirable
2. **Confirm** that all cameras actually started/stopped
3. Record per-camera **start times to ~1 s accuracy**
4. Work **wirelessly over 10 m+** with no venue WLAN (own access point allowed; cheap
   per-camera helper hardware allowed)
5. Prevent the classic failure modes: wrong mode, missed start, camera nudged at
   start (invalidating extrinsics)
6. Preferably **collect the recorded files** to the PC automatically, named by camera and
   capture time
7. **Switch the camera LEDs** used for reflective-marker capture and record their state
8. Be operable from a **mobile phone** through a local web UI
9. Leave room for **sub-millisecond synchronized LED patterns** (not built initially)

Remote triggering alone eliminates failure mode "camera moves when the start button is
pressed" — nobody touches a rigged camera after extrinsics calibration.

---

## 1. Architecture overview

Star topology on a dedicated capture WLAN. Cameras and peripherals are separate kinds of
device:

```
  Phone (web UI)        Qt desktop app
        │  HTTP + WebSocket  │
  ┌─────┴────────────────────┴──┐
  │ Controller PC               │
  │ posetrak-capture daemon     │
  │ MQTT broker · time reference│
  └──────────────┬──────────────┘
                 │ WiFi
  ┌──────────────┴──────────────┐
  │ Travel router (capture WLAN)│
  └──┬──────────┬──────────┬────┘
     │ HTTPS    │ ADB      │ MQTT
     │ (COHN)   │          │
  ┌──┴─────┐ ┌──┴──────┐ ┌─┴───────────────────────────┐
  │ Hero13 │ │ Android │ │ Nodes (ESP32)               │
  │  (xN)  │ │ phones  │ │  ble_bridge ──BLE──► Hero11,│
  └────────┘ └─────────┘ │                     Insta360│
                         │  led_ring   (on any camera) │
                         │  beacon     (free-standing) │
                         └─────────────────────────────┘
```

Key decisions (rationale in the analysis doc):

- **Own WLAN via travel router.** WiFi covers the 10 m+ distances reliably; BLE does not.
  The router also isolates the capture network and gives phones and Hero13s something to
  join.
- **Direct control wherever the camera can join the WLAN.** Hero13s are controlled with
  the Open GoPro HTTP(S) API over COHN; Android phones over wireless ADB. No extra
  hardware.
- **Nodes are peripherals with roles.** An ESP32 node has one or more *roles*:
  `ble_bridge` (relay BLE control for a Hero11, Insta360 or phone-HID target), `led_ring`
  (switch camera-mounted LEDs), `beacon` (sync LED). Nodes are associated with a camera
  where it matters (`mounted_on`) but are independent registry entities.
- **Capture daemon with thin clients.** All device logic lives in an asyncio service; the
  Qt app and the web UI are both clients of its HTTP/WebSocket API.
- **MQTT broker on the PC** for node traffic. Retained status topics give an
  always-current device view; last-will marks a node offline when it drops;
  QoS 1 covers command delivery over flaky radio.
- **Scheduled, not immediate, actuation on nodes.** Commands that drive LEDs carry a
  target time on the node's synchronized clock (see §5 and the node protocol), so
  sub-ms synchronization can be introduced later without changing the controller.

### Design principle: confirm where possible, verify-on-collect always

Not every camera can confirm "recording started" (Insta360 cannot). Rather than pretend,
the system is explicit about three confidence levels per camera, shown in the UI:

| Level | Meaning | Cameras |
|---|---|---|
| **Confirmed** | positive status readback (encoding flag / camera pipeline active) | GoPro (Hero13, Hero11), Android |
| **Commanded** | command delivered to the camera, no readback | Insta360, iPhone |
| **Manual** | operator started it by hand | anything, as fallback |

Independently, the **collection step verifies every file** after the fact — fps,
resolution, duration, and timestamps are probed and compared against the expected camera
mode and capture window. A wrong-mode or short-file problem is caught the same day with
the rig still standing, not weeks later at pose-extraction time.

---

## 2. Hardware

### Bill of materials (10-camera setup)

Assumes 8 Hero13s, 1 Insta360 and 1 phone, with marker lighting on every camera.

| Item | Qty | Unit ≈ | Notes |
|---|---|---|---|
| Travel router (GL.iNet GL-A1300 / GL-MT3000 class) | 1 | €50 | dual-band, USB-powerable; 2.4 GHz for range, 5 GHz for offload |
| Node: ESP32-S3 or C3 dev board (e.g. Seeed XIAO) | 1 per camera needing a bridge or LED ring | €7 | WiFi + BLE 5, tiny |
| Node battery: 500 mAh LiPo + charger circuit | per node | €4 | days of standby, USB-C charge |
| Node case + mounting (printed case, velcro/clamp) | per node | €2 | attach to camera clamp/tripod |
| LED ring and driver | per camera in marker mode | tbd | designed with the lighting hardware |
| Sync beacon: ESP32 + 1–3 W LED + driver + battery | 1–2 | €15 | visible to all cameras; 2 units for opposing camera arcs |
| Powered USB hub, ≥10 ports ("collection dock") | 1 | €30 | Hero11 / Insta360 offload and charging after teardown |
| USB cables (USB-C) | ~10 | €3 | |

Without marker lighting, Hero13s and phones need no per-camera hardware; only the
Hero11 and Insta360 do. Phones and the PC need no extra hardware.

### Node hardware notes

- A status RGB LED mirrors node state (offline / connected / armed / recording / error)
  so a glance across the room shows rig health even without the UI.
- One push-button for pairing/identify.
- Nodes are generic and interchangeable: roles and the target camera are assigned from
  the controller at registration time and stored in NVS, so a spare node can replace a
  failed one in the field.
- 1 node : 1 camera by default. The Insta360 role is a BLE *peripheral* (it advertises as
  a remote) while the GoPro role is a *central*; do not plan on mixing them in one node.
- Firmware is developed in a separate Rust repository. This repository owns only the
  protocol contract ([node-protocol.md](node-protocol.md)).

---

## 3. Software components

### 3.1 Capture daemon — `python/pipeline/capture/`

New Python package, run as a standalone service (`posetrak-capture`) that serves an HTTP
API and a WebSocket event stream (§3.4). It runs on asyncio, which matches the Open GoPro
SDK and `bleak`.

```
python/pipeline/capture/
├── service.py        # CaptureController: orchestration + state machines
├── api.py            # HTTP + WebSocket server, auth, control lease
├── devices.py        # Device abstraction + per-type drivers
├── gopro_cohn.py     # Hero13: COHN provisioning, HTTPS control, media API
├── mqtt_link.py      # broker mgmt (embedded mosquitto or amqtt), topic schema
├── nodes.py          # node registry, roles, scheduled commands, clock monitoring
├── adb_driver.py     # Android control: pairing, connect, intents, keyevents, dumpsys
├── collect.py        # collection service + verify-on-collect
├── lighting.py       # LED ring + beacon control, capture lighting state
└── protocol.py       # MQTT message dataclasses, versioned (see node-protocol.md)
```

**Device model.** Each controllable camera is a `CaptureDevice` bound to a registry
`camera_instances` row, with a driver:

- `GoProCohnDriver` — Hero13: direct HTTPS to the camera over the capture WLAN
- `GoProNodeDriver` — Hero11: MQTT to a node with the `ble_bridge` role targeting a GoPro
- `Insta360NodeDriver` — MQTT to a `ble_bridge` node in Insta360 remote mode
- `BleHidNodeDriver` — MQTT to a `ble_bridge` node in HID mode (iPhone / Android fallback)
- `AdbDriver` — direct ADB over the WLAN
- `ManualDriver` — no automation; the UI shows operator instructions and asks for
  a manual tick (safety net so a broken node never blocks a shoot)

**Peripherals** are `PeripheralDevice` objects bound to a registry `peripherals` row
(node id, role, optional `mounted_on` camera). They have their own state (link, clock sync
quality, battery) and appear in the device view next to cameras, but take no part in the
recording state machine.

**Per-camera state machine:**

```
OFFLINE → DISCOVERED → READY → ARMED → RECORDING → STOPPING → STOPPED → COLLECTED
                         │        │
                         └── fault states carry a reason (low battery, SD full,
                             mode mismatch, link lost) and are always recoverable
                             by falling back to ManualDriver
```

- `READY`: link up, battery/storage OK (where readable)
- `ARMED`: pre-flight passed — GoPros have had their preset pushed *and read back*;
  checklist items acknowledged for cameras without remote mode control; if the capture
  uses marker lighting, every `led_ring` node is online with an acceptable clock
- `RECORDING`: per the device's confidence level (confirmed / commanded)

**Orchestration.** `start_capture()` fans out start commands to all armed devices in
parallel, collects confirmations with a timeout (default 5 s), and returns a per-camera
result wall. Cameras are *not* required to start simultaneously (the brief allows offset;
LED sync aligns frames later) — what matters is that every start is known and
timestamped. `stop_capture()` mirrors this. Any camera failing to confirm raises a
prominent alert while the shot can still be saved by one person walking to one camera.

**Hero13 provisioning.** A guided flow (desktop UI) pairs each Hero13 over BLE next to
the PC, joins it to the capture WLAN, and stores its COHN credentials in a local keystore
beside the controller configuration (never in a session DB that may be shared). Discovery
on capture day resolves the camera's address on the WLAN; `control_endpoints` stores the
identity, not the address.

### 3.2 Node firmware (separate repository)

Responsibilities that the controller relies on; implementation choices are the firmware
repository's. The interface is [node-protocol.md](node-protocol.md).

- Join the capture WLAN (credentials provisioned over USB serial), connect to the broker,
  publish retained status on connect, on every state change, and periodically; LWT →
  offline.
- Keep a node clock synchronized to the controller and report the method and error
  estimate in status.
- Execute commands at their scheduled time and report the time actually executed.
- Roles:
  - `ble_bridge` / GoPro: maintain BLE connection and keep-alive to the assigned camera
    (by BLE address); wake; apply preset/settings on `arm` and read back; start/stop
    shutter; publish camera status (encoding, battery, SD remaining).
  - `ble_bridge` / Insta360: hold the emulated-remote pairing; wake via manufacturer
    advertisement payload (captured per camera at registration); shutter on start/stop.
    Status is command-delivery only.
  - `ble_bridge` / HID: present as a BLE HID keyboard paired to the phone; send volume-key
    / shutter keycodes.
  - `led_ring`: set LED state and brightness, now or at a scheduled time.
  - `beacon`: play a blink pattern at a scheduled time and report per-flash timestamps.

Existing open-source references for the BLE roles: Open GoPro's own demos,
`pchwalek/insta360_ble_esp32`.

### 3.3 Messaging

The MQTT topic and message schema, versioning and clock model are specified in
[node-protocol.md](node-protocol.md). The controller treats the protocol version in node
status as a compatibility gate: a node with an unsupported version appears in the device
view as faulty and is excluded from arming.

### 3.4 Control API and UIs

**API.** The daemon exposes:

- `GET /api/state` — snapshot of all devices, the current capture, and the control lease
- `WebSocket /api/events` — state changes, per-camera confirmations, alerts
- `POST /api/capture/{arm,start,stop}`, `POST /api/lighting`, `POST /api/checklist/{id}`
- `POST /api/lease` — take or release the control lease

Provisioning, collection review and destructive operations (card cleanup) are exposed
only to the desktop client.

**Web UI (mobile-first).** Single-page app served as static files by the daemon,
written without a Node build chain (Preact or htmx). It covers the operational loop:

- Device view: one tile per camera and peripheral — link, battery, storage,
  mode-verified, recording state, confidence level, node clock quality. All-green = safe
  to start.
- Large ARM / START / STOP controls with the confirmation wall and per-camera failure
  banners; manual-override tick per camera.
- Marker lighting toggle (applies to all `led_ring` nodes), and per-node identify.
- Rigging checklist items that need an operator tick.

Access:

- Reachable at a fixed name (`posetrak.local`, mDNS) plus a QR code shown in the desktop
  app that carries the URL and an access token.
- The token is required for every API call; the capture WLAN is also WPA2-protected.
- Any number of clients can observe. One holds the **control lease** and is the only one
  whose START/STOP/lighting commands are accepted; taking over from another client is an
  explicit action that is logged.
- A client that disconnects never changes capture state. The daemon is the authority.

**Desktop UI.** The capture-day panel extends the main viewer/editor (`python/app/ui/`),
next to the existing `CapturePanel`, as a client of the same API. It adds what the web UI
leaves out: Hero13 provisioning, the collection view (per-file progress and the
verify-on-collect report), and node registration.

### 3.5 Collection service and verify-on-collect

Runs after capture. Hero13s and Android phones can be collected over the WLAN right after
stopping; Hero11 and Insta360 cameras go on the dock after teardown.

1. **Enumerate** per type: Hero13 → Open GoPro HTTP media list over COHN; Hero11 → wired
   USB HTTP media list; Insta360 → mass-storage scan; Android → `adb shell ls` of DCIM,
   filtered to the capture window.
2. **Match** files to captures. For a capture the daemon controls, the camera's media
   list is read when the cameras are armed and again after they have stopped, and the
   videos that appeared in between are the capture's; no clock is involved. Files recorded
   without the daemon have to be matched by time: the creation time of the per-file
   metadata and of the container is UTC, while the one in the media list is the camera's
   local time written as if it were UTC, and the camera's clock may be off, so a match by
   time needs a tolerance and a camera clock check.
3. **Copy** to `<offload dir>/capture-<number>/<camera label>/<camera folder>/`, SHA-256
   on the fly, resuming interrupted copies. Wireless downloads run in parallel with a
   concurrency cap so the WLAN is not saturated. The cameras are not armed again until
   copying has finished or been cancelled.
4. **Verify**: size against what the camera announced, and the detected capture mode
   against the one the camera was armed with (§3.6); a mismatch raises an alert.
5. **Register** (see §3.6): create/complete `capture_videos` rows (`file_path`, `actual_fps`,
   frame range) and write provenance (§4). The capture wizard's manual
   "add videos" step becomes a review step.

### 3.6 Registry integration: identity, capture modes, credentials

The capture daemon takes everything about a physical camera from the registry. The
daemon configuration names which cameras take part in a capture and under which
configuration; it does not repeat serial numbers or settings.

**Identity.** `control_endpoints` links a `camera_instances` row to its control method and
device identity (`gopro_serial`, `ble_mac`, `adb_serial`, ...). Provisioning starts from a
registry camera: `provision <label>` resolves the serial from the endpoint row. If the
camera has none, a BLE scan lists the cameras in range, the operator picks one, and the
link is stored. `camera_instances.serial_number` is entered by hand and is often empty,
so it is not relied on for matching. The link is the endpoint row, made once by the
operator. Once the camera is reachable, provisioning reads the full manufacturer serial
from it and offers to fill an empty `serial_number` (a differing value is reported, never
overwritten). That serial also appears in the metadata of the camera's video files, so it
later lets mode detection and the capture wizard identify the instance of a video taken
outside the daemon, and warns when a link points at the wrong registry camera.

**Capture modes.** A *mode* is a `camera_modes` row of the camera's model. Intrinsic
calibrations are attached to modes, so a capture is only useful for tracking when it
used a mode that exists in the registry. The picker therefore offers the modes of each
camera's model. `camera_modes` has no lens, performance or exposure fields and the set of
parameters differs per model, so a mode also carries the settings its driver applies:

```sql
ALTER TABLE camera_modes ADD COLUMN name TEXT;               -- unique within the model
ALTER TABLE camera_modes ADD COLUMN capture_settings TEXT;   -- JSON object, driver-specific
```

The name is what setups and the operator refer to (`4K 120 linear`). Whether a camera's
settings can be set remotely is a property of its driver (below), not of the mode, so the
same mode row serves a camera that is configured over the link and one that is set by hand.

`capture_settings` holds the keys of the driver's settings vocabulary (for the Hero13
those of `gopro_profile.translate`: resolution, fps, lens, bit_rate, performance, or a raw
numeric setting id). The driver derives what it can from the mode's width, height and fps
(for the Hero13 resolution and frame rate, not the lens) and `capture_settings` is applied on
top, so a mode only lists what the standard fields cannot tell and may override a derived
value. The settings that define a mode's *image* (resolution, fps, lens) are part of
the mode identity and select its intrinsics; settings that do not (exposure, bit rate)
may differ between captures within one mode.

**Capture setups.** Users define named *capture setups* in the daemon configuration. A
setup says which mode each camera uses when it is selected, either per camera label or
per camera model, plus optional per-camera overrides:

```toml
[setup.default]
mode = { "GoPro Hero13" = "4K 120 linear", "GoPro Hero11" = "4K 60 linear" }
[setup.default.camera.gopro13_03]
mode = "2.7K 120 linear"          # override
[setup.default.camera.gopro13_03.settings]
exposure_time = "1/480"           # extra driver parameter on top of the mode
```

The operator picks a setup in the UI (default if there is only one) and may override a
camera's mode for one capture. Cameras of different models in one rig get different
modes by construction. Resolution of "which mode does camera X use" is: per-capture
override, then the setup's per-camera entry, then its per-model entry; a camera that
resolves to nothing is reported as not ready.

**Driver contract.** Every driver implements:

- `supported_settings() -> dict[str, SettingSpec]`: the vocabulary of parameters the
  driver understands with their types and allowed values. The UI renders per-capture
  adjustments (exposure time etc.) from it, so a new parameter is a driver change only.
- `settings_control` capability: `driver` (settings can be applied and verified over the
  control channel; Hero13, later Hero11) or `manual` (Android, iPhone, Insta360 Ace Pro 2
  cannot be configured remotely). For `manual` cameras arm does not write settings; it
  shows the mode the operator must select on the device and the checklist records the
  confirmation. A `manual` camera is never reported as ready on the controller's say-so
  alone.
- `describe_mode(mode) -> list[str]`: the instruction text for the manual case.
- `read_settings()` where the camera can report them, so arm can verify even a camera
  the operator set by hand.

The mode actually used is recorded per camera in `capture_events` (`camera_mode_id`,
`settings_json` with the settings as read back, or as commanded for `manual` cameras).

**Mode detection from video files.** Calibration footage and any other video recorded
outside the capture daemon must land on the right mode without the camera being connected.
Detection is a function of the file (`pipeline.capture.mode_detect`), independent of any
control connection:

- `read_video_facts(path)` reads the recording details with `exiftool`: image size, frame
  rate, and where the file carries them make, model, serial number and field of view.
- `detect_mode(facts, modes)` ranks the registry modes of the camera's model. Each match
  lists the evidence that agrees and what could not be checked; only a mode for which
  everything checked out counts as certain, and a file is assigned without asking only when
  exactly one mode is certain.
- `match_registry(registry, facts)` finds the registry camera by the serial number in the
  file (the registry serial, or the end of it for a camera linked for control before its
  serial was recorded), and takes the modes of its model; without a registered serial the
  camera model named in the file decides.

What the files carry decides the rules, and the findings are these. A GoPro file states its
lens (`FieldOfView`) and the camera's full serial number in its telemetry header, and the
metadata the camera serves over its HTTP API does not state the lens, so detection has to
read the file. Cameras record NTSC frame rates (119.88 for a nominal 120, 239.76 for 240),
so frame rates match within one percent. Image size and frame rate alone do not tell
modes of the same size apart (linear and superview at 4K 120 fps look the same to them),
which is why the lens is part of a mode. Other cameras register their own readers of what
their files state; the generic fallback has only size and frame rate and never reaches
certainty for a mode that states a lens. The result is used in three places:

1. Collection verify (§3.5): the detected mode is compared with the commanded one;
   disagreement gives `verify_status = 'mode_mismatch'`.
2. The capture wizard: videos added by hand get `shot_videos.camera_mode_id`
   pre-filled. A unique match with high confidence is assigned silently; an ambiguous
   one (for instance linear and wide lens indistinguishable from the metadata) is
   presented as a choice with the evidence.
3. Calibration import: the matched mode selects its default intrinsics calibration.

Registration (`pipeline.capture.register`) applies the same rules when a file is added to a
capture. The mode recorded for the file is the one the file agrees with, and the mode the
camera was armed with only fills in when the file cannot decide between modes that include
it. A file that contradicts the armed mode is recorded with the mode it is, and flagged
(`verify_status = 'mode_mismatch'`); a file that fits no registered mode gets no mode. The
registry rows of the camera and the mode are copied into the session on the way.

**Credentials.** COHN credentials belong to the camera, not to the PC, and any machine
on the WLAN can use them, so they live in the registry with the rest of the machine
setup, but in a table of their own:

```sql
CREATE TABLE control_credentials (
    camera_instance_id TEXT NOT NULL,       -- references camera_instances(id)
    ssid               TEXT NOT NULL,
    ip_address         TEXT NOT NULL,
    username           TEXT NOT NULL,
    password           TEXT NOT NULL,
    certificate        TEXT NOT NULL,
    PRIMARY KEY (camera_instance_id, ssid)
);
```

The password gives full control of the camera. This table exists only in the registry
and is never copied into session databases or into registry exports, which differs from
`control_endpoints`, `camera_models` and `camera_instances`, whose rows sessions embed. It
is defined in `registry_private_schema.sql`, which only the registry applies, and a test
checks that a new session lacks it. The password is stored as plain text, so the registry
file is as sensitive as the cameras; its protection is that of the user's profile directory.
Credentials saved in files by older versions are read as a fallback until
`posetrak-capture import-credentials` copies them into the registry.

**Operator flow.**

1. *Add a camera* (CLI first, web UI later): choose or create a registry camera
   instance, pick the device from a scan, choose the WLAN, provision. The endpoint,
   credentials and instance end up linked.
2. *Define modes* for each camera model in the registry (settings editor in the
   desktop app); a mode can be created from a camera's current settings.
3. *Define setups* in the daemon configuration.
4. *Capture*: pick a setup, optionally override per camera, arm. Arm applies and
   verifies settings on `driver` cameras and lists the manual instructions for the rest.

---

## 4. Data model additions

### Registry DB

```sql
-- How a physical camera is controlled. One row per camera_instance that is
-- capture-controllable; cameras without a row default to 'manual'.
CREATE TABLE control_endpoints (
    camera_instance_id TEXT PRIMARY KEY,        -- references camera_instances(id)
    method             TEXT NOT NULL,           -- 'gopro_cohn' | 'gopro_ble' |
                                                -- 'insta360_ble' | 'adb' | 'ble_hid' |
                                                -- 'manual'
    gopro_serial       TEXT,                    -- gopro_cohn: stable camera identity
                                                -- (COHN credentials live in the local
                                                -- keystore, not here)
    ble_mac            TEXT,                    -- gopro_ble: camera BLE address
    wake_payload       BLOB,                    -- insta360_ble: per-unit wake adv payload
    adb_serial         TEXT,                    -- adb: persistent device serial
    notes              TEXT
);

-- ESP32 nodes. A node may carry several roles.
CREATE TABLE peripherals (
    node_id                TEXT PRIMARY KEY,
    label                  TEXT NOT NULL,
    roles                  TEXT NOT NULL,       -- JSON array: 'ble_bridge' | 'led_ring' |
                                                -- 'beacon'
    mounted_on_camera_id   TEXT,                -- references camera_instances(id); NULL for
                                                -- a free-standing beacon
    notes                  TEXT
);
```

Node-to-target assignment for a `ble_bridge` (which camera's BLE address it controls) is
configuration in the node's NVS and the controller's device configuration, so a spare
node can be repurposed in the field.

### Session DB

```sql
-- One row per camera per capture: the capture-control event log.
-- Timestamps are ISO-8601 UTC on the controller clock unless noted.
CREATE TABLE capture_events (
    capture_id          TEXT NOT NULL REFERENCES captures(id),
    camera_instance_id  TEXT NOT NULL,          -- references registry: camera_instances(id)
    start_cmd_ts        TEXT,                   -- start command sent
    start_ack_ts        TEXT,                   -- node/ADB/camera executed the command
    start_confirm_ts    TEXT,                   -- positive recording confirmation (NULL if none)
    stop_cmd_ts         TEXT,
    stop_ack_ts         TEXT,
    stop_confirm_ts     TEXT,
    camera_mode_id      TEXT,                   -- mode the camera was armed with (§3.6);
                                                -- references registry: camera_modes(id)
    settings_json       TEXT,                   -- settings read back (or commanded, for
                                                -- manual cameras)
    confidence          TEXT NOT NULL,          -- 'confirmed' | 'commanded' | 'manual'
    status              TEXT NOT NULL,          -- 'ok' | 'failed' | 'operator_override'
    notes               TEXT,
    PRIMARY KEY (capture_id, camera_instance_id)
);

-- Peripheral actions during a capture: LED state changes, beacon blinks.
CREATE TABLE capture_peripheral_events (
    id                  INTEGER PRIMARY KEY,
    capture_id          TEXT NOT NULL REFERENCES captures(id),
    node_id             TEXT NOT NULL,          -- references registry: peripherals(node_id)
    op                  TEXT NOT NULL,          -- 'led_set' | 'blink'
    scheduled_ts        TEXT,                   -- target time on the controller clock
    actual_ts           TEXT,                   -- time reported by the node, mapped to the
                                                -- controller clock
    clock_err_us        INTEGER,                -- node's error estimate at execution
    params_json         TEXT                    -- brightness, pattern, ...
);

-- Intended lighting for the capture (whether the markers were lit).
ALTER TABLE captures ADD COLUMN marker_lighting TEXT;  -- 'off' | 'on'

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
   at execution, and (where possible) at confirmation. Hero13 and ADB commands are
   timestamped by the controller around the HTTP/ADB call. Node clocks are synchronized
   to the PC over the WLAN, so `start_ack_ts` bounds the true recording start to a few
   hundred ms even for unconfirmed cameras. This alone meets the requirement.
2. **Camera RTC discipline** — makes in-file timestamps agree with the log, which the
   verify-on-collect step exploits:
   - GoPro: **Labs precision-time QR** displayed full-screen by the posetrak app at the
     start of the capture day; walk the phone/laptop past each GoPro (or each GoPro past
     the screen) before rigging.
   - Android: phones NTP-sync themselves; ensure they were online recently
     (checklist item).
   - Insta360: RTC syncs when the camera last connected to its phone app —
     checklist item on capture day.
3. **Sync beacon** — at capture start and end the controller schedules a distinctive
   blink pattern and logs the per-flash timestamps. The existing LED fine-sync pipeline
   gains (a) a guaranteed, well-formed sync event in every capture without anyone running
   around with a lamp, and (b) absolute-time anchoring of the common timeline. Sub-frame
   *relative* sync stays the job of the existing cross-correlation step; nothing about
   the tracker's inputs changes.

### Node clock and future sub-ms synchronization

Node commands that drive LEDs carry a target time on the node clock. Each node reports
the method it uses to track the controller clock and the current error estimate:

- `sntp` — periodic SNTP exchange over the WLAN (tens of ms). Sufficient for beacon
  blinks and LED on/off.
- Future, e.g. a broadcast sync frame all nodes timestamp (ESP-NOW or WiFi TSF) or a
  wired trigger, for structured lighting that needs sub-ms agreement between cameras.

The controller refuses to arm a capture whose lighting plan needs better precision than
the worst node reports. For the initial on/off lighting and the beacon this limit is set
loosely, so SNTP nodes pass. Introducing a better method is a firmware and protocol-minor
change; the controller scheduling code is unchanged.

---

## 6. Capture-day workflow

1. **Base setup** — router on, PC on WLAN, `posetrak-capture` running. Hero13s, nodes and
   phones appear on the device view as they connect; the operator can switch to the
   phone web UI from here.
2. **Time-set** — Labs QR shown on screen, GoPros synced (once per day).
3. **Rig** — mount cameras + their nodes, aim, then run extrinsics calibration as today.
   LEDs can be toggled remotely for rigging checks. From here on, no one touches a camera.
4. **Pre-flight (ARM)** — controller wakes all cameras; pushes and read-back-verifies
   GoPro presets; checks battery/storage where readable; verifies node clocks if
   lighting is on; operator ticks the checklist items for Insta360 modes and phone camera
   settings. The device view must be all green.
5. **START** — fan-out start; confirmation wall within ~5 s; lighting set per the capture
   plan; sync beacon fires its start pattern. Any red tile: fix or override before the
   action starts.
6. **Record** — live monitoring (GoPro encoding flag, phone camera pipeline, node and
   camera link health). A camera dying mid-capture raises an alert *during* the take,
   not after.
7. **STOP** — fan-out stop + beacon end pattern; confirmation wall; capture row and
   `capture_events` finalised; LEDs off.
8. **Collect** — Hero13 and Android downloads can start over the WLAN immediately;
   Hero11 and Insta360 cameras go on the dock after teardown. Files are renamed, hashed,
   verified, and registered as `capture_videos`; the verification report is reviewed
   before leaving the venue.

---

## 7. Implementation phases

Ordered so that each phase delivers standalone value, the highest-priority cameras come
first, and the riskiest external dependencies (Hero13 COHN behaviour, Insta360) are
isolated behind spikes.

**Phase 1 — Daemon, API and Hero13 control.** `posetrak-capture` service, HTTP/WebSocket
API with token auth and control lease, `gopro_cohn` driver (provisioning, preset push and
read-back, start/stop, status), `capture_events` schema, first web UI (device view,
arm/start/stop). Gated on spikes S7 and S8. Deliverable: the Hero13 fleet fully automated
with confirmation, from desktop or phone.

**Phase 2 — Collection.** Collection service with verify-on-collect for all camera types
(works on manually started cameras too), Hero13 parallel wireless download (gated on S9),
provenance schema, desktop collection view.

**Phase 3 — Node protocol and sync beacon.** Protocol spec implemented in the controller
(`nodes.py`, `mqtt_link.py`), tested against a mock node and then the Rust firmware;
`peripherals` table, beacon scheduling, `sync_anchors` integration, Labs QR time-set
screen.

**Phase 4 — Marker lighting.** `led_ring` role, `captures.marker_lighting`,
`capture_peripheral_events`, lighting toggle in web and desktop UI, arm-time checks.
Downstream dot detection can read the recorded lighting state.

**Phase 5 — Hero11 and Insta360.** `ble_bridge` nodes and drivers, dock offload for
Hero11 (Open GoPro wired HTTP) and Insta360 (mass storage). Insta360 runs spikes S1/S2
first; if the protocol does not work on the Ace Pro 2 those cameras stay on `ManualDriver`
with checklist support and the architecture is unchanged. In parallel, apply for the
official Insta360 SDK.

**Phase 6 — Android, iPhone and sub-ms groundwork.** ADB driver (with smoke tests per
phone model), `ble_hid` role for the iPhone path, and the spike on a sub-ms node
synchronization method (S10).

## 8. Risks and verification spikes

| # | Risk / unknown | Impact | Spike / mitigation |
|---|---|---|---|
| S1 | Ace Pro 2 may not accept the emulated GPS-remote protocol | Insta360 control blocked | Buy a €15 compatible BLE remote, confirm behaviour on the actual camera, sniff its traffic (nRF52 dongle + Wireshark) before writing any firmware |
| S2 | No known Insta360 recording-state feedback | Insta360 stays at `commanded` confidence | Sniff for state in advertisement data / notifications during S1; otherwise rely on verify-on-collect + operator glance |
| S3 | OEM differences in stock-app key-event handling (esp. OnePlus); settings like "volume = shutter" may reset | Phone start/stop unreliable | Per-model smoke-test script; KeyMapper as per-device hardening; `ble_hid` node as fallback |
| S4 | Hero11 BLE link stability over multi-hour sessions (sleep, keep-alive) | Missed starts | Node auto-reconnect + wake logic; soak test; device view makes link loss visible before it matters |
| S5 | Open GoPro wired-USB media download quirks on Hero11 Mini | Hero11 offload | Verify early with one camera; MTP fallback |
| S6 | Required capture presets (e.g. high-fps modes) not all settable via the Open GoPro API on a given model | Mode enforcement partial | Enumerate needed modes against the Open GoPro settings matrix for Hero13 and Hero11; anything unsettable becomes a checklist item + verify-on-collect |
| S7 | Hero13 COHN link stability, battery drain and heat while recording at mocap frame rates, behaviour on camera auto-sleep | Hero13 control and the "leave cameras on" policy | Multi-hour soak with 2+ cameras recording at the target preset; log link drops and temperature warnings; decide wake policy |
| S8 | Waking a Hero13 that is off, from the PC at capture distances | Missed starts if cameras are off | Measure BLE wake range outdoors; fall back to leaving cameras on or waking from a nearby node |
| S9 | Aggregate WLAN throughput for parallel Hero13 downloads | Collection time; WLAN contention with control traffic | Measure with 6–10 cameras on 5 GHz; tune concurrency cap; keep the wired path as fallback |
| S10 | Method for sub-ms synchronization between LED nodes (ESP-NOW / TSF / wired) | Structured lighting | Prototype in the firmware repository; measure achievable jitter between two nodes with a logic analyzer |
| S11 | Rust BLE on ESP32 supporting both central (GoPro) and peripheral (Insta360 remote) roles | Hero11 / Insta360 bridge | Confirm against the chosen Rust BLE stack before Phase 5 |
| — | ADB wireless-debugging port changes / pairing loss after reboot | Phone control friction | mDNS re-discovery in `adb_driver`; re-pair procedure documented in checklist |
| — | 2.4 GHz congestion at venues | Command latency | Router on least-congested channel; commands are tiny and QoS 1 retries; start fan-out tolerates seconds of skew by design |
| — | Unauthorized client on the capture WLAN starting/stopping cameras | Lost or ruined takes | WPA2 + API token + control lease; destructive operations desktop-only |

## 9. Explicitly out of scope

- Frame-accurate hardware genlock / timecode (LED fine sync already provides sub-frame
  alignment; see analysis §4)
- Live video preview/monitoring on the controller or phone (bandwidth-prohibitive; the
  device view monitors *state*, not pixels)
- Exposure/white-balance changes mid-capture
- Structured-lighting patterns and the sub-ms sync mechanism themselves; this design only
  keeps the node protocol and scheduling model compatible with them
- Internet-facing access to the web UI; it is for the capture WLAN only
