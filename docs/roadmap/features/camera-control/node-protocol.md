# Node Protocol

Wire contract between the Posetrak capture controller
([camera-control-design.md](camera-control-design.md)) and ESP32 nodes (camera-mounted
bridges and LED rings, sync beacons). The controller is implemented in this repository;
node firmware is developed in a separate repository and implements this document.

Status: draft, protocol version `1`.

## 1. Principles

- **Nodes are peripherals with roles**, not camera bridges. A node declares the roles it
  supports; the controller activates and configures them.
- **Actuation is scheduled.** A command that drives an output carries the time it must
  take effect, expressed on the node clock. "As soon as possible" is the special case of
  a time in the past.
- **Nodes report truth.** Every response carries the time the node actually executed the
  command and its clock error estimate at that moment. The controller logs both.
- **Transport is replaceable, semantics are not.** Messages are JSON over MQTT; the
  clock model and command semantics do not depend on that choice.
- **Compatibility by version gate.** The protocol version is in node status; the
  controller refuses to arm with a node whose major version it does not support.

## 2. Transport

- MQTT 3.1.1 or 5 against the broker on the controller PC, on the capture WLAN.
- Command and response messages: QoS 1. Status: retained, QoS 1.
- Each node sets a last-will on its status topic publishing `{"state":"offline"}`
  (retained).
- Node ids are stable strings (for example derived from the MAC address) and are the
  `<node_id>` segment in topics below.

## 3. Topics

```
posetrak/node/<node_id>/status      node → controller, retained
posetrak/node/<node_id>/cmd         controller → node
posetrak/node/<node_id>/resp        node → controller
posetrak/node/<node_id>/event       node → controller, unsolicited (button press, fault)
posetrak/time/sync                  controller → all nodes, clock reference (see §5)
```

All payloads are UTF-8 JSON objects.

## 4. Messages

### 4.1 Status

Published on connect, on every change, and at least every 5 s.

```json
{
  "proto": "1.0",
  "state": "online",
  "fw_version": "0.3.1",
  "roles": ["ble_bridge", "led_ring"],
  "role_config": {
    "ble_bridge": {"target": "gopro", "ble_addr": "AA:BB:CC:DD:EE:FF"},
    "led_ring":   {"channels": 1, "max_brightness": 255}
  },
  "battery_v": 3.92,
  "ip": "192.168.8.41",
  "clock": {"method": "sntp", "err_us": 12000, "synced": true},
  "led": {"on": false, "brightness": 0},
  "camera": {"link": "connected", "recording": false, "battery_pct": 81,
             "sd_remaining_s": 5400}
}
```

- `roles` lists what the firmware supports; `role_config` holds the active configuration
  assigned by the controller (persisted in node NVS).
- `camera` is present only when a `ble_bridge` role is active. Fields the target camera
  does not report are omitted, not zeroed.
- `clock.method` is one of `none`, `sntp`, `radio_sync` (broadcast sync frame) or
  `wired`. `err_us` is the node's current estimate of its offset error relative to the
  controller clock (1σ).

### 4.2 Commands

Every command has a controller-assigned `seq` (unique per node), and an optional `at`.

```json
{"seq": 42, "op": "led_set", "at": 1760712345.250, "on": true, "brightness": 200}
```

- `at` is a time on the node clock mapped to Unix seconds with fractional part
  (controller time scale). Omitted or in the past means "now".
- If a command's `at` is more than 5 s ahead the node rejects it with `too_far`.
- A node executes commands for the same output in `at` order.

Operations:

| `op` | Roles | Parameters | Effect |
|---|---|---|---|
| `assign` | any | `role`, `config` | Set and persist role configuration |
| `identify` | any | `duration_s` | Flash status LED |
| `led_set` | `led_ring` | `on`, `brightness` (0-255) | Set LED state |
| `blink` | `beacon` | `pattern` (see §6), `at` required | Play a pattern |
| `wake` | `ble_bridge` | | Wake the target camera |
| `arm` | `ble_bridge` | `preset`/`settings` | Apply camera settings and read back |
| `start` | `ble_bridge` | | Start recording on the target camera |
| `stop` | `ble_bridge` | | Stop recording |
| `time_probe` | any | | Reply immediately with the node clock (§5) |

Camera-control ops (`wake`, `arm`, `start`, `stop`) are not scheduled for sub-ms
precision: `at` is honored but only to the precision of the BLE link.

### 4.3 Responses

```json
{"seq": 42, "ok": true, "ts_executed": 1760712345.2503,
 "clock_err_us": 12000, "ts_camera_ack": null}
{"seq": 43, "ok": false, "error": "too_far"}
```

- `ts_executed`: the node clock time at which the output actually changed (or the command
  was issued to the camera).
- `ts_camera_ack`: for `ble_bridge`, the time the camera acknowledged, when it does.
- `error` is one of: `bad_request`, `unsupported_role`, `too_far`, `busy`,
  `camera_unreachable`, `camera_rejected`, `internal`.
- For `blink`, the response additionally carries `ts_actual: [...]`, the time of each
  flash edge.

### 4.4 Events

Unsolicited: `{"event": "button", "ts": ...}`, `{"event": "fault", "reason": "...",
"ts": ...}`, `{"event": "camera_link", "link": "lost", "ts": ...}`.

## 5. Clock model

The controller needs to schedule actions on many nodes at the same instant and to relate
node-reported times to its own clock.

- **Reference.** The controller PC clock. Time values on the wire are Unix seconds with
  fractional part.
- **Current method (`sntp`).** The controller publishes an NTP-style reference over SNTP
  on the capture WLAN; nodes track it. Expected error is tens of ms with jitter. Nodes
  report `err_us` including the observed round-trip spread.
- **`time_probe`.** The controller can measure each node's offset at any time: it sends
  `time_probe` with its own send time and the node answers with its receive and send
  times, NTP-style. The controller records offsets in the event log for post-hoc
  correction regardless of which synchronization method is in use.
- **Future methods.** `radio_sync` (all nodes timestamp a broadcast frame, ESP-NOW or
  WiFi TSF) and `wired` are reserved values for `clock.method`. They need no protocol
  change other than the firmware reporting them; the broadcast frame itself is
  firmware-internal and not carried on MQTT. `posetrak/time/sync` is reserved for
  methods that distribute a reference over MQTT.
- **Arming rule.** The controller compares `err_us` of the nodes involved in a capture
  with the precision the capture plan needs (loose for LED on/off and beacon blinks,
  tight for structured lighting) and refuses to arm if any node is worse.

## 6. Blink patterns

A pattern is a list of `[on_us, off_us]` pairs for the beacon LED, starting at `at`:

```json
{"seq": 7, "op": "blink", "at": 1760712400.0,
 "pattern": {"id": "start", "steps": [[200000, 200000], [200000, 600000], [600000, 200000]]}}
```

Pattern ids are chosen by the controller and echoed in the response so that the sync
pipeline can match detected transitions to the commanded pattern. The node plays the
steps against its clock and reports each edge in `ts_actual`.

## 7. Versioning and compatibility

- `proto` is `major.minor`. Minor increments add optional fields, ops or `error` values
  that a peer may ignore. A major increment is breaking.
- Unknown fields must be ignored. Unknown `op`s are rejected with `bad_request`.
- The controller records the `proto` and `fw_version` of every node in the capture's
  event log.

## 8. Testing

- The controller repository provides a **mock node** (Python) that implements this
  protocol, including an adjustable clock error, injected failures and delays, and is
  used for the controller test suite.
- The firmware repository is expected to run the same message fixtures (JSON examples
  under `python/tests/fixtures/node_protocol/` in this repository) against its
  serialization layer.
- Hardware-in-the-loop checks (blink timing against a logic analyzer, `err_us` accuracy)
  belong to the firmware repository.

## 9. Open questions

- Whether `time_probe` suffices for post-hoc correction or a continuous offset stream
  should be published by nodes.
- MQTT QoS 1 duplicates: commands are idempotent per `seq`; nodes must de-duplicate by
  `seq` for the last N commands.
- Authentication of nodes to the broker (shared secret for now; per-node credentials if
  the capture WLAN is ever shared).
