# Camera control, Phase 1: implementation plan

Plan for Phase 1 of [camera-control-design.md](camera-control-design.md) (§7): the capture
daemon, the Hero13 driver over COHN, the control API with a first mobile web UI, and the
capture event log. Option rationale is in
[camera-control-analysis.md](camera-control-analysis.md); the node protocol is not touched
in this phase.

## 1. Deliverable

From a phone browser on the capture WLAN, an operator can arm, start and stop a group of
Hero13 cameras. Before the take the presets are pushed and read back; after START every
camera is confirmed recording (or flagged); STOP is confirmed the same way. Every command
and confirmation is timestamped into the session DB. A camera that fails to confirm, or
drops off the WLAN mid-take, raises an alert in the UI. The desktop app can start the daemon
and show the URL/QR for the web UI.

### Networks

The cameras must work on more than one WLAN: the capture access point used on location
(cameras, laptop, phone, access point only) and the home network used for development and
testing. The daemon therefore has the notion of a *network profile* (name, SSID, address
resolution method, optional DHCP reservations), selected automatically from the PC's current
WLAN or explicitly by the operator. Whether a Hero13 can hold COHN credentials for several
SSIDs and pick the available one on power-up is unverified and is the first question WP0
answers (see below); the design degrades gracefully if it cannot (re-provisioning per
network, from the PC, takes a few minutes per camera).

### Acceptance test (on hardware)

With at least 4 Hero13s on a travel router (the on-location network), a PC, and a phone:

1. Provision all cameras once from the PC.
2. Power-cycle the cameras; they rejoin the WLAN and appear in the web UI without PC action
   other than starting the daemon.
3. Arm with a mocap profile (resolution / fps / lens); a deliberately mis-set camera is
   corrected by the preset push or blocked at arm with a clear reason.
4. START from the phone: all cameras report `RECORDING` within 5 s. `capture_events` has
   command, ack and confirm timestamps for each.
5. Switch off one camera's WiFi mid-take: alert within 10 s; the other cameras keep
   recording and STOP still works.
6. Close and reopen the phone browser mid-take: state is intact, capture unaffected.
7. A second phone is view-only until it takes the control lease.
8. Repeat steps 2–4 on the home network, then power-cycle the cameras with both networks
   in range and with only one in range: each time the cameras join the network the PC is on
   (or the one that is available) without per-camera action.

### Not in Phase 1

Collection and verify-on-collect (Phase 2), MQTT broker and nodes (Phase 3), lighting
(Phase 4), Hero11 / Insta360 / Android / iPhone drivers, Hero13 offload, Labs time-set
screen, a full Qt capture panel. The device abstraction must not preclude them, but no code
is written for them.

## 2. Work packages

Sizes are relative (S ≈ a day, M ≈ several days, L ≈ a week or more). Each package ends in a
cohesive commit (or a few), with unit tests passing and docs updated.

### WP0: Hardware probe and spikes (M, needs 2+ Hero13s)

Throwaway script `python/tools/prototypes/gopro_cohn_probe.py` that exercises the Open GoPro
SDK against real cameras. Answers, before any design is frozen in code:

- **Multiple networks.** Can a camera keep COHN credentials for two or more SSIDs (home
  network and the capture access point) and join whichever is in range when powered on? If
  both are in range, which one wins and can that be controlled? Does switching networks
  require re-provisioning, and how long does that take? What happens to the stored COHN
  credentials and certificate per network?
- **S8 / discovery.** After BLE provisioning, how does the PC find a camera's address on the
  WLAN (BLE-reported IP, mDNS, router DHCP reservation by MAC)? Expect to adopt DHCP
  reservations plus re-resolution as the robust option.
- **COHN API surface.** Provisioning steps, credential and certificate handling, command
  latency over HTTPS, which preset and setting IDs the mocap profile needs (S6, Hero13).
- **S7 soak.** Two or more cameras recording at the target preset for an hour or more with
  COHN connected: link drops, reconnect time, temperature warnings, battery drain, behaviour
  when the camera would normally auto-sleep.
- **Wake policy.** Can the PC wake a powered-off Hero13 over BLE at realistic distance, or
  is "leave cameras on after rigging" the Phase 1 policy?
- **SDK fit.** Does the Open GoPro Python SDK install and work on Python 3.13 / Windows, and
  does it cover COHN, or do we speak the HTTP API directly?

WP0 can start on the home network now. The on-location network is exercised once a travel
router is available (a few days); a phone hotspot is an acceptable stand-in for the
single-AP behaviour meanwhile. A home network may have client isolation or a mesh that
affects discovery; the probe records this.

Output: findings appended to the analysis doc and a "Phase 1 decisions" note in the design
doc (discovery method, wake policy, SDK vs direct HTTP). WP3 starts from these; WP1, WP2 and
WP6 do not depend on them and can proceed in parallel.

### WP1: Package skeleton and tooling (S)

- `python/pipeline/capture/` package: `service.py` (stub), `api.py` (stub), `devices.py`,
  `config.py`, `__main__.py`; console script `posetrak-capture` in `pyproject.toml`.
- New dependency group `capture` (HTTP/WebSocket server, Open GoPro SDK or HTTP client, BLE
  library for provisioning, mDNS, test helpers such as `pytest-asyncio`). Dependencies for
  the field must work without internet at runtime.
- Daemon configuration in a TOML file in the user config directory: session DB path,
  listen address and port, access token, capture profile, camera list, and the network
  profiles (see Networks above). Command-line
  overrides for all of it.
- Logging to a file next to the config, with per-device context, so a failed take can be
  reconstructed.

Tests: config loading and validation; `posetrak-capture --help` smoke test.

### WP2: Schema and DB access (S–M)

- Session migration (next number in `python/posetrak/db/sql/migrations/`, plus
  `session_schema.sql`): `capture_events` as in the design doc, with `PRIMARY KEY
  (capture_id, camera_instance_id)`.
- Registry migration and `registry_schema.sql`: `control_endpoints` with `method` including
  `'gopro_cohn'` and `gopro_serial`. `peripherals` is not created until Phase 3.
- Accessors in `python/posetrak/db/` (new `capture_control.py`): create a capture, upsert
  and read `capture_events`, read/write `control_endpoints`. Follow the existing patterns in
  `db.py` for migration application and connection setup.
- Update `docs/data-model-and-storage.md`.

Open point: the capture record is created at START (so the event log has a parent row), not
after the fact by the capture wizard. The wizard's "add videos" step must tolerate an
existing capture with events but no videos yet. Verify against the current wizard in
WP2 and note any needed change.

Tests: migration applies to an existing DB; accessors round-trip; the idempotent upsert of an
event row.

### WP3: Device abstraction and Hero13 driver (L; depends on WP0)

- `CameraDriver` protocol in `devices.py`: `connect()`, `arm(profile)`, `start()`, `stop()`,
  `status()` plus an async stream of status changes, `close()`. Drivers return structured
  results (ok / fail, command-sent / executed / confirmed timestamps) rather than raising for
  expected failures.
- `ManualDriver`: no automation, operator confirms in the UI. Implemented now because it is
  the fallback for any camera that is not yet automated.
- `gopro_cohn.py`:
  - provisioning flow (BLE pairing, join a chosen network profile, obtain and store COHN
    credentials) exposed as `posetrak-capture provision --network <name>`, repeatable per
    network and per camera; stored credentials are keyed by camera and network;
  - connect/reconnect with backoff, address resolution per the WP0 decision and the
    active network profile; if the camera appears on a network other than the PC's it is
    reported as "on another network", not as offline;
  - `arm`: push the profile's settings, read them back, compare, report differences;
  - `start` / `stop`: shutter command, then poll status until the encoding flag matches
    (confirmation), with a timeout;
  - status: battery, SD remaining, encoding, temperature warnings, link state;
  - credentials in a local keystore under the user config directory, never in a session DB.
- Capture profile: a TOML description (resolution, fps, lens, anything else the WP0
  settings matrix requires) and the `camera_modes` row it should correspond to, so Phase 2
  verification can compare files against the same source of truth.

Tests: against the fake camera from WP6 (see below); real-camera runs are manual and
covered by the acceptance test.

### WP4: Orchestration and state machine (M)

- `CaptureController` in `service.py`: per-camera state machine from the design doc
  (OFFLINE … STOPPED; COLLECTED is Phase 2), fault states with reasons, fallback to
  `ManualDriver`.
- `arm()`: parallel pre-flight over all devices, checks (link, battery, storage, profile
  read-back), operator checklist items, aggregate result.
- `start_capture()` / `stop_capture()`: create the capture row, fan out in parallel, collect
  confirmations with a timeout (default 5 s), write `capture_events`, return a per-camera
  result; a single failing camera never aborts the others.
- Live monitoring during a take: status stream feeds alerts (link lost, battery low, SD
  nearly full, overheating).
- A single asyncio event bus carries state changes to the API layer.

Tests (with fake cameras): all-success; one camera rejecting START; one confirming late; one
dropping mid-take; STOP with a camera already stopped; operator override; two overlapping
START requests rejected.

### WP5: API, auth and control lease (M)

- HTTP + WebSocket server per design §3.4: `GET /api/state`, `WebSocket /api/events`,
  `POST /api/capture/{arm,start,stop}`, `POST /api/checklist/{id}`, `POST /api/lease`.
  Lighting endpoints arrive in Phase 4.
- Access token required on every request, supplied through the URL fragment/QR on first load
  and kept in the browser; token generated on first run, stored in the daemon config.
- Control lease: one holder at a time; commands from non-holders get a clear refusal;
  explicit takeover is logged; a lease with a dead client expires after a timeout.
- Startup detects the PC's current WLAN, selects the matching network profile and prints
  the URL and QR code; the base URL uses the PC's WLAN address (mDNS name as
  a convenience where the phone supports it).
- Windows: the first run needs an inbound firewall rule; document it and offer a helper
  command rather than relying on the OS prompt.

Tests: auth failures; lease acquisition, refusal, takeover and expiry; event stream delivers
state changes in order; snapshot after reconnect equals live state.

### WP6: Fake Open GoPro camera and test harness (M; start early)

A small HTTP(S) server implementing the subset of the Open GoPro API the driver uses, with
programmable behaviour: settings acceptance, encoding flag delay, command failures, link
drops, slow responses. Test fixtures spin up N fakes in-process. This is what makes WP3–WP5
testable without hardware and is kept in `python/tests/capture/`.

### WP7: Mobile web UI (M)

- Static single-page app under `python/pipeline/capture/web/`, served by the daemon; no
  build step; all JavaScript and CSS vendored in the repo so it works with no internet.
- Screens: device wall (tiles with link, battery, storage, mode-verified, state,
  confidence), arm/start/stop controls with the confirmation wall, per-camera failure
  banners and manual override, checklist ticks, lease indicator and takeover, alerts.
- Layout designed for a phone held in one hand outdoors: large controls, high contrast,
  START requires a confirm gesture to avoid accidental takes, wake lock while a capture is
  running, reconnect automatically after a network blip.

Tests: the API is exercised in WP5; UI checks are a manual script plus a few headless-browser
smoke tests if a harness is cheap to add.

### WP8: Desktop integration (S)

Minimal: a "Capture control" action in the main window that starts the daemon (or detects a
running one), shows the web UI URL and a QR code, and opens the page in the system browser.
Hero13 provisioning stays on the command line in this phase. No new Qt widgets beyond that
dialog; the full Qt panel and in-app device wall are deferred.

### WP9: Docs, hardware validation, status (S)

- Run the acceptance test on real cameras and record the results in the status document.
- `docs/` how-to for field setup: router, provisioning, starting the daemon, joining from a
  phone, firewall, troubleshooting.
- Update the design doc with the decisions from WP0, `status.md` (state and open questions),
  and regenerate the roadmap index.

## 3. Order and dependencies

```
WP0 (hardware) ─────────────┐
WP1 ── WP2 ─────────────────┤
WP6 (fake camera) ──────────┼── WP3 ── WP4 ── WP5 ── WP7 ── WP8 ── WP9
```

WP1, WP2 and WP6 can start immediately and in parallel with WP0. WP3 waits for the WP0
decisions. The web UI (WP7) can be developed against the fake cameras as soon as WP5 exists.

## 4. Testing strategy

- Unit and integration tests with fake cameras run in CI like the rest of `pytest
  python/tests/`; they cover the orchestration, API and DB behaviour, including the failure
  cases listed under WP4.
- Hardware behaviour (COHN stability, wake, throughput of status polling alongside
  recording) is checked in WP0 and the WP9 acceptance run, not in CI.
- The fake camera is derived from the same API subset the real driver uses, so API drift is
  caught when the driver is updated against the real SDK.

## 5. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| COHN link unreliable while recording at mocap presets | Missed or unconfirmed starts | WP0 soak first; reconnect/backoff in the driver; fall back to "leave cameras on, BLE as backup" |
| Open GoPro SDK lacks COHN support or fails on Python 3.13 / Windows | Driver rework | WP0 checks; HTTP API is documented, so a thin direct client is the fallback |
| Camera cannot hold or choose between several networks | Needs re-provisioning when moving between home and location | WP0 answers it; network profiles and per-network credentials make re-provisioning a routine command; a dedicated capture SSID that is never available at home avoids ambiguity |
| Phone cannot resolve the `.local` name (common on Android) | Operator can't reach the UI | QR carries the IP address; name is a convenience |
| Windows firewall / network profile blocks the daemon | UI unreachable | Documented setup and a helper command; test on a clean machine in WP9 |
| Concurrent SQLite access by the daemon and the Qt app | Locked DB | Daemon is the only writer during a capture; WAL mode; short transactions |
| Capture row created before videos exist breaks the capture wizard | Broken flow | Checked in WP2 |
| Scope creep toward a full Qt panel | Delay | Explicitly deferred (WP8) |

## 6. Decisions

Settled in review:

- **Web framework:** FastAPI with uvicorn.
- **Process model:** the daemon is a separate process; the desktop app starts or attaches
  to it.
- **COHN credentials:** a file in the user config directory, keyed by camera and network,
  never in the session DB.
- **Web UI:** Preact + htm vendored as plain ES modules, no build step.
- **Provisioning:** CLI only in Phase 1.
- **Hardware for WP0:** Hero13s, a PC with Bluetooth and the home network are available
  now; a travel router follows in a few days.

Open, to be answered by WP0 and recorded in the design doc:

- Multi-SSID behaviour of COHN (see WP0 and Networks).
- Address discovery method and wake policy.
