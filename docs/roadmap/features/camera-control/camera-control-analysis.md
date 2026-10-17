# Camera Control — Options Analysis

Companion to [camera-control-brief.md](camera-control-brief.md). This document analyses the
control, timing, and file-offload options for each camera type in scope, and the
communication-architecture options between them and a central controller. The resulting
system proposal is in [camera-control-design.md](camera-control-design.md).

## Summary of findings

| Camera | Remote start/stop | Start confirmation | Mode set/verify | Wireless range path | File offload | Confidence |
|---|---|---|---|---|---|---|
| GoPro Hero13 Black | ✅ Open GoPro HTTP over COHN | ✅ status (encoding flag) | ✅ full preset control | Joins capture WLAN directly (COHN); BLE only for provisioning/wake | ✅ HTTP media API over the WLAN (parallel) or wired USB | Medium-high — official API; COHN field behaviour to be spiked (S7-S9) |
| GoPro Hero11 Black Mini | ✅ Open GoPro BLE | ✅ BLE status (encoding flag) | ✅ full preset control via BLE | BLE via per-camera bridge node | ✅ Open GoPro wired USB (HTTP) | High — official, documented API |
| Insta360 Ace Pro 2 | ✅ BLE (GPS Action Remote emulation) | ❌ none known (spike needed) | ⚠️ mode *cycle* only, no absolute set | BLE via per-camera bridge node | ✅ USB mass storage ("U-disk" mode) | Medium — reverse-engineered protocol, Ace Pro 2 untested |
| Android (Pixel, OnePlus) | ✅ ADB over WiFi key events into stock camera app | ✅ `dumpsys media.camera` / file mtime probe | ⚠️ partial (app state not scriptable; verify-on-collect) | Phone joins capture WLAN directly | ✅ `adb pull` (wireless or USB) | Medium-high — standard tooling, per-OEM quirks |
| iPhone (future) | ✅ BLE HID volume-key remote | ❌ | ❌ | BLE via bridge node (HID role) | ⚠️ manual / Files app | Low — path exists, not designed here |

Every camera type in scope can be started and stopped remotely; the differences are in
*confirmation* and *mode enforcement*, which drives several design decisions (see the
verify-on-collect principle in the design doc).

---

## 1a. GoPro Hero13 Black

The Hero13 is on the [Open GoPro](https://gopro.github.io/OpenGoPro/docs/) compatibility
list and, unlike the Hero11 generation, supports **COHN (Camera On Home Network)**: the
camera joins an existing WLAN as a client instead of acting as its own access point. This
removes the property that made WiFi unusable for multi-camera control (a controller can
join only one camera AP at a time).

With COHN the controller PC and all cameras share the capture WLAN and the PC speaks the
Open GoPro HTTP(S) API to each camera directly. The full command set is available:
presets and settings, shutter, status (encoding flag, battery, SD remaining), media list
and download. No bridge node is involved in control.

Workflow:

- **Provisioning (once per camera and WLAN).** The camera is paired over BLE and given the
  capture WLAN credentials; it then obtains COHN credentials (and a self-signed
  certificate) that the controller stores locally. This needs BLE range, so it is done
  with the camera next to the PC.
- **Capture day.** The camera joins the WLAN when powered on; the controller finds it and
  connects over HTTPS.
- **Wake.** A camera that is off is not reachable over WiFi. BLE can still wake it, but
  BLE at 10 m+ has the range problem described in §6. Options are to leave cameras
  powered on after rigging, or to wake from a nearby node. Which is acceptable depends on
  the battery and thermal behaviour measured in spike S7.
- **Offload.** All cameras are on one network, so media can be downloaded in parallel over
  WiFi without hopping between access points. Aggregate throughput on a shared 5 GHz
  channel is the limit (spike S9); wired USB remains available as a fallback.
- **Timing.** [GoPro Labs](https://gopro.github.io/labs/) precision-time QR works as for
  the Hero11 (§1b).

Unknowns to verify on hardware: how long a COHN connection stays up while recording at
mocap frame rates, battery drain and heat with WiFi active, behaviour when the camera
auto-sleeps, and reconnect time after a WLAN drop.

## 1b. GoPro Hero11 Black Mini

### Control: Open GoPro (official)

The Hero11 Mini is on the official [Open GoPro](https://gopro.github.io/OpenGoPro/docs/)
compatibility list (HERO9 through HERO12 family, including HERO11 Mini). The API is
available over three transports:

- **BLE** — full command set: load presets / preset groups, set individual settings
  (resolution, fps, lens/FOV), start/stop shutter, query status. Status includes the
  *encoding active* flag, i.e. genuine confirmation that recording is running, plus
  battery level and SD-card capacity/remaining time.
- **WiFi (camera AP mode)** — HTTP server with the same command set. **Not usable for
  multi-camera control**: each camera is its own access point and a controller can join
  only one AP at a time. The Hero11 generation does *not* support COHN (Camera On Home
  Network — camera joins an existing WLAN as a client); that arrived with HERO12.
- **Wired USB** — the camera enumerates as a network device (NCM) and exposes the same
  HTTP server, including the media list and media download endpoints. This is the most
  attractive offload path (see §5).

**Why BLE for control when USB offers the same HTTP API?** The command set is indeed
shared across transports, but control and offload happen in different phases with
different physics. During capture the cameras are rigged 10 m+ from the PC, so wired USB
control would require a USB host *at each camera* — and speaking Open GoPro over USB
means acting as a USB NCM host (the camera enumerates as a network adapter) with an IP
stack on top, which is beyond ESP32-class hardware; the per-camera node would grow into a
Pi-class SBC (≈2× cost, ~30 s boot, far higher idle power, SD-card fragility — see §6).
A USB host is also expected to supply 5 V, so the node battery would end up charging the
camera, and the open USB door + hanging cable adds a mechanical path for exactly the
mount-nudge failure the system is meant to eliminate; BLE is contactless and can
additionally wake a sleeping camera. At offload time the situation inverts: the cameras
are unrigged and next to the PC, which is a first-class USB host, so bulk transfer runs
over the wired HTTP API there (§5). Each transport is used in the phase whose physical
constraints it already satisfies. (In a *permanent* studio rig with wall power and cabled
cameras, wired-everything would become attractive and the bridge nodes would disappear —
but that is not the portable-capture scenario in the brief.)

Practical notes:

- A camera that is "off" with wireless connections enabled still advertises over BLE and
  can be woken remotely — no need to touch the camera after mounting.
- BLE requires a keep-alive; connections must be maintained or re-established by whatever
  talks to the camera.
- BLE range is nominally 10 m line-of-sight but degrades badly outdoors/through bodies.
  With camera-to-controller distances of 10 m+ and 4–10 cameras, direct PC-BLE to all
  cameras is not dependable (see §6).

### Timing: GoPro Labs

[GoPro Labs firmware](https://gopro.github.io/labs/) is available for the Hero11 Mini
([announcement](https://gopro.com/en/us/news/copy-of-gopro-labs-welcomes-hero11-black-mini-support)).
Its [precision date/time QR code](https://gopro.github.io/labs/control/precisiontime/)
sets the camera RTC to sub-second accuracy from an animated QR code — the camera clock
drifts, so this is done at the start of each capture day. With Labs time-set, the
creation timestamps in the recorded MP4s are themselves accurate to well under the
required ~1 s, independent of the control path. Labs also supports QR-triggered settings
and recording, which is a useful manual fallback but not a primary control path (requires
pointing each camera at a screen).

### Offload

Open GoPro wired USB gives an HTTP media API (list, download, delete) — scriptable,
robust, and faster than camera WiFi. Alternative: one-at-a-time WiFi AP offload
(slow, serial, requires the controller to hop networks) — rejected as primary path.

---

## 2. Insta360 Ace Pro 2

This is the problem camera: Insta360 has **no open control API**.

### Official options (all inadequate)

- **Insta360 SDK** — exists, but access is [application-gated](https://www.insta360.com/developer/home)
  (business application, approval process), historically focused on the 360° cameras, and
  redistribution terms are unclear. Not a dependable foundation for an open-source-adjacent
  tool, but worth applying in parallel — if granted it could replace the reverse-engineered
  path.
- **Insta360 mobile app** — controls one camera at a time over the camera's WiFi AP; no
  multi-cam, no external automation surface.
- **Commercial BLE remotes** — Insta360's GPS Action Remote and cheap third-party remotes
  explicitly support the Ace Pro 2. This proves the camera speaks a BLE remote protocol;
  it just isn't documented.

### Reverse-engineered BLE remote protocol

The GPS Action Remote protocol has been reverse engineered
([write-up by Patrick Chwalek](https://medium.com/@patrickchwalek/ble-control-of-insta360-cameras-7bf6894648a4),
code at [pchwalek/insta360_ble_esp32](https://github.com/pchwalek/insta360_ble_esp32);
related: [X3 ESP32 remote on Hackaday](https://hackaday.io/project/188975-insta360-x3-ble-remote-control-with-esp32),
[WiFi protocol notes](https://www.rigacci.org/wiki/doku.php/doc/appunti/hardware/insta360_one_rs_wifi_reverse_engineering)).
An ESP32 advertises as "Insta360 GPS Remote" and writes to a characteristic to emulate
button presses. Verified capabilities (on X3 / RS 1-inch):

- **Shutter** (starts/stops recording in the current mode)
- **Mode cycle** (relative only — cannot set an absolute mode)
- **Screen toggle / power off**
- **Remote wake** of a powered-off camera via manufacturer-specific advertisement data

Known limitations, which the design must absorb:

- **No status feedback** — the remote protocol is fire-and-forget. There is no known way
  to confirm over BLE that recording actually started. (Spike: check whether the Ace Pro 2
  reflects recording state in its BLE advertisement or notifies on any characteristic.)
- **Wake payloads differ per model** and possibly per unit — must be captured per camera
  (one-time sniffing step at device-registration time).
- **Ace Pro 2 specifically is untested** with this code. Third-party remote compatibility
  strongly suggests the same protocol family, but this is the single highest-risk
  assumption in the whole design → verification spike S1 in the design doc.

### Mode enforcement

Not achievable remotely (mode cycle is relative and unconfirmed). Mitigation is
procedural + verify-on-collect: the camera mode is set by hand at rigging time following
an in-app checklist, and after offload the actual files are probed (fps, resolution) and
compared against the expected camera mode from the DB.

### Offload

The Ace Pro 2 supports USB mass-storage ("U-disk") mode — files are directly readable
over USB. Simple and scriptable.

---

## 3. Android phones (Pixel, OnePlus; others best-effort)

Constraint from the brief: high-fps / high-res modes are only available in the vendor
camera app, so the *stock camera app* must do the recording — ruling out custom
camera-API apps (and third-party apps like Blackmagic Camera, which also lack the
vendor-tuned high-speed modes on most devices).

That leaves *driving the stock app by input injection*. Two viable mechanisms:

### Option A — ADB over WiFi (recommended)

Android 11+ supports wireless debugging: one-time pairing per phone, after which the
controller PC can connect over the capture WLAN (mDNS advertises the current port).
The controller can then:

- Wake the device and keep the screen on (`input keyevent WAKEUP`, `svc power stayon true`)
- Launch the stock camera straight into video mode
  (`am start -a android.media.action.VIDEO_CAMERA`)
- Start/stop recording via key events — volume key with the "volume key = shutter"
  camera setting (Pixel), or `KEYCODE_CAMERA`; in the video tab the shutter key toggles
  recording
- **Confirm state**: `dumpsys media.camera` shows whether the camera pipeline is active;
  a new growing file in `DCIM/` confirms recording; screenshots are available as a
  debugging aid
- **Offload**: `adb pull` of the new files, over the same WLAN (no cable) or USB

One control path that also provides confirmation *and* collection, with zero extra
hardware. Risks: OEM variation in key-event handling (OnePlus shutter-key behaviour must
be verified — spike S3), Developer Options must stay enabled, wireless-debugging pairing
persistence across reboots.

### Option B — BLE HID remote emulation (fallback)

Cheap "selfie remotes" are BLE HID keyboards sending volume keys; stock camera apps
treat these as shutter. A bridge node (same hardware as the GoPro/Insta360 nodes) can
present a BLE HID personality paired to the phone. Notes: BLE *headset* (AVRCP) volume
does **not** reach the camera app — it must be HID; key remapping apps (e.g. KeyMapper)
can harden the mapping per OEM. No confirmation channel, no offload — use only if ADB
proves unreliable on a given phone.

### iPhone path (nice-to-have)

Same BLE HID mechanism triggers the iOS stock camera (volume keys record video), so the
bridge-node HID personality covers iPhone start/stop with zero additional design.
Confirmation and automated offload are not solved (no ADB equivalent); files come off via
USB/Files app manually. Acceptable for the "path exists" requirement in the brief.

---

## 4. Timing and synchronisation

The brief asks for ~1 s start-time accuracy per recording; sub-frame sync continues to
come from the existing LED-based fine sync (`video_sync.py` → `SyncConfig`/`SyncPoint`).
Three layers, cheapest first:

1. **Controller-side event log** — the controller timestamps every command sent and every
   acknowledgement/confirmation received, against its own clock, into the session DB.
   For confirmed paths (GoPro, ADB) this alone bounds the start time to well under 1 s.
   For unconfirmed paths (Insta360) it records the command time, which is still within
   ~1 s of actual start in practice.
2. **Camera RTC discipline** — GoPro Labs QR time-set at the start of the day makes GoPro
   file timestamps trustworthy; Android phones NTP-sync themselves; Insta360 RTC is set
   when it last talked to its app (document as a rigging-checklist item).
3. **Sync beacon (automated LED sync)** — a controller-triggered high-brightness LED node
   that blinks a known pattern at logged times at the start/end of each capture. This
   replaces the manual "clap/LED hunt" and — because blink times are logged on the global
   clock — anchors the existing fine-sync pipeline to absolute time. Reuses the same node
   hardware as the camera bridges.

Millisecond-level *hardware* genlock (Timecode Systems / Tentacle Sync style) is neither
needed (LED sync already provides sub-frame alignment) nor possible on these cameras, and
commercial timecode boxes cost more per unit than the cameras' bridge nodes — rejected.

---

## 5. File collection

Per-camera-type offload paths (all scriptable):

| Camera | Path | Notes |
|---|---|---|
| GoPro Hero13 | COHN → Open GoPro HTTP media API over the WLAN | parallel; shares WLAN bandwidth |
| GoPro Hero11 | wired USB → Open GoPro HTTP media API | list + download + optional delete; fast |
| Insta360 | wired USB → mass storage | plain file copy |
| Android | `adb pull` over WLAN or USB | can run wirelessly during teardown |

Wireless offload of Hero11 and Insta360 cameras is rejected as the primary path: the Hero11
would require serial one-at-a-time AP hopping at ~30 MB/s, and Insta360 WiFi is
undocumented. Hero13s on the capture WLAN (COHN) are the exception and can be downloaded
in parallel over the network. A USB "collection dock" (powered hub + the controller PC) that the cameras
are plugged into after teardown is faster, simpler, and doubles as charging. Wireless
offload can be added later per-type without changing the collection model.

Collection is also the natural point for **verification**: probe each retrieved file
(ffprobe — fps, resolution, duration, creation time), compare against the expected
camera mode and capture duration, and flag mismatches. This converts the "camera was in
the wrong mode" failure from *silent ruin* to *detected the same day* even for cameras
where mode enforcement is impossible (Insta360, phones).

---

## 6. Communication architecture options

Constraints: 10 m+ camera spacing, no venue WLAN (own AP allowed), 4–10 cameras scaling
larger, BLE-only cameras in the mix.

### Option 1 — Direct PC BLE to all cameras

PC with BLE dongle talks to every camera. Rejected: BLE range at 10 m+ with bodies in
the way is unreliable; Windows multi-connection BLE stacks are flaky at 5+ concurrent
connections; phones would still need WiFi/ADB anyway.

### Option 2 — Capture WLAN, direct control where possible, bridge nodes where not (recommended)

A travel router creates a dedicated capture WLAN. Hero13s and Android phones join it and
are controlled directly. Small ESP32-based nodes are mounted
next to each BLE-only camera (velcro/clamp, LiPo powered); each node speaks BLE to its
camera over a ~0.5 m link (rock solid) and WiFi to the controller. Android phones and Hero13s join
the WLAN directly, no node needed for control. WiFi at 10–30 m outdoors with a decent router is
dependable, and the star topology scales linearly by adding nodes.

The brief explicitly blesses this: *"OK to have additional cheap equipment connected to
cameras (e.g. microcontroller to action cameras)"*. ESP32 nodes cost ~€8–12 including
battery; one firmware image with selectable *roles* (Open GoPro BLE client for the Hero11,
Insta360 remote emulation, BLE HID, plus the peripheral roles in §7) covers every case. One node can serve 2–3
co-located cameras (NimBLE supports multiple concurrent connections) — but note the
Insta360 personality *advertises as* a remote (peripheral role) while the GoPro
personality is a central; mixing both on one node is a firmware complication, so plan
1 node : 1 camera by default.

Raspberry Pi Zero 2 W was considered as the node platform instead (full Linux, Python,
easier debugging) but loses on price (~2×), boot time (~30 s vs instant), power (hours vs
days on the same battery) and SD-card fragility. ESP32 chosen; nothing in the
architecture prevents a Pi-based node speaking the same MQTT protocol if a future camera
needs more horsepower — the main thing that would force a Pi is wired USB control of the
camera (USB NCM host), which was considered and rejected for the capture phase (see §1,
"Why BLE for control").

### Option 3 — Commercial multi-cam remotes

GoPro's "The Remote" (GoPro-only, max 5 cameras, no logging), Insta360 GPS Remote
(Insta360-only), no cross-brand product exists, none covers phones and none reports
into a PC. Rejected.

---

## 7. Marker lighting and fine synchronization

Reflective-dot capture needs LEDs around each camera lens, switched on only while markers
are in use (they interfere with pose-based capture and drain batteries). Two consequences
for the architecture:

1. **Peripherals are separate from cameras.** An LED ring is not part of any camera's
   control API, so it is driven by a node regardless of how the camera itself is
   controlled. A Hero13 controlled over COHN still carries a node when lighting is needed.
   Nodes therefore have roles (`led_ring`, `beacon`, `ble_bridge`) and an optional
   association with a camera, instead of being defined as a camera bridge.
2. **The lighting state is part of the capture record.** Whether the LEDs were on, and at
   what brightness, determines whether dot detection can work. It is logged per capture
   like camera mode.

**Future structured lighting.** Patterns that vary across cameras (for example temporal
codes that identify which camera's LED a dot reflects) require LEDs in different cameras
to be synchronized to well under a millisecond. Time sources, in increasing precision:

| Method | Precision | Notes |
|---|---|---|
| SNTP to the controller over WLAN | ~5-50 ms, jittery | Adequate for beacon blinks and LED on/off. |
| Broadcast sync frame (ESP-NOW, or WiFi beacon TSF timestamps) | ~10-100 µs plausible | All nodes timestamp the same radio event; needs firmware support and validation. |
| Wired trigger or shared clock line | ~µs | Only if radio sync is not enough. |

Because the method is not yet chosen, the node protocol (see
[node-protocol.md](node-protocol.md)) makes every actuating command *scheduled*
against a node clock and reports the node's current sync method and error estimate. The
controller logic is independent of which method is used, so the better methods can be
introduced later without changing it.

## 8. Mobile control through a local web UI

The operator is often away from the PC (rigging, adjusting lighting, or at a remote site
with the PC in a bag or vehicle). A phone on the capture WLAN can reach a web UI served by
the controller PC, with no app installation and no internet.

This requires the capture service to run independently of the desktop GUI, exposing an
HTTP API and an event stream that both the Qt application and the web UI use as clients.
Design points (details in the design doc):

- The web UI covers the operational loop (device status, arm, start, stop, lighting,
  checklist). Review and editing stay in the desktop application.
- The commands can start and stop recordings and delete files, so access is gated by a
  token, and only one client at a time holds the right to send start/stop.
- A phone dropping off the WLAN must not affect a running capture.

## Sources

- [Open GoPro documentation](https://gopro.github.io/OpenGoPro/docs/) — supported cameras, BLE/WiFi/USB APIs
- [Open GoPro Python SDK](https://gopro.github.io/OpenGoPro/python_sdk/)
- [Open GoPro BLE API reference](https://gopro.github.io/OpenGoPro/ble/index.html)
- [GoPro Labs](https://gopro.github.io/labs/) and [HERO11 Black + Mini support announcement](https://gopro.com/en/us/news/copy-of-gopro-labs-welcomes-hero11-black-mini-support)
- [GoPro Labs precision date/time QR](https://gopro.github.io/labs/control/precisiontime/)
- [BLE Control of Insta360 Cameras — Patrick Chwalek](https://medium.com/@patrickchwalek/ble-control-of-insta360-cameras-7bf6894648a4) and [pchwalek/insta360_ble_esp32](https://github.com/pchwalek/insta360_ble_esp32)
- [Insta360 X3 BLE remote with ESP32 — Hackaday](https://hackaday.io/project/188975-insta360-x3-ble-remote-control-with-esp32)
- [Insta360 ONE RS WiFi protocol reverse engineering](https://www.rigacci.org/wiki/doku.php/doc/appunti/hardware/insta360_one_rs_wifi_reverse_engineering)
- [Insta360 developer portal](https://www.insta360.com/developer/home) (SDK application)
- [Insta360 Ace Pro 2 Bluetooth device compatibility](https://onlinemanual.insta360.com/acepro2/en-us/specs/compatibility/bluetooth)
- [Controlling a phone camera with a remote — Štěpán Zákostelecký](https://stepanzak.cc/blog/phone-camera-remote-control/) (BLE HID / key-event behaviour of stock camera apps)
