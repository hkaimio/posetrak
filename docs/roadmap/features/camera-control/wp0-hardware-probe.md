# Camera control: hardware probe runbook

How to run `python/tools/prototypes/gopro_cohn_probe.py` against real Hero13 cameras to
answer the open questions of the [Phase 1 plan](phase1-implementation-plan.md) (WP0). Record
the results in the table at the end and then fold them into the analysis and design docs.

## Setup (once)

A scratch environment keeps the SDK out of the main project environment:

```bash
uv venv probe-venv --python 3.13
uv pip install --python probe-venv/Scripts/python.exe open-gopro bleak
```

Run the commands below as `probe-venv/Scripts/python.exe python/tools/prototypes/gopro_cohn_probe.py <command>`
(abbreviated as `probe` below). Work from a scratch directory: the script writes its files
to `./gopro_probe/` (override with `--db-dir`). That directory contains camera passwords and
must not be committed.

Prerequisites:

- The cameras' wireless connections are enabled and the cameras are powered on, charged and
  have an SD card. A Hero13 that is powered on advertises over BLE.
- The PC has Bluetooth on and is connected to the WiFi network the cameras will join. For the
  home network use the 2.4 GHz band or a combined SSID.
- Cameras are identified by the last four digits of the serial number, as in the BLE name
  `GoPro 1234`. `probe scan` lists them.
- Phone apps (GoPro Quik) must not be connected to the cameras; they hold the BLE link.

## 0. Pair each camera with the PC (once per camera)

`scan` only needs the camera to advertise, but every command after it needs a bonded BLE
connection, so each camera must be paired with the PC once:

1. On the camera: Preferences → Connections → Connect Device → GoPro Quik App. The camera
   now waits in pairing mode.
2. Run `probe networks 1234` for that camera. Windows shows a pairing prompt (usually no PIN)
   the first time the probe connects; accept it.
3. From then on the camera can be in its normal state.

If pairing or the first command fails: close GoPro Quik and switch off Bluetooth on phones that
have been paired with the camera (only one central should hold it); remove "GoPro 1234" from
Windows Settings → Bluetooth & devices; on the camera choose Preferences → Connections →
Reset Connections; then repeat from step 1.

On Windows the SDK refuses to use the WiFi adapter unless the system language is US English.
The probe only uses BLE, so it disables that check
(`_wifi_adapter.ensure_us_english = lambda: None` near the top of the script).

## 1. First contact (one camera)

```bash
probe scan                      # cameras advertising, with signal strength
probe networks 1234             # COHN status, and WiFi networks the camera sees,
                                #   flags CONFIGURED = already provisioned
probe provision 1234 --ssid HomeNet --password '...'
probe state 1234 --ssid HomeNet --repeat 10 --interval 1
```

Note: whether `provision` worked (COHN `state` connected, an IP address printed, credentials
saved), how long each step took, and the median state latency.

Quit the BLE connection to this camera from other apps first if `networks` cannot connect.

## 2. Several cameras

Provision each camera as in step 1, then:

```bash
probe state 1234 5678 9012 3456 --ssid HomeNet --repeat 10
probe record 1234 5678 9012 3456 --ssid HomeNet --seconds 10
probe dump 1234 --ssid HomeNet        # writes gopro_probe/dump_1234.json
```

`record` prints, per camera, the command latency, the time from sending START to the camera
reporting that it is encoding, and the spread between cameras. Run it three to five times.
`dump` gives the settings and presets for the mocap profile; before dumping, set the camera
manually to the preset you intend to use for mocap (resolution, fps, lens, and so on).

## 3. Power-cycle and rediscovery

Power the cameras off and on (button). Without touching the PC, run:

```bash
probe state 1234 5678 9012 3456 --ssid HomeNet
```

Note: how long it takes for each camera to rejoin the network and answer; whether the IP
address stayed the same (compare with `gopro_probe/<id>.json`); whether any camera needs a
BLE nudge. Also record whether the router's DHCP lease list shows the cameras and whether the
home network has client isolation (the PC cannot reach the camera at all).

## 4. Wake from off

```bash
probe wake 1234 --ssid HomeNet --power-down       # the script powers the camera down
probe wake 1234 --ssid HomeNet                    # or: power it off yourself first
```

Note: whether the BLE connect wakes the camera, the time from the attempt until COHN answers,
and repeat it from realistic distances (5 m, 10 m, with a body in between, outdoors if
possible). `probe scan --seconds 10` at each distance shows the signal strength.

## 5. Soak

Two or more cameras on the mocap preset, charged, with SD cards:

```bash
probe soak 1234 5678 --ssid HomeNet --minutes 60 --interval 5
```

The script starts recording, polls state, writes a CSV to `gopro_probe/soak_<time>.csv`, stops
the recording and prints a summary (failed polls, latency, link drops and their length). Watch
for `NOT ENCODING` (a camera stopped by itself) and `SYSTEM HOT` messages. Afterwards note the
battery percentage drop and whether the camera is warm to the touch. Repeat once with the
cameras in sun or a warm room if outdoor shoots matter, and once with a longer (2 h) run if the
first one is clean.

## 6. Multiple networks (needs a second network)

With a second SSID available (the travel router, or a phone hotspot):

```bash
probe networks 1234                                 # both SSIDs should be flagged CONFIGURED
                                                    #   once provisioned on each
probe provision 1234 --ssid TravelAP --password '...'
probe state 1234 --ssid HomeNet                      # does the first network's login still work?
probe join 1234 --ssid HomeNet                       # switch back by BLE
probe join 1234 --ssid TravelAP
```

Then power-cycle the camera in three situations and see which network it joins:

1. only HomeNet in range
2. only TravelAP in range
3. both in range

Note: whether provisioning the second network invalidated the first network's COHN
credentials (`state` on the first SSID fails after step 2), whether `join` reports matching
stored credentials, how the camera chooses with both in range, and how long switching takes.
If credentials are invalidated, record how long re-provisioning takes per camera.

## Results

Fill this in. Paste the corresponding probe output next to the results where it helps.

| Question | Result |
|---|---|
| SDK works on Windows / Python 3.13 | |
| Provisioning time per camera (BLE, join, COHN) | |
| How cameras are found on the WLAN (IP stable? DHCP reservation needed?) | |
| Command latency (state, shutter) | |
| START → confirmed encoding, 4 cameras, spread | |
| Rejoin time after power-cycle | |
| Wake from off over BLE; range limit | |
| Soak: link drops, hot warnings, battery drain per hour | |
| Camera keeps credentials for two networks? | |
| Which network wins with both in range? | |
| Settings required for the mocap profile exist and read back | |
| Anything else surprising | |

## Findings so far (Hero13, Windows 11, Python 3.13)

- **Pairing is required.** BLE commands only work after the camera has been paired with
  the PC (step 0). Once paired, `networks` lists the nearby networks; a camera that has not
  been provisioned yet shows none of them as configured.
- **COHN provisioning works.** `provision` joins the camera to the chosen network and
  returns its IP address and credentials.
- **Overlapping VPN routes hide the camera.** If a VPN tunnel (WireGuard) routes the same
  subnet as the capture network, the PC sends traffic for the camera into the tunnel and every
  request times out although the camera is fine. Check with
  `Find-NetRoute -RemoteIPAddress <camera ip>`: the interface must be the WiFi adapter. The
  controller should warn about this at startup. On location, use an access-point subnet that
  does not overlap any VPN route.
- **Do not use `requests` (urllib3) to talk to the camera.** With Python's `requests` the camera
  accepts the connection and then closes it without a response after about 2 s, on almost
  every attempt. `curl`, `http.client` and `httpx` get the answer in 0.1-0.3 s on every attempt
  with the same credentials, so the camera is not at fault. The cause on the urllib3 side was
  not found (TLS options, headers, proxy settings and keep-alive were ruled out). The
  controller uses `httpx`, with an SSL context pinned to the camera's certificate:
  `create_default_context(cadata=<cert>)`, `check_hostname = False`, and the
  `VERIFY_X509_STRICT` and `VERIFY_X509_PARTIAL_CHAIN` flags cleared (the certificate has no
  authority key identifier and does not match the IP address; Python 3.13 rejects it in strict
  mode).
- **Latency.** Ping to the camera varies between 20 and 320 ms (WiFi power saving on the
  camera); state requests take 150-250 ms, and the first request after a pause up to 700 ms.
- **Idle battery drain.** One camera lost 5 percentage points in about 11 minutes while idle on
  WiFi with BLE closed (90% → 85%). Compare with a camera that is on without WiFi before
  deciding how long cameras can be left on.
- **The SDK drops the PC's WiFi connection.** On Windows, closing the SDK's camera object
  runs `netsh wlan disconnect` on the PC's WiFi adapter, even for BLE-only use (it always builds
  a host WiFi controller). Every BLE command, `provision` included, therefore ended with the PC
  disconnected from its WLAN. The probe passes the SDK a no-op WiFi adapter; the production
  driver must do the same, or avoid the SDK's camera object for BLE and use `bleak` directly for
  the few provisioning commands.
- **Start and stop timing (three Hero13s, one WLAN, commands fanned out together).**
  START: the shutter request returns after 1.1-1.5 s (the camera answers once it is starting to
  encode) and the encoding flag is up at 1.3-1.5 s; the three cameras confirm within 0.22 s of each
  other. STOP: the request returns after 0.24-0.31 s, but the encoding flag only drops 1.65-1.8 s
  after the command (the camera finalizes the file); spread 0.14 s. The 5 s confirmation timeout
  in the design has plenty of margin.
- **Status polling** of three cameras every second: 150-250 ms per request, no failures in 30
  polls. Battery on one camera was 12%: cameras must be charged before an acceptance run.

## Soak results, first run (2026-10-17, two Hero13s, home WLAN, 5 s polling)

Camera 4713 recorded video; camera 4243 turned out to be in timelapse mode, so only its link
statistics are representative.

| | 4713 (video) | 4243 (timelapse) |
|---|---|---|
| Polls / failed | 284 ok until shutdown | 385 / 0 over 38 min |
| State request latency | median 219 ms, p95 290 ms, max 836 ms | median 215 ms, p95 400 ms, max 1.0 s |
| SD usage | about 2.9 GB per 5 min (9.4 MB/s) | about 35 MB per 5 min |
| Battery | 64% → 35% in 20 min (about 85% per hour) | 75% → 42% in 35 min (about 56% per hour) |
| System hot flag | first at 19.9 min | never |
| Outcome | link lost at 24.8 min: thermal shutdown | still running at 38 min |

- **The COHN link itself is stable.** No failed poll on the surviving camera in 38 minutes, with
  no reconnects; the only link loss was the camera shutting down.
- **Thermal limit.** The video camera raised the system-hot flag after about 20 minutes of
  recording and powered off about 5 minutes later, with WiFi/COHN active and polling. The
  camera had also been on for roughly an hour beforehand (state polling, short recordings), so
  the starting temperature was not cold. The preset in use was not recorded in this run (the
  probe now saves each camera's settings before a soak starts).
- **Timelapse mode was started silently.** `record` and `soak` send the shutter command to
  whatever mode the camera is in, and a camera in timelapse mode "records" without complaint.
  This is exactly the wrong-mode failure the system is meant to catch: the Hero13 driver must
  push the capture preset and read the mode back at arm time, and refuse to arm on a mismatch.

Consequences for the design:

- The system-hot flag is the only early warning (about 5 minutes) and must raise a high-priority
  alert the moment it appears. The controller already alerts on it.
- Long continuous takes are limited by heat, not by the link. Takes should be as short as the
  action allows, with the cameras stopped between takes to cool; cameras should not be left
  recording while waiting.
- Open questions for the next run: which preset runs cool enough for the longest planned take,
  whether the camera runs cooler with WiFi off (start by hand on two identical cameras, one
  with COHN polling and one without), and how long a camera needs to cool between takes.
