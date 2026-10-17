# Controlling cameras for motion capture

## Problem statement

Controlling multiple cameras manually during multi-view motion capture is difficult and error prone. The main issues that often happen and can ruin a capture:

- Camera is in wrong mode (resolution, frame rate, FOV, distortion correction, ...)
- Some cameras do not start, or stop too early
- Camera moves when the start button is pressed, invalidating extrinsics calibration

In addition, collecting the recorded videos after capture is time consuming and error prone.

Marker-based capture adds a further control task: the reflective dots are lit by LEDs
around each camera lens, and these must be on during marker capture and off otherwise.

## Camera fleet and priority

| Priority | Camera | Role |
|---|---|---|
| 1 | GoPro Hero13 Black | Bulk of the fleet. Can join a WiFi access point, so many can be controlled simultaneously over one network. |
| 2 | GoPro Hero11 Black Mini, Insta360 Ace Pro 2 | Still in use; BLE-only control. |
| 3 | Android phones (Pixel, OnePlus; others best effort) | Stock camera app is required for high-fps / high-res modes. |
| 4 | iPhone and other cameras | Nice to have; there should be a path, not a full design. |

## Key requirements

- Must support a heterogeneous set of cameras in a single capture, in the priority order above.
- Typical capture has 4-10 cameras, but should scale to larger numbers.
- Mocap requires high frame rate and resolution, which on phones are usually only supported by the phone's own camera app, so capture must be done using that.
- It must be possible to start & stop all cameras participating in a capture from a single device. Must get confirmation when all cameras are started/stopped. Some differences in start times between cameras are OK, but knowing the start time of each recording to ~1 s accuracy would help a lot.
- Distances between cameras can be 10 m or more; strong preference for wireless communication between camera locations and the central controller. It is OK to have additional cheap equipment connected to cameras (e.g. a microcontroller for cameras that cannot be directly controlled wirelessly).
- Usually there is no prebuilt WLAN in the capture location; fine to bring an own access point if needed.
- Preferred additional feature: after stopping capture the video files are copied to the controller PC with names that indicate the camera and capture time.
- **Remote LED control.** For marker-based capture the LEDs around each camera lens must be switchable remotely, and the state used for a capture must be recorded with it.
- **Mobile control.** At remote locations the operator must be able to run the capture from a mobile phone. The Posetrak software runs on a PC, so it must provide a local web UI for capture control.

## Direction to keep open (not required initially)

Structured lighting with the camera LEDs, which needs sub-millisecond synchronization of
the LEDs across cameras. This will require a microcontroller node at every camera, even
those that can be controlled directly from the PC. The system architecture must not
preclude this: peripherals are first-class devices separate from cameras, and their
commands are scheduled against a synchronized clock rather than executed on arrival.

## Related hardware developed separately

The sync beacon and the camera-mounted nodes are ESP32 based. Their firmware is developed
in a separate repository, in Rust. This repository defines the wire protocol they share
with the controller ([node-protocol.md](node-protocol.md)).
