# Controlling cameras for motion capture

## Problem statement

Contorlling multiple cameras manulally sduring multi-view motion capture is difficult and error prone. The main issues that often happen and can ruin a capture

- Camera is in wrong mode (resolution, frame rate, FOV, distortion correction, ...)
- Some cameras do not start, or stop too early
- Camera moves when the start button is pressed, invalidating extrinsics calibration


I addition, collecting the recorded videos after capture is time consuming and error prone

## Key requirements

- Must support heterogenous set of cameras for single capture.  at least GoPro cameras (mini 11), Insta360 cameras (ace2 pro), Android phones (at leat pixel & OnePlus, others preferably). iPhone & additional camera supprot nice to have features, at least should hava path for supporting those
- Typical capture has 4-10 cameras, but should scale to larger about
- mocap requires hig frame rate 6 resolution which usually are only supported by android phone's own camera app, so capture must be doe using that
- It must be possible to start & stop all cameras participating in capture from single device (PC preferred). Must get confirmation when all cameras are started/stopped. Some differences in start times between cameras are OK, but getting the start time of each reacording at ~1s accuracy would help a lot
- Distances between cameras can be 10 m or more; strong preference fr wireless commmunication between camera locations & the central controller device. It is OK to have additional cheap equipment connected to cameras (e.g. microcontroller to action cameras that cannot be directly controlled wirelessly)
- Usually there is no available prebuilt WLAN in the capture location; fine to bring own access pint if needed
- Preferred additional feature: after stopping capture the video files are copied to the controller PC  with names that indicate the camera & capture time.
