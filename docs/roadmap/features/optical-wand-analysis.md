# Shared video observations and optical-wand timing

The headless `posetrak.analysis` modules allow independent analyses to share
native video frames and ArUco observations without requiring synchronization first.
The optical timing algorithms and target/waveform profiles live in the optional
`sync_device` package; capture identities, mode/calibration selection and session
configuration selection belong to Posetrak.

## Native video stream

`VideoFrames(path, video_id)` yields `NativeFrame` packets with native ordinal,
integer PTS/rational time base, image dimensions and PTS-segment identity. PTS
resets or dimension changes start an index segment while ordinals continue.
The index is marked complete only after EOF; interrupted passes stay partial.
Frame numbering is never reconstructed from nominal FPS.

`frame.image.bgr()` converts lazily. An even-aligned `(x, y, width, height)` region
uses a native crop before conversion. Odd regions use exact BGR slicing to avoid
YUV chroma alignment silently shifting image coordinates. Converted arrays are
read-only and borrowed: consumers must finish reading or copy them before advancing.
Display rotation requires an explicit image-domain/calibration policy.

## Shared detection

`SharedAruco` uses the existing `app.setup.fiducial_markers.ArucoDetector`. It retains
dictionary/ID/corners, native frame identity, detector/version fingerprint and
image-coordinate convention. Duplicate IDs in one frame are excluded from reuse.
`save`/`load` persist marker observations with explicit source/mode provenance;
stale sources or detector settings are rejected. No image buffers are stored.

`run_shared_pass` accepts synchronous `Consumer` callbacks with independent stride
and marker requirements. It decodes once and detects markers at most once per
requested frame. Consumers get the same observation tuple. No queued frames are
retained; a cancellation callback is checked between frames. Slow consumers apply
backpressure rather than causing an unbounded image queue.

For example, a marker-body consumer can retain raw corner observations while a
stationary LED locator accumulates a temporal maximum from the same frames.
Metric calibration/undistortion belongs to the geometry consumer, not the shared
native-pixel observation cache.

## Optical adapter

`analyze_optical_wand(request, profile, ...)` supplies Posetrak-owned video and
marker inputs to the timing package and returns a versioned result plus counters.
`native_frames` and `marker_provider` may be injected from a host analysis job;
they need not reopen a decoder or repeat ArUco detection. The file CLI uses the
same API:

```powershell
python -m posetrak.analysis.optical_wand `
  --request capture-analysis-request.json --profile t-wand-profile.json `
  --output analysis-result.json --cache analysis-cache --workers 1
```

Make the local sync-device package available in the interpreter environment for
this optional adapter. Other analysis modules do not import it. No Bluetooth or
capture daemon is imported by the file analysis path. Worker count is bounded,
and defaults to one; native 4K decoding/detection can require several GiB per process.

The request supplies stable capture/video/camera IDs, ordered chapter records,
mode/calibration metadata, target/waveform/run identity and analysis/export policy.
Use registered file modes and calibration IDs, not merely the settings requested
when arming a camera. Wait for complete offload/registration before analysis.

Timing results retain coarse/fine/ambiguous/missing/failed outcomes, native PTS
segments, evidence intervals, selected bank/image rows and unknown uncertainty.
Bank fallback never averages independently biased clocks: reference changes are
explicit and remain uncalibrated. Complete optical loss has no observed support.
Long-gap prediction and calibrated timing covariance require separate qualification.

No session writes or configuration selection occur in this adapter. The capture
job/UI must later choose whether a quality-qualified new config may be attached;
it must not silently replace an existing selected config. The legacy sync importer
still needs explicit chapter assembly for multiple files from one camera instance.

## Checks before broader integration

Tests cover native VFR/nonzero PTS, explicit reset domains, exact crop coordinates,
read-only images, one decode/detection pass for multiple consumers, cancellation,
source-qualified marker persistence and host-owned input injection. A full recording
comparison verified native PTS/ordinals/LED intensities and fitted clock equality
between the standalone and Posetrak adapters. Three-camera shared-input prefix
benchmarks halved decode/detection counts; full-job memory and packaging still need
profiling before choosing production worker isolation and concurrency budgets.
