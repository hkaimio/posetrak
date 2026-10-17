# Detection pipeline: one pass, pluggable box source, SOMA-77 as a native pose model

## 1. Problem

Person pose detection has three code paths today, each with its own frame loop:

| Path | Entry | Boxes | Frame loop |
|---|---|---|---|
| YOLOX + RTMPose | `posetrak detect run`, `DetectionPipeline` (`posetrak/detection/pipeline.py`) | YOLOX with its tracker, per frame | PyAV, single-threaded decode, one camera after another |
| Segmentation | `detect run --segmentation`, GUI Cutie panel; `PoseWorker` (`app/pose/pose_worker.py`) | Tight box of each mask label | `cv2.VideoCapture`, per-frame PNG decode of the mask, resize to native size |
| SOMA-77 prototype | `python/tools/prototypes/soma77_extract.py` | Copied from an existing detection run | PyAV, batches of eight frames |

The SOMA model therefore needs a source run, which for YOLOX footage means a full RTMPose pass
only to get boxes (about 35 minutes per trial in the spike). The hand refinement pass decodes the
video a second time (`HandRefinementPipeline`, `iter_frames`).

## 2. Design

One `detect run` command and one frame loop, with two plug points:

```
BoxSource ──boxes per (camera, frame)──▶ PoseEstimator ──keypoints (+ candidates)──▶ writer
   ├─ YoloxBoxSource       (detector + its tracker; track ids are fragments)
   ├─ MaskBoxSource        (seg_quality_run + revision; track id = mask label = subject label)
   └─ RunBoxSource         (person_detections of an existing run, with its track assignments)
```

- `BoxSource.boxes(camera, video_frame, image) -> list[PersonDetection]`. The mask source also
  exposes the decoded mask of the frame so the estimator can gate heatmaps with it.
- `PoseEstimator` (existing protocol in `backends.py`) gets a batched form,
  `estimate_batch(items) -> list[list[PoseResult]]`, over crops from several frames and cameras.
  RTMPose and SOMA-77 both implement it; the single-frame `estimate` stays for callers that need it.
- SOMA-77 (`backends_soma77.py`) is registered as estimator `soma77-vitpose-h`, with the options the
  spike validated: flip test, crop enlargement 1.6, parabolic refinement, decoding at cell centres
  (index + 0.5), heatmap gating by the person's mask when the box source provides masks. Optional
  candidate output (several maxima per keypoint) is specified in
  [keypoint-candidates](../keypoint-candidates/keypoint-candidates-design.md).
- The run records its inputs: `detection_runs.seg_quality_run_id` and `seg_revision` when masks
  provided boxes or gated heatmaps, `source_detection_run_id` when boxes were copied from a run, and
  every estimator option in `config_json` (see the architecture and schema summary).
- A mask-sourced run is finalised automatically, as today. A YOLOX-sourced run goes through track
  to person assignment ([track-person-assignment](../track-person-assignment/track-person-assignment-design.md)).
- Hand refinement runs inside the same loop on the frame already decoded, instead of a second pass
  ([hand refinement without segmentation](../hand-detection-refinement/hand-refinement-without-segmentation-design.md)).
- `PoseWorker` and the GUI dialogs call the same loop; `pose_worker.py`'s own loop is removed.

### Feeding the GPU

The loop is split into three stages connected by bounded queues:

1. **Decode** — one producer per camera video (threads, or processes if decode holds the GIL in
   practice), yielding frames of the range. Each producer also asks the box source for boxes, so
   mask decoding runs here too.
2. **Infer** — one consumer that collects crops from all producers into batches of a fixed size
   (16 crops for SOMA, the size measured at 12 ms per crop) and runs the model.
3. **Write** — one thread that encodes preview crops and writes rows in batches.

The GPU then waits only when all producers together cannot keep up, which the profiling below
shows or rules out.

Decoding is single-threaded today on purpose: closing a container with threaded decoding before
the end of the file can hang in FFmpeg (`frame_source.py`). Parallelism comes from decoding several
cameras at once, not from threaded decoding of one; hardware decoding is an option to measure.

### Variable frame rate

The pose model need not run on every frame when the persons are far apart. Per global time `t`,
the loop decides from the boxes of all cameras whether `t` is a **full-rate** instant:

- any two persons' enlarged boxes overlap in any camera, or
- any person's box moves faster than a threshold (box heights per second), or
- the instant lies within a margin of a full-rate instant.

Other instants are processed at every `N`th frame (default 1, i.e. off). Every camera skips the
same instants. The box source still runs on every frame (YOLOX tracking needs it); only the pose
model is skipped. Skipped frames have no rows; the tracker predicts through them. The same rule can
later drive the segmentation propagation rate, which is part of the segmentation feature.

This only pays when the pose model, not decoding, dominates the time; it is built after the
profiling and only if it does.

## 3. Profiling plan

Where the time goes is not known. GPU utilisation readings so far are single samples (about 96%
during the SOMA pass, 45-50% during the RTMPose pass). Code reading gives hypotheses only:

- 4K video decoded single-threaded, one camera at a time (`_iter_frames_av`, `thread_type = "NONE"`).
- A JPEG preview crop encoded per detection inside the loop (`DetectionBatchWriter.add_frame`).
- The segmentation route decodes a PNG mask per frame, resizes it to native size with nearest
  neighbour and scans it with `np.where` per label (`pose_worker.py`).
- RTMPose runs one frame's crops per call (no batching across frames); SOMA runs the flip test,
  doubling the model work.

Measurement on a fixed clip (10 s of the five-camera labelled trial), optimized environment, nothing
else running:

1. Per-stage wall time per frame, from timers around decode, box source, crop preparation, model
   call, decoding of the output, crop encoding and database write, enabled by an environment
   variable (the same pattern as `POSETRAK_PROFILE_FRAME_STEP` in the tracker).
2. A sampling profile of the whole process (`py-spy record --native`) to catch time outside the
   timers (GIL, conversion to `ndarray`, colour conversion).
3. GPU utilisation sampled once a second for the whole run (`nvidia-smi dmon -s u`), reported as
   mean and distribution, not single readings.
4. Routes: YOLOX + RTMPose-133; segmentation boxes + RTMPose-133; segmentation boxes + SOMA-77
   with and without flip test; RTMPose with hand refinement on and off.
5. Decode alone (frames read and dropped) per camera, with PyAV software decode and with hardware
   decode if available, as the floor any pipeline change can reach.

The result is a table of milliseconds per frame per stage and route; the pipeline work starts from
the largest entry.

### Instrumentation

`POSETRAK_PROFILE_DETECTION=1` switches on per-stage timing in the detection loop
(`posetrak/detection/stage_timing.py`); any other value is the path of a JSON file that receives the numbers
in addition to the table printed at exit. Stages: `decode`, `box_source`, `crop_preprocess`, `inference`,
`heatmap_decode`, `mask_gate`, `hand_model`, `preview_encode`, `db_write`; the hand pass reports its own
stages under `hand_pass/`. Times are exclusive (a nested stage is not counted in its parent). With the
variable unset the decorated functions and iterators are the original objects (a unit test checks this).
The SOMA-77 prototype route (`soma77_extract.py`) has no hooks of its own; its stages are measured from
outside with `python/tools/prototypes/profile_detection_routes.py`, which wraps the video container, the mask
reader and the connection and runs the prototype code unchanged.

### Results

Material: the five-camera, three-person trial, 10 s (46-56 s of the capture timeline), 6,000 camera-frames
(three 4K cameras at 120 fps, two FHD phone cameras at 120 fps), person masks of an edited segmentation run,
RTX 4080 SUPER, PyAV software decode, `onnxruntime-gpu` 1.23. Every run wrote its detection run into a copy
of the session database. Build: the commit that adds the instrumentation (no other change to the loop). Seconds are
totals over the clip; GPU utilisation is `nvidia-smi` sampled once a second (mean / 10th / 90th
percentile). "Masks + SOMA" is the prototype route (boxes copied from an existing segmentation-sourced run,
heatmaps gated by the person mask, batches of eight frames), for one of the three persons; the other
routes handle all three persons of every frame.

| Route | wall s | decode | box source | crop + preprocess | inference | heatmap decode + gate | hand pass | preview encode | db write | outside stages | GPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decode alone (PyAV) | 124 | 121 | - | - | - | - | - | - | - | 3 | 2 / 1 / 3 |
| YOLOX + RTMPose-133 | 395 | 135 | 143 | 17 | 90 | 2 | - | 3 | 1 | 4 | 42 / 30 / 51 |
| Masks + RTMPose-133, no hand pass | 437 | 47 | 149 | 19 | 212 | 2 | - | 3 | 1 | 4 | 44 / 31 / 58 |
| Masks + RTMPose-133, hand pass | 918 | 45 | 147 | 19 | 206 | 2 | 485 | 3 | 1 | 9 | 39 / 28 / 52 |
| Masks + SOMA-77, flip test, one person | 339 | 124 | 40 | 5 | 144 | 5 | - | - | 0 | 21 | 40 / 0 / 96 |
| Masks + SOMA-77, no flip test, one person | 280 | 134 | 41 | 5 | 73 | 5 | - | - | 0 | 21 | 22 / 0 / 69 |

Commands (shell, `PYTHONUTF8=1`, the variable set to a JSON path, GPU sampler `nvidia-smi
--query-gpu=utilization.gpu,... -l 1` alongside):

- `profile_detection_routes.py decode SESSION SHOT 46 56`
- `posetrak --session SESSION detect run --capture C --sync S --start 46 --end 56 --detector yolox-x --pose-model rtmpose-l-133kp`
- the same with `--segmentation RUN --no-refine-hands` and `--segmentation RUN --refine-hands`
- `profile_detection_routes.py soma SESSION SOURCE_RUN gem-x/onnx/vitpose.onnx 46 56 --mask-run RUN --mask-label 1`, and with
  `--no-flip` (inference of the unflipped crop only; the keypoints it writes are not meant to be used)

Per-unit costs behind the table:

| Unit | Cost |
|---|---|
| Decode of one 4K camera-frame (PyAV, software) | 27-31 ms (36.8 s for 1,199 frames of the slowest camera); FHD phone camera 9 ms |
| Decode of the same frames with `cv2.VideoCapture` (segmentation route) | 7.5-7.8 ms mean over all five cameras |
| YOLOX box source | 23.8 ms per frame |
| Mask box source in the RTMPose route (PNG decode, nearest resize to native, scan per label) | 24.9 ms per frame, three persons |
| Mask decode and grow in the SOMA route | 6.7 ms per frame, one person |
| RTMPose-133 inference | 5.7 ms per crop in the YOLOX route, 11.5-11.9 ms per crop in the mask route |
| SOMA-77 inference | 192 ms per batch of 16 crops (eight frames with their flipped copies): 12 ms per crop; 97 ms per eight crops without flip |
| Hand pass | 19.5 ms per body crop (both hands), plus a second decode of the whole clip (22.6 ms per frame) |
| Preview JPEG encode | 0.16 ms per crop |

Sampling profile of the slowest route (masks + RTMPose-133 with the hand pass; `py-spy record --native -r 20
-f raw --pid <process>`, 21,021 samples of the process; the run took 1,105 s with the profiler attached
against 918 s without): 51% of the samples are inside the ONNX Runtime `run` call (body and hand models), 46%
inside the hand pass's `detect_hand_in_crop`, 19% in the body pose `estimate`, and 13% in the mask handling
`_bboxes_from_mask` (nearest-neighbour resize of the mask to the native resolution and the scan per label)
against 4% for reading and decoding the stored PNG mask. The leaf functions are dominated by waiting for
the GPU: `NtGdiDdDDISubmitCommandToHwQueue` (command submission to the GPU queue) 22% and the CUDA runtime's
synchronisation 9%, then libavcodec 11.5% (the hand pass's second decode) and OpenCV 7%. No time outside the
timed stages stands out; the main thread spends about a third of its time waiting for the GPU while the GPU
itself is on average 39% busy, which is the signature of a serial loop in which decode, mask handling and
inference never overlap.

What this says about the hypotheses above:

- **Decode is a large and serial cost, the model is not the only bound.** Decoding alone takes 124 s for the
  10 s of capture (12 times real time, camera after camera). In the SOMA route it is 37-48% of the wall time
  and the GPU is idle while it runs: the utilisation is bimodal (10th percentile 0%, 90th percentile 96%,
  mean 40%), not a steady 96%; the earlier single readings of 96% are consistent with samples taken during the inference bursts.
- **Once decoding is overlapped the model dominates.** The slowest camera decodes in 37 s; SOMA inference for
  one person is 144 s with the flip test and 73 s without. A decode thread per camera would hide the decode
  (projection, not measured), and the bound would become inference (144 s for one person, about 430 s for
  three), which is also where a variable frame rate and the flip test would pay.
- **The prototype route decodes the video once per person and per pass.** Three persons need three passes
  (about 3 x 339 s), after a source pass for the boxes (395 s from YOLOX, 437 s from masks). A one-pass
  pipeline with overlapped decode would be bounded by inference, about 150 s per person (projection), so
  the acceptance target (twice the prototype's throughput) leaves a wide margin, including the source pass
  the prototype needs.
- **The YOLOX box source is as expensive as decoding** (36% of the route) and larger than the RTMPose model
  itself. The mask box source of the RTMPose route is also costly (34%): PNG decode, a nearest-neighbour resize
  of the mask to 4K and a scan per label. Both are cheaper in the SOMA route (mask decode 12%).
- **Preview encoding is not a problem** (0.7%), and nothing is lost by keeping it in the loop; it can move to
  the writer thread without effect on the numbers.
- **The hand pass doubles the RTMPose mask route** (918 s against 437 s): the hand model and a second decode
  of the clip. Inside the frame loop the second decode disappears (136 s); the hand model remains
  (349 s, 38% of the route) and is a candidate for batching.
- **Unexplained:** RTMPose inference is twice as expensive per crop in the segmentation route (11.5-11.9 ms)
  as in the YOLOX route (5.7 ms) for the same model, device and options. A stand-alone loop of the estimator
  with `cv2.VideoCapture` reads and a Qt core application gives 6.8 ms per crop, so neither explains it;
  the cause is not known and is left for the unified loop to remove or reveal.

## 4. Model weights and licence

The GEM-X model weights are not in the repository and must not be: the repository is Apache-2.0 and
MIT and REUSE-clean, and the weights are under other terms.

What the published terms say (read from the Hugging Face model card of `nvidia/GEM-X`, the NVIDIA
Open Model License Agreement and the DINOv3 License Agreement):

- The `nvidia/GEM-X` repository holds both the 3D model and the 2D model used here
  (`vitpose.pth`, `onnx/vitpose.onnx` with `.data`). It is not gated. Its card says the source code is
  under Apache-2.0 and "use of the associated model is governed by the NVIDIA Open Model License
  Agreement". The `LICENSE` file in the repository is the Apache-2.0 text; the card is what
  distinguishes code from weights.
- The NVIDIA Open Model License allows commercial use and redistribution. Redistributing the model
  requires giving recipients a copy of the agreement and a "Notice" text file containing "Licensed by
  NVIDIA Corporation under the NVIDIA Open Model License". Rights terminate if a guardrail of the model
  is bypassed without a substitute, or if the licensee starts copyright or patent litigation over the
  model. Use must be consistent with NVIDIA's Trustworthy AI terms. Derivative models belong to their
  maker.
- The model card's use-case restrictions forbid processing video of individuals without legal
  authorisation and, where required, consent, synthetic impersonation, unauthorised surveillance, and
  uses violating privacy or biometric law.
- The 2D model is a DINOv3 ViT-H backbone with a heatmap head (GEM-X's `vitpose_extractor.py` builds it
  as `Dinov3_ViTPose_huge_metrosim_256x192` from the `facebookresearch/dinov3` code). The model card
  does not say whether the backbone was initialised from Meta's DINOv3 weights. NVIDIA's listing of
  the corresponding NIM names the DINOv3 licence among its terms. If it applies, the DINOv3 License
  allows use, modification and redistribution of the materials and derivative works only under that
  agreement, with a copy of it and "Built with DINOv3" displayed in the product's documentation or
  interface; it prohibits military, ITAR-controlled, nuclear, espionage and weapons uses, and Meta may
  change its terms.

What follows for posetrak:

- The weights are downloaded on first use from the Hugging Face repository into the user's model
  cache, the way RTMPose checkpoints already are, after the user has been shown both licence texts
  and accepted them. The accepted licence version is stored with the downloaded file.
- The installer does not bundle the weights; bundling would require the notice file and both
  agreements in the installer, and waits for the answer on the DINOv3 terms.
- The about box and the user documentation carry the NVIDIA notice line and "Built with DINOv3".
- SOMA-77 is the recommended model: the detection dialogs and `detect run` preselect it and offer
  the download with the licences. A user who declines the licences, or has no weights, gets
  RTMPose-133 (with ViTPose where configured), and the pipeline says why. Both stay selectable.
- Whether the DINOv3 terms apply to `vitpose.pth` is a question for NVIDIA; until answered they are
  assumed to apply.

### Open item: licence question to NVIDIA

Status: drafted, not sent. The answer is recorded here, with the date it was received and who gave it,
and decides whether the installer may bundle the weights (plan owner decision 1) and which notices the
about box carries.

> Subject: Licence terms of `vitpose.pth` / `onnx/vitpose.onnx` in `nvidia/GEM-X`
>
> We are an open-source motion-capture project (Apache-2.0) that would like to use the 2D pose model
> from the `nvidia/GEM-X` Hugging Face repository (`vitpose.pth`, and the ONNX export
> `onnx/vitpose.onnx` with its `.data` file) as an optional pose detector. The model card says that use
> of the model is governed by the NVIDIA Open Model License Agreement. GEM-X's `vitpose_extractor.py`
> builds the network as `Dinov3_ViTPose_huge_metrosim_256x192` from the `facebookresearch/dinov3`
> code, and the listing of the corresponding NIM names the DINOv3 License Agreement among its terms.
> We would like to ask:
>
> 1. Was the backbone of `vitpose.pth` initialised from Meta's DINOv3 weights? If it was, do the terms of
>    the DINOv3 License Agreement apply to `vitpose.pth` and to the ONNX export, in addition to the
>    NVIDIA Open Model License Agreement, for (a) use and (b) redistribution?
> 2. If both apply, which notices must accompany a copy of the weights that our software downloads
>    from `nvidia/GEM-X` onto the user's machine at the user's request (the NVIDIA notice line, "Built
>    with DINOv3", copies of both agreements), and where must they be shown (about box, documentation,
>    download dialog)?
> 3. May we include the weights in an installer or other package that we distribute, provided those
>    notices and agreements are included, or should users always download them from your repository?
> 4. Do any of the use restrictions of either agreement go beyond those in the model card's use-case
>    section (consent for processing video of individuals, unauthorised surveillance, biometric law)?
>    Our software is used for body motion capture of consenting performers in studio and sport settings;
>    it does not identify people.
> 5. Is a new release of the model planned that would change the licence terms or the file layout?
>
> Please reply to the project maintainer address. We will keep your answer in the repository's design
> documentation so that downstream users can rely on it.

Sources to attach: the model card of `nvidia/GEM-X`, the NVIDIA Open Model License Agreement and the
DINOv3 License Agreement at the versions read when the draft was written.

## 5. Acceptance

- One `detect run` produces a SOMA-77 run from YOLOX boxes, from masks and from an existing run,
  without a prior RTMPose pass.
- On the profiling clip, the SOMA route's mean GPU utilisation is at least 85%, or the profile shows
  which non-GPU stage bounds it and that stage is at its decode floor.
- Throughput of the SOMA route from masks is at least twice that of the prototype route
  (RTMPose source run plus `soma77_extract.py`) on the same clip.
- Keypoints of the native SOMA run equal those of the prototype on the same boxes (same options)
  to within 0.5 px, apart from the documented decode change of half a cell.
