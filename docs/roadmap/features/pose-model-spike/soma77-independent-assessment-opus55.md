# Independent assessment of the GEM-X / SOMA-77 and candidate-keypoint spike

Status: a review of `soma77-spike-report.md`, written without reading the other assessment in this
directory. Sources: the report, the plan (`soma77-pose-spike-plan.md`) and the working notes the
report points to. Nothing was re-run; every number below is the spike's own. Where a statement is an
inference of this review and not a measurement, it says so. External methods named in section 5 are
quoted from memory and their claims and licences were not checked.

## 1. Verdict

The spike is honest and well instrumented: it reports its negative results, it found two real
tracker bugs, and its by-products (labeller, event list, incremental extraction) are worth keeping.
Its conclusions about what does not work at the candidate-selection level are credible.

Three things weaken what it concludes beyond that:

1. **The original question was never answered with a control.** The multi-person work ran only the
   new model. There is no ViTPose-L result on the same labels or the same 17 events, so the spike
   cannot say whether the new model is better, equal or worse in the footage that matters, nor
   whether the invisible-keypoint problem is a property of this model or of top-down heatmap
   detectors in general.
2. **The effort went into the stage with the least headroom.** The spike's own numbers say the
   choice of 2D candidate is not the binding constraint in easy ranges, and that in hard ranges the
   lever is visibility, not position. Eleven selection ideas were tried anyway; the measurement
   model and the estimator, where the evidence points, were treated as fixed.
3. **Several headline figures are less solid than they read**, because the labels were made by
   choosing among the candidates, the tolerance scales with a box that shrinks in falls, and the 3D
   reference is a tracker run with the same model biases (section 3).

Recommendation in one line: do not replace the detector now; spend a few days closing the missing
controls (section 6, step 1), then put the effort into the keypoint-to-skeleton measurement model,
robust estimation over short windows, and 3D correction, not into further 2D selection.

## 2. What the evidence supports, read from outside

**The model is a better 2D detector that does not produce a better 3D result.** On the rolling-fall
clip the share of joints far from the projected reference was 9% against 33% on the worst camera and
lower on most others, yet the tracked 3D error was level or worse (arms 9.5 / 8.5 cm against
6.9 / 6.2). A 2D advantage that disappears in 3D is lost between the detector and the state: in the
joint offsets, the noise weighting, or the skeleton. The notes list those hypotheses and none was
tested. This is the most important unexplained result of the single-person phase.

**The gate into the multi-person phase passed on visual judgement.** The plan's gate asked for
"clearly better somewhere that matters, or equal and cheaper". The measured result was "about as
accurate, not clearly better"; the model is also not cheaper (a 3.4 GB ViT-H, flip test, three
workarounds, per-subject offsets). The retargeting comparison was not blind and is confounded: a
trunk with keypoints will retarget better than one without, whatever the detector's accuracy.

**Selection is saturated, and the spike shows why.** In the first labelled range, a tracker fed the
operator's labels was no closer to the labels than one fed the strongest peak (median 0.05 of the
box height in all variants). In the hard range, the working notes contain an ablation the report
omits and which is the most decisive measurement of the whole spike: joint distance to the label-fed
run (median / 90th percentile) is 4.1 / 25.2 cm for the proposal, 3.5 / 19.1 cm when the operator's
positions replace the picks on visible keypoints but the picks on invisible ones are kept, and
1.7 / 15.9 cm when the proposal's positions are kept but the picks on operator-marked invisible
keypoints are dropped. Removing invisible picks is worth several times more than getting the
positions right. The report says this in words (section 8, item 2) but without the numbers that
make it a design constraint.

**The skeleton cannot explain the labels.** Even the label-fed run projects a keypoint within 3% of
the box height only about a quarter of the time. The report files this under "limits what a gate
can do". Read from outside it is a finding about the measurement model: the residual between
skeleton and keypoints is two to three times the tolerance used to call a pick correct. No
selection method can be validated, or exploited, more finely than the model that consumes it.

## 3. Methodological concerns

- **Labels were chosen from candidates.** The labeller offers accept, reject, reassign and
  not-visible; about 89% of labelled frames are accepted proposals. "A correct candidate exists on
  99% of visible keypoints" is therefore close to true by construction: a visible keypoint with no
  good candidate could only be labelled from another person's crop or marked not visible. The
  invisible class may contain "visible, but the model produced nothing usable". A few hundred
  frames labelled by free placement, without candidates on screen, would calibrate both figures.
- **The tolerance shrinks where the motion is hard.** 3% of the box height is several centimetres
  for a standing person and much less for a person on the mat. Part of the fall from 99% to 93% in
  the hard windows may be the metric. A body-scale normalisation (a bone length at the keypoint's
  depth) removes the pose dependence.
- **The 3D reference is not ground truth.** "Distance to the label-fed run" measures sensitivity to
  the picks. The claim that events "come from the picks, not from the tracker" assumes the label-fed
  run passes those events; the report does not say it was reviewed in video like the others. One
  noted window has a foot 60 cm off with picks that are 0-8% wrong, which looks like a tracker or
  observability failure.
- **The hardest event has no labels.** The high fall at about 51-53 s, the largest disagreement
  between runs, lies in the planned range that was not labelled, and its person has no labels in the
  range that overlaps it.
- **Small and reused sample.** One session, one trial of 28 s, three people in identical clothing,
  about 9 s labelled, one labeller. Eleven ideas were tuned against the same two ranges. The
  failure flag is 12 of 15 events (an interval of roughly 50-95%) with no precision: the number of
  flagged windows that are not events is not reported, and it decides whether the flag saves work.
- **Promising variants stopped before the tracker.** Pose-guided gating cut invisible picks from
  86-92% to 33-43%, the largest reduction of anything tried, and no tracker run with it is reported.
  Given the ablation above it is the variant most likely to have shown a tracker-level gain.
- **Metric composition.** 48 of the 77 keypoints are finger joints, which the spike found
  unreliable from the body crop. Some event distances are dominated by finger joints (up to 90 cm).
  Body and hands should be reported separately everywhere.
- **Questions the plan asked and the report leaves open.** Integration cost (plan question 4) has
  no estimate, the integration-form decision (replace, fuse, trunk only) is not taken, editing
  effort was a planned metric and was not measured, and the licences are still unread.

## 4. Assumptions worth challenging

**"Fast enough."** 12 ms per crop is fast; the pipeline is not. One person over 28 s of five cameras
takes about 15 minutes, roughly 30 times real time per person, about 100 times for three. An hour
of three-person capture is days of GPU time. If the product is meant for sessions and not clips,
throughput is a first-order requirement: drop the flip test where it does not pay, try half
precision and TensorRT, and detect at a lower rate than 120 Hz outside flagged windows.

**"Occlusion is the cause."** Occlusion, speed, motion blur, rolling shutter and synchronisation
error all peak in the same half second of a throw. The spike attributes the failures to occlusion
and identity without separating the others. Two cameras needed an agreement radius of up to 60 px
against 12 px for the rest, which already indicates calibration or synchronisation error in this
capture. The operator's labels allow a detector-free check: triangulate the labels themselves and
plot each camera's residual against keypoint speed. A residual that grows with speed is timing, not
occlusion.

**"Four cameras are the given."** In the hard windows half of all camera-keypoint observations are
invisible. Nobody counted how many cameras see each joint at the failure times; a joint seen by
fewer than two cameras is unobservable whatever is selected. Four usable views for three people in
contact is sparse. Recalibrating the fifth camera is listed under "later and larger" but adds a
quarter more views for a day of work, and a high viewpoint looking down between the bodies is the
classical answer to people occluding each other at ground level. Capture-side fixes are dismissed
for uncontrolled captures only; this is a controlled capture.

**"A fixed offset per bone is a calibration."** The spike measured that the ankle displacement is
21% of the segment upright and 12% in rolls, then fitted rigid offsets on upright poses of another
subject, against a reference that carries the old detector's 2-3 cm lateral bias, and applied them
to falls. The measurement model is most wrong exactly where the failures are. The model outputs
joint centres of its own skeleton; the alternative is to stop bending them onto a different
skeleton: triangulate the joint centres, estimate bone lengths and attachment points per subject
from that subject's own footage, and either track a skeleton with the detector's joint definition
or learn the mapping as pose dependent. The end use (retargeting to a character rig) does not
require the existing skeleton's joint definition.

**"Observations are independent."** The filter treats each camera and keypoint as an independent
measurement. The spike's own data shows the errors are correlated in every direction: across
neighbouring keypoints (the neighbours of a wrong pick are wrong in 57-74% of cases), in time (a
wrong pick follows the occluder along a smooth tracklet), and across cameras when the model
guesses. Correlated errors treated as independent make the filter overconfident, and an
overconfident filter with a hard Mahalanobis gate then rejects the correct observations while it
recovers. That would explain "wrong for a while after a fall" better than the process model alone.
The report contains no filter diagnostics (NIS, outlier counts, gate decisions) for the event
windows, although the tooling for it exists in the repository.

**"The estimator is a causal filter."** The pipeline is offline: extraction takes minutes, a
smoother already runs, and the failures are shorter than half a second with good frames on both
sides. That is the easy case for batch estimation and the hard case for a forward filter, which
must commit to an association and a gate at each step. A sliding-window least-squares problem over
one or two seconds, with a heavy-tailed loss and each keypoint's candidates as a mixture, is the
standard treatment of ambiguous, outlier-ridden measurements in visual SLAM. The report's direction
3 (probabilistic association, then forward and reverse passes) approximates it from inside the
filter; the direct formulation is probably simpler to prototype and has no latency argument against
it.

**"Association happens in the image."** All selection ideas scored 2D candidates, with
triangulation agreement as one feature. Neighbour grouping failed for long bones because a 2D
distance says nothing about a long bone, but a 3D bone length is invariant. Triangulating candidate
pairs across cameras first and assigning 3D candidates to persons under bone-length constraints
uses the one strong piece of geometry the spike left out.

**"Correction means editing keypoints."** Per-camera, per-keypoint labelling consumed most of the
spike and yields 385 tracks per person. The failures are few (17 in 28 s) and short. One posed 3D
keyframe inside each failure, used as a strong constraint by the smoother, may repair what hundreds
of 2D edits repair. The report ranks 3D correction fifth; with this failure shape it belongs near
the top, and it can be tested without building UI (section 6, step 4).

**"Visibility must come from geometry because detector signals failed."** A six-feature logistic
model is not the ceiling of what the image can tell. The backbone features see whether the joint is
there; a hand-built summary of the heatmap does not. More to the point, the required quality is
unknown: a classifier with AUC 0.9 gave nothing and the oracle gave a large gain. Corrupting the
oracle visibility labels at controlled error rates and re-running the tracker gives the operating
point a visibility estimator has to reach, from any source, before another one is built.

## 5. Viewpoints the spike did not take

- **Two detectors as a mutual check.** Both models have run on this footage. One is trained on real
  images, the other on synthetic renders with every joint labelled; their failure modes differ.
  Their 2D disagreement on shared joints is a failure and visibility signal that costs nothing to
  compute, and likely a better triage flag than the disagreement of two runs fed by the same
  detector. It also suggests the low-risk integration form: keep the current detector and add the
  new model's trunk, hip and head keypoints, which is where its measured and judged advantage lies.
- **Detectors built for this problem.** The operator's idea of feeding the mask to the model was
  shelved because it needs retraining. Published models exist that are conditioned on an instance
  mask for crowded scenes, and others that output presence and visibility probabilities per
  keypoint (from memory: MaskPose in the BBox-Mask-Pose work, ProbPose, and conditioned top-down
  models such as BUCTD). A day of testing one on the labelled ranges is cheaper than any of the
  heuristics tried. Unverified; check licences.
- **Fingers from the body crop are noise.** The spike showed it (0.20 of the hand size against
  0.05-0.07 from a hand crop) and kept producing, storing, linking and scoring 48 finger keypoints
  per person and camera. Dropping them from the body pass removes most of the candidate volume,
  the labelling volume and the metric contamination at once.
- **Cost per minute of footage as the metric.** The stated aim is mostly automatic tracking with
  targeted manual help. The quantity to optimise is operator minutes per minute of capture to an
  acceptable result, and none of the spike's metrics measure it.
- **Adaptive process noise.** Before a learned motion prior, inflate the process noise when the
  innovation statistics spike, so recovery after a fall takes frames and not seconds. It is a few
  lines in the filter and directly addresses the one process-model complaint in the report.

## 6. Recommended next steps

In order. Step 1 is a few days and decides the shape of everything after it.

1. **Close the controls.**
   - Score the existing ViTPose-L runs on the 17 events and against the labels (shared joints).
   - Have the operator review the label-fed runs in video, blind and mixed with the others.
   - Run the visibility corruption sweep to get the required operating point.
   - Run the tracker on the pose-guided gate picks.
   - Triangulate the labels: per-camera residual against speed, and cameras-seeing-joint counts in
     the event windows.
   - Pull filter diagnostics for the event windows: is the filter overconfident before the failure
     and does it reject correct observations after it.
2. **Decide the integration form.** On present evidence: do not replace. Add the trunk, hip and head
   keypoints of the new model to the existing pipeline, drop body-crop fingers, read the licences,
   and re-score the event table when the announced release arrives. Revisit if step 1 shows the
   current detector clearly worse on the events.
3. **Fix the measurement model before the association.** Per-subject, self-calibrated attachment
   points and bone lengths from triangulated joint centres of the subject's own clip, including
   non-upright poses; observation noise from the unclipped peak per body group instead of a
   saturated confidence. Success criterion: the label-fed run projects within the labelling
   tolerance most of the time, not a quarter of it.
4. **Test 3D keyframes as the correction primitive.** Take the label-fed pose at the midpoint of
   each scored event, feed it as one strong observation to the proposal-fed run, and count the
   events that disappear. If most do, build 3D correction and the triage flag, and stop 2D
   per-keypoint editing as the main workflow.
5. **Prototype robust windowed estimation** on the event windows only: candidates as a mixture,
   heavy-tailed loss, bone-length-consistent 3D association, compared event by event with the
   filter. Move to a filter-internal variant only if a causal estimate is a real requirement.
6. **Capture side.** Recalibrate the fifth camera, check synchronisation from the residual-speed
   plot, and add an elevated view in the next capture.
7. **Only then** a learned visibility head, a motion prior for gap filling, or body rendering for
   occlusion prediction; step 1 will say which of them, if any, the remaining failures need.

Stop: further local 2D selection heuristics, and extending per-keypoint 2D ground truth beyond the
small free-placement calibration set and the one unlabelled high fall.

## 7. What to keep from the spike

- The event-window evaluation and the event list; extend it to other trials before anything else is
  tuned on this one.
- The oracle ablation method (label-fed, visibility-only, position-only runs). It is the spike's
  most effective tool and should be the template for every future "where is the headroom" question.
- The two tracker fixes, tracking-run names, mask gating, the 1.6 crop and parabolic refinement.
- The labeller as an inspection and evaluation tool, not as a production editing workflow.
