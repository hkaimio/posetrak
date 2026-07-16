# Pose detection improvements for aikido capture — analysis

Captured 2026-07-16 (discussion with Claude). Brainstorm of detection- and
tracker-side improvements for the aikido-specific failure modes, beyond
what is already implemented or planned elsewhere. Deliberately excludes
what the error-improvements Phase 5 plan already covers
(`docs/roadmap/features/error-improvements/phase5-cross-person-plan.md`:
cross-person relative observations, contact gating, collision detection).

## Problem statement

Aikido technique capture is close to worst-case for 2D pose estimation:

- Many common poses are far out of distribution for pose models (ukemi,
  fast throws with the person strongly bent or upside down).
- Practitioners are very close and occlude each other, making limb
  ownership ambiguous.
- Wrist grabs put one person's hand on the other's arm; detectors
  associate A's hand with B's arm.
- The hakama hides exact knee location.
- All practitioners wear near-identical uniforms, weakening person
  re-identification.
- Hip keypoints are systematically biased when the person is seen from
  the back and/or there is a large spine–thigh angle.

Constraint: prefer solutions that work "in the wild" — no markers, no
special uniforms, no lab-only setup.

Already done (baseline this analysis builds on):

- RTMPose → ViTPose switch.
- Cutie video segmentation with temporal memory; crop areas derived from
  segmentation instead of detector bboxes.
- Keypoint confidence down-weighted outside the person's segmentation
  mask.
- Separate hand pose estimator run on the area where the body model
  predicts the hand should be (hand-detection-refinement Idea 3).

Suggestions below are roughly ordered by expected payoff per effort.

## Tier 1 — cheap, targeted wins exploiting existing infrastructure

### 1. Rotation-canonicalized crops for the detector

The single biggest stated problem — ukemi and inverted/strongly bent
poses — has a well-known cause: 2D pose models are trained on
overwhelmingly upright people, and accuracy degrades badly past ~90° of
body rotation. But this pipeline has something almost no pose-estimator
deployment has: a per-frame 3D prediction of each person's torso
orientation from the UKF.

Project the predicted spine axis into each camera, rotate the crop so
the person is upright, run ViTPose, rotate the keypoints back. An
upside-down ukemi becomes an in-distribution upright pose from the
model's perspective. This is the same trick as hand-detection-refinement
(use the tracker's prediction to condition the detector), applied to
orientation instead of location, and needs no training.

Fallback when the prediction is untrusted in a given frame: test-time
augmentation — run the crop at 0/90/180/270° and keep the
highest-confidence result.

### 2. Mask the other person out of the crop

Down-weighting keypoints outside the Cutie mask is a post-hoc
correction — the detector has already "seen" the other person and stolen
their limbs. Stronger: before running ViTPose on person A's crop, paint
over the pixels belonging to person B's mask (gray fill, blur, or mean
color). Top-down estimators hallucinate limb assignments precisely
because a second person is visible in the crop; removing them at the
input attacks the wrist-grab and limb-ownership problems directly.

Main risk is at the actual contact boundary where masks touch — consider
eroding B's mask slightly, or applying this only when the persons
overlap substantially. Worth an A/B test on a few known-bad sequences.

### 3. Fine-tune ViTPose on self-harvested labels

On the "which base model" question: **stay with ViTPose**. It is already
integrated, fine-tunes well with small datasets in the mmpose ecosystem,
and its plain-ViT backbone is exactly the architecture that benefits
from domain data. The interesting part is where the labels come from —
two free sources:

- **Manual keypoint edits.** Every correction made in the editing UI is
  a ground-truth label on a hard example — by construction the failure
  cases, which is exactly what active learning wants. Harvest
  `pose_observation_edits` plus the surrounding crop as training
  samples.
- **Multi-view-consistent pseudo-labels.** Frames where tracking
  converged well (low NIS, multi-camera inliers) give pseudo-labels for
  free: reproject the smoothed 3D solution into each camera — including
  cameras where the 2D detection was wrong or missing. The reprojection
  into a back-view camera teaches the model where the hip *actually* is
  when seen from behind, directly targeting the hip-bias problem. This
  self-training loop (3D-consistency-filtered pseudo-labels → fine-tune
  2D detector → better 3D) is a known, effective pattern for
  multi-camera rigs.

If switching bases rather than fine-tuning is ever on the table, the one
alternative worth evaluating is Meta's Sapiens family — pretrained on
~300M human images, notably robust on unusual poses and occlusion — but
the integration cost is real and fine-tuning ViTPose probably captures
most of the benefit.

### 4. Capture-side improvements that still count as "in the wild"

No markers or special uniforms required:

- **Fast shutter.** Motion blur during fast throws silently destroys
  detections. Force 1/500 s or faster and accept higher ISO / add
  ambient light — action cameras default to shutter speeds that smear a
  fast ukemi across many pixels.
- **One elevated / near-overhead camera.** Drastically reduces
  person-person occlusion and makes two-person assignment nearly trivial
  in that view. Even one camera on a balcony or tall pole gives the
  triangulator and identity logic a view where the practitioners rarely
  overlap.

## Tier 2 — tracker-side improvements (moderate effort)

### 5. Reassign swapped detections instead of just rejecting them

The Mahalanobis gate currently discards a keypoint that matches the
wrong person. But "person A's wrist detection is an outlier for A *and*
a good inlier for B's predicted wrist" is a swap signature that can be
acted on: reassign the detection rather than losing it. The Phase 5 plan
already mentions detecting this signature defensively (skip the anchor);
the offensive version — a small cross-person data-association step over
rejected keypoints before finalizing each frame's observation set —
recovers data in exactly the contact windows where the filter is most
starved for it. Fits naturally into the `MultiPersonTracker`
orchestrator.

### 6. View-dependent bias handling for hips (and knees)

The hip error — wrong when seen from the back or with strong hip
flexion — is a systematic *bias*, and the UKF assumes zero-mean noise,
so it doesn't just add noise, it pulls the solution. Since the tracker
knows body orientation relative to each camera, hip/knee observation
noise can be made anisotropic and view-dependent: inflate (or skip) hip
observations from cameras looking at the person's back, or learn a small
per-keypoint correction offset as a function of relative view angle from
converged-tracking data. Item 3's pseudo-label fine-tuning attacks the
same problem from the detector side; this attacks it from the filter
side and is much quicker to ship.

### 7. Floor-contact constraints for the lower body (ZUPT)

The hakama hides the knee, and no detector fix will conjure a joint the
camera cannot see — but aikido provides a strong substitute prior: feet
(and in suwari-waza, knees) are planted on a known floor plane a large
fraction of the time. Detect stationary-foot periods (low predicted foot
velocity + stable ankle/toe detections) and add pseudo-measurements:
foot sole on the floor plane, zero velocity while planted. This is the
classic ZUPT idea from inertial tracking, fits the UKF observation
framework naturally, and constrains the knee through the kinematic chain
(hip + ankle + segment lengths + floor leave the knee only one place to
be). The existing relative-observation infrastructure suggests a
"world-plane contact" observation type would be a contained change.

### 8. Feed tracking back into segmentation identity

Similar uniforms mean Cutie's ID swaps happen at exactly the crossings
where everything else fails too. They can be audited for free: project
each tracked skeleton into each camera and check it lands inside its own
person's mask. A sustained mismatch means the masks swapped — flag it in
the UI (dovetails with first-release backlog item 5, surfacing
crisis-debugging patterns) or auto-correct the mask labels for
downstream crops.

## Assessment of the two originally proposed ideas

### Contact / technique detection

Split this in two:

- A general "detect where persons touch" model is expensive to train,
  and the Phase 5 geometric gating (tracked marker distance +
  hysteresis) may already provide most of the gating signal — wait for
  Phase 5 measurements before investing.
- The tractable, high-value subset is a **wrist-grab classifier**. Grabs
  are the one contact type that occurs at predefined marker locations
  (hand on wrist), they are ubiquitous in aikido, and a positive
  detection justifies a much stronger constraint than the generic
  proximity gate — essentially "these two markers coincide," which the
  `PAIR_DIFF`/anchor machinery can express with tight noise. Training
  data is cheap: labelling grab intervals per wrist pair is far faster
  than per-keypoint annotation.
- Technique classification (nikyo etc.) ranks low for *tracking*
  accuracy — it helps annotation and analysis, but converting "this is
  nikyo" into a usable measurement constraint is a long road; the grab
  detector yields the constraint directly.

### Fine-tuning the pose detector

Do it; base it on ViTPose; let the editing UI and the tracker generate
the dataset rather than annotating from scratch (see item 3).

## Longer-term bet — learned motion prior

The constant-velocity process model is weakest exactly during throws —
fast, coordinated, highly non-linear motion. A learned motion prior (a
small sequence model trained on cleaned captures, HuMoR-style, used as
the process model or as an additional pseudo-measurement during
high-NIS windows) would help most where detection helps least.
Significant work, and it only becomes attractive once a library of
cleaned aikido sequences has accumulated — which everything above
accelerates.

## Recommended starting picks

Three to start with: rotation-canonicalized crops (1), other-person
masking (2), and the edit-harvesting fine-tune loop (3). They are
independent of the Phase 5 work, all work "in the wild," and each
targets one of the top three failure modes directly.
