# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""mask_treatment.py — suppress everyone-but-the-target-person in a crop
before pose estimation, using an existing segmentation mask.

Phase 1 of docs/roadmap/features/segmentation-pose-treatment/
segmentation-pose-treatment-design.md: during a grab, a top-down pose
model's crop shows both people at once, and it's the model's own
judgement -- not anything the pipeline controls -- which one it actually
estimates joints for. Validated by an offline study
(python/tools/segmentation_mask_steering_experiment.py) across ~40,000
metric rows on real grab footage: "feather2" (soften everything outside
the target's own mask with a blur + contrast reduction, blended in over
a narrow band from the mask boundary rather than a hard cutoff) measurably
improves which body the keypoints land on, without the confident
hallucination hard/blur cutoffs were observed to produce once they erase
a grabbing hand's palm entirely.

Only meaningful against a mask a human has actually verified (see the
design doc's "why this can't be a live, automatic step") -- there is no
guard here against a bad mask, because the callers this is wired into
(PoseWorker, reading curated seg_masks rows) only ever have a verified one
in the first place.
"""
from __future__ import annotations

import cv2
import numpy as np

#: Half-width of the soft transition band, in pixels, outward from the
#: mask boundary -- narrower than the "feather" treatment tried first
#: (15px), which left the band immediately around a person almost
#: unblurred (alpha only reaches ~0.07 at 1px out) -- weakest exactly
#: where a grabbing hand sits.
FEATHER_PX = 10

#: Gaussian blur kernel size. Strong enough that shape/motion survives but
#: fine detail (fingers) doesn't.
BLUR_KSIZE = 45

#: 0 = flatten fully to gray, 1 = no contrast reduction. Applied on top of
#: the blur so a masked-out region gives the pose model even less to
#: hallucinate a hand from than a merely-blurred one.
CONTRAST_FACTOR = 0.4

FILL_GRAY = 127


def suppress_others(frame: np.ndarray, mask: np.ndarray, target_label: int) -> np.ndarray:
    """Return a copy of *frame* with every pixel outside *target_label*'s
    region in *mask* blurred and contrast-reduced, blended in smoothly
    over a narrow band from the mask boundary (the "feather2" treatment).

    Parameters
    ----------
    frame:
        (H, W, 3) BGR image, same resolution as *mask*.
    mask:
        (H, W) uint8 labeled mask -- 0 = background, label = a person.
    target_label:
        Which label to preserve; every other pixel (background AND any
        other tracked person) is treated.
    """
    target = (mask == target_label).astype(np.uint8)
    dist = cv2.distanceTransform(1 - target, cv2.DIST_L2, 3)
    alpha = np.clip(dist / FEATHER_PX, 0.0, 1.0)[..., None]  # 0 inside, 1 far outside

    blurred = cv2.GaussianBlur(frame, (BLUR_KSIZE, BLUR_KSIZE), 0)
    low_contrast = blurred.astype(np.float32) * CONTRAST_FACTOR + FILL_GRAY * (1 - CONTRAST_FACTOR)

    return (frame.astype(np.float32) * (1 - alpha) + low_contrast * alpha).astype(np.uint8)
