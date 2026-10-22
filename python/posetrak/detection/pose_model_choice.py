# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pose_model_choice.py — which pose model a detection run uses, and how its estimator is built.

SOMA-77 is the recommended model and the default of ``detect run`` and the detection dialogs. Its weights are
downloaded after the user accepts their licences (``model_weights``); a user who declines, or whose machine cannot
fetch them, gets RTMPose-133 instead, with the reason returned to the caller to show.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from posetrak.detection.backends import PoseEstimator
from posetrak.detection.model_weights import AcceptCallback, LicenceDeclined, download_gemx, soma77_model_path

_log = logging.getLogger(__name__)

SOMA_MODEL = "soma77-vitpose-h"
DEFAULT_POSE_MODEL = SOMA_MODEL
FALLBACK_POSE_MODEL = "rtmpose-l-133kp"
#: Pose models offered in the dialogs and accepted by ``detect run``, recommended first.
POSE_MODELS = (SOMA_MODEL, "rtmpose-l-133kp", "vitpose-l-133kp")


def is_soma(pose_model: str) -> bool:
    return pose_model.startswith("soma77")


def resolve_pose_model(
    requested: str | None,
    accept: AcceptCallback | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[str, str | None]:
    """The pose model to run for *requested* (None: the default), and why it differs from it, if it does.

    A SOMA-77 request with no usable weights offers the download through *accept* (which shows the licence
    texts and returns whether the user accepted); without *accept*, or if it declines or the download fails,
    the result is RTMPose-133 and the reason.
    """
    model = requested or DEFAULT_POSE_MODEL
    if not is_soma(model) or soma77_model_path() is not None:
        return model, None
    if accept is None:
        return FALLBACK_POSE_MODEL, (
            f"{model}: the weights are not downloaded; run with the licences accepted (--accept-licences) to fetch them")
    try:
        download_gemx(accept, progress)
        return model, None
    except LicenceDeclined as exc:
        return FALLBACK_POSE_MODEL, f"{model}: {exc}"
    except Exception as exc:  # network, disk, hub errors: the run can still be made with RTMPose
        _log.warning("SOMA-77 weights download failed", exc_info=True)
        return FALLBACK_POSE_MODEL, f"{model}: the weights could not be downloaded ({exc})"


def create_estimator(pose_model: str, device: str | None = None, **options) -> PoseEstimator:
    """The estimator for a resolved *pose_model*; *options* go to its constructor.

    ``candidates`` (a :class:`~posetrak.detection.kp_candidates.CandidateOptions`) is accepted for SOMA-77 only.
    """
    if options.get("candidates") is None:
        options.pop("candidates", None)
    elif not is_soma(pose_model):
        raise ValueError(f"Keypoint candidates are available for SOMA-77 runs only, not for {pose_model}.")
    if is_soma(pose_model):
        from posetrak.detection.backends_soma77 import Soma77Estimator

        return Soma77Estimator(device=device, **options)
    from posetrak.detection.backends_rtmpose import RTMPoseEstimator

    return RTMPoseEstimator(model_name=pose_model, device=device, **options)
