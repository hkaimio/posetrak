# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""What a pose model contributes to the windowed choice of keypoint candidates.

The tracker (``posetrak-tracker track --kp-choice-model FILE``) reads a small CSV with, for the pose model of the
sequence, the distance limit of every pair of skeleton neighbours (in box heights) and the class of every keypoint
(for the mask term of the choice). See docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md,
section 10.4. Lines::

    pair,<landmark>,<landmark>,<limit in box heights>
    class,<landmark>,<class>

Landmarks are qualified like the markers' (``soma77:Hips``).
"""
from __future__ import annotations

from pathlib import Path

from posetrak.detection.pose_models import SOMA77

#: Keypoint classes the mask term can be restricted to.
KEYPOINT_CLASSES = ("body", "hands", "feet")


def soma77_keypoint_class(name: str) -> str:
    """Class of a SOMA-77 keypoint: the hand joints and fingers, the foot joints and toes, or the rest of the body."""
    if "Hand" in name:
        return "hands"
    if "Foot" in name or "Toe" in name:
        return "feet"
    return "body"


def choice_model_csv(pose_model: str | None) -> str | None:
    """The choice model of a pose model as CSV text, ``None`` for a model that has none (only SOMA-77 has)."""
    if not pose_model or not pose_model.startswith("soma77"):
        return None
    from posetrak.detection.soma77_neighbour_limits import SOMA77_NEIGHBOUR_LIMITS

    source = SOMA77.source
    lines = [f"# choice model of the {source} pose model; written by posetrak.detection.kp_choice_model"]
    lines += [f"pair,{source}:{a},{source}:{b},{limit}" for (a, b), limit in SOMA77_NEIGHBOUR_LIMITS.items()]
    lines += [f"class,{source}:{name},{soma77_keypoint_class(name)}" for name in SOMA77.keypoint_names]
    return "\n".join(lines) + "\n"


def write_choice_model(pose_model: str | None, path: Path) -> Path | None:
    """Write the choice model of *pose_model* to *path*; ``None`` (nothing written) when the model has none."""
    text = choice_model_csv(pose_model)
    if text is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
