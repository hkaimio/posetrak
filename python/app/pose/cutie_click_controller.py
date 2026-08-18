# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""cutie_click_controller.py — SAM2-backed interactive click segmentation.

Wraps Meta's own ``sam2`` package (Apache-2.0) with a stateful interface:
accumulate positive/negative clicks per person label, re-run the SAM2
decoder after each change, return a combined (H, W) uint8 labeled mask.

Replaces the earlier ultralytics.SAM-based implementation (see git
history) as part of removing the AGPL-3.0 ``ultralytics`` dependency --
see docs/license-analysis.md. ``sam2.SAM2ImagePredictor`` separates
image encoding (set_image(), the expensive ~200ms-on-GPU step) from
decoding (predict(), fast) explicitly, so set_image() now only runs
once per frame instead of once per click -- push_point() no longer
re-runs the image encoder at all, which the old ultralytics wrapper had
no way to avoid (it re-ran the whole model, encoder included, on every
predict() call). See "A fast cached-encoder path can be added later"
in the old version's docstring -- this is that path.

Falls back gracefully when sam2 is not installed.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    from sam2.build_sam import build_sam2_hf as _build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor as _SAM2ImagePredictor
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class ClickController:
    """Stateful SAM2 click segmentation for one video frame at a time.

    Usage
    -----
    ctrl = ClickController("facebook/sam2.1-hiera-base-plus")
    ctrl.set_image(frame_bgr)                        # register current frame
    mask = ctrl.push_point(1, x, y)                 # positive click, person 1
    mask = ctrl.push_point(1, x2, y2, positive=False)  # negative click
    mask = ctrl.clear_person(2)
    ctrl.clear_all()
    """

    def __init__(self, model_name: str = "facebook/sam2.1-hiera-base-plus") -> None:
        self._model_name = model_name
        self._predictor = None
        self._image_bgr: np.ndarray | None = None
        self._h = self._w = 0
        # label (1-based int) → [(x, y, is_positive)]
        self._clicks: dict[int, list[tuple[int, int, bool]]] = {}
        self._mask: np.ndarray = np.zeros((0, 0), dtype=np.uint8)
        # Base mask loaded from the DB for the current frame.  Persons without
        # live clicks show their base pixels; persons with live clicks have their
        # SAM2 result painted over their base region.
        self._base_mask: np.ndarray | None = None
        # True once set_image() has actually run the (expensive) encoder for
        # the current self._image_bgr -- avoids re-encoding on every click.
        self._image_encoded = False

        if _AVAILABLE:
            try:
                sam_model = _build_sam2_hf(model_name, device=_auto_device())
                self._predictor = _SAM2ImagePredictor(sam_model)
            except Exception as e:
                log.warning("Failed to load SAM2 model %s: %s", model_name, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._predictor is not None

    def set_image(self, frame_bgr: np.ndarray) -> None:
        """Register a new frame and clear all click and base-mask state."""
        self._image_bgr = frame_bgr
        self._h, self._w = frame_bgr.shape[:2]
        self._clicks.clear()
        self._base_mask = None
        self._mask = np.zeros((self._h, self._w), dtype=np.uint8)
        self._image_encoded = False

    def set_base_mask(self, labeled: np.ndarray | None) -> np.ndarray:
        """Load a stored labeled mask as the base layer for this frame.

        Persons that have no live SAM2 clicks keep their base pixels.
        Persons that do have live clicks have their SAM2 result painted
        over their base region (the old base pixels for that label are
        discarded before the SAM2 mask is applied).

        Call after set_image() whenever a stored mask exists for the frame.
        Returns the updated display mask.
        """
        import cv2
        if labeled is None:
            self._base_mask = None
        else:
            if labeled.shape[:2] != (self._h, self._w):
                labeled = cv2.resize(
                    labeled, (self._w, self._h), interpolation=cv2.INTER_NEAREST
                )
            self._base_mask = labeled.astype(np.uint8)
        return self._run_predictions()

    def push_point(
        self, label: int, x: int, y: int, positive: bool = True
    ) -> np.ndarray:
        """Add a click for person *label* at image coords (x, y).

        Returns the updated (H, W) uint8 labeled mask.
        """
        self._clicks.setdefault(label, []).append((x, y, positive))
        return self._run_predictions()

    def clear_person(self, label: int) -> np.ndarray:
        """Remove all clicks for person *label*.  Returns updated labeled mask."""
        self._clicks.pop(label, None)
        return self._run_predictions()

    def clear_all(self) -> None:
        """Remove all live clicks and the base mask; reset to blank."""
        self._clicks.clear()
        self._base_mask = None
        self._mask = np.zeros((self._h, self._w), dtype=np.uint8)

    def get_mask(self) -> np.ndarray:
        return self._mask

    def click_count(self, label: int) -> int:
        return len(self._clicks.get(label, []))

    def get_all_clicks(self) -> dict[int, list[tuple[int, int, bool]]]:
        """Return {label: [(x, y, positive), ...]} for all persons."""
        return {lbl: list(pts) for lbl, pts in self._clicks.items() if pts}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_image_encoded(self) -> bool:
        """Run SAM2's image encoder for self._image_bgr if not done yet.

        Returns False if there's no image or encoding failed.
        """
        if self._image_encoded:
            return True
        if not self.available or self._image_bgr is None:
            return False
        try:
            import cv2
            rgb = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(rgb)
            self._image_encoded = True
            return True
        except Exception:
            log.exception("SAM2 set_image failed")
            return False

    def _run_predictions(self) -> np.ndarray:
        import cv2

        # Start from the base mask so persons without live clicks are preserved.
        if self._base_mask is not None and self._base_mask.shape == (self._h, self._w):
            combined = self._base_mask.copy()
        else:
            combined = np.zeros((self._h, self._w), dtype=np.uint8)

        if self._image_bgr is None:
            self._mask = combined
            return combined

        for label, clicks in self._clicks.items():
            if not clicks:
                continue
            points = [[x, y] for x, y, _ in clicks]
            labels = [1 if pos else 0 for _, _, pos in clicks]

            # Clear this person's base pixels before applying SAM2 result
            # so the SAM2 mask fully replaces their old region.
            combined[combined == label] = 0

            m = self._predict(points, labels)
            if m is not None:
                if m.shape != (self._h, self._w):
                    m = cv2.resize(
                        m.astype(np.uint8), (self._w, self._h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                combined[m] = label

        self._mask = combined
        return combined

    def _predict(
        self, points: list[list[int]], labels: list[int]
    ) -> np.ndarray | None:
        """Run SAM2 with point prompts; return (H, W) bool mask or None."""
        if not self._ensure_image_encoded():
            return None
        try:
            masks, scores, _logits = self._predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=False,
            )
            if masks is None or len(masks) == 0:
                log.debug("SAM2 returned no masks")
                return None
            m = masks[0].astype(bool)
            log.debug("SAM2 mask shape=%s score=%.3f any=%s", m.shape, scores[0], m.any())
            return m
        except Exception:
            log.exception("SAM2 predict failed")
            return None
