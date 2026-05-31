"""cutie_click_controller.py — SAM2-backed interactive click segmentation.

Wraps ultralytics.SAM with a stateful interface: accumulate positive/negative
clicks per person label, re-run SAM2 after each change, return a combined
(H, W) uint8 labeled mask.

Each call to push_point() re-runs the image encoder + decoder (~200 ms on
GPU).  A fast cached-encoder path can be added later; for Phase 2 interactive
use 200 ms per click is acceptable.

Falls back gracefully when ultralytics is not installed.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    from ultralytics import SAM as _SAM
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class ClickController:
    """Stateful SAM2 click segmentation for one video frame at a time.

    Usage
    -----
    ctrl = ClickController("sam2.1_b.pt")
    ctrl.set_image(frame_bgr)                        # register current frame
    mask = ctrl.push_point(1, x, y)                 # positive click, person 1
    mask = ctrl.push_point(1, x2, y2, positive=False)  # negative click
    mask = ctrl.clear_person(2)
    ctrl.clear_all()
    """

    def __init__(self, model_path: str = "sam2.1_b.pt") -> None:
        self._model_path = model_path
        self._sam = None
        self._image_bgr: np.ndarray | None = None
        self._h = self._w = 0
        # label (1-based int) → [(x, y, is_positive)]
        self._clicks: dict[int, list[tuple[int, int, bool]]] = {}
        self._mask: np.ndarray = np.zeros((0, 0), dtype=np.uint8)
        # Base mask loaded from the DB for the current frame.  Persons without
        # live clicks show their base pixels; persons with live clicks have their
        # SAM2 result painted over their base region.
        self._base_mask: np.ndarray | None = None

        if _AVAILABLE:
            try:
                self._sam = _SAM(model_path)
            except Exception as e:
                log.warning("Failed to load SAM2 model %s: %s", model_path, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._sam is not None

    def set_image(self, frame_bgr: np.ndarray) -> None:
        """Register a new frame and clear all click and base-mask state."""
        self._image_bgr = frame_bgr
        self._h, self._w = frame_bgr.shape[:2]
        self._clicks.clear()
        self._base_mask = None
        self._mask = np.zeros((self._h, self._w), dtype=np.uint8)

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
        if not self.available or self._image_bgr is None:
            return None
        try:
            results = self._sam.predict(
                self._image_bgr,
                points=points,
                labels=labels,
                verbose=False,
            )
            if not results or results[0].masks is None:
                log.debug("SAM2 returned no masks")
                return None
            m = results[0].masks.data[0].cpu().numpy().astype(bool)
            log.debug("SAM2 mask shape=%s any=%s", m.shape, m.any())
            return m
        except Exception:
            log.exception("SAM2 predict failed")
            return None
