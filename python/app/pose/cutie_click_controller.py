"""cutie_click_controller.py — SAM2-backed interactive click segmentation.

Wraps ultralytics.SAM (SAM2 backend) with a stateful interface: accumulate
positive/negative clicks per person label, re-run SAM2 after each change,
return a combined (H, W) uint8 labeled mask.

The image encoder runs once per frame (set_image); subsequent clicks reuse
the cached features via prompt_inference, so each click costs only the
lightweight SAM2 decoder (~30 ms on GPU).

If ultralytics is not installed the controller degrades gracefully: all
methods still work but always return empty masks.
"""
from __future__ import annotations

import numpy as np

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
    ctrl.set_image(frame_bgr)          # encode image (slow ~200 ms, once per frame)
    mask = ctrl.push_point(1, x, y)    # add positive click for person label 1
    mask = ctrl.push_point(1, x2, y2, positive=False)  # negative click
    mask = ctrl.clear_person(2)        # remove all clicks for person 2
    ctrl.clear_all()                   # reset
    """

    def __init__(self, model_path: str = "sam2.1_b.pt") -> None:
        self._model_path = model_path
        self._sam = None
        self._predictor = None      # ultralytics Predictor, lazily initialised
        self._cached_im = None      # preprocessed image tensor (after set_image)

        self._image_bgr: np.ndarray | None = None
        self._h = self._w = 0
        # label (1-based int) → [(x, y, is_positive)]
        self._clicks: dict[int, list[tuple[int, int, bool]]] = {}
        self._mask: np.ndarray = np.zeros((0, 0), dtype=np.uint8)

        if _AVAILABLE:
            try:
                self._sam = _SAM(model_path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if SAM2 is loaded and ready."""
        return self._sam is not None

    def set_image(self, frame_bgr: np.ndarray) -> None:
        """Set a new frame, encode image features, clear all click state.

        This is the slow call (~200 ms on GPU); subsequent push_point() calls
        reuse the cached features and are much faster.
        """
        self._image_bgr = frame_bgr
        self._h, self._w = frame_bgr.shape[:2]
        self._clicks.clear()
        self._mask = np.zeros((self._h, self._w), dtype=np.uint8)
        self._cached_im = None

        if not self.available:
            return
        try:
            pred = self._get_predictor()
            pred.set_image(frame_bgr)
            # Cache the preprocessed tensor AND fix self.batch so that
            # prompt_inference uses the correct original-image dimensions
            # (orig_hw).  set_image() does not set self.batch — only the
            # full stream_inference path does — so without this fix,
            # prompt_inference would scale point coords against the dummy
            # 64×64 image used to initialise the predictor.
            for batch in pred.dataset:
                pred.batch = batch
                self._cached_im = pred.preprocess(batch[1])
                break
        except Exception:
            self._cached_im = None

    def push_point(
        self,
        label: int,
        x: int,
        y: int,
        positive: bool = True,
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
        """Remove all clicks for all persons and reset the mask."""
        self._clicks.clear()
        self._mask = np.zeros((self._h, self._w), dtype=np.uint8)

    def get_mask(self) -> np.ndarray:
        """Return the current (H, W) uint8 labeled mask."""
        return self._mask

    def click_count(self, label: int) -> int:
        """Total click count (positive + negative) for *label*."""
        return len(self._clicks.get(label, []))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_predictor(self):
        """Return the ultralytics Predictor, initialising it on first call."""
        if self._predictor is not None:
            return self._predictor
        # A dummy predict() call is needed to initialise the predictor object.
        self._sam.predict(
            np.zeros((64, 64, 3), dtype=np.uint8),
            points=[[32, 32]],
            labels=[1],
            verbose=False,
            imgsz=64,
        )
        self._predictor = self._sam.predictor
        return self._predictor

    def _run_predictions(self) -> np.ndarray:
        """Re-run SAM2 for every person that has clicks; return combined mask."""
        import cv2

        combined = np.zeros((self._h, self._w), dtype=np.uint8)
        if self._image_bgr is None:
            self._mask = combined
            return combined

        for label, clicks in self._clicks.items():
            if not clicks:
                continue
            pts = np.array([[x, y] for x, y, _ in clicks], dtype=np.float32)
            lbls = np.array([1 if pos else 0 for _, _, pos in clicks], dtype=np.int32)

            m = self._sam2_predict(pts, lbls)
            if m is not None:
                if m.shape != (self._h, self._w):
                    m = cv2.resize(
                        m.astype(np.uint8), (self._w, self._h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                combined[m] = label

        self._mask = combined
        return combined

    def _sam2_predict(
        self, points: np.ndarray, labels: np.ndarray
    ) -> np.ndarray | None:
        """Run one SAM2 prompt; return (H, W) bool mask or None on failure."""
        try:
            if self._cached_im is not None and self._predictor is not None:
                # Fast path: reuse cached image encoder output.
                # Wrap points/labels as (1, N, 2) / (1, N) numpy arrays so
                # torch.as_tensor() receives a single ndarray, not a list of
                # ndarrays (which triggers a slow-path warning).
                pts_3d = points[np.newaxis]        # (1, N, 2)
                lbl_2d = labels[np.newaxis]        # (1, N)
                masks, _scores, _logits = self._predictor.prompt_inference(
                    self._cached_im, points=pts_3d, labels=lbl_2d
                )
                return masks[0].cpu().numpy().astype(bool)
            else:
                # Fallback: full predict() (re-encodes image, ~200 ms).
                results = self._sam.predict(
                    self._image_bgr,
                    points=[points.tolist()],
                    labels=[labels.tolist()],
                    verbose=False,
                )
                if results and results[0].masks is not None:
                    return results[0].masks.data[0].cpu().numpy().astype(bool)
        except Exception:
            pass
        return None
