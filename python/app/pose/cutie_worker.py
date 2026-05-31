"""cutie_worker.py — QThread wrapper for Cutie bidirectional mask propagation.

Runs one tracking pass (forward or backward) on a single camera video,
emitting mask_ready(frame_idx, mask) per frame so the UI can update live
and persist results to the DB.

Requires:
  - CUTIE_DIR env var or the default ../tests/Cutie sibling directory
  - torch, torchvision, omegaconf, hydra-core, einops (all in posetrak venv)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


class CutieWorker(QThread):
    """Runs a single Cutie forward or backward tracking pass.

    Signals
    -------
    mask_ready(frame_idx, mask):
        Emitted for every tracked frame. mask is (H, W) uint8 labeled array
        (label 0 = background, 1..N = persons in persons_ordered order).
    progress(current, total):
        Emitted every 50 frames so the UI can show a progress indicator.
    finished():
        Emitted when the pass completes or is stopped.
    error(message):
        Emitted if an unrecoverable error occurs.
    """

    mask_ready = Signal(int, object)   # frame_idx, np.ndarray
    progress   = Signal(int, int)      # frames_done, total_frames
    finished   = Signal()
    error      = Signal(str)

    def __init__(
        self,
        video_path: str,
        init_frame: int,
        init_mask: np.ndarray,          # (H, W) uint8 labeled mask
        persons_ordered: list[str],     # label 1..N → person name (for logging)
        first_frame: int,               # inclusive track range start
        last_frame: int,                # inclusive track range end
        direction: str,                 # "forward" or "backward"
        device: str = "cuda",
        max_internal_size: int = 480,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._init_frame = init_frame
        self._init_mask = init_mask.copy()
        self._persons = persons_ordered
        self._first = first_frame
        self._last = last_frame
        self._direction = direction
        self._device = device
        self._max_internal_size = max_internal_size
        self._stop_requested = False

    def stop(self) -> None:
        """Request the worker to stop after the current frame."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # QThread.run
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            if self._direction == "forward":
                self._run_forward()
            else:
                self._run_backward()
        except Exception:
            log.exception("CutieWorker error")
            self.error.emit("Cutie tracking failed — see console for details.")
        finally:
            self.finished.emit()

    # ------------------------------------------------------------------
    # Internal — Cutie setup
    # ------------------------------------------------------------------

    def _load_cutie(self):
        """Import Cutie (via sys.path) and return a loaded model."""
        import sys
        from pathlib import Path
        from pipeline.pose.segmentation import _find_cutie_dir
        cutie_dir = _find_cutie_dir()
        if str(cutie_dir) not in sys.path:
            sys.path.insert(0, str(cutie_dir))

        from cutie.utils.get_default_model import get_default_model
        model = get_default_model()
        return model

    def _new_processor(self, model):
        from cutie.inference.inference_core import InferenceCore
        proc = InferenceCore(model, cfg=model.cfg)
        proc.max_internal_size = self._max_internal_size
        return proc

    @staticmethod
    def _to_tensor(bgr: np.ndarray, device: str):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return to_tensor(Image.fromarray(rgb)).to(device).float()

    # ------------------------------------------------------------------
    # Forward pass: init_frame → last_frame
    # ------------------------------------------------------------------

    def _run_forward(self) -> None:
        import torch

        log.info("CutieWorker: forward  [%d, %d]  init=%d",
                 self._init_frame, self._last, self._init_frame)
        model = self._load_cutie()
        proc = self._new_processor(model)
        objects = list(range(1, len(self._persons) + 1))
        init_t = torch.from_numpy(self._init_mask).to(self._device)

        total = self._last - self._init_frame + 1
        done = 0

        cap = cv2.VideoCapture(self._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._init_frame)
        initialized = False

        try:
            with torch.inference_mode(), torch.amp.autocast(self._device):
                for fi in range(self._init_frame, self._last + 1):
                    if self._stop_requested:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img_t = self._to_tensor(frame, self._device)
                    if not initialized:
                        out = proc.step(img_t, init_t, objects=objects)
                        initialized = True
                    else:
                        out = proc.step(img_t)
                    labeled = proc.output_prob_to_mask(out).cpu().numpy()
                    self.mask_ready.emit(fi, labeled)
                    done += 1
                    if done % 50 == 0:
                        self.progress.emit(done, total)
        finally:
            cap.release()

        log.info("CutieWorker: forward done — %d frames", done)

    # ------------------------------------------------------------------
    # Backward pass: init_frame → first_frame
    # ------------------------------------------------------------------

    def _run_backward(self) -> None:
        import torch

        if self._init_frame <= self._first:
            log.info("CutieWorker: nothing to track backward (init == first frame)")
            return

        log.info("CutieWorker: backward  [%d, %d]  init=%d",
                 self._first, self._init_frame - 1, self._init_frame)
        model = self._load_cutie()
        proc = self._new_processor(model)
        objects = list(range(1, len(self._persons) + 1))
        init_t = torch.from_numpy(self._init_mask).to(self._device)

        # Read backward-range frames into memory first (sequential read is faster).
        bwd_frames: list[tuple[int, np.ndarray]] = []
        cap = cv2.VideoCapture(self._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._first)
        for fi in range(self._first, self._init_frame):
            ret, frame = cap.read()
            if not ret:
                break
            bwd_frames.append((fi, frame))
        cap.release()

        total = len(bwd_frames)
        done = 0

        cap2 = cv2.VideoCapture(self._video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, self._init_frame)
        ret, init_img = cap2.read()
        cap2.release()
        if not ret:
            self.error.emit(f"Cannot read init frame {self._init_frame}")
            return

        try:
            with torch.inference_mode(), torch.amp.autocast(self._device):
                # Seed with the init frame mask.
                img_t = self._to_tensor(init_img, self._device)
                proc.step(img_t, init_t, objects=objects)

                for fi, frame in reversed(bwd_frames):
                    if self._stop_requested:
                        break
                    img_t = self._to_tensor(frame, self._device)
                    out = proc.step(img_t)
                    labeled = proc.output_prob_to_mask(out).cpu().numpy()
                    self.mask_ready.emit(fi, labeled)
                    done += 1
                    if done % 50 == 0:
                        self.progress.emit(done, total)
        finally:
            pass

        log.info("CutieWorker: backward done — %d frames", done)
