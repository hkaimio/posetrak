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
import time

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

    # Emitted in batches rather than per-frame to avoid flooding the main thread's
    # event queue.  Each payload is a list of (frame_idx, png_bytes) tuples.
    mask_ready    = Signal(object)     # list[tuple[int, bytes]]
    progress      = Signal(int, int)   # frames_done, total_frames
    # Named tracking_done (not "finished") to avoid shadowing ambiguity with
    # QThread::finished, which Qt emits via its own internal mechanism after
    # run() returns.  Using the same name risks the connection binding to
    # QThread::finished, which is posted to the event queue through a different
    # path and can arrive before our batch signals.
    tracking_done = Signal()
    error         = Signal(str)

    def __init__(
        self,
        video_path: str,
        init_frame: int,
        init_mask: np.ndarray,          # (H, W) uint8 labeled mask at max_dim resolution
        persons_ordered: list[str],     # label 1..N → person name (for logging)
        first_frame: int,               # inclusive track range start
        last_frame: int,                # inclusive track range end
        direction: str,                 # "forward" or "backward"
        device: str = "cuda",
        max_internal_size: int = 480,
        max_dim: int = 1920,            # downscale video to this before Cutie; must match FrameCache
        model=None,                     # pre-loaded Cutie model; if None, loaded in thread
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
        self._max_dim = max_dim
        self._preloaded_model = model   # provided by runner on 2nd+ jobs
        self._loaded_model = None       # set during run(); retrieved by runner after finish
        self._stop_requested = False

    def get_loaded_model(self):
        """Return the Cutie model after run() has completed (None if not loaded)."""
        return self._loaded_model or self._preloaded_model

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
            log.info("CutieWorker: emitting tracking_done  t=%.3f", time.monotonic())
            self.tracking_done.emit()

    # ------------------------------------------------------------------
    # Internal — Cutie setup
    # ------------------------------------------------------------------

    def _load_cutie(self):
        """Return the Cutie model, using the pre-loaded one if available.

        Loading the model calls hydra.initialize() which is not safe to call
        from a background thread more than once per process.  The runner
        caches the model after the first job and passes it here so that Hydra
        is only ever initialised once.
        """
        if self._preloaded_model is not None:
            return self._preloaded_model

        import sys
        from pipeline.pose.segmentation import _find_cutie_dir
        cutie_dir = _find_cutie_dir()
        if str(cutie_dir) not in sys.path:
            sys.path.insert(0, str(cutie_dir))

        try:
            from hydra.core.global_hydra import GlobalHydra
            GlobalHydra.instance().clear()
        except Exception:
            pass

        from cutie.utils.get_default_model import get_default_model
        model = get_default_model()
        self._loaded_model = model   # runner retrieves this after finish
        return model

    def _new_processor(self, model):
        from cutie.inference.inference_core import InferenceCore
        proc = InferenceCore(model, cfg=model.cfg)
        proc.max_internal_size = self._max_internal_size
        return proc

    @staticmethod
    def _to_tensor(bgr: np.ndarray, device: str, max_dim: int = 0):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        if max_dim > 0:
            h, w = bgr.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return to_tensor(Image.fromarray(rgb)).to(device).float()

    # ------------------------------------------------------------------
    # Forward pass: init_frame → last_frame
    # ------------------------------------------------------------------

    # Number of frames to accumulate before emitting a batch signal.
    # Fewer, larger signals prevent flooding the main thread's event queue.
    _BATCH = 50

    def _run_forward(self) -> None:
        import torch

        t0 = time.monotonic()
        log.info("CutieWorker: forward  [%d, %d]  init=%d  t=%.3f",
                 self._init_frame, self._last, self._init_frame, t0)
        model = self._load_cutie()
        log.info("CutieWorker: model ready  t=%.3f  (%.2fs to load)",
                 time.monotonic(), time.monotonic() - t0)
        proc = self._new_processor(model)
        objects = list(range(1, len(self._persons) + 1))
        init_t = torch.from_numpy(self._init_mask).to(self._device)

        total = self._last - self._init_frame + 1
        done = 0
        batch_num = 0
        batch: list[tuple[int, bytes]] = []

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
                    img_t = self._to_tensor(frame, self._device, self._max_dim)
                    del frame
                    if not initialized:
                        out = proc.step(img_t, init_t, objects=objects)
                        initialized = True
                    else:
                        out = proc.step(img_t)
                    del img_t
                    labeled = proc.output_prob_to_mask(out).cpu().numpy().astype(np.uint8)
                    del out
                    ok, png_buf = cv2.imencode(".png", labeled)
                    del labeled
                    if ok:
                        batch.append((fi, bytes(png_buf)))
                    done += 1
                    if len(batch) >= self._BATCH:
                        batch_num += 1
                        log.debug("CutieWorker: emitting batch %d  frames=%d-%d  done=%d/%d  t=%.3f",
                                  batch_num, batch[0][0], batch[-1][0], done, total, time.monotonic() - t0)
                        self.mask_ready.emit(batch)
                        batch = []
                        self.progress.emit(done, total)
            if batch:
                batch_num += 1
                log.debug("CutieWorker: emitting final batch %d  frames=%d-%d  done=%d/%d  t=%.3f",
                          batch_num, batch[0][0], batch[-1][0], done, total, time.monotonic() - t0)
                self.mask_ready.emit(batch)
                batch = []
        finally:
            cap.release()

        log.info("CutieWorker: inference done — %d frames in %.2fs  (%d batches)  t=%.3f",
                 done, time.monotonic() - t0, batch_num, time.monotonic())

        # Explicitly free InferenceCore and flush CUDA memory before thread exits so the
        # next job starts with a clean GPU state and doesn't double-count live tensors.
        del proc
        try:
            if torch.cuda.is_available():
                log.debug("CutieWorker: CUDA sync+empty_cache  t=%.3f", time.monotonic() - t0)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        log.info("CutieWorker: forward complete — %d frames  total=%.2fs", done, time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Backward pass: init_frame → first_frame
    # ------------------------------------------------------------------

    def _run_backward(self) -> None:
        import torch

        if self._init_frame <= self._first:
            log.info("CutieWorker: nothing to track backward (init == first frame)")
            return

        t0 = time.monotonic()
        log.info("CutieWorker: backward  [%d, %d]  init=%d  t=%.3f",
                 self._first, self._init_frame - 1, self._init_frame, t0)
        model = self._load_cutie()
        proc = self._new_processor(model)
        objects = list(range(1, len(self._persons) + 1))
        init_t = torch.from_numpy(self._init_mask).to(self._device)

        # Read backward-range frames into memory, scale + JPEG-compress immediately.
        # Raw 4K ndarray per frame is ~24 MB; JPEG at 1920p is ~150 KB — 100× smaller.
        # Without this, a 1700-frame backward pass on a 4K gopro would consume ~40 GB.
        # Use a deque so frames are popped and freed as they are processed (right-to-left
        # gives highest frame first, matching the reverse-chronological processing order).
        from collections import deque
        bwd_deque: deque[tuple[int, bytes]] = deque()
        cap = cv2.VideoCapture(self._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._first)
        for fi in range(self._first, self._init_frame):
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if self._max_dim > 0 and max(h, w) > self._max_dim:
                scale = self._max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            del frame
            if ok:
                bwd_deque.append((fi, buf.tobytes()))
        cap.release()
        n_bwd = len(bwd_deque)
        log.info("CutieWorker: backward buffer — %d frames, ~%.0f MB",
                 n_bwd, n_bwd * 150 / 1024)

        total = n_bwd
        done = 0

        cap2 = cv2.VideoCapture(self._video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, self._init_frame)
        ret, init_img = cap2.read()
        cap2.release()
        if not ret:
            self.error.emit(f"Cannot read init frame {self._init_frame}")
            return

        batch: list[tuple[int, bytes]] = []
        batch_num = 0

        with torch.inference_mode(), torch.amp.autocast(self._device):
            # Seed with the init frame mask.
            img_t = self._to_tensor(init_img, self._device, self._max_dim)
            del init_img
            proc.step(img_t, init_t, objects=objects)
            del img_t

            while bwd_deque:
                if self._stop_requested:
                    break
                fi, jpeg_bytes = bwd_deque.pop()   # pop from right = highest frame first
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                del jpeg_bytes
                img_t = self._to_tensor(frame, self._device, self._max_dim)
                del frame
                out = proc.step(img_t)
                del img_t
                labeled = proc.output_prob_to_mask(out).cpu().numpy().astype(np.uint8)
                del out
                ok, png_buf = cv2.imencode(".png", labeled)
                del labeled
                if ok:
                    batch.append((fi, bytes(png_buf)))
                done += 1
                if len(batch) >= self._BATCH:
                    batch_num += 1
                    log.debug("CutieWorker: emitting batch %d  frames=%d-%d  done=%d/%d  t=%.3f",
                              batch_num, batch[0][0], batch[-1][0], done, total, time.monotonic() - t0)
                    self.mask_ready.emit(batch)
                    batch = []
                    self.progress.emit(done, total)
        if batch:
            batch_num += 1
            log.debug("CutieWorker: emitting final batch %d  frames=%d-%d  done=%d/%d  t=%.3f",
                      batch_num, batch[0][0], batch[-1][0], done, total, time.monotonic() - t0)
            self.mask_ready.emit(batch)
            batch = []

        log.info("CutieWorker: inference done — %d frames in %.2fs  (%d batches)  t=%.3f",
                 done, time.monotonic() - t0, batch_num, time.monotonic())

        # Explicitly free InferenceCore and flush CUDA memory before thread exits.
        del proc
        del bwd_deque
        try:
            if torch.cuda.is_available():
                log.debug("CutieWorker: CUDA sync+empty_cache  t=%.3f", time.monotonic() - t0)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        log.info("CutieWorker: backward complete — %d frames  total=%.2fs", done, time.monotonic() - t0)
