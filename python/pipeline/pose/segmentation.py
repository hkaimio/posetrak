# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""segmentation.py — Cutie-based per-person segmentation for keypoint quality scoring.

This module is part of the pipeline described in
docs/segmentation-keypoint-weighting-design.md.

Overview
--------
``CutieSegmentor`` wraps Cutie (XMem++ successor) for video object segmentation.
It accepts a video path, a single initialization frame, and per-person bounding
boxes (or a pre-built labeled mask for manual workflows), then propagates masks
**bidirectionally** — forward to end_frame and backward to start_frame — so that
the full clip is covered from a single init point.

Quality scores
--------------
The quality score is a float32 value in ``{1.0, 0.5, 0.0, -1.0}``:

  1.0  — keypoint is clearly inside the person mask (after erosion)
  0.5  — keypoint is in the eroded boundary zone — uncertain; partial inflation
  0.0  — keypoint is outside the person mask
 -1.0  — sentinel: no mask data available for this frame/person

Initialization
--------------
Two modes are supported:

**Automatic (default):**
  rtmlib's YOLOX detects persons in the init frame; SAM2 generates per-person
  masks from those bboxes; the masks are merged into a Cutie-format labeled
  init mask.  Requires ``rtmlib`` and ``sam2`` to be installed.

**Manual:**
  The caller provides a pre-built ``(H, W)`` uint8 labeled mask directly via
  the ``init_mask`` parameter.  Pixel value 0 = background; value *k* = the
  *k*-th person in the ``persons`` dict (insertion order, 1-indexed).  This
  supports the UI workflow where the user clicks on each person once — faster
  than stitching and eliminates the need for the detector or SAM2 at runtime.

Typical usage — automatic init
-------------------------------
::

    from pipeline.pose.segmentation import CutieSegmentor

    persons = {
        "Harri": np.array([554, 194, 754, 748]),   # bbox xyxy
        "Tommi": np.array([783, 105, 1080, 779]),
    }

    seg = CutieSegmentor(device="cuda")
    seg.process_video(
        "path/to/video.mp4",
        init_frame=270,
        persons=persons,
    )

    scores = seg.get_keypoint_scores(300, "Harri", keypoints_xy)  # (133,)

Typical usage — manual init
----------------------------
::

    # UI supplies a labeled uint8 (H, W) mask for frame 270
    seg.process_video(
        "path/to/video.mp4",
        init_frame=270,
        persons={"Harri": ..., "Tommi": ...},
        init_mask=labeled_mask_from_ui,   # bypasses YOLO + SAM
    )

Cutie dependency
----------------
Cutie is expected at the path given by the ``CUTIE_DIR`` environment variable,
or auto-detected relative to this project at ``../../tests/Cutie`` (i.e.
``/home/harri/projects/tests/Cutie`` by default).  It must be a clone of
https://github.com/hkchengrex/Cutie with ``cutie-base-mega.pth`` in
``weights/``.  The Cutie venv (``<CUTIE_DIR>/venv/``) must have
``sam2`` and ``rtmlib`` installed if automatic init is used.

Legacy class
------------
``SAM2Segmentor`` (SAM2VideoPredictor-based) is retained at the bottom of this
file for reference.  It is superseded by ``CutieSegmentor`` and should not be
used in new code.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cutie directory resolution
# ---------------------------------------------------------------------------

def _find_cutie_dir() -> Path:
    """Return the Cutie project directory.

    Search order:
    1. ``CUTIE_DIR`` environment variable (explicit override)
    2. Platform data directory: ``%LOCALAPPDATA%\\posetrak\\Cutie`` (Windows)
       or ``~/.local/share/posetrak/Cutie`` (Linux/macOS)
    3. Legacy fallback: ``<project-root>/../tests/Cutie``
    """
    import platform

    env = os.environ.get("CUTIE_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
        raise EnvironmentError(f"CUTIE_DIR={env!r} is not a directory")

    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path.home() / ".local" / "share"
    platform_candidate = base / "posetrak" / "Cutie"
    if platform_candidate.is_dir():
        return platform_candidate

    # Legacy: sibling tests/Cutie next to project root
    project_root = Path(__file__).parents[3]
    legacy_candidate = project_root.parent / "tests" / "Cutie"
    if legacy_candidate.is_dir():
        return legacy_candidate

    if platform.system() == "Windows":
        install_dir = platform_candidate
        install_cmd = f'git clone https://github.com/hkchengrex/Cutie "{install_dir}"'
    else:
        install_dir = platform_candidate
        install_cmd = f"git clone https://github.com/hkchengrex/Cutie {install_dir}"

    raise EnvironmentError(
        f"Cannot find Cutie (https://github.com/hkchengrex/Cutie).\n"
        f"Install it to the default location:\n\n"
        f"    {install_cmd}\n\n"
        f"Or set CUTIE_DIR to the path of an existing clone."
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Score returned when a keypoint is clearly inside the (eroded) person mask.
SCORE_INSIDE: float = 1.0

#: Score returned when a keypoint is in the boundary zone.
SCORE_BOUNDARY: float = 0.5

#: Score returned when a keypoint is outside the person mask.
SCORE_OUTSIDE: float = 0.0

#: Sentinel value meaning "no mask data available for this frame/person".
SCORE_UNAVAILABLE: float = -1.0

#: Number of RTMPose-133 wholebody keypoints stored per frame.
N_KEYPOINTS: int = 133


# ---------------------------------------------------------------------------
# Blob encode / decode helpers  (unchanged from SAM2Segmentor)
# ---------------------------------------------------------------------------

def encode_scores(scores: np.ndarray) -> bytes:
    """Encode an (N,) float32 array as little-endian bytes for DB storage."""
    return np.asarray(scores, dtype="<f4").tobytes()


def decode_scores(blob: bytes, n: int = N_KEYPOINTS) -> np.ndarray:
    """Decode little-endian float32 bytes back to an (n,) float32 array."""
    return np.frombuffer(blob, dtype="<f4").copy()


# ---------------------------------------------------------------------------
# CutieSegmentor
# ---------------------------------------------------------------------------

class CutieSegmentor:
    """Tracks persons through a video clip using Cutie (XMem++) segmentation.

    Bidirectional propagation from a single initialization frame:

    - **Forward pass**: ``init_frame`` → ``end_frame``
    - **Backward pass**: ``init_frame`` → ``start_frame`` (frames fed in
      reverse temporal order to a fresh Cutie InferenceCore)

    Both passes are seeded from the same labeled init mask, giving complete
    coverage of the clip without UKF warm-up penalties at either end.

    Parameters
    ----------
    device:
        Torch device string: ``"cuda"``, ``"cuda:1"``, ``"cpu"``.
    max_internal_size:
        Cutie's internal processing resolution (shorter edge in pixels).
        480 is a good balance; reduce to 360 for a ~30 % speedup at some
        quality cost.
    erosion_px:
        Pixels to erode from the binary mask before scoring keypoints.
        Points in the eroded zone receive ``SCORE_BOUNDARY`` rather than
        ``SCORE_INSIDE``.  Use 3–5 px for typical 1080p footage.
    """

    def __init__(
        self,
        device: str = "cuda",
        max_internal_size: int = 480,
        erosion_px: int = 5,
    ) -> None:
        self._device = device
        self._max_internal_size = max_internal_size
        self._erosion_px = erosion_px

        # frame_idx → person_id → (H, W) bool mask
        self._masks: dict[int, dict[str, np.ndarray]] = {}
        self._person_ids: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        init_frame: int,
        persons: dict[str, np.ndarray],
        init_mask: np.ndarray | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Run Cutie bidirectionally and store per-frame masks.

        Parameters
        ----------
        video_path:
            Path to the video file.
        init_frame:
            Frame index (0-based) used to seed Cutie.  Should be a "clean"
            frame where all persons are clearly visible and well-separated.
        persons:
            Ordered mapping from person ID to ``(4,)`` xyxy bbox in video
            pixels.  Used for automatic detector+SAM2 init when ``init_mask`` is
            ``None``.  When ``init_mask`` is provided, the dict keys define
            the person IDs and their order must match mask label indices
            (first key → label 1, second key → label 2, …).
        init_mask:
            Optional pre-built ``(H, W)`` uint8 labeled mask for
            ``init_frame``.  Pixel value 0 = background; value *k* = the
            *k*-th entry in ``persons`` (1-indexed).  Pass this from the UI
            when the user has already annotated the init frame; skips YOLO
            and SAM entirely.
        start_frame:
            First frame of the segment (0-based, inclusive).
        end_frame:
            Last frame of the segment (exclusive).  ``None`` = end of video.
        verbose:
            Print per-300-frame progress.
        """
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        cutie_dir = _find_cutie_dir()
        if str(cutie_dir) not in sys.path:
            sys.path.insert(0, str(cutie_dir))

        from cutie.inference.inference_core import InferenceCore
        from cutie.utils.get_default_model import get_default_model

        video_path = Path(video_path)
        self._person_ids = list(persons.keys())
        n = len(self._person_ids)
        objects_list = list(range(1, n + 1))

        # ── Video metadata ────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if end_frame is None:
            end_frame = total

        logger.info(
            "CutieSegmentor: %s  segment [%d, %d)  init_frame=%d  %d persons",
            video_path.name, start_frame, end_frame, init_frame, n,
        )

        # ── Load Cutie model ──────────────────────────────────────────────
        logger.debug("Loading Cutie model…")
        cutie_model = get_default_model()

        # ── Build or validate init mask ───────────────────────────────────
        if init_mask is None:
            logger.debug("Building init mask via YOLO + SAM…")
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
            ret, init_frame_img = cap.read()
            cap.release()
            if not ret:
                raise ValueError(
                    f"Cannot read init frame {init_frame} from {video_path}"
                )
            init_mask = self._build_init_mask(
                init_frame_img, persons, video_path
            )
        else:
            init_mask = np.asarray(init_mask, dtype=np.uint8)

        init_mask_tensor = torch.from_numpy(init_mask).to(self._device)

        # ── Helper: frame → tensor ────────────────────────────────────────
        def frame_to_tensor(bgr: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return to_tensor(Image.fromarray(rgb)).to(self._device).float()

        # ── Helper: labeled mask → per-person bool masks ──────────────────
        def store_labeled(fi: int, labeled: np.ndarray) -> None:
            frame_masks: dict[str, np.ndarray] = {}
            for idx, pid in enumerate(self._person_ids):
                m = labeled == (idx + 1)
                if m.any():
                    frame_masks[pid] = m
            if frame_masks:
                self._masks[fi] = frame_masks

        # ── Forward pass: init_frame → end_frame ─────────────────────────
        logger.info("CutieSegmentor: forward pass [%d, %d)…", init_frame, end_frame)
        fwd_processor = InferenceCore(cutie_model, cfg=cutie_model.cfg)
        fwd_processor.max_internal_size = self._max_internal_size

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
        initialized = False

        with torch.inference_mode(), torch.amp.autocast(self._device):
            for fi in range(init_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                img_t = frame_to_tensor(frame)

                if not initialized:
                    out = fwd_processor.step(img_t, init_mask_tensor,
                                             objects=objects_list)
                    initialized = True
                else:
                    out = fwd_processor.step(img_t)

                labeled = fwd_processor.output_prob_to_mask(out).cpu().numpy()
                store_labeled(fi, labeled)

                if verbose and fi % 300 == 0 and fi > init_frame:
                    logger.info("  fwd %d / %d", fi, end_frame)

        cap.release()
        logger.info("  forward pass done — %d frames stored", end_frame - init_frame)

        # ── Backward pass: init_frame-1 → start_frame ────────────────────
        if init_frame > start_frame:
            logger.info(
                "CutieSegmentor: backward pass [%d, %d)…",
                start_frame, init_frame,
            )
            bwd_processor = InferenceCore(cutie_model, cfg=cutie_model.cfg)
            bwd_processor.max_internal_size = self._max_internal_size

            # Read the backward-range frames into memory, then iterate reversed.
            # For typical init_frame ~ a few hundred this is small (<1 GB).
            logger.debug(
                "  reading %d frames for backward pass…",
                init_frame - start_frame,
            )
            bwd_frames: list[tuple[int, np.ndarray]] = []
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for fi in range(start_frame, init_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                bwd_frames.append((fi, frame))
            cap.release()

            with torch.inference_mode(), torch.amp.autocast(self._device):
                # Seed backward processor with init_mask (same as forward)
                cap2 = cv2.VideoCapture(str(video_path))
                cap2.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
                ret, init_frame_img = cap2.read()
                cap2.release()

                img_t = frame_to_tensor(init_frame_img)
                bwd_processor.step(img_t, init_mask_tensor, objects=objects_list)

                # Process backward range in reverse temporal order
                for fi, frame in reversed(bwd_frames):
                    img_t = frame_to_tensor(frame)
                    out = bwd_processor.step(img_t)
                    labeled = bwd_processor.output_prob_to_mask(out).cpu().numpy()
                    store_labeled(fi, labeled)

                    if verbose and fi % 300 == 0:
                        logger.info("  bwd %d / %d", fi, start_frame)

            logger.info(
                "  backward pass done — %d frames stored",
                init_frame - start_frame,
            )

        logger.info(
            "CutieSegmentor: total masks stored: %d frames", len(self._masks)
        )

    # ------------------------------------------------------------------

    def get_mask(self, frame_idx: int, person_id: str) -> np.ndarray | None:
        """Return the ``(H, W)`` bool mask for a person at a given frame.

        Returns ``None`` if no data is available.
        """
        frame_data = self._masks.get(frame_idx)
        if frame_data is None:
            return None
        return frame_data.get(person_id)

    def get_keypoint_scores(
        self,
        frame_idx: int,
        person_id: str,
        keypoints_xy: np.ndarray,
        erosion_px: int | None = None,
    ) -> np.ndarray:
        """Compute per-keypoint quality scores for a person at a frame.

        Parameters
        ----------
        frame_idx:
            Absolute frame index in the video file.
        person_id:
            Person identifier as passed to :meth:`process_video`.
        keypoints_xy:
            ``(N, 2)`` float array of pixel coordinates ``[x, y]``.
        erosion_px:
            Override the instance-level ``erosion_px``.

        Returns
        -------
        float32 array of shape ``(N,)``.
        Values: ``SCORE_INSIDE``, ``SCORE_BOUNDARY``, ``SCORE_OUTSIDE``,
        or ``SCORE_UNAVAILABLE``.
        """
        erosion_px = erosion_px if erosion_px is not None else self._erosion_px
        mask = self.get_mask(frame_idx, person_id)
        n = len(keypoints_xy)

        if mask is None:
            return np.full(n, SCORE_UNAVAILABLE, dtype=np.float32)

        return _score_keypoints(mask, keypoints_xy, erosion_px)

    def get_all_scores_for_frame(
        self,
        frame_idx: int,
        keypoints_per_person: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Score all persons in one frame.

        Parameters
        ----------
        frame_idx:
            Absolute frame index.
        keypoints_per_person:
            Mapping from person_id to ``(N, 2)`` keypoint array.

        Returns
        -------
        Mapping from person_id to float32 score array of shape ``(N,)``.
        """
        return {
            pid: self.get_keypoint_scores(frame_idx, pid, kpts)
            for pid, kpts in keypoints_per_person.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_init_mask(
        self,
        frame: np.ndarray,
        persons: dict[str, np.ndarray],
        video_path: Path,
    ) -> np.ndarray:
        """Run a detector + SAM2 on *frame* to produce a labeled init mask.

        Persons are ordered by x-centre of their detected bbox so that
        left-to-right order is preserved regardless of detector output order.
        The ``persons`` dict provides a fallback: if the detector finds fewer
        persons than the dict has entries, the provided bboxes are used
        directly for the missing ones.

        Returns
        -------
        ``(H, W)`` uint8 array, values 0..N (0 = background).
        """
        try:
            from posetrak.detection.backends_rtmdet import _KNOWN_MODELS
            from rtmlib.tools.object_detection import YOLOX
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "rtmlib and sam2 are required for automatic init.  "
                "Install with: pip install rtmlib sam2\n"
                "Alternatively, pass init_mask= for manual initialisation."
            ) from exc

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        h, w = frame.shape[:2]
        n_persons = len(persons)

        # Detector: single-frame person detection, no tracking needed here.
        url, input_size = _KNOWN_MODELS["yolox-x"]
        yolox = YOLOX(
            url, model_input_size=input_size, mode="human",
            score_thr=0.30, backend="onnxruntime", device=device,
        )
        boxes_xyxy = yolox(frame)

        if boxes_xyxy is not None and len(boxes_xyxy) >= n_persons:
            bboxes = np.asarray(boxes_xyxy, dtype=float)
            x_centers = (bboxes[:, 0] + bboxes[:, 2]) / 2
            bboxes = bboxes[np.argsort(x_centers)[:n_persons]]
        else:
            # Fall back to the bboxes supplied by the caller
            logger.warning(
                "Detector found fewer than %d persons; using provided bboxes",
                n_persons,
            )
            bboxes = np.array(list(persons.values()), dtype=float)

        # SAM2 single-frame masks, one box prompt at a time (SAM2ImagePredictor
        # has no batched-box API); the image is only encoded once, reused for
        # every box.
        sam_model = build_sam2_hf("facebook/sam2.1-hiera-base-plus", device=device)
        predictor = SAM2ImagePredictor(sam_model)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        init_mask = np.zeros((h, w), dtype=np.uint8)
        for j, box in enumerate(bboxes[:n_persons]):
            masks, _scores, _logits = predictor.predict(
                box=np.asarray(box, dtype=np.float32), multimask_output=False,
            )
            m = masks[0] > 0.5
            if m.shape != (h, w):
                m = cv2.resize(
                    m.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            init_mask[m] = j + 1   # higher index overwrites in overlap zone

        return init_mask


# ---------------------------------------------------------------------------
# Shared scoring helper (used by both CutieSegmentor and SAM2Segmentor)
# ---------------------------------------------------------------------------

def _score_keypoints(
    mask: np.ndarray,
    keypoints_xy: np.ndarray,
    erosion_px: int,
) -> np.ndarray:
    """Core keypoint scoring logic (erosion-based boundary zone).

    Parameters
    ----------
    mask: (H, W) bool
    keypoints_xy: (N, 2) float [x, y]
    erosion_px: erosion kernel radius in pixels

    Returns
    -------
    float32 (N,) with values SCORE_INSIDE / SCORE_BOUNDARY / SCORE_OUTSIDE /
    SCORE_UNAVAILABLE.
    """
    h, w = mask.shape
    n = len(keypoints_xy)
    scores = np.full(n, SCORE_OUTSIDE, dtype=np.float32)

    if erosion_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
        )
        mask_eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    else:
        mask_eroded = mask

    for i, (x, y) in enumerate(keypoints_xy):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            scores[i] = SCORE_UNAVAILABLE
        elif mask_eroded[yi, xi]:
            scores[i] = SCORE_INSIDE
        elif mask[yi, xi]:
            scores[i] = SCORE_BOUNDARY
        # else stays SCORE_OUTSIDE

    return scores
