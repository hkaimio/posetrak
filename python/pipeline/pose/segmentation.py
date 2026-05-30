"""segmentation.py — SAM2-based per-person segmentation for keypoint quality scoring.

This module is part of the pipeline described in
docs/segmentation-keypoint-weighting-design.md.

Overview
--------
``SAM2Segmentor`` wraps the Ultralytics SAM2VideoPredictor.  It accepts a video
path and initial per-person bounding boxes, streams SAM2 through the clip, and
returns a per-keypoint quality score (float32 in [0, 1]) for every frame.

The quality score is:
  1.0  — keypoint is clearly inside the person mask (after erosion)
  0.5  — keypoint is in the eroded boundary zone (inside raw mask but outside
          eroded mask) — uncertain; partial inflation
  0.0  — keypoint is outside the person mask

A score of -1.0 is the sentinel "no data available for this frame/person".

Typical usage
-------------
::

    from pipeline.pose.segmentation import SAM2Segmentor

    # one entry per person: person_id -> (init_frame_idx, bbox_xyxy)
    persons = {
        "Harri": (0, np.array([554, 194, 754, 748])),
        "Tommi": (0, np.array([783, 105, 1080, 779])),
    }

    seg = SAM2Segmentor(model_name="sam2.1_b.pt", device="cuda")
    seg.process_video("path/to/video.mp4", persons)

    # After processing, query any frame
    scores = seg.get_keypoint_scores(42, "Harri", keypoints_xy)  # (133,)

Re-initialisation
-----------------
SAM2 is initialised once per call to ``process_video``.  For multi-segment
workflows, call ``process_video`` again with a different start frame and bbox.
Scores from multiple calls are merged into the same internal store.

Limitations (prototype)
-----------------------
* Full video is processed in one streaming pass — no mid-video re-init yet.
* Masks are binary (Ultralytics SAM2 postprocessing thresholds the logits).
  A three-level score (1.0 / 0.5 / 0.0) approximates the probability gradient
  at the silhouette boundary via the erosion zone.
* Only forward propagation is supported.  RTS-style backward pass is a TODO.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Score returned when a keypoint is clearly inside the (eroded) person mask.
SCORE_INSIDE: float = 1.0

#: Score returned when a keypoint is in the boundary zone (inside raw mask but
#: outside eroded mask).  Triggers partial inflation.
SCORE_BOUNDARY: float = 0.5

#: Score returned when a keypoint is outside the person mask.
SCORE_OUTSIDE: float = 0.0

#: Sentinel value meaning "no SAM2 data available for this frame/person".
SCORE_UNAVAILABLE: float = -1.0

#: Number of RTMPose-133 wholebody keypoints stored per frame.
N_KEYPOINTS: int = 133


# ---------------------------------------------------------------------------
# Blob encode / decode helpers
# ---------------------------------------------------------------------------

def encode_scores(scores: np.ndarray) -> bytes:
    """Encode an (N,) float32 array as little-endian bytes for DB storage.

    Args:
        scores: 1-D float32 array of length N_KEYPOINTS.

    Returns:
        Raw bytes (N * 4 bytes, little-endian float32).
    """
    arr = np.asarray(scores, dtype="<f4")  # little-endian float32
    return arr.tobytes()


def decode_scores(blob: bytes, n: int = N_KEYPOINTS) -> np.ndarray:
    """Decode little-endian float32 bytes back to an (n,) float32 array.

    Args:
        blob: Raw bytes from the DB.
        n: Expected number of values (default: N_KEYPOINTS = 133).

    Returns:
        float32 array of shape (n,).
    """
    return np.frombuffer(blob, dtype="<f4").copy()


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class SAM2Segmentor:
    """Tracks persons through a video clip using SAM2 and scores keypoints.

    Parameters
    ----------
    model_name:
        Ultralytics model weight filename, e.g. ``"sam2.1_b.pt"`` (fast) or
        ``"sam2.1_l.pt"`` (accurate).  Auto-downloaded on first use.
    device:
        Torch device string: ``"cuda"``, ``"cuda:1"``, ``"cpu"``.
    erosion_px:
        Pixels to erode from the binary mask before scoring keypoints.  Points
        in the eroded zone receive ``SCORE_BOUNDARY`` rather than
        ``SCORE_INSIDE``, signalling silhouette-boundary uncertainty.
    reinit_interval:
        Re-initialise SAM2 tracking from the current YOLO bbox every N frames.
        0 disables periodic re-init (only segment-boundary re-init applies).
        Not yet implemented — placeholder for future work.
    """

    def __init__(
        self,
        model_name: str = "sam2.1_b.pt",
        device: str = "cuda",
        erosion_px: int = 5,
        reinit_interval: int = 60,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._erosion_px = erosion_px
        self._reinit_interval = reinit_interval

        # Stores: frame_idx -> person_id -> (H, W) bool mask (raw, not eroded)
        self._masks: dict[int, dict[str, np.ndarray]] = {}

        # Ordered list of person IDs, matching SAM2 object indices
        self._person_ids: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        persons: dict[str, tuple[int, np.ndarray]],
        start_frame: int = 0,
        end_frame: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Run SAM2 on a video segment and store per-frame masks.

        Existing masks in ``self._masks`` are preserved; new frames are added.
        Call this method once per tracking segment (each segment corresponds to
        a contiguous assignment in the stitcher).

        Parameters
        ----------
        video_path:
            Path to the video file.
        persons:
            Mapping from person ID to ``(init_frame_relative, bbox_xyxy)``.
            ``init_frame_relative`` is the frame index *within the segment*
            (usually 0) where the bbox is valid.  ``bbox_xyxy`` is an
            ``(4,)`` float array ``[x1, y1, x2, y2]`` in original video
            pixels.
        start_frame:
            First frame of the segment (0-based, inclusive) in video-file
            coordinates.  Frames before this are skipped.
        end_frame:
            Last frame of the segment (exclusive).  ``None`` means process to
            end of video.
        verbose:
            Print progress every 100 frames.
        """
        try:
            from ultralytics.models.sam import SAM2VideoPredictor
        except ImportError as exc:
            raise ImportError(
                "ultralytics >= 8.3 is required for SAM2.  "
                "Install with: pip install ultralytics"
            ) from exc

        video_path = str(video_path)
        self._person_ids = list(persons.keys())

        # Build the initial bbox array in the order of self._person_ids
        # (object index 0..N-1 in SAM2 matches this order)
        init_bboxes = np.array(
            [persons[pid][1] for pid in self._person_ids], dtype=float
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if end_frame is None:
            end_frame = total_frames

        logger.info(
            "SAM2: processing %s frames [%d, %d) for %d persons",
            Path(video_path).name, start_frame, end_frame, len(self._person_ids),
        )

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=1024,
            model=self._model_name,
            save=False,
            verbose=verbose,
            device=self._device,
        )
        predictor = SAM2VideoPredictor(overrides=overrides)

        # SAM2VideoPredictor requires the full video as source.
        # We seek to start_frame using vid_stride trick: not straightforward,
        # so for the prototype we stream from 0 and skip early frames.
        # TODO: use av or ffmpeg-based seeking for large offsets.
        abs_frame = 0
        for result in predictor(
            source=video_path,
            bboxes=init_bboxes,
            stream=True,
        ):
            if abs_frame < start_frame:
                abs_frame += 1
                continue
            if abs_frame >= end_frame:
                break

            if verbose and abs_frame % 100 == 0:
                logger.info("  SAM2 frame %d / %d", abs_frame, end_frame)

            if result.masks is not None and len(result.masks) > 0:
                masks_hw = result.masks.data.cpu().numpy()  # (N, H, W) bool
                frame_masks: dict[str, np.ndarray] = {}
                for obj_idx, pid in enumerate(self._person_ids):
                    if obj_idx < len(masks_hw):
                        frame_masks[pid] = masks_hw[obj_idx].astype(bool)
                self._masks[abs_frame] = frame_masks

            abs_frame += 1

        logger.info(
            "SAM2: stored masks for %d frames", len(self._masks)
        )

    def get_mask(self, frame_idx: int, person_id: str) -> np.ndarray | None:
        """Return the (H, W) bool mask for a person at a given frame.

        Parameters
        ----------
        frame_idx:
            Absolute frame index in the video file.
        person_id:
            Person identifier as passed to :meth:`process_video`.

        Returns
        -------
        bool array of shape ``(H, W)``, or ``None`` if no data is available.
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

        For each keypoint, checks whether its pixel coordinate falls inside the
        SAM2 person mask.  The mask is eroded by ``erosion_px`` pixels first to
        flag silhouette-boundary points as uncertain.

        Parameters
        ----------
        frame_idx:
            Absolute frame index.
        person_id:
            Person identifier.
        keypoints_xy:
            ``(N, 2)`` float array of pixel coordinates ``[x, y]`` in the
            original (undistorted) video frame.  Typically the 133 RTMPose
            wholebody keypoints.
        erosion_px:
            Override the instance-level ``erosion_px``.

        Returns
        -------
        float32 array of shape ``(N,)``.
        Values: ``SCORE_INSIDE`` (1.0), ``SCORE_BOUNDARY`` (0.5),
        ``SCORE_OUTSIDE`` (0.0), or ``SCORE_UNAVAILABLE`` (-1.0).
        """
        erosion_px = erosion_px if erosion_px is not None else self._erosion_px
        mask = self.get_mask(frame_idx, person_id)
        n = len(keypoints_xy)

        if mask is None:
            return np.full(n, SCORE_UNAVAILABLE, dtype=np.float32)

        h, w = mask.shape
        scores = np.full(n, SCORE_OUTSIDE, dtype=np.float32)

        # Erode the mask to detect the boundary zone
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
                # Out of frame — treat as unavailable
                scores[i] = SCORE_UNAVAILABLE
                continue
            if mask_eroded[yi, xi]:
                scores[i] = SCORE_INSIDE
            elif mask[yi, xi]:
                scores[i] = SCORE_BOUNDARY
            # else stays SCORE_OUTSIDE

        return scores

    def get_all_scores_for_frame(
        self,
        frame_idx: int,
        keypoints_per_person: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Convenience wrapper: score all persons in one frame.

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
    # Streaming generator (for memory-efficient processing)
    # ------------------------------------------------------------------

    def iter_scores(
        self,
        video_path: str | Path,
        persons: dict[str, tuple[int, np.ndarray]],
        keypoints_fn,
        start_frame: int = 0,
        end_frame: int | None = None,
        verbose: bool = False,
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Stream (frame_idx, scores_per_person) without storing full masks.

        Use this when memory is a concern (long videos).  Masks are computed
        and discarded frame-by-frame; only the scores for the provided
        keypoints are returned.

        Parameters
        ----------
        video_path, persons, start_frame, end_frame, verbose:
            Same as :meth:`process_video`.
        keypoints_fn:
            Callable ``(frame_idx) -> dict[person_id, np.ndarray]`` returning
            ``(N, 2)`` keypoint arrays for the frame, or an empty dict if no
            pose data is available.

        Yields
        ------
        ``(frame_idx, {person_id: scores_array})``
        """
        try:
            from ultralytics.models.sam import SAM2VideoPredictor
        except ImportError as exc:
            raise ImportError(
                "ultralytics >= 8.3 is required.  pip install ultralytics"
            ) from exc

        video_path = str(video_path)
        person_ids = list(persons.keys())
        init_bboxes = np.array(
            [persons[pid][1] for pid in person_ids], dtype=float
        )

        if end_frame is None:
            cap = cv2.VideoCapture(video_path)
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        overrides = dict(
            conf=0.25, task="segment", mode="predict", imgsz=1024,
            model=self._model_name, save=False, verbose=verbose,
            device=self._device,
        )
        predictor = SAM2VideoPredictor(overrides=overrides)

        abs_frame = 0
        for result in predictor(source=video_path, bboxes=init_bboxes, stream=True):
            if abs_frame < start_frame:
                abs_frame += 1
                continue
            if abs_frame >= end_frame:
                break

            kpts_dict = keypoints_fn(abs_frame)

            scores_dict: dict[str, np.ndarray] = {}
            if result.masks is not None and len(result.masks) > 0:
                masks_hw = result.masks.data.cpu().numpy()
                h, w = masks_hw.shape[1], masks_hw.shape[2]

                for obj_idx, pid in enumerate(person_ids):
                    if obj_idx >= len(masks_hw):
                        continue
                    mask = masks_hw[obj_idx].astype(bool)
                    kpts = kpts_dict.get(pid)
                    if kpts is None:
                        continue

                    # Erode
                    if self._erosion_px > 0:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * self._erosion_px + 1, 2 * self._erosion_px + 1),
                        )
                        mask_eroded = cv2.erode(
                            mask.astype(np.uint8), kernel
                        ).astype(bool)
                    else:
                        mask_eroded = mask

                    n = len(kpts)
                    scores = np.full(n, SCORE_OUTSIDE, dtype=np.float32)
                    for i, (x, y) in enumerate(kpts):
                        xi, yi = int(round(x)), int(round(y))
                        if not (0 <= xi < w and 0 <= yi < h):
                            scores[i] = SCORE_UNAVAILABLE
                        elif mask_eroded[yi, xi]:
                            scores[i] = SCORE_INSIDE
                        elif mask[yi, xi]:
                            scores[i] = SCORE_BOUNDARY
                    scores_dict[pid] = scores

            yield abs_frame, scores_dict
            abs_frame += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _erode_mask(self, mask: np.ndarray, erosion_px: int) -> np.ndarray:
        if erosion_px <= 0:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
        )
        return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
