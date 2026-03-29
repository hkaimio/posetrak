"""led_sync.py — LED-based multi-camera synchronisation algorithm.

Extracts a per-frame brightness-change signal from a small ROI in each video,
then uses peak detection, DTW event matching, and RANSAC affine fitting to
compute a global-time mapping for every camera relative to a chosen reference.

This module is pure Python / NumPy (SciPy is used when available for better
peak detection and spline interpolation, but the module degrades gracefully
without it).  All Qt / UI code lives in page_sync.py.

Algorithm overview
------------------
1. ``extract_brightness_changes`` — read every frame, crop to the LED ROI,
   compute signed max-delta between consecutive frames.
2. ``detect_events`` — normalise signal, smooth, detect peaks/troughs with
   sub-sample parabolic refinement → event times in seconds.
3. ``dtw_match_event_times`` — align event sequences from two cameras using
   DTW on the time axis.
4. ``ransac_affine_fit`` → ``build_time_map`` — fit a robust affine
   (or piecewise-monotone) mapping t_cam → t_global.
5. ``run_led_sync`` — orchestrate steps 1-4 for all cameras and return a
   ``LedSyncResult`` with per-camera ``CameraSyncResult`` objects that carry
   the global frame times and quality metrics needed by the UI.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

try:
    from scipy.signal import find_peaks as _sp_find_peaks
    _HAS_SCIPY = True
except Exception:
    _sp_find_peaks = None
    _HAS_SCIPY = False

try:
    from scipy.interpolate import PchipInterpolator as _PchipInterpolator
    _HAS_PCHIP = True
except Exception:
    _PchipInterpolator = None
    _HAS_PCHIP = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ROI:
    """Axis-aligned region of interest in full video-frame pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    def to_slice(self) -> tuple[slice, slice]:
        """Return (row_slice, col_slice) for numpy-array indexing."""
        r1, r2 = min(self.y1, self.y2), max(self.y1, self.y2)
        c1, c2 = min(self.x1, self.x2), max(self.x1, self.x2)
        return slice(r1, r2), slice(c1, c2)

    @property
    def is_valid(self) -> bool:
        return abs(self.x2 - self.x1) > 0 and abs(self.y2 - self.y1) > 0


@dataclasses.dataclass
class CameraSyncResult:
    """Synchronisation result for one camera."""
    camera_instance_id: str
    shot_video_id: str
    fps_used: float
    n_events: int           # events detected in this camera's signal
    n_pairs: int            # DTW-matched pairs
    n_inliers: int          # RANSAC inliers
    map_type: str           # 'reference' | 'affine' | 'pchip' | 'shift_only'
    resid_std_s: float      # affine-fit residual std in seconds (0 for reference)
    frame_times: np.ndarray  # shape (N,) — t_global for each frame index
    brightness: np.ndarray   # shape (N,) — raw brightness-change signal


@dataclasses.dataclass
class LedSyncResult:
    """Result from running LED sync across all cameras in one shot."""
    cameras: list[CameraSyncResult]
    ref_camera_idx: int


# ---------------------------------------------------------------------------
# Brightness extraction
# ---------------------------------------------------------------------------


def extract_brightness_changes(
    file_path: str,
    roi: ROI,
    fps_override: Optional[float] = None,
    progress_cb=None,
) -> tuple[np.ndarray, float]:
    """Extract a per-frame brightness-change signal from a video ROI.

    Parameters
    ----------
    file_path:
        Path to the video file.
    roi:
        Region of interest in full video-frame pixel coordinates.
    fps_override:
        Use this fps instead of the container fps when provided.
    progress_cb:
        Optional ``callback(frame_idx: int, total_frames: int)`` called after
        each frame so callers can display progress.

    Returns
    -------
    (changes, fps)
        ``changes`` is a float64 array of length *frame_count*.  The first
        element is always 0.0 (no previous frame to diff against).
        ``fps`` is the fps value actually used (override or container).
    """
    import cv2  # imported lazily so the module loads without OpenCV

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {file_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_container = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps_override) if fps_override is not None else fps_container

    row_sl, col_sl = roi.to_slice()
    changes: list[float] = [0.0]
    prev: np.ndarray | None = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        patch = frame[row_sl, col_sl].astype(np.float64)
        if prev is not None:
            diff = patch - prev
            mx, mn = float(diff.max()), float(diff.min())
            changes.append(mx if abs(mx) >= abs(mn) else mn)
        prev = patch
        frame_idx += 1
        if progress_cb is not None and total > 0:
            progress_cb(frame_idx, total)

    cap.release()
    return np.array(changes, dtype=np.float64), fps


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / (sd + eps)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="same")


def _correlate_fft(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation via FFT.  Returns (corr, lags)."""
    n, m = len(x), len(y)
    xx, yy = zscore(x), zscore(y)
    L = 1
    while L < n + m - 1:
        L <<= 1
    X, Y = np.fft.rfft(xx, L), np.fft.rfft(yy, L)
    c = np.fft.irfft(Y * np.conj(X), L)
    c = np.concatenate([c[-(n - 1):], c[:m]])
    lags = np.arange(-(n - 1), m)
    return c, lags


def _find_peaks_numpy(
    x: np.ndarray,
    distance: int = 1,
    threshold: float = 0.0,
    polarity: str = "pos",
) -> np.ndarray:
    """Minimal NumPy-only peak detector (fallback when SciPy is absent)."""
    if polarity == "neg":
        x = -x
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if threshold != 0.0:
        cand = cand[x[cand] >= threshold]
    if distance > 1 and cand.size > 0:
        sel: list[int] = []
        last = -distance
        for i in cand:
            if i - last >= distance:
                sel.append(int(i))
                last = int(i)
        cand = np.array(sel, dtype=int)
    return cand


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


def detect_events(
    sig: np.ndarray,
    fs: float,
    min_sep_s: float = 0.2,
    prominence: float = 1.5,
    polarity: str = "both",
    smooth_win: int = 5,
    use_derivative: bool = False,
) -> np.ndarray:
    """Detect brightness-change events and return their times in seconds.

    Normalises the signal via z-score, applies optional smoothing and
    differencing, then detects positive/negative peaks.  Sub-sample precision
    is achieved via parabolic interpolation around each detected peak.
    """
    x = zscore(sig)
    if smooth_win > 1:
        x = moving_average(x, smooth_win)
    if use_derivative:
        x = np.concatenate([[0.0], np.diff(x)])

    dist = max(1, int(min_sep_s * fs))

    if _HAS_SCIPY and _sp_find_peaks is not None:
        peaks_pos = (
            _sp_find_peaks(x, distance=dist, prominence=prominence)[0]
            if polarity in ("both", "pos") else np.array([], int)
        )
        peaks_neg = (
            _sp_find_peaks(-x, distance=dist, prominence=prominence)[0]
            if polarity in ("both", "neg") else np.array([], int)
        )
    else:
        thr = np.mean(x) + prominence * np.std(x)
        peaks_pos = (
            _find_peaks_numpy(x, distance=dist, threshold=thr, polarity="pos")
            if polarity in ("both", "pos") else np.array([], int)
        )
        peaks_neg = (
            _find_peaks_numpy(-x, distance=dist, threshold=thr, polarity="pos")
            if polarity in ("both", "neg") else np.array([], int)
        )

    idx = np.sort(np.unique(np.concatenate([peaks_pos, peaks_neg])))

    t: list[float] = []
    N = len(x)
    for n in idx:
        if 1 <= n < N - 1:
            y0, y1, y2 = x[n - 1], x[n], x[n + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        else:
            delta = 0.0
        t.append((n + float(np.clip(delta, -1.0, 1.0))) / fs)

    return np.array(t, dtype=float)


# ---------------------------------------------------------------------------
# DTW matching
# ---------------------------------------------------------------------------


def dtw_match_event_times(
    tA: np.ndarray,
    tB: np.ndarray,
    band_s: Optional[float] = None,
) -> np.ndarray:
    """DTW on two sorted event-time sequences.  Returns matched index pairs."""
    n, m = len(tA), len(tB)
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int)

    INF = 1e18
    C = np.full((n + 1, m + 1), INF, dtype=float)
    C[0, 0] = 0.0
    P = np.zeros((n + 1, m + 1, 2), dtype=int)

    if band_s is not None:
        valid_js: list[np.ndarray] = []
        for i in range(1, n + 1):
            diffs = np.abs(tB - tA[i - 1])
            mask = diffs <= band_s
            js = np.where(mask)[0] + 1
            if js.size == 0:
                js = np.array([np.argmin(diffs) + 1], dtype=int)
            valid_js.append(js)
    else:
        valid_js = [np.arange(1, m + 1, dtype=int) for _ in range(n)]

    for i in range(1, n + 1):
        for j in valid_js[i - 1]:
            cost = abs(tA[i - 1] - tB[j - 1])
            opts = (C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
            k = int(np.argmin(opts))
            C[i, j] = opts[k] + cost
            P[i, j] = ((i - 1, j), (i, j - 1), (i - 1, j - 1))[k]

    i, j = n, m
    pairs: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        pi, pj = P[i, j]
        if pi == i - 1 and pj == j - 1:
            pairs.append((i - 1, j - 1))
        i, j = int(pi), int(pj)
    pairs.reverse()
    return np.array(pairs, dtype=int)


# ---------------------------------------------------------------------------
# Robust affine fit
# ---------------------------------------------------------------------------


def ransac_affine_fit(
    t_ref: np.ndarray,
    t_cam: np.ndarray,
    max_err_s: float = 0.01,
    n_iter: int = 800,
    rng: Optional[np.random.Generator] = None,
) -> tuple[tuple[float, float], np.ndarray]:
    """Fit t_ref ≈ a * t_cam + b with RANSAC.  Returns ((a, b), inlier_mask)."""
    assert t_ref.shape == t_cam.shape
    N = len(t_ref)
    if rng is None:
        rng = np.random.default_rng(1234)

    best_inliers: np.ndarray = np.array([], dtype=int)
    best_model = (1.0, 0.0)
    if N < 2:
        return best_model, best_inliers

    for _ in range(n_iter):
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        denom = t_cam[i] - t_cam[j]
        if abs(denom) < 1e-12:
            continue
        a = (t_ref[i] - t_ref[j]) / denom
        b = t_ref[i] - a * t_cam[i]
        inliers = np.where(np.abs(t_ref - (a * t_cam + b)) <= max_err_s)[0]
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_model = (a, b)

    if best_inliers.size >= 2:
        B = np.vstack([t_cam[best_inliers], np.ones(best_inliers.size)]).T
        a_r, b_r = np.linalg.lstsq(B, t_ref[best_inliers], rcond=None)[0]
        best_model = (float(a_r), float(b_r))

    return best_model, best_inliers


def build_time_map(
    t_ref: np.ndarray,
    t_cam: np.ndarray,
    use_piecewise_threshold_s: float = 0.008,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Fit t_ref = f(t_cam).  Returns (callable, meta_dict).

    Uses an affine LS fit; upgrades to monotone piecewise interpolation if
    residual std exceeds *use_piecewise_threshold_s*.
    """
    B = np.vstack([t_cam, np.ones_like(t_cam)]).T
    a, b = np.linalg.lstsq(B, t_ref, rcond=None)[0]
    resid_std = float(np.std(t_ref - (a * t_cam + b)))

    if resid_std <= use_piecewise_threshold_s or len(t_cam) < 4:
        a_f, b_f = float(a), float(b)
        return (lambda t, _a=a_f, _b=b_f: _a * t + _b), {
            "type": "affine", "a": a_f, "b": b_f, "resid_std_s": resid_std,
        }

    order = np.argsort(t_cam)
    x, y = t_cam[order], t_ref[order]

    if _HAS_PCHIP and _PchipInterpolator is not None:
        f = _PchipInterpolator(x, y, extrapolate=True)
        return (lambda t, _f=f: np.asarray(_f(t))), {
            "type": "pchip", "knots": len(x), "resid_std_s": resid_std,
        }

    # Fallback: monotone piecewise linear with linear extrapolation
    slope_lo = (y[1] - y[0]) / (x[1] - x[0]) if len(x) > 1 else 1.0
    slope_hi = (y[-1] - y[-2]) / (x[-1] - x[-2]) if len(x) > 1 else 1.0

    def _f_lin(t: np.ndarray, _x=x, _y=y, _sl=slope_lo, _sh=slope_hi) -> np.ndarray:
        return np.interp(
            t, _x, _y,
            left=_y[0] + (t - _x[0]) * _sl,
            right=_y[-1] + (t - _x[-1]) * _sh,
        )

    return _f_lin, {
        "type": "linear_piecewise", "knots": len(x), "resid_std_s": resid_std,
    }


# ---------------------------------------------------------------------------
# Single-camera sync
# ---------------------------------------------------------------------------


def _sync_one_camera(
    sig_ref: np.ndarray,
    fs_ref: float,
    sig_cam: np.ndarray,
    fs_cam: float,
    event_cfg: dict | None = None,
    dtw_band_s: float = 1.0,
    ransac_max_err_s: float = 0.01,
) -> dict:
    """Synchronise one camera to the reference.  Returns internal info dict."""
    cfg = event_cfg or dict(
        min_sep_s=0.2, prominence=2.0, polarity="both",
        smooth_win=5, use_derivative=False,
    )
    t_ref_events = detect_events(sig_ref, fs_ref, **cfg)
    t_cam_events = detect_events(sig_cam, fs_cam, **cfg)

    # Fallback: cross-correlation shift if too few events
    if len(t_ref_events) < 2 or len(t_cam_events) < 2:
        n, m = len(sig_ref), len(sig_cam)
        t_r = np.arange(n) / fs_ref
        t_c = np.arange(m) / fs_cam
        sig_cam_interp = np.interp(t_r, t_c, zscore(sig_cam))
        corr, lags = _correlate_fft(zscore(sig_ref), sig_cam_interp)
        k = int(np.argmax(corr))
        if 1 <= k < len(corr) - 1:
            y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
            delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        else:
            delta = 0.0
        offset_s = float((lags[k] + delta) / fs_ref)
        return dict(
            f_map=lambda t, _o=offset_s: t + _o,
            meta={"type": "shift_only", "offset_s": offset_s, "resid_std_s": 0.0},
            n_events=len(t_cam_events),
            n_pairs=0,
            n_inliers=0,
        )

    pairs = dtw_match_event_times(t_ref_events, t_cam_events, band_s=dtw_band_s)
    A = t_ref_events[pairs[:, 0]]
    B_ev = t_cam_events[pairs[:, 1]]
    (_, _), inliers = ransac_affine_fit(A, B_ev, max_err_s=ransac_max_err_s)

    if inliers.size < 2:
        # Not enough inliers; fall back to LS on all pairs
        inliers = np.arange(len(A))

    f_map, meta = build_time_map(A[inliers], B_ev[inliers])
    meta["events_cam"] = len(t_cam_events)
    meta["events_ref"] = len(t_ref_events)
    return dict(
        f_map=f_map,
        meta=meta,
        n_events=len(t_cam_events),
        n_pairs=len(pairs),
        n_inliers=int(inliers.size),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_led_sync(
    signals: list[np.ndarray],
    fps_list: list[float],
    cam_ids: list[str],
    video_ids: list[str],
    ref_cam: int = 0,
    event_cfg: dict | None = None,
    dtw_band_s: float = 1.0,
    ransac_max_err_s: float = 0.01,
) -> LedSyncResult:
    """Synchronise all cameras using LED brightness-change signals.

    Parameters
    ----------
    signals:
        Per-camera brightness arrays from ``extract_brightness_changes``.
    fps_list:
        Per-camera fps (after any user overrides).
    cam_ids, video_ids:
        Camera-instance ID and shot_video_id for each camera (same order).
    ref_cam:
        Index of the reference camera (identity map, global time = local time).

    Returns
    -------
    ``LedSyncResult`` with one ``CameraSyncResult`` per camera.
    """
    K = len(signals)
    assert K == len(fps_list) == len(cam_ids) == len(video_ids)

    cam_results: list[CameraSyncResult] = []

    # Reference camera — identity map
    ref_sig = signals[ref_cam]
    ref_fps = fps_list[ref_cam]
    ref_frames = np.arange(len(ref_sig), dtype=int)
    ref_times = ref_frames / ref_fps
    ref_events = detect_events(ref_sig, ref_fps, **(event_cfg or {}))

    for k in range(K):
        if k == ref_cam:
            cam_results.append(CameraSyncResult(
                camera_instance_id=cam_ids[k],
                shot_video_id=video_ids[k],
                fps_used=ref_fps,
                n_events=len(ref_events),
                n_pairs=0,
                n_inliers=0,
                map_type="reference",
                resid_std_s=0.0,
                frame_times=ref_times.copy(),
                brightness=ref_sig.copy(),
            ))
            continue

        info = _sync_one_camera(
            sig_ref=ref_sig, fs_ref=ref_fps,
            sig_cam=signals[k], fs_cam=fps_list[k],
            event_cfg=event_cfg, dtw_band_s=dtw_band_s,
            ransac_max_err_s=ransac_max_err_s,
        )
        f_map = info["f_map"]
        meta = info["meta"]

        N_k = len(signals[k])
        t_local = np.arange(N_k) / fps_list[k]
        t_global = np.asarray(f_map(t_local), dtype=float)

        cam_results.append(CameraSyncResult(
            camera_instance_id=cam_ids[k],
            shot_video_id=video_ids[k],
            fps_used=fps_list[k],
            n_events=info["n_events"],
            n_pairs=info["n_pairs"],
            n_inliers=info["n_inliers"],
            map_type=meta.get("type", "affine"),
            resid_std_s=meta.get("resid_std_s", 0.0),
            frame_times=t_global,
            brightness=signals[k].copy(),
        ))

    return LedSyncResult(cameras=cam_results, ref_camera_idx=ref_cam)
