import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    import pandas as pd
    from typing import List, Tuple, Callable, Dict, Optional


    return Callable, Dict, List, Optional, Path, Tuple, cv2, mo, np, pd, plt


@app.cell
def _(Path):
    video_dir = Path("C:/temp/aikido-2025-11-15-all/Harri_aihanmi_katatedori_ikkyo/videos")
    return (video_dir,)


@app.cell
def _(mo, video_dir):
    _available_mp4_files = sorted(list(video_dir.glob("*.mp4")))
    _video_dropdown_options = {file.name: file for file in _available_mp4_files}

    # Set the initial value of the dropdown.
    # Prioritize the existing 'video_file' if it's in the list of available files,
    # otherwise pick the first available file, or None if no files are found.
    # The dropdown's 'value' parameter expects one of the keys (file names).
    _initial_selected_file_name = None
    if _initial_selected_file_name is None and _available_mp4_files:
        _initial_selected_file_name = _available_mp4_files[0].name

    video_path_selector = mo.ui.dropdown(
        _video_dropdown_options,
        label="Select MP4 Video File",
        value=_initial_selected_file_name
    )

    video_path_selector

    return (video_path_selector,)


@app.cell
def _():
    return


@app.cell
def _(video_path_selector):
    video_path_selector.value
    return


@app.cell
def _(cv2, plt, video_path_selector):
    video = cv2.VideoCapture(video_path_selector.value)
    frames = []
    video.set(cv2.CAP_PROP_POS_FRAMES, 48)
    _ret, first_frame = video.read()
    video.release()
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    return first_frame, frames


@app.cell
def _(cv2, first_frame, plt):
    _size = 200
    x = 2457
    y = 1110
    plt.imshow(cv2.cvtColor(first_frame[y-_size//2:y+_size//2, x-_size//2:x+_size//2], cv2.COLOR_BGR2RGB))


    return x, y


@app.cell
def _(cv2, first_frame, plt, x, y):
    _size=20
    plt.imshow(cv2.cvtColor(first_frame[y-_size//2:y+_size//2, x-_size//2:x+_size//2], cv2.COLOR_BGR2RGB))
    return


@app.cell
def _(x, y):
    print(f"({y-10}, {y+10}, {x- 10}, {x+10})")
    return


@app.cell
def _():
    led_locs = {
        "cam1": (1750, 1770, 1390, 1410),
        "cam2": (1325, 1345, 2915, 2935),
        "cam3": (1340, 1360, 940, 960),
        "cam4": (1430, 1450, 2430, 2450),
        "cam5": (525, 545, 760, 780),
        "cam6": (1100, 1120, 2447, 2467)
    }

    return (led_locs,)


@app.cell
def _(cv2, led_locs, np, video_dir):
    def calc_frame_change(frame, prev_frame):
        _diff = frame - prev_frame
        _max_diff = _diff.max()
        _min_diff = _diff.min()
        return _max_diff if abs(_max_diff) > abs(_min_diff) else _min_diff

    changes = {}

    for _cam, _led_loc in led_locs.items():
        print(f"Processing {_cam} with LED location {_led_loc}...")
        changes[_cam] = [0.0]
        _video = cv2.VideoCapture(video_dir/f"{_cam}.mp4")
        _prev_frame = None

        while True:
            _ret, _frame = _video.read()
            if not _ret:
                break
            _frame = _frame[_led_loc[0]:_led_loc[1], _led_loc[2]:_led_loc[3]].astype(np.float64)
            if _prev_frame is not None:
                changes[_cam].append(calc_frame_change(_frame, _prev_frame))
            _prev_frame = _frame
        _video.release()
        print(f"Finished processing {_cam}. Detected {len(changes[_cam])} changes.")
    return (changes,)


@app.cell
def _(changes, plt):
    for _cam_name, _series_data in changes.items():
        plt.plot(_series_data, label=_cam_name)

    plt.title("Changes over Time for Each Camera LED")
    plt.xlabel("Frame Index")
    plt.ylabel("Change in Pixel Intensity")
    plt.legend()
    plt.grid(True)
    plt.gca()
    return


@app.cell
def _(changes, np, synchronize_cameras):
    cam_series = []
    fps = [120, 120, 120, 60, 120, 60]
    for _c in range(6):
        cam_series.append(np.array(changes[f"cam{_c+1}"]))
    results = synchronize_cameras(cam_series, fps_list=fps, ref_cam=0, event_cfg=dict(min_sep_s=0.5, prominence=2.0, polarity='both', smooth_win=3, use_derivative=False))
    return cam_series, fps, results


@app.cell
def _(results):
    results
    return


@app.cell
def _(results):
    results[2]["df"]
    return


@app.cell
def _(cam_series, plt, results):
    plt.plot(results[5]["df"].t_global, cam_series[5])
    return


@app.cell
def _(cam_series, np, results):
    import plotly.graph_objects as go

    _fig = go.Figure()

    for n in [0, 5]:
        _df_cam = results[n]["df"]
        _fig.add_trace(go.Scatter(
            x=_df_cam["t_global"],
            y=cam_series[n],
            mode='lines',
            name=f"Camera {n+1} (Global Time)",
            hovertemplate=(
                "<b>Camera %{customdata[0]}</b><br>"
                "Global Time: %{x:.2f} s<br>"
                "Intensity Change: %{y:.2f}<br>"
                "Local Frame: %{customdata[1]}<br>"
                "Local Time: %{customdata[2]:.2f} s<extra></extra>"
            ),
            customdata=np.stack([
                np.full(len(_df_cam), n + 1),
                _df_cam["frame"],
                _df_cam["t_local"]
            ], axis=-1)
        ))

    _fig.update_layout(
        title="Synchronized LED Intensity Changes over Global Time",
        xaxis_title="Global Time (seconds)",
        yaxis_title="Change in Pixel Intensity (Normalized)",
        hovermode="x unified",
        legend_title="Camera"
    )

    _fig
    return


@app.cell
def _(fps, results):
    import json
    sync_data = {}
    for _cam in range(5):
        _camsync = {}
        _camsync["fps"] = fps[_cam]
        _syncpts = []
        _camsync["syncpoints"] = _syncpts
        for _, _pt in results[_cam]["df"].iterrows():
            _syncpts.append({"frame": int(_pt["frame"]), "timestamp": _pt["t_global"]})
        sync_data[f"cam{_cam+1}"] = _camsync


    with open("sync_data.json", "w") as f:
        json.dump(sync_data, f, indent=2)
    return


@app.cell
def _(frames, plt):
    plt.imshow(frames[48])
    return


@app.cell
def _(Callable, Dict, List, Optional, Tuple, np, pd):
    # sync_blinking_led.py
    # Prototype: multi-camera synchronization from blinking LED brightness time series
    # Works with NumPy-only; uses SciPy if available for nicer filters/interpolators.

    # --- Optional SciPy imports (fall back to NumPy methods if missing) ---
    try:
        from scipy.signal import find_peaks as sp_find_peaks
        _HAS_SCIPY = True
    except Exception:
        sp_find_peaks = None
        _HAS_SCIPY = False

    try:
        from scipy.interpolate import PchipInterpolator
        _HAS_PCHIP = True
    except Exception:
        PchipInterpolator = None
        _HAS_PCHIP = False


    # ---------------------------
    # Utility helpers
    # ---------------------------

    def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        return (x - mu) / (sd + eps)


    def moving_average(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return x.copy()
        k = np.ones(w) / w
        return np.convolve(x, k, mode='same')


    def simple_highpass(x: np.ndarray, w: int) -> np.ndarray:
        """High-pass via DC removal with a moving average of width w."""
        return x - moving_average(x, w)


    def correlate_fft(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalized cross-correlation y vs x using FFT.
        Returns (corr, lags), with lags compatible with numpy.correlate 'full'.
        """
        n = len(x)
        m = len(y)
        # zero-mean, unit variance
        xx = zscore(x)
        yy = zscore(y)
        # next power of two for speed
        L = 1
        while L < n + m - 1:
            L <<= 1
        X = np.fft.rfft(xx, L)
        Y = np.fft.rfft(yy, L)
        c = np.fft.irfft(Y * np.conj(X), L)
        # roll to 'full' alignment (lags from -(n-1) to +(m-1))
        c = np.concatenate([c[-(n-1):], c[:m]])
        lags = np.arange(-(n-1), m)
        return c, lags


    # ---------------------------
    # Peak / event detection
    # ---------------------------

    def find_peaks_numpy(x: np.ndarray,
                         distance: int = 1,
                         threshold: float = 0.0,
                         polarity: str = 'pos') -> np.ndarray:
        """
        Simple local-maximum based peak detector using NumPy only.
        - distance: minimum separation (samples)
        - threshold: minimum peak amplitude (on x as-is)
        - polarity: 'pos' (maxima) or 'neg' (minima)
        Returns indices of peaks.
        """
        if polarity == 'neg':
            x = -x
        # Local maxima: x[i-1] < x[i] >= x[i+1]
        # Avoid first/last sample
        cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
        # Amplitude threshold
        if threshold != 0.0:
            cand = cand[x[cand] >= threshold]
        # Enforce minimum distance by greedy suppression
        if distance > 1 and cand.size > 0:
            sel = []
            last = -distance
            for i in cand:
                if i - last >= distance:
                    sel.append(i)
                    last = i
            cand = np.array(sel, dtype=int)
        return cand


    def detect_events(sig: np.ndarray,
                      fs: float,
                      min_sep_s: float = 0.2,
                      prominence: float = 1.5,
                      polarity: str = 'both',
                      smooth_win: int = 5,
                      use_derivative: bool = False) -> np.ndarray:
        """
        Detect events (spikes or edges) in a brightness signal.
        - Normalizes via z-score
        - Optional smoothing
        - Optional derivative to emphasize transitions
        - Detects positive and/or negative peaks and merges them

        Returns event times in seconds (sub-frame refined using parabolic interpolation).
        """
        x = zscore(sig)
        if smooth_win > 1:
            x = moving_average(x, smooth_win)

        if use_derivative:
            x = np.concatenate([[0.0], np.diff(x)])

        dist = max(1, int(min_sep_s * fs))

        # Choose detector
        if _HAS_SCIPY and sp_find_peaks is not None:
            peaks_pos, _ = sp_find_peaks(x, distance=dist, prominence=prominence) if polarity in ('both', 'pos') else (np.array([], int), {})
            peaks_neg, _ = sp_find_peaks(-x, distance=dist, prominence=prominence) if polarity in ('both', 'neg') else (np.array([], int), {})
        else:
            # Fallback: simple detector with z-threshold for prominence
            thr = np.mean(x) + prominence * np.std(x)
            peaks_pos = find_peaks_numpy(x, distance=dist, threshold=thr, polarity='pos') if polarity in ('both', 'pos') else np.array([], int)
            peaks_neg = find_peaks_numpy(x, distance=dist, threshold=thr, polarity='pos') if polarity in ('both', 'neg') else np.array([], int)
            # For minima, we used the same on -x via polarity='pos' route in fallback; above line mimics that choice.

        idx = np.sort(np.unique(np.concatenate([peaks_pos, peaks_neg])))

        # Sub-sample refinement via parabolic interpolation on each local extremum
        # For a peak at n, fit parabola through (n-1, n, n+1) to find fractional offset.
        t = []
        N = len(x)
        for n in idx:
            if 1 <= n < N - 1:
                y0, y1, y2 = x[n-1], x[n], x[n+1]
                denom = (y0 - 2*y1 + y2)
                if abs(denom) > 1e-12:
                    delta = 0.5 * (y0 - y2) / denom  # in [-1, +1] typically
                else:
                    delta = 0.0
            else:
                delta = 0.0
            t.append((n + np.clip(delta, -1.0, 1.0)) / fs)

        return np.array(t, dtype=float)


    # ---------------------------
    # DTW matching on event times
    # ---------------------------

    def dtw_match_event_times(tA: np.ndarray,
                              tB: np.ndarray,
                              band_s: Optional[float] = None) -> np.ndarray:
        """
        DTW between two increasing sequences of event times.
        Cost = |tA[i] - tB[j]| (seconds).
        Optionally restrict warping with a Sakoe-Chiba band (seconds).
        Returns list of matched index pairs [[iA, iB], ...] in order.
        """
        n, m = len(tA), len(tB)
        if n == 0 or m == 0:
            return np.empty((0, 2), dtype=int)

        # Build cost matrix with optional band
        INF = 1e18
        C = np.full((n + 1, m + 1), INF, dtype=float)
        C[0, 0] = 0.0
        P = np.zeros((n + 1, m + 1, 2), dtype=int)

        # Precompute allowed j range per i if band is given
        if band_s is not None:
            # Simple heuristic: restrict j such that |tA[i-1]-tB[j-1]| <= band_s
            valid_js = []
            for i in range(1, n + 1):
                diffs = np.abs(tB - tA[i - 1])
                mask = diffs <= band_s
                js = np.where(mask)[0] + 1  # shift by 1 for DP matrix
                if js.size == 0:
                    # If empty, widen slightly to avoid dead-ends
                    js = np.array([np.argmin(diffs) + 1], dtype=int)
                valid_js.append(js)
        else:
            valid_js = [np.arange(1, m + 1, dtype=int) for _ in range(n)]

        # DP
        for i in range(1, n + 1):
            js = valid_js[i - 1]
            for j in js:
                cost = abs(tA[i - 1] - tB[j - 1])
                # options: up (i-1,j), left (i,j-1), diag (i-1,j-1)
                opts = (C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
                k = int(np.argmin(opts))
                C[i, j] = opts[k] + cost
                if k == 0:
                    P[i, j] = (i - 1, j)
                elif k == 1:
                    P[i, j] = (i, j - 1)
                else:
                    P[i, j] = (i - 1, j - 1)

        # Backtrack to get pairs (keep only diagonal moves)
        i, j = n, m
        pairs = []
        while i > 0 and j > 0:
            pi, pj = P[i, j]
            if pi == i - 1 and pj == j - 1:
                pairs.append((i - 1, j - 1))
            i, j = pi, pj
        pairs.reverse()
        return np.array(pairs, dtype=int)


    # ---------------------------
    # Robust affine fit (RANSAC)
    # ---------------------------

    def ransac_affine_fit(t_ref: np.ndarray,
                          t_cam: np.ndarray,
                          max_err_s: float = 0.01,
                          n_iter: int = 800,
                          rng: Optional[np.random.Generator] = None) -> Tuple[Tuple[float, float], np.ndarray]:
        """
        Fit t_ref ≈ a * t_cam + b with RANSAC.
        Returns ((a, b), inlier_mask_indices)
        """
        assert t_ref.shape == t_cam.shape
        N = len(t_ref)
        if rng is None:
            rng = np.random.default_rng(1234)

        best_inliers = np.array([], dtype=int)
        best_model = (1.0, 0.0)
        if N < 2:
            return best_model, best_inliers

        for _ in range(n_iter):
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            # Two-point affine solve
            denom = (t_cam[i] - t_cam[j])
            if abs(denom) < 1e-12:
                continue
            a = (t_ref[i] - t_ref[j]) / denom
            b = t_ref[i] - a * t_cam[i]
            err = np.abs(t_ref - (a * t_cam + b))
            inliers = np.where(err <= max_err_s)[0]
            if inliers.size > best_inliers.size:
                best_inliers = inliers
                best_model = (a, b)

        # Refine using least squares on inliers (if enough)
        if best_inliers.size >= 2:
            B = np.vstack([t_cam[best_inliers], np.ones(best_inliers.size)]).T
            a_ref, b_ref = np.linalg.lstsq(B, t_ref[best_inliers], rcond=None)[0]
            best_model = (a_ref, b_ref)

        return best_model, best_inliers


    # ---------------------------
    # Mapping choice: affine or monotone piecewise
    # ---------------------------

    def build_time_map(t_ref: np.ndarray,
                       t_cam: np.ndarray,
                       use_piecewise_if_resid_std_over_s: float = 0.008) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict]:
        """
        Given matched event times (t_ref, t_cam), fit an affine map, evaluate residuals.
        If residuals are large -> return a monotone piecewise (PCHIP if available, else linear).
        Returns (f_map, meta) where f_map maps t_cam -> t_ref.
        """
        # Affine LS
        B = np.vstack([t_cam, np.ones_like(t_cam)]).T
        a, b = np.linalg.lstsq(B, t_ref, rcond=None)[0]
        resid = t_ref - (a * t_cam + b)
        resid_std = float(np.std(resid))

        if resid_std <= use_piecewise_if_resid_std_over_s or len(t_cam) < 4:
            meta = {'type': 'affine', 'a': float(a), 'b': float(b), 'resid_std_s': resid_std}
            return (lambda t: a * t + b), meta

        # Piecewise monotone fit
        order = np.argsort(t_cam)
        x = t_cam[order]
        y = t_ref[order]

        if _HAS_PCHIP and PchipInterpolator is not None:
            f = PchipInterpolator(x, y, extrapolate=True)
            f_map = lambda t: f(t)
            meta = {'type': 'pchip', 'knots': len(x), 'resid_std_s': resid_std}
        else:
            # Fallback: monotone linear interpolation
            def f_lin(tt: np.ndarray) -> np.ndarray:
                return np.interp(tt, x, y, left=y[0] + (tt - x[0]) * (y[1]-y[0]) / (x[1]-x[0]),
                                      right=y[-1] + (tt - x[-1]) * (y[-1]-y[-2]) / (x[-1]-x[-2]))
            f_map = f_lin
            meta = {'type': 'linear_piecewise', 'knots': len(x), 'resid_std_s': resid_std}

        return f_map, meta


    # ---------------------------
    # Main: synchronize one camera to reference using events
    # ---------------------------

    def sync_one_camera_to_ref(sig_ref: np.ndarray, fs_ref: float,
                               sig_cam: np.ndarray, fs_cam: float,
                               event_cfg: dict = None,
                               dtw_band_s: Optional[float] = 1.0,
                               ransac_max_err_s: float = 0.01) -> Dict:
        """
        Returns dict with:
        - 'map': callable t_ref = map(t_cam)
        - 'meta': info about the fit
        - 'pairs': matched event index pairs
        - 't_ref_events': times of events (ref)
        - 't_cam_events': times of events (cam)
        """
        if event_cfg is None:
            event_cfg = dict(min_sep_s=0.2, prominence=2.0, polarity='both', smooth_win=5, use_derivative=False)

        # 1) Detect events with sub-sample refinement
        t_ref_events = detect_events(sig_ref, fs_ref, **event_cfg)
        t_cam_events = detect_events(sig_cam, fs_cam, **event_cfg)

        # Guard clause: if too few events, fall back to NCC over the whole signal
        if len(t_ref_events) < 2 or len(t_cam_events) < 2:
            # naive: just align by maximizing cross-correlation (shift only)
            # Resample cam to ref grid by index-time
            n = len(sig_ref); m = len(sig_cam)
            # upsample both to same length by linear interp on time
            tR = np.arange(n) / fs_ref
            tC = np.arange(m) / fs_cam
            # Interpolate cam onto ref time
            sig_cam_interp = np.interp(tR, tC, zscore(sig_cam))
            corr, lags = correlate_fft(zscore(sig_ref), sig_cam_interp)
            k = np.argmax(corr)
            # sub-sample refine
            if 1 <= k < len(corr) - 1:
                y0, y1, y2 = corr[k-1], corr[k], corr[k+1]
                delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-12)
            else:
                delta = 0.0
            offset_s = (lags[k] + delta) / fs_ref
            f_map = lambda t: t + offset_s
            return dict(
                map=f_map,
                meta={'type': 'shift_only', 'offset_s': float(offset_s), 'note': 'few events, NCC fallback'},
                pairs=np.empty((0, 2), dtype=int),
                t_ref_events=t_ref_events,
                t_cam_events=t_cam_events
            )

        # 2) DTW match event times
        pairs = dtw_match_event_times(t_ref_events, t_cam_events, band_s=dtw_band_s)

        # 3) Build robust affine with RANSAC on matched pairs
        A = t_ref_events[pairs[:, 0]]
        B = t_cam_events[pairs[:, 1]]
        (a, b), inliers = ransac_affine_fit(A, B, max_err_s=ransac_max_err_s)

        # 4) Optionally upgrade to piecewise if residuals large
        f_map, meta = build_time_map(A[inliers], B[inliers])

        # metadata
        meta_all = dict(meta)
        meta_all.update({
            'events_ref': int(len(t_ref_events)),
            'events_cam': int(len(t_cam_events)),
            'pairs': int(len(pairs)),
            'inliers': int(len(inliers)),
        })
        return dict(
            map=f_map,
            meta=meta_all,
            pairs=pairs,
            t_ref_events=t_ref_events,
            t_cam_events=t_cam_events
        )


    # ---------------------------
    # Batch: synchronize multiple cameras to a reference
    # ---------------------------

    def synchronize_cameras(signals: List[np.ndarray],
                            fps_list: List[float],
                            ref_cam: int = 0,
                            event_cfg: dict = None,
                            dtw_band_s: Optional[float] = 1.0,
                            ransac_max_err_s: float = 0.01) -> Dict[int, Dict]:
        """
        For each camera k, compute mapping t_global (ref time) = f_k(t_local).
        Returns dict:
          cam_index -> {
             'df': DataFrame with columns ['frame', 't_local', 't_global'],
             'meta': info about fit
          }
        """
        K = len(signals)
        assert K == len(fps_list)
        results = {}

        # Reference timeline: identity
        Nref = len(signals[ref_cam])
        fs_ref = fps_list[ref_cam]
        frames_ref = np.arange(Nref, dtype=int)
        t_local_ref = frames_ref / fs_ref
        df_ref = pd.DataFrame({
            'frame': frames_ref,
            't_local': t_local_ref,
            't_global': t_local_ref,  # identity
        })
        results[ref_cam] = {'df': df_ref, 'meta': {'type': 'reference', 'fps': fs_ref}}

        # Sync others
        for k in range(K):
            if k == ref_cam:
                continue
            info = sync_one_camera_to_ref(
                sig_ref=signals[ref_cam], fs_ref=fs_ref,
                sig_cam=signals[k], fs_cam=fps_list[k],
                event_cfg=event_cfg, dtw_band_s=dtw_band_s,
                ransac_max_err_s=ransac_max_err_s
            )
            f_map = info['map']

            Nk = len(signals[k])
            fs_k = fps_list[k]
            frames = np.arange(Nk, dtype=int)
            t_local = frames / fs_k
            t_global = f_map(t_local)

            df = pd.DataFrame({
                'frame': frames,
                't_local': t_local,
                't_global': t_global
            })
            results[k] = {'df': df, 'meta': info['meta']}

        return results


    # ---------------------------
    # Example usage (commented)
    # ---------------------------
    # if __name__ == "__main__":
    #     # Example with synthetic signals; replace with your real arrays.
    #     fsA = 60.0
    #     fsB = 60.0
    #     T = 20.0
    #     tA = np.arange(int(T * fsA)) / fsA
    #     tB = np.arange(int(T * fsB)) / fsB
    #
    #     # Synthetic blinking: spikes every ~1s with noise
    #     rng = np.random.default_rng(0)
    #     xA = 0.05 * rng.standard_normal(len(tA))
    #     xB = 0.05 * rng.standard_normal(len(tB))
    #
    #     # Create three spike events with different shapes
    #     for tt in [5.0, 10.0, 15.5]:
    #         xA += np.exp(-0.5 * ((tA - tt) / 0.03) ** 2) * 5.0
    #         xB += np.exp(-0.5 * ((tB - (1.002 * tt + 0.12)) / 0.03) ** 2) * 4.5  # drift + offset
    #
    #     signals = [xA, xB]
    #     fps_list = [fsA, fsB]
    #     out = synchronize_cameras(signals, fps_list, ref_cam=0,
    #                               event_cfg=dict(min_sep_s=0.5, prominence=2.0, polarity='both', smooth_win=3, use_derivative=False),
    #                               dtw_band_s=1.0,
    #                               ransac_max_err_s=0.02)
    #
    #     for cam, res in out.items():
    #         print(f"Camera {cam}: meta =", res['meta'])
    #         print(res['df'].head())
    return (synchronize_cameras,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
