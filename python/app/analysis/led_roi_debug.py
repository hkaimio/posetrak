import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from pathlib import Path
    from scipy.ndimage import gaussian_filter
    return Path, gaussian_filter, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## LED ROI debug — raw video inspection & algorithm comparison
    """)
    return


@app.cell
def _():
    CAMERAS = [
        {"name": "ace2pro",     "w": 23, "h": 25, "fps": 119.88},
        {"name": "gopromini-01","w": 26, "h": 22, "fps": 119.88},
        {"name": "gopromini-02","w": 21, "h": 19, "fps": 119.88},
        {"name": "instax3",     "w": 29, "h": 26, "fps":  59.94},
        {"name": "pixel9",      "w": 11, "h": 12, "fps": 118.88},
        {"name": "r5",          "w": 15, "h": 15, "fps":  59.94},
    ]

    RAW_FILES = [
        "/tmp/led_rois/cam0_ace2pro_23x25.raw",
        "/tmp/led_rois/cam1_gopromini01_26x22.raw",
        "/tmp/led_rois/cam2_gopromini02_21x19.raw",
        "/tmp/led_rois/cam3_instax3_29x26.raw",
        "/tmp/led_rois/cam4_pixel9_11x12.raw",
        "/tmp/led_rois/cam5_r5_15x15.raw",
    ]
    return CAMERAS, RAW_FILES


@app.cell
def _(CAMERAS, Path, RAW_FILES, np):
    def load_raw(path, w, h):
        """Load a gray8 raw video file into array (N, h, w) uint8.

        ffmpeg rounds odd crop widths up to the next even number for alignment,
        so the actual row stride may differ from the requested width.  We detect
        the true stride by trying w, w+1 (and symmetrically for h) until the
        file size divides evenly into whole frames.
        """
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        total = len(data)
        for w_try in [w, w + (w % 2), w - (w % 2)]:
            for h_try in [h, h + (h % 2), h - (h % 2)]:
                if w_try <= 0 or h_try <= 0:
                    continue
                frame_bytes = w_try * h_try
                if frame_bytes > 0 and total % frame_bytes == 0:
                    n = total // frame_bytes
                    print(f"  → stride detected: {w_try}x{h_try} (requested {w}x{h}), {n} frames")
                    return data.reshape(n, h_try, w_try)
        # fallback: truncate to fit requested dimensions
        n = total // (w * h)
        return data[: n * w * h].reshape(n, h, w)

    all_frames = []
    for _cam, _path in zip(CAMERAS, RAW_FILES):
        try:
            frames = load_raw(_path, _cam["w"], _cam["h"])
            all_frames.append(frames)
            print(f"{_cam['name']}: {len(frames)} frames ({len(frames)/_cam['fps']:.1f} s), "
                  f"pixel range [{frames.min()},{frames.max()}]")
        except Exception as e:
            all_frames.append(None)
            print(f"{_cam['name']}: FAILED — {e}")

    return (all_frames,)


@app.cell
def _(gaussian_filter, np):
    def extract_max_signed(frames):
        """Current algorithm: signed max-magnitude pixel diff."""
        out = np.zeros(len(frames))
        for i in range(1, len(frames)):
            d = frames[i].astype(float) - frames[i-1].astype(float)
            mx, mn = d.max(), d.min()
            out[i] = mx if abs(mx) >= abs(mn) else mn
        return out

    def extract_mean_diff(frames):
        """Mean of per-pixel diff — robust to motion edges, loses small LEDs."""
        out = np.zeros(len(frames))
        for i in range(1, len(frames)):
            out[i] = float((frames[i].astype(float) - frames[i-1].astype(float)).mean())
        return out

    def extract_blur_max(frames, sigma=2.0):
        """Gaussian blur then signed max-magnitude — suppresses motion fringing."""
        out = np.zeros(len(frames))
        for i in range(1, len(frames)):
            d = frames[i].astype(float) - frames[i-1].astype(float)
            d_blur = gaussian_filter(d, sigma=sigma)
            mx, mn = d_blur.max(), d_blur.min()
            out[i] = mx if abs(mx) >= abs(mn) else mn
        return out

    def extract_percentile(frames, pct=95):
        """Signed: take pct-th percentile of |diff|, preserve sign of that pixel."""
        out = np.zeros(len(frames))
        for i in range(1, len(frames)):
            d = frames[i].astype(float) - frames[i-1].astype(float)
            flat = d.ravel()
            pos = flat[flat > 0]
            neg = flat[flat < 0]
            p = np.percentile(pos, pct) if len(pos) else 0.0
            n = np.percentile(-neg, pct) if len(neg) else 0.0
            out[i] = p if p >= n else -n
        return out

    ALGORITHMS = {
        "max_signed (current)": extract_max_signed,
        "blur_max σ=1":  lambda f: extract_blur_max(f, sigma=1),
        "blur_max σ=2":  lambda f: extract_blur_max(f, sigma=2),
        "blur_max σ=3":  lambda f: extract_blur_max(f, sigma=3),
        "mean_diff":     extract_mean_diff,
        "percentile 95": lambda f: extract_percentile(f, pct=95),
    }
    return (ALGORITHMS,)


@app.cell
def _(CAMERAS, mo):
    cam_selector = mo.ui.dropdown(
        options={c["name"]: i for i, c in enumerate(CAMERAS)},
        value="ace2pro",
        label="Camera",
    )
    frame_slider = mo.ui.slider(0, 1000, value=0, step=1, label="Start frame")
    window_slider = mo.ui.slider(50, 500, value=200, step=50, label="Window (frames)")
    mo.hstack([cam_selector, frame_slider, window_slider])
    return cam_selector, frame_slider, window_slider


@app.cell
def _(CAMERAS, all_frames, cam_selector, frame_slider, mo, plt):
    _cam_idx = cam_selector.value
    _frames = all_frames[_cam_idx]
    _cam = CAMERAS[_cam_idx]

    if _frames is None:
        _result = mo.md(f"**No data for {_cam['name']}**")
    else:
        _f = min(frame_slider.value, len(_frames) - 1)
        _patch = _frames[_f]

        fig_frame, axes = plt.subplots(1, 3, figsize=(10, 3))

        # Raw frame
        axes[0].imshow(_patch, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0].set_title(f"Frame {_f}  ({_cam['name']})")
        axes[0].axis("off")

        # Diff to next frame (if available)
        if _f + 1 < len(_frames):
            _diff = _frames[_f + 1].astype(float) - _patch.astype(float)
            _lim = max(abs(_diff.max()), abs(_diff.min()), 1)
            axes[1].imshow(_diff, cmap="RdBu_r", vmin=-_lim, vmax=_lim, interpolation="nearest")
            axes[1].set_title(f"Diff frame {_f}→{_f+1}\nrange [{_diff.min():.0f}, {_diff.max():.0f}]")
        else:
            axes[1].set_visible(False)
        axes[1].axis("off")

        # Pixel value histogram
        axes[2].hist(_patch.ravel(), bins=32, range=(0, 255), color="steelblue")
        axes[2].set_title("Pixel histogram")
        axes[2].set_xlabel("Gray value")

        fig_frame.tight_layout()
        _result = fig_frame

    _result


@app.cell
def _(
    ALGORITHMS,
    CAMERAS,
    all_frames,
    cam_selector,
    frame_slider,
    mo,
    np,
    plt,
    window_slider,
):
    _cam_idx = cam_selector.value
    _frames = all_frames[_cam_idx]
    _cam = CAMERAS[_cam_idx]

    if _frames is None:
        _result2 = mo.md("No data")
    else:
        _f0 = frame_slider.value
        _f1 = min(_f0 + window_slider.value, len(_frames))
        _clip = _frames[_f0:_f1]

        fig_sig, axes_sig = plt.subplots(
            len(ALGORITHMS), 1, figsize=(14, 2.2 * len(ALGORITHMS)), sharex=True
        )

        for ax, (algo_name, algo_fn) in zip(axes_sig, ALGORITHMS.items()):
            sig = algo_fn(_clip)
            t = (np.arange(len(sig)) + _f0) / _cam["fps"]
            ax.plot(t, sig, lw=0.8)
            ax.axhline(0, color="k", lw=0.4, ls="--")
            ax.set_ylabel(algo_name, fontsize=8)
            ax.set_ylim(-260, 260)
            mean_val = sig.mean()
            ax.set_title(
                f"mean={mean_val:.1f}  std={sig.std():.1f}  "
                f"range=[{sig.min():.0f},{sig.max():.0f}]",
                fontsize=8, loc="right",
            )

        axes_sig[-1].set_xlabel("Time (s)")
        fig_sig.suptitle(f"{_cam['name']}  frames {_f0}–{_f1}", fontsize=10)
        fig_sig.tight_layout()
        _result2 = fig_sig

    _result2


@app.cell
def _(mo):
    mo.md("""
    ### All cameras — algorithm overview (first 500 frames each)
    """)
    return


@app.cell
def _(ALGORITHMS, mo):
    _algo_names = list(ALGORITHMS.keys())
    algo_pick = mo.ui.dropdown(
        options=_algo_names,
        value=_algo_names[0],
        label="Algorithm for overview",
    )
    algo_pick
    return (algo_pick,)


@app.cell
def _(CAMERAS, mo):
    _max_frames = max(c["fps"] for c in CAMERAS) * 3600  # generous upper bound
    overview_start = mo.ui.slider(0, 5000, value=0, step=100, label="Overview start frame (per camera)")
    overview_start


@app.cell
def _(ALGORITHMS, CAMERAS, algo_pick, all_frames, mo, np, overview_start, plt):
    _algo_fn = ALGORITHMS[algo_pick.value]
    _f0_ov = overview_start.value
    _N = 6
    fig_all, axes_all = plt.subplots(_N, 1, figsize=(14, 2.5 * _N), sharex=False)

    for _k, (_cam, _frames, _ax) in enumerate(zip(CAMERAS, all_frames, axes_all)):
        if _frames is None:
            _ax.set_title(f"cam{_k} {_cam['name']}: no data")
            continue
        _f1_ov = min(_f0_ov + 500, len(_frames))
        _clip = _frames[_f0_ov:_f1_ov]
        _sig = _algo_fn(_clip)
        _t = (np.arange(len(_sig)) + _f0_ov) / _cam["fps"]
        _ax.plot(_t, _sig, lw=0.7)
        _ax.axhline(0, color="k", lw=0.4, ls="--")
        _ax.set_ylim(-260, 260)
        _ax.set_ylabel(f"cam{_k}\n{_cam['name']}", fontsize=8)
        _ax.set_title(
            f"frames {_f0_ov}–{_f1_ov}  mean={_sig.mean():.1f}  std={_sig.std():.1f}",
            fontsize=8, loc="right",
        )

    axes_all[-1].set_xlabel("Time (s)")
    fig_all.suptitle(f"Algorithm: {algo_pick.value}", fontsize=10)
    fig_all.tight_layout()
    fig_all


if __name__ == "__main__":
    app.run()
