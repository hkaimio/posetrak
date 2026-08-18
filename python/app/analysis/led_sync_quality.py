# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    return Path, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## LED sync quality analysis — per-camera event and drift diagnostics
    """)
    return


@app.cell
def _(Path, mo):
    _default = str(Path(__file__).parents[3] / "cpp/tests/data/led_brightness_04.npz")
    npz_path = mo.ui.text(value=_default, label="NPZ path", full_width=True)
    npz_path
    return (npz_path,)


@app.cell
def _(npz_path):
    from app.setup.led_sync import (
        load_brightness_dump, run_led_sync, detect_events,
        ransac_affine_fit, dtw_match_event_times,
    )
    dump = load_brightness_dump(npz_path.value)
    result = run_led_sync(**dump)

    CAM_NAMES = [c.split("-", 4)[-1] if "-" in c else c for c in dump["cam_ids"]]
    K = len(dump["signals"])
    LED_PERIOD_S = 0.197   # approximate; refined below
    print(f"Loaded {K} cameras")
    for _k in range(K):
        _cr = result.cameras[_k]
        print(f"  cam{_k} {CAM_NAMES[_k]}: {_cr.n_inliers} inliers/{_cr.n_pairs} pairs  "
              f"map={_cr.map_type}  resid={_cr.resid_std_s*1000:.1f}ms")
    return (
        CAM_NAMES,
        K,
        LED_PERIOD_S,
        detect_events,
        dump,
        ransac_affine_fit,
        result,
    )


@app.cell
def _(CAM_NAMES, K, LED_PERIOD_S, detect_events, dump, np):
    EVENT_CFG = dict(min_sep_s=0.15, prominence=1.5, polarity="both", smooth_win=5)

    all_events = []
    for _k in range(K):
        _sig = dump["signals"][_k]
        _fps = dump["fps_list"][_k]
        _ev = detect_events(_sig, _fps, **EVENT_CFG)
        all_events.append(_ev)
        _dur = len(_sig) / _fps
        _expected = _dur / (LED_PERIOD_S / 2)  # both transitions per cycle
        _gaps = np.diff(_ev) if len(_ev) > 1 else np.array([])
        _long_gaps = _gaps[_gaps > 2 * LED_PERIOD_S]
        print(f"  cam{_k} {CAM_NAMES[_k]}: {len(_ev)} events "
              f"({len(_ev)/_expected*100:.0f}% of expected), "
              f"{len(_long_gaps)} gaps > 2×period  "
              f"({_long_gaps.sum():.1f}s total occluded)")
    return (all_events,)


@app.cell
def _(all_events, dump, np):
    _ref = dump["ref_cam"]
    _ev = all_events[_ref]
    if len(_ev) > 1:
        _ivals = np.diff(_ev)
        # LED half-period (between consecutive ON→OFF or OFF→ON transitions)
        _half = np.median(_ivals[_ivals < 0.5])   # exclude long gaps
        LED_PERIOD_REFINED = float(_half * 2)
        print(f"Refined LED period: {LED_PERIOD_REFINED*1000:.1f} ms  "
              f"(half-period {_half*1000:.1f} ms)")
    else:
        LED_PERIOD_REFINED = 0.197
    return (LED_PERIOD_REFINED,)


@app.cell(hide_code=True)
def _():
    # placeholder so the cell above can reference this name without circular dep
    all_frames_events_note = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Inter-event intervals over time
    """)
    return


@app.cell
def _(CAM_NAMES, K, LED_PERIOD_REFINED, all_events, dump, np, plt):
    fig_iev, axes_iev = plt.subplots(K, 1, figsize=(14, 2.5 * K), sharex=False)

    for _k in range(K):
        _ax = axes_iev[_k]
        _ev = all_events[_k]
        _rough = dump["rough_offsets"][_k]
        if len(_ev) < 2:
            _ax.set_title(f"cam{_k} {CAM_NAMES[_k]}: too few events")
            continue

        _ivals = np.diff(_ev)
        _t_mid = (_ev[:-1] + _rough)   # global-time midpoint of each interval

        # colour: green = near LED half-period, red = long gap (occlusion)
        _expected_half = LED_PERIOD_REFINED / 2
        _colors = np.where(_ivals > 3 * _expected_half, "red",
                  np.where(_ivals > 1.5 * _expected_half, "orange", "steelblue"))

        _ax.vlines(_t_mid, 0, _ivals * 1000, colors=_colors, linewidth=0.6, alpha=0.7)
        _ax.axhline(LED_PERIOD_REFINED / 2 * 1000, color="k", lw=0.8, ls="--",
                    label=f"half-period {LED_PERIOD_REFINED/2*1000:.0f} ms")
        _ax.axhline(LED_PERIOD_REFINED * 3 / 2 * 1000, color="orange", lw=0.6, ls=":",
                    label="1.5× half-period")
        _ax.set_ylabel(f"cam{_k}\n{CAM_NAMES[_k]}", fontsize=8)
        _ax.set_ylim(0, min(_ivals.max() * 1000 * 1.1, 2000))
        _ax.set_xlabel("Global time (s)")

        _long = _ivals[_ivals > 3 * _expected_half]
        _ax.set_title(
            f"{len(_ev)} events  |  {len(_long)} occlusion gaps  "
            f"({_long.sum():.1f}s total)",
            fontsize=8, loc="right",
        )
        if _k == 0:
            _ax.legend(fontsize=7, loc="upper right")

    fig_iev.suptitle("Inter-event intervals  (red = likely occlusion > 3× half-period)", fontsize=10)
    fig_iev.tight_layout()
    fig_iev
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Rolling fps estimate from sync map  (deviation from nominal = clock drift or fit error)
    """)
    return


@app.cell
def _(CAM_NAMES, K, dump, np, plt, result):
    _WINDOW_S = 5.0   # seconds of camera time per estimate

    fig_fps, axes_fps = plt.subplots(K, 1, figsize=(14, 2.2 * K), sharex=False)

    for _k in range(K):
        _ax = axes_fps[_k]
        _cr = result.cameras[_k]
        _fps_nom = dump["fps_list"][_k]
        _ft = _cr.frame_times
        _N = len(_ft)
        _win = max(10, int(_WINDOW_S * _fps_nom))

        _t_mid_r, _fps_local = [], []
        for _i in range(0, _N - _win, _win // 2):
            _j = min(_i + _win, _N - 1)
            _dt_global = _ft[_j] - _ft[_i]
            if abs(_dt_global) > 1e-6:
                _fps_est = (_j - _i) / _dt_global
                _t_mid_r.append((_ft[_i] + _ft[_j]) / 2)
                _fps_local.append(_fps_est)

        if _fps_local:
            _t_arr = np.array(_t_mid_r)
            _f_arr = np.array(_fps_local)
            _ax.plot(_t_arr, _f_arr, lw=0.8, color="steelblue")
            _ax.axhline(_fps_nom, color="k", lw=0.8, ls="--", label=f"nominal {_fps_nom} fps")
            _ax.set_ylim(_fps_nom * 0.97, _fps_nom * 1.03)
            _ax.set_ylabel(f"cam{_k}\n{CAM_NAMES[_k]}", fontsize=8)
            _ax.set_xlabel("Global time (s)")
            _deviation = np.std(_f_arr - _fps_nom)
            _ax.set_title(f"fps std dev: {_deviation:.3f}  map={_cr.map_type}", fontsize=8, loc="right")
            if _k == 0:
                _ax.legend(fontsize=7)

    fig_fps.suptitle(f"Rolling {_WINDOW_S:.0f}s fps estimate  (flat = good clock, drift = sync error)", fontsize=10)
    fig_fps.tight_layout()
    fig_fps
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### RANSAC pair residuals over time  (growing residuals = fit diverges)
    """)
    return


@app.cell
def _(CAM_NAMES, K, all_events, dump, np, plt, ransac_affine_fit):
    _ref = dump["ref_cam"]
    _t_ref = all_events[_ref]
    _fps_ref = dump["fps_list"][_ref]

    fig_res, axes_res = plt.subplots(K - 1, 1, figsize=(14, 2.5 * (K - 1)), sharex=False)
    if K - 1 == 1:
        axes_res = [axes_res]

    _ax_idx = 0
    for _k in range(K):
        if _k == _ref:
            continue
        _ax = axes_res[_ax_idx]
        _ax_idx += 1

        _t_cam = all_events[_k]
        _offset = dump["rough_offsets"][_k]
        _cam_shifted = _t_cam + _offset

        # NN matching (same logic as _sync_one_camera)
        _pairs = []
        for _j, _tc in enumerate(_cam_shifted):
            _ins = int(np.searchsorted(_t_ref, _tc))
            _best_i, _best_d = -1, 1.5
            for _ic in (_ins - 1, _ins):
                if 0 <= _ic < len(_t_ref):
                    _d = abs(_t_ref[_ic] - _tc)
                    if _d < _best_d:
                        _best_d, _best_i = _d, _ic
            if _best_i >= 0:
                _pairs.append((_best_i, _j))

        if not _pairs:
            _ax.set_title(f"cam{_k} {CAM_NAMES[_k]}: no pairs")
            continue

        _pairs_arr = np.array(_pairs)
        _A = _t_ref[_pairs_arr[:, 0]]
        _B = _t_cam[_pairs_arr[:, 1]]
        (_, _b), _inliers = ransac_affine_fit(_A, _B, max_err_s=0.01)

        _resid_all = _A - (_B + _offset)   # residual in global time
        _colors = np.where(np.isin(np.arange(len(_A)), _inliers), "steelblue", "red")

        _ax.scatter(_A, _resid_all * 1000, c=_colors, s=4, alpha=0.6)
        _ax.axhline(0, color="k", lw=0.6)
        _ax.set_ylabel(f"cam{_k}\n{CAM_NAMES[_k]}", fontsize=8)
        _ax.set_xlabel("Reference time (s)")
        _ax.set_ylim(-300, 300)
        _ax.set_title(
            f"{len(_inliers)}/{len(_A)} inliers  (blue=inlier, red=outlier)",
            fontsize=8, loc="right",
        )

    fig_res.suptitle("Pair residuals: ref_event_time − (cam_event_time + rough_offset)", fontsize=10)
    fig_res.tight_layout()
    fig_res
    return


if __name__ == "__main__":
    app.run()
