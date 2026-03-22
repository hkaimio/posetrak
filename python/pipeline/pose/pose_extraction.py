"""
pose_extraction.py  —  Marimo app for Stage 1: Person Detection & Pose Extraction

Covers P1b (Marimo port) and P1c (multi-camera in one session).

Usage:
    cd harritests
    marimo edit pose_extraction.py        # interactive editing mode
    marimo run  pose_extraction.py        # read-only app mode

Cell dependency graph (simplified):
  _config_ui ──► _parse_config ──► _run_yolo ──► _stitcher_state ──► _stitcher_ui
                                               └──► _assignment_controls
  _do_assignment ──► (updates timelines state, triggers _stitcher_ui re-render)
  _rtmpose_config ──► _run_rtmpose ──► _confidence_plots
                                    └──► _export_controls ──► _do_export
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _imports():
    import marimo as mo
    import sys
    import hashlib
    import pickle
    import time
    import json
    from pathlib import Path
    import numpy as np
    import cv2
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; marimo renders via HTML
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    sys.path.insert(0, str(Path(__file__).parent))
    from poseanalysis import (
        analyze_video_with_yolo_tracker,
        NamedPersonTimeline,
        VideoData,
        MultiVideoPoseDataset,
        _av_read_single_frame,
        AV_AVAILABLE,
    )
    av_read_frame = _av_read_single_frame  # public alias for cross-cell use
    return (
        AV_AVAILABLE,
        MultiVideoPoseDataset,
        Path,
        VideoData,
        analyze_video_with_yolo_tracker,
        av_read_frame,
        cv2,
        hashlib,
        mo,
        np,
        pickle,
        plt,
        time,
    )


@app.cell
def _cache_helpers(Path, hashlib, pickle):

    def _video_key(video_path: str) -> str:
        p = Path(video_path)
        st = p.stat()
        return hashlib.md5(f"{video_path}:{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:16]

    def load_yolo_cache(video_path: str, cache_dir: Path):
        p = cache_dir / f"yolo_{_video_key(video_path)}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def save_yolo_cache(video_path: str, cache_dir: Path, data):
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"yolo_{_video_key(video_path)}.pkl"
        with open(p, "wb") as f:
            pickle.dump(data, f)

    return load_yolo_cache, save_yolo_cache


@app.cell
def _config_ui(mo):
    project_folder_ui = mo.ui.text(
        value="C:/temp/aikido-2025-11-15-all/Harri_aihanmi_katatedori_ikkyo",
        label="Project folder",
        full_width=True,
    )
    camera_nums_ui = mo.ui.text(
        value="1,2,3,4,5,6",
        label="Camera numbers (comma-separated)",
    )
    persons_ui = mo.ui.text(
        value="Harri,Timo",
        label="Person names (comma-separated)",
    )
    yolo_model_ui = mo.ui.dropdown(
        options={"yolo11n (fast)": "yolo11n.pt", "yolo11x (accurate)": "yolo11x.pt"},
        value="yolo11x (accurate)",
        label="YOLO model",
    )
    tracker_conf_ui = mo.ui.text(
        value="harritests/yolo_tracker_conf.yaml",
        label="Tracker config YAML",
    )
    device_ui = mo.ui.dropdown(
        options=["cuda:0", "cuda:1", "cpu"],
        value="cuda:0",
        label="Device",
    )
    run_yolo_btn = mo.ui.run_button(label="▶  Run YOLO tracking for all cameras")

    mo.vstack([
        mo.md("# 🎬 Pose Extraction"),
        mo.md("## Configuration"),
        mo.hstack([
            mo.vstack([project_folder_ui, camera_nums_ui, persons_ui]),
            mo.vstack([yolo_model_ui, tracker_conf_ui, device_ui]),
        ]),
        run_yolo_btn,
    ])
    return (
        camera_nums_ui,
        device_ui,
        persons_ui,
        project_folder_ui,
        run_yolo_btn,
        tracker_conf_ui,
        yolo_model_ui,
    )


@app.cell
def _parse_config(
    Path,
    camera_nums_ui,
    device_ui,
    persons_ui,
    project_folder_ui,
    tracker_conf_ui,
    yolo_model_ui,
):
    project_folder = Path(project_folder_ui.value.strip())
    video_folder   = project_folder / "videos"
    json_folder    = project_folder / "pose"
    cache_dir      = project_folder / ".pose_cache"
    camera_nums    = [int(x.strip()) for x in camera_nums_ui.value.split(",") if x.strip()]
    named_persons  = [x.strip() for x in persons_ui.value.split(",") if x.strip()]
    yolo_model     = yolo_model_ui.value
    tracker_conf   = tracker_conf_ui.value.strip() or None
    device         = device_ui.value
    return (
        cache_dir,
        camera_nums,
        device,
        json_folder,
        named_persons,
        project_folder,
        tracker_conf,
        video_folder,
        yolo_model,
    )


@app.cell
def _run_yolo(
    analyze_video_with_yolo_tracker,
    cache_dir,
    camera_nums,
    device,
    load_yolo_cache,
    mo,
    named_persons,
    run_yolo_btn,
    save_yolo_cache,
    time,
    tracker_conf,
    video_folder,
    yolo_model,
):
    mo.stop(not run_yolo_btn.value, mo.md(
        "_Click **Run YOLO tracking** above to start._"
    ))

    yolo_data = {}
    _msgs = []

    for _cam in camera_nums:
        _vpath = str(video_folder / f"cam{_cam}.mp4")
        _cached = load_yolo_cache(_vpath, cache_dir)
        if _cached is not None:
            yolo_data[_cam] = _cached
            _msgs.append(f"- **cam{_cam}**: loaded from cache ✓")
        else:
            _t0 = time.perf_counter()
            _tl, _nf = analyze_video_with_yolo_tracker(
                _vpath, named_persons,
                tracker_config=tracker_conf,
                model_name=yolo_model,
                device=device,
            )
            yolo_data[_cam] = (_tl, _nf)
            save_yolo_cache(_vpath, cache_dir, (_tl, _nf))
            _e = time.perf_counter() - _t0
            _msgs.append(
                f"- **cam{_cam}**: {_nf} frames, {len(_tl.detections)} tracks"
                f" in {_e:.1f}s ✓"
            )

    mo.callout(
        mo.md("**YOLO tracking complete**\n\n" + "\n".join(_msgs) +
              "\n\n_Assign persons in the stitcher below._"),
        kind="success",
    )
    return (yolo_data,)


@app.cell
def _stitcher_state(mo, yolo_data):
    _initial = {cam: tl for cam, (tl, _) in yolo_data.items()}
    timelines, set_timelines = mo.state(_initial)
    return set_timelines, timelines


@app.cell
def _stitcher_cam_ui(mo, yolo_data):
    mo.stop(not yolo_data)
    _cams = [f"cam{c}" for c in sorted(yolo_data.keys())]
    stitch_cam_ui = mo.ui.dropdown(options=_cams, value=_cams[0], label="Camera")
    mo.vstack([mo.md("## Person–Track Stitcher"), stitch_cam_ui])
    return (stitch_cam_ui,)


@app.cell
def _stitcher_heatmap_data(
    mo,
    named_persons,
    np,
    stitch_cam_ui,
    timelines,
    yolo_data,
):
    """Build integer z-matrix + colorscale for go.Heatmap (one value per person)."""
    mo.stop(not yolo_data)
    _cam     = int(stitch_cam_ui.value.replace("cam", ""))
    _tl_dict = timelines()
    _tl      = _tl_dict.get(_cam)
    _nf      = yolo_data[_cam][1] if _cam in yolo_data else 0
    _yolo_ids = sorted(_tl.detections.keys()) if _tl else []

    # z value: 0=absent, 1=unassigned, 2+N=person N
    _n_persons = len(named_persons)
    _p_idx = {p: i for i, p in enumerate(named_persons)}
    _z = None
    _colorscale = None
    if _tl and _yolo_ids and _nf > 0:
        _z = np.zeros((len(_yolo_ids), _nf), dtype=np.float32)
        for _yi, _yid in enumerate(_yolo_ids):
            for _fn, _dr in _tl.detections[_yid].items():
                if _fn < _nf:
                    _z[_yi, _fn] = (
                        _p_idx[_dr.person_name] + 2
                        if _dr.person_name in _p_idx
                        else 1
                    )
        # build discrete colorscale: levels 0..n_persons+1
        _levels = _n_persons + 2
        def _hex(name):
            c = _tl.person_colors.get(name, _tl.person_colors[None])
            import matplotlib.colors as _mc
            r, g, b, _ = _mc.to_rgba(c)
            return f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"
        _colorscale = [
            [0,             "rgb(240,240,240)"],   # absent
            [1/(_levels-1), _hex(None)],           # unassigned
        ]
        for _i, _p in enumerate(named_persons):
            _colorscale.append([(2+_i)/(_levels-1), _hex(_p)])
        # ensure last stop == 1.0
        _colorscale[-1][0] = 1.0

    cam_heatmap_data = (_yolo_ids, _nf, _z, _colorscale)
    return (cam_heatmap_data,)


@app.cell
def _stitcher_sel_state(mo, yolo_data):
    mo.stop(not yolo_data)
    sel_frame, set_sel_frame = mo.state(0)
    sel_yid,   set_sel_yid   = mo.state(-1)   # -1 = not yet set
    return sel_frame, sel_yid, set_sel_frame, set_sel_yid


@app.cell
def _stitcher_heatmap_cell(cam_heatmap_data, mo, sel_frame, sel_yid):
    """Build plotly heatmap widget. Returns it; never reads .value here."""
    import plotly.graph_objects as _pgo

    _yolo_ids, _nf, _z, _colorscale = cam_heatmap_data
    mo.stop(not _yolo_ids or _z is None, mo.md("_Run YOLO tracking first._"))

    _n   = len(_yolo_ids)
    _sf  = min(sel_frame(), max(_nf - 1, 0))
    _syid = sel_yid() if sel_yid() in _yolo_ids else _yolo_ids[0]
    _syi = _yolo_ids.index(_syid)

    # Use sequential y-indices 0..n-1 so every row has equal height.
    # customdata carries the real YOLO ID for the hover label.
    _y_idx = list(range(_n))
    _row_h = 28   # px per row
    _fig_h = min(max(150, _n * _row_h + 60), 800)

    _fig = _pgo.Figure(_pgo.Heatmap(
        z=_z,
        x=list(range(_nf)),
        y=_y_idx,
        colorscale=_colorscale,
        zmin=0, zmax=len(_colorscale) - 1,
        showscale=False,
        customdata=[[_yolo_ids[i]] * _nf for i in range(_n)],
        hovertemplate="frame %{x}<br>ID %{customdata}<extra></extra>",
    ))
    # red vline — spans all rows (y in sequential index space)
    _fig.add_shape(type="line",
        x0=_sf, x1=_sf, y0=-0.5, y1=_n - 0.5,
        line=dict(color="red", width=2), xref="x", yref="y")
    # selected-row highlight (also in index space)
    _fig.add_shape(type="rect",
        x0=-0.5, x1=_nf - 0.5,
        y0=_syi - 0.5, y1=_syi + 0.5,
        line=dict(color="rgba(0,0,0,0.4)", width=1),
        fillcolor="rgba(255,255,255,0.15)",
        xref="x", yref="y")
    _fig.update_layout(
        height=_fig_h,
        margin=dict(l=55, r=4, t=10, b=36),
        xaxis=dict(title="Frame", showgrid=False),
        yaxis=dict(
            tickmode="array",
            tickvals=_y_idx,
            ticktext=[str(y) for y in _yolo_ids],
            autorange="reversed",
            showgrid=False,
            range=[-0.5, _n - 0.5],   # lock range so vline always reaches edge
        ),
        clickmode="event", dragmode=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
    )
    heatmap_plot = mo.ui.plotly(_fig)
    return (heatmap_plot,)


@app.cell
def _stitcher_view(
    AV_AVAILABLE,
    av_read_frame,
    cam_heatmap_data,
    cv2,
    heatmap_plot,
    mo,
    named_persons,
    sel_frame,
    sel_yid,
    set_sel_frame,
    set_sel_yid,
    set_timelines,
    stitch_cam_ui,
    timelines,
    video_folder,
):
    import base64 as _b64

    _cam      = int(stitch_cam_ui.value.replace("cam", ""))
    _yolo_ids, _nf, _z, _colorscale = cam_heatmap_data
    mo.stop(not _yolo_ids or _z is None)

    # ── resolve selection (click overrides state) ─────────────────────────
    _cur_frame = min(sel_frame(), max(_nf - 1, 0))
    _cur_yid   = sel_yid() if sel_yid() in _yolo_ids else _yolo_ids[0]

    _pts = heatmap_plot.value or []
    if _pts:
        _pt = _pts[0]
        _cf = max(0, min(_nf - 1, int(round(float(_pt.get("x", _cur_frame))))))
        # y is now a sequential index; map back to YOLO ID
        _yi_raw = int(round(float(_pt.get("y", _yolo_ids.index(_cur_yid)))))
        _cy = _yolo_ids[_yi_raw] if 0 <= _yi_raw < len(_yolo_ids) else _cur_yid
        if _cy not in _yolo_ids:
            _cy = _cur_yid
        if _cf != _cur_frame or _cy != _cur_yid:
            set_sel_frame(_cf)
            set_sel_yid(_cy)
            _cur_frame, _cur_yid = _cf, _cy

    _tl_dict = timelines()
    _tl      = _tl_dict.get(_cam)

    # ── crop image ────────────────────────────────────────────────────────
    _crop_html = ""
    if _tl and _cur_yid in _tl.detections:
        _dets = _tl.detections[_cur_yid]
        _det  = _dets.get(_cur_frame)
        _uf   = _cur_frame
        if _det is None and _dets:
            _uf  = min(_dets.keys(), key=lambda f: abs(f - _cur_frame))
            _det = _dets[_uf]
        if _det and AV_AVAILABLE:
            _img = av_read_frame(str(video_folder / f"cam{_cam}.mp4"), _uf)
            if _img is not None:
                _cx, _cy2, _bw, _bh = _det.x, _det.y, _det.w, _det.h
                _x1 = max(0, int(_cx - _bw/2));  _x2 = min(_img.shape[1], int(_cx + _bw/2))
                _y1 = max(0, int(_cy2 - _bh/2)); _y2 = min(_img.shape[0], int(_cy2 + _bh/2))
                _crop = _img[_y1:_y2, _x1:_x2]
                if _crop.size > 0:
                    _th = 220
                    _sc = _th / max(_crop.shape[0], 1)
                    _crop = cv2.resize(_crop, (max(1, int(_crop.shape[1]*_sc)), _th))
                    _ok, _cbuf = cv2.imencode(".jpg", _crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if _ok:
                        _enc = _b64.b64encode(_cbuf.tobytes()).decode()
                        _crop_html = (
                            f'<img src="data:image/jpeg;base64,{_enc}" '
                            f'style="max-width:100%;border-radius:6px;margin:4px 0;display:block">'
                        )

    # ── assignment state ──────────────────────────────────────────────────
    _assigned = sorted({
        d.person_name
        for d in (_tl.detections.get(_cur_yid) or {}).values()
        if d.person_name
    }) if _tl else []
    _assigned_str = ", ".join(_assigned) or "—"

    from_here_sw = mo.ui.switch(label="From here onward", value=False)

    def _assign_cb(_person, _yid=_cur_yid, _cn=_cam):
        def _cb(_):
            _cur = dict(timelines())
            _t = _cur.get(_cn)
            if _t and _yid in _t.detections:
                _from = sel_frame() if from_here_sw.value else 0
                _t.assign_yolo_to_person(_yid, _person, from_frame=_from)
                _cur[_cn] = _t
                set_timelines(_cur)
        return _cb

    def _clear_cb(_, _yid=_cur_yid, _cn=_cam):
        _cur = dict(timelines())
        _t = _cur.get(_cn)
        if _t and _yid in _t.detections:
            _t.assign_yolo_to_person(_yid, None)
            _cur[_cn] = _t
            set_timelines(_cur)

    _assign_btns = [
        mo.ui.button(
            label=p,
            on_change=_assign_cb(p),
            kind="success" if p in _assigned else "neutral",
        )
        for p in named_persons
    ]
    _clear_btn = mo.ui.button(label="✕ clear", on_change=_clear_cb, kind="danger")

    # ── right panel ───────────────────────────────────────────────────────
    _right = mo.vstack([
        mo.Html(
            f'<div style="font-size:0.9em;line-height:1.6;padding-bottom:4px">'
            f'<b style="font-size:1.1em">ID {_cur_yid}</b>'
            f' <span style="color:#888">frame {_cur_frame}</span><br>'
            f'<span style="color:#555">→ {_assigned_str}</span></div>'
        ),
        from_here_sw,
        mo.hstack(_assign_btns + [_clear_btn], gap="0.4rem", wrap=True),
        mo.Html(_crop_html) if _crop_html else mo.md(""),
    ], gap="0.4rem")

    # ── layout: heatmap takes ~75 %, detail panel ~25 % ──────────────────
    mo.Html(
        f'<div style="display:flex;gap:1rem;align-items:flex-start">'
        f'<div style="flex:3;min-width:0">{mo.as_html(heatmap_plot).text}</div>'
        f'<div style="flex:1;min-width:240px">{mo.as_html(_right).text}</div>'
        f'</div>'
    )
    return


@app.cell
def _rtmpose_config(mo):
    # Each option value is a dict with 'url' and 'input_size' (H, W).
    _MODELS = {
        "RTMPose-l (133 kp, recommended)": {
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
            "input_size": (288, 384),
        },
        "ViTPose++-l (133 kp)": {
            "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx",
            "input_size": (192, 256),
        },
    }
    rtm_model_ui = mo.ui.dropdown(
        options=_MODELS,
        value="RTMPose-l (133 kp, recommended)",
        label="RTMPose model",
    )
    rtm_device_ui = mo.ui.dropdown(
        options=["cuda",  "cpu"], value="cuda",
        label="RTMPose device",
    )
    conf_thr_ui = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3,
        label="Confidence threshold (plot guideline)",
    )
    run_rtmpose_btn = mo.ui.run_button(label="▶  Run RTMPose for all cameras")

    mo.vstack([
        mo.md("## RTMPose — Pose Estimation"),
        mo.md("_Complete person assignments above first._"),
        mo.hstack([rtm_model_ui, rtm_device_ui, conf_thr_ui]),
        run_rtmpose_btn,
    ])
    return conf_thr_ui, rtm_device_ui, rtm_model_ui, run_rtmpose_btn


@app.cell
def _run_rtmpose(
    VideoData,
    camera_nums,
    mo,
    named_persons,
    rtm_device_ui,
    rtm_model_ui,
    run_rtmpose_btn,
    time,
    timelines,
    video_folder,
):
    mo.stop(not run_rtmpose_btn.value, mo.md(
        "_Assign persons, then click **Run RTMPose**._"
    ))

    from rtmlib.tools.pose_estimation import RTMPose
    from rtmlib.tools.object_detection import YOLOX

    _model_cfg = rtm_model_ui.value  # dict with 'url' and 'input_size'
    _pose_model = RTMPose(
        _model_cfg["url"],
        model_input_size=_model_cfg["input_size"],
        to_openpose=False,
        backend="onnxruntime",
        device=rtm_device_ui.value,
    )
    _det_model = YOLOX(
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        model_input_size=(640, 640),
        backend="onnxruntime",
        device=rtm_device_ui.value,
    )

    rtmpose_data = {}
    _msgs = []

    for _cam in camera_nums:
        _tl = timelines().get(_cam)
        if _tl is None:
            _msgs.append(f"- **cam{_cam}**: no timeline — skipped")
            continue
        _assigned = sum(len(_tl.timelines.get(n, {})) for n in named_persons)
        if _assigned == 0:
            _msgs.append(f"- **cam{_cam}**: ⚠ no persons assigned — skipped")
            continue
        _t0 = time.perf_counter()
        _vdata = VideoData(
            video_path=str(video_folder / f"cam{_cam}.mp4"),
            det_model=_det_model,
            pose_model=_pose_model,
            start_frame=1,
            end_frame=None,
            named_person_timeline=_tl,
        )
        _e = time.perf_counter() - _t0
        rtmpose_data[_cam] = _vdata
        _msgs.append(f"- **cam{_cam}**: {_vdata.get_frame_count()} frames in {_e:.1f}s ✓")

    mo.callout(
        mo.md("**RTMPose complete**\n\n" + "\n".join(_msgs)), kind="success"
    )
    return (rtmpose_data,)


@app.cell
def _confidence_plots(conf_thr_ui, mo, named_persons, np, plt, rtmpose_data):
    mo.stop(not rtmpose_data)

    figs = []
    for _cam, _vd in sorted(rtmpose_data.items()):
        _fns = _vd.get_frame_numbers()
        if not _fns:
            continue
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.axhline(conf_thr_ui.value, color="gray", lw=0.8, ls="--",
                   label=f"thr {conf_thr_ui.value:.2f}")
        for _p in named_persons:
            _c = [
                float(np.mean(_vd.frame_data[f][_p].scores))
                if f in _vd.frame_data and _p in _vd.frame_data[f]
                else float("nan")
                for f in _fns
            ]
            ax.plot(_fns, _c, label=_p, lw=0.8, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean keypoint score")
        ax.set_title(f"cam{_cam} — Pose confidence over time")
        ax.legend(fontsize=8)
        plt.tight_layout()
        figs.append(mo.as_html(fig))
        plt.close(fig)

    conf_plots = mo.vstack([
        mo.md("## Confidence over time (R1.2)"),
        mo.md(
            "_Dips below the dashed threshold indicate frames needing "
            "bounding-box correction._"
        ),
        *figs,
    ])
    conf_plots
    return


@app.cell
def _pose_viewer_ui(mo, rtmpose_data):
    mo.stop(not rtmpose_data)
    _cams = [f"cam{c}" for c in sorted(rtmpose_data.keys())]
    _vd0  = rtmpose_data[sorted(rtmpose_data.keys())[0]]
    _fns0 = _vd0.get_frame_numbers()

    pose_cam_ui = mo.ui.dropdown(
        options=_cams, value=_cams[0], label="Camera",
    )
    pose_frame_ui = mo.ui.slider(
        start=min(_fns0), stop=max(_fns0), step=1, value=_fns0[0],
        label="Frame", full_width=True,
    )
    pose_thr_ui = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3,
        label="Keypoint threshold",
    )

    mo.vstack([
        mo.md("## Pose Viewer & Correction"),
        mo.hstack([pose_cam_ui, pose_thr_ui]),
        pose_frame_ui,
    ])
    return pose_cam_ui, pose_frame_ui, pose_thr_ui


@app.cell
def _pose_viewer(
    AV_AVAILABLE,
    cv2,
    mo,
    named_persons,
    pose_cam_ui,
    pose_frame_ui,
    pose_thr_ui,
    rtmpose_data,
):
    import base64 as _b64

    mo.stop(not rtmpose_data)

    _cam = int(pose_cam_ui.value.replace("cam", ""))
    _vd  = rtmpose_data.get(_cam)
    mo.stop(_vd is None, mo.md(f"_No data for {pose_cam_ui.value}._"))

    _fns = _vd.get_frame_numbers()
    _fn  = pose_frame_ui.value
    # Snap to nearest available frame number
    if _fn not in _vd.frame_data:
        _fn = min(_fns, key=lambda f: abs(f - _fn))

    _thr = pose_thr_ui.value

    def _to_jpg_b64(img_bgr, max_h=500):
        _h, _w = img_bgr.shape[:2]
        if _h > max_h:
            _sc  = max_h / _h
            img_bgr = cv2.resize(img_bgr, (int(_w * _sc), max_h))
        _ok, _buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return _b64.b64encode(_buf.tobytes()).decode() if _ok else ""

    # ── Main frame with all skeletons ────────────────────────────────────
    _main_enc = ""
    try:
        _main_img = _vd.draw_frame(_fn, thr=_thr)
        _main_enc = _to_jpg_b64(_main_img, max_h=540)
    except Exception as _e:
        pass

    # ── Per-person crops ─────────────────────────────────────────────────
    _person_cards = []
    for _p in named_persons:
        if _fn in _vd.frame_data and _p in _vd.frame_data[_fn]:
            _pd = _vd.frame_data[_fn][_p]
            _mean_score = float(
                __import__("numpy").mean(_pd.scores)
            )
            try:
                _crop_img = _vd.draw_person(_fn, _p, thr=_thr)
                _enc = _to_jpg_b64(_crop_img, max_h=280)
                _card = mo.Html(
                    f'<div style="text-align:center">'
                    f'<div style="font-weight:600;margin-bottom:4px">{_p}'
                    f' <small style="font-weight:normal;color:#888">'
                    f'conf&nbsp;{_mean_score:.2f}</small></div>'
                    f'<img src="data:image/jpeg;base64,{_enc}" '
                    f'style="max-height:280px;border-radius:6px">'
                    f'</div>'
                )
            except Exception:
                _card = mo.md(f"**{_p}** — _(crop error)_")
        else:
            _card = mo.Html(
                f'<div style="text-align:center;color:#aaa;padding:8px">'
                f'<b>{_p}</b><br><small>not detected</small></div>'
            )
        _person_cards.append(_card)

    _main_html = (
        f'<img src="data:image/jpeg;base64,{_main_enc}" '
        f'style="width:100%;border-radius:6px">'
        if _main_enc else '<div style="color:#aaa">_(no image)_</div>'
    )

    mo.Html(
        f'<div style="display:flex;gap:1rem;align-items:flex-start">'
        f'<div style="flex:3;min-width:0">{_main_html}</div>'
        f'<div style="flex:1;min-width:200px">'
        + "".join(
            f'<div style="margin-bottom:12px">{mo.as_html(c).text}</div>'
            for c in _person_cards
        )
        + "</div></div>"
    )
    return


@app.cell
def _export_controls(mo, rtmpose_data):
    mo.stop(not rtmpose_data)
    export_btn = mo.ui.run_button(label="💾  Export all cameras → OpenPose JSON + HDF5")
    mo.vstack([mo.md("## Export"), export_btn])
    return (export_btn,)


@app.cell
def _do_export(
    MultiVideoPoseDataset,
    export_btn,
    json_folder,
    mo,
    project_folder,
    rtmpose_data,
):
    mo.stop(not export_btn.value)

    json_folder.mkdir(parents=True, exist_ok=True)
    _ds = MultiVideoPoseDataset(str(project_folder / "data.h5"))
    _msgs = []

    for _cam, _vd in sorted(rtmpose_data.items()):
        _name = f"cam{_cam}"
        _ds.save_video_data(_vd, video_name=_name)
        _ds.export_video_to_openpose_json(_name, str(json_folder))
        _msgs.append(f"- **cam{_cam}**: → `{json_folder / (_name + '_json')}/` ✓")

    mo.callout(
        mo.md("## Export complete\n\n" + "\n".join(_msgs)), kind="success"
    )
    return


if __name__ == "__main__":
    app.run()
