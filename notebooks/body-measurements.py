import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path
    import json
    import yaml
    import math
    import tomllib
    return Path, go, json, make_subplots, math, mo, np, pd, tomllib, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Key body measurements (triangulated from inlier observations)
    """)
    return


@app.cell
def _(mo):
    config_input = mo.ui.text(
        value="",
        label="Tracking config TOML",
        full_width=True,
    )
    config_input
    return (config_input,)


@app.cell
def _(Path, config_input, mo, tomllib):
    _p = Path(config_input.value)
    if not config_input.value or not _p.exists():
        cameras_path = Path("/dev/null")
        skeleton_path = Path("/dev/null")
        output_dir = Path("/dev/null")
        mo.stop(True, mo.callout(mo.md(f"Config not found: `{_p}`"), kind="danger"))
    else:
        with open(_p, "rb") as _f:
            _cfg = tomllib.load(_f)
        cameras_path = Path(_cfg["data"]["cameras"])
        skeleton_path = Path(_cfg["data"]["skeleton"])
        output_dir = Path(_cfg["output"]["directory"])
    return cameras_path, output_dir, skeleton_path


@app.cell(hide_code=True)
def _(math, np):
    def _rodrigues(rvec):
        v = np.array(rvec, dtype=float)
        angle = float(np.linalg.norm(v))
        if angle < 1e-10:
            return np.eye(3)
        ax = v / angle
        c, s = math.cos(angle), math.sin(angle)
        t = 1.0 - c
        x, y, z = ax
        return np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
        ])

    def load_cameras(toml_path, tomllib):
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        cameras = {}
        for key, vals in data.items():
            if not key.startswith("cam") or key == "metadata":
                continue
            try:
                cam_id = int(key[3:]) - 1  # cam1 → 0, cam2 → 1, …
            except ValueError:
                continue
            K = np.array(vals["matrix"], dtype=float)
            R = _rodrigues(vals["rotation"])
            t = np.array(vals["translation"], dtype=float)
            dist = np.array(vals.get("distortions", [0.0, 0.0, 0.0, 0.0]), dtype=float)
            P = K @ np.hstack([R, t.reshape(3, 1)])
            cameras[cam_id] = {"K": K, "dist": dist, "P": P}
        return cameras

    def undistort_point(px, py, K, dist):
        k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        if abs(k1) < 1e-9 and abs(k2) < 1e-9 and abs(p1) < 1e-9 and abs(p2) < 1e-9:
            return px, py
        xn = (px - cx) / fx
        yn = (py - cy) / fy
        x0, y0 = xn, yn
        for _ in range(5):
            r2 = xn*xn + yn*yn
            radial = 1.0 + k1*r2 + k2*r2*r2
            dx = 2.0*p1*xn*yn + p2*(r2 + 2.0*xn*xn)
            dy = p1*(r2 + 2.0*yn*yn) + 2.0*p2*xn*yn
            xn = (x0 - dx) / radial
            yn = (y0 - dy) / radial
        return xn*fx + cx, yn*fy + cy

    def triangulate_dlt(observations, Ps):
        rows = []
        for (u, v), P in zip(observations, Ps):
            rows.append(u * P[2] - P[0])
            rows.append(v * P[2] - P[1])
        A = np.array(rows, dtype=float)
        _, s, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pos = X[:3] / X[3]
        cond = float(s[0] / s[-2]) if s[-2] > 1e-12 else float("inf")
        return pos, cond
    return load_cameras, triangulate_dlt, undistort_point


@app.cell(hide_code=True)
def _(math, np):
    def _rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _R_from_zyx(zyx):
        if not zyx:
            return np.eye(3)
        return _rx(zyx[2]) @ _ry(zyx[1]) @ _rz(zyx[0])

    def fk_rest_pose(joints: list) -> dict:
        """Return {joint_name: world_position_array} at rest pose (all angles = 0)."""
        by_name = {j["name"]: j for j in joints}
        ordered, visited = [], set()

        def visit(n):
            if n in visited:
                return
            visited.add(n)
            p = by_name[n].get("parent")
            if p:
                visit(p)
            ordered.append(n)

        for j in joints:
            visit(j["name"])

        transforms = {}
        for name in ordered:
            jnt = by_name[name]
            offset = np.array(jnt.get("offset") or [0.0, 0.0, 0.0], dtype=float)
            R = _R_from_zyx(jnt.get("orientation"))
            parent = jnt.get("parent")
            if parent is None:
                transforms[name] = (offset.copy(), R)
            else:
                p_pos, p_R = transforms[parent]
                transforms[name] = (p_pos + p_R @ offset, p_R @ R)

        return {n: pos for n, (pos, _) in transforms.items()}
    return (fk_rest_pose,)


@app.cell
def _(cameras_path, load_cameras, mo, tomllib):
    if not cameras_path.exists():
        cameras = {}
        mo.stop(True, mo.callout(mo.md(f"Cameras not found: `{cameras_path}`"), kind="danger"))
    else:
        cameras = load_cameras(cameras_path, tomllib)
    return (cameras,)


@app.cell
def _(cameras, mo, np, output_dir, pd, triangulate_dlt, undistort_point):
    _obs_path = output_dir / "observations.csv"
    if not _obs_path.exists():
        wide_df = pd.DataFrame()
        mo.stop(True, mo.callout(mo.md(f"Not found: `{_obs_path}`"), kind="danger"))
    elif not cameras:
        wide_df = pd.DataFrame()
        mo.stop(True, mo.callout(mo.md("No cameras loaded."), kind="danger"))
    else:
        _obs = pd.read_csv(_obs_path)
        # used_in_tracking is written as "true"/"false" strings by the C++ exporter
        _inliers = _obs[_obs["used_in_tracking"].isin([True, "true"])].copy()

        records = []
        for (frame, mrk_name), grp in _inliers.groupby(["frame", "marker_name"]):
            cam_obs: dict[int, tuple[float, float]] = {}
            for row in grp.itertuples(index=False):
                cam_id = int(row.camera_id)
                if cam_id not in cameras:
                    continue
                cam = cameras[cam_id]
                u, v = undistort_point(row.pixel_x, row.pixel_y, cam["K"], cam["dist"])
                cam_obs[cam_id] = (u, v)

            if len(cam_obs) < 2:
                continue

            pos, cond = triangulate_dlt(
                list(cam_obs.values()),
                [cameras[cid]["P"] for cid in cam_obs],
            )
            if cond > 200 or not np.all(np.isfinite(pos)):
                continue

            records.append({"frame": frame, "marker_name": mrk_name,
                            "x": pos[0], "y": pos[1], "z": pos[2]})

        if not records:
            wide_df = pd.DataFrame()
        else:
            _tri = pd.DataFrame(records)
            _pv = _tri.pivot_table(
                index="frame", columns="marker_name", values=["x", "y", "z"]
            )
            # Flatten: (x, MRK-knee.L) → MRK-knee.L.x
            _pv.columns = [f"{mrk}.{ax}" for ax, mrk in _pv.columns]
            wide_df = _pv.reset_index()

        mo.md(
            f"Triangulated **{len(records)}** (frame, marker) pairs "
            f"from {len(_inliers)} inlier observations across "
            f"{_inliers['frame'].nunique()} frames."
        )
    return (wide_df,)


@app.cell
def _(fk_rest_pose, mo, np, skeleton_path, yaml):
    if not skeleton_path.exists():
        tmpl_ref = {}
        jp = {}
        mo.stop(True, mo.callout(mo.md(f"Skeleton not found: `{skeleton_path}`"), kind="danger"))
    else:
        with open(skeleton_path) as _f:
            _skel = yaml.safe_load(_f)
        jp = fk_rest_pose(_skel.get("joints", []))

        def _d(a, b):
            return float(np.linalg.norm(jp[a] - jp[b]))

        def _mid(a, b):
            return (jp[a] + jp[b]) / 2.0

        tmpl_ref = {
            "shin":           (_d("shin.L", "foot.L") + _d("shin.R", "foot.R")) / 2,
            "femur":          (_d("thigh.L", "shin.L") + _d("thigh.R", "shin.R")) / 2,
            "upper_arm":      (_d("upper_arm.L", "forearm.L") + _d("upper_arm.R", "forearm.R")) / 2,
            "lower_arm":      (_d("forearm.L", "hand.L") + _d("forearm.R", "hand.R")) / 2,
            "torso_height":   float(np.linalg.norm(_mid("shoulder.L", "shoulder.R") - _mid("thigh.L", "thigh.R"))),
            "shoulder_width": _d("shoulder.L", "shoulder.R"),
            "head":           float(np.linalg.norm(jp["head"] - _mid("shoulder.L", "shoulder.R"))),
        }
    return jp, tmpl_ref


@app.cell
def _(np, pd, wide_df):
    def _dist(m1, m2):
        try:
            dx = wide_df[f"{m1}.x"] - wide_df[f"{m2}.x"]
            dy = wide_df[f"{m1}.y"] - wide_df[f"{m2}.y"]
            dz = wide_df[f"{m1}.z"] - wide_df[f"{m2}.z"]
            return np.sqrt(dx**2 + dy**2 + dz**2)
        except KeyError:
            return pd.Series(float("nan"), index=wide_df.index)

    def _mid_dist(m1a, m1b, m2a, m2b):
        mx1 = (wide_df[f"{m1a}.x"] + wide_df[f"{m1b}.x"]) / 2
        my1 = (wide_df[f"{m1a}.y"] + wide_df[f"{m1b}.y"]) / 2
        mz1 = (wide_df[f"{m1a}.z"] + wide_df[f"{m1b}.z"]) / 2
        mx2 = (wide_df[f"{m2a}.x"] + wide_df[f"{m2b}.x"]) / 2
        my2 = (wide_df[f"{m2a}.y"] + wide_df[f"{m2b}.y"]) / 2
        mz2 = (wide_df[f"{m2a}.z"] + wide_df[f"{m2b}.z"]) / 2
        return np.sqrt((mx1-mx2)**2 + (my1-my2)**2 + (mz1-mz2)**2)

    if wide_df.empty:
        meas_df = pd.DataFrame()
    else:
        meas_df = pd.DataFrame({"frame": wide_df["frame"]})
        meas_df["shin"] = (
            _dist("MRK-knee.L", "MRK-Ankle.L") + _dist("MRK-knee.R", "MRK-Ankle.R")
        ) / 2
        meas_df["femur"] = (
            _dist("MRK-hip.L", "MRK-knee.L") + _dist("MRK-hip.R", "MRK-knee.R")
        ) / 2
        meas_df["upper_arm"] = (
            _dist("MRK-shoulder.L", "MRK-elbow.L") + _dist("MRK-shoulder.R", "MRK-elbow.R")
        ) / 2
        meas_df["lower_arm"] = (
            _dist("MRK-elbow.L", "MRK-wrist.L") + _dist("MRK-elbow.R", "MRK-wrist.R")
        ) / 2
        meas_df["torso_height"] = _mid_dist(
            "MRK-shoulder.L", "MRK-shoulder.R", "MRK-hip.L", "MRK-hip.R"
        )
        meas_df["shoulder_width"] = _dist("MRK-shoulder.L", "MRK-shoulder.R")
        meas_df["head"] = _mid_dist(
            "MRK-ear.L", "MRK-ear.R", "MRK-shoulder.L", "MRK-shoulder.R"
        )
    return (meas_df,)


@app.cell
def _(meas_df, mo):
    if meas_df.empty:
        frame_range = mo.ui.range_slider(start=0, stop=100, value=[0, 100], label="Frame range")
    else:
        _frames = meas_df["frame"].values
        _lo = int(_frames[len(_frames) // 4])
        _hi = int(_frames[3 * len(_frames) // 4])
        frame_range = mo.ui.range_slider(
            start=int(_frames[0]),
            stop=int(_frames[-1]),
            value=[_lo, _hi],
            label="Frame range for median",
            step=1,
        )
    frame_range
    return (frame_range,)


@app.cell
def _(frame_range, meas_df):
    _lo, _hi = frame_range.value
    MEAS_KEYS = ["shin", "femur", "upper_arm", "lower_arm", "torso_height", "shoulder_width", "head"]
    if meas_df.empty:
        range_medians = {k: 0.0 for k in MEAS_KEYS}
    else:
        _sel = meas_df[(meas_df["frame"] >= _lo) & (meas_df["frame"] <= _hi)]
        range_medians = {k: float(_sel[k].median()) for k in MEAS_KEYS}
    return MEAS_KEYS, range_medians


@app.cell
def _(MEAS_KEYS, frame_range, go, make_subplots, meas_df, tmpl_ref):
    MEAS_LABELS = {
        "shin":           "Shin (knee → ankle)",
        "femur":          "Femur (hip → knee)",
        "upper_arm":      "Upper arm (shoulder → elbow)",
        "lower_arm":      "Lower arm (elbow → wrist)",
        "torso_height":   "Torso height (hip → shoulder)",
        "shoulder_width": "Shoulder width (L → R)",
        "head":           "Head proxy (shoulder → ear)",
    }

    _fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[MEAS_LABELS[k] for k in MEAS_KEYS],
        shared_xaxes=True,
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
    )

    _lo, _hi = frame_range.value

    for _i, _k in enumerate(MEAS_KEYS):
        _row = _i // 2 + 1
        _col = _i % 2 + 1

        if not meas_df.empty and _k in meas_df.columns:
            _vals = meas_df[_k] * 100  # → cm

            _fig.add_trace(go.Scatter(
                x=meas_df["frame"], y=_vals,
                mode="lines", line=dict(width=1, color="lightsteelblue"),
                showlegend=False,
            ), row=_row, col=_col)

            _smooth = _vals.rolling(15, center=True, min_periods=1).median()
            _fig.add_trace(go.Scatter(
                x=meas_df["frame"], y=_smooth,
                mode="lines", line=dict(width=2, color="steelblue"),
                showlegend=False,
            ), row=_row, col=_col)

        if _k in tmpl_ref:
            _v = tmpl_ref[_k] * 100
            _fig.add_hline(
                y=_v, line_dash="dot", line_color="gray",
                annotation_text=f"tmpl {_v:.1f} cm",
                annotation_font_size=9,
                row=_row, col=_col,
            )

        _fig.add_vrect(
            x0=_lo, x1=_hi,
            fillcolor="rgba(255,200,0,0.15)", layer="below", line_width=0,
            row=_row, col=_col,
        )

    _fig.update_yaxes(title_text="cm")
    _fig.update_layout(
        height=860,
        title="Key body measurements over time  (dotted = template rest-pose reference, yellow band = selected range)",
        margin=dict(t=60),
    )
    _fig
    return (MEAS_LABELS,)


@app.cell
def _(mo, range_medians):
    def _num(key, label):
        v = range_medians.get(key, 0.0)
        return mo.ui.number(
            value=round(v * 100, 1) if v else 0.0,
            start=1.0, stop=300.0, step=0.1,
            label=label,
        )

    inp_shin           = _num("shin",           "Shin (cm)")
    inp_femur          = _num("femur",          "Femur (cm)")
    inp_upper_arm      = _num("upper_arm",      "Upper arm (cm)")
    inp_lower_arm      = _num("lower_arm",      "Lower arm (cm)")
    inp_torso_height   = _num("torso_height",   "Torso height (cm)")
    inp_shoulder_width = _num("shoulder_width", "Shoulder width (cm)")
    inp_head           = _num("head",           "Head / shoulder→ear (cm)")

    mo.vstack([
        mo.md("### Override values  *(pre-filled with median in selected frame range)*"),
        mo.hstack([
            mo.vstack([inp_shin, inp_femur, inp_upper_arm, inp_lower_arm]),
            mo.vstack([inp_torso_height, inp_shoulder_width, inp_head]),
        ]),
    ])
    return (
        inp_femur,
        inp_head,
        inp_lower_arm,
        inp_shin,
        inp_shoulder_width,
        inp_torso_height,
        inp_upper_arm,
    )


@app.cell
def _(
    MEAS_KEYS,
    MEAS_LABELS,
    inp_femur,
    inp_head,
    inp_lower_arm,
    inp_shin,
    inp_shoulder_width,
    inp_torso_height,
    inp_upper_arm,
    mo,
    pd,
    range_medians,
    tmpl_ref,
):
    _inp_map = {
        "shin":           inp_shin,
        "femur":          inp_femur,
        "upper_arm":      inp_upper_arm,
        "lower_arm":      inp_lower_arm,
        "torso_height":   inp_torso_height,
        "shoulder_width": inp_shoulder_width,
        "head":           inp_head,
    }

    chosen = {k: v.value / 100.0 for k, v in _inp_map.items()}  # back to metres

    _rows = []
    for _k in MEAS_KEYS:
        _t   = tmpl_ref.get(_k, float("nan"))
        _med = range_medians.get(_k, float("nan"))
        _c   = chosen[_k]
        _rows.append({
            "Measurement":            MEAS_LABELS[_k],
            "Template current (cm)":  round(_t * 100, 1),
            "Tracking median (cm)":   round(_med * 100, 1),
            "Chosen (cm)":            round(_c * 100, 1),
            "Scale vs template":      round(_c / _t, 3) if _t else float("nan"),
        })

    mo.vstack([
        mo.md("### Summary"),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ])
    return (chosen,)


@app.cell
def _(chosen, json, mo, output_dir):
    def _export_json(_):
        out = {
            "measurements": {
                k: {"value": round(v, 6), "unit": "m"}
                for k, v in chosen.items()
            },
        }
        p = output_dir / "body-measurements.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as _f:
            json.dump(out, _f, indent=2)
        return f"Written → {p}"

    export_btn = mo.ui.button(label="Export body-measurements.json", on_click=_export_json)
    export_btn
    return


@app.cell
def _(jp, mo, np, output_dir):
    def _cylinder(c1, c2, r=0.01, n=8):
        """Return (verts, tris) for a capped cylinder from c1 to c2, 0-indexed."""
        c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
        ax = c2 - c1
        length = np.linalg.norm(ax)
        if length < 1e-9:
            return [], []
        ax /= length
        perp = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(ax, perp); u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ring1 = [c1 + r * (np.cos(a) * u + np.sin(a) * w) for a in ang]
        ring2 = [c2 + r * (np.cos(a) * u + np.sin(a) * w) for a in ang]
        verts = ring1 + ring2 + [c1.copy(), c2.copy()]
        tris = []
        for i in range(n):
            j = (i + 1) % n
            tris += [(i, j, n + j), (i, n + j, n + i)]
        for i in range(n):
            j = (i + 1) % n
            tris += [(2 * n, j, i), (2 * n + 1, n + i, n + j)]
        return verts, tris

    def _build_scale_obj(jp):
        if not jp:
            return ""

        def _ys(names):
            vals = [jp[k][1] for k in names if k in jp]
            return float(np.mean(vals)) if vals else None

        knee_y  = _ys(["shin.L", "shin.R"])
        hip_y   = _ys(["thigh.L", "thigh.R"])
        shldr_y = _ys(["upper_arm.L", "upper_arm.R"])
        ear_y   = jp["head"][1] if "head" in jp else (shldr_y or 0.0) + 0.25

        xs = [jp[k][0] for k in ["shoulder.L", "shoulder.R", "hand.L", "hand.R"] if k in jp]
        x_min = min(xs) - 0.05 if xs else -0.5
        x_max = max(xs) + 0.05 if xs else 0.5
        z_ctr = float(np.mean([jp[k][2] for k in ["shoulder.L", "shoulder.R"] if k in jp] or [0.0]))

        y_bot = jp["foot.L"][1] if "foot.L" in jp else 0.0
        y_top = ear_y + 0.15

        all_verts = []
        all_groups = []

        def _add(name, c1, c2):
            verts, tris = _cylinder(c1, c2)
            if not verts:
                return
            off = len(all_verts)
            all_verts.extend(verts)
            all_groups.append((name, [(t[0]+off, t[1]+off, t[2]+off) for t in tris]))

        for label, y in [
            ("rail_knee",     knee_y),
            ("rail_hip",      hip_y),
            ("rail_shoulder", shldr_y),
            ("rail_ear",      ear_y),
        ]:
            if y is not None:
                _add(label, [x_min, y, z_ctr], [x_max, y, z_ctr])

        for label, jname in [
            ("pole_shoulder_L", "upper_arm.L"),
            ("pole_shoulder_R", "upper_arm.R"),
            ("pole_elbow_L",    "forearm.L"),
            ("pole_elbow_R",    "forearm.R"),
            ("pole_wrist_L",    "hand.L"),
            ("pole_wrist_R",    "hand.R"),
        ]:
            if jname in jp:
                x, z = jp[jname][0], jp[jname][2]
                _add(label, [x, y_bot, z], [x, y_top, z])

        lines = ["# Scale reference — body-measurements.py", ""]
        for v in all_verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        lines.append("")
        for name, tris in all_groups:
            lines.append(f"g {name}")
            for t in tris:
                lines.append(f"f {t[0]+1} {t[1]+1} {t[2]+1}")
            lines.append("")
        return "\n".join(lines)

    def _export_obj(_):
        obj_str = _build_scale_obj(jp)
        if not obj_str:
            return "No skeleton loaded"
        out = output_dir / "body-scale-ref.obj"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(obj_str)
        return f"Written → {out}"

    export_obj_btn = mo.ui.button(label="Export body-scale-ref.obj", on_click=_export_obj)
    export_obj_btn
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
