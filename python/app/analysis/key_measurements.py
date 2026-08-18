# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

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
    return Path, go, json, make_subplots, math, mo, np, pd, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Key body measurements
    """)
    return


@app.cell
def _(mo):
    tracking_csv_input = mo.ui.text(
        value="/mnt/d/mocap/2026-03-10-posetrak-test/Harri_aihanmi_katatedori_ikkyo/posetrak/2026-03-13-3/harri/tracking_results.csv",
        label="tracking_results.csv",
        full_width=True,
    )
    skeleton_input = mo.ui.text(
        value="/home/harri/projects/posetrak/tracking_tests/harri-scaled-skeleton-ri.yaml",
        label="Skeleton YAML",
        full_width=True,
    )
    output_json_input = mo.ui.text(
        value="/home/harri/projects/posetrak/tracking_tests/harri-measurements.json",
        label="Output measurements JSON",
        full_width=True,
    )
    mo.vstack([tracking_csv_input, skeleton_input, output_json_input])
    return output_json_input, skeleton_input, tracking_csv_input


@app.cell
def _(Path, mo, pd, tracking_csv_input):
    _p = Path(tracking_csv_input.value)
    if not _p.exists():
        wide_df = pd.DataFrame()
        mo.stop(True, mo.callout(mo.md(f"File not found: `{_p}`"), kind="danger"))
    else:
        _raw = pd.read_csv(_p)
        _pv = _raw.pivot_table(
            index="frame", columns="marker_name", values=["x_3d", "y_3d", "z_3d"]
        )
        # Flatten: (x_3d, MRK-knee.L) → MRK-knee.L.x
        _pv.columns = [f"{mrk}.{ax[0]}" for ax, mrk in _pv.columns]
        wide_df = _pv.reset_index()
    return (wide_df,)


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
def _(Path, fk_rest_pose, mo, np, skeleton_input, yaml):
    _p = Path(skeleton_input.value)
    if not _p.exists():
        tmpl_ref = {}
        jp = {}
        mo.stop(True, mo.callout(mo.md(f"Skeleton not found: `{_p}`"), kind="danger"))
    else:
        with open(_p) as _f:
            _skel = yaml.safe_load(_f)
        jp = fk_rest_pose(_skel.get("joints", []))

        def _d(a, b):
            return float(np.linalg.norm(jp[a] - jp[b]))

        def _mid(a, b):
            return (jp[a] + jp[b]) / 2.0

        # Template reference distances (joint-origin to joint-origin approximations).
        # shin.L = knee joint, foot.L = ankle joint; thigh.L = hip joint, etc.
        tmpl_ref = {
            "shin":           (_d("shin.L", "foot.L") + _d("shin.R", "foot.R")) / 2,
            "femur":          (_d("thigh.L", "shin.L") + _d("thigh.R", "shin.R")) / 2,
            "upper_arm":      (_d("upper_arm.L", "forearm.L") + _d("upper_arm.R", "forearm.R")) / 2,
            "lower_arm":      (_d("forearm.L", "hand.L") + _d("forearm.R", "hand.R")) / 2,
            "torso_height":   float(np.linalg.norm(_mid("shoulder.L", "shoulder.R") - _mid("thigh.L", "thigh.R"))),
            "shoulder_width": _d("shoulder.L", "shoulder.R"),
            "head":           float(np.linalg.norm(jp["head"] - _mid("shoulder.L", "shoulder.R"))),
        }
    return (tmpl_ref, jp)


@app.cell
def _(wide_df):
    wide_df
    return


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
        return np.sqrt((mx1 - mx2)**2 + (my1 - my2)**2 + (mz1 - mz2)**2)

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

            # Raw signal (faint)
            _fig.add_trace(go.Scatter(
                x=meas_df["frame"], y=_vals,
                mode="lines", line=dict(width=1, color="lightsteelblue"),
                showlegend=False,
            ), row=_row, col=_col)

            # Rolling median (15-frame window) for readability
            _smooth = _vals.rolling(15, center=True, min_periods=1).median()
            _fig.add_trace(go.Scatter(
                x=meas_df["frame"], y=_smooth,
                mode="lines", line=dict(width=2, color="steelblue"),
                showlegend=False,
            ), row=_row, col=_col)

        # Template reference line
        if _k in tmpl_ref:
            _v = tmpl_ref[_k] * 100
            _fig.add_hline(
                y=_v, line_dash="dot", line_color="gray",
                annotation_text=f"tmpl {_v:.1f} cm",
                annotation_font_size=9,
                row=_row, col=_col,
            )

        # Selected frame range shading
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
def _(Path, chosen, json, mo, output_json_input, skeleton_input):
    def _export(_):
        out = {
            "skeleton": skeleton_input.value,
            "measurements": {
                k: {"value": round(v, 6), "unit": "m"}
                for k, v in chosen.items()
            },
        }
        p = Path(output_json_input.value)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as _f:
            json.dump(out, _f, indent=2)
        return f"Written → {p}"

    export_btn = mo.ui.button(label="Export measurements.json", on_click=_export)
    mo.vstack([export_btn, mo.md("*(click once; result appears in terminal)*")])
    return


@app.cell
def _(Path, jp, mo, np, output_json_input):
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
        # end caps: ring1 winding reversed (faces outward from c1), ring2 normal
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

        # X extent for horizontal rails (widest arm span + margin)
        xs = [jp[k][0] for k in ["shoulder.L", "shoulder.R", "hand.L", "hand.R"] if k in jp]
        x_min = min(xs) - 0.05 if xs else -0.5
        x_max = max(xs) + 0.05 if xs else 0.5
        z_ctr = float(np.mean([jp[k][2] for k in ["shoulder.L", "shoulder.R"] if k in jp] or [0.0]))

        # Y extent for vertical poles
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
            all_groups.append((name, [(t[0] + off, t[1] + off, t[2] + off) for t in tris]))

        # Horizontal rails at key heights
        for label, y in [
            ("rail_knee",     knee_y),
            ("rail_hip",      hip_y),
            ("rail_shoulder", shldr_y),
            ("rail_ear",      ear_y),
        ]:
            if y is not None:
                _add(label, [x_min, y, z_ctr], [x_max, y, z_ctr])

        # Vertical poles at shoulder / elbow / wrist X positions (both sides)
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

        lines = ["# Scale reference — key-measurements.py", ""]
        for v in all_verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        lines.append("")
        for name, tris in all_groups:
            lines.append(f"g {name}")
            for t in tris:
                lines.append(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}")
            lines.append("")
        return "\n".join(lines)

    def _export_obj(_):
        obj_str = _build_scale_obj(jp)
        if not obj_str:
            return "No skeleton loaded"
        stem = Path(output_json_input.value).stem
        out = Path(output_json_input.value).parent / f"{stem}_scale_ref.obj"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(obj_str)
        return f"Written → {out}"

    export_obj_btn = mo.ui.button(label="Export scale_ref.obj", on_click=_export_obj)
    mo.vstack([
        export_obj_btn,
        mo.md(
            "Horizontal rails at knee / hip / shoulder / ear heights; "
            "vertical poles at shoulder / elbow / wrist X positions (rest / T-pose)."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
