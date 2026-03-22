import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path
    return Path, go, mo, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Scale calibration debug viewer
    """)
    return


@app.cell
def _(mo):
    csv_path_input = mo.ui.text(
        value="/home/harri/projects/posetrak/tracking_tests/harri-scaled-skeleton-ri-debug.csv",
        label="Debug CSV path",
        full_width=True,
    )
    csv_path_input
    return (csv_path_input,)


@app.cell
def _(Path, csv_path_input, mo, pd):
    _p = Path(csv_path_input.value)
    if not _p.exists():
        df = pd.DataFrame()
        mo.stop(True, mo.callout(mo.md(f"File not found: `{_p}`"), kind="danger"))
    else:
        df = pd.read_csv(_p)
        # Coerce numeric columns that may have been read as strings due to empty cells
        for _col in ["tri_dist", "model_dist", "rest_pose_chord", "scale_estimate",
                     "prox_cond", "dist_cond"]:
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce")
        df["accepted"] = df["accepted"].astype(str).str.lower() == "true"
    return (df,)


@app.cell(hide_code=True)
def _(df, mo):
    if df.empty:
        mo.stop(True)
    groups = sorted(df["group"].unique())
    group_selector = mo.ui.dropdown(
        options=groups,
        value=groups[0],
        label="Scale group",
    )
    group_selector
    return (group_selector,)


@app.cell
def _(df, group_selector):
    gdf = df[df["group"] == group_selector.value].copy()
    joints_in_group = sorted(gdf["joint"].unique())
    return gdf, joints_in_group


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Per-frame distances and scale estimates
    """)
    return


@app.cell
def _(gdf, go, group_selector, joints_in_group, px):
    import plotly.graph_objects as _go

    _is_chain = gdf["rest_pose_chord"].notna().any() and (gdf["rest_pose_chord"] > 0).any()
    _rest_chord = float(gdf["rest_pose_chord"].dropna().iloc[0]) if _is_chain else None

    fig_dist = go.Figure()

    _colors = px.colors.qualitative.Plotly
    for _i, _joint in enumerate(joints_in_group):
        _jdf = gdf[gdf["joint"] == _joint]
        _c = _colors[_i % len(_colors)]

        # Triangulated distance (solid line)
        fig_dist.add_trace(go.Scatter(
            x=_jdf["frame"], y=_jdf["tri_dist"],
            mode="lines",
            name=f"{_joint} tri_dist",
            line=dict(color=_c, width=2),
            legendgroup=_joint,
        ))

        # Tracked model chord (dashed, lighter — reference only)
        if _jdf["model_dist"].notna().any():
            fig_dist.add_trace(go.Scatter(
                x=_jdf["frame"], y=_jdf["model_dist"],
                mode="lines",
                name=f"{_joint} model_dist (tracked)",
                line=dict(color=_c, width=1, dash="dash"),
                opacity=0.5,
                legendgroup=_joint,
                legendgrouptitle=dict(text=_joint) if _i == 0 else {},
            ))

    # Rest-pose chord: single horizontal reference line
    if _rest_chord is not None:
        fig_dist.add_hline(
            y=_rest_chord,
            line_dash="dot",
            line_color="black",
            annotation_text=f"rest-pose chord ({_rest_chord:.3f} m)",
            annotation_position="right",
        )

    fig_dist.update_layout(
        title=f"Marker pair distances — group '{group_selector.value}'",
        xaxis_title="Frame",
        yaxis_title="Distance (m)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig_dist
    return


@app.cell
def _(gdf, go, group_selector, joints_in_group, px):
    _colors = px.colors.qualitative.Plotly

    fig_scale = go.Figure()

    for _i, _joint in enumerate(joints_in_group):
        _jdf = gdf[gdf["joint"] == _joint]
        _c = _colors[_i % len(_colors)]
        _acc = _jdf[_jdf["accepted"]]
        _rej = _jdf[~_jdf["accepted"] & _jdf["scale_estimate"].notna()]

        # Accepted estimates
        fig_scale.add_trace(go.Scatter(
            x=_acc["frame"], y=_acc["scale_estimate"],
            mode="markers",
            name=f"{_joint} accepted",
            marker=dict(color=_c, size=4, symbol="circle"),
            legendgroup=_joint,
        ))
        # Rejected estimates (with hover showing reason)
        if not _rej.empty:
            fig_scale.add_trace(go.Scatter(
                x=_rej["frame"], y=_rej["scale_estimate"],
                mode="markers",
                name=f"{_joint} rejected",
                marker=dict(color=_c, size=4, symbol="x", opacity=0.3),
                customdata=_rej[["reject_reason"]].values,
                hovertemplate="%{y:.4f}<br>%{customdata[0]}<extra></extra>",
                legendgroup=_joint,
            ))

    fig_scale.add_hline(y=1.0, line_dash="dot", line_color="gray",
                        annotation_text="scale=1.0", annotation_position="right")

    fig_scale.update_layout(
        title=f"Per-frame scale estimates — group '{group_selector.value}'",
        xaxis_title="Frame",
        yaxis_title="Scale estimate",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig_scale
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Scale estimate distribution
    """)
    return


@app.cell
def _(gdf, go, group_selector, joints_in_group, np, px):
    fig_hist = go.Figure()
    _colors = px.colors.qualitative.Plotly

    for _i, _joint in enumerate(joints_in_group):
        _vals = gdf[(gdf["joint"] == _joint) & gdf["accepted"]]["scale_estimate"].dropna()
        if _vals.empty:
            continue
        _median = float(np.median(_vals))
        _p90 = float(np.percentile(_vals, 90))
        fig_hist.add_trace(go.Histogram(
            x=_vals, nbinsx=50,
            name=_joint,
            marker_color=_colors[_i % len(_colors)],
            opacity=0.6,
        ))
        # Vertical lines for median and P90
        for _v, _label, _dash in [(_median, "median", "dash"), (_p90, "P90", "dot")]:
            fig_hist.add_vline(
                x=_v, line_dash=_dash,
                line_color=_colors[_i % len(_colors)],
                annotation_text=f"{_joint} {_label}={_v:.3f}",
                annotation_position="top right",
            )

    fig_hist.update_layout(
        barmode="overlay",
        title=f"Scale estimate distribution — group '{group_selector.value}' (accepted only)",
        xaxis_title="Scale estimate",
        yaxis_title="Count",
        height=340,
    )
    fig_hist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Triangulation quality
    """)
    return


@app.cell
def _(gdf, go, group_selector, joints_in_group, px):
    _colors = px.colors.qualitative.Plotly
    fig_cond = go.Figure()

    for _i, _joint in enumerate(joints_in_group):
        _jdf = gdf[gdf["joint"] == _joint]
        _c = _colors[_i % len(_colors)]
        for _endpoint, _col in [("prox", "prox_cond"), ("dist", "dist_cond")]:
            _vals = _jdf[_col]
            if _vals.notna().any():
                fig_cond.add_trace(go.Scatter(
                    x=_jdf["frame"], y=_vals.clip(upper=500),
                    mode="lines",
                    name=f"{_joint} {_endpoint} cond",
                    line=dict(color=_c, width=1,
                              dash="solid" if _endpoint == "prox" else "dash"),
                    legendgroup=_joint,
                ))

    fig_cond.add_hline(y=200, line_dash="dot", line_color="red",
                       annotation_text="cond threshold (200)", annotation_position="right")
    fig_cond.update_layout(
        title=f"DLT condition numbers — group '{group_selector.value}' (clipped at 500)",
        xaxis_title="Frame",
        yaxis_title="Condition number",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig_cond
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Rejection breakdown
    """)
    return


@app.cell
def _(gdf, mo):
    _rej = gdf[~gdf["accepted"]].copy()
    _rej["reject_reason"] = _rej["reject_reason"].fillna("").replace("", "accepted")

    _counts = (
        _rej.groupby(["joint", "reject_reason"])
        .size()
        .reset_index(name="count")
        .sort_values(["joint", "count"], ascending=[True, False])
    )
    _total = len(gdf)
    _acc = gdf["accepted"].sum()

    mo.vstack([
        mo.stat(
            label="Accepted samples",
            value=f"{_acc} / {_total}",
            caption=f"{100*_acc/_total:.1f}%",
        ),
        mo.ui.table(_counts, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### All groups summary
    """)
    return


@app.cell
def _(df, mo, np, pd):
    _rows = []
    for _group, _gdf in df.groupby("group"):
        for _joint, _jdf in _gdf.groupby("joint"):
            _acc = _jdf[_jdf["accepted"]]["scale_estimate"].dropna()
            _all = _jdf["scale_estimate"].dropna()
            _rest = _jdf["rest_pose_chord"].dropna()
            _rows.append({
                "group": _group,
                "joint": _joint,
                "n_accepted": len(_acc),
                "n_total": len(_jdf),
                "median": float(np.median(_acc)) if len(_acc) else float("nan"),
                "P90": float(np.percentile(_acc, 90)) if len(_acc) else float("nan"),
                "IQR": float(np.percentile(_acc, 75) - np.percentile(_acc, 25)) if len(_acc) > 1 else float("nan"),
                "rest_pose_chord": float(_rest.iloc[0]) if len(_rest) else float("nan"),
                "top_reject": _jdf[~_jdf["accepted"]]["reject_reason"].value_counts().index[0]
                              if (~_jdf["accepted"]).any() else "",
            })
    _summary = pd.DataFrame(_rows)
    mo.ui.table(_summary.round(4), selection=None)
    return


if __name__ == "__main__":
    app.run()
