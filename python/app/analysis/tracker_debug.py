import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import os
    return Path, mo, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data source
    """)
    return


@app.cell
def _(mo):
    source_selector = mo.ui.radio(
        options={"CSV directory": "csv", "Session DB": "db"},
        value="CSV directory",
        label="Data source",
    )
    source_selector
    return (source_selector,)


@app.cell
def _(mo, os, source_selector):
    _is_db = source_selector.value == "db"

    # DB inputs
    db_path_input = mo.ui.text(
        value=os.getenv("POSETRAK_SESSION_DB", ""),
        label="Session DB path",
        full_width=True,
    )

    person_id_input = mo.ui.number(
        value=0,
        start=0,
        stop=99,
        step=1,
        label="Person ID",
    )

    smoothed_checkbox = mo.ui.checkbox(value=False, label="Load smoothed results")

    # CSV input
    csv_dir_input = mo.ui.text(
        value="/mnt/d/mocap/2026-03-10-posetrak-test/Harri_shomenuchi_iriminage_korkea/posetrak/2026-03-19-1/harri",
        label="Tracking result directory",
        full_width=True,
    )

    mo.vstack([
        db_path_input,
        mo.hstack([person_id_input, smoothed_checkbox]),
    ]) if _is_db else csv_dir_input
    return csv_dir_input, db_path_input, person_id_input, smoothed_checkbox


@app.cell
def _(db_path_input, mo, source_selector):
    import sqlite3 as _sqlite3_rs
    import json as _json_rs
    from pathlib import Path as _Path_rs

    _is_db = source_selector.value == "db"
    _db_path_rs = db_path_input.value.strip()
    _options: dict[str, str] = {}

    if _is_db and _db_path_rs and _Path_rs(_db_path_rs).exists():
        try:
            _conn_rs = _sqlite3_rs.connect(_db_path_rs, check_same_thread=False)
            _conn_rs.row_factory = _sqlite3_rs.Row
            _run_rows = _conn_rs.execute(
                """
                SELECT tr.id, tr.ran_at, tr.active_camera_ids,
                       COUNT(res.tracker_step) AS n_frames
                FROM tracking_runs tr
                LEFT JOIN tracking_results res
                    ON res.run_id = tr.id
                   AND res.person_id = 0
                   AND res.is_smoothed = 0
                GROUP BY tr.id
                ORDER BY tr.ran_at DESC
                """
            ).fetchall()
            _conn_rs.close()
            for _r in _run_rows:
                _cams = _json_rs.loads(_r["active_camera_ids"] or "[]")
                _lbl = (
                    f"{_r['ran_at']}  "
                    f"[{_r['n_frames']} frames, {len(_cams)} cams]  "
                    f"{_r['id'][:8]}…"
                )
                _options[_lbl] = _r["id"]
        except Exception:
            pass

    run_selector = mo.ui.dropdown(
        options=_options,
        label="Tracking run",
        full_width=True,
    )
    run_selector if _is_db else mo.md("")
    return (run_selector,)


@app.cell
def _(
    Path,
    csv_dir_input,
    db_path_input,
    mo,
    pd,
    person_id_input,
    run_selector,
    smoothed_checkbox,
    source_selector,
):
    import sys as _sys
    _project_root = str(Path(__file__).parent.parent)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    _is_db = source_selector.value == "db"

    if _is_db:
        _db_path = db_path_input.value.strip()
        _run_id = run_selector.value or ""
        if not _db_path or not _run_id:
            marker_tracking_df = pd.DataFrame()
            tracking_stats = pd.DataFrame()
            joint_angles = pd.DataFrame()
            root_pose = pd.DataFrame()
            cov_diag_df = pd.DataFrame()
            observations_df = pd.DataFrame()
            projected_markers_df = pd.DataFrame()
            joint_type_map = {}
            _db_mode = True
            mo.stop(True, mo.callout(mo.md("Enter session DB path and select a run above."), kind="warn"))
        else:
            try:
                from posetrak.db.load_session import load_tracking_run_with_markers
                _data = load_tracking_run_with_markers(
                    _db_path,
                    _run_id,
                    person_id=int(person_id_input.value),
                    smoothed=bool(smoothed_checkbox.value),
                )
                marker_tracking_df = _data["marker_positions_df"].rename(columns={
                    "x_3d": "x_3d", "y_3d": "y_3d", "z_3d": "z_3d",
                })
                tracking_stats = _data["tracking_stats_df"].rename(columns={
                    "num_inliers": "num_inliers",
                })
                # Add placeholder columns expected by existing cells
                if not tracking_stats.empty:
                    tracking_stats["num_observations"] = tracking_stats["num_inliers"]
                    tracking_stats["mean_reprojection_error"] = float("nan")
                    tracking_stats["max_reprojection_error"] = float("nan")
                    tracking_stats = tracking_stats.rename(columns={"timestamp": "timestamp"})
                joint_angles = _data["joint_angles_df"]
                root_pose = _data["root_pose_df"].rename(columns={"timestamp": "timestamp"})
                cov_diag_df = _data.get("cov_diag_df", pd.DataFrame())
                from posetrak.db.load_session import load_obs_results as _load_obs
                observations_df, projected_markers_df = _load_obs(
                    _db_path, _run_id, person_id=int(person_id_input.value)
                )
                # Derive per-frame inlier/observation counts from obs_results blob
                # (more reliable than n_inlier_observations in tracking_results).
                if not observations_df.empty and not tracking_stats.empty:
                    _obs_counts = (
                        observations_df.groupby("frame")
                        .agg(
                            num_observations=("is_outlier", "count"),
                            num_inliers=("is_outlier", lambda x: int((~x.astype(bool)).sum())),
                        )
                        .reset_index()
                    )
                    tracking_stats = tracking_stats.drop(
                        columns=["num_inliers"], errors="ignore"
                    ).merge(_obs_counts, on="frame", how="left").fillna(0)
                    tracking_stats["num_observations"] = tracking_stats["num_observations"].astype(int)
                    tracking_stats["num_inliers"] = tracking_stats["num_inliers"].astype(int)
                elif not tracking_stats.empty:
                    tracking_stats["num_observations"] = tracking_stats["num_inliers"]
                # Build joint info map from the raw skeleton YAML.
                # Stores {joint_name: {"type", "parent", "limits"}} for the angle plot.
                import yaml as _yaml
                _skel_data = _yaml.safe_load(_data["skeleton_yaml"])
                joint_type_map = {
                    jd["name"]: {
                        "type": jd.get("type", "fixed"),
                        "parent": jd.get("parent"),
                        "limits": jd.get("limits", {}),
                    }
                    for jd in _skel_data.get("joints", [])
                    if jd.get("type", "fixed") not in ("root", "fixed")
                }
                _db_mode = True
            except Exception as _e:
                marker_tracking_df = pd.DataFrame()
                tracking_stats = pd.DataFrame()
                joint_angles = pd.DataFrame()
                root_pose = pd.DataFrame()
                cov_diag_df = pd.DataFrame()
                observations_df = pd.DataFrame()
                projected_markers_df = pd.DataFrame()
                joint_type_map = {}
                _db_mode = True
                mo.stop(True, mo.callout(mo.md(f"Error loading from DB: `{_e}`"), kind="danger"))
    else:
        _result_dir = Path(csv_dir_input.value)
        _db_mode = False
        cov_diag_df = pd.DataFrame()
        joint_type_map = {}  # not available from CSV; fall back to zero-filter in plot cell
        if not _result_dir.exists():
            marker_tracking_df = pd.DataFrame()
            tracking_stats = pd.DataFrame()
            joint_angles = pd.DataFrame()
            root_pose = pd.DataFrame()
            observations_df = pd.DataFrame()
            projected_markers_df = pd.DataFrame()
            mo.stop(True, mo.callout(mo.md(f"Directory not found: `{_result_dir}`"), kind="danger"))
        else:
            import numpy as _np
            marker_tracking_df = pd.read_csv(_result_dir / "tracking_results.csv")
            tracking_stats = pd.read_csv(_result_dir / "tracking_stats.csv")
            joint_angles = pd.read_csv(_result_dir / "joint_angles.csv")
            root_pose = pd.read_csv(_result_dir / "root_pose.csv")
            _obs_path = _result_dir / "observations.csv"
            _proj_path = _result_dir / "marker_projections.csv"
            if _obs_path.exists():
                observations_df = pd.read_csv(_obs_path)
            else:
                observations_df = pd.DataFrame()
            if _proj_path.exists():
                projected_markers_df = pd.read_csv(_proj_path)
                projected_markers_df["error_dist"] = _np.sqrt(
                    projected_markers_df["error_x"]**2
                    + projected_markers_df["error_y"]**2
                )
            else:
                projected_markers_df = pd.DataFrame()

    _db_mode
    return (
        cov_diag_df,
        joint_angles,
        joint_type_map,
        marker_tracking_df,
        observations_df,
        projected_markers_df,
        root_pose,
        tracking_stats,
    )


@app.cell
def _(marker_tracking_df, mo):
    if marker_tracking_df.empty:
        frame_selector = mo.ui.slider(0, 1, value=0, label="Select Frame", show_value=True)
    else:
        frame_selector = mo.ui.slider(
            marker_tracking_df["frame"].min(),
            marker_tracking_df["frame"].max(),
            value=marker_tracking_df["frame"].min(),
            label="Select Frame",
            show_value=True,
        )
    frame_selector
    return (frame_selector,)


@app.cell(hide_code=True)
def _(frame_selector, marker_tracking_df):
    import plotly.express as px

    _df_selected_frame = marker_tracking_df[marker_tracking_df.frame == frame_selector.value]

    fig = px.scatter_3d(
        _df_selected_frame,
        x="x_3d",
        y="y_3d",
        z="z_3d",
        hover_name="marker_name",
        title=f"3D Marker Positions for Frame {frame_selector.value}",
        color_discrete_sequence=["green"],
    )

    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
        )
    )
    fig
    return (px,)


@app.cell(hide_code=True)
def _(mo, observations_df, projected_markers_df):
    if observations_df.empty and projected_markers_df.empty:
        mo.callout(mo.md("No observation/projection data available."), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Camera projections
    """)
    return


@app.cell(hide_code=True)
def _(mo, observations_df):
    if observations_df.empty:
        proj_frame_selector = mo.ui.slider(0, 1, value=0, label="Select Frame", show_value=True)
        camera_selector = mo.ui.dropdown(options=[0], value=0, label="Select Camera")
    else:
        proj_frame_selector = mo.ui.slider(
            observations_df["frame"].min(),
            observations_df["frame"].max(),
            value=observations_df["frame"].min(),
            label="Select Frame",
            show_value=True,
        )
        camera_selector = mo.ui.dropdown(
            options=observations_df["camera_id"].unique().tolist(),
            value=observations_df["camera_id"].unique().tolist()[0],
            label="Select Camera",
        )
    mo.hstack([proj_frame_selector, camera_selector])
    return camera_selector, proj_frame_selector


@app.cell(hide_code=True)
def _(
    camera_selector,
    mo,
    observations_df,
    pd,
    proj_frame_selector,
    projected_markers_df,
    px,
):
    mo.stop(observations_df.empty or projected_markers_df.empty)
    _filtered_observations_2d = observations_df[
        (observations_df["frame"] == proj_frame_selector.value)
        & (observations_df["camera_id"] == camera_selector.value)
    ].copy()
    _filtered_projections_2d = projected_markers_df[
        (projected_markers_df["frame"] == proj_frame_selector.value)
        & (projected_markers_df["camera_id"] == camera_selector.value)
    ].copy()
    _filtered_observations_2d["type"] = "Observation"
    _filtered_projections_2d["type"] = "Projected Marker"
    _filtered_observations_2d = _filtered_observations_2d.rename(columns={"pixel_x": "x_2d", "pixel_y": "y_2d"})
    _filtered_projections_2d = _filtered_projections_2d.rename(columns={"proj_x": "x_2d", "proj_y": "y_2d"})
    _combined_2d_df = pd.concat([_filtered_observations_2d, _filtered_projections_2d], ignore_index=True)
    _marker_size_mapping = {"Observation": 5, "Projected Marker": 3}
    _combined_2d_df["_marker_size"] = _combined_2d_df["type"].map(_marker_size_mapping)
    _2d_marker_plot = px.scatter(
        _combined_2d_df,
        x="x_2d",
        y="y_2d",
        color="type",
        color_discrete_map={"Observation": "green", "Projected Marker": "red"},
        size="_marker_size",
        hover_name="marker_name",
        title=f"2D Marker Projections for Frame {proj_frame_selector.value}, Camera {camera_selector.value}",
        labels={"x_2d": "X (pixels)", "y_2d": "Y (pixels)", "type": "Marker Type"},
    )
    _2d_marker_plot.update_layout(hovermode="closest")
    _2d_marker_plot.update_yaxes(autorange="reversed")
    _2d_marker_plot
    return


@app.cell
def _(observations_df):
    observations_df
    return


@app.cell
def _(projected_markers_df):
    projected_markers_df
    return


@app.cell
def _(mo, projected_markers_df, px):
    mo.stop(projected_markers_df.empty)
    _inlier_projections_df = projected_markers_df[projected_markers_df["is_outlier"] == False]
    _max_error_markers_per_frame = (
        _inlier_projections_df.loc[_inlier_projections_df.groupby("frame")["error_dist"].idxmax()]
    )
    _max_error_marker_counts = _max_error_markers_per_frame["marker_name"].value_counts().reset_index()
    _max_error_marker_counts.columns = ["marker_name", "count"]
    _histogram_max_error_marker = px.bar(
        _max_error_marker_counts,
        x="marker_name",
        y="count",
        title="Marker Names Most Frequently Having the Largest Reprojection Error",
        labels={"marker_name": "Marker Name", "count": "Number of Frames with Max Error"},
        hover_data=["marker_name", "count"],
    )
    _histogram_max_error_marker.update_xaxes(categoryorder="total descending")
    _histogram_max_error_marker
    return


@app.cell
def _(mo, projected_markers_df, px):
    mo.stop(projected_markers_df.empty)
    _inlier_error_dist = projected_markers_df[projected_markers_df["is_outlier"] == False]
    _marker_error_stats = (
        _inlier_error_dist.groupby("marker_name")["error_dist"]
        .agg(["mean", "median"])
        .reset_index()
    )
    _marker_error_stats_long = _marker_error_stats.melt(
        id_vars=["marker_name"],
        value_vars=["mean", "median"],
        var_name="metric",
        value_name="error_value",
    )
    _error_stats_plot = px.bar(
        _marker_error_stats_long,
        x="marker_name",
        y="error_value",
        color="metric",
        barmode="group",
        title="Mean and Median Reprojection Error per Marker (Inliers Only)",
        labels={
            "marker_name": "Marker Name",
            "error_value": "Error (pixels)",
            "metric": "Statistic",
        },
        hover_data=["marker_name", "metric", "error_value"],
    )
    _error_stats_plot.update_xaxes(categoryorder="total descending")
    _error_stats_plot.update_layout(hovermode="x unified")
    _error_stats_plot
    return


@app.cell
def _(mo, tracking_stats):
    mo.stop(not tracking_stats.empty)
    mo.callout(mo.md("No tracking stats available."), kind="warn")
    return


@app.cell
def _(mo, px, tracking_stats):
    mo.stop(tracking_stats.empty)
    _tracking_stats_long = tracking_stats.melt(
        id_vars=["frame"],
        value_vars=["num_observations", "num_inliers"],
        var_name="metric",
        value_name="count",
    )
    _tracking_stats_plot = px.line(
        _tracking_stats_long,
        x="frame",
        y="count",
        color="metric",
        title="Number of Observations and Inliers Over Frames",
        labels={
            "frame": "Frame",
            "count": "Count",
            "metric": "Tracking Metric",
        },
        hover_data=["frame", "metric", "count"],
    )
    _tracking_stats_plot.update_layout(hovermode="x unified")
    _tracking_stats_plot
    return


@app.cell
def _(mo, px, tracking_stats):
    mo.stop(tracking_stats.empty or tracking_stats["mean_reprojection_error"].isna().all())
    _tracking_stats_long = tracking_stats.melt(
        id_vars=["frame"],
        value_vars=["mean_reprojection_error", "max_reprojection_error"],
        var_name="metric",
        value_name="err",
    )
    _tracking_stats_plot = px.line(
        _tracking_stats_long,
        x="frame",
        y="err",
        color="metric",
        title="Reprojection Error Over Frames",
        labels={
            "frame": "Frame",
            "err": "Error (pixels)",
            "metric": "Tracking Metric",
        },
        hover_data=["frame", "metric", "err"],
    )
    _tracking_stats_plot.update_layout(hovermode="x unified")
    _tracking_stats_plot
    return


@app.cell
def _(joint_angles, mo):
    if joint_angles.empty:
        joint_name_selector = mo.ui.dropdown(options=["(none)"], value="(none)", label="Select Joint Name")
    else:
        joint_name_selector = mo.ui.dropdown(
            options=sorted(joint_angles["joint_name"].unique().tolist()),
            value=sorted(joint_angles["joint_name"].unique().tolist())[0],
            label="Select Joint Name",
        )
    joint_name_selector
    return (joint_name_selector,)


@app.cell(hide_code=True)
def _(joint_angles, joint_name_selector, joint_type_map, mo, px):
    mo.stop(joint_angles.empty)
    _filtered_joint_angles = joint_angles[
        joint_angles["joint_name"] == joint_name_selector.value
    ]
    _jinfo = joint_type_map.get(joint_name_selector.value)
    _jtype = _jinfo["type"] if _jinfo else None

    _xyz_colors = {"angle_x": "red", "angle_y": "green", "angle_z": "blue"}

    # --- Joint info header ---
    if _jinfo:
        _lim = _jinfo.get("limits") or {}
        if _lim:
            _lim_parts = [f"{ax}: [{v[0]:.3f}, {v[1]:.3f}]" for ax, v in _lim.items()]
            _lim_str = ",  ".join(_lim_parts)
        else:
            _lim_str = "none"
        _info_md = mo.md(
            f"**Name:** {joint_name_selector.value} &nbsp;|&nbsp; "
            f"**Type:** {_jtype} &nbsp;|&nbsp; "
            f"**Parent:** {_jinfo.get('parent') or '—'} &nbsp;|&nbsp; "
            f"**Limits:** {_lim_str}"
        )
    else:
        _lim = {}
        _info_md = mo.md("")

    # --- Active angle components ---
    if _jtype in ("revolute", "prismatic"):
        _active_components = ["angle_x"]
    elif _jtype in ("ball", "spherical"):
        _active_components = ["angle_x", "angle_y", "angle_z"]
    else:
        # CSV mode or unknown type: fall back to showing only non-zero components.
        _active_components = [
            c for c in ["angle_x", "angle_y", "angle_z"]
            if _filtered_joint_angles[c].abs().max() > 0
        ]
    _joint_angles_long = _filtered_joint_angles.melt(
        id_vars=["frame", "joint_name"],
        value_vars=_active_components,
        var_name="angle_component",
        value_name="angle_value",
    )
    _joint_angle_plot = px.line(
        _joint_angles_long,
        x="frame",
        y="angle_value",
        color="angle_component",
        color_discrete_map=_xyz_colors,
        title=f"Joint Angles for {joint_name_selector.value}",
        labels={
            "frame": "Frame",
            "angle_value": "Angle (radians)",
            "angle_component": "Angle Component",
        },
        hover_data=["frame", "angle_component", "angle_value"],
    )
    # Draw limit lines (dotted, same color as the corresponding axis)
    _axis_to_component = {"x": "angle_x", "y": "angle_y", "z": "angle_z"}
    for _ax, _bounds in _lim.items():
        _comp = _axis_to_component.get(_ax)
        _color = _xyz_colors.get(_comp, "gray")
        if _comp in _active_components:
            for _bound in _bounds:
                _joint_angle_plot.add_hline(
                    y=_bound,
                    line_dash="dot",
                    line_color=_color,
                    line_width=1,
                    opacity=0.6,
                )
    _joint_angle_plot.update_layout(hovermode="x unified")
    mo.vstack([_info_md, _joint_angle_plot])
    return


@app.cell(hide_code=True)
def _(cov_diag_df, joint_name_selector, joint_type_map, mo, px):
    mo.stop(cov_diag_df.empty)
    _filtered_cov = cov_diag_df[cov_diag_df["joint_name"] == joint_name_selector.value]
    mo.stop(_filtered_cov.empty)

    _jinfo_cov = joint_type_map.get(joint_name_selector.value)
    _jtype_cov = _jinfo_cov["type"] if _jinfo_cov else None
    if _jtype_cov in ("revolute", "prismatic"):
        _std_components = ["std_x"]
    elif _jtype_cov in ("ball", "spherical"):
        _std_components = ["std_x", "std_y", "std_z"]
    else:
        _std_components = [c for c in ["std_x", "std_y", "std_z"]
                           if _filtered_cov[c].max() > 0]

    _cov_long = _filtered_cov.melt(
        id_vars=["frame", "joint_name"],
        value_vars=_std_components,
        var_name="component",
        value_name="std_value",
    )
    _std_colors = {"std_x": "red", "std_y": "green", "std_z": "blue"}
    _cov_plot = px.line(
        _cov_long,
        x="frame",
        y="std_value",
        color="component",
        color_discrete_map=_std_colors,
        title=f"UKF angle std dev (σ) for {joint_name_selector.value}",
        labels={
            "frame": "Frame",
            "std_value": "Std dev (radians)",
            "component": "Component",
        },
        hover_data=["frame", "component", "std_value"],
    )
    _cov_plot.update_layout(hovermode="x unified")
    _cov_plot


@app.cell
def _(mo, px, root_pose):
    mo.stop(root_pose.empty)
    _root_pose_long = root_pose.melt(
        id_vars=["frame"],
        value_vars=["pos_x", "pos_y", "pos_z"],
        var_name="position_component",
        value_name="position_value",
    )
    _root_pose_plot = px.line(
        _root_pose_long,
        x="frame",
        y="position_value",
        color="position_component",
        color_discrete_map={"pos_x": "red", "pos_y": "green", "pos_z": "blue"},
        title="Root Pose Position Over Frames",
        labels={
            "frame": "Frame",
            "position_value": "Position (m)",
            "position_component": "Position Component",
        },
        hover_data=["frame", "position_component", "position_value"],
    )
    _root_pose_plot.update_layout(hovermode="x unified")
    _root_pose_plot
    return


@app.cell
def _(mo, px, root_pose):
    mo.stop(root_pose.empty)
    _root_pose_long = root_pose.melt(
        id_vars=["frame"],
        value_vars=["quat_x", "quat_y", "quat_z", "quat_w"],
        var_name="quaternion_component",
        value_name="quaternion_value",
    )
    _root_pose_plot = px.line(
        _root_pose_long,
        x="frame",
        y="quaternion_value",
        color="quaternion_component",
        color_discrete_map={
            "quat_x": "red",
            "quat_y": "green",
            "quat_z": "blue",
            "quat_w": "purple",
        },
        title="Root Pose Quaternion Over Frames",
        labels={
            "frame": "Frame",
            "quaternion_value": "Quaternion Component Value",
            "quaternion_component": "Quaternion Component",
        },
        hover_data=["frame", "quaternion_component", "quaternion_value"],
    )
    _root_pose_plot.update_layout(hovermode="x unified")
    _root_pose_plot
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
