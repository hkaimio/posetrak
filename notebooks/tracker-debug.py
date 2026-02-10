import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    return Path, mo, pd


@app.cell
def _(Path):
    #result_dir = Path("/home/harri/projects/posetrak/tracking_tests/alpha_0_1")
    result_dir = Path("/home/harri/projects/posetrak/tracking_tests/cpp-python-comparison/cpp_results")
    return (result_dir,)


@app.cell
def _(result_dir):
    marker_track_path = result_dir / "tracking_results.csv"
    return (marker_track_path,)


@app.cell
def _(marker_track_path, pd):
    marker_tracking_df = pd.read_csv(marker_track_path)
    return (marker_tracking_df,)


@app.cell
def _(pd, result_dir):
    tracking_stats = pd.read_csv(result_dir / "tracking_stats.csv")
    return (tracking_stats,)


@app.cell
def _(px, tracking_stats):
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
def _(px, tracking_stats):
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
def _(marker_tracking_df, mo):
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
def _(pd, result_dir):
    observations_df = pd.read_csv(result_dir / "observations.csv")
    projected_markers_df = pd.read_csv(result_dir / "marker_projections.csv")
    return observations_df, projected_markers_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Camera projections
    """)
    return


@app.cell(hide_code=True)
def _(mo, observations_df):
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
    observations_df,
    pd,
    proj_frame_selector,
    projected_markers_df,
    px,
):
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

    # Rename columns to a common name for concatenation
    _filtered_observations_2d = _filtered_observations_2d.rename(columns={"pixel_x": "x_2d", "pixel_y": "y_2d"})
    _filtered_projections_2d = _filtered_projections_2d.rename(columns={"proj_x": "x_2d", "proj_y": "y_2d"})

    _combined_2d_df = pd.concat([_filtered_observations_2d, _filtered_projections_2d], ignore_index=True)

    _color_map_2d = {"Observation": "green", "Projected Marker": "red"}
    # Define a mapping for marker sizes
    _marker_size_mapping = {"Observation": 5, "Projected Marker": 3}
    # Create a new column in the DataFrame for marker sizes based on the 'type'
    _combined_2d_df["_marker_size"] = _combined_2d_df["type"].map(_marker_size_mapping)

    _2d_marker_plot = px.scatter(
        _combined_2d_df,
        x="x_2d",
        y="y_2d",
        color="type",
        color_discrete_map=_color_map_2d,
        # Use the new numerical column for marker size
        size="_marker_size",
        hover_name="marker_name",
        title=f"2D Marker Projections for Frame {proj_frame_selector.value}, Camera {camera_selector.value}",
        labels={
            "x_2d": "X (pixels)",
            "y_2d": "Y (pixels)",
            "type": "Marker Type",
        },
    )

    _2d_marker_plot.update_layout(
        hovermode="closest"
    )

    # Reverse the y-axis to make Y increase downward
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
def _(pd, result_dir):
    joint_angles = pd.read_csv(result_dir / "joint_angles.csv")
    return (joint_angles,)


@app.cell
def _(joint_angles, mo):
    joint_name_selector = mo.ui.dropdown(
        options=sorted(joint_angles["joint_name"].unique().tolist()),
        value=sorted(joint_angles["joint_name"].unique().tolist())[0],
        label="Select Joint Name",
    )
    joint_name_selector
    return (joint_name_selector,)


@app.cell(hide_code=True)
def _(joint_angles, joint_name_selector, px):
    _filtered_joint_angles = joint_angles[
        joint_angles["joint_name"] == joint_name_selector.value
    ]

    _joint_angles_long = _filtered_joint_angles.melt(
        id_vars=["frame", "joint_name"],
        value_vars=["angle_x", "angle_y", "angle_z"],
        var_name="angle_component",
        value_name="angle_value",
    )

    _joint_angle_plot = px.line(
        _joint_angles_long,
        x="frame",
        y="angle_value",
        color="angle_component",
        title=f"Joint Angles for {joint_name_selector.value}",
        labels={
            "frame": "Frame",
            "angle_value": "Angle (radians)",
            "angle_component": "Angle Component",
        },
        hover_data=["frame", "angle_component", "angle_value"]
    )

    _joint_angle_plot.update_layout(
        hovermode="x unified"
    )
    _joint_angle_plot
    return


@app.cell
def _(pd, result_dir):
    root_pose = pd.read_csv(result_dir / "root_pose.csv")
    return (root_pose,)


@app.cell
def _(px, root_pose):
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
def _(px, root_pose):
    _root_pose_long = root_pose.melt(
        id_vars=["frame"],
        value_vars=["quat_x", "quat_y", "quat_w", "quat_z", "quat_w"],
        var_name="quaternion_component",
        value_name="quaternion_value",
    )

    _root_pose_plot = px.line(
        _root_pose_long,
        x="frame",
        y="quaternion_value",
        color="quaternion_component",
        title="Root Pose Quaternion Over Frames",
        labels={
            "frame": "Frame",
            "quaternion_value": "Position (m)",
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
