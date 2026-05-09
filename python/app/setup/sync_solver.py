"""sync_solver.py — Graph-based sync offset solver.

Takes pairwise anchor observations (sync_anchors / sync_anchor_observations)
and computes a consistent set of SyncPoints ready to write to sync_configs /
sync_points via DBContext.write_sync_config().

Algorithm
---------
1.  Build an undirected graph: nodes = video IDs, edges = shared anchors.
2.  BFS from the reference video, assigning each reachable video a time offset
    such that all anchor observations map to the same global timestamp.
3.  Emit one SyncPoint per anchor observation for each reachable video.

The output SyncPoints feed the existing piecewise-linear interpolation in
SyncTable / the tracker unchanged.  Multiple anchor observations per video
automatically enable drift correction via that interpolation.

Known limitation: cycles in the graph use only the first BFS path; redundant
constraints are not used to improve the fit.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from app.setup.db_context import CaptureVideoInfo, SyncAnchorObservation, SyncPoint


@dataclass
class SolveResult:
    """Output of :func:`solve_sync_graph`."""
    sync_points: list[SyncPoint]
    connected_video_ids: set[str]
    isolated_video_ids: set[str]


def check_connectivity(
    anchors: list[tuple[str, list[SyncAnchorObservation]]],
    video_ids: list[str],
) -> tuple[bool, list[str]]:
    """Return ``(all_connected, isolated_video_ids)``.

    A video is *connected* if it is reachable from the first video in
    *video_ids* through the anchor graph.  Videos that have no anchor
    observations at all are always isolated.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for _, obs_list in anchors:
        vids = [o.shot_video_id for o in obs_list]
        for i, vid_a in enumerate(vids):
            for vid_b in vids[i + 1:]:
                adj[vid_a].add(vid_b)
                adj[vid_b].add(vid_a)

    if not video_ids:
        return True, []

    visited: set[str] = {video_ids[0]}
    queue: deque[str] = deque([video_ids[0]])
    while queue:
        node = queue.popleft()
        for neighbour in adj[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    isolated = [vid for vid in video_ids if vid not in visited]
    return len(isolated) == 0, isolated


def solve_sync_graph(
    anchors: list[tuple[str, list[SyncAnchorObservation]]],
    videos: list[CaptureVideoInfo],
    reference_video_id: str | None = None,
) -> SolveResult:
    """Compute consistent SyncPoints from pairwise anchor observations.

    Parameters
    ----------
    anchors:
        Output of ``DBContext.get_anchor_observations()`` —
        ``[(anchor_id, [obs, …]), …]``.
    videos:
        All ``CaptureVideoInfo`` rows for the capture.
    reference_video_id:
        Video whose offset is defined as 0.  If ``None``, the video with the
        most anchor appearances is chosen automatically.

    Returns
    -------
    SolveResult
        ``sync_points`` can be passed directly to
        ``DBContext.write_sync_config()``.  Isolated videos are omitted from
        ``sync_points`` and listed in ``isolated_video_ids``.
    """
    fps_map = {v.id: (v.actual_fps or 30.0) for v in videos}
    all_video_ids = [v.id for v in videos]

    # Index anchors by video so BFS can find neighbours quickly.
    video_to_anchors: dict[str, list[tuple[str, list[SyncAnchorObservation]]]] = (
        defaultdict(list)
    )
    for anchor_id, obs_list in anchors:
        for obs in obs_list:
            video_to_anchors[obs.shot_video_id].append((anchor_id, obs_list))

    # Choose reference: most-connected video by default.
    if reference_video_id is None:
        if not all_video_ids:
            return SolveResult([], set(), set())
        reference_video_id = max(
            all_video_ids,
            key=lambda vid: len(video_to_anchors.get(vid, [])),
        )

    # BFS: offsets[vid] means  t_global = (frame + subframe)/fps + offset
    offsets: dict[str, float] = {reference_video_id: 0.0}
    queue: deque[str] = deque([reference_video_id])
    used_anchors: set[str] = set()

    while queue:
        cur_vid = queue.popleft()
        cur_fps = fps_map.get(cur_vid, 30.0)
        cur_offset = offsets[cur_vid]

        for anchor_id, obs_list in video_to_anchors.get(cur_vid, []):
            if anchor_id in used_anchors:
                continue
            used_anchors.add(anchor_id)

            cur_obs = next(
                (o for o in obs_list if o.shot_video_id == cur_vid), None
            )
            if cur_obs is None:
                continue

            t_anchor = (cur_obs.video_frame + cur_obs.subframe) / cur_fps + cur_offset

            for obs in obs_list:
                if obs.shot_video_id == cur_vid or obs.shot_video_id in offsets:
                    continue
                other_fps = fps_map.get(obs.shot_video_id, 30.0)
                offsets[obs.shot_video_id] = (
                    t_anchor - (obs.video_frame + obs.subframe) / other_fps
                )
                queue.append(obs.shot_video_id)

    # Emit one SyncPoint per anchor observation for each reachable video.
    # Use shot_video_id as camera_instance_id to avoid PK collisions when
    # multiple videos share the same camera_instance_id ("__unassigned__").
    # Deduplicate by (shot_video_id, video_frame) — a camera can appear in
    # multiple anchors and would otherwise produce duplicate rows.
    sync_points: list[SyncPoint] = []
    seen: set[tuple[str, int]] = set()
    for _anchor_id, obs_list in anchors:
        for obs in obs_list:
            if obs.shot_video_id not in offsets:
                continue
            key = (obs.shot_video_id, obs.video_frame)
            if key in seen:
                continue
            seen.add(key)
            fps = fps_map.get(obs.shot_video_id, 30.0)
            t = (obs.video_frame + obs.subframe) / fps + offsets[obs.shot_video_id]
            sync_points.append(SyncPoint(
                camera_instance_id=obs.shot_video_id,
                shot_video_id=obs.shot_video_id,
                video_frame=obs.video_frame,
                timestamp_s=t,
            ))

    connected = set(offsets.keys())
    isolated = set(all_video_ids) - connected
    return SolveResult(
        sync_points=sync_points,
        connected_video_ids=connected,
        isolated_video_ids=isolated,
    )
