"""sync_solver.py — Graph-based sync offset solver.

Takes pairwise anchor observations (sync_anchors / sync_anchor_observations)
and computes a consistent set of SyncPoints ready to write to sync_configs /
sync_points via DBContext.write_sync_config().

Algorithm
---------
1.  Build an undirected graph: nodes = video IDs, edges = shared anchors.
    Pre-compute all anchor pairs between each ordered camera pair so that
    the BFS step can see them collectively rather than one at a time.
2.  BFS from the reference video:
    - If 2+ anchor pairs connect a new camera to an already-solved camera,
      fit  frame = fps * global_time + C  by least squares and derive both
      the effective fps and the time offset.  This corrects nominal fps
      values that are wrong (e.g. a 120 fps camera reported as 30 fps).
    - If only 1 anchor pair is available, use the nominal fps from the
      video file and solve for the offset alone.
3.  Emit one SyncPoint per anchor observation for each reachable video,
    timestamped with the effective (possibly corrected) fps.

Priority
--------
LED sync results (written into sync_configs by the LED dialog) are
authoritative for the cameras they cover.  The graph solver is called
afterwards only for cameras not covered by LED sync; the caller in
page_sync._merge_led_and_graph is responsible for that split.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from app.setup.db_context import CaptureVideoInfo, SyncAnchorObservation, SyncPoint


@dataclass
class SolveResult:
    """Output of :func:`solve_sync_graph`."""
    sync_points: list[SyncPoint]
    connected_video_ids: set[str]
    isolated_video_ids: set[str]
    # effective fps per video (may differ from nominal when derived from 2+
    # anchor pairs via least-squares); reference video always has nominal fps.
    effective_fps: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _lstsq_fps_offset(
    global_times: list[float],
    local_frames: list[float],
) -> tuple[float | None, float | None]:
    """Fit  frame = fps * global_time + C  by ordinary least squares.

    Returns ``(fps, offset)`` where ``offset = -C / fps``, i.e.
    ``global_time = frame / fps + offset``.
    Returns ``(None, None)`` if the system is degenerate.
    """
    n = len(global_times)
    if n < 2:
        return None, None
    sx = sum(global_times)
    sy = sum(local_frames)
    sxx = sum(t * t for t in global_times)
    sxy = sum(t * f for t, f in zip(global_times, local_frames))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-10:          # all observations at the same instant
        return None, None
    m = (n * sxy - sx * sy) / denom  # effective fps
    c = (sy - m * sx) / n
    if m <= 0:                        # fps must be positive
        return None, None
    return m, -c / m                  # (fps, offset)


def _build_pair_obs(
    anchors: list[tuple[str, list[SyncAnchorObservation]]],
) -> dict[tuple[str, str], list[tuple[int, float, int, float]]]:
    """Return a mapping (vid_a, vid_b) → [(frame_a, sub_a, frame_b, sub_b), …].

    Every anchor that involves both vid_a and vid_b contributes one entry.
    Both orderings (a→b and b→a) are stored so BFS can look up neighbours
    in either direction.
    """
    pair_obs: dict[tuple[str, str], list[tuple[int, float, int, float]]] = (
        defaultdict(list)
    )
    for _anchor_id, obs_list in anchors:
        for i in range(len(obs_list)):
            for j in range(len(obs_list)):
                if i == j:
                    continue
                oa, ob = obs_list[i], obs_list[j]
                pair_obs[(oa.shot_video_id, ob.shot_video_id)].append((
                    oa.video_frame, oa.subframe,
                    ob.video_frame, ob.subframe,
                ))
    return pair_obs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_pair_fps_inconsistency(
    anchors: list[tuple[str, list[SyncAnchorObservation]]],
    vid_a_id: str,
    vid_b_id: str,
    fps_a: float,
    fps_b: float,
    threshold: float = 0.10,
) -> tuple[float, float] | None:
    """Check if 2+ anchor pairs between two cameras imply an fps inconsistency.

    Treats each camera as the reference in turn and fits the other's fps via OLS.

    Returns
    -------
    (fps_b_if_a_correct, fps_a_if_b_correct)
        The implied fps for each camera assuming the other is accurate.
        Both will differ from their nominal values by more than *threshold*.
    None
        Fewer than 2 shared anchors, or deviation is within *threshold*.
    """
    pairs: list[tuple[float, float]] = []
    for _anchor_id, obs_list in anchors:
        by_vid = {o.shot_video_id: o for o in obs_list}
        if vid_a_id in by_vid and vid_b_id in by_vid:
            oa, ob = by_vid[vid_a_id], by_vid[vid_b_id]
            pairs.append((oa.video_frame + oa.subframe, ob.video_frame + ob.subframe))

    if len(pairs) < 2:
        return None

    # Treat A as reference → derive implied fps for B.
    fps_b_eff, _ = _lstsq_fps_offset(
        [fa / fps_a for fa, _fb in pairs],
        [fb for _fa, fb in pairs],
    )
    if fps_b_eff is None or fps_b_eff <= 0:
        return None
    if abs(fps_b_eff - fps_b) / fps_b <= threshold:
        return None

    # Treat B as reference → derive implied fps for A.
    fps_a_eff, _ = _lstsq_fps_offset(
        [fb / fps_b for _fa, fb in pairs],
        [fa for fa, _fb in pairs],
    )
    if fps_a_eff is None or fps_a_eff <= 0:
        fps_a_eff = fps_a * fps_b / fps_b_eff  # geometric fallback

    return fps_b_eff, fps_a_eff


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
        ``effective_fps`` maps each connected video to the fps that was used
        (may differ from the nominal value stored in the video file).
    """
    nom_fps: dict[str, float] = {v.id: (v.actual_fps or 30.0) for v in videos}
    all_video_ids = [v.id for v in videos]

    if not all_video_ids:
        return SolveResult([], set(), set())

    # Pre-compute all anchor pairs between each ordered (cur, new) camera pair.
    pair_obs = _build_pair_obs(anchors)

    # Build adjacency graph (undirected) for BFS connectivity.
    adj: dict[str, set[str]] = defaultdict(set)
    for (va, vb) in pair_obs:
        adj[va].add(vb)

    # Choose reference: most-connected video.
    if reference_video_id is None:
        reference_video_id = max(
            all_video_ids,
            key=lambda vid: len(adj.get(vid, set())),
        )

    # BFS — solve for offset and effective fps of each reachable video.
    offsets: dict[str, float] = {reference_video_id: 0.0}
    eff_fps: dict[str, float] = {reference_video_id: nom_fps.get(reference_video_id, 30.0)}
    queue: deque[str] = deque([reference_video_id])

    while queue:
        cur_vid = queue.popleft()
        cur_fps = eff_fps[cur_vid]
        cur_offset = offsets[cur_vid]

        for new_vid in adj.get(cur_vid, set()):
            if new_vid in offsets:
                continue  # already solved via an earlier BFS path

            pairs = pair_obs.get((cur_vid, new_vid), [])
            if not pairs:
                continue

            nom = nom_fps.get(new_vid, 30.0)

            if len(pairs) >= 2:
                # Compute global times for every anchor observation in cur_vid,
                # then fit  frame_new = fps_new * t_global + C  by OLS.
                global_times = [
                    (fa + sa) / cur_fps + cur_offset
                    for fa, sa, _fb, _sb in pairs
                ]
                local_frames = [float(fb + sb) for _fa, _sa, fb, sb in pairs]
                fps_new, off_new = _lstsq_fps_offset(global_times, local_frames)
                if fps_new is None or fps_new <= 0:
                    # Degenerate — fall back to single-anchor with nominal fps.
                    fps_new = nom
                    fa, sa, fb, sb = pairs[0]
                    t = (fa + sa) / cur_fps + cur_offset
                    off_new = t - (fb + sb) / fps_new
            else:
                # Single anchor — use nominal fps, solve for offset only.
                fps_new = nom
                fa, sa, fb, sb = pairs[0]
                t = (fa + sa) / cur_fps + cur_offset
                off_new = t - (fb + sb) / fps_new

            offsets[new_vid] = off_new
            eff_fps[new_vid] = fps_new
            queue.append(new_vid)

    # Emit one SyncPoint per anchor observation for each reachable video.
    # Use effective fps (possibly corrected) for the timestamp.
    # Deduplicate by (shot_video_id, video_frame).
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
            fps = eff_fps.get(obs.shot_video_id, nom_fps.get(obs.shot_video_id, 30.0))
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
        effective_fps=dict(eff_fps),
    )
