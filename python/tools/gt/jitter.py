# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""jitter.py — high-frequency noise of tracked joint positions, its episodes and what they coincide with.

Library and command line. ``measure`` compares tracking runs, ``causes`` relates the jitter episodes of one run to the
events of its keypoint choice and filter, ``synthetic`` checks the metric on a known signal.

The metric. The 3D position of each body joint (joint origins; finger joints left out) is sampled at 60 Hz. The
high-frequency part is the residual of a Savitzky-Golay fit of order 3 over 0.2 s. The fit absorbs part of white noise,
so the residual is divided by ``sqrt(1 - h0)``, with ``h0`` the centre coefficient of the filter, which makes its RMS
equal to the noise of the signal; the corrected 3D residual is what is reported (mm). Jerk is the third derivative of
a Savitzky-Golay fit of order 4 over the same span (m/s^3). A time series is the RMS over a sliding 0.5 s window (30
samples) of the squared residual averaged over the joints of a group, advanced one sample at a time. The fit's span
stays well below the window, so the fit does not follow the window's own variation.

Episodes. An episode of a joint group is a run of consecutive samples whose series stays above a threshold of that
group. The threshold is a high percentile (default 99) of the series of the reference runs (the tracker on edited
keypoints), pooled over the subjects: noise of the tracker alone on good input then seldom crosses it. Fast genuine
motion raises the residual too, which the reference runs contain as well, so an episode that is also an episode of the
reference run, in the same group of the same subject, is motion and not jitter of the run under test.

Causes. For the steps of an episode and the markers of its joint group the rate of each event is compared with the
rate over the whole trial: tracklet switch of the pick or a jump of the observation from the fit of its own track, a missing
observation, a run of missing observations, an outlier rejection, a change in the number of cameras that observe the
marker, and a spike of the filter's NIS. The share of the variance of the jitter energy that the events explain is
the R^2 of a regression of the group's energy on the event densities.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_coeffs, savgol_filter

FS = 60.0
SPAN_S = 0.2
WINDOW_S = 0.5
ORDER = 3
JERK_ORDER = 4
GROUPS = {
    "torso": ("hips", "spine1", "spine2", "neck1", "neck2", "head", "shoulder.L", "shoulder.R"),
    "arms": ("upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R"),
    "hands": ("hand.L", "hand.R"),
    "legs": ("thigh.L", "thigh.R", "shin.L", "shin.R"),
    "feet": ("foot.L", "foot.R", "toe.L", "toe.R"),
}
JUMP_PX = 12.0         # distance of an observation from the fit of its own track that counts as a jump (about three times the noise of a still keypoint)
MISSING_RUN_STEPS = 5  # consecutive missing observations that count as a run
NIS_SPIKE_FACTOR = 2.0  # NIS above this multiple of its median over one second


def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 else n + 1


def span_samples(fs: float = FS) -> int:
    return _odd(SPAN_S * fs)


def noise_gain(fs: float = FS) -> float:
    """Factor that turns the RMS of the fit residual of white noise into the RMS of the noise."""
    n = span_samples(fs)
    h0 = savgol_coeffs(n, ORDER)[n // 2]
    return 1.0 / np.sqrt(1.0 - h0)


def residual(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Noise-corrected residual of ``x[T, 3]`` from its fit; NaN where the fit is not interior or ``x`` has gaps."""
    n = span_samples(fs)
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x).all(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= n:
            seg = x[a:b]
            r = (seg - savgol_filter(seg, n, ORDER, axis=0)) * noise_gain(fs)
            r[: n // 2] = np.nan
            r[len(seg) - n // 2:] = np.nan
            out[a:b] = r
    return out


def jerk(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Third derivative (m/s^3 for metres) of the fit of ``x[T, 3]``; NaN at the edges and over gaps."""
    n = span_samples(fs)
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x).all(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= n:
            d = savgol_filter(x[a:b], n, JERK_ORDER, deriv=3, delta=1.0 / fs, axis=0)
            d[: n // 2] = np.nan
            d[b - a - n // 2:] = np.nan
            out[a:b] = d
    return out


def sliding_rms(energy: np.ndarray, fs: float = FS) -> np.ndarray:
    """RMS over a centred window of ``WINDOW_S`` of a series of squared residuals; NaN where the window is incomplete."""
    w = int(round(WINDOW_S * fs))
    kernel = np.ones(w) / w
    ok = np.isfinite(energy)
    clean = np.where(ok, energy, 0.0)
    num = np.convolve(clean, kernel, "same")
    cnt = np.convolve(ok.astype(float), kernel, "same")
    out = np.where(cnt > 0.999, num, np.nan)
    return np.sqrt(out)


@dataclass
class Episode:
    start: float
    end: float
    peak_time: float
    peak_rms: float
    excess: float
    joints: list[str]


def find_episodes(times: np.ndarray, series: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Index ranges ``[a, b)`` of the runs of consecutive samples above the threshold."""
    above = np.nan_to_num(series, nan=0.0) > threshold
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


# --- joint positions -------------------------------------------------------------------------------------------------

class RunPose:
    """Joint frames of a tracking run from its smoothed states, or from the filtered ones when it has none or when asked."""

    def __init__(self, conn: sqlite3.Connection, run: str, states: str = "auto") -> None:
        from posetrak.markers.bones import BodyPose

        run_id = conn.execute("SELECT id FROM tracking_runs WHERE id LIKE ?", (run + "%",)).fetchone()[0]
        self.run_id = run_id
        smoothed = conn.execute("SELECT COUNT(*) FROM tracking_results WHERE run_id=? AND is_smoothed=1", (run_id,)).fetchone()[0]
        self.smoothed = smoothed > 0 and states != "filtered"
        self.pose = BodyPose(conn, run_id)
        if not self.smoothed:
            rows = conn.execute("SELECT timestamp_s, state FROM tracking_results WHERE run_id=? AND is_smoothed=0 "
                                "AND tracking_lost=0 ORDER BY tracker_step", (run_id,)).fetchall()
            self.pose.times = np.array([r[0] for r in rows])
            self.pose._blobs = [bytes(r[1]) for r in rows]
            self.pose._cache = {}

    def positions(self, t0: float, t1: float, fs: float = FS) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Times ``[T]``, joint names and positions ``[J, T, 3]`` (NaN where the run has no state) of the body joints."""
        names = [j for g in GROUPS.values() for j in g]
        times = np.arange(t0, t1, 1.0 / fs)
        pos = np.full((len(names), len(times), 3), np.nan)
        for i, t in enumerate(times):
            frames = self.pose.joint_frames(float(t))
            if frames is None:
                continue
            for j, n in enumerate(names):
                if n in frames:
                    pos[j, i] = frames[n][:3, 3]
        return times, names, pos


def group_of(name: str) -> str:
    return next(g for g, joints in GROUPS.items() if name in joints)


def figures(pos: np.ndarray, names: list[str]) -> dict:
    """RMS residual (mm) and RMS jerk (m/s^3) per joint group over the trial."""
    res = np.stack([residual(pos[j]) for j in range(len(names))])      # [J, T, 3]
    jk = np.stack([jerk(pos[j]) for j in range(len(names))])
    out = {}
    for g, joints in GROUPS.items():
        idx = [names.index(j) for j in joints]
        out[g] = {"rms_mm": float(1000 * np.sqrt(np.nanmean(np.sum(res[idx] ** 2, axis=2)))),
                  "jerk": float(np.sqrt(np.nanmean(np.sum(jk[idx] ** 2, axis=2))))}
    return out


def group_speed(pos: np.ndarray, names: list[str], fs: float = FS) -> dict[str, np.ndarray]:
    """Per sample, the RMS speed (m/s) of the joints of each group, from a fit of order 3 over 0.2 s."""
    n = span_samples(fs)
    sp = np.full(pos.shape[:2], np.nan)
    for j in range(len(names)):
        ok = np.isfinite(pos[j]).all(axis=1)
        if ok.all():
            sp[j] = np.linalg.norm(savgol_filter(pos[j], n, ORDER, deriv=1, delta=1.0 / fs, axis=0), axis=1)
    return {g: np.sqrt(np.nanmean(sp[[names.index(j) for j in joints]] ** 2, axis=0)) for g, joints in GROUPS.items()}


def group_energy(pos: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Per sample, the squared corrected residual (m^2) averaged over the joints of each group."""
    res = np.stack([residual(pos[j]) for j in range(len(names))])
    sq = np.sum(res ** 2, axis=2)
    return {g: np.nanmean(sq[[names.index(j) for j in joints]], axis=0) for g, joints in GROUPS.items()}


def thresholds(reference_energies: list[dict[str, np.ndarray]], percentile: float = 99.0) -> dict[str, float]:
    """Per group, the percentile of the sliding RMS (m) pooled over the reference runs."""
    out = {}
    for g in GROUPS:
        pooled = np.concatenate([sliding_rms(e[g]) for e in reference_energies])
        out[g] = float(np.nanpercentile(pooled, percentile))
    return out


def episodes_of(times, names, pos, energy, thr) -> dict[str, list[Episode]]:
    res = np.stack([residual(pos[j]) for j in range(len(names))])
    sq = np.sum(res ** 2, axis=2)
    out = {}
    for g, joints in GROUPS.items():
        series = sliding_rms(energy[g])
        idx = [names.index(j) for j in joints]
        eps = []
        for a, b in find_episodes(times, series, thr[g]):
            seg = np.nan_to_num(series[a:b])
            peak = a + int(np.argmax(seg))
            # the glitch itself: the sample with the largest residual energy inside the episode
            lo, hi = max(0, a - 15), min(len(times), b + 15)
            e_in = np.nan_to_num(np.nansum(sq[idx][:, lo:hi], axis=0))
            by_joint = np.nansum(sq[idx][:, lo:hi], axis=1)
            order = np.argsort(by_joint)[::-1]
            eps.append(Episode(float(times[a]), float(times[b - 1]), float(times[lo + int(np.argmax(e_in))]), float(seg.max()),
                               float(np.sum(seg / thr[g] - 1)), [joints[i] for i in order[:3]]))
        out[g] = eps
    return out


def overlaps(a: Episode, others: list[Episode], pad: float = 0.25) -> bool:
    return any(a.start - pad <= o.end and o.start - pad <= a.end for o in others)


# --- observation level jitter (2D) -----------------------------------------------------------------------------------

def observation_residual(obs: "Observations") -> np.ndarray:
    """Distance (px) of each used position observation from the fit of order 3 over 0.2 s of its own track; NaN elsewhere.

    The track is the used observations of one marker in one camera over consecutive steps. The distance does not
    depend on the tracker's prediction, so a jump of the observation is not mistaken for a move of the pose.
    """
    A = obs.A
    n = _odd(SPAN_S * obs.rate)
    used = (A[..., 5] == 1) & (A[..., 7] == 0) & np.isfinite(A[..., 0])
    out = np.full(A.shape[:3], np.nan)
    for c in range(A.shape[1]):
        for m in range(A.shape[2]):
            edges = np.flatnonzero(np.diff(np.concatenate([[0], used[:, c, m].astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= n:
                    xy = A[a:b, c, m, 0:2].astype(float)
                    out[a:b, c, m] = np.linalg.norm(xy - savgol_filter(xy, n, ORDER, axis=0), axis=1)
    return out


def observation_jitter(conn: sqlite3.Connection, run: str, t0: float, t1: float, person_id: int = 0,
                       still_px_per_step: float = 0.5) -> dict:
    """RMS 2D residual (px) of the used position observations of a run, all and while the keypoint stays still.

    The residual is that of a fit of order 3 over 0.2 s of the observation of one marker in one camera (runs of
    consecutive used steps); a keypoint is still where the fit moves less than ``still_px_per_step`` per step.
    """
    obs = load_obs(conn, run, person_id)
    n = _odd(SPAN_S * obs.rate)
    h0 = savgol_coeffs(n, ORDER)[n // 2]
    gain = 1.0 / np.sqrt(1.0 - h0)
    sel = (obs.times >= t0) & (obs.times < t1)
    A = obs.A[sel]
    used = (A[..., 5] == 1) & (A[..., 7] == 0) & np.isfinite(A[..., 0])
    sq_all, sq_still = [], []
    for c in range(A.shape[1]):
        for m in range(A.shape[2]):
            ok = used[:, c, m]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a < n + 2:
                    continue
                xy = A[a:b, c, m, 0:2].astype(float)
                fit = savgol_filter(xy, n, ORDER, axis=0)
                r = ((xy - fit) * gain)[n // 2: -(n // 2)]
                speed = np.linalg.norm(np.gradient(fit, axis=0), axis=1)[n // 2: -(n // 2)]
                e = np.sum(r ** 2, axis=1)
                sq_all.append(e)
                sq_still.append(e[speed < still_px_per_step])
    cat = lambda xs: float(np.sqrt(np.concatenate(xs).mean())) if xs and sum(map(len, xs)) else float("nan")
    return {"rms_px": cat(sq_all), "rms_px_still": cat(sq_still), "n": int(sum(map(len, sq_all)))}


def _track_residuals(frames, x, y, n, gain, still_px):
    """Squared residuals (px^2) of 2D tracks over consecutive frames, all and where the fit is still."""
    allr, stillr = [], []
    order = np.argsort(frames)
    f, xy = frames[order], np.column_stack([x[order], y[order]])
    edges = np.flatnonzero(np.diff(f) != 1) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(f)]):
        if b - a >= n + 2:
            fit = savgol_filter(xy[a:b], n, ORDER, axis=0)
            r = ((xy[a:b] - fit) * gain)[n // 2: -(n // 2)]
            speed = np.linalg.norm(np.gradient(fit, axis=0), axis=1)[n // 2: -(n // 2)]
            e = np.sum(r ** 2, axis=1)
            allr.append(e)
            stillr.append(e[speed < still_px])
    return allr, stillr


def pick_jitter(conn: sqlite3.Connection, run: str, person_id: int = 0, still_px_per_frame: float = 0.5) -> dict:
    """2D jitter (px) of the candidates a run picked and of the strongest maxima of the same keypoint-frames.

    Tracks are the picks of one keypoint in one camera over consecutive video frames; the residual is that of a fit of
    order 3 over 0.2 s, corrected for the fit's share of white noise as in the 3D metric. Keypoints that the pick
    changed tracklet for appear in both tracks, so the difference between the two figures is what the choice adds.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kp_gt

    obs = load_obs(conn, run, person_id)
    candidates = kp_gt.read_sequence_candidates(conn, obs.sequence_id, person_id)
    cams = sorted(candidates)
    dt = np.median(np.diff([t for _, t, _ in candidates[cams[0]]]))
    n = _odd(SPAN_S / dt)
    gain = 1.0 / np.sqrt(1.0 - savgol_coeffs(n, ORDER)[n // 2])
    picks, strongest, _ = kp_gt.tracker_picks(conn, run, cams, person_id=person_id)
    out = {}
    for name, pk in (("pick", picks), ("strongest", strongest)):
        allr, stillr = [], []
        for c in range(len(cams)):
            for k in np.unique(pk.keypoint[pk.camera == c]):
                sel = (pk.camera == c) & (pk.keypoint == k)
                a, b = _track_residuals(pk.frame[sel], pk.x[sel].astype(float), pk.y[sel].astype(float), n, gain, still_px_per_frame)
                allr += a
                stillr += b
        cat = lambda xs: float(np.sqrt(np.concatenate(xs).mean())) if xs and sum(map(len, xs)) else float("nan")
        out[name] = {"rms_px": cat(allr), "rms_px_still": cat(stillr), "n": int(sum(map(len, allr)))}
    return out


# --- events ----------------------------------------------------------------------------------------------------------

@dataclass
class Observations:
    times: np.ndarray        # [S] step times (s)
    A: np.ndarray            # [S, C, M, 8] observation results, see the CLAUDE.md note on obs_blob
    cameras: list[str]       # camera labels
    markers: list[str]
    nis: np.ndarray          # [S]
    rate: float
    run_id: str
    sequence_id: str


def load_obs(conn: sqlite3.Connection, run: str, person_id: int = 0) -> Observations:
    row = conn.execute("SELECT id, observation_sequence_id, active_camera_ids, marker_names FROM tracking_runs WHERE id LIKE ?",
                       (run + "%",)).fetchone()
    run_id, seq = row[0], row[1]
    cams, markers = json.loads(row[2]), json.loads(row[3])
    t = {s: (ts, nis) for s, ts, nis in conn.execute(
        "SELECT tracker_step, timestamp_s, nis_value FROM tracking_results WHERE run_id=? AND person_id=? AND is_smoothed=0",
        (run_id, person_id))}
    steps, blobs = [], []
    for s, blob in conn.execute("SELECT tracker_step, obs_blob FROM tracking_obs_results WHERE run_id=? AND person_id=? "
                                "ORDER BY tracker_step", (run_id, person_id)):
        if s in t:
            steps.append(s)
            blobs.append(np.frombuffer(bytes(blob), "<f4").reshape(len(cams), len(markers), 8))
    times = np.array([t[s][0] for s in steps])
    rate = 1.0 / float(np.median(np.diff(times)))
    return Observations(times, np.stack(blobs), cams, markers, np.array([t[s][1] for s in steps]), rate, run_id, seq)


def marker_parents(conn: sqlite3.Connection, run_id: str) -> dict[str, str]:
    y = conn.execute("SELECT s.yaml_content FROM skeletons s JOIN tracking_runs r ON r.skeleton_id = s.id WHERE r.id=?",
                     (run_id,)).fetchone()[0]
    return {m["name"]: m["parent"] for m in yaml.safe_load(y)["markers"]}


def event_arrays(conn: sqlite3.Connection, obs: Observations, person_id: int = 0) -> dict[str, np.ndarray]:
    """Boolean event arrays ``[S, C, M]`` of a run (see the module comment for the definitions)."""
    A = obs.A
    S, C, M, _ = A.shape
    pos_mode = A[..., 7] == 0
    used = (A[..., 5] == 1) & np.isfinite(A[..., 0])
    ev = {"missing": ~np.isfinite(A[..., 0]), "outlier": A[..., 6] == 1}
    res = observation_residual(obs)
    ev["jump"] = np.nan_to_num(res, nan=0.0) > JUMP_PX
    run = np.zeros((S, C, M), bool)
    count = np.zeros((C, M), int)
    for s in range(S):
        count = np.where(ev["missing"][s], count + 1, 0)
        run[s] = count >= MISSING_RUN_STEPS
    ev["missing_run"] = run
    n_used = used.sum(axis=1)                                  # [S, M]
    change = np.zeros((S, M), bool)
    change[1:] = n_used[1:] != n_used[:-1]
    ev["camera_count"] = np.broadcast_to(change[:, None, :], (S, C, M)).copy()
    w = int(round(obs.rate))
    med = np.array([np.nanmedian(obs.nis[max(0, s - w // 2): s + w // 2 + 1]) for s in range(S)])
    ev["nis_spike"] = np.broadcast_to((obs.nis > NIS_SPIKE_FACTOR * med)[:, None, None], (S, C, M)).copy()
    ev["tracklet_switch"] = _switches(conn, obs, person_id)
    return ev


def _switches(conn: sqlite3.Connection, obs: Observations, person_id: int) -> np.ndarray:
    """Pick events of a windowed run: the picked tracklet of a keypoint differs from that of the previous frame."""
    out = np.zeros(obs.A.shape[:3], bool)
    picks = {r[0]: r[1:] for r in conn.execute(
        "SELECT camera_instance_id, n_frames, n_keypoints, video_frames, tracklet_ids FROM tracking_kp_picks "
        "WHERE run_id=? AND person_id=?", (obs.run_id, person_id))}
    if not picks:
        return out
    camera_of = {r[0]: r[1] for r in conn.execute("SELECT id, label FROM camera_instances")}
    names = {r[0]: r[1] for r in conn.execute(
        "SELECT keypoint_idx, name FROM pose_sequence_keypoints WHERE sequence_id=? AND obs_source='kp_candidates'", (obs.sequence_id,))}
    marker_of = {kp: obs.markers.index(n.split(":", 1)[1]) for kp, n in names.items() if n.split(":", 1)[-1] in obs.markers}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kp_gt

    candidates = kp_gt.read_sequence_candidates(conn, obs.sequence_id, person_id)
    for cam, (n, k, frames, tracklets) in picks.items():
        if camera_of.get(cam) not in obs.cameras or cam not in candidates:
            continue
        c = obs.cameras.index(camera_of[cam])
        time_of = {f: t for f, t, _ in candidates[cam]}
        frames = np.frombuffer(frames, "<i4")
        tl = np.frombuffer(tracklets, "<i4").reshape(n, k)
        sw = (tl[1:] >= 0) & (tl[:-1] >= 0) & (tl[1:] != tl[:-1])
        for i, kp in zip(*np.nonzero(sw)):
            if kp in marker_of and frames[i + 1] in time_of:
                s = int(np.argmin(np.abs(obs.times - time_of[frames[i + 1]])))
                out[s, c, marker_of[kp]] = True
    return out


def coincidence(ev: dict[str, np.ndarray], obs: Observations, parents: dict[str, str], group: str,
                a: float, b: float) -> dict[str, tuple[float, float]]:
    """Per event, its rate inside [a, b] and over the whole trial, on the markers of a joint group's joints."""
    cols = [i for i, m in enumerate(obs.markers) if parents.get(m) in GROUPS[group]]
    inside = (obs.times >= a) & (obs.times <= b)
    return {k: (float(v[inside][:, :, cols].mean()), float(v[:, :, cols].mean())) for k, v in ev.items()}


def explained(times, energy, ev, obs, parents, group, speed, fs: float = FS) -> dict:
    """Variance of the group's smoothed jitter (RMS) that the smoothed event densities explain.

    Fast motion raises both the jitter and the event rates, so the regression always has the group's speed as its
    first regressor: ``motion`` is the R^2 of the speed alone, ``all`` that of the speed and every event, and each
    event's entry the gain in R^2 when it is added to the speed alone.
    """
    cols = [i for i, m in enumerate(obs.markers) if parents.get(m) in GROUPS[group]]
    step_of = np.array([int(np.argmin(np.abs(obs.times - t))) for t in times])
    w = int(round(WINDOW_S * fs))
    smooth = lambda x: np.convolve(np.nan_to_num(x), np.ones(w) / w, "same")
    y = np.sqrt(smooth(energy[group]))
    ok = np.isfinite(sliding_rms(energy[group]))
    X = {k: smooth(v[step_of][:, :, cols].mean(axis=(1, 2)).astype(float)) for k, v in ev.items()}

    v = smooth(speed[group])

    def r2(keys, motion=True):
        design = np.column_stack([np.ones(ok.sum())] + ([v[ok]] if motion else []) + [X[k][ok] for k in keys])
        beta, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
        resid = y[ok] - design @ beta
        return float(1 - resid.var() / y[ok].var())

    base = r2([])
    return {"motion": base, "all": r2(list(X)), **{k: r2([k]) - base for k in X}}


# --- command line ----------------------------------------------------------------------------------------------------

def synthetic_check(sigma: float = 0.004, seed: int = 0) -> dict:
    """A smooth motion plus white noise of known standard deviation: the metric must return the noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 20, 1.0 / FS)
    smooth = np.column_stack([0.3 * np.sin(2 * np.pi * 0.7 * t), 0.2 * np.sin(2 * np.pi * 1.3 * t + 1), 0.1 * np.sin(2 * np.pi * 0.4 * t)])
    noise = rng.normal(0, sigma, smooth.shape)
    r = residual(smooth + noise)
    rms = float(np.sqrt(np.nanmean(np.sum(r ** 2, axis=1))))
    clean = jerk(smooth)
    expect = np.sqrt(np.mean(np.sum(np.column_stack([0.3 * (2 * np.pi * 0.7) ** 3 * np.cos(2 * np.pi * 0.7 * t),
                                                     0.2 * (2 * np.pi * 1.3) ** 3 * np.cos(2 * np.pi * 1.3 * t + 1),
                                                     0.1 * (2 * np.pi * 0.4) ** 3 * np.cos(2 * np.pi * 0.4 * t)]) ** 2, axis=1)))
    return {"noise_3d_rms": sigma * np.sqrt(3), "measured": rms, "smooth_residual": float(np.sqrt(np.nanmean(np.sum(residual(smooth) ** 2, axis=1)))),
            "jerk_measured": float(np.sqrt(np.nanmean(np.sum(clean ** 2, axis=1)))), "jerk_expected": float(expect)}


def _runs(path: str) -> dict[str, dict[str, str]]:
    return json.load(open(path))


def cmd_measure(args) -> int:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    runs = _runs(args.runs)
    data = {}
    for variant, persons in runs.items():
        for person, run in persons.items():
            if run is None:
                continue
            rp = RunPose(conn, run, args.states)
            times, names, pos = rp.positions(args.start, args.end)
            data[(variant, person)] = (rp, times, names, pos, group_energy(pos, names))
            print(f"{variant:12s} {person:6s} {run}  states {'smoothed' if rp.smoothed else 'filtered'}", file=sys.stderr)
    subjects = args.reference_subjects.split(",") if args.reference_subjects else None
    ref = [v[4] for (variant, person), v in data.items() if variant == args.reference and (subjects is None or person in subjects)]
    thr = thresholds(ref, args.percentile)
    out = {"threshold_rms_mm": {g: 1000 * v for g, v in thr.items()}, "figures": {}, "episodes": {}, "observation_2d": {}}
    for (variant, person), (rp, times, names, pos, energy) in data.items():
        out["figures"][f"{variant}.{person}"] = figures(pos, names)
        eps = episodes_of(times, names, pos, energy, thr)
        out["episodes"][f"{variant}.{person}"] = {g: [e.__dict__ for e in es] for g, es in eps.items()}
        if args.obs:
            out["observation_2d"][f"{variant}.{person}"] = observation_jitter(conn, rp.run_id, args.start, args.end)
    # episodes of each run that the reference run of the subject does not have
    for (variant, person), (rp, times, names, pos, energy) in data.items():
        if variant == args.reference or (args.reference, person) not in data:
            continue
        mine = out["episodes"][f"{variant}.{person}"]
        ref_eps = out["episodes"][f"{args.reference}.{person}"]
        for g in GROUPS:
            ref_g = [Episode(**e) for e in ref_eps[g]]
            for e in mine[g]:
                e["in_reference"] = overlaps(Episode(**e), ref_g)
    json.dump(out, open(args.out, "w"), indent=1)
    _print_tables(out, runs)
    return 0


def _print_tables(out: dict, runs: dict) -> None:
    print("threshold of the sliding 0.5 s RMS per group (mm):", {g: round(v, 1) for g, v in out["threshold_rms_mm"].items()})
    print("\nRMS residual (mm) / RMS jerk (m/s^3) over the trial, per run and group")
    print(f"{'run':16s}" + "".join(f"{g:>16s}" for g in GROUPS))
    for k, f in out["figures"].items():
        print(f"{k:16s}" + "".join(f"{f[g]['rms_mm']:8.1f}/{f[g]['jerk']:6.0f} " for g in GROUPS))
    print("\nepisodes (count / total seconds / not in the reference run)")
    for k, d in out["episodes"].items():
        cells = []
        for g in GROUPS:
            es = d[g]
            new = "  -" if es and "in_reference" not in es[0] else f"{sum(not e.get('in_reference', False) for e in es):3d}"
            cells.append(f"{len(es):3d}/{sum(e['end'] - e['start'] for e in es):5.1f}/{new}")
        print(f"{k:16s}" + "".join(f"{c:>16s}" for c in cells))
    if out["observation_2d"]:
        print("\n2D residual of the used observations (px): all / keypoint still")
        for k, v in out["observation_2d"].items():
            print(f"{k:16s} {v['rms_px']:6.2f} / {v['rms_px_still']:6.2f}   (n={v['n']})")


def cmd_causes(args) -> int:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    report = json.load(open(args.measure))
    runs = _runs(args.runs)[args.variant]
    result = {}
    for person, run in runs.items():
        rp = RunPose(conn, run, args.states)
        times, names, pos = rp.positions(args.start, args.end)
        energy = group_energy(pos, names)
        speed = group_speed(pos, names)
        obs = load_obs(conn, run)
        parents = marker_parents(conn, obs.run_id)
        ev = event_arrays(conn, obs)
        res = {"events_per_marker_camera_step": {k: float(v.mean()) for k, v in ev.items()}, "episodes": [], "explained": {}}
        for g in GROUPS:
            res["explained"][g] = explained(times, energy, ev, obs, parents, g, speed)
            for e in report["episodes"][f"{args.variant}.{person}"][g]:
                co = coincidence(ev, obs, parents, g, e["start"], e["end"])
                res["episodes"].append({"group": g, **e, "rates": {k: {"episode": a, "trial": b} for k, (a, b) in co.items()}})
        result[person] = res
        print(person, "events per marker-camera-step:", {k: round(v, 4) for k, v in res["events_per_marker_camera_step"].items()})
        for g in GROUPS:
            print(f"  {g:6s} R^2 of the jitter on the events: " + ", ".join(f"{k} {v:.2f}" for k, v in res["explained"][g].items()))
    json.dump(result, open(args.out, "w"), indent=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("measure", "causes"):
        p = sub.add_parser(name)
        p.add_argument("--db", required=True)
        p.add_argument("--runs", required=True, help="JSON: variant -> {subject -> tracking run id}")
        p.add_argument("--start", type=float, default=43.5)
        p.add_argument("--end", type=float, default=66.4)
        p.add_argument("--states", choices=["auto", "filtered"], default="auto",
                       help="'auto' uses the smoothed states of a run when it has them")
        p.add_argument("--out", required=True)
        if name == "measure":
            p.add_argument("--reference", default="edited", help="variant whose runs set the thresholds")
            p.add_argument("--percentile", type=float, default=99.0)
            p.add_argument("--reference-subjects", help="comma separated subjects whose reference runs set the thresholds (default all)")
            p.add_argument("--obs", action="store_true", help="also the 2D jitter of the observations")
        else:
            p.add_argument("--measure", required=True, help="the JSON written by measure")
            p.add_argument("--variant", default="window")
    p = sub.add_parser("picks")
    p.add_argument("--db", required=True)
    p.add_argument("--run", action="append", required=True, help="tracking run id of a run with picks (repeatable)")
    sub.add_parser("synthetic")
    args = ap.parse_args(argv)
    if args.cmd == "picks":
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        for run in args.run:
            print(run, json.dumps(pick_jitter(conn, run)))
        return 0
    if args.cmd == "synthetic":
        print(json.dumps(synthetic_check(), indent=1))
        return 0
    return {"measure": cmd_measure, "causes": cmd_causes}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
