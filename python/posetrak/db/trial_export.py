"""trial_export.py — Export and import trial data between posetrak session databases.

All reads from the source DB use a read-only URI connection so corrupted or
foreign DBs are opened without triggering Python-side migrations.  Each table
copy is wrapped independently so a bad table (e.g. pose_observation_edits with
B-tree corruption) can be skipped via ``skip_tables`` without aborting the run.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ExportScope(Enum):
    CAPTURE_ONLY   = "capture-only"
    TRIAL_ONLY     = "trial-only"
    DETECTION_ONLY = "detection-only"
    FULL           = "full"


@dataclass
class AnchorSpec:
    trial_ids:        list[str] = field(default_factory=list)
    capture_ids:      list[str] = field(default_factory=list)
    detection_ids:    list[str] = field(default_factory=list)
    tracking_run_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.trial_ids,
            self.capture_ids,
            self.detection_ids,
            self.tracking_run_ids,
        ])


@dataclass
class TableResult:
    table: str
    rows_copied: int
    error: str | None = None


@dataclass
class ExportResult:
    anchor: AnchorSpec
    tables: list[TableResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(t.error is None for t in self.tables)

    @property
    def total_rows(self) -> int:
        return sum(t.rows_copied for t in self.tables)


# ---------------------------------------------------------------------------
# Internal: plan (set of IDs to copy per table)
# ---------------------------------------------------------------------------


@dataclass
class _Plan:
    # Registry tables
    camera_model_ids:    set[str] = field(default_factory=set)
    camera_mode_ids:     set[str] = field(default_factory=set)
    camera_instance_ids: set[str] = field(default_factory=set)
    intrinsics_ids:      set[str] = field(default_factory=set)
    skeleton_ids:        set[str] = field(default_factory=set)
    config_ids:          set[str] = field(default_factory=set)
    # Session tables
    session_ids:         set[str] = field(default_factory=set)
    extrinsic_cal_ids:   set[str] = field(default_factory=set)
    capture_ids:         set[str] = field(default_factory=set)
    sync_config_ids:     set[str] = field(default_factory=set)
    sync_anchor_ids:     set[str] = field(default_factory=set)
    trial_ids:           set[str] = field(default_factory=set)
    detection_run_ids:   set[str] = field(default_factory=set)
    seg_quality_ids:     set[str] = field(default_factory=set)
    sequence_ids:        set[str] = field(default_factory=set)
    tracking_run_ids:    set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Internal: low-level copy helpers
# ---------------------------------------------------------------------------


def _ph(n: int) -> str:
    return ", ".join(["?"] * n)


def _fetch_col(
    conn: sqlite3.Connection,
    table: str,
    col: str,
    where_col: str,
    where_ids: set[str],
) -> set[str]:
    """Return {row[col] for row in table WHERE where_col IN where_ids}."""
    if not where_ids:
        return set()
    try:
        rows = conn.execute(
            f"SELECT {col} FROM {table} WHERE {where_col} IN ({_ph(len(where_ids))})",
            list(where_ids),
        ).fetchall()
    except sqlite3.DatabaseError:
        return set()
    return {r[0] for r in rows if r[0] is not None}


def _copy_by_ids(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    ids: set[str],
    *,
    dry_run: bool = False,
) -> TableResult:
    if not ids:
        return TableResult(table=table, rows_copied=0)
    try:
        rows = src.execute(
            f"SELECT * FROM {table} WHERE id IN ({_ph(len(ids))})",
            list(ids),
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        return TableResult(table=table, rows_copied=0, error=str(exc))
    return _insert_rows(src, dst, table, rows, dry_run=dry_run)


def _copy_where(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    col: str,
    parent_ids: set[str],
    *,
    dry_run: bool = False,
) -> TableResult:
    if not parent_ids:
        return TableResult(table=table, rows_copied=0)
    try:
        rows = src.execute(
            f"SELECT * FROM {table} WHERE {col} IN ({_ph(len(parent_ids))})",
            list(parent_ids),
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        return TableResult(table=table, rows_copied=0, error=str(exc))
    return _insert_rows(src, dst, table, rows, dry_run=dry_run)


def _insert_rows(
    src: sqlite3.Connection,
    dst: sqlite3.Connection | None,
    table: str,
    rows: list[sqlite3.Row],
    *,
    dry_run: bool = False,
) -> TableResult:
    if not rows:
        return TableResult(table=table, rows_copied=0)

    # In dry-run mode we just count — no column inspection or writes needed.
    if dry_run:
        return TableResult(table=table, rows_copied=len(rows))

    assert dst is not None
    src_cols = list(rows[0].keys())
    try:
        dst_cols = {r[1] for r in dst.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError as exc:
        return TableResult(table=table, rows_copied=0, error=str(exc))

    use_cols = [c for c in src_cols if c in dst_cols]
    if not use_cols:
        return TableResult(table=table, rows_copied=0, error="no matching columns")

    col_names = ", ".join(use_cols)
    placeholders = _ph(len(use_cols))
    sql = f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"
    count = 0
    try:
        for row in rows:
            cur = dst.execute(sql, tuple(row[c] for c in use_cols))
            count += cur.rowcount
    except sqlite3.DatabaseError as exc:
        return TableResult(table=table, rows_copied=count, error=str(exc))
    return TableResult(table=table, rows_copied=count)


# ---------------------------------------------------------------------------
# Internal: dependency resolution
# ---------------------------------------------------------------------------


def _resolve_plan(
    src: sqlite3.Connection,
    anchor: AnchorSpec,
    scope: ExportScope,
) -> _Plan:
    p = _Plan()

    # Seed from anchor
    p.trial_ids         = set(anchor.trial_ids)
    p.capture_ids       = set(anchor.capture_ids)
    p.detection_run_ids = set(anchor.detection_ids)
    p.tracking_run_ids  = set(anchor.tracking_run_ids)

    # Empty anchor = export everything reachable
    if anchor.is_empty():
        try:
            p.capture_ids       |= {r[0] for r in src.execute("SELECT id FROM captures")}
            p.trial_ids         |= {r[0] for r in src.execute("SELECT id FROM trials")}
            p.detection_run_ids |= {r[0] for r in src.execute("SELECT id FROM detection_runs")}
            p.tracking_run_ids  |= {r[0] for r in src.execute("SELECT id FROM tracking_runs")}
        except sqlite3.DatabaseError:
            pass

    # ── Phase 1: Walk DOWN from captures/trials based on scope ──────────────

    if scope != ExportScope.CAPTURE_ONLY:
        p.trial_ids |= _fetch_col(src, "trials", "id", "capture_id", p.capture_ids)

    if scope in (ExportScope.DETECTION_ONLY, ExportScope.FULL):
        p.detection_run_ids |= _fetch_col(src, "detection_runs", "id", "shot_id",   p.capture_ids)
        p.detection_run_ids |= _fetch_col(src, "detection_runs", "id", "trial_id",  p.trial_ids)
        p.sequence_ids      |= _fetch_col(src, "pose_observation_sequences", "id", "detection_run_id", p.detection_run_ids)
        p.sequence_ids      |= _fetch_col(src, "pose_observation_sequences", "id", "shot_id", p.capture_ids)

    if scope == ExportScope.FULL:
        p.tracking_run_ids |= _fetch_col(src, "tracking_runs", "id", "observation_sequence_id", p.sequence_ids)

    # ── Phase 2: Walk UP from all collected items ───────────────────────────

    # tracking_runs → sequences, configs, skeletons, extrinsics, sync
    if p.tracking_run_ids:
        try:
            for row in src.execute(
                "SELECT observation_sequence_id, tracker_config_id, skeleton_id, "
                f"extrinsic_calibration_id, sync_config_id FROM tracking_runs "
                f"WHERE id IN ({_ph(len(p.tracking_run_ids))})",
                list(p.tracking_run_ids),
            ).fetchall():
                p.sequence_ids.add(row[0])
                if row[1]: p.config_ids.add(row[1])
                if row[2]: p.skeleton_ids.add(row[2])
                if row[3]: p.extrinsic_cal_ids.add(row[3])
                if row[4]: p.sync_config_ids.add(row[4])
        except sqlite3.DatabaseError:
            pass

    # sequences → captures, detection_runs, sync_configs
    if p.sequence_ids:
        try:
            for row in src.execute(
                "SELECT shot_id, detection_run_id, sync_config_id "
                f"FROM pose_observation_sequences WHERE id IN ({_ph(len(p.sequence_ids))})",
                list(p.sequence_ids),
            ).fetchall():
                p.capture_ids.add(row[0])
                if row[1]:
                    p.detection_run_ids.add(row[1])
                if row[2]: p.sync_config_ids.add(row[2])
        except sqlite3.DatabaseError:
            pass

    # detection_runs → captures, sync_configs, trials
    if p.detection_run_ids:
        try:
            for row in src.execute(
                "SELECT shot_id, sync_config_id, trial_id FROM detection_runs "
                f"WHERE id IN ({_ph(len(p.detection_run_ids))})",
                list(p.detection_run_ids),
            ).fetchall():
                p.capture_ids.add(row[0])
                if row[1]: p.sync_config_ids.add(row[1])
                if row[2] and scope != ExportScope.CAPTURE_ONLY:
                    p.trial_ids.add(row[2])
        except sqlite3.DatabaseError:
            pass

    # trials → captures
    if p.trial_ids:
        p.capture_ids |= _fetch_col(src, "trials", "capture_id", "id", p.trial_ids)

    # captures → sessions, extrinsics
    if p.capture_ids:
        try:
            for row in src.execute(
                "SELECT session_id, extrinsic_calibration_id FROM captures "
                f"WHERE id IN ({_ph(len(p.capture_ids))})",
                list(p.capture_ids),
            ).fetchall():
                p.session_ids.add(row[0])
                if row[1]: p.extrinsic_cal_ids.add(row[1])
        except sqlite3.DatabaseError:
            pass

    # captures → sync_configs, sync_anchors
    p.sync_config_ids |= _fetch_col(src, "sync_configs", "id", "shot_id", p.capture_ids)
    p.sync_anchor_ids |= _fetch_col(src, "sync_anchors", "id", "shot_id", p.capture_ids)

    # captures → videos → cameras
    if p.capture_ids:
        try:
            for row in src.execute(
                "SELECT camera_instance_id, camera_mode_id, intrinsics_calibration_id "
                f"FROM capture_videos WHERE shot_id IN ({_ph(len(p.capture_ids))})",
                list(p.capture_ids),
            ).fetchall():
                p.camera_instance_ids.add(row[0])
                if row[1]: p.camera_mode_ids.add(row[1])
                if row[2]: p.intrinsics_ids.add(row[2])
        except sqlite3.DatabaseError:
            pass

    # cameras → models
    p.camera_model_ids |= _fetch_col(src, "camera_instances", "camera_model_id", "id", p.camera_instance_ids)
    p.camera_model_ids |= _fetch_col(src, "camera_modes",     "camera_model_id", "id", p.camera_mode_ids)

    # seg_quality_runs is capture-scoped (shot_id/trial_id of its own, see
    # docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md),
    # not tied to any one detection run -- fetched by shot_id once
    # p.capture_ids is fully settled (every walk-up/walk-down pass above
    # may still add captures), not tied to `scope`, since a segmentation
    # can exist -- and be worth exporting -- before any detection run does.
    p.seg_quality_ids |= _fetch_col(src, "seg_quality_runs", "id", "shot_id", p.capture_ids)

    return p


# ---------------------------------------------------------------------------
# Internal: execute a plan
# ---------------------------------------------------------------------------


def _execute_plan(
    src: sqlite3.Connection,
    dst: sqlite3.Connection | None,
    plan: _Plan,
    scope: ExportScope,
    include_cache: bool,
    skip_tables: set[str],
    dry_run: bool,
    on_progress: Callable[[str], None] | None,
) -> list[TableResult]:
    results: list[TableResult] = []

    def _log(tr: TableResult) -> None:
        results.append(tr)
        if on_progress:
            status = f"  {tr.table}: {tr.rows_copied} rows"
            if tr.error:
                status += f" [WARNING: {tr.error}]"
            on_progress(status)

    def by_ids(table: str, ids: set[str]) -> None:
        if table in skip_tables:
            if on_progress:
                on_progress(f"  {table}: skipped (--skip-tables)")
            return
        _log(_copy_by_ids(src, dst, table, ids, dry_run=dry_run))

    def by_parent(table: str, col: str, parent_ids: set[str]) -> None:
        if table in skip_tables:
            if on_progress:
                on_progress(f"  {table}: skipped (--skip-tables)")
            return
        _log(_copy_where(src, dst, table, col, parent_ids, dry_run=dry_run))

    # ── Registry tables (always included) ───────────────────────────────────
    by_ids("camera_models",           plan.camera_model_ids)
    by_ids("camera_modes",            plan.camera_mode_ids)
    by_ids("camera_instances",        plan.camera_instance_ids)
    by_ids("intrinsics_calibrations", plan.intrinsics_ids)
    if scope == ExportScope.FULL:
        by_ids("skeletons",       plan.skeleton_ids)
        by_ids("tracker_configs", plan.config_ids)

    # ── Session infrastructure ───────────────────────────────────────────────
    by_ids("mocap_sessions",           plan.session_ids)
    by_parent("session_cameras",          "session_id",                plan.session_ids)
    by_ids("extrinsic_calibrations",   plan.extrinsic_cal_ids)
    by_parent("extrinsic_entries",        "extrinsic_calibration_id",  plan.extrinsic_cal_ids)
    by_ids("captures",                 plan.capture_ids)
    by_parent("capture_videos",           "shot_id",                   plan.capture_ids)
    by_ids("sync_configs",             plan.sync_config_ids)
    by_parent("sync_points",              "sync_config_id",             plan.sync_config_ids)
    by_ids("sync_anchors",             plan.sync_anchor_ids)
    by_parent("sync_anchor_observations", "sync_anchor_id",             plan.sync_anchor_ids)

    if scope != ExportScope.CAPTURE_ONLY:
        by_ids("trials", plan.trial_ids)

    # ── Detection data ───────────────────────────────────────────────────────
    if scope in (ExportScope.DETECTION_ONLY, ExportScope.FULL):
        by_ids("detection_runs",              plan.detection_run_ids)
        by_parent("detection_keypoints",       "detection_run_id", plan.detection_run_ids)
        by_parent("person_detections",         "detection_run_id", plan.detection_run_ids)
        by_parent("person_tracks",             "detection_run_id", plan.detection_run_ids)
        by_parent("detection_track_assignments","detection_run_id", plan.detection_run_ids)
        by_ids("seg_quality_runs",             plan.seg_quality_ids)
        by_parent("keypoint_obs_quality",      "seg_run_id",       plan.seg_quality_ids)
        by_parent("seg_masks",                 "seg_quality_run_id", plan.seg_quality_ids)
        by_ids("pose_observation_sequences",   plan.sequence_ids)
        by_parent("sequence_persons",          "sequence_id",      plan.sequence_ids)
        by_parent("pose_observations",         "sequence_id",      plan.sequence_ids)
        by_parent("pose_observation_edits",    "sequence_id",      plan.sequence_ids)

    # ── Tracking data ────────────────────────────────────────────────────────
    if scope == ExportScope.FULL:
        by_ids("tracking_runs",            plan.tracking_run_ids)
        by_parent("tracking_run_persons",  "run_id", plan.tracking_run_ids)
        by_parent("tracking_results",      "run_id", plan.tracking_run_ids)
        by_parent("tracking_obs_results",  "run_id", plan.tracking_run_ids)

    # ── Optional frame cache ─────────────────────────────────────────────────
    if include_cache and plan.capture_ids:
        cv_ids: set[str] = {
            r[0]
            for r in src.execute(
                f"SELECT id FROM capture_videos WHERE shot_id IN ({_ph(len(plan.capture_ids))})",
                list(plan.capture_ids),
            ).fetchall()
        }
        by_parent("frame_cache_entries", "shot_video_id", cv_ids)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def open_source_readonly(path: Path) -> sqlite3.Connection:
    """Open *path* read-only via URI; no migration code is called."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def export_trials(
    src: sqlite3.Connection,
    dst: sqlite3.Connection | None,
    anchor: AnchorSpec,
    *,
    scope: ExportScope,
    include_cache: bool = False,
    skip_tables: set[str] = frozenset(),
    on_progress: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> ExportResult:
    """Copy rows matching *anchor* from *src* into *dst* at the given *scope*.

    When ``dry_run=True``, *dst* may be ``None``; rows are counted from the
    source but nothing is written.  Otherwise *dst* must already have the
    session schema applied (use ``create_session()`` or ``open_session()``).
    """
    result = ExportResult(anchor=anchor)

    try:
        plan = _resolve_plan(src, anchor, scope)
    except sqlite3.DatabaseError as exc:
        result.tables.append(TableResult(table="<resolve>", rows_copied=0, error=str(exc)))
        return result

    if not dry_run:
        assert dst is not None
        dst.execute("PRAGMA foreign_keys = OFF")

    result.tables = _execute_plan(src, dst, plan, scope, include_cache, skip_tables, dry_run, on_progress)

    if not dry_run:
        assert dst is not None
        dst.commit()
        dst.execute("PRAGMA foreign_keys = ON")

    return result


def import_trials(
    src: sqlite3.Connection,
    dst_session: sqlite3.Connection,
    dst_registry: sqlite3.Connection | None = None,
    anchor: AnchorSpec | None = None,
    *,
    skip_tables: set[str] = frozenset(),
    dry_run: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> ExportResult:
    """Import rows from *src* (a previously exported session DB) into *dst_session*.

    If *dst_registry* is also provided, the camera/skeleton/config registry
    tables are mirrored there as well (INSERT OR IGNORE).  This is the
    ``camera import-session`` equivalent at the session level.

    When *dry_run* is ``True``, rows are counted but not written; the result
    reports how many rows would be copied.
    """
    if anchor is None:
        anchor = AnchorSpec()

    result = ExportResult(anchor=anchor)

    try:
        plan = _resolve_plan(src, anchor, ExportScope.FULL)
    except sqlite3.DatabaseError as exc:
        result.tables.append(TableResult(table="<resolve>", rows_copied=0, error=str(exc)))
        return result

    # Copy everything into dst_session
    dst_session.execute("PRAGMA foreign_keys = OFF")
    result.tables = _execute_plan(
        src, dst_session, plan, ExportScope.FULL,
        include_cache=False, skip_tables=skip_tables,
        dry_run=dry_run, on_progress=on_progress,
    )
    if not dry_run:
        dst_session.commit()
    dst_session.execute("PRAGMA foreign_keys = ON")

    # Mirror registry tables to dst_registry
    if dst_registry is not None:
        if on_progress:
            on_progress("Registry sync:")
        dst_registry.execute("PRAGMA foreign_keys = OFF")
        registry_results = _execute_plan(
            src, dst_registry, plan, ExportScope.FULL,
            include_cache=False,
            skip_tables=skip_tables | {
                # Only registry tables are meaningful here; skip session-only tables
                "mocap_sessions", "session_cameras", "extrinsic_calibrations",
                "extrinsic_entries", "captures", "capture_videos",
                "sync_configs", "sync_points", "sync_anchors", "sync_anchor_observations",
                "trials", "detection_runs", "detection_keypoints", "person_detections",
                "person_tracks", "detection_track_assignments",
                "seg_quality_runs", "keypoint_obs_quality", "seg_masks",
                "pose_observation_sequences", "sequence_persons", "pose_observations",
                "pose_observation_edits", "tracking_runs", "tracking_run_persons",
                "tracking_results", "tracking_obs_results", "frame_cache_entries",
            },
            dry_run=dry_run, on_progress=on_progress,
        )
        if not dry_run:
            dst_registry.commit()
        dst_registry.execute("PRAGMA foreign_keys = ON")
        result.tables.extend(registry_results)

    return result
