"""Tests for trial list / export / import CLI commands and the trial_export library."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import create_session, generate_id, open_session
from posetrak.db.trial_export import (
    AnchorSpec,
    ExportScope,
    export_trials,
    import_trials,
    open_source_readonly,
)


# ---------------------------------------------------------------------------
# Helpers for building a seeded session DB
# ---------------------------------------------------------------------------


def _make_full_session(path: Path) -> dict:
    """Create a session DB with one of everything; return IDs."""
    conn = create_session(path)

    ids: dict = {}
    ids["camera_model"]  = generate_id()
    ids["camera_mode"]   = generate_id()
    ids["camera_inst"]   = generate_id()
    ids["intrinsics"]    = generate_id()
    ids["skeleton"]      = generate_id()
    ids["config"]        = generate_id()
    ids["session"]       = generate_id()
    ids["extrinsic_cal"] = generate_id()
    ids["capture"]       = generate_id()
    ids["video"]         = generate_id()
    ids["sync"]          = generate_id()
    ids["trial"]         = generate_id()
    ids["detection"]     = generate_id()
    ids["sequence"]      = generate_id()
    ids["tracking_run"]  = generate_id()

    conn.execute("PRAGMA foreign_keys = OFF")

    # Registry tables
    conn.execute("INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?,?,?)",
                 (ids["camera_model"], "TestCo", "CamX"))
    conn.execute(
        "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, nominal_fps) VALUES (?,?,?,?,?)",
        (ids["camera_mode"], ids["camera_model"], 1920, 1080, 60.0),
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?,?,?)",
        (ids["camera_inst"], ids["camera_model"], "cam-a"),
    )
    conn.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, fx, fy, cx, cy, distortion_model) VALUES (?,?,?,?,?,?,?,?)",
        (ids["intrinsics"], ids["camera_mode"], "2026-01-01", 1000.0, 1000.0, 960.0, 540.0, "radtan"),
    )
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) VALUES (?,?,?,?)",
        (ids["skeleton"], "TestSkel", "joints: []", "2026-01-01"),
    )
    conn.execute(
        "INSERT INTO tracker_configs (id, name, created_at) VALUES (?,?,?)",
        (ids["config"], "default", "2026-01-01"),
    )

    # Session tables
    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES (?,?)",
                 (ids["session"], "2026-01-01"))
    conn.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at) VALUES (?,?,?)",
        (ids["extrinsic_cal"], ids["session"], "2026-01-01"),
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, extrinsic_calibration_id, capture_number, label) VALUES (?,?,?,?,?)",
        (ids["capture"], ids["session"], ids["extrinsic_cal"], 1, "cap-1"),
    )
    conn.execute(
        "INSERT INTO capture_videos "
        "(id, shot_id, camera_instance_id, file_path, first_video_frame, last_video_frame, actual_fps,"
        " camera_mode_id, intrinsics_calibration_id) VALUES (?,?,?,?,?,?,?,?,?)",
        (ids["video"], ids["capture"], ids["camera_inst"], "/fake/a.mp4", 0, 900, 30.0,
         ids["camera_mode"], ids["intrinsics"]),
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES (?,?)",
        (ids["sync"], ids["capture"]),
    )
    conn.execute(
        "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id, video_frame, timestamp_s) "
        "VALUES (?,?,?,?,?)",
        (ids["sync"], ids["camera_inst"], ids["video"], 0, 0.0),
    )
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) VALUES (?,?,?,?,?)",
        (ids["trial"], ids["capture"], "take-1", 5.0, 25.0),
    )
    conn.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, "
        "detector_model, pose_model, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (ids["detection"], ids["capture"], ids["sync"], ids["trial"],
         5.0, 25.0, "yolo", "rtmpose", "2026-01-01"),
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id) VALUES (?,?,?,?,?,?)",
        (ids["sequence"], ids["capture"], ids["sync"], 5.0, 25.0, ids["detection"]),
    )
    conn.execute(
        "INSERT INTO tracking_runs "
        "(id, observation_sequence_id, tracker_config_id, skeleton_id, extrinsic_calibration_id,"
        " sync_config_id, ran_at, posetrak_version, active_camera_ids, marker_names) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (ids["tracking_run"], ids["sequence"], ids["config"], ids["skeleton"],
         ids["extrinsic_cal"], ids["sync"], "2026-01-01", "0.1", "[]", "[]"),
    )

    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.close()
    return ids


# ---------------------------------------------------------------------------
# Library: export_trials
# ---------------------------------------------------------------------------


class TestExportTrials:
    def test_capture_only_scope(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        dst_path = tmp_path / "dst.db"
        dst = create_session(dst_path)
        src = open_source_readonly(src_path)

        anchor = AnchorSpec(capture_ids=[ids["capture"]])
        result = export_trials(src, dst, anchor, scope=ExportScope.CAPTURE_ONLY)

        dst.commit()
        assert result.success

        # Capture infrastructure present
        assert dst.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM camera_instances").fetchone()[0] == 1

        # Detection / tracking not copied
        assert dst.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0
        assert dst.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0] == 0
        assert dst.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 0

        src.close(); dst.close()

    def test_trial_only_scope(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        dst_path = tmp_path / "dst.db"
        dst = create_session(dst_path)
        src = open_source_readonly(src_path)

        anchor = AnchorSpec(trial_ids=[ids["trial"]])
        result = export_trials(src, dst, anchor, scope=ExportScope.TRIAL_ONLY)

        dst.commit()
        assert result.success
        assert dst.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0] == 0
        assert dst.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 0

        src.close(); dst.close()

    def test_detection_only_scope(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        dst_path = tmp_path / "dst.db"
        dst = create_session(dst_path)
        src = open_source_readonly(src_path)

        anchor = AnchorSpec(detection_ids=[ids["detection"]])
        result = export_trials(src, dst, anchor, scope=ExportScope.DETECTION_ONLY)

        dst.commit()
        assert result.success
        assert dst.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM pose_observation_sequences").fetchone()[0] == 1
        # trial is included (detection_run has trial_id)
        assert dst.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 0

        src.close(); dst.close()

    def test_full_scope(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        dst_path = tmp_path / "dst.db"
        dst = create_session(dst_path)
        src = open_source_readonly(src_path)

        anchor = AnchorSpec(tracking_run_ids=[ids["tracking_run"]])
        result = export_trials(src, dst, anchor, scope=ExportScope.FULL)

        dst.commit()
        assert result.success
        assert dst.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM tracker_configs").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0] == 1

        src.close(); dst.close()

    def test_empty_anchor_exports_all(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        _make_full_session(src_path)
        dst_path = tmp_path / "dst.db"
        dst = create_session(dst_path)
        src = open_source_readonly(src_path)

        result = export_trials(src, dst, AnchorSpec(), scope=ExportScope.FULL)

        dst.commit()
        assert result.success
        assert dst.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
        assert dst.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 1

        src.close(); dst.close()

    def test_skip_tables(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        # Inject a row into pose_observation_edits so we can verify it's skipped
        conn = sqlite3.connect(src_path)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO pose_observation_edits (id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask)"
            " VALUES (?,?,?,?,?,?)",
            (generate_id(), ids["sequence"], ids["camera_inst"], 10,
             b"\x00" * 16, b"\x01"),
        )
        conn.commit(); conn.close()

        dst_path = tmp_path / "dst.db"
        dst = create_session(dst_path)
        src = open_source_readonly(src_path)

        anchor = AnchorSpec(capture_ids=[ids["capture"]])
        result = export_trials(
            src, dst, anchor,
            scope=ExportScope.DETECTION_ONLY,
            skip_tables={"pose_observation_edits"},
        )
        dst.commit()

        assert result.success
        assert dst.execute("SELECT COUNT(*) FROM pose_observation_edits").fetchone()[0] == 0
        # Other tables still present
        assert dst.execute("SELECT COUNT(*) FROM pose_observation_sequences").fetchone()[0] == 1

        src.close(); dst.close()

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        src = open_source_readonly(src_path)

        anchor = AnchorSpec(capture_ids=[ids["capture"]])
        result = export_trials(src, None, anchor, scope=ExportScope.FULL, dry_run=True)

        assert result.success
        assert result.total_rows > 0  # something would be copied
        src.close()

    def test_idempotent_import(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        dst_path = tmp_path / "dst.db"
        dst = create_session(dst_path)
        src = open_source_readonly(src_path)

        anchor = AnchorSpec(capture_ids=[ids["capture"]])
        export_trials(src, dst, anchor, scope=ExportScope.FULL)
        dst.commit()

        # Second call should not error (INSERT OR IGNORE)
        src2 = open_source_readonly(src_path)
        result2 = export_trials(src2, dst, anchor, scope=ExportScope.FULL)
        dst.commit()
        assert result2.success

        src.close(); src2.close(); dst.close()


# ---------------------------------------------------------------------------
# Library: import_trials
# ---------------------------------------------------------------------------


class TestImportTrials:
    def test_import_from_export(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        ids = _make_full_session(src_path)
        export_path = tmp_path / "export.db"

        # Export first
        src = open_source_readonly(src_path)
        dst_export = create_session(export_path)
        export_trials(src, dst_export, AnchorSpec(), scope=ExportScope.FULL)
        dst_export.commit(); dst_export.close(); src.close()

        # Import into a fresh session
        target_path = tmp_path / "target.db"
        target = create_session(target_path)
        export_src = open_source_readonly(export_path)

        result = import_trials(export_src, target)
        assert result.success
        assert target.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        assert target.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 1

        export_src.close(); target.close()

    def test_dry_run_import(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.db"
        _make_full_session(src_path)
        target_path = tmp_path / "target.db"
        target = create_session(target_path)
        src = open_source_readonly(src_path)

        result = import_trials(src, target, dry_run=True)
        assert result.success
        assert result.total_rows > 0
        # Nothing was written
        assert target.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 0

        src.close(); target.close()


# ---------------------------------------------------------------------------
# CLI: trial list
# ---------------------------------------------------------------------------


class TestTrialList:
    def test_list_trials(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "session.db"
        ids = _make_full_session(db_path)

        result = cli_runner.invoke(main, ["--session", str(db_path), "trial", "list"])
        assert result.exit_code == 0, result.output
        assert ids["trial"][:8] in result.output
        assert "take-1" in result.output

    def test_list_falls_back_to_captures(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "session.db"
        conn = create_session(db_path)
        sess_id = generate_id()
        cap_id = generate_id()
        conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES (?,?)", (sess_id, "2026-01-01"))
        conn.execute(
            "INSERT INTO captures (id, session_id, capture_number, label) VALUES (?,?,?,?)",
            (cap_id, sess_id, 1, "cap-only"),
        )
        conn.commit(); conn.close()

        result = cli_runner.invoke(main, ["--session", str(db_path), "trial", "list"])
        assert result.exit_code == 0, result.output
        assert "cap-only" in result.output

    def test_list_json_mode(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "session.db"
        _make_full_session(db_path)

        result = cli_runner.invoke(
            main, ["--session", str(db_path), "--json", "trial", "list"]
        )
        assert result.exit_code == 0, result.output
        obj = json.loads(result.output.strip())
        assert "id" in obj
        assert "capture_label" in obj

    def test_requires_session(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        registry = tmp_path / "reg.db"
        from posetrak.db.db import create_registry
        create_registry(registry).close()
        result = cli_runner.invoke(main, ["--registry", str(registry), "trial", "list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: export
# ---------------------------------------------------------------------------


class TestExportCommand:
    def test_export_capture(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        out = tmp_path / "out.db"

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--capture", ids["capture"],
            "--scope", "capture-only",
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

        conn = sqlite3.connect(out)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0
        conn.close()

    def test_export_capture_prefix(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Abbreviated capture ID (8-char prefix) should be accepted."""
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        out = tmp_path / "out.db"

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--capture", ids["capture"][:8],   # prefix only
            "--scope", "capture-only",
        ])
        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(out)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        conn.close()

    def test_export_trial_detection_scope(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        out = tmp_path / "out.db"

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--trial", ids["trial"],
            "--scope", "detection-only",
        ])
        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(out)
        assert conn.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 0
        conn.close()

    def test_export_full_scope(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        out = tmp_path / "out.db"

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--tracking-run", ids["tracking_run"],
        ])
        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(out)
        assert conn.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 1
        conn.close()

    def test_export_dry_run_no_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        out = tmp_path / "out.db"

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--capture", ids["capture"],
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert not out.exists()
        assert "dry-run" in result.output.lower() or "rows" in result.output

    def test_export_fails_if_output_exists(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        out = tmp_path / "out.db"
        out.write_text("")

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--capture", ids["capture"],
        ])
        assert result.exit_code != 0

    def test_export_skip_tables(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        out = tmp_path / "out.db"

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--capture", ids["capture"],
            "--scope", "detection-only",
            "--skip-tables", "pose_observation_edits",
        ])
        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(out)
        assert conn.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0] == 1
        conn.close()

    def test_export_all_when_no_anchor(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_full_session(src)
        out = tmp_path / "out.db"

        result = cli_runner.invoke(main, [
            "--session", str(src),
            "export", str(out),
            "--scope", "full",
        ])
        assert result.exit_code == 0, result.output
        conn = sqlite3.connect(out)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        conn.close()

    def test_export_requires_session(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        registry = tmp_path / "reg.db"
        from posetrak.db.db import create_registry
        create_registry(registry).close()
        result = cli_runner.invoke(main, [
            "--registry", str(registry),
            "export", str(tmp_path / "out.db"),
        ])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: import
# ---------------------------------------------------------------------------


class TestImportCommand:
    def _export_full(self, src_path: Path, export_path: Path) -> None:
        src = open_source_readonly(src_path)
        dst = create_session(export_path)
        export_trials(src, dst, AnchorSpec(), scope=ExportScope.FULL)
        dst.commit(); dst.close(); src.close()

    def test_import_into_session(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_full_session(src)
        export_path = tmp_path / "export.db"
        self._export_full(src, export_path)

        target = tmp_path / "target.db"
        create_session(target).close()

        result = cli_runner.invoke(main, [
            "--session", str(target),
            "import", str(export_path),
        ])
        assert result.exit_code == 0, result.output

        conn = open_session(target)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM tracking_runs").fetchone()[0] == 1
        conn.close()

    def test_import_capture_prefix(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Abbreviated capture ID prefix should be resolved from the source DB."""
        src = tmp_path / "src.db"
        ids = _make_full_session(src)
        export_path = tmp_path / "export.db"
        self._export_full(src, export_path)

        target = tmp_path / "target.db"
        create_session(target).close()

        result = cli_runner.invoke(main, [
            "--session", str(target),
            "import", str(export_path),
            "--capture", ids["capture"][:8],
        ])
        assert result.exit_code == 0, result.output
        conn = open_session(target)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        conn.close()

    def test_import_dry_run(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_full_session(src)
        export_path = tmp_path / "export.db"
        self._export_full(src, export_path)

        target = tmp_path / "target.db"
        create_session(target).close()

        result = cli_runner.invoke(main, [
            "--session", str(target),
            "import", str(export_path),
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(target)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 0
        conn.close()

    def test_import_idempotent(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_full_session(src)
        export_path = tmp_path / "export.db"
        self._export_full(src, export_path)

        target = tmp_path / "target.db"
        create_session(target).close()

        for _ in range(2):
            result = cli_runner.invoke(main, [
                "--session", str(target),
                "import", str(export_path),
            ])
            assert result.exit_code == 0, result.output

        conn = open_session(target)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        conn.close()

    def test_import_skip_tables(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_full_session(src)
        export_path = tmp_path / "export.db"
        self._export_full(src, export_path)

        target = tmp_path / "target.db"
        create_session(target).close()

        result = cli_runner.invoke(main, [
            "--session", str(target),
            "import", str(export_path),
            "--skip-tables", "pose_observation_edits",
        ])
        assert result.exit_code == 0, result.output

    def test_import_requires_session(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_full_session(src)
        registry = tmp_path / "reg.db"
        from posetrak.db.db import create_registry
        create_registry(registry).close()

        result = cli_runner.invoke(main, [
            "--registry", str(registry),
            "import", str(src),
        ])
        assert result.exit_code != 0
