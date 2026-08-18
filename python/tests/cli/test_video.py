# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for video list / locate / relocate commands and updated capture show."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import create_session, generate_id


# ---------------------------------------------------------------------------
# Fixture: seeded session with two captures, two videos each
# ---------------------------------------------------------------------------


def _make_session(path: Path) -> dict:
    """Session with 2 captures × 2 cameras (4 videos total)."""
    conn = create_session(path)
    ids: dict = {}

    for key in ["cam_model", "cam_a", "cam_b", "session", "ext_cal", "cap1", "cap2"]:
        ids[key] = generate_id()
    for key in ["vid_a1", "vid_b1", "vid_a2", "vid_b2"]:
        ids[key] = generate_id()

    conn.execute("PRAGMA foreign_keys = OFF")

    conn.execute("INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?,?,?)",
                 (ids["cam_model"], "Co", "Cam"))
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?,?,?)",
                 (ids["cam_a"], ids["cam_model"], "cam-a"))
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?,?,?)",
                 (ids["cam_b"], ids["cam_model"], "cam-b"))

    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES (?,?)",
                 (ids["session"], "2026-01-01"))
    conn.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at) VALUES (?,?,?)",
        (ids["ext_cal"], ids["session"], "2026-01-01"),
    )

    for cap_key, cap_num, cap_label in [("cap1", 1, "morning"), ("cap2", 2, "afternoon")]:
        conn.execute(
            "INSERT INTO captures (id, session_id, extrinsic_calibration_id, capture_number, label) "
            "VALUES (?,?,?,?,?)",
            (ids[cap_key], ids["session"], ids["ext_cal"], cap_num, cap_label),
        )

    # Two videos per capture (cam-a and cam-b)
    for vid_key, shot_key, cam_key, fp in [
        ("vid_a1", "cap1", "cam_a", "/old/mount/cap1/cam_a.mp4"),
        ("vid_b1", "cap1", "cam_b", "/old/mount/cap1/cam_b.mp4"),
        ("vid_a2", "cap2", "cam_a", "/old/mount/cap2/cam_a.mp4"),
        ("vid_b2", "cap2", "cam_b", "/old/mount/cap2/cam_b.mp4"),
    ]:
        conn.execute(
            "INSERT INTO capture_videos "
            "(id, shot_id, camera_instance_id, file_path, first_video_frame, last_video_frame, actual_fps) "
            "VALUES (?,?,?,?,?,?,?)",
            (ids[vid_key], ids[shot_key], ids[cam_key], fp, 0, 900, 30.0),
        )

    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.close()
    return ids


# ---------------------------------------------------------------------------
# video list
# ---------------------------------------------------------------------------


class TestVideoList:
    def test_lists_all_videos(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, ["--session", str(db), "video", "list"])
        assert result.exit_code == 0, result.output
        assert "cam_a.mp4" in result.output
        assert "cam_b.mp4" in result.output
        # 4 videos total
        assert result.output.count(".mp4") == 4

    def test_filter_by_capture(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "list",
            "--capture", ids["cap1"],
        ])
        assert result.exit_code == 0, result.output
        assert "cap1" in result.output
        assert "cap2" not in result.output

    def test_filter_by_capture_prefix(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "list",
            "--capture", ids["cap1"][:8],
        ])
        assert result.exit_code == 0, result.output
        assert result.output.count(".mp4") == 2

    def test_exists_column_no(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        _make_session(db)

        result = cli_runner.invoke(main, ["--session", str(db), "video", "list"])
        assert result.exit_code == 0, result.output
        assert "NO" in result.output  # paths don't exist on disk

    def test_exists_column_yes(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        # Create a real file and update the path
        real_file = tmp_path / "real.mp4"
        real_file.write_bytes(b"")
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db)
        conn.execute("UPDATE capture_videos SET file_path = ? WHERE id = ?",
                     (str(real_file), ids["vid_a1"]))
        conn.commit(); conn.close()

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "list",
            "--capture", ids["cap1"],
        ])
        assert result.exit_code == 0, result.output
        assert "yes" in result.output

    def test_json_mode(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "--json", "video", "list",
            "--capture", ids["cap1"],
        ])
        assert result.exit_code == 0, result.output
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 2
        obj = json.loads(lines[0])
        assert "file_path" in obj
        assert "camera" in obj

    def test_requires_session(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        from posetrak.db.db import create_registry
        reg = tmp_path / "r.db"
        create_registry(reg).close()
        result = cli_runner.invoke(main, ["--registry", str(reg), "video", "list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# video locate
# ---------------------------------------------------------------------------


class TestVideoLocate:
    def test_updates_path(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "locate",
            ids["vid_a1"], "/new/path/cam_a.mp4",
        ])
        assert result.exit_code == 0, result.output
        assert "/new/path/cam_a.mp4" in result.output
        assert "/old/mount" in result.output  # old path shown

        # Verify DB was updated
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db)
        row = conn.execute("SELECT file_path FROM capture_videos WHERE id = ?",
                           (ids["vid_a1"],)).fetchone()
        conn.close()
        assert row[0] == "/new/path/cam_a.mp4"

    def test_prefix_accepted(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "locate",
            ids["vid_a1"][:8], "/new/path/cam_a.mp4",
        ])
        assert result.exit_code == 0, result.output

    def test_no_change_when_same(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "locate",
            ids["vid_a1"], "/old/mount/cap1/cam_a.mp4",
        ])
        assert result.exit_code == 0, result.output
        assert "unchanged" in result.output.lower()

    def test_bad_id_fails(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "locate",
            "deadbeef", "/new/path.mp4",
        ])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# video relocate
# ---------------------------------------------------------------------------


class TestVideoRelocate:
    def test_bulk_replace(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "relocate",
            "--from", "/old/mount",
            "--to", "/new/mount",
        ])
        assert result.exit_code == 0, result.output
        assert "4" in result.output  # 4 videos updated

        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db)
        rows = conn.execute("SELECT file_path FROM capture_videos").fetchall()
        conn.close()
        for (fp,) in rows:
            assert fp.startswith("/new/mount"), fp

    def test_dry_run_does_not_write(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "relocate",
            "--from", "/old/mount",
            "--to", "/new/mount",
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "would" in result.output.lower() or "old/mount" in result.output

        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db)
        rows = conn.execute("SELECT file_path FROM capture_videos").fetchall()
        conn.close()
        for (fp,) in rows:
            assert fp.startswith("/old/mount"), fp  # unchanged

    def test_no_match(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "relocate",
            "--from", "/does/not/exist",
            "--to", "/new",
        ])
        assert result.exit_code == 0, result.output
        assert "No video" in result.output

    def test_filter_by_capture(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "relocate",
            "--from", "/old/mount",
            "--to", "/new/mount",
            "--capture", ids["cap1"],
        ])
        assert result.exit_code == 0, result.output
        assert "2" in result.output

        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db)
        unchanged = conn.execute(
            "SELECT COUNT(*) FROM capture_videos WHERE file_path LIKE '/old/%' AND shot_id = ?",
            (ids["cap2"],)
        ).fetchone()[0]
        conn.close()
        assert unchanged == 2  # cap2 untouched

    def test_capture_prefix_accepted(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "relocate",
            "--from", "/old/mount",
            "--to", "/new/mount",
            "--capture", ids["cap1"][:8],
        ])
        assert result.exit_code == 0, result.output

    def test_trailing_slash_normalised(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Trailing slash on --from or --to must not cause path concatenation."""
        import sqlite3 as _sqlite3
        db = tmp_path / "s.db"
        _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "video", "relocate",
            "--from", "/old/mount/",   # trailing slash
            "--to", "/new/mount",      # no trailing slash
        ])
        assert result.exit_code == 0, result.output

        conn = _sqlite3.connect(db)
        rows = conn.execute("SELECT file_path FROM capture_videos").fetchall()
        conn.close()
        for (fp,) in rows:
            # separator must be preserved — no concatenation like /new/mountcap1/...
            assert fp.startswith("/new/mount/"), repr(fp)


# ---------------------------------------------------------------------------
# capture show — videos section
# ---------------------------------------------------------------------------


class TestCaptureShowVideos:
    def test_shows_videos(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "capture", "show", ids["cap1"],
        ])
        assert result.exit_code == 0, result.output
        assert "Videos" in result.output
        assert "cam_a.mp4" in result.output
        assert "cam_b.mp4" in result.output

    def test_json_includes_videos(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db = tmp_path / "s.db"
        ids = _make_session(db)

        result = cli_runner.invoke(main, [
            "--session", str(db), "--json", "capture", "show", ids["cap1"],
        ])
        assert result.exit_code == 0, result.output
        obj = json.loads(result.output.strip())
        assert "videos" in obj
        assert len(obj["videos"]) == 2
        file_paths = {v["file_path"] for v in obj["videos"]}
        assert "/old/mount/cap1/cam_a.mp4" in file_paths
