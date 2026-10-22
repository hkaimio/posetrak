# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the posetrak track CLI commands."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import generate_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(args: list[str], session_path: Path | None = None) -> "click.testing.Result":
    runner = CliRunner()
    base_args: list[str] = []
    if session_path is not None:
        base_args = ["--session", str(session_path)]
    return runner.invoke(main, base_args + args, catch_exceptions=False)


def _seed_run(session_db_path: Path) -> str:
    conn = sqlite3.connect(str(session_db_path))
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = generate_id()
    conn.execute(
        "INSERT INTO tracking_runs "
        "(id, observation_sequence_id, tracker_config_id, skeleton_id, "
        " extrinsic_calibration_id, sync_config_id, ran_at, posetrak_version, "
        " active_camera_ids, marker_names) "
        "VALUES (?,?,?,?,?,?,?,'0.1.0','[]','[]')",
        (run_id, generate_id(), generate_id(), generate_id(),
         generate_id(), generate_id(), now),
    )
    conn.commit()
    conn.close()
    return run_id


# ---------------------------------------------------------------------------
# track list
# ---------------------------------------------------------------------------


class TestTrackList:
    def test_empty(self, seeded_session_db_path: Path) -> None:
        result = _invoke(["track", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""

    def test_lists_run(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["track", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert run_id[:8] in result.output

    def test_filters_by_sequence_prefix(self, seeded_session_db_path: Path, capture_id: str, sync_id: str) -> None:
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.execute(
            "INSERT INTO pose_observation_sequences (id, shot_id, sync_config_id, time_start_s, time_end_s) "
            "VALUES ('seq-abc', ?, ?, 0, 1)", (capture_id, sync_id),
        )
        conn.commit()
        conn.close()
        other = _seed_run(seeded_session_db_path)
        conn = sqlite3.connect(str(seeded_session_db_path))
        mine = generate_id()
        conn.execute(
            "INSERT INTO tracking_runs (id, observation_sequence_id, tracker_config_id, skeleton_id, "
            " extrinsic_calibration_id, sync_config_id, ran_at, posetrak_version, active_camera_ids, marker_names) "
            "VALUES (?,?,?,?,?,?,?,'0.1.0','[]','[]')",
            (mine, "seq-abc", generate_id(), generate_id(), generate_id(), generate_id(),
             dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

        result = _invoke(["track", "list", "--sequence", "seq-a"], seeded_session_db_path)

        assert result.exit_code == 0, result.output
        assert mine[:8] in result.output and other[:8] not in result.output

    def test_json_mode(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["--json", "track", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["id"] == run_id
        assert "skeleton_id" in obj
        assert "ran_at" in obj
        assert "posetrak_version" in obj

    def test_no_session(self) -> None:
        result = _invoke(["track", "list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# track show
# ---------------------------------------------------------------------------


class TestTrackShow:
    def test_show_by_prefix(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["track", "show", run_id[:8]], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert run_id in result.output
        assert "0.1.0" in result.output

    def test_show_json(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["--json", "track", "show", run_id], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        obj = json.loads(result.output.strip())
        assert obj["id"] == run_id

    def test_unknown_id(self, seeded_session_db_path: Path) -> None:
        result = _invoke(["track", "show", "nonexistent"], seeded_session_db_path)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# track export
# ---------------------------------------------------------------------------


class TestTrackExport:
    @patch("posetrak.cli.track.export_bvh")
    def test_export_bvh(self, mock_export, seeded_session_db_path: Path, tmp_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.bvh")
        result = _invoke(
            ["track", "export", "bvh", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs.args[0] == output
        assert call_kwargs.kwargs["run_id"] == run_id
        assert call_kwargs.kwargs["session_db"] == str(seeded_session_db_path)

    @patch("posetrak.cli.track.export_gltf")
    def test_export_gltf(self, mock_export, seeded_session_db_path: Path, tmp_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.glb")
        result = _invoke(
            ["track", "export", "gltf", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs.args[0] == output
        assert call_kwargs.kwargs["run_id"] == run_id
        assert call_kwargs.kwargs["session_db"] == str(seeded_session_db_path)

    @patch("posetrak.cli.track.export_usd")
    def test_export_usd(self, mock_export, seeded_session_db_path: Path, tmp_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.usda")
        result = _invoke(
            ["track", "export", "usd", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs.args[0] == output
        assert call_kwargs.kwargs["run_id"] == run_id
        assert call_kwargs.kwargs["session_db"] == str(seeded_session_db_path)

    @patch("posetrak.cli.track.export_usd", side_effect=ImportError("usd-core not installed"))
    def test_export_usd_missing_package(
        self, mock_export, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.usda")
        result = _invoke(
            ["track", "export", "usd", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code != 0

    def test_export_no_session(self, tmp_path: Path) -> None:
        result = _invoke(["track", "export", "bvh", "abc123", str(tmp_path / "out.bvh")])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# track run
# ---------------------------------------------------------------------------


def _seed_run_prerequisites(db_path: Path) -> tuple[str, str]:
    """Seed a skeleton and a pose_observation_sequence; return (seq_id, skel_id)."""
    import datetime as dt
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    skel_id = generate_id()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO skeletons (id, name, yaml_content, created_at)"
        " VALUES (?, 'TestSkel', '{}', ?)",
        (skel_id, now),
    )

    # Re-use the capture and sync_config seeded by the fixture
    capture_row = conn.execute("SELECT id FROM captures LIMIT 1").fetchone()
    sync_row = conn.execute("SELECT id FROM sync_configs LIMIT 1").fetchone()

    seq_id = generate_id()
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s)"
        " VALUES (?, ?, ?, 0.0, 10.0)",
        (seq_id, capture_row["id"], sync_row["id"]),
    )
    conn.commit()
    conn.close()
    return seq_id, skel_id


class TestTrackRun:
    @patch("posetrak.cli.track.run_tracker")
    def test_run_names_the_missing_pose_model(self, mock_run, seeded_session_db_path: Path, tmp_path: Path) -> None:
        from posetrak.detection.pose_models import SOMA77
        from posetrak.skeleton.compose import compose_default

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.execute("UPDATE skeletons SET yaml_content = ? WHERE id = ?",
                     (compose_default("male", ["coco133", "hand21"]), skel_id))
        conn.executemany(
            "INSERT INTO pose_sequence_keypoints (sequence_id, obs_source, keypoint_idx, name, source)"
            " VALUES (?, 'body', ?, ?, ?)", [(seq_id, *row) for row in SOMA77.manifest_rows()])
        conn.commit()
        conn.close()
        result = _invoke(["track", "run", "--sequence", seq_id, "--skeleton", skel_id,
                          "--output-dir", str(tmp_path / "out")], seeded_session_db_path)
        assert result.exit_code != 0
        assert "soma77" in result.output and "no marker bound" in result.output
        mock_run.assert_not_called()

    @patch("posetrak.cli.track.run_tracker")
    def test_run_basic(self, mock_run, seeded_session_db_path: Path, tmp_path: Path) -> None:
        from posetrak.tracker.runner import TrackerResult
        run_id = generate_id()
        mock_run.return_value = TrackerResult(exit_code=0, run_id=run_id)

        fake_binary = tmp_path / "fake-binary"
        fake_binary.touch()

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            [
                "track", "run",
                "--sequence", seq_id[:8],
                "--skeleton", skel_id[:8],
                "--output-dir", str(tmp_path / "out"),
                "--binary", str(fake_binary),
            ],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        # sequence_id and skeleton_id are positional args 1 and 2
        assert call_kwargs.args[1] == seq_id
        assert call_kwargs.args[2] == skel_id

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_creates_config_row(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import TrackerResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = TrackerResult(exit_code=0, run_id=generate_id())

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            [
                "track", "run",
                "--sequence", seq_id,
                "--skeleton", skel_id,
                "--output-dir", str(tmp_path / "out"),
                "--calib-noise-std", "42.0",
            ],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        cfg = conn.execute(
            "SELECT * FROM tracker_configs WHERE name = 'cli-run' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert cfg is not None
        assert cfg["measurement_noise_std"] == pytest.approx(42.0)

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_with_the_windowed_choice_records_it_and_hands_the_model_to_the_binary(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import TrackerResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = TrackerResult(exit_code=0, run_id=generate_id())

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.execute("UPDATE pose_observation_sequences SET pose_model = 'soma77-v1' WHERE id = ?", (seq_id,))
        conn.commit()
        conn.close()
        result = _invoke(
            ["track", "run", "--sequence", seq_id, "--skeleton", skel_id, "--output-dir", str(tmp_path / "out"),
             "--candidates", "--kp-choice", "window", "--kp-support-window", "1.5", "--kp-rounds", "2"],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        cfg = conn.execute("SELECT * FROM tracker_configs WHERE name = 'cli-run' ORDER BY created_at DESC LIMIT 1").fetchone()
        conn.close()
        assert (cfg["kp_candidates_enabled"], cfg["kp_candidate_choice"]) == (1, "window")
        assert (cfg["kp_choice_support_window_s"], cfg["kp_choice_rounds"]) == (1.5, 2)
        kwargs = mock_run.call_args.kwargs
        assert kwargs["kp_choice_model_path"] == tmp_path / "out" / "kp_choice_model.csv"
        assert kwargs["kp_choice_model_path"].exists()
        assert kwargs["kp_confidence_scale"] == 5.0

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_refuses_a_mask_option_of_the_windowed_choice_without_masks(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            ["track", "run", "--sequence", seq_id, "--skeleton", skel_id, "--output-dir", str(tmp_path / "out"),
             "--candidates", "--kp-choice", "window", "--kp-strict-filter"],
            seeded_session_db_path,
        )
        assert result.exit_code != 0
        assert "--seg-run" in result.output
        mock_run.assert_not_called()

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_prints_run_id(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import TrackerResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        expected_run_id = generate_id()
        mock_run.return_value = TrackerResult(exit_code=0, run_id=expected_run_id)

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            ["track", "run", "--sequence", seq_id, "--skeleton", skel_id,
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        assert expected_run_id in result.output

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_nonzero_exit(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import TrackerResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = TrackerResult(exit_code=1, run_id=None)

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            ["track", "run", "--sequence", seq_id, "--skeleton", skel_id,
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code == 1

    def test_run_no_session(self) -> None:
        result = _invoke(["track", "run", "--sequence", "abc", "--skeleton", "def"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# track run-persons
# ---------------------------------------------------------------------------


def _seed_trial_with_person(
    db_path: Path,
    *,
    person_name: str = "Alice",
    with_default_skeleton: bool = True,
    n_detection_runs: int = 1,
) -> dict:
    """Seed a trial, a capture_persons row, and *n_detection_runs* detection
    runs each with a sequence + sequence_persons row observing that person.

    Returns a dict with trial_id, skel_id, person_id, and seq_ids (one per
    detection run, in creation order).
    """
    import datetime as dt
    from posetrak.db.manage_person import create_person

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    capture_row = conn.execute("SELECT id FROM captures LIMIT 1").fetchone()
    sync_row = conn.execute("SELECT id FROM sync_configs LIMIT 1").fetchone()
    capture_id = capture_row["id"]
    sync_id = sync_row["id"]

    skel_id = generate_id()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO skeletons (id, name, yaml_content, created_at)"
        " VALUES (?, 'TestSkel', '{}', ?)",
        (skel_id, now),
    )

    trial_id = generate_id()
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s)"
        " VALUES (?, ?, 'take 1', 0.0, 10.0)",
        (trial_id, capture_id),
    )
    conn.commit()

    person_id = create_person(
        conn, capture_id, person_name,
        default_skeleton_id=skel_id if with_default_skeleton else None,
    )

    seq_ids = []
    for i in range(n_detection_runs):
        dr_id = generate_id()
        conn.execute(
            "INSERT INTO detection_runs"
            " (id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s,"
            "  detector_model, pose_model, created_at)"
            " VALUES (?, ?, ?, ?, 0.0, 10.0, 'yolo', 'rtmpose', ?)",
            (dr_id, capture_id, sync_id, trial_id, now),
        )
        seq_id = generate_id()
        conn.execute(
            "INSERT INTO pose_observation_sequences"
            " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id)"
            " VALUES (?, ?, ?, 0.0, 10.0, ?)",
            (seq_id, capture_id, sync_id, dr_id),
        )
        conn.execute(
            "INSERT INTO sequence_persons (sequence_id, person_id, person_name, capture_person_id)"
            " VALUES (?, 0, ?, ?)",
            (seq_id, person_name, person_id),
        )
        seq_ids.append(seq_id)

    conn.commit()
    conn.close()
    return {
        "trial_id": trial_id, "skel_id": skel_id, "person_id": person_id, "seq_ids": seq_ids,
    }


class TestResolveTrialPersons:
    """Unit tests for the pure resolution function, no CLI involved."""

    def test_resolves_single_person(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_persons

        info = _seed_trial_with_person(seeded_session_db_path)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row

        resolved = resolve_trial_persons(conn, info["trial_id"], ["Alice"])
        assert len(resolved) == 1
        assert resolved[0].sequence_id == info["seq_ids"][0]
        assert resolved[0].skeleton_id == info["skel_id"]
        assert resolved[0].time_start_s == pytest.approx(0.0)
        assert resolved[0].time_end_s == pytest.approx(10.0)

    def test_unknown_trial_raises(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_persons

        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        with pytest.raises(ValueError, match="trial not found"):
            resolve_trial_persons(conn, "does-not-exist", ["Alice"])

    def test_undefined_person_raises(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_persons

        info = _seed_trial_with_person(seeded_session_db_path)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        with pytest.raises(ValueError, match="No person named 'Bob'"):
            resolve_trial_persons(conn, info["trial_id"], ["Bob"])

    def test_missing_default_skeleton_raises(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_persons

        info = _seed_trial_with_person(seeded_session_db_path, with_default_skeleton=False)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        with pytest.raises(ValueError, match="no default skeleton"):
            resolve_trial_persons(conn, info["trial_id"], ["Alice"])

    def test_ambiguous_detection_runs_raises(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_persons

        info = _seed_trial_with_person(seeded_session_db_path, n_detection_runs=2)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        with pytest.raises(ValueError, match="Ambiguous"):
            resolve_trial_persons(conn, info["trial_id"], ["Alice"])

    def test_no_observations_raises(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_persons
        from posetrak.db.manage_person import create_person

        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        capture_id = conn.execute("SELECT id FROM captures LIMIT 1").fetchone()["id"]
        trial_id = generate_id()
        conn.execute(
            "INSERT INTO trials (id, capture_id, name) VALUES (?, ?, 'take 1')",
            (trial_id, capture_id),
        )
        create_person(conn, capture_id, "Alice", default_skeleton_id=generate_id())
        conn.commit()

        with pytest.raises(ValueError, match="No detection-run observations"):
            resolve_trial_persons(conn, trial_id, ["Alice"])

    def test_resolves_multiple_persons(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_persons

        info_a = _seed_trial_with_person(seeded_session_db_path, person_name="Alice")
        # Reuse the same trial for a second person.
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        from posetrak.db.manage_person import create_person
        capture_id = conn.execute("SELECT id FROM captures LIMIT 1").fetchone()["id"]
        sync_id = conn.execute("SELECT id FROM sync_configs LIMIT 1").fetchone()["id"]
        skel_id = generate_id()
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO skeletons (id, name, yaml_content, created_at)"
            " VALUES (?, 'TestSkel2', '{}', ?)",
            (skel_id, now),
        )
        person_id = create_person(conn, capture_id, "Bob", default_skeleton_id=skel_id)
        dr_id = generate_id()
        conn.execute(
            "INSERT INTO detection_runs"
            " (id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s,"
            "  detector_model, pose_model, created_at)"
            " VALUES (?, ?, ?, ?, 0.0, 10.0, 'yolo', 'rtmpose', ?)",
            (dr_id, capture_id, sync_id, info_a["trial_id"], now),
        )
        seq_id = generate_id()
        conn.execute(
            "INSERT INTO pose_observation_sequences"
            " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id)"
            " VALUES (?, ?, ?, 0.0, 10.0, ?)",
            (seq_id, capture_id, sync_id, dr_id),
        )
        conn.execute(
            "INSERT INTO sequence_persons (sequence_id, person_id, person_name, capture_person_id)"
            " VALUES (?, 0, 'Bob', ?)",
            (seq_id, person_id),
        )
        conn.commit()

        resolved = resolve_trial_persons(conn, info_a["trial_id"], ["Alice", "Bob"])
        assert [r.name for r in resolved] == ["Alice", "Bob"]
        assert resolved[1].sequence_id == seq_id


class TestTrackRunPersons:
    @patch("posetrak.cli.track.run_multi_person_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_persons_basic(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import MultiPersonResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        run_id = generate_id()
        mock_run.return_value = MultiPersonResult(exit_code=0, run_ids=[run_id])

        info = _seed_trial_with_person(seeded_session_db_path)
        result = _invoke(
            [
                "track", "run-persons",
                "--trial", info["trial_id"],
                "--persons", "Alice",
                "--output-dir", str(tmp_path / "out"),
            ],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        person_specs = mock_run.call_args.args[1]
        assert len(person_specs) == 1
        assert person_specs[0].sequence_id == info["seq_ids"][0]
        assert person_specs[0].skeleton_id == info["skel_id"]
        assert f"Alice: {run_id}" in result.output

    @patch("posetrak.cli.track.run_multi_person_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_persons_uses_trial_default_config_when_none_given(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import MultiPersonResult
        from posetrak.db.manage_config import BASELINE_CONFIG_ID
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = MultiPersonResult(exit_code=0, run_ids=[generate_id()])

        info = _seed_trial_with_person(seeded_session_db_path)
        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--persons", "Alice",
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        person_specs = mock_run.call_args.args[1]
        assert person_specs[0].config_id == BASELINE_CONFIG_ID

    @patch("posetrak.cli.track.run_multi_person_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_persons_multiple_names(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import MultiPersonResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = MultiPersonResult(
            exit_code=0, run_ids=[generate_id(), generate_id()]
        )

        info = _seed_trial_with_person(seeded_session_db_path, person_name="Alice")
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        from posetrak.db.manage_person import create_person
        import datetime as dt
        capture_id = conn.execute("SELECT id FROM captures LIMIT 1").fetchone()["id"]
        sync_id = conn.execute("SELECT id FROM sync_configs LIMIT 1").fetchone()["id"]
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        skel_id = generate_id()
        conn.execute(
            "INSERT OR IGNORE INTO skeletons (id, name, yaml_content, created_at)"
            " VALUES (?, 'TestSkel2', '{}', ?)",
            (skel_id, now),
        )
        person_id = create_person(conn, capture_id, "Bob", default_skeleton_id=skel_id)
        dr_id = generate_id()
        conn.execute(
            "INSERT INTO detection_runs"
            " (id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s,"
            "  detector_model, pose_model, created_at)"
            " VALUES (?, ?, ?, ?, 0.0, 10.0, 'yolo', 'rtmpose', ?)",
            (dr_id, capture_id, sync_id, info["trial_id"], now),
        )
        seq_id = generate_id()
        conn.execute(
            "INSERT INTO pose_observation_sequences"
            " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id)"
            " VALUES (?, ?, ?, 0.0, 10.0, ?)",
            (seq_id, capture_id, sync_id, dr_id),
        )
        conn.execute(
            "INSERT INTO sequence_persons (sequence_id, person_id, person_name, capture_person_id)"
            " VALUES (?, 0, 'Bob', ?)",
            (seq_id, person_id),
        )
        conn.commit()
        conn.close()

        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--persons", "Alice,Bob",
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        person_specs = mock_run.call_args.args[1]
        assert len(person_specs) == 2

    def test_run_persons_unknown_name_fails(self, seeded_session_db_path: Path, tmp_path: Path) -> None:
        info = _seed_trial_with_person(seeded_session_db_path)
        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--persons", "Bob",
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code != 0
        assert "No person named 'Bob'" in result.output

    def test_run_persons_missing_default_skeleton_fails(
        self, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        info = _seed_trial_with_person(seeded_session_db_path, with_default_skeleton=False)
        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--persons", "Alice",
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code != 0
        assert "no default skeleton" in result.output

    def test_run_persons_ambiguous_fails(self, seeded_session_db_path: Path, tmp_path: Path) -> None:
        info = _seed_trial_with_person(seeded_session_db_path, n_detection_runs=2)
        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--persons", "Alice",
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code != 0
        assert "Ambiguous" in result.output

    def test_run_persons_empty_names_fails(self, seeded_session_db_path: Path, tmp_path: Path) -> None:
        info = _seed_trial_with_person(seeded_session_db_path)
        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--persons", " , ",
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code != 0

    def test_run_persons_no_session(self) -> None:
        result = _invoke(["track", "run-persons", "--trial", "abc", "--persons", "Alice"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# track run-persons with capture objects
# ---------------------------------------------------------------------------


def _seed_object_in_trial(
    db_path: Path,
    trial_id: str,
    *,
    name: str = "ball",
    time_range: tuple[float, float] = (2.0, 8.0),
    n_sequences: int = 1,
) -> dict:
    """Add a capture object and *n_sequences* finalised object sequences to an
    existing trial (see _seed_trial_with_person); returns object_id, skel_id,
    seq_ids."""
    import datetime as dt

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    capture_id = conn.execute("SELECT capture_id FROM trials WHERE id = ?", (trial_id,)).fetchone()["capture_id"]
    sync_id = conn.execute("SELECT id FROM sync_configs LIMIT 1").fetchone()["id"]

    skel_id = generate_id()
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) VALUES (?, 'BallSkel', '{}', ?)",
        (skel_id, now),
    )
    object_id = generate_id()
    conn.execute(
        "INSERT INTO capture_objects (id, capture_id, name, marker_body_definition_id, created_at)"
        " VALUES (?, ?, ?, 'body-def', ?)",
        (object_id, capture_id, name, now),
    )
    seq_ids = []
    for _ in range(n_sequences):
        dr_id = generate_id()
        conn.execute(
            "INSERT INTO detection_runs"
            " (id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, detector_model,"
            "  pose_model, created_at, capture_object_id)"
            " VALUES (?, ?, ?, ?, ?, ?, 'aruco:DICT_4X4_50', '', ?, ?)",
            (dr_id, capture_id, sync_id, trial_id, *time_range, now, object_id),
        )
        seq_id = generate_id()
        conn.execute(
            "INSERT INTO pose_observation_sequences"
            " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (seq_id, capture_id, sync_id, *time_range, dr_id),
        )
        seq_ids.append(seq_id)
    conn.commit()
    conn.close()
    return {"object_id": object_id, "skel_id": skel_id, "seq_ids": seq_ids}


class TestResolveTrialObjects:
    def test_resolves_object_sequence_and_explicit_skeleton(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_objects

        info = _seed_trial_with_person(seeded_session_db_path)
        obj = _seed_object_in_trial(seeded_session_db_path, info["trial_id"])
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row

        resolved = resolve_trial_objects(conn, info["trial_id"], [("ball", obj["skel_id"][:8], (1.0, 2.0, 3.0))])

        assert len(resolved) == 1
        assert resolved[0].sequence_id == obj["seq_ids"][0]
        assert resolved[0].skeleton_id == obj["skel_id"]
        assert resolved[0].seed_position == (1.0, 2.0, 3.0)
        assert (resolved[0].time_start_s, resolved[0].time_end_s) == (2.0, 8.0)

    def test_unknown_object_is_reported(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_objects

        info = _seed_trial_with_person(seeded_session_db_path)
        obj = _seed_object_in_trial(seeded_session_db_path, info["trial_id"])
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row

        with pytest.raises(ValueError, match="No object named 'pen'"):
            resolve_trial_objects(conn, info["trial_id"], [("pen", obj["skel_id"], None)])

    def test_two_sequences_for_one_object_are_ambiguous(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_objects

        info = _seed_trial_with_person(seeded_session_db_path)
        obj = _seed_object_in_trial(seeded_session_db_path, info["trial_id"], n_sequences=2)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row

        with pytest.raises(ValueError, match="Ambiguous"):
            resolve_trial_objects(conn, info["trial_id"], [("ball", obj["skel_id"], None)])

    def test_object_without_a_sequence_in_the_trial_is_reported(self, seeded_session_db_path: Path) -> None:
        from posetrak.cli.track import resolve_trial_objects

        info = _seed_trial_with_person(seeded_session_db_path)
        obj = _seed_object_in_trial(seeded_session_db_path, info["trial_id"], n_sequences=0)
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row

        with pytest.raises(ValueError, match="No finalised sequence.*detect it with --trial"):
            resolve_trial_objects(conn, info["trial_id"], [("ball", obj["skel_id"], None)])


class TestRunPersonsWithObjects:
    @patch("posetrak.cli.track.run_multi_person_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_person_and_object_are_tracked_together_over_the_common_range(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import MultiPersonResult

        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = MultiPersonResult(exit_code=0, run_ids=[generate_id(), generate_id()])

        info = _seed_trial_with_person(seeded_session_db_path)   # person sequence 0..10 s
        obj = _seed_object_in_trial(seeded_session_db_path, info["trial_id"], time_range=(2.0, 8.0))
        result = _invoke(
            [
                "track", "run-persons", "--trial", info["trial_id"], "--persons", "Alice",
                "--object", f"ball={obj['skel_id'][:8]}@0.1,-0.9,1.0",
                "--output-dir", str(tmp_path / "out"),
            ],
            seeded_session_db_path,
        )

        assert result.exit_code == 0, result.output
        person_specs = mock_run.call_args.args[1]
        assert [s.sequence_id for s in person_specs] == [info["seq_ids"][0], obj["seq_ids"][0]]
        assert person_specs[1].skeleton_id == obj["skel_id"]
        assert mock_run.call_args.kwargs["start_time"] == 2.0
        assert mock_run.call_args.kwargs["end_time"] == 8.0
        assert [s.seed_position for s in person_specs] == [None, (0.1, -0.9, 1.0)]

    @patch("posetrak.cli.track.run_multi_person_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_explicit_start_and_end_override_the_common_range(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import MultiPersonResult

        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = MultiPersonResult(exit_code=0, run_ids=[generate_id(), generate_id()])

        info = _seed_trial_with_person(seeded_session_db_path)
        obj = _seed_object_in_trial(seeded_session_db_path, info["trial_id"], time_range=(2.0, 8.0))
        result = _invoke(
            [
                "track", "run-persons", "--trial", info["trial_id"], "--persons", "Alice",
                "--object", f"ball={obj['skel_id'][:8]}", "--start-time", "3.0", "--end-time", "4.0",
                "--output-dir", str(tmp_path / "out"),
            ],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        assert mock_run.call_args.kwargs["start_time"] == 3.0
        assert mock_run.call_args.kwargs["end_time"] == 4.0

    def test_no_subjects_is_an_error(self, seeded_session_db_path: Path) -> None:
        info = _seed_trial_with_person(seeded_session_db_path)
        result = _invoke(["track", "run-persons", "--trial", info["trial_id"]], seeded_session_db_path)
        assert result.exit_code != 0
        assert "at least one subject" in result.output

    def test_malformed_objects_option_is_an_error(self, seeded_session_db_path: Path) -> None:
        info = _seed_trial_with_person(seeded_session_db_path)
        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--object", "ball"],
            seeded_session_db_path,
        )
        assert result.exit_code != 0
        assert "NAME=SKELETON" in result.output

    def test_malformed_object_position_is_an_error(self, seeded_session_db_path: Path) -> None:
        info = _seed_trial_with_person(seeded_session_db_path)
        result = _invoke(
            ["track", "run-persons", "--trial", info["trial_id"], "--object", "ball=abcd@1,2"],
            seeded_session_db_path,
        )
        assert result.exit_code != 0
        assert "X,Y,Z" in result.output

    @patch("posetrak.cli.track.run_multi_person_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_each_object_carries_its_own_seed(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import MultiPersonResult

        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = MultiPersonResult(exit_code=0, run_ids=[generate_id()] * 3)

        info = _seed_trial_with_person(seeded_session_db_path)
        first = _seed_object_in_trial(seeded_session_db_path, info["trial_id"], time_range=(2.0, 8.0), name="ball")
        second = _seed_object_in_trial(seeded_session_db_path, info["trial_id"], time_range=(2.0, 8.0), name="sword")
        result = _invoke(
            [
                "track", "run-persons", "--trial", info["trial_id"], "--persons", "Alice",
                "--object", f"ball={first['skel_id'][:8]}@1,2,3",
                "--object", f"sword={second['skel_id'][:8]}",
                "--output-dir", str(tmp_path / "out"),
            ],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        assert [s.seed_position for s in mock_run.call_args.args[1]] == [None, (1.0, 2.0, 3.0), None]


def test_a_run_config_made_from_a_hierarchical_base_is_hierarchical(tmp_path):
    """The stage rows select hierarchical mode and active_joint_groups restricts the
    parent stage; a run config made from the base must carry both."""
    from posetrak.cli.track import _build_run_config
    from posetrak.db.db import create_session
    from posetrak.db.manage_config import HIERARCHICAL_CONFIG_ID, seed_hierarchical_tracker_config

    conn = create_session(tmp_path / "s.db")
    seed_hierarchical_tracker_config(conn)
    new_id = _build_run_config(
        conn, base_config_id=HIERARCHICAL_CONFIG_ID, process_noise_std=None, process_noise_vel_std=None,
        velocity_half_life_s=None, measurement_noise_std=None, pose_noise_std=None,
        outlier_threshold=None, tracker_fps=None, use_relative_obs=False,
        relative_min_conf=None, cross_pair_radius=None, cross_pair_max_n=None,
        velocity_cam_labels=None, sequence_cameras=[],
    )
    stages = conn.execute(
        "SELECT group_name FROM tracker_config_stages WHERE tracker_config_id = ? ORDER BY group_name",
        (new_id,),
    ).fetchall()
    assert [r["group_name"] for r in stages] == ["HandL", "HandR"]
    row = conn.execute("SELECT active_joint_groups FROM tracker_configs WHERE id = ?", (new_id,)).fetchone()
    assert json.loads(row["active_joint_groups"]) == ["main"]


def test_a_run_config_made_from_a_base_keeps_every_other_setting(tmp_path):
    """Settings the CLI does not override (dot assignment, segmentation mask, ...)
    must be carried over from the base config, not reset to defaults."""
    from posetrak.cli.track import _build_run_config
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "s.db")
    conn.execute(
        "INSERT INTO tracker_configs (id, name, created_at, dot_seg_mask_enabled,"
        " dot_seg_mask_hard_distance_px, dot_assignment_gate_mahalanobis, outlier_threshold)"
        " VALUES ('base', 'b', '2026-01-01', 1, 70.0, 12.5, 6.0)"
    )
    new_id = _build_run_config(
        conn, base_config_id="base", process_noise_std=None, process_noise_vel_std=None,
        velocity_half_life_s=None, measurement_noise_std=None, pose_noise_std=None,
        outlier_threshold=9.0, tracker_fps=None, use_relative_obs=False,
        relative_min_conf=None, cross_pair_radius=None, cross_pair_max_n=None,
        velocity_cam_labels=None, sequence_cameras=[],
    )
    row = conn.execute("SELECT * FROM tracker_configs WHERE id = ?", (new_id,)).fetchone()
    assert (row["dot_seg_mask_enabled"], row["dot_seg_mask_hard_distance_px"]) == (1, 70.0)
    assert row["dot_assignment_gate_mahalanobis"] == 12.5
    assert row["outlier_threshold"] == 9.0   # the CLI override still wins
    assert row["parent_id"] == "base"


def _config_with_options(tmp_path, base_columns="", **options):
    from posetrak.cli.track import _apply_assignment_options, _build_run_config
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "s.db")
    conn.execute("INSERT INTO tracker_configs (id, name, created_at) VALUES ('base', 'b', '2026-01-01')")
    if base_columns:
        conn.execute(f"UPDATE tracker_configs SET {base_columns} WHERE id = 'base'")
    new_id = _build_run_config(
        conn, base_config_id="base", process_noise_std=None, process_noise_vel_std=None,
        velocity_half_life_s=None, measurement_noise_std=None, pose_noise_std=None, outlier_threshold=None,
        tracker_fps=None, use_relative_obs=False, relative_min_conf=None, cross_pair_radius=None,
        cross_pair_max_n=None, velocity_cam_labels=None, sequence_cameras=[])
    defaults = dict(candidates=None, strict_mask=False, mask_min_fraction=None, mask_window=None,
                    multi_view_agreement=False, agreement_px=None, agreement_factor=None)
    _apply_assignment_options(conn, new_id, **{**defaults, **options})
    return conn.execute("SELECT * FROM tracker_configs WHERE id = ?", (new_id,)).fetchone()


def test_candidates_option_sets_only_the_candidate_column(tmp_path):
    row = _config_with_options(tmp_path, candidates=True)
    assert row["kp_candidates_enabled"] == 1
    assert row["dot_seg_mask_enabled"] is None and row["dot_cross_view_corroboration_px"] is None
    (tmp_path / "off").mkdir()
    assert _config_with_options(tmp_path / "off", candidates=False)["kp_candidates_enabled"] == 0


def test_kp_cost_options_set_their_own_columns(tmp_path):
    row = _config_with_options(tmp_path, kp_peak_weight=1.5, kp_tracklet_multiplier=3.0)
    assert (row["kp_candidate_peak_weight"], row["kp_candidate_tracklet_multiplier"]) == (1.5, 3.0)
    assert row["kp_candidates_enabled"] is None


def test_no_option_leaves_the_base_configs_candidate_setting(tmp_path):
    assert _config_with_options(tmp_path, "kp_candidates_enabled = 1")["kp_candidates_enabled"] == 1


def test_strict_mask_and_agreement_set_their_own_settings_with_base_values_kept(tmp_path):
    row = _config_with_options(tmp_path, "dot_cross_view_corroboration_px = 20.0",
                               strict_mask=True, multi_view_agreement=True, agreement_factor=2.5)
    assert (row["dot_seg_mask_enabled"], row["dot_seg_mask_min_tracklet_fraction"]) == (1, 0.5)
    assert (row["dot_cross_view_corroboration_px"], row["dot_corroboration_gate_multiplier"]) == (20.0, 2.5)
    assert row["kp_candidates_enabled"] is None


def test_window_choice_options_set_their_own_columns_and_nothing_else(tmp_path):
    from posetrak.cli.track import _window_choice_changes

    changes = _window_choice_changes(
        kp_choice="window", support_window=1.5, delta=0.5, beta=0.4, gamma=0.1, rounds=2, skip_cameras=("cam-b",),
        sequence_cameras=["cam-a", "cam-b", "cam-c"], strict_filter=True, strict_window=8, strict_min_fraction=0.6,
        mask_weight=0.3, mask_window=2.5, mask_classes="body,feet")
    assert changes == {
        "kp_candidate_choice": "window", "kp_choice_support_window_s": 1.5, "kp_choice_delta": 0.5,
        "kp_choice_beta": 0.4, "kp_choice_gamma": 0.1, "kp_choice_rounds": 2,
        "kp_strict_filter_window_frames": 8, "kp_strict_filter_min_fraction": 0.6, "kp_mask_term_weight": 0.3,
        "kp_mask_term_window_s": 2.5, "kp_mask_term_classes": "body,feet", "kp_strict_filter": 1,
        "kp_choice_skip_camera_ids": "[1]"}
    none = dict(kp_choice=None, support_window=None, delta=None, beta=None, gamma=None, rounds=None, skip_cameras=(),
                sequence_cameras=[], strict_filter=None, strict_window=None, strict_min_fraction=None,
                mask_weight=None, mask_window=None, mask_classes=None)
    assert _window_choice_changes(**none) == {}
    assert _window_choice_changes(**{**none, "strict_filter": False}) == {"kp_strict_filter": 0}


def test_window_choice_options_reject_an_unknown_camera_and_class():
    import click
    from posetrak.cli.track import _window_choice_changes

    none = dict(kp_choice=None, support_window=None, delta=None, beta=None, gamma=None, rounds=None, skip_cameras=(),
                sequence_cameras=["cam-a"], strict_filter=None, strict_window=None, strict_min_fraction=None,
                mask_weight=None, mask_window=None, mask_classes=None)
    with pytest.raises(click.ClickException, match="cam-z"):
        _window_choice_changes(**{**none, "skip_cameras": ("cam-z",)})
    with pytest.raises(click.ClickException, match="faces"):
        _window_choice_changes(**{**none, "mask_classes": "body,faces"})


def test_window_choice_settings_land_in_the_config_row(tmp_path):
    row = _config_with_options(tmp_path, candidates=True,
                               choice={"kp_candidate_choice": "window", "kp_choice_gamma": 0.1})
    assert (row["kp_candidates_enabled"], row["kp_candidate_choice"], row["kp_choice_gamma"]) == (1, "window", 0.1)
    assert row["kp_choice_beta"] is None  # not given: the tracker's default (0.3) applies
