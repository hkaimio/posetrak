"""runner.py — subprocess wrapper for the posetrak-tracker binary.

Pure Python, no Qt. Both the CLI and the UI use this module; the UI runs it
inside a QThread so the event loop stays responsive.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Development build fallback: optbuild relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEVBUILD_BINARY = _REPO_ROOT / "optbuild" / "cli" / "posetrak-tracker"


@dataclass
class TrackerResult:
    exit_code: int
    run_id: str | None  # parsed from "tracking_run_id: UUID\n" in output


@dataclass
class PersonRunSpec:
    """One person's (sequence, skeleton, tracker_config, person_id) tuple for
    a ``--person`` multi-person tracking run -- see
    ``run_multi_person_tracker()``."""

    sequence_id: str
    skeleton_id: str
    config_id: str
    person_id: int


@dataclass
class MultiPersonResult:
    exit_code: int
    # run_ids[i] is person i's tracking_run_id (None if the binary exited
    # before emitting it for that person), in the same order as the `persons`
    # list passed to run_multi_person_tracker().
    run_ids: list[str | None]


def default_binary_path() -> Path:
    """Return the tracker binary path.

    Prefers ~/.posetrak/posetrak-tracker (installed location) and falls back
    to optbuild/cli/posetrak-tracker (developer build).
    """
    user_bin = Path.home() / ".posetrak" / "posetrak-tracker"
    if user_bin.exists():
        return user_bin
    return _DEVBUILD_BINARY


def run_tracker(
    session_path: Path,
    sequence_id: str,
    skeleton_id: str,
    config_id: str,
    output_dir: Path,
    *,
    binary_path: Path | None = None,
    person_id: int = 0,
    start_time: float | None = None,
    end_time: float | None = None,
    smooth: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> TrackerResult:
    """Run the posetrak-tracker binary as a subprocess.

    Blocks until the binary exits. Each output line (split on both ``\\n`` and
    ``\\r``) is forwarded to ``on_progress`` if provided. The binary emits a
    ``tracking_run_id: <UUID>`` line on success; this is parsed and returned in
    ``TrackerResult.run_id``.

    Parameters
    ----------
    session_path:
        Path to the session .db file.
    sequence_id:
        pose_observation_sequences.id to track.
    skeleton_id:
        skeletons.id to use.
    config_id:
        tracker_configs.id row already written to the session DB.
    output_dir:
        Directory where the tracker writes its CSV output files.
    binary_path:
        Explicit path to the tracker binary. Defaults to
        ``default_binary_path()``.
    person_id:
        Person index within the sequence (0 for single-person sessions).
    start_time:
        Optional start time in seconds (passed to binary as ``--start-time``).
    end_time:
        Optional end time in seconds (passed to binary as ``--end-time``).
    smooth:
        Whether to enable RTS smoothing (``--smooth`` flag). Default True.
    on_progress:
        Callback invoked for each non-empty output line. Called from whichever
        thread calls ``run_tracker()``.

    Returns
    -------
    TrackerResult
        Exit code and parsed run_id (None if the binary failed before emitting
        it).
    """
    binary = binary_path or default_binary_path()

    args = [
        str(binary), "track",
        "--session-db", str(session_path),
        "--sequence", sequence_id,
        "--skeleton", skeleton_id,
        "--tracker-config", config_id,
        "--person-id", str(person_id),
        "--output-dir", str(output_dir),
    ]
    if start_time is not None:
        args += ["--start-time", str(start_time)]
    if end_time is not None:
        args += ["--end-time", str(end_time)]
    if smooth:
        args.append("--smooth")

    run_id: str | None = None

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )
    assert proc.stdout is not None

    for raw_line in proc.stdout:
        # The binary uses \r for in-place progress updates; split on both so
        # each update reaches on_progress as a distinct line.
        for line in raw_line.split("\r"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"tracking_run_id:\s*(\S+)", line)
            if m:
                run_id = m.group(1)
            if on_progress is not None:
                on_progress(line)

    proc.wait()
    return TrackerResult(exit_code=proc.returncode, run_id=run_id)


def _build_multi_person_args(
    binary: Path,
    session_path: Path,
    persons: list[PersonRunSpec],
    output_dir: Path,
    *,
    start_time: float | None,
    end_time: float | None,
    smooth: bool,
) -> list[str]:
    """Build the CLI argument list for a ``--person``-mode multi-person run.

    Pulled out of run_multi_person_tracker() as a pure function so the
    argument shape (repeated 4-value ``--person`` groups, per
    cli/track.cpp's ``->expected(4)->take_all()`` option) is unit-testable
    without spawning a subprocess.
    """
    args = [
        str(binary), "track",
        "--session-db", str(session_path),
        "--output-dir", str(output_dir),
    ]
    for p in persons:
        args += ["--person", p.sequence_id, p.skeleton_id, p.config_id, str(p.person_id)]
    if start_time is not None:
        args += ["--start-time", str(start_time)]
    if end_time is not None:
        args += ["--end-time", str(end_time)]
    if smooth:
        args.append("--smooth")
    return args


def run_multi_person_tracker(
    session_path: Path,
    persons: list[PersonRunSpec],
    output_dir: Path,
    *,
    binary_path: Path | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    smooth: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> MultiPersonResult:
    """Run the posetrak-tracker binary in ``--person`` multi-person mode.

    Tracks every person in *persons* through the same session DB in one
    process, interleaved frame-by-frame (Stage 1 of the cross-person
    relative observations plan -- see
    docs/roadmap/features/error-improvements/phase5-cross-person-plan.md).
    Each person's output lands in ``<output_dir>/person_<index>/`` (index
    matching *persons*' order), same convention as
    ``run_multi_person_track_from_db()`` in cli/track.cpp.

    Parameters
    ----------
    session_path:
        Path to the session .db file.
    persons:
        One ``PersonRunSpec`` per person to track together. At least 2 is the
        point of this mode; a single entry works too (equivalent to
        ``run_tracker()`` but through the multi-person code path).
    output_dir:
        Parent directory; each person's CSVs land in
        ``output_dir/person_<index>/``.
    binary_path:
        Explicit path to the tracker binary. Defaults to
        ``default_binary_path()``.
    start_time, end_time:
        Optional sequence time-range override, applied to every person.
    smooth:
        Whether to enable RTS smoothing (``--smooth`` flag). Default True.
    on_progress:
        Callback invoked for each non-empty output line.

    Returns
    -------
    MultiPersonResult
        Exit code and each person's parsed run_id (in *persons* order).
    """
    binary = binary_path or default_binary_path()
    args = _build_multi_person_args(
        binary, session_path, persons, output_dir,
        start_time=start_time, end_time=end_time, smooth=smooth,
    )

    run_ids: list[str | None] = [None] * len(persons)
    next_person_idx = 0

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )
    assert proc.stdout is not None

    for raw_line in proc.stdout:
        for line in raw_line.split("\r"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"tracking_run_id:\s*(\S+)", line)
            if m and next_person_idx < len(run_ids):
                # finalize_person_context() prints one "tracking_run_id:" line
                # per person, in the same order MultiPersonTracker::run()
                # finalizes them -- i.e. `persons` order.
                run_ids[next_person_idx] = m.group(1)
                next_person_idx += 1
            if on_progress is not None:
                on_progress(line)

    proc.wait()
    return MultiPersonResult(exit_code=proc.returncode, run_ids=run_ids)
