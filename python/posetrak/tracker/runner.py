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
