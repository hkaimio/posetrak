"""Posetrak MCP diagnostic server.

Exposes read-only tools for inspecting tracking runs in a session database.
Intended for use with Claude Desktop, Claude Code, or any MCP-compatible client.

Usage:
    uv run python python/app/mcp/server.py --db-path /path/to/session.db

.mcp.json example:
    {
      "mcpServers": {
        "posetrak": {
          "command": "uv",
          "args": ["run", "python", "python/app/mcp/server.py",
                   "--db-path", "/path/to/session.db"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from app.mcp import db as _db
from app.mcp.tools import coverage as _coverage
from app.mcp.tools import diagnostics as _diag
from app.mcp.tools import geometry as _geo
from app.mcp.tools import runs as _runs

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "posetrak-diagnostics",
    instructions=(
        "Diagnostic tools for posetrak motion capture tracking runs. "
        "Start with list_tracking_runs to see what's in the database, "
        "then get_run_info on a specific run to understand its configuration. "
        "Use get_filter_stats to find divergence windows, get_camera_coverage "
        "to see which cameras see which body parts, get_observation_gaps to "
        "quantify how far the filter has drifted, get_camera_geometry to "
        "diagnose parallax problems, and get_edit_coverage to find uneditied "
        "keypoints that may be causing errors."
    ),
)

# Module-level DB path set at startup
_db_path: Path | None = None


def _conn():
    if _db_path is None:
        raise RuntimeError("Server started without --db-path.")
    return _db.connect_readonly(_db_path)


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
def list_tracking_runs() -> str:
    """List all tracking runs in the session database.

    Shows run ID, skeleton name, timestamp, time range, and persons tracked.
    Use this first to identify the run_id you want to investigate.
    """
    with _conn() as conn:
        return _runs.list_tracking_runs(conn)


@mcp.tool()
def get_run_info(run_id: str) -> str:
    """Get full context for a tracking run.

    Returns: tracker config (including measurement_noise_std and the
    maximum accepted pixel gap), camera list with labels, time range,
    and all skeleton markers with their obs_blob indices.

    Start here when beginning a diagnostic session for a specific run.
    """
    with _conn() as conn:
        return _runs.get_run_info(conn, run_id)


@mcp.tool()
def get_filter_stats(run_id: str, start_s: float, end_s: float) -> str:
    """Per-step NIS/DOF and covariance condition number.

    NIS/DOF > 1.5: filter is overconfident — state has likely drifted from
    reality; observations are surprising it.

    NIS/DOF < 0.3: filter is very underconfident — measurement_noise_std
    may be too large; the filter is not extracting available information.

    Covariance condition number > 1e6: covariance is ill-conditioned in at
    least one direction (often the depth direction when cameras have poor
    parallax for a particular marker).

    Anomalous windows are summarised at the top of the output.
    """
    with _conn() as conn:
        return _diag.get_filter_stats(conn, run_id, start_s, end_s)


@mcp.tool()
def get_camera_coverage(
    run_id: str,
    start_s: float,
    end_s: float,
    markers: list[str],
) -> str:
    """Per-step inlier/outlier/absent grid for specified markers and all cameras.

    I = inlier (observation accepted by filter)
    x = outlier (observation rejected by Mahalanobis gate)
    . = no observation for that camera/step

    markers: list of skeleton marker names as they appear in get_run_info,
    e.g. ["Ankle.R", "Knee.R", "Hip.L"].

    Use this to identify which cameras contribute useful observations for
    specific body parts, and to spot windows where coverage drops.
    """
    with _conn() as conn:
        return _coverage.get_camera_coverage(conn, run_id, start_s, end_s, markers)


@mcp.tool()
def get_observation_gaps(
    run_id: str,
    start_s: float,
    end_s: float,
    markers: list[str],
) -> str:
    """Actual vs predicted pixel positions per camera for specified markers.

    Shows the pixel gap between where the filter predicts each marker should
    project and where it is actually observed. Large gaps (≥30px, flagged *)
    indicate the filter's 3-D estimate has drifted from what the cameras see.

    Gaps accepted as inliers despite being large (e.g. 40–70px) indicate
    measurement_noise_std is too large and the Mahalanobis gate is too loose.

    markers: skeleton marker names, e.g. ["Ankle.R", "Ankle.L"].
    """
    with _conn() as conn:
        return _diag.get_observation_gaps(conn, run_id, start_s, end_s, markers)


@mcp.tool()
def get_camera_geometry(run_id: str) -> str:
    """Camera 3-D world positions, viewing directions, and pairwise parallax.

    For each pair of cameras, reports:
    - baseline (metres between camera centres)
    - angle between viewing directions (degrees)
    - depth quality: GOOD (≥90°, near-opposite), MODERATE, or POOR (<45°)

    POOR pairs have the same-side camera geometry problem: a marker that is
    only visible to cameras with POOR mutual parallax will have its 3-D depth
    underdetermined, causing the filter to drift in that direction.
    """
    with _conn() as conn:
        return _geo.get_camera_geometry(conn, run_id)


@mcp.tool()
def get_edit_coverage(run_id: str) -> str:
    """Which HALPE keypoints are edited per camera in this run's observation sequence.

    Shows edited keypoint names per camera with frame ranges.
    More importantly, flags key body landmarks (hips, knees, ankles) that
    are NOT edited in cameras that have other edits — these are likely
    candidates for wrong detections that the filter is following uncorrected.
    """
    with _conn() as conn:
        return _coverage.get_edit_coverage(conn, run_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Posetrak MCP diagnostic server (read-only)"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to a posetrak session .db file",
    )
    args, _ = parser.parse_known_args()

    global _db_path
    _db_path = Path(args.db_path)
    if not _db_path.exists():
        print(f"Error: database not found: {_db_path}", file=sys.stderr)
        sys.exit(1)

    mcp.run()


if __name__ == "__main__":
    main()
