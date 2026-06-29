"""posetrak CLI — main entry point.

Provides the top-level Click group with global options and wires in all
sub-command groups.

Global options
--------------
--registry PATH   Registry DB path (default: $POSETRAK_REGISTRY or ~/.posetrak/registry.db)
--session PATH    Session DB path (no default; commands that need it fail clearly if absent)
--json            Emit JSONL on list/show commands instead of human-readable tables
-v / --verbose    Increase log level
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from posetrak.db.db import DEFAULT_REGISTRY_PATH


# ---------------------------------------------------------------------------
# Global Click group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--registry",
    envvar="POSETRAK_REGISTRY",
    default=str(DEFAULT_REGISTRY_PATH),
    metavar="PATH",
    show_default=True,
    help="Path to the registry .db file.",
)
@click.option(
    "--session",
    "session",
    envvar="POSETRAK_SESSION_DB",
    default=None,
    metavar="PATH",
    help="Path to the session .db file.",
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    default=False,
    help="Emit JSONL output on list/show commands.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase log verbosity (repeat for more).",
)
@click.pass_context
def main(
    ctx: click.Context,
    registry: str,
    session: str | None,
    json_mode: bool,
    verbose: int,
) -> None:
    """posetrak — motion capture database management CLI."""
    ctx.ensure_object(dict)
    ctx.obj["registry"] = registry
    ctx.obj["session"] = session
    ctx.obj["json_mode"] = json_mode
    ctx.obj["verbose"] = verbose

    if verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Register sub-command groups
# ---------------------------------------------------------------------------

from posetrak.cli.registry import (  # noqa: E402
    registry_group,
    camera_model_group,
    camera_mode_group,
    camera_group,
    calib_group,
)
from posetrak.cli.session import (  # noqa: E402
    session_group,
    capture_group,
    extrinsics_group,
    sync_group,
)
from posetrak.cli.skeleton import skeleton_group  # noqa: E402
from posetrak.cli.config import config_group  # noqa: E402
from posetrak.cli.pose import pose_group  # noqa: E402
from posetrak.cli.detect import detect_group  # noqa: E402
from posetrak.cli.track import track_group  # noqa: E402
from posetrak.cli.trial import trial_group  # noqa: E402

main.add_command(registry_group, "registry")
main.add_command(camera_model_group, "camera-model")
main.add_command(camera_mode_group, "camera-mode")
main.add_command(camera_group, "camera")
main.add_command(calib_group, "calib")
main.add_command(skeleton_group, "skeleton")
main.add_command(config_group, "config")
main.add_command(session_group, "session")
main.add_command(capture_group, "capture")
main.add_command(extrinsics_group, "extrinsics")
main.add_command(sync_group, "sync")
main.add_command(pose_group, "pose")
main.add_command(detect_group, "detect")
main.add_command(track_group, "track")
main.add_command(trial_group, "trial")
