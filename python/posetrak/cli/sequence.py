# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""sequence commands: finalise-object, add-dots.

A tracking subject reads its observations from a ``pose_observation_sequences``
row. These commands make the sequences of tracked objects and attach
reflective-dot candidates to sequences that lack them.
"""

from __future__ import annotations

import click

from posetrak.cli.detect import _resolve_capture_object
from posetrak.cli.session import _open_session_required, _resolve
from posetrak.db.object_marker_run import derive_object_marker_run
from posetrak.db.sequence_dots import add_dots_to_sequence


@click.group("sequence")
def sequence_group() -> None:
    """Build and extend observation sequences."""


@sequence_group.command("finalise-object")
@click.option(
    "--detection-run", "run", required=True, metavar="UUID",
    help="A marker run (`detect run --type aruco|dots`); prefix accepted. Bound to the object "
         "unless --object is given.",
)
@click.option(
    "--object", "object_ref", default=None, metavar="NAME|ID",
    help="Take this object's markers out of a run bound to no object (or to another one), for "
         "props detected together in one pass. Makes a run of its own for the object first.",
)
@click.option(
    "--with-dots", is_flag=True, default=False,
    help="With --object: also copy the run's reflective-dot candidates, for an object whose "
         "marker body has dots.",
)
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def sequence_finalise_object(obj: dict, run: str, object_ref: str | None, with_dots: bool, notes: str) -> None:
    """Make the observation sequence of a tracked object from a marker run.

    Copies the run's coded-marker corners and reflective-dot candidates into one
    new sequence. Finalising a run again replaces its sequence, unless that
    sequence already has tracking results or manual edits.

    Several props can be detected in one pass over the video: run
    `detect run --type aruco --marker-ids ...` once with the ids of all of them,
    then finalise once per prop with --object. Each prop gets a run of its own
    (derived from the shared one, which is not changed) and a sequence.
    """
    from app.pose.finalise import finalise_object_to_db

    if with_dots and object_ref is None:
        raise click.UsageError("--with-dots needs --object.")
    conn = _open_session_required(obj)
    try:
        run_id = _resolve(conn, "detection_runs", run)
        try:
            if object_ref is not None:
                shot_id = conn.execute("SELECT shot_id FROM detection_runs WHERE id = ?", (run_id,)).fetchone()[0]
                run_id = derive_object_marker_run(
                    conn, run_id, _resolve_capture_object(conn, shot_id, object_ref), with_dots=with_dots,
                )
            sequence_id = finalise_object_to_db(conn, run_id, notes=notes)
        except (ValueError, RuntimeError) as exc:
            raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()
    if object_ref is not None:
        click.echo(f"detection_run_id: {run_id}")
    click.echo(f"sequence_id: {sequence_id}")


@sequence_group.command("add-dots")
@click.option(
    "--detection-run", "run", required=True, metavar="UUID",
    help="A marker run that detected dots (`detect run --type dots`); prefix accepted.",
)
@click.option("--sequence", required=True, metavar="UUID", help="Sequence to extend (prefix accepted).")
@click.option(
    "--replace", is_flag=True, default=False,
    help="Delete the sequence's existing dot rows first. Refused once the sequence is tracked or edited.",
)
@click.pass_obj
def sequence_add_dots(obj: dict, run: str, sequence: str, replace: bool) -> None:
    """Attach a run's reflective-dot candidates to an existing sequence.

    For dots worn on a person: the person's sequence comes from a pose run, and
    the dots come from a separate dots run over the same capture and sync
    configuration. Body and hand rows of the sequence are not touched.
    """
    conn = _open_session_required(obj)
    try:
        run_id = _resolve(conn, "detection_runs", run)
        sequence_id = _resolve(conn, "pose_observation_sequences", sequence)
        try:
            result = add_dots_to_sequence(conn, run_id, sequence_id, replace=replace)
        except (ValueError, RuntimeError) as exc:
            raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()
    if result.skipped_no_camera or result.skipped_no_timestamp:
        click.echo(
            f"Skipped {result.skipped_no_camera} rows with no camera and "
            f"{result.skipped_no_timestamp} rows with no sync timestamp.", err=True,
        )
    click.echo(f"Added {result.inserted} dot rows to sequence {sequence_id}.")
