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

from posetrak.cli.session import _open_session_required, _resolve
from posetrak.db.sequence_dots import add_dots_to_sequence


@click.group("sequence")
def sequence_group() -> None:
    """Build and extend observation sequences."""


@sequence_group.command("finalise-object")
@click.option(
    "--detection-run", "run", required=True, metavar="UUID",
    help="An object-bound marker run (`detect run --type aruco|dots --object NAME`); prefix accepted.",
)
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def sequence_finalise_object(obj: dict, run: str, notes: str) -> None:
    """Make the observation sequence of a tracked object from its marker run.

    Copies the run's coded-marker corners and reflective-dot candidates into one
    new sequence. Finalising a run again replaces its sequence, unless that
    sequence already has tracking results or manual edits.
    """
    from app.pose.finalise import finalise_object_to_db

    conn = _open_session_required(obj)
    try:
        run_id = _resolve(conn, "detection_runs", run)
        try:
            sequence_id = finalise_object_to_db(conn, run_id, notes=notes)
        except (ValueError, RuntimeError) as exc:
            raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()
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
