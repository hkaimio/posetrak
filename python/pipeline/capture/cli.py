# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""cli.py — command-line entry point of the capture daemon (``posetrak-capture``)."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import click

from pipeline.capture.config import (
    ConfigError,
    default_config_path,
    load_config,
    new_token,
    render_default_config,
)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_dir: Path, *, verbose: bool = False) -> Path:
    """Log to the console and to a rotating file in *log_dir*; return the file path.

    The file is what makes a failed take reconstructable afterwards, so it is
    kept at INFO even when the console is quiet.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "capture.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        if getattr(handler, "_posetrak_capture", False):
            root.removeHandler(handler)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    console = logging.StreamHandler()
    for handler in (file_handler, console):
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._posetrak_capture = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    return log_path


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Configuration file (default: {default_config_path()}).",
)
@click.pass_context
def main(ctx: click.Context, config_path: Path | None) -> None:
    """Remote capture control for the posetrak camera fleet."""
    ctx.obj = config_path or default_config_path()


@main.command("init-config")
@click.option("--session-db", type=click.Path(path_type=Path), default=None, help="Session database to record into.")
@click.option("--force", is_flag=True, help="Overwrite an existing configuration file.")
@click.pass_obj
def init_config(config_path: Path, session_db: Path | None, force: bool) -> None:
    """Write a starter configuration file with a fresh access token."""
    if config_path.exists() and not force:
        raise click.ClickException(f"{config_path} already exists (use --force to overwrite)")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_default_config(new_token(), session_db), encoding="utf-8")
    click.echo(f"wrote {config_path}")


@main.command("show-config")
@click.pass_obj
def show_config(config_path: Path) -> None:
    """Validate the configuration file and print what it describes."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"config:   {config_path}")
    click.echo(f"listen:   {cfg.daemon.host}:{cfg.daemon.port}")
    click.echo(f"session:  {cfg.daemon.session_db or '(not set)'}")
    click.echo(f"profile:  {cfg.profile.name} {cfg.profile.settings}")
    for net in cfg.networks:
        click.echo(f"network:  {net.name} (ssid {net.ssid!r}, discovery {net.discovery})")
    for cam in cfg.cameras:
        click.echo(f"camera:   {cam.label} ({cam.method})")


if __name__ == "__main__":
    main()
