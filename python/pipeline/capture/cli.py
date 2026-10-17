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
    DEFAULT_CONFIG_DIR,
    CaptureConfig,
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


@main.command("run")
@click.option("--demo", type=int, default=0, metavar="N", help="Serve N simulated cameras and a temporary database.")
@click.option("--host", default=None, help="Address to listen on (default: from the configuration).")
@click.option("--port", type=int, default=None, help="Port to listen on (default: from the configuration).")
@click.option("-v", "--verbose", is_flag=True, help="Log debug messages.")
@click.pass_obj
def run(config_path: Path, demo: int, host: str | None, port: int | None, verbose: bool) -> None:
    """Run the capture daemon and print the address of the web UI."""
    import uvicorn

    from pipeline.capture.api import create_app
    from pipeline.capture.config import new_token, parse_config
    from pipeline.capture.daemon import controller_factory
    from pipeline.capture.network import access_url, lan_addresses, qr_text

    try:
        if demo and not config_path.exists():
            cfg: CaptureConfig = parse_config({"daemon": {"token": new_token()}})
        else:
            cfg = load_config(config_path)
        if not cfg.daemon.token:
            raise ConfigError("daemon.token is empty (create a configuration with `posetrak-capture init-config`)")
        factory = controller_factory(cfg, demo_cameras=demo)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    listen_host = host or cfg.daemon.host
    listen_port = port or cfg.daemon.port
    log_path = setup_logging(DEFAULT_CONFIG_DIR, verbose=verbose)
    addresses = lan_addresses() if listen_host in ("0.0.0.0", "::") else [listen_host]
    click.echo(f"log file: {log_path}")
    for addr in addresses:
        click.echo(f"web UI:   {access_url(addr, listen_port, cfg.daemon.token)}")
    if addresses:
        url = access_url(addresses[0], listen_port, cfg.daemon.token)
        try:
            click.echo(qr_text(url))
        except UnicodeEncodeError:
            click.echo(qr_text(url, ascii_only=True))
    app = create_app(factory, cfg.daemon.token)
    uvicorn.run(app, host=listen_host, port=listen_port, log_level="debug" if verbose else "info")


@main.command("firewall-rule")
@click.option("--apply", "apply_rule", is_flag=True, help="Add the rule now (needs an administrator shell).")
@click.pass_obj
def firewall_rule(config_path: Path, apply_rule: bool) -> None:
    """Show (or add) the Windows firewall rule that lets phones reach the daemon."""
    import subprocess

    try:
        port = load_config(config_path).daemon.port
    except ConfigError:
        from pipeline.capture.config import DEFAULT_PORT

        port = DEFAULT_PORT
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule", "name=posetrak-capture",
        "dir=in", "action=allow", "protocol=TCP", f"localport={port}", "profile=private,public",
    ]
    click.echo(" ".join(cmd))
    if apply_rule:
        result = subprocess.run(cmd, capture_output=True, text=True)
        click.echo((result.stdout + result.stderr).strip())
        if result.returncode:
            raise click.ClickException("netsh failed; run this from a shell started as administrator")


if __name__ == "__main__":
    main()
