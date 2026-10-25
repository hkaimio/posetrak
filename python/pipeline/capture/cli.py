# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""cli.py — command-line entry point of the capture daemon (``posetrak-capture``)."""

from __future__ import annotations

import logging
import logging.handlers
import sqlite3
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
from pipeline.capture.registry_link import RegistryCamera

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
        click.echo(f"camera:   {cam.label} ({cam.method or 'from the registry'})")


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


def _load_optional_config(config_path: Path) -> CaptureConfig | None:
    if not config_path.exists():
        return None
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _open_registry(cfg: CaptureConfig | None) -> sqlite3.Connection:
    from posetrak.db.db import DEFAULT_REGISTRY_PATH, open_registry

    return open_registry((cfg.daemon.registry_db if cfg else None) or DEFAULT_REGISTRY_PATH)


def _choose_nearby_camera(registry: sqlite3.Connection, label: str, *, scan_s: float) -> str:
    """Scan over Bluetooth and let the operator pick which camera is *label*; return its device serial."""
    import asyncio

    from pipeline.capture.gopro_ble import scan_cameras
    from pipeline.capture.registry_link import cameras_by_gopro_serial

    linked = cameras_by_gopro_serial(registry)
    click.echo(f"{label} is not linked to a camera yet; scanning for cameras ({scan_s:g} s) ...")
    nearby = asyncio.run(scan_cameras(scan_s))
    if not nearby:
        raise click.ClickException("no GoPro cameras found; switch the camera on, or give its serial with --serial")
    for i, cam in enumerate(nearby, 1):
        owner = f"  (already {linked[cam.serial]})" if cam.serial in linked else ""
        click.echo(f"  {i}. GoPro {cam.serial}  signal {cam.rssi if cam.rssi is not None else '?'}{owner}")
    pick = click.prompt(f"Which one is {label}?", type=click.IntRange(1, len(nearby)))
    return nearby[pick - 1].serial


def _device_serial(registry: sqlite3.Connection, camera: RegistryCamera, serial: str | None, *, scan_s: float) -> str:
    """Device serial of a registry camera: the one it is linked to, the one given, or a scanned pick."""
    if camera.gopro_serial and serial and camera.gopro_serial != serial:
        raise click.ClickException(f"{camera.label} is linked to camera {camera.gopro_serial}, not {serial}")
    return serial or camera.gopro_serial or _choose_nearby_camera(registry, camera.label, scan_s=scan_s)


_SERIAL_HELP = "Device serial (the digits in the camera's Bluetooth name) when the camera is not linked yet."


@main.command("import-credentials")
@click.option("--delete-files", is_flag=True, help="Delete each credential file once its entries are in the registry.")
@click.pass_obj
def import_credentials(config_path: Path, delete_files: bool) -> None:
    """Copy camera credentials saved in files by older versions into the registry.

    A file belongs to a camera by its device serial, so the camera must already be linked
    to a registry camera (`provision <label>` links it); files of other cameras are left alone.
    """
    from pipeline.capture.credentials import CredentialStore, RegistryCredentialStore

    cfg = _load_optional_config(config_path)
    registry = _open_registry(cfg)
    try:
        legacy = CredentialStore(cfg.daemon.credentials_dir if cfg else None)
        store = RegistryCredentialStore(registry, legacy)
        imported = store.import_legacy()
        for serial, count in imported.items():
            if count:
                click.echo(f"  camera {serial}: {count} WLAN entries imported")
                if delete_files:
                    legacy._path(serial).unlink()
            else:
                click.echo(f"  camera {serial}: not linked to a registry camera, left alone")
        if not imported:
            click.echo(f"no credential files in {legacy.directory}")
    finally:
        registry.close()


@main.command("offload")
@click.argument("labels", nargs=-1, required=True)
@click.option(
    "--dest", type=click.Path(path_type=Path), default=Path("offload"), show_default=True,
    help="Directory to copy to; files go under <dest>/<label>/<camera folder>/.",
)
@click.option("--list", "list_only", is_flag=True, help="List the recordings on the cameras instead of copying them.")
@click.option("--last", type=int, default=None, metavar="N", help="Only the N most recent recordings of each camera.")
@click.option("--all-files", is_flag=True, help="Copy photos and other files too, not only videos.")
@click.pass_obj
def offload(config_path: Path, labels: tuple[str, ...], dest: Path, list_only: bool, last: int | None, all_files: bool) -> None:
    """Copy the recordings of cameras over the network.

    LABELS are camera labels from the camera registry. Each file is checked against the size
    the camera announced and gets a JSON file with its SHA-256 and the camera's metadata
    beside it. Files copied before are skipped and an interrupted copy resumes, so the
    command can be repeated. Do not run it while the cameras are recording.
    """
    import asyncio
    import datetime

    from pipeline.capture.collect import is_video, offload_camera
    from pipeline.capture.config import CameraConfig
    from pipeline.capture.credentials import CredentialStore, RegistryCredentialStore
    from pipeline.capture.daemon import build_driver, resolve_registry_cameras
    from pipeline.capture.devices import OffloadError

    cfg = _load_optional_config(config_path)
    registry = _open_registry(cfg)
    try:
        try:
            resolved = resolve_registry_cameras(registry, [CameraConfig(label) for label in labels])
            store = RegistryCredentialStore(registry, CredentialStore(cfg.daemon.credentials_dir if cfg else None))
            drivers = [build_driver(c, cfg, store) for c, _ in resolved]
        except ConfigError as exc:
            raise click.ClickException(str(exc)) from exc
    finally:
        registry.close()
    for driver in drivers:
        if not driver.supports_offload:
            raise click.ClickException(f"{driver.label}: this camera type cannot be offloaded over the network")

    def when(ts: int | None) -> str:
        if ts is None:
            return "?"
        return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    async def run() -> None:
        for driver in drivers:
            await driver.connect()
            try:
                if driver.status.link != "up":
                    raise OffloadError(f"{driver.label}: camera not reachable")
                if list_only:
                    files = [f for f in await driver.list_media() if all_files or is_video(f)]
                    for f in sorted(files, key=lambda f: (f.created or 0, f.path))[-last if last else None :]:
                        size = f"{f.size / 1e6:9.1f} MB" if f.size is not None else "         ? MB"
                        click.echo(f"  {driver.label}  {f.path:<26} {size}  created {when(f.created)} (camera clock)")
                    continue
                shown = {"name": None}

                def progress(file, done, total) -> None:  # noqa: ANN001
                    if shown["name"] != file.path:
                        shown["name"] = file.path
                        click.echo(f"  {driver.label}  {file.path} ...", nl=False)
                    if total is not None and done >= total:
                        click.echo(f" {done / 1e6:.1f} MB")

                results = await offload_camera(driver, dest, include_all=all_files, last=last, progress=progress)
                for r in results:
                    if r.skipped:
                        click.echo(f"  {driver.label}  {r.file.path}: already copied")
                if not results:
                    click.echo(f"  {driver.label}: no recordings")
            finally:
                await driver.close()

    try:
        asyncio.run(run())
    except OffloadError as exc:
        raise click.ClickException(str(exc)) from exc
    if not list_only:
        click.echo(f"copied under {dest.resolve()}")


@main.command("detect-mode")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_obj
def detect_mode_command(config_path: Path, files: tuple[Path, ...]) -> None:
    """Tell which registry camera and capture mode video files were recorded in.

    The camera is found by the serial number in the file, the mode by the image size, the
    frame rate and, where the file states it, the lens. Needs exiftool on the PATH.
    """
    from pipeline.capture.mode_detect import ProbeError, match_registry, read_video_facts

    registry = _open_registry(_load_optional_config(config_path))
    try:
        for path in files:
            try:
                found = match_registry(registry, read_video_facts(path))
            except ProbeError as exc:
                raise click.ClickException(str(exc)) from exc
            facts = found.detection.facts
            details = ", ".join(
                x
                for x in (
                    f"{facts.make or ''} {facts.model or ''}".strip(),
                    f"serial {facts.serial_number}" if facts.serial_number else "",
                    f"{facts.width}x{facts.height} at {facts.fps:g} fps",
                    f"lens {facts.lens}" if facts.lens else "no lens in the file",
                    *(f"{k.replace('_', ' ')} {v}" for k, v in facts.extra.items()),
                )
                if x
            )
            click.echo(f"{path.name}: {details}")
            click.echo(f"  camera: {found.label or 'not identified'}" + (f" ({found.model_name})" if found.model_name else ""))
            for note in found.notes:
                click.echo(f"  note: {note}")
            matches = found.detection.matches
            if not matches:
                click.echo("  mode: none of the registered modes fits")
            for m in matches:
                certainty = "certain" if m.confidence >= 0.99 else f"unsure ({', '.join(m.unchecked)} not checked)"
                click.echo(f"  mode: {m.mode.name} - {certainty}")
    finally:
        registry.close()


@main.command("settings")
def settings() -> None:
    """List the settings a capture mode can specify for a Hero13, with their allowed values.

    These are the keys of a mode's ``capture_settings`` (``posetrak camera-mode add
    --settings '{"lens": "linear"}'``) and of a setup's per-camera ``settings``. A
    number as key is passed to the camera as a raw setting id with a raw option value.
    """
    from pipeline.capture.gopro_profile import SUPPORTED

    click.echo("Hero13 (gopro_cohn):")
    for name, choices in SUPPORTED.items():
        click.echo(f"  {name:<12} {', '.join(str(c) for c in choices)}")
    click.echo("  <number>     raw Open GoPro setting id, with the raw option value")
    click.echo("Which values a camera accepts depends on the others (frame rates on the resolution, lenses on both);")
    click.echo("a setting the camera refuses is reported when the cameras are armed.")


@main.command("cameras")
@click.option("--scan", "scan_s", type=float, default=6.0, show_default=True, help="Seconds to scan over Bluetooth.")
@click.pass_obj
def cameras(config_path: Path, scan_s: float) -> None:
    """List the GoPro cameras in Bluetooth range and the registry camera each is linked to."""
    import asyncio

    from pipeline.capture.gopro_ble import scan_cameras
    from pipeline.capture.registry_link import cameras_by_gopro_serial

    registry = _open_registry(_load_optional_config(config_path))
    try:
        linked = cameras_by_gopro_serial(registry)
    finally:
        registry.close()
    for cam in asyncio.run(scan_cameras(scan_s)):
        rssi = cam.rssi if cam.rssi is not None else "?"
        click.echo(f"  GoPro {cam.serial}  signal {rssi}  {linked.get(cam.serial, '(not linked)')}")


@main.command("networks")
@click.argument("label")
@click.option("--serial", default=None, help=_SERIAL_HELP)
@click.option("--scan", "scan_s", type=float, default=6.0, show_default=True, help="Seconds to scan over Bluetooth.")
@click.pass_obj
def networks(config_path: Path, label: str, serial: str | None, scan_s: float) -> None:
    """List the WLANs a Hero13 sees over Bluetooth and which it is provisioned on.

    LABEL is the camera's label in the camera registry.
    """
    import asyncio

    from pipeline.capture.gopro_ble import list_networks
    from pipeline.capture.registry_link import LinkError, find_camera

    registry = _open_registry(_load_optional_config(config_path))
    try:
        try:
            camera = find_camera(registry, label)
        except LinkError as exc:
            raise click.ClickException(str(exc)) from exc
        device = _device_serial(registry, camera, serial, scan_s=scan_s)
    finally:
        registry.close()
    for net in asyncio.run(list_networks(device)):
        marks = ",".join(m for m, on in (("provisioned", net.configured), ("joined", net.associated)) if on)
        click.echo(f"  {net.ssid:<28} {net.frequency_mhz} MHz  bars {net.signal_bars}  {marks}")


@main.command("provision")
@click.argument("label")
@click.option("--serial", default=None, help=_SERIAL_HELP)
@click.option("--scan", "scan_s", type=float, default=6.0, show_default=True, help="Seconds to scan over Bluetooth.")
@click.option("--network", "network_name", default=None, help="Name of a [[network]] block in the configuration.")
@click.option("--ssid", default=None, help="SSID to join, when no configured network is named.")
@click.option("--password", prompt=True, hide_input=True, help="WLAN password (prompted if omitted).")
@click.option("--force", is_flag=True, help="Create new COHN credentials even if the camera has some.")
@click.pass_obj
def provision(
    config_path: Path,
    label: str,
    serial: str | None,
    scan_s: float,
    network_name: str | None,
    ssid: str | None,
    password: str,
    force: bool,
) -> None:
    """Set a registry camera up for control over a WLAN, over Bluetooth.

    LABEL is the camera's label in the camera registry. A camera not linked to a
    physical Hero13 yet is picked from a Bluetooth scan (or given with --serial)
    and the link is stored in the registry. The camera joins the WLAN, creates
    COHN credentials there, and they are stored under the user configuration
    directory. Repeat for every WLAN the camera will be used on (home, location
    access point). A camera that was never paired with this computer must be in
    pairing mode.
    """
    import asyncio

    from pipeline.capture.credentials import CredentialStore, RegistryCredentialStore
    from pipeline.capture.gopro_ble import ProvisioningError
    from pipeline.capture.gopro_ble import provision as run_provision
    from pipeline.capture.registry_link import (
        LinkError,
        SerialCheck,
        check_link,
        find_camera,
        link_gopro,
        reconcile_serial,
        serial_matches,
    )

    cfg = _load_optional_config(config_path)
    if network_name:
        if cfg is None:
            raise click.ClickException(f"--network needs a configuration file ({config_path})")
        try:
            ssid = cfg.network(network_name).ssid
        except ConfigError as exc:
            raise click.ClickException(str(exc)) from exc
    if not ssid:
        raise click.ClickException("give --network or --ssid")
    registry = _open_registry(cfg)
    store = RegistryCredentialStore(registry, CredentialStore(cfg.daemon.credentials_dir if cfg else None))
    try:
        try:
            camera = find_camera(registry, label)
            device = _device_serial(registry, camera, serial, scan_s=scan_s)
            check_link(registry, camera, device)
        except LinkError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"provisioning {label} (camera {device}) on {ssid!r} ...")
        try:
            result = asyncio.run(run_provision(device, ssid, password, force=force))
        except ProvisioningError as exc:
            raise click.ClickException(str(exc)) from exc
        full = result.serial_number
        fill = False
        if full is not None:
            if not serial_matches(full, device):
                raise click.ClickException(f"the camera reports serial {full}, which does not end in {device}; nothing was stored")
            outcome = reconcile_serial(registry, camera, full, fill=False)
            if outcome is SerialCheck.DIFFERS:
                raise click.ClickException(
                    f"the camera's serial {full} differs from {camera.serial_number} in the registry for {label}; "
                    "this is probably a different camera. Nothing was stored."
                )
            if outcome is SerialCheck.FILLED:
                fill = click.confirm(f"Record serial number {full} for {label} in the registry?", default=True)
        link_gopro(registry, camera, device)
        store.save(device, result.credentials)
        if fill and full is not None:
            reconcile_serial(registry, camera, full)
    except LinkError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        registry.close()
    click.echo(f"done: {label} is at {result.credentials.ip_address} on {ssid!r}; credentials saved in the registry")
    click.echo("Give the camera a fixed address in the router and list it under [network.addresses] if the address changes.")


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
