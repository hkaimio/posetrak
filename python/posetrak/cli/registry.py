"""Registry, camera-model, camera-mode, camera, and calib commands."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click

from posetrak.db.db import (
    REGISTRY_SCHEMA_VERSION,
    create_camera_instance,
    create_camera_model,
    create_camera_mode,
    create_registry,
    get_project_root,
    get_schema_version,
    list_camera_instances,
    list_camera_models,
    list_camera_modes,
    open_registry,
    resolve_id_prefix,
    set_project_root,
)
from posetrak.db.import_calib_toml import import_calib_toml
from posetrak.db.import_calib_h5 import import_calib_h5

from posetrak.cli._output import abbrev_id, print_record, print_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_registry(path: str) -> sqlite3.Connection:
    """Open registry or exit with a clear error."""
    try:
        return open_registry(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening registry: {exc}") from exc


def _open_source(obj: dict) -> tuple[sqlite3.Connection, str]:
    """Return (conn, label) — session DB if --session is set, else registry."""
    session = obj.get("session")
    if session:
        from posetrak.db.db import open_session
        try:
            return open_session(Path(session)), "session"
        except (FileNotFoundError, ValueError, sqlite3.DatabaseError) as exc:
            raise click.ClickException(f"Error opening session: {exc}") from exc
    return _open_registry(obj["registry"]), "registry"


def _import_table(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Copy rows from src table into dst using INSERT OR IGNORE.

    Returns (imported, skipped) counts.  Columns present in src but absent
    from dst are silently dropped so schema-version differences don't abort
    the whole import.
    """
    try:
        src_rows = src.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
    except sqlite3.DatabaseError as exc:
        raise click.ClickException(f"Cannot read {table} from session: {exc}") from exc

    if not src_rows:
        return 0, 0

    src_cols = list(src_rows[0].keys())
    dst_cols = {row[1] for row in dst.execute(f"PRAGMA table_info({table})")}
    use_cols = [c for c in src_cols if c in dst_cols]

    placeholders = ", ".join("?" for _ in use_cols)
    col_names = ", ".join(use_cols)
    insert_sql = f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"  # noqa: S608

    imported = skipped = 0
    existing_ids = {r[0] for r in dst.execute(f"SELECT id FROM {table}")}  # noqa: S608
    for row in src_rows:
        if row["id"] in existing_ids:
            skipped += 1
            continue
        if not dry_run:
            dst.execute(insert_sql, tuple(row[c] for c in use_cols))
        imported += 1
    return imported, skipped


def _resolve(conn: sqlite3.Connection, table: str, prefix: str | None) -> str | None:
    """Resolve a UUID prefix; returns None if prefix is None."""
    if prefix is None:
        return None
    try:
        return resolve_id_prefix(conn, table, prefix)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _parse_camera_modes(values: tuple[str, ...]) -> str | dict[str, str]:
    """Parse --camera-mode values into a homogeneous UUID or per-camera dict."""
    values = list(values)
    if len(values) == 1 and "=" not in values[0]:
        return values[0]

    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise click.UsageError(
                f"--camera-mode value {v!r} is ambiguous — "
                "mix of plain UUID and cam=UUID is not allowed. "
                "Use either a single UUID or one 'camN=UUID' per camera."
            )
        key, _, uuid = v.partition("=")
        if not key or not uuid:
            raise click.UsageError(f"Malformed --camera-mode value: {v!r}")
        result[key] = uuid
    return result


def _parse_camera_instances(
    values: tuple[str, ...] | None,
) -> str | dict[str, str] | None:
    """Parse --camera-instance values into homogeneous UUID, per-camera dict, or None."""
    if not values:
        return None
    values = list(values)
    if len(values) == 1 and "=" not in values[0]:
        return values[0]
    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise click.UsageError(f"--camera-instance value {v!r} is ambiguous.")
        key, _, uuid = v.partition("=")
        if not key or not uuid:
            raise click.UsageError(f"Malformed --camera-instance value: {v!r}")
        result[key] = uuid
    return result


# ---------------------------------------------------------------------------
# registry group
# ---------------------------------------------------------------------------


@click.group("registry")
def registry_group() -> None:
    """Manage the registry database."""


@registry_group.command("init")
@click.pass_obj
def registry_init(obj: dict) -> None:
    """Create a new registry database."""
    path = Path(obj["registry"])
    try:
        conn = create_registry(path)
        conn.close()
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Registry created: {path}")


@registry_group.command("info")
@click.pass_obj
def registry_info(obj: dict) -> None:
    """Print registry schema version and settings."""
    registry = _open_registry(obj["registry"])
    try:
        version = get_schema_version(registry)
        project_root = get_project_root(registry)
        n_models = registry.execute("SELECT COUNT(*) FROM camera_models").fetchone()[0]
        n_modes = registry.execute("SELECT COUNT(*) FROM camera_modes").fetchone()[0]
        n_intrinsics = registry.execute(
            "SELECT COUNT(*) FROM intrinsics_calibrations"
        ).fetchone()[0]
    finally:
        registry.close()

    if obj["json_mode"]:
        print_record(
            {
                "registry": obj["registry"],
                "schema_version": version,
                "expected_schema_version": REGISTRY_SCHEMA_VERSION,
                "project_root": str(project_root) if project_root else None,
                "camera_models": n_models,
                "camera_modes": n_modes,
                "intrinsics": n_intrinsics,
            },
            json_mode=True,
        )
    else:
        click.echo(f"Registry: {obj['registry']}")
        click.echo(f"  schema version : {version} (expected {REGISTRY_SCHEMA_VERSION})")
        click.echo(f"  project_root   : {project_root}")
        click.echo(f"  camera models  : {n_models}")
        click.echo(f"  camera modes   : {n_modes}")
        click.echo(f"  intrinsics     : {n_intrinsics}")


# ---------------------------------------------------------------------------
# camera-model group
# ---------------------------------------------------------------------------


@click.group("camera-model")
def camera_model_group() -> None:
    """Manage camera hardware models."""


@camera_model_group.command("add")
@click.option("--manufacturer", default="", metavar="S")
@click.option("--model-name", default="", metavar="S")
@click.option("--sensor-size", default="", metavar="S")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def camera_model_add(
    obj: dict,
    manufacturer: str,
    model_name: str,
    sensor_size: str,
    notes: str,
) -> None:
    """Register a camera hardware model in the registry."""
    registry = _open_registry(obj["registry"])
    try:
        model_id = create_camera_model(
            registry,
            manufacturer=manufacturer,
            model_name=model_name,
            sensor_size=sensor_size or None,
            notes=notes or None,
        )
    finally:
        registry.close()
    click.echo(f"camera_model_id: {model_id}")


@camera_model_group.command("list")
@click.pass_obj
def camera_model_list(obj: dict) -> None:
    """List camera models (registry, or session DB if --session is given)."""
    conn, _src = _open_source(obj)
    try:
        rows = list_camera_models(conn)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No camera models registered.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=["id", "manufacturer", "model_name", "sensor_size"],
        json_mode=obj["json_mode"],
    )


# ---------------------------------------------------------------------------
# camera-mode group
# ---------------------------------------------------------------------------


@click.group("camera-mode")
def camera_mode_group() -> None:
    """Manage camera capture modes."""


@camera_mode_group.command("add")
@click.option("--model-id", required=True, metavar="UUID",
              help="ID of the parent camera_models row")
@click.option("--width", type=int, default=0, metavar="N",
              help="Image width in pixels (default: 0 = unknown)")
@click.option("--height", type=int, default=0, metavar="N",
              help="Image height in pixels (default: 0 = unknown)")
@click.option("--fps", type=float, default=0.0, metavar="F",
              help="Nominal frames per second (default: 0 = unknown)")
@click.option("--codec", default="", metavar="S")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def camera_mode_add(
    obj: dict,
    model_id: str,
    width: int,
    height: int,
    fps: float,
    codec: str,
    notes: str,
) -> None:
    """Register a capture mode (resolution/fps) for a camera model."""
    registry = _open_registry(obj["registry"])
    try:
        mode_id = create_camera_mode(
            registry,
            model_id,
            width_px=width,
            height_px=height,
            nominal_fps=fps,
            codec=codec or None,
            notes=notes or None,
        )
    except sqlite3.IntegrityError as exc:
        registry.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        registry.close()
    click.echo(f"camera_mode_id: {mode_id}")


@camera_mode_group.command("list")
@click.option("--model-id", default="", metavar="UUID",
              help="Filter by camera model ID (or unique prefix)")
@click.pass_obj
def camera_mode_list(obj: dict, model_id: str) -> None:
    """List camera modes (registry, or session DB if --session is given)."""
    conn, _src = _open_source(obj)
    try:
        resolved_model_id = _resolve(conn, "camera_models", model_id or None)
        rows = list_camera_modes(conn, camera_model_id=resolved_model_id)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No camera modes registered.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=["id", "camera_model_id", "width_px", "height_px", "nominal_fps", "codec"],
        json_mode=obj["json_mode"],
    )


# ---------------------------------------------------------------------------
# camera group (camera-instance)
# ---------------------------------------------------------------------------


@click.group("camera")
def camera_group() -> None:
    """Manage physical camera units."""


@camera_group.command("add")
@click.option("--model-id", required=True, metavar="UUID",
              help="camera_models.id (or unique prefix)")
@click.option("--label", required=True, metavar="S",
              help="Human-readable label (e.g. 'cam1')")
@click.option("--serial", default="", metavar="S",
              help="Camera serial number (optional)")
@click.pass_obj
def camera_add(obj: dict, model_id: str, label: str, serial: str) -> None:
    """Register a physical camera unit in the registry."""
    registry = _open_registry(obj["registry"])
    try:
        resolved_model_id = _resolve(registry, "camera_models", model_id)
        instance_id = create_camera_instance(
            registry,
            resolved_model_id,
            label=label,
            serial_number=serial or None,
        )
    except Exception as exc:  # noqa: BLE001
        registry.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        registry.close()
    click.echo(f"camera_instance_id: {instance_id}  label={label!r}")


@camera_group.command("list")
@click.option("--model-id", default="", metavar="UUID",
              help="Filter by camera model ID (or unique prefix)")
@click.pass_obj
def camera_list(obj: dict, model_id: str) -> None:
    """List camera instances (registry, or session DB if --session is given)."""
    conn, _src = _open_source(obj)
    try:
        rows = list_camera_instances(conn, camera_model_id=model_id or None)
        models = {
            r["id"]: f"{r['manufacturer'] or ''} {r['model_name'] or ''}".strip()
            for r in conn.execute("SELECT * FROM camera_models").fetchall()
        }
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No camera instances registered.")
        return

    display_rows = []
    for row in rows:
        d = dict(row)
        d["model"] = models.get(row["camera_model_id"], "")
        display_rows.append(d)

    print_table(
        display_rows,
        columns=["id", "label", "serial_number", "camera_model_id", "model"],
        json_mode=obj["json_mode"],
    )


@camera_group.command("import-session")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be imported without writing anything.")
@click.pass_obj
def camera_import_session(obj: dict, dry_run: bool) -> None:
    """Copy camera data from a session DB into the registry.

    Reads camera_models, camera_modes, camera_instances, and
    intrinsics_calibrations from the session DB (--session required) and
    inserts any rows not already present in the registry.  Existing rows
    (matched by UUID) are left unchanged, so the command is safe to re-run
    and handles the case where a model already exists but some of its modes
    or calibrations are missing.
    """
    session_path = obj.get("session")
    if not session_path:
        raise click.ClickException("--session is required for 'camera import-session'.")

    # Open source read-only to avoid running migrations on a potentially
    # corrupted session DB.
    try:
        src = sqlite3.connect(f"file:{Path(session_path)}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
    except sqlite3.DatabaseError as exc:
        raise click.ClickException(f"Cannot open session: {exc}") from exc

    dst = _open_registry(obj["registry"])

    if dry_run:
        click.echo("Dry run — nothing will be written.")
    else:
        # Disable FK checks for the import so we can insert camera_modes
        # (which reference intrinsics_calibrations) before the calibrations
        # themselves are inserted.
        dst.execute("PRAGMA foreign_keys = OFF")

    tables = [
        "camera_models",
        "camera_modes",
        "camera_instances",
        "intrinsics_calibrations",
    ]
    total_imported = 0
    try:
        for table in tables:
            try:
                n_imp, n_skip = _import_table(src, dst, table, dry_run=dry_run)
            except click.ClickException as exc:
                click.echo(f"  WARNING: {exc.format_message()}", err=True)
                n_imp, n_skip = 0, 0
            verb = "would import" if dry_run else "imported"
            click.echo(
                f"  {table:<30} {n_imp:>4} {verb},  {n_skip:>4} already present"
            )
            total_imported += n_imp
        if not dry_run:
            dst.commit()
            dst.execute("PRAGMA foreign_keys = ON")
    finally:
        src.close()
        dst.close()

    if not dry_run and total_imported == 0:
        click.echo("Nothing to import — registry is already up to date.")


@camera_group.command("show")
@click.argument("instance_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def camera_show(obj: dict, instance_id: str) -> None:
    """Show full details for one camera instance including calibration history."""
    registry = _open_registry(obj["registry"])
    try:
        resolved_id = _resolve(registry, "camera_instances", instance_id)
        inst = registry.execute(
            "SELECT ci.*, cm.manufacturer, cm.model_name, cm.sensor_size "
            "FROM camera_instances ci "
            "JOIN camera_models cm ON cm.id = ci.camera_model_id "
            "WHERE ci.id = ?",
            (resolved_id,),
        ).fetchone()
        if inst is None:
            raise click.ClickException(
                f"Camera instance {resolved_id!r} not found"
            )

        modes = registry.execute(
            "SELECT * FROM camera_modes WHERE camera_model_id = ? ORDER BY rowid",
            (inst["camera_model_id"],),
        ).fetchall()

        calibrations = registry.execute(
            """
            SELECT ic.*, cm.width_px, cm.height_px, cm.nominal_fps
            FROM intrinsics_calibrations ic
            JOIN camera_modes cm ON cm.id = ic.camera_mode_id
            WHERE cm.camera_model_id = ?
            ORDER BY ic.calibrated_at DESC
            """,
            (inst["camera_model_id"],),
        ).fetchall()
    finally:
        registry.close()

    if obj["json_mode"]:
        import json as _json
        print_record(
            {
                "id": inst["id"],
                "label": inst["label"],
                "serial_number": inst["serial_number"],
                "camera_model_id": inst["camera_model_id"],
                "manufacturer": inst["manufacturer"],
                "model_name": inst["model_name"],
                "sensor_size": inst["sensor_size"],
                "modes": [dict(m) for m in modes],
                "calibrations": [dict(c) for c in calibrations],
            },
            json_mode=True,
        )
        return

    manufacturer = inst["manufacturer"] or ""
    model_name = inst["model_name"] or ""
    sensor = f"  sensor={inst['sensor_size']}" if inst["sensor_size"] else ""

    click.echo(f"Instance:  {inst['id']}")
    click.echo(f"Label:     {inst['label']}")
    click.echo(f"Serial:    {inst['serial_number'] or '(none)'}")
    click.echo(f"Model:     {manufacturer} {model_name}{sensor}  [{inst['camera_model_id']}]")

    click.echo(f"\nCapture modes ({len(modes)}):")
    for mode in modes:
        fps = f"{mode['nominal_fps']:.3g}" if mode["nominal_fps"] else "?"
        res = (
            f"{mode['width_px']}x{mode['height_px']}"
            if mode["width_px"] and mode["height_px"]
            else "?x?"
        )
        codec = f"  {mode['codec']}" if mode["codec"] else ""
        click.echo(f"  {mode['id']}  {res} @ {fps} fps{codec}")

    click.echo(f"\nIntrinsics calibrations ({len(calibrations)}):")
    if not calibrations:
        click.echo("  (none)")
    else:
        click.echo(f"  {'id':<36}  {'date':<12}  {'mode':<16}  {'rms':>6}  maps  tool")
        click.echo("  " + "-" * 95)
        for cal in calibrations:
            rms = f"{cal['rms_error']:.4f}" if cal["rms_error"] is not None else "     ?"
            has_maps = "yes" if cal["undistort_mapx"] else " no"
            res = f"{cal['width_px']}x{cal['height_px']}" if cal["width_px"] else "?x?"
            fps = f"{cal['nominal_fps']:.3g}" if cal["nominal_fps"] else "?"
            mode_desc = f"{res}@{fps}"
            tool = cal["calibration_tool"] or ""
            click.echo(
                f"  {cal['id']}  {cal['calibrated_at']:<12}  {mode_desc:<16}  {rms}  {has_maps}   {tool}"
            )


# ---------------------------------------------------------------------------
# calib group
# ---------------------------------------------------------------------------


@click.group("calib")
def calib_group() -> None:
    """Manage intrinsic calibrations."""


@calib_group.command("import")
@click.option("--calib", required=True, metavar="TOML_PATH",
              help="Path to the Pose2Sim calibration TOML file")
@click.option("--camera-mode", required=True, multiple=True, metavar="SPEC",
              help=(
                  "Camera mode assignment. Two forms: "
                  "(1) a single UUID applies to all cameras in the file; "
                  "(2) one or more 'camN=UUID' pairs for per-camera assignment."
              ))
@click.option("--tool", default="pose2sim", metavar="S",
              help="Calibration tool name (default: pose2sim)")
@click.option("--distortion-model", default="radtan", metavar="S",
              help="Distortion model (default: radtan)")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def calib_import(
    obj: dict,
    calib: str,
    camera_mode: tuple[str, ...],
    tool: str,
    distortion_model: str,
    notes: str,
) -> None:
    """Import intrinsic calibration from a Pose2Sim TOML file."""
    camera_modes = _parse_camera_modes(camera_mode)
    registry = _open_registry(obj["registry"])
    try:
        result = import_calib_toml(
            registry,
            Path(calib),
            camera_modes,
            calibration_tool=tool,
            distortion_model=distortion_model,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001
        registry.close()
        raise click.ClickException(f"Error importing calibration: {exc}") from exc
    finally:
        registry.close()

    n = len(result.camera_instance_ids)
    mode_desc = camera_modes if isinstance(camera_modes, str) else "per-camera"
    click.echo(f"Imported {Path(calib).name}: {n} camera(s)  mode={mode_desc}")
    for label, iid in result.camera_instance_ids.items():
        intr_id = result.intrinsics_ids[label]
        click.echo(f"  {label!r}  instance={iid}  intrinsics={intr_id}")
    if result.skipped:
        click.echo(f"  skipped: {', '.join(sorted(result.skipped))}")


@calib_group.command("import-h5")
@click.argument("h5_file", metavar="H5_FILE")
@click.option("--camera-mode", required=True, metavar="UUID",
              help="camera_modes.id (or unique prefix) to associate with this calibration")
@click.option("--camera-instance", default=None, metavar="UUID",
              help="Optional camera_instances.id for the notes field")
@click.option("--no-maps", is_flag=True, default=False,
              help="Skip storing undistortion maps (saves ~3 MB per camera)")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def calib_import_h5(
    obj: dict,
    h5_file: str,
    camera_mode: str,
    camera_instance: str | None,
    no_maps: bool,
    notes: str,
) -> None:
    """Import intrinsic calibration from an HDF5 file."""
    registry = _open_registry(obj["registry"])
    try:
        camera_mode_id = _resolve(registry, "camera_modes", camera_mode)
        result = import_calib_h5(
            registry,
            Path(h5_file),
            camera_mode_id,
            camera_instance_id=camera_instance or None,
            store_maps=not no_maps,
            notes=notes or "",
        )
    except Exception as exc:  # noqa: BLE001
        registry.close()
        raise click.ClickException(f"Error importing HDF5 calibration: {exc}") from exc
    finally:
        registry.close()

    click.echo(f"intrinsics_id: {result.intrinsics_id}")
    if result.camera_name:
        click.echo(f"camera_name: {result.camera_name}")


@calib_group.command("list")
@click.pass_obj
def calib_list(obj: dict) -> None:
    """List camera instances and their intrinsics calibrations."""
    # Prefer session DB if provided, else registry.
    session_path = obj.get("session")
    if session_path:
        from posetrak.db.db import open_session
        try:
            conn = open_session(Path(session_path))
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        conn = _open_registry(obj["registry"])

    try:
        rows = conn.execute(
            "SELECT ci.id, ci.label, ic.id, ic.calibrated_at, ic.fx, ic.fy, ic.cx, ic.cy"
            " FROM camera_instances ci"
            " JOIN intrinsics_calibrations ic ON ic.camera_mode_id IN"
            "   (SELECT id FROM camera_modes WHERE camera_model_id = ci.camera_model_id)"
            " ORDER BY ci.label, ic.calibrated_at"
        ).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("(no calibrations found)")
        return

    display_rows = [
        {
            "camera_key": r[1],
            "instance_id": r[0],
            "intrinsics_id": r[2],
            "calibrated_at": r[3],
            "fx": f"{r[4]:.1f}" if r[4] is not None else "",
            "fy": f"{r[5]:.1f}" if r[5] is not None else "",
        }
        for r in rows
    ]
    if obj["json_mode"]:
        for row in display_rows:
            import json as _json
            sys.stdout.write(_json.dumps(row) + "\n")
    else:
        print_table(
            display_rows,
            columns=["camera_key", "instance_id", "intrinsics_id", "calibrated_at", "fx", "fy"],
            json_mode=False,
        )


# ---------------------------------------------------------------------------
# Wire camera-* sub-groups into the registry group so they appear as
# top-level commands in main.py
# ---------------------------------------------------------------------------
# These are exported and registered separately in main.py as top-level groups.
