# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""gopro_cohn_probe.py — hardware probe for controlling Hero13 cameras over COHN.

Answers the open questions of the camera-control Phase 1 plan (WP0) on real
cameras:

  * Can a camera hold COHN credentials for several WiFi networks, and join
    whichever is in range?  (`networks`, `provision`, `join`)
  * How is a camera found on the WLAN, and how long do commands take?
    (`state`, `record`)
  * Does a COHN link survive an hour of recording at a mocap preset?  (`soak`)
  * Can BLE wake a camera that is powered off?  (`wake`)
  * Which settings/presets exist for the mocap profile?  (`dump`)

Everything is kept under --db-dir (default ./gopro_probe): per-camera COHN
credentials keyed by (camera, SSID) in <id>.json, and CSV/JSON results.  The
credentials file contains camera passwords; do not commit it.

Cameras are addressed by the last four digits of their serial number, the
same identifier as in the camera's BLE name ("GoPro 1234").

Install into a scratch environment:   uv pip install open-gopro bleak httpx
Examples:
  python gopro_cohn_probe.py scan
  python gopro_cohn_probe.py networks 1234
  python gopro_cohn_probe.py provision 1234 --ssid HomeNet --password ...
  python gopro_cohn_probe.py join 1234 --ssid HomeNet
  python gopro_cohn_probe.py record 1234 5678 --ssid HomeNet --seconds 10
  python gopro_cohn_probe.py soak 1234 5678 --ssid HomeNet --minutes 60
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import ssl
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# GoPro status ids used here (Open GoPro "Camera Statuses").
ST_SYSTEM_HOT = "6"
ST_ENCODING = "10"
ST_SD_REMAINING_KB = "54"
ST_BATTERY_PERCENT = "70"

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Credential store: <db_dir>/<camera_id>.json  ->  {ssid: {ip, username, ...}}
# ---------------------------------------------------------------------------


def load_creds(db_dir: Path, camera_id: str) -> dict[str, dict]:
    path = db_dir / f"{camera_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_cred(db_dir: Path, camera_id: str, ssid: str, cred: dict) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    creds = load_creds(db_dir, camera_id)
    creds[ssid] = cred
    (db_dir / f"{camera_id}.json").write_text(json.dumps(creds, indent=2))


# ---------------------------------------------------------------------------
# BLE side (SDK): scan, inspect, provision, join, wake
# ---------------------------------------------------------------------------


async def cmd_scan(args: argparse.Namespace) -> None:
    from bleak import BleakScanner

    log(f"scanning BLE for {args.seconds:.0f} s ...")
    found = await BleakScanner.discover(timeout=args.seconds, return_adv=True)
    rows = [
        (adv.rssi, dev.name, dev.address)
        for dev, adv in found.values()
        if dev.name and dev.name.startswith("GoPro")
    ]
    for rssi, name, address in sorted(rows, reverse=True):
        print(f"  {name:<14} {address}  RSSI {rssi} dBm")
    if not rows:
        print("  no GoPro advertising (cameras need wireless connections enabled)")


def _null_wifi_adapter():
    """A WiFi adapter that never touches the host's WiFi.

    The SDK builds a host WiFi controller even for BLE-only use, and on
    Windows closing the camera object runs `netsh wlan disconnect`, which drops
    the PC from its WLAN (and the camera's COHN link with it) at the end of
    every BLE command.  This stand-in does nothing.
    """
    from open_gopro.network.wifi.controller import SsidState, WifiController

    class NullWifi(WifiController):
        def __init__(self, interface=None, password=None) -> None:
            super().__init__(interface, password)

        async def connect(self, ssid, password, timeout=15) -> bool:
            return False

        async def disconnect(self) -> bool:
            return True

        def current(self):
            return None, SsidState.DISCONNECTED

        def available_interfaces(self) -> list[str]:
            return []

        def power(self, power: bool) -> bool:
            return False

        @property
        def is_on(self) -> bool:
            return False

    return NullWifi


def _open_ble(camera_id: str, db_dir: Path):
    from open_gopro import WirelessGoPro

    db_dir.mkdir(parents=True, exist_ok=True)
    return WirelessGoPro(
        target=camera_id,
        interfaces={WirelessGoPro.Interface.BLE},
        cohn_db=db_dir / f"sdk_{camera_id}.json",
        wifi_adapter=_null_wifi_adapter(),
    )


def _flags(flags: int) -> str:
    from open_gopro.models.proto import EnumScanEntryFlags as F

    names = [
        ("CONFIGURED", F.SCAN_FLAG_CONFIGURED),
        ("ASSOCIATED", F.SCAN_FLAG_ASSOCIATED),
        ("BEST", F.SCAN_FLAG_BEST_SSID),
        ("OPEN", F.SCAN_FLAG_OPEN),
    ]
    return ",".join(n for n, f in names if flags & f) or "-"


async def cmd_networks(args: argparse.Namespace) -> None:
    """List networks the camera sees and which of them it has provisioned."""
    async with _open_ble(args.camera, args.db_dir) as gopro:
        await gopro.cohn.wait_until_ready()
        st = gopro.cohn.status
        log(
            f"COHN: status={st.status} state={st.state} enabled={st.enabled} "
            f"ssid={st.ssid!r} ip={st.ipaddress!r} mac={st.macaddress}"
        )
        res = await gopro.access_point.scan_wifi_networks()
        for e in res.unwrap().entries:
            print(
                f"  {e.ssid:<28} {e.signal_frequency_mhz} MHz "
                f"bars={e.signal_strength_bars} flags={_flags(e.scan_entry_flags)}"
            )


def _save_current_cohn(args: argparse.Namespace, gopro, ssid: str) -> None:
    creds = gopro.cohn.credentials
    if creds is None or not creds.is_complete:
        log("no complete COHN credentials to save")
        return
    save_cred(
        args.db_dir,
        args.camera,
        ssid,
        {
            "ip_address": creds.ip_address,
            "username": creds.username,
            "password": creds.password,
            "certificate": creds.certificate,
        },
    )
    log(f"saved credentials for {args.camera} on {ssid!r} (ip {creds.ip_address})")


async def cmd_provision(args: argparse.Namespace) -> None:
    """Join a new network and (re)provision COHN on it.

    Provisioning clears the camera's COHN certificate first.  To learn whether
    that invalidates the credentials of another network, run `state` against
    the previously provisioned SSID afterwards.
    """
    async with _open_ble(args.camera, args.db_dir) as gopro:
        await gopro.cohn.wait_until_ready()
        t0 = time.monotonic()
        res = await gopro.access_point.connect(args.ssid, args.password)
        log(f"camera joined {args.ssid!r}: {res} ({time.monotonic() - t0:.1f} s)")
        t0 = time.monotonic()
        res = await gopro.cohn.configure(force_reprovision=args.force)
        log(f"COHN configure: {res} ({time.monotonic() - t0:.1f} s)")
        st = gopro.cohn.status
        log(f"COHN status: state={st.state} ssid={st.ssid!r} ip={st.ipaddress!r}")
        _save_current_cohn(args, gopro, args.ssid)


async def cmd_join(args: argparse.Namespace) -> None:
    """Ask the camera (over BLE) to join an already-provisioned network."""
    async with _open_ble(args.camera, args.db_dir) as gopro:
        await gopro.cohn.wait_until_ready()
        t0 = time.monotonic()
        res = await gopro.access_point.connect(args.ssid, "")
        log(f"join {args.ssid!r}: {res} ({time.monotonic() - t0:.1f} s)")
        await asyncio.sleep(2)
        st = gopro.cohn.status
        log(
            f"COHN status: status={st.status} state={st.state} ssid={st.ssid!r} "
            f"ip={st.ipaddress!r} username={st.username!r}"
        )
        creds = load_creds(args.db_dir, args.camera).get(args.ssid)
        if creds:
            same = (creds["username"], creds["password"]) == (st.username, st.password)
            log(f"stored credentials for this SSID match camera: {same}")
            log(f"stored ip {creds['ip_address']} vs camera-reported {st.ipaddress}")
        else:
            log("no stored credentials for this SSID")


async def cmd_wake(args: argparse.Namespace) -> None:
    """Test whether a powered-off camera can be woken over BLE.

    Power the camera off with its button first (or pass --power-down to do it
    over BLE), then this reconnects over BLE and waits for COHN to answer.
    """
    if args.power_down:
        async with _open_ble(args.camera, args.db_dir) as gopro:
            await gopro.ble_command.power_down()
            log("power_down sent")
        await asyncio.sleep(args.settle)
    log("trying BLE connect (this is the wake attempt) ...")
    t0 = time.monotonic()
    async with _open_ble(args.camera, args.db_dir) as gopro:
        log(f"BLE connected after {time.monotonic() - t0:.1f} s")
        await gopro.cohn.wait_until_ready()
        st = gopro.cohn.status
        log(f"COHN: state={st.state} ssid={st.ssid!r} ip={st.ipaddress!r}")
        cred = load_creds(args.db_dir, args.camera).get(st.ssid or args.ssid)
        if not cred:
            return
    cam = Cam(args.camera, args.ssid or st.ssid, cred)
    for _ in range(60):
        ok, _, _ = await cam.get_state()
        if ok:
            log(f"COHN reachable {time.monotonic() - t0:.1f} s after BLE connect")
            return
        await asyncio.sleep(1)
    log("COHN did not become reachable within 60 s")


# ---------------------------------------------------------------------------
# COHN side (plain HTTPS): state, record, soak, dump
# ---------------------------------------------------------------------------


@dataclass
class Cam:
    """One camera reached over COHN."""

    camera_id: str
    ssid: str
    cred: dict

    def __post_init__(self) -> None:
        import httpx

        # The camera's certificate has no authority key identifier and the
        # name does not match its IP address, so verify against the pinned
        # certificate only, without hostname checking or X.509 strict mode.
        ctx = ssl.create_default_context(cadata=self.cred["certificate"])
        ctx.check_hostname = False
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        ctx.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
        # httpx, not requests: urllib3 requests are closed by the camera
        # after ~2 s (see the camera-control analysis doc); httpx and
        # http.client are answered in ~0.1-0.3 s.
        self.client = httpx.AsyncClient(
            base_url=f"https://{self.cred['ip_address']}",
            verify=ctx,
            headers={"Authorization": _basic(self.cred)},
        )
        self.link_down_since: float | None = None
        self.drops: list[float] = []

    async def request(self, path: str, timeout: float = 5.0) -> tuple[bool, float, dict | None]:
        t0 = time.monotonic()
        try:
            r = await self.client.get(path, timeout=timeout)
            r.raise_for_status()
            data = r.json() if r.content else {}
            ok = True
        except Exception as exc:  # noqa: BLE001  (probe: record any failure)
            data, ok = {"error": repr(exc)[:200]}, False
        dt = time.monotonic() - t0
        now = time.monotonic()
        if ok and self.link_down_since is not None:
            self.drops.append(now - self.link_down_since)
            log(f"{self.camera_id}: link back after {self.drops[-1]:.1f} s")
            self.link_down_since = None
        elif not ok and self.link_down_since is None:
            self.link_down_since = now - dt
            log(f"{self.camera_id}: link lost: {data}")
        return ok, dt, data

    async def get_state(self) -> tuple[bool, float, dict | None]:
        return await self.request("/gopro/camera/state", timeout=3.0)

    async def shutter(self, on: bool) -> tuple[bool, float, dict | None]:
        return await self.request(f"/gopro/camera/shutter/{'start' if on else 'stop'}")

    async def wait_encoding(self, want: bool, timeout: float = 10.0) -> float | None:
        """Poll until the encoding flag equals `want`; return elapsed seconds."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            ok, _, data = await self.get_state()
            if ok and bool(data["status"].get(ST_ENCODING)) == want:
                return time.monotonic() - t0
            await asyncio.sleep(0.1)
        return None


def _basic(cred: dict) -> str:
    from base64 import b64encode

    return "Basic " + b64encode(f"{cred['username']}:{cred['password']}".encode()).decode()


def _cams(args: argparse.Namespace) -> list[Cam]:
    cams = []
    for cid in args.cameras:
        cred = load_creds(args.db_dir, cid).get(args.ssid)
        if not cred:
            sys.exit(f"no stored credentials for {cid} on {args.ssid!r}; run provision")
        cams.append(Cam(cid, args.ssid, cred))
    return cams


async def cmd_state(args: argparse.Namespace) -> None:
    cams = _cams(args)
    for _ in range(args.repeat):
        results = await asyncio.gather(*(c.get_state() for c in cams))
        for c, (ok, dt, data) in zip(cams, results):
            if ok:
                s = data["status"]
                print(
                    f"  {c.camera_id} {dt * 1000:5.0f} ms  enc={s.get(ST_ENCODING)} "
                    f"batt={s.get(ST_BATTERY_PERCENT)}% hot={s.get(ST_SYSTEM_HOT)} "
                    f"sd={s.get(ST_SD_REMAINING_KB)} kB"
                )
            else:
                print(f"  {c.camera_id} FAILED after {dt * 1000:.0f} ms: {data}")
        await asyncio.sleep(args.interval)


async def _fan(cams: list[Cam], on: bool) -> dict[str, dict]:
    """Send shutter to all cameras at once; time command and confirmation."""
    t_send = time.monotonic()

    async def one(c: Cam) -> tuple[str, dict]:
        ok, dt, data = await c.shutter(on)
        t_ack = time.monotonic() - t_send
        confirm = await c.wait_encoding(on) if ok else None
        return c.camera_id, {
            "cmd_ok": ok,
            "cmd_latency_s": round(dt, 3),
            "ack_s": round(t_ack, 3),
            "confirm_s": None if confirm is None else round(time.monotonic() - t_send, 3),
        }

    return dict(await asyncio.gather(*(one(c) for c in cams)))


def _print_fan(label: str, res: dict[str, dict]) -> None:
    log(label)
    for cid, r in res.items():
        print(f"  {cid}: {r}")
    confirms = [r["confirm_s"] for r in res.values() if r["confirm_s"] is not None]
    if len(confirms) > 1:
        print(f"  confirm spread: {max(confirms) - min(confirms):.3f} s")


async def cmd_record(args: argparse.Namespace) -> None:
    cams = _cams(args)
    _print_fan("START", await _fan(cams, True))
    await asyncio.sleep(args.seconds)
    _print_fan("STOP", await _fan(cams, False))


async def cmd_soak(args: argparse.Namespace) -> None:
    cams = _cams(args)
    args.db_dir.mkdir(parents=True, exist_ok=True)
    out = args.db_dir / f"soak_{datetime.now():%Y%m%d_%H%M%S}.csv"
    _print_fan("START", await _fan(cams, True))
    end = time.monotonic() + args.minutes * 60
    latencies: dict[str, list[float]] = {c.camera_id: [] for c in cams}
    fails: dict[str, int] = {c.camera_id: 0 for c in cams}
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "camera", "ok", "latency_ms", "encoding", "battery_pct", "hot", "sd_kb"])
        try:
            while time.monotonic() < end:
                results = await asyncio.gather(*(c.get_state() for c in cams))
                for c, (ok, dt, data) in zip(cams, results):
                    s = data["status"] if ok else {}
                    w.writerow(
                        [
                            datetime.now().isoformat(timespec="seconds"),
                            c.camera_id,
                            int(ok),
                            round(dt * 1000),
                            s.get(ST_ENCODING),
                            s.get(ST_BATTERY_PERCENT),
                            s.get(ST_SYSTEM_HOT),
                            s.get(ST_SD_REMAINING_KB),
                        ]
                    )
                    if ok:
                        latencies[c.camera_id].append(dt * 1000)
                        if not s.get(ST_ENCODING):
                            log(f"{c.camera_id}: NOT ENCODING (stopped on its own)")
                        if s.get(ST_SYSTEM_HOT):
                            log(f"{c.camera_id}: SYSTEM HOT")
                    else:
                        fails[c.camera_id] += 1
                f.flush()
                await asyncio.sleep(args.interval)
        except KeyboardInterrupt:
            log("interrupted; stopping")
    _print_fan("STOP", await _fan(cams, False))
    print("\nsummary")
    for c in cams:
        lat = latencies[c.camera_id]
        med = f"{statistics.median(lat):.0f}" if lat else "-"
        p99 = f"{sorted(lat)[int(len(lat) * 0.99) - 1]:.0f}" if len(lat) > 10 else "-"
        print(
            f"  {c.camera_id}: polls ok={len(lat)} failed={fails[c.camera_id]} "
            f"latency median={med} ms p99={p99} ms  link drops={len(c.drops)} "
            f"(longest {max(c.drops, default=0):.1f} s)"
        )
    print(f"log: {out}")


async def cmd_dump(args: argparse.Namespace) -> None:
    """Dump full state and presets to JSON (for the settings matrix)."""
    cam = _cams(args)[0]
    state = await cam.get_state()
    presets = await cam.request("/gopro/camera/presets/get")
    path = args.db_dir / f"dump_{cam.camera_id}.json"
    path.write_text(json.dumps({"state": state[2], "presets": presets[2]}, indent=2))
    log(f"wrote {path}")


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-dir", type=Path, default=Path("gopro_probe"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--seconds", type=float, default=8)
    s.set_defaults(fn=cmd_scan)

    for name, fn in (("networks", cmd_networks),):
        s = sub.add_parser(name)
        s.add_argument("camera")
        s.set_defaults(fn=fn)

    s = sub.add_parser("provision")
    s.add_argument("camera")
    s.add_argument("--ssid", required=True)
    s.add_argument("--password", required=True)
    s.add_argument("--force", action="store_true", help="force COHN re-provisioning even if credentials exist")
    s.set_defaults(fn=cmd_provision)

    s = sub.add_parser("join")
    s.add_argument("camera")
    s.add_argument("--ssid", required=True)
    s.set_defaults(fn=cmd_join)

    s = sub.add_parser("wake")
    s.add_argument("camera")
    s.add_argument("--ssid", default="")
    s.add_argument("--power-down", action="store_true")
    s.add_argument("--settle", type=float, default=10)
    s.set_defaults(fn=cmd_wake)

    def with_cameras(name, fn, **extra):
        s = sub.add_parser(name)
        s.add_argument("cameras", nargs="+")
        s.add_argument("--ssid", required=True)
        for k, v in extra.items():
            s.add_argument(f"--{k}", type=type(v), default=v)
        s.set_defaults(fn=fn)

    with_cameras("state", cmd_state, repeat=1, interval=1.0)
    with_cameras("record", cmd_record, seconds=10.0)
    with_cameras("soak", cmd_soak, minutes=60.0, interval=5.0)
    with_cameras("dump", cmd_dump)

    args = p.parse_args()
    asyncio.run(args.fn(args))


if __name__ == "__main__":
    main()
