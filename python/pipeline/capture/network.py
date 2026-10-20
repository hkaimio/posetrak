# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Finding the address phones should use, and showing it as a QR code.

Phones on the capture WLAN reach the daemon by IP address. The ``.local`` name
is a convenience that Android browsers often cannot resolve, so the QR code
carries the address.
"""

from __future__ import annotations

import ipaddress
import socket


def lan_addresses() -> list[str]:
    """IPv4 addresses this machine could be reached on, most likely first.

    The first entry is the address of the interface the operating system would
    use to reach an arbitrary outside host (the WLAN, normally); loopback and
    link-local addresses are left out.
    """
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # a documentation address; nothing is sent
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    seen: list[str] = []
    for addr in found:
        if addr.startswith(("127.", "169.254.", "0.")) or addr in seen:
            continue
        seen.append(addr)
    return seen


def access_url(host: str, port: int, token: str) -> str:
    """URL that opens the web UI and hands it the access token in the fragment.

    A fragment is never sent to the server or written to access logs.
    """
    return f"http://{host}:{port}/#token={token}"


def qr_text(url: str, *, ascii_only: bool = False) -> str:
    """Render *url* as a QR code for a terminal.

    The default uses half-block characters, two QR rows per text line. With
    ``ascii_only`` (for consoles that cannot encode them) each module is two
    plain characters wide. Light modules are drawn as filled cells, so the code
    is scannable on a dark terminal theme.
    """
    import qrcode

    qr = qrcode.QRCode(border=2)
    qr.add_data(url)
    qr.make(fit=True)
    matrix = qr.get_matrix()  # True = dark module
    if ascii_only:
        return "\n".join("".join("  " if dark else "##" for dark in row) for row in matrix)
    if len(matrix) % 2:
        matrix.append([False] * len(matrix[0]))
    blocks = {(True, True): " ", (True, False): "▄", (False, True): "▀", (False, False): "█"}
    return "\n".join(
        "".join(blocks[(t, b)] for t, b in zip(top, bottom)) for top, bottom in zip(matrix[::2], matrix[1::2])
    )


def route_warning(camera_ip: str) -> str | None:
    """Explain a route that would keep traffic to a camera off the WLAN it is on.

    A VPN that routes the camera's subnet (the same private range as the
    capture WLAN) into a tunnel makes every request time out. When this
    machine has an address in the camera's /24 but the operating system would
    send traffic to the camera from a different one, return a description;
    otherwise None. A camera on a network this machine is not on gives no
    warning: it is simply not reachable from here.
    """
    try:
        camera = ipaddress.ip_address(camera_ip)
        subnet = ipaddress.ip_network(f"{camera}/24", strict=False)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((camera_ip, 9))
            chosen = ipaddress.ip_address(s.getsockname()[0])
    except (ValueError, OSError):
        return None
    on_subnet = any(ipaddress.ip_address(a) in subnet for a in lan_addresses())
    if on_subnet and chosen not in subnet:
        return (
            f"traffic to {camera_ip} would leave from {chosen}, not from this machine's address on {subnet}; "
            "a VPN or another route may be shadowing the capture WLAN"
        )
    return None
