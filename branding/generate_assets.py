# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the installer/docs assets derived from the Posetrak logo.

The logo artwork itself (posetrak-logo-abstract.png, posetrak-logo-aikido.png
in this directory) is Nelli Kaimio's -- see REUSE.toml. Run this script after
replacing either master file to regenerate every derived asset consistently.

Usage (from the repo root):
    uv run python branding/generate_assets.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANDING = REPO_ROOT / "branding"
ABSTRACT = BRANDING / "posetrak-logo-abstract.png"
AIKIDO = BRANDING / "posetrak-logo-aikido.png"

WIN_ASSETS = REPO_ROOT / "packaging" / "windows" / "assets"
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"


def fit_and_pad(im: Image.Image, size: tuple[int, int], bg: tuple[int, int, int, int]) -> Image.Image:
    """Resize *im* to fit within *size* preserving aspect ratio (no crop, no
    distortion), then center it on a *bg*-filled canvas of exactly *size*."""
    im = im.copy()
    im.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGBA", size, bg)
    x = (size[0] - im.width) // 2
    y = (size[1] - im.height) // 2
    canvas.paste(im, (x, y), im if im.mode == "RGBA" else None)
    return canvas


def main() -> None:
    WIN_ASSETS.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)

    abstract = Image.open(ABSTRACT).convert("RGBA")
    aikido = Image.open(AIKIDO).convert("RGBA")

    # Installer icon (.ico, multi-resolution, keeps transparency).
    ico_path = WIN_ASSETS / "posetrak-icon.ico"
    abstract.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
    print("wrote", ico_path)

    # Installer wizard small image (top-right, every page). Inno's
    # WizardSmallImageFile wants a BMP; flatten onto white (the modern
    # wizard style's own background) rather than leaving alpha in a BMP.
    small = fit_and_pad(abstract, (55, 58), (255, 255, 255, 255)).convert("RGB")
    small_path = WIN_ASSETS / "wizard-small.bmp"
    small.save(small_path)
    print("wrote", small_path, small.size)

    # Installer wizard banner (left side of Welcome/Finished pages).
    # posetrak-logo-aikido.png is already opaque white to its own edges, so
    # padding with white to reach the tall/narrow target aspect blends in
    # rather than adding a visible border -- preserves the full image with
    # no crop or distortion.
    banner = fit_and_pad(aikido, (164, 314), (255, 255, 255, 255)).convert("RGB")
    banner_path = WIN_ASSETS / "wizard-banner.bmp"
    banner.save(banner_path)
    print("wrote", banner_path, banner.size)

    # Docs site header logo (keeps transparency; Material scales via CSS).
    logo = abstract.copy()
    logo.thumbnail((256, 256), Image.LANCZOS)
    logo_path = DOCS_ASSETS / "logo.png"
    logo.save(logo_path)
    print("wrote", logo_path, logo.size)

    # Docs site favicon.
    favicon = abstract.copy()
    favicon.thumbnail((48, 48), Image.LANCZOS)
    favicon_path = DOCS_ASSETS / "favicon.png"
    favicon.save(favicon_path)
    print("wrote", favicon_path, favicon.size)

    # Docs home page banner (docs/index.md).
    banner_docs = aikido.copy()
    banner_docs.thumbnail((480, 480), Image.LANCZOS)
    banner_docs_path = DOCS_ASSETS / "banner.png"
    banner_docs.save(banner_docs_path)
    print("wrote", banner_docs_path, banner_docs.size)


if __name__ == "__main__":
    main()
