# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Translating a capture profile into Open GoPro video settings.

A capture profile names settings symbolically (``resolution = "4K"``,
``fps = 120``, ``lens = "linear"``) so that one profile can describe the same
look on different camera types. This module holds the Hero13 mapping to the
numeric setting ids and option values of the Open GoPro API, and the mode the
settings only make sense in.

A key that is a plain number (``"108" = 1``) is passed through as a raw
setting id and option value, for settings without a symbolic name.

The settings are applied in a fixed order because the options a camera offers
depend on the earlier ones: the frame rates available depend on the
resolution, and the lenses on both.
"""

from __future__ import annotations

from typing import Any, Final

#: ``camera/state`` status ids
ST_OVERHEATING: Final = "6"
ST_ENCODING: Final = "10"
ST_REMAINING_VIDEO_S: Final = "35"
ST_SD_REMAINING_KB: Final = "54"
ST_BATTERY_PERCENT: Final = "70"
ST_FLAT_MODE: Final = "89"

#: ``status[ST_FLAT_MODE]`` value of the video mode; a time lapse or photo mode
#: accepts the shutter command and records the wrong thing.
FLAT_MODE_VIDEO: Final = 12
#: Preset group that selects the video mode (``presets/set_group``).
PRESET_GROUP_VIDEO: Final = 1000

_RESOLUTION: Final[dict[str, int]] = {
    "5.3K": 100,
    "5K": 24,
    "4K": 1,
    "2.7K": 4,
    "1440": 7,
    "1080": 9,
    "720": 12,
}
_FPS: Final[dict[int, int]] = {240: 0, 120: 1, 100: 2, 90: 3, 60: 5, 50: 6, 30: 8, 25: 9, 24: 10, 200: 13}
_LENS: Final[dict[str, int]] = {
    "wide": 0,
    "narrow": 2,
    "superview": 3,
    "linear": 4,
    "max_superview": 7,
    "linear_horizon_leveling": 8,
    "hyperview": 9,
    "linear_horizon_lock": 10,
    "max_hyperview": 11,
}
_BIT_RATE: Final[dict[str, int]] = {"standard": 0, "high": 1}
_PERFORMANCE: Final[dict[str, int]] = {"max": 0, "extended_battery": 1, "tripod": 2}

#: symbolic name -> (setting id, option lookup), in the order the settings are applied
_SYMBOLIC: Final[dict[str, tuple[int, dict[Any, int]]]] = {
    "resolution": (2, _RESOLUTION),
    "fps": (3, _FPS),
    "lens": (121, _LENS),
    "bit_rate": (182, _BIT_RATE),
    "performance": (173, _PERFORMANCE),
}


class ProfileError(ValueError):
    """The profile asks for something this camera type cannot do."""


def _normalise(name: str, value: Any) -> Any:
    if name == "fps":
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if name == "resolution":
        return str(value).strip().upper()
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def translate(settings: dict[str, Any]) -> dict[int, int]:
    """Map profile settings to ``{setting id: option value}`` in application order.

    Raises
    ------
    ProfileError
        On a name or value the Hero13 mapping does not know.
    """
    wanted: dict[int, int] = {}
    raw: dict[int, int] = {}
    for name, value in settings.items():
        if name.isdigit():
            raw[int(name)] = int(value)
        elif name not in _SYMBOLIC:
            raise ProfileError(f"unknown profile setting {name!r} (known: {', '.join(_SYMBOLIC)}, or a numeric setting id)")
    for name, (setting_id, options) in _SYMBOLIC.items():
        if name not in settings:
            continue
        key = _normalise(name, settings[name])
        if key not in options:
            raise ProfileError(f"unknown value {settings[name]!r} for {name} (known: {', '.join(map(str, options))})")
        wanted[setting_id] = options[key]
    wanted.update(raw)
    return wanted


def label(setting_id: int, option: Any) -> tuple[str, str]:
    """Human-readable ``(name, value)`` of a setting id and option, for messages."""
    for name, (sid, options) in _SYMBOLIC.items():
        if sid == setting_id:
            for text, value in options.items():
                if value == option:
                    return name, str(text)
            return name, f"option {option}"
    return f"setting {setting_id}", f"option {option}"
