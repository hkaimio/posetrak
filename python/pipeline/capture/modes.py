# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Choosing a capture mode for every camera.

A capture mode is a named registry ``camera_modes`` row of the camera's model.
Which mode a camera records in comes from a *setup* in the configuration: an
entry for the camera's model, optionally overridden for one camera, and the
operator may override either for a single capture. The settings that make the
camera record in the mode come from its driver, with the setup's per-camera
settings applied on top.

Without any setup the configured profile's settings go to every camera, as
before modes existed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from pipeline.capture.config import SetupConfig
from pipeline.capture.devices import CameraDriver, ModeSpec


class ModeError(ValueError):
    """The requested setup or mode choice cannot be used."""


@dataclass(frozen=True)
class ModeChoice:
    """The mode one camera is armed with."""

    #: registry ``camera_modes.id``; None when the settings are the profile's
    mode_id: str | None
    name: str | None
    settings: dict[str, Any] = field(default_factory=dict)
    #: why no mode could be chosen for the camera (it is then not ready)
    problem: str | None = None


@dataclass(frozen=True)
class PlannedCamera:
    """What the planner needs to know about one camera."""

    label: str
    #: normalised names the camera's model is known by in setups
    model_keys: tuple[str, ...]
    #: the model as shown to the operator
    model: str
    modes: tuple[ModeSpec, ...]
    driver: CameraDriver

    @property
    def mode_names(self) -> list[str]:
        return [m.name for m in self.modes if m.name]


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def model_keys(manufacturer: str | None, model_name: str | None) -> tuple[str, ...]:
    """The names a setup may use for a camera model: the model name, or with the manufacturer."""
    name = (model_name or "").strip()
    full = f"{(manufacturer or '').strip()} {name}".strip()
    return tuple(dict.fromkeys(_norm(k) for k in (name, full) if k))


def load_modes(registry: sqlite3.Connection, camera_model_id: str) -> tuple[ModeSpec, ...]:
    """The named capture modes of a camera model."""
    import json

    rows = registry.execute(
        "SELECT id, name, width_px, height_px, nominal_fps, capture_settings FROM camera_modes "
        "WHERE camera_model_id = ? AND name IS NOT NULL ORDER BY name",
        (camera_model_id,),
    ).fetchall()
    return tuple(
        ModeSpec(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]) if r[5] else {}) for r in rows
    )


class ModePlanner:
    """Resolves setups and overrides into a mode for every camera.

    Parameters
    ----------
    cameras:
        The cameras that may take part in a capture.
    setups:
        The configured setups.
    default_setup:
        The setup used when none is requested; None means the only setup.
    profile_settings:
        Settings used for every camera when no setup is configured.
    """

    def __init__(
        self,
        cameras: list[PlannedCamera],
        setups: tuple[SetupConfig, ...] = (),
        default_setup: str | None = None,
        profile_settings: dict[str, Any] | None = None,
    ) -> None:
        self._cameras = {c.label: c for c in cameras}
        self._setups = {s.name: s for s in setups}
        self._default = default_setup
        self._profile = dict(profile_settings or {})

    @property
    def setup_names(self) -> list[str]:
        return list(self._setups)

    @property
    def default_setup(self) -> str | None:
        """The setup used when none is requested, if one is determined."""
        if self._default is not None:
            return self._default
        return next(iter(self._setups)) if len(self._setups) == 1 else None

    def options(self, label: str) -> list[str]:
        """Names of the modes a camera can be armed with."""
        return self._cameras[label].mode_names

    def settings_control(self, label: str) -> str:
        return self._cameras[label].driver.settings_control

    def plan(
        self, labels: list[str], setup: str | None = None, overrides: dict[str, str] | None = None
    ) -> dict[str, ModeChoice]:
        """Choose a mode for each of *labels*.

        Raises
        ------
        ModeError
            If the setup is unknown, none was chosen where several exist, or an
            override names an unknown camera. A camera whose mode cannot be chosen
            gets a :class:`ModeChoice` with a ``problem`` instead.
        """
        overrides = dict(overrides or {})
        for label in overrides:
            if label not in self._cameras:
                raise ModeError(f"unknown camera {label!r}")
        chosen = self._pick_setup(setup)
        return {label: self._choose(self._cameras[label], chosen, overrides.get(label)) for label in labels}

    def _pick_setup(self, name: str | None) -> SetupConfig | None:
        if not self._setups:
            if name:
                raise ModeError(f"unknown setup {name!r} (no setups are configured)")
            return None
        name = name or self.default_setup
        if name is None:
            raise ModeError(f"choose a setup: {', '.join(self._setups)}")
        if name not in self._setups:
            raise ModeError(f"unknown setup {name!r} (known: {', '.join(self._setups)})")
        return self._setups[name]

    def _choose(self, cam: PlannedCamera, setup: SetupConfig | None, override: str | None) -> ModeChoice:
        entry = setup.cameras.get(cam.label) if setup else None
        wanted = override or (entry.mode if entry else None)
        if wanted is None and setup is not None:
            by_model = {_norm(k): v for k, v in setup.models.items()}
            wanted = next((by_model[k] for k in cam.model_keys if k in by_model), None)
        if wanted is None:
            if setup is None and not override:
                return ModeChoice(None, None, dict(self._profile))
            return ModeChoice(None, None, problem=f"no capture mode for {cam.model} in setup {setup.name!r}" if setup else "no capture mode")
        spec = next((m for m in cam.modes if m.name == wanted), None)
        if spec is None:
            known = ", ".join(cam.mode_names) or "none"
            return ModeChoice(None, wanted, problem=f"{wanted!r} is not a capture mode of {cam.model} (modes: {known})")
        try:
            settings = cam.driver.mode_settings(spec)
        except ValueError as exc:
            return ModeChoice(spec.id, spec.name, problem=f"mode {wanted!r}: {exc}")
        if entry:
            settings.update(entry.settings)
        return ModeChoice(spec.id, spec.name, settings)

    def describe(self) -> dict[str, Any]:
        """What an operator needs to pick a setup and modes, JSON-serializable."""
        return {
            "setups": self.setup_names,
            "default": self.default_setup,
            "cameras": {
                label: {
                    "model": cam.model,
                    "modes": cam.mode_names,
                    "settings_control": cam.driver.settings_control,
                    "settings": sorted(cam.driver.supported_settings()),
                }
                for label, cam in self._cameras.items()
            },
        }
