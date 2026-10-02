"""Named EFIT configurations a pipeline can select by name.

A preset is a named :class:`EFITScientificConfig`; the relative sigma floor it
assumes is part of that configuration (``constraints.sigma_floor``), applied by
:func:`vaft.code.efit.generate_kfile` for every caller.

``statistical_891`` -- the default since 2026-10-01
    The working setting of the #891 weight study: statistical sigma
    calibrated self-consistently on the reference shots (probe x3.62, loop
    x2.15 over the stored sigma, a 2 % floor on both), the diamagnetic flux
    fitted at x16, Ip sigma 20 % (x4 of 5 %), profile basis KPPCUR = 2,
    KFFCUR = 1, and an exit on psi convergence alone (ERRMIN 1e-4 with SAICON
    out of reach, NXITER 1).  It is :class:`EFITScientificConfig`'s defaults,
    so selecting it is the same as selecting nothing.  The calibration rests on
    39915's flat-top; see ``workflow/efit_uncertainty_calibration``.

``routine``
    The legacy production configuration, the defaults before 2026-10-01:
    legacy weights, a (2,2) basis, EFIT's own termination, no floor.  Its
    beta_p sits near zero on the reference shots (#386); kept for the studies
    recorded with it and as an explicit opt-out.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .config import EFITScientificConfig, routine_scientific_config

#: The families the sigma floor applies to: the multi-channel arrays.  The
#: scalar Ip and diamagnetic-flux nodes carry their own relative sigma.
FLOOR_FAMILIES = ("bpol_probe", "flux_loop")

#: The file a k-file stage writes beside its manifest to say which preset built
#: it; the EFIT-collection stage reads it back into ``code.parameters``.
PRESET_RECORD = "efit_preset.json"

#: SAICON high enough that the chi-square never gates EFIT's exit: the fit
#: stops on ERRMIN and a chi-square stall, and chi-square is judged afterwards.
PSI_ONLY_SAICON = 1.0e10

#: (515 - 1) // NXITER with NXITER = 1: EFIT's iteration arrays hold 515 (#171).
PSI_ONLY_MAX_ITERATIONS = 514


#: The preset a run uses when it names none.
DEFAULT_PRESET = "statistical_891"


@dataclass(frozen=True)
class EFITPreset:
    """A named scientific configuration."""

    name: str
    description: str
    scientific: EFITScientificConfig = field(default_factory=EFITScientificConfig)

    @property
    def sigma_floor(self) -> float:
        """The configuration's relative sigma floor (``constraints.sigma_floor``)."""
        return self.scientific.constraints.sigma_floor

    @property
    def floor_families(self) -> tuple[str, ...]:
        return self.scientific.constraints.sigma_floor_families

    def prepare_constraints(self, ods) -> tuple[Any, list[dict[str, Any]]]:
        """A copy of ``ods`` with this preset's sigma floor applied, and what it changed.

        ``generate_kfile`` applies the same floor itself; this is for a caller
        that wants to see the floored constraints, and applying it twice
        changes nothing (the floor is set by the measurements, not the sigma).
        """
        out = copy.deepcopy(ods)
        return out, apply_sigma_floor(out, self.sigma_floor, self.floor_families)

    def record(self) -> dict[str, Any]:
        """What a stage writes to say this preset built its inputs."""
        return {
            "name": self.name,
            "sigma_floor": self.sigma_floor,
            "floor_families": list(self.floor_families),
            "scientific": self.scientific.to_dict(),
            "scientific_sha256": self.scientific.sha256,
        }


def apply_sigma_floor(ods, fraction: float, families: Iterable[str] = FLOOR_FAMILIES) -> list[dict[str, Any]]:
    """Raise each fitted channel's ``measured_error_upper`` to ``fraction`` x its family's median |m|.

    Per slice and per family, over the channels with a positive weight.
    Mutates ``ods`` and returns, per slice and family, the floor and how many
    channels it raised; ``fraction == 0`` changes nothing.  The same rule as
    the #891 calibration driver, which the ``statistical_891`` sigma assumes.
    """
    changes: list[dict[str, Any]] = []
    if not fraction:
        return changes
    for index in range(len(ods["equilibrium.time_slice"])):
        root = f"equilibrium.time_slice.{index}.constraints"
        for family in families:
            path = f"{root}.{family}"
            if path not in ods:
                continue
            channels = []
            for j in range(len(ods[path])):
                node = f"{path}.{j}"
                try:
                    weight = float(ods[f"{node}.weight"])
                    measured = float(ods[f"{node}.measured"])
                except Exception:  # noqa: BLE001 - a node without both is not fitted
                    continue
                if weight > 0.0 and np.isfinite(measured):
                    channels.append((j, measured))
            if not channels:
                continue
            floor = float(fraction) * float(np.median([abs(m) for _j, m in channels]))
            raised = 0
            for j, _m in channels:
                key = f"{path}.{j}.measured_error_upper"
                current = float(ods[key]) if key in ods else float("nan")
                if not np.isfinite(current) or abs(current) < floor:
                    ods[key] = floor
                    raised += 1
            changes.append({"slice": index, "family": family, "floor": floor, "raised": raised,
                            "fitted": len(channels)})
    return changes


PRESETS: dict[str, EFITPreset] = {
    "statistical_891": EFITPreset(
        name="statistical_891",
        description="#891 working setting (default): statistical sigma, diamagnetic fitted, (2,1), psi-only exit",
        scientific=EFITScientificConfig(),
    ),
    "routine": EFITPreset(
        name="routine",
        description="legacy routine configuration (before 2026-10-01): legacy weights, (2,2), EFIT termination",
        scientific=routine_scientific_config(),
    ),
}


def preset_of(scientific: EFITScientificConfig) -> EFITPreset | None:
    """The preset whose scientific configuration ``scientific`` is, if any.

    What a stage that resolved a configuration from a payload (``--config``)
    rather than from a name should record: a payload that resolves to a
    preset is that preset, and a product built from it must not be
    ``unrecorded``.  Matched on the configuration hash.
    """
    sha = scientific.sha256
    for preset in PRESETS.values():
        if preset.scientific.sha256 == sha:
            return preset
    return None


def efit_preset(name: str) -> EFITPreset:
    """The preset called ``name``; a ``ValueError`` names the known ones."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown EFIT preset {name!r}; known: {sorted(PRESETS)}") from None


__all__ = [
    "DEFAULT_PRESET",
    "EFITPreset",
    "FLOOR_FAMILIES",
    "PRESETS",
    "PRESET_RECORD",
    "PSI_ONLY_MAX_ITERATIONS",
    "PSI_ONLY_SAICON",
    "apply_sigma_floor",
    "efit_preset",
    "preset_of",
]
