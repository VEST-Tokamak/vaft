"""Named EFIT configurations a pipeline can select by name.

A preset is everything a run needs beyond the constraints product: the
scientific configuration written into the k-file, and a relative sigma floor
applied to the constraints before the k-file is written.  The floor is part of
the preset rather than of :class:`EFITScientificConfig` so that the routine
configuration, its ``to_dict()`` and its ``sha256`` stay exactly as they were.

``routine``
    The production configuration: legacy weights, (2,2) basis, EFIT's own
    termination defaults.  Selecting it is the same as selecting nothing.

``statistical_891``
    The working setting of the #891 weight study (2026-09-29): statistical
    sigma calibrated self-consistently on the reference shots (probe x3.62,
    loop x2.15 over the stored sigma, a 2 % floor on both), the diamagnetic
    flux fitted at x16, Ip sigma 20 % (x4 of 5 %), profile basis
    KPPCUR = 2, KFFCUR = 1, and an exit on psi convergence alone
    (ERRMIN 1e-4 with SAICON out of reach, NXITER 1).  The calibration rests
    on 39915's flat-top; see ``workflow/efit_uncertainty_calibration``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np

from .config import EFITScientificConfig

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


@dataclass(frozen=True)
class EFITPreset:
    """A named scientific configuration plus the constraint sigma floor it assumes."""

    name: str
    description: str
    scientific: EFITScientificConfig = field(default_factory=EFITScientificConfig)
    #: Relative floor: each fitted channel's sigma is raised to at least this
    #: fraction of its family's median |measured|, per slice.  0 changes nothing.
    sigma_floor: float = 0.0
    floor_families: tuple[str, ...] = FLOOR_FAMILIES

    def __post_init__(self) -> None:
        if not (math.isfinite(self.sigma_floor) and self.sigma_floor >= 0.0):
            raise ValueError("sigma_floor must be a finite, non-negative fraction")

    def prepare_constraints(self, ods) -> tuple[Any, list[dict[str, Any]]]:
        """A copy of ``ods`` with this preset's sigma floor applied, and what it changed."""
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


def _statistical_891() -> EFITPreset:
    base = EFITScientificConfig()
    constraints = replace(
        base.constraints,
        uncertainty_mode="standard_deviation",
        uncertainty_scales={
            **base.constraints.uncertainty_scales,
            # uncertainty_scales divides sigma: a multiplier m is a scale 1/m.
            "bpol_probe": 1.0 / 3.62,
            "flux_loop": 1.0 / 2.15,
            "plasma_current": 1.0 / 4.0,
            "diamagnetic_flux": 1.0 / 16.0,
        },
    )
    profile = replace(base.profile, kppcur=2, kffcur=1)
    numerics = replace(
        base.numerics,
        inner_iterations=1,
        error_minimum=1.0e-4,
        chi_squared_target=PSI_ONLY_SAICON,
        max_iterations=PSI_ONLY_MAX_ITERATIONS,
    )
    return EFITPreset(
        name="statistical_891",
        description="#891 working setting: statistical sigma, diamagnetic fitted, (2,1), psi-only exit",
        scientific=replace(base, constraints=constraints, profile=profile, numerics=numerics),
        sigma_floor=0.02,
    )


PRESETS: dict[str, EFITPreset] = {
    "routine": EFITPreset(name="routine", description="production configuration (legacy weights, (2,2))"),
    "statistical_891": _statistical_891(),
}


def efit_preset(name: str) -> EFITPreset:
    """The preset called ``name``; a ``ValueError`` names the known ones."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown EFIT preset {name!r}; known: {sorted(PRESETS)}") from None


__all__ = [
    "EFITPreset",
    "FLOOR_FAMILIES",
    "PRESETS",
    "PRESET_RECORD",
    "PSI_ONLY_MAX_ITERATIONS",
    "PSI_ONLY_SAICON",
    "apply_sigma_floor",
    "efit_preset",
]
