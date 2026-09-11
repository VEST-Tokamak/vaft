"""Configuration and result types for the NEO backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ...base import CodeResult
from .._types import GACODEConfig

#: NEO's own defaults, from `neo/bin/neo_parse.py`. Repeated here rather than
#: left implicit because a run whose resolution was never chosen is not
#: reproducible: `input.neo` records only what it is given, and NEO fills the
#: rest silently.
NEO_DEFAULTS: Mapping[str, Any] = {
    "N_ENERGY": 6,
    "N_XI": 17,
    "N_THETA": 17,
    "N_RADIAL": 1,
    "RMIN_OVER_A": 0.5,
    "COLLISION_MODEL": 4,
    "PROFILE_MODEL": 2,
    "PROFILE_ERAD0_MODEL": 1,
    "ROTATION_MODEL": 1,
    "SPITZER_MODEL": 0,
    "EQUILIBRIUM_MODEL": 0,
    "SILENT_FLAG": 0,
    "IPCCW": -1,
    "BTCCW": -1,
}

#: `PROFILE_MODEL=2` is the mode that reads `input.gacode`; `1` is the local
#: mode where every profile quantity comes from `input.neo` itself.
PROFILE_MODEL_EXPERIMENTAL = 2


@dataclass(frozen=True)
class NEOConfig(GACODEConfig):
    """A NEO run's numerical settings, on top of the shared GACODE runtime.

    Every field lands in `input.neo` verbatim and is carried into the result's
    provenance, so a stored result says what produced it.

    Attributes
    ----------
    n_energy, n_xi, n_theta
        Velocity-space and poloidal resolution. NEO does not estimate its own
        discretisation error, so a convergence scan is the caller's job and
        these are the knobs for it.
    n_radial
        Number of radial points solved. With ``PROFILE_MODEL=2`` these are
        placed relative to ``rmin_over_a``.
    rmin_over_a
        Normalised minor radius of the (first) surface to solve.
    collision_model
        4 is the full linearised Fokker-Planck operator.
    profile_model
        2 reads ``input.gacode``; 1 takes local parameters from ``input.neo``.
    rotation_model
        1 ignores rotation; 2 includes the sonic-rotation terms.
    n_species
        Total species count, electrons included. Defaults to the species in the
        supplied profile.
    extra_parameters
        Additional ``KEY=VALUE`` pairs written verbatim, for NEO settings this
        class does not model. They are recorded in provenance like any other.
    """

    n_energy: int = 6
    n_xi: int = 17
    n_theta: int = 17
    n_radial: int = 1
    rmin_over_a: float = 0.5
    collision_model: int = 4
    profile_model: int = PROFILE_MODEL_EXPERIMENTAL
    profile_erad0_model: int = 1
    rotation_model: int = 1
    spitzer_model: int = 0
    equilibrium_model: int = 0
    ipccw: int = -1
    btccw: int = -1
    n_species: Optional[int] = None
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("n_energy", "n_xi", "n_theta", "n_radial"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1; got {getattr(self, name)!r}")
        if not 0.0 < float(self.rmin_over_a) < 1.0:
            raise ValueError(
                f"rmin_over_a must lie in (0, 1); got {self.rmin_over_a!r}. "
                "NEO solves a flux surface, and neither the axis nor the "
                "separatrix is one."
            )
        # NEO's own limits (neo/src/neo_check.f90), refused here so that the
        # message names the setting rather than arriving through out.neo.run.
        if int(self.n_theta) % 2 == 0:
            raise ValueError(f"n_theta must be odd for NEO; got {self.n_theta!r}")
        if self.n_species is not None and int(self.n_species) > 6:
            raise ValueError(f"NEO supports at most 6 species; got {self.n_species!r}")
        if self.n_species is not None and int(self.n_species) < 2:
            raise ValueError(
                f"n_species counts electrons too, so it is at least 2; got "
                f"{self.n_species!r}"
            )


@dataclass
class NEOResult(CodeResult):
    """A NEO run: its exit status, its files, and its native output.

    Subclasses :class:`vaft.code.base.CodeResult`, and follows NUBEAM in
    requiring the native container for ``ok`` -- and goes further, requiring it
    to report a completed solve. NEO exits zero after rejecting its input, so a
    zero status alone is not success, and nor is the presence of output files.
    """

    outputs_native: Optional[Any] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and self.outputs_native is not None
            and self.outputs_native.solved
        )
