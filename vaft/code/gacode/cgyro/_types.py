"""CGYRO's run configuration, result, and formalism metadata.

CGYRO is the GACODE suite's local delta-f gyrokinetic solver (continuum, flux tube,
linear or nonlinear). It shares the suite's runtime and profile spine with NEO and TGLF,
so this module adds only what is CGYRO's own. Three differences from TGLF that a copied
TGLF idiom gets wrong:

* the launcher takes ``-e``, ``-n`` **and** ``-nomp`` (``$GACODEHOME/cgyro/bin/cgyro``);
  TGLF's rejects the last one;
* species order is free, and the convention GACODE itself uses (``PROFILE_MODEL=2``,
  ``cgyro_make_profiles.F90``) puts the **electrons last**, where TGLF puts them first;
* **the frequency sign depends on the field orientation.** CGYRO's eigenvalue is
  ``phi ~ exp(-i omega t)`` and ``cgyro_make_profiles`` prints which sign the ion
  diamagnetic direction has for this ``IPCCW``/``BTCCW`` (``q*rho < 0`` means ion
  direction is ``omega > 0``). The native frequency is kept as written; the
  orientation-free one is a derived quantity, see
  :attr:`~vaft.code.gacode.cgyro.outputs.CgyroOutputs.frequency_ion_negative`.

Every key CGYRO knows is **not** transcribed here, unlike ``TGLF_DEFAULT_SCALARS``: the
launcher's parse step (``cgyro_parse.py``) writes ``input.cgyro.gen`` with every key
resolved against CGYRO's own defaults, and that file is kept in the run directory as the
complete record. A second, hand-copied default table would be a place to drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .._types import GACODEConfig
from ...base import CodeResult

#: CGYRO's own species cap (`cgyro_parse.py` declares `n = 11`).
MAX_SPECIES = 11

#: The field model, named for what it includes. ``N_FIELD`` counts fields, so the names
#: are the contract and the integer is CGYRO's encoding of it (#1354: the field model is
#: recorded explicitly, separately from the equilibrium beta that drives it).
FIELD_MODELS: Mapping[str, int] = {
    "es": 1,                # phi only
    "em-aperp": 2,          # phi + A_parallel (delta B_perp)
    "em-aperp-bpar": 3,     # phi + A_parallel + delta B_parallel
}

#: The TGLF configuration each CGYRO field model is compared against, in Lane T's
#: naming (#1482): ``USE_BPER`` is A_parallel, ``USE_BPAR`` is delta B_parallel.
TGLF_FIELD_MODEL: Mapping[str, str] = {
    "es": "es",
    "em-aperp": "em-bper",
    "em-aperp-bpar": "em-bper-bpar",
}

__all__ = [
    "CGYROConfig",
    "CGYROResult",
    "FIELD_MODELS",
    "MAX_SPECIES",
    "TGLF_FIELD_MODEL",
    "formalism",
    "plasma_formalism",
]


@dataclass(frozen=True)
class CGYROConfig(GACODEConfig):
    """One CGYRO run's settings: numerics and model choices, never plasma state.

    The plasma state (geometry, gradients, species, beta, collisionality) comes from the
    profile through :func:`~vaft.code.gacode.cgyro.inputs.prepare_cgyro_input`; this
    class holds only what a convergence scan or a modelling choice changes. Anything not
    named reaches ``input.cgyro`` through :attr:`extra_parameters`, upper-cased and
    merged last.

    The linear defaults are a starting point for VEST's low-aspect-ratio surfaces, not a
    converged resolution: the resolution is a result of the convergence scan, and is
    recorded with every run.
    """

    #: Velocity-space and real-space resolution.
    n_energy: int = 8
    n_xi: int = 24
    n_theta: int = 32
    n_radial: int = 8
    #: Toroidal modes. 1 is a single linear mode at :attr:`ky`.
    n_toroidal: int = 1
    #: ``k_y rho_s`` of the n=1 mode (the binormal wavenumber of a linear run).
    ky: float = 0.3
    box_size: int = 1
    nonlinear: bool = False
    field_model: str = "es"
    #: CGYRO's default, the Sugama operator.
    collision_model: int = 4
    #: The initial step; with the adaptive method CGYRO adjusts it.
    delta_t: float = 0.01
    #: 1 adaptive (Cash-Karp), CGYRO's 0 is fixed-step RK4. Adaptive is the default
    #: because fixed ``DELTA_T=0.01`` exceeds CGYRO's integration-error limit on VEST
    #: surfaces with kinetic electrons (48224 r/a=0.7, 2026-10-01); adaptive settles
    #: near 0.0035 and converges by t ~ 17 a/c_s.
    delta_t_method: int = 1
    #: Simulation time in ``a/c_s``. A linear run stops earlier when it converges.
    max_time: float = 200.0
    #: Linear convergence tolerance on the fractional frequency error.
    freq_tol: float = 1.0e-3
    print_step: int = 100
    #: Continue from ``bin.cgyro.restart`` instead of clearing the directory.
    restart: bool = False
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.field_model not in FIELD_MODELS:
            raise ValueError(
                f"field_model is one of {', '.join(FIELD_MODELS)}; got {self.field_model!r}"
            )
        for name in ("n_energy", "n_xi", "n_theta", "n_radial", "n_toroidal", "box_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1; got {getattr(self, name)!r}")
        if self.nonlinear and int(self.n_toroidal) < 2:
            raise ValueError("a nonlinear run needs n_toroidal > 1 (n=0 is the zonal mode)")
        if not float(self.ky) > 0.0:
            raise ValueError(f"ky must be positive; got {self.ky!r}")

    @property
    def n_field(self) -> int:
        return FIELD_MODELS[self.field_model]

    @property
    def regime(self) -> str:
        return "nonlinear" if self.nonlinear else "linear"

    def resolution(self) -> dict[str, Any]:
        """The numerical settings a result depends on, for provenance and convergence."""
        return {
            "n_energy": int(self.n_energy),
            "n_xi": int(self.n_xi),
            "n_theta": int(self.n_theta),
            "n_radial": int(self.n_radial),
            "n_toroidal": int(self.n_toroidal),
            "box_size": int(self.box_size),
            "ky": float(self.ky),
            "delta_t": float(self.delta_t),
            "delta_t_method": int(self.delta_t_method),
            "max_time": float(self.max_time),
            "freq_tol": float(self.freq_tol),
        }


def plasma_formalism(config: Optional[CGYROConfig] = None):
    """The solver-neutral #1727 record of a CGYRO run (:class:`vaft.code.formalism.PlasmaFormalism`).

    Local delta-f gyrokinetics on a closed flux surface, linear or nonlinear, with the
    field model coarsened to electrostatic / electromagnetic.  What is CGYRO's own --
    the ``em-aperp`` / ``em-aperp-bpar`` split, the continuum representation, Miller
    geometry, gyrokinetic electrons -- stays in ``extensions``, from which
    :func:`formalism` rebuilds the #1353 record, so the two cannot disagree.
    """
    from ...formalism import PlasmaFormalism

    configuration = config or CGYROConfig()
    return PlasmaFormalism(
        scientific_operation="turbulent_transport" if configuration.nonlinear else "microstability",
        bulk_description="kinetic",
        kinetic_equation="gyrokinetic",
        kinetic_population=("all",),
        distribution_formulation="delta_f",
        orbit_representation="gyrocenter",
        spatial_domain="local",
        topology_domain="closed_flux_surface",
        regime=configuration.regime,
        field_model="electrostatic" if configuration.field_model == "es" else "electromagnetic",
        solver="cgyro",
        extensions={
            "field_model": configuration.field_model,
            "numerical_representation": "continuum",
            "geometry_model": "miller",
            "species_model": "kinetic_electrons",
        },
    )


def formalism(
    config: Optional[CGYROConfig] = None, *, solver_version: Optional[str] = None
) -> dict[str, Any]:
    """The #1353 formalism record of a CGYRO run.

    Fixed by the solver except for the field model and the regime. ``spatial_domain`` is
    ``local`` -- never a bare ``global`` flag: CGYRO's global-spectral mode is a later,
    separately named phase (#1354 non-goal) and is not equivalent to GENE-global or GTC.

    Every value is read from :func:`plasma_formalism`, the solver-neutral record
    (#1727), so the two records of one run cannot disagree; the keys and values are
    unchanged from #1353.
    """
    record = plasma_formalism(config)
    extensions = record.extensions
    return {
        "distribution_formulation": record.distribution_formulation,
        "spatial_domain": record.spatial_domain,
        "numerical_representation": extensions["numerical_representation"],
        "field_model": extensions["field_model"],
        "regime": record.regime,
        "topology_domain": record.topology_domain,
        "geometry_model": extensions["geometry_model"],
        "species_model": extensions["species_model"],
        "solver": record.solver,
        "solver_version": solver_version,
    }


@dataclass
class CGYROResult(CodeResult):
    """A CGYRO run: exit status, files, native output, and two separate verdicts.

    #1354 keeps "the program ran" apart from "the physics is usable":

    ``executed``
        Exit 0 and CGYRO wrote its normal ``EXIT:`` line with no ``ERROR:`` line.
        GACODE codes exit 0 after rejecting input, so the status alone proves nothing.
    ``ok``
        ``executed`` and the native result is finite (:attr:`CgyroOutputs.solved`).
    ``qualified``
        ``ok`` and the physics passed its own criterion -- for a linear run, CGYRO's
        frequency converged below ``FREQ_TOL`` rather than stopping at ``MAX_TIME``.
        A nonlinear run is never qualified here: saturation is judged on the flux
        time trace by the caller, with the averaging window recorded.
    """

    outputs_native: Optional[Any] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def executed(self) -> bool:
        native = self.outputs_native
        return (
            self.returncode == 0
            and native is not None
            and not native.errors
            and native.exit_message is not None
        )

    @property
    def ok(self) -> bool:
        return self.executed and self.outputs_native.solved

    @property
    def qualified(self) -> bool:
        return self.ok and self.outputs_native.converged
