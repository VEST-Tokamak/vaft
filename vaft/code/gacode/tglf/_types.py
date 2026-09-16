"""TGLF's own defaults, and the run configuration VAFT resolves against them.

Transcribed from ``$GACODEHOME/tglf/bin/tglf_defaults.py`` for the reason
:data:`~vaft.code.gacode.neo._types.NEO_DEFAULTS` exists: ``input.tglf`` records only
what it is given, so a file that omits a setting cannot afterwards be told apart from
one that chose TGLF's default deliberately. Writing every key out makes a run
reproducible from its own directory.

Four places where TGLF differs from NEO, each of which a copied NEO idiom gets wrong:

* the launcher takes ``-e`` and ``-n`` and **rejects anything else** -- there is no
  ``-nomp``, and an unknown flag exits 1 (``$GACODEHOME/tglf/bin/tglf``);
* booleans are Fortran logicals, ``.true.``/``.false.``, not ``1``/``0``;
* **species 1 is the electrons** (``ZS_1 = -1``), where NEO puts ions first;
* up to 7 species, where NEO caps at 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .._types import GACODEConfig
from ...base import CodeResult

#: The most species TGLF accepts, electrons included (`tglf_defaults.py` indexes 1..7).
MAX_SPECIES = 7

#: Scalar settings, keyed exactly as `input.tglf` spells them.
TGLF_DEFAULT_SCALARS: Mapping[str, Any] = {
    "UNITS": 'GYRO',
    "USE_TRANSPORT_MODEL": True,
    "GEOMETRY_FLAG": 1,
    "WRITE_WAVEFUNCTION_FLAG": 0,
    "SIGN_BT": 1.0,
    "SIGN_IT": 1.0,
    "THETA_TRAPPED": 0.7,
    "WDIA_TRAPPED": 0.0,
    "PARK": 1.0,
    "GHAT": 1.0,
    "GCHAT": 1.0,
    "WD_ZERO": 0.1,
    "LINSKER_FACTOR": 0.0,
    "GRADB_FACTOR": 0.0,
    "FILTER": 2.0,
    "DAMP_PSI": 0.0,
    "DAMP_SIG": 0.0,
    "IFLUX": True,
    "USE_BPER": False,
    "USE_BPAR": False,
    "USE_MHD_RULE": True,
    "USE_BISECTION": True,
    "USE_INBOARD_DETRAPPED": False,
    "IBRANCH": -1,
    "NMODES": 2,
    "NBASIS_MAX": 4,
    "NBASIS_MIN": 2,
    "NXGRID": 16,
    "NKY": 12,
    "USE_AVE_ION_GRID": False,
    "ADIABATIC_ELEC": False,
    "ALPHA_MACH": 0.0,
    "ALPHA_E": 1.0,
    "ALPHA_P": 1.0,
    "ALPHA_QUENCH": 0.0,
    "ALPHA_ZF": 1.0,
    "XNU_FACTOR": 1.0,
    "DEBYE_FACTOR": 1.0,
    "ETG_FACTOR": 1.25,
    "RLNP_CUTOFF": 18.0,
    "SAT_RULE": 0,
    "KYGRID_MODEL": 1,
    "XNU_MODEL": 2,
    "VPAR_MODEL": 0,
    "VPAR_SHEAR_MODEL": 1,
    "NS": 2,
    "KY": 0.3,
    "WIDTH": 1.65,
    "WIDTH_MIN": 0.3,
    "NWIDTH": 21,
    "FIND_WIDTH": True,
    "VEXB_SHEAR": 0.0,
    "VEXB": 0.0,
    "BETAE": 0.0,
    "XNUE": 0.0,
    "ZEFF": 1.0,
    "DEBYE": 0.0,
    "NEW_EIKONAL": True,
    "RMIN_SA": 0.5,
    "RMAJ_SA": 3.0,
    "Q_SA": 2.0,
    "SHAT_SA": 1.0,
    "ALPHA_SA": 0.0,
    "XWELL_SA": 0.0,
    "THETA0_SA": 0.0,
    "B_MODEL_SA": 1,
    "FT_MODEL_SA": 1,
    "RMIN_LOC": 0.5,
    "RMAJ_LOC": 3.0,
    "ZMAJ_LOC": 0.0,
    "DRMINDX_LOC": 1.0,
    "DRMAJDX_LOC": 0.0,
    "DZMAJDX_LOC": 0.0,
    "KAPPA_LOC": 1.0,
    "S_KAPPA_LOC": 0.0,
    "DELTA_LOC": 0.0,
    "S_DELTA_LOC": 0.0,
    "ZETA_LOC": 0.0,
    "S_ZETA_LOC": 0.0,
    "SHAPE_COS0": 0.0,
    "SHAPE_S_COS0": 0.0,
    "SHAPE_COS1": 0.0,
    "SHAPE_S_COS1": 0.0,
    "SHAPE_COS2": 0.0,
    "SHAPE_S_COS2": 0.0,
    "SHAPE_COS3": 0.0,
    "SHAPE_S_COS3": 0.0,
    "SHAPE_COS4": 0.0,
    "SHAPE_S_COS4": 0.0,
    "SHAPE_COS5": 0.0,
    "SHAPE_S_COS5": 0.0,
    "SHAPE_COS6": 0.0,
    "SHAPE_S_COS6": 0.0,
    "SHAPE_SIN3": 0.0,
    "SHAPE_S_SIN3": 0.0,
    "SHAPE_SIN4": 0.0,
    "SHAPE_S_SIN4": 0.0,
    "SHAPE_SIN5": 0.0,
    "SHAPE_S_SIN5": 0.0,
    "SHAPE_SIN6": 0.0,
    "SHAPE_S_SIN6": 0.0,
    "Q_LOC": 2.0,
    "Q_PRIME_LOC": 16.0,
    "P_PRIME_LOC": 0.0,
    "BETA_LOC": 0.0,
    "KX0_LOC": 0.0,
    "NN_MAX_ERROR": -1.0,
}

#: Per-species settings, written as `<KEY>_<n>` for n = 1..NS.
TGLF_DEFAULT_SPECIES: Mapping[str, Any] = {
    "AS": 1.0,
    "MASS": 0.0002723,
    "RLNS": 1.0,
    "RLTS": 3.0,
    "TAUS": 1.0,
    "VNS_SHEAR": 0.0,
    "VPAR": 0.0,
    "VPAR_SHEAR": 0.0,
    "VTS_SHEAR": 0.0,
    "ZS": -1.0,
}

__all__ = [
    "MAX_SPECIES",
    "TGLFConfig",
    "TGLFResult",
    "TGLF_DEFAULT_SCALARS",
    "TGLF_DEFAULT_SPECIES",
]


@dataclass(frozen=True)
class TGLFConfig(GACODEConfig):
    """One TGLF run's settings.

    Only the knobs VAFT resolves deliberately are named; everything else reaches the
    file through :attr:`extra_parameters`, which is merged verbatim and upper-cased so
    an unmodelled TGLF key never needs this class to change.

    ``n_omp`` is inherited from :class:`GACODEConfig` and is deliberately **not** passed
    to the launcher: TGLF's own script hardwires ``NOMP=1`` and rejects the flag.
    """

    #: Saturation rule. 0-3; the electromagnetic rules are 2 and 3.
    sat_rule: int = 0
    #: Solve the transport problem (quasilinear fluxes) rather than a single mode.
    use_transport_model: bool = True
    #: Miller (1) or s-alpha (0) geometry.
    geometry_flag: int = 1
    #: Include the perpendicular and parallel magnetic perturbations.
    use_bper: bool = False
    use_bpar: bool = False
    #: Number of species including electrons; resolved from the profile when None.
    n_species: Optional[int] = None
    extra_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_species is not None:
            if not 2 <= int(self.n_species) <= MAX_SPECIES:
                raise ValueError(
                    f"TGLF takes between 2 and {MAX_SPECIES} species including "
                    f"electrons; got n_species={self.n_species}"
                )
        if not 0 <= int(self.sat_rule) <= 3:
            raise ValueError(f"SAT_RULE is 0, 1, 2 or 3; got {self.sat_rule}")
        if int(self.geometry_flag) not in (0, 1):
            raise ValueError(
                f"GEOMETRY_FLAG is 0 (s-alpha) or 1 (Miller); got {self.geometry_flag}"
            )


@dataclass
class TGLFResult(CodeResult):
    """A TGLF run: its exit status, its files, and its native output.

    ``ok`` requires the native container to report a completed solve, following
    :class:`~vaft.code.gacode.neo.NEOResult`: the launcher creates ``out.tglf.run``
    before the solve starts, so neither a zero exit status nor the presence of output
    proves anything.
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
