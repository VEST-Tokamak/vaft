"""Ideal-GPEC's profile output: resonant metrics and mapped perturbed fields.

``gpec_profile_output_n<mode>.nc`` is the file the resonant analysis lives in:
the rational surfaces GPEC found, the pitch-resonant flux and current on each
of them, island widths and Chirikov parameters, and the perturbed field and
displacement mapped onto the magnetic-coordinate grid.

Conventions encoded here, verified against the shipped DIII-D 147131 @ 2300 ms
example (GPEC ``v1.5.5-378-gf06e6ab``) and the shot-48226 VEST reference run:

- **Complex values** use the same leading length-2 ``i`` dimension as the
  control and cylindrical files; :func:`vaft.code.gpec._netcdf.complex_var`
  decodes both layouts.
- **``n_tor``** comes from the ``n`` global attribute, never the filename
  (convention C-08: the legacy readers fell back to ``n=1``).  A
  filename/attribute mismatch warns and trusts the attribute.
- **``helicity``** is a global attribute of this file (GPEC writes the
  ``ipd*btd`` product it ran with).  It is recorded verbatim, and it is what
  decides whether the real-space ``*_fun`` quantities in this file are
  conjugated relative to the spectral ones -- see
  :data:`vaft.machine_mapping.conventions.VEST_GPEC_COIL_DIRECTIONS`.  No
  conjugation is applied here: this container is a transcript.
- **Rational-surface arrays** are ordered as GPEC wrote them (increasing
  ``psi_n_rational``); ``q_rational`` is the safety factor there, so the
  resonant poloidal mode number is ``m = n_tor * q_rational``.
- **``Phi_res``/``Phi_res_v``** are fluxes *normalized by the surface area*,
  hence tesla; ``I_res`` is amperes; island widths are in ``psi_n``.  The
  ``units`` attribute of every variable is preserved in :attr:`units` rather
  than being reinterpreted.
- **Spectral vs real space.**  ``*_fun`` variables are on the ``theta_dcon``
  grid; the others are on ``m_out`` (or ``m_pest`` for
  ``Jbgradpsi_pest``).  Both are kept in native order and native dimension
  order; reorientation belongs to the mapping layer.

Everything the file carries is preserved: the named fields are the resonant
metrics and the primary perturbed quantities, and every other variable is
decoded into :attr:`extras` under its native name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._netcdf import complex_var

__all__ = ["GpecProfileOutput", "read_gpec_profile_output"]

#: Rational-surface quantities kept as named fields, in file order.
_RATIONAL_REAL = (
    "q_rational",
    "dqdpsi_n_rational",
    "area_rational",
    "w_isl",
    "w_isl_v",
    "w_isl_v_crit",
    "Phi_res_crit",
    "K_isl",
    "K_isl_v",
)

#: Complex rational-surface quantities.
_RATIONAL_COMPLEX = ("Phi_res", "Phi_res_v", "Delta", "B_pen", "I_res")

#: Radial (``psi_n``) profiles kept as named fields.
_RADIAL_REAL = ("q", "rmean_n", "dvdpsi_n")

#: Real 2-D geometry on ``(theta_dcon, psi_n)``.
_GEOMETRY_REAL = ("R", "z", "B", "R_n", "z_n")

#: Primary perturbed quantities; the rest of the spectral and real-space
#: variables go to ``extras`` under their native names.
_PRIMARY_COMPLEX = ("b_n", "b_n_fun", "xi_n", "xi_n_fun")


@dataclass
class GpecProfileOutput:
    """Profile output (``gpec_profile_output_n<mode>.nc``).

    Field names mirror the netCDF variable names.  Every array is optional
    because GPEC writes different subsets depending on the namelist, and a
    trimmed file (a committed regression extract, say) must still read.
    """

    n_tor: int
    machine: str = ""
    shot: int = 0
    time: float = 0.0
    version: str = ""
    helicity: Optional[float] = None
    # Coordinates
    psi_n: Optional[np.ndarray] = None
    psi_n_rational: Optional[np.ndarray] = None
    theta_dcon: Optional[np.ndarray] = None
    m_out: Optional[np.ndarray] = None
    m_pest: Optional[np.ndarray] = None
    # Rational-surface metrics
    q_rational: Optional[np.ndarray] = None
    dqdpsi_n_rational: Optional[np.ndarray] = None
    area_rational: Optional[np.ndarray] = None  # [m^2]
    Phi_res: Optional[np.ndarray] = None  # complex, area-normalized flux [T]
    Phi_res_v: Optional[np.ndarray] = None  # complex, vacuum [T]
    Phi_res_crit: Optional[np.ndarray] = None  # [T]
    Delta: Optional[np.ndarray] = None  # complex, resonance parameter [-]
    B_pen: Optional[np.ndarray] = None  # complex, penetrated field [T]
    I_res: Optional[np.ndarray] = None  # complex, resonant current [A]
    w_isl: Optional[np.ndarray] = None  # saturated island width [psi_n]
    w_isl_v: Optional[np.ndarray] = None  # vacuum island width [psi_n]
    w_isl_v_crit: Optional[np.ndarray] = None  # [psi_n]
    K_isl: Optional[np.ndarray] = None  # Chirikov [-]
    K_isl_v: Optional[np.ndarray] = None  # Chirikov, vacuum [-]
    # Radial profiles and geometry
    q: Optional[np.ndarray] = None
    rmean_n: Optional[np.ndarray] = None
    dvdpsi_n: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None  # (theta_dcon, psi_n) [m]
    z: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None  # equilibrium field strength [T]
    R_n: Optional[np.ndarray] = None  # radial unit normal
    z_n: Optional[np.ndarray] = None
    # Primary perturbed quantities
    b_n: Optional[np.ndarray] = None  # complex (m_out, psi_n) [T]
    b_n_fun: Optional[np.ndarray] = None  # complex (theta_dcon, psi_n) [T]
    xi_n: Optional[np.ndarray] = None  # complex (m_out, psi_n) [m]
    xi_n_fun: Optional[np.ndarray] = None  # complex (theta_dcon, psi_n) [m]
    #: Every other variable, decoded (complex where the ``i`` dimension is
    #: present) and keyed by its native name.
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    #: Native ``units`` attribute per variable, verbatim.
    units: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rational(self) -> int:
        """Number of rational surfaces in the file."""
        return 0 if self.psi_n_rational is None else int(self.psi_n_rational.size)

    @property
    def m_rational(self) -> Optional[np.ndarray]:
        """Resonant poloidal mode number ``n * q`` at each rational surface.

        GPEC solves for the surfaces where ``q = m / n``, so this is integral
        up to the root-finding tolerance; it is derived rather than read
        because the profile file stores only ``q_rational``.
        """
        if self.q_rational is None:
            return None
        return self.n_tor * np.asarray(self.q_rational, dtype=float)

    def resonant_table(self) -> dict[str, np.ndarray]:
        """The rational-surface quantities present, as one aligned mapping.

        Keys are the native names plus ``m_rational``; every value has one
        entry per rational surface, so a caller can build a table without
        knowing which optional metrics this run wrote.
        """
        table: dict[str, np.ndarray] = {}
        if self.psi_n_rational is not None:
            table["psi_n_rational"] = self.psi_n_rational
        if self.m_rational is not None:
            table["m_rational"] = self.m_rational
        for name in (*_RATIONAL_REAL, *_RATIONAL_COMPLEX):
            value = getattr(self, name, None)
            if value is not None:
                table[name] = value
        for name, value in self.extras.items():
            if (
                self.psi_n_rational is not None
                and getattr(value, "shape", ()) == self.psi_n_rational.shape
            ):
                table[name] = value
        return table


def _read_profile(path: Path) -> GpecProfileOutput:
    import xarray as xr

    from ._gpec_output import _attr_n_tor, _plain_attrs, _real_var

    with xr.open_dataset(path) as ds:
        n_tor = _attr_n_tor(ds, path)
        attrs = _plain_attrs(ds)
        units = {
            name: str(ds[name].attrs["units"])
            for name in ds.variables
            if "units" in ds[name].attrs
        }
        named = {
            name: (complex_var(ds, name) if name in _named_complex() else _real_var(ds, name))
            for name in _named_variables()
        }
        helicity = attrs.get("helicity")
        extras: dict[str, np.ndarray] = {}
        for name in ds.variables:
            if name in named or name in ("i",):
                continue
            extras[name] = (
                complex_var(ds, name)
                if "i" in ds[name].dims
                else np.asarray(ds[name].values)
            )
        return GpecProfileOutput(
            n_tor=n_tor,
            machine=str(attrs.get("machine", "")),
            shot=int(float(attrs.get("shot", 0) or 0)),
            time=float(attrs.get("time", 0.0) or 0.0),
            version=str(attrs.get("version", "")),
            helicity=None if helicity is None else float(helicity),
            extras=extras,
            units=units,
            attrs=attrs,
            **named,
        )


def _named_complex() -> tuple[str, ...]:
    return (*_RATIONAL_COMPLEX, *_PRIMARY_COMPLEX)


def _named_variables() -> tuple[str, ...]:
    return (
        "psi_n",
        "psi_n_rational",
        "theta_dcon",
        "m_out",
        "m_pest",
        *_RATIONAL_REAL,
        *_RATIONAL_COMPLEX,
        *_RADIAL_REAL,
        *_GEOMETRY_REAL,
        *_PRIMARY_COMPLEX,
    )


def read_gpec_profile_output(path: str | Path) -> GpecProfileOutput:
    """Read one ``gpec_profile_output_n<mode>.nc`` file."""
    return _read_profile(Path(path))
