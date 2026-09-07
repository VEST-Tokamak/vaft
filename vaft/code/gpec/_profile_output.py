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
  hence tesla; ``I_res`` is amperes; island widths are in ``psi_n``.  Units
  are copied from the file rather than reinterpreted, and GPEC leaves many
  variables without a ``units`` attribute (``q_rational``, ``Delta``,
  ``K_isl``, ``B`` and every coordinate among them), so :attr:`units` is a
  partial mapping -- look up with ``.get``.
- **Spectral vs real space.**  ``*_fun`` variables are on the ``theta_dcon``
  grid; the others are on ``m_out`` (or ``m_pest`` for
  ``Jbgradpsi_pest``).  Both are kept in native order and native dimension
  order; reorientation belongs to the mapping layer.

Everything the file carries is preserved.  Decoding is decided per variable
by the file itself -- complex when the variable has the ``i`` dimension, its
native dtype otherwise -- and the result goes to a named field when one
exists and to :attr:`extras` otherwise, so a run that writes a quantity in an
unexpected shape is never dropped.  :attr:`dims` records each variable's
dimensions (with ``i`` removed), which is what
:meth:`GpecProfileOutput.resonant_table` selects on.

The profile file is by far the largest of a run's outputs (144 MB for the
DIII-D example, against 2 MB for the cylindrical file), so
:attr:`vaft.code.gpec.GpecIdealResult.profile` reads it lazily on first
access rather than whenever a run directory is opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._netcdf import complex_var, float_attr, int_attr, scalar_attr

__all__ = ["GpecProfileOutput", "read_gpec_profile_output", "read_resonant_table"]


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

#: Primary perturbed quantities; every other spectral and real-space variable
#: goes to ``extras`` under its native name.
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
    #: Native ``units`` attribute, for the variables that carry one.
    units: dict[str, str] = field(default_factory=dict)
    #: Native dimensions per variable, with the complex ``i`` axis removed.
    dims: dict[str, tuple[str, ...]] = field(default_factory=dict)
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

        Selected by *dimension*, not by length: a variable belongs here when
        the file gives it the ``psi_n_rational`` dimension alone.  Matching on
        length instead would fold in whatever else happened to have as many
        entries as there are rational surfaces -- ``coil_index`` and
        ``coil_name`` in a kinetic run, for instance.  ``m_rational`` is added
        because it is derived rather than stored.
        """
        table: dict[str, np.ndarray] = {}
        for name in self.dims:
            if self.dims[name] != ("psi_n_rational",):
                continue
            value = getattr(self, name, None) if name in _NAMED_FIELDS else self.extras.get(name)
            if value is not None:
                table[name] = value
        if self.m_rational is not None:
            table["m_rational"] = self.m_rational
        return table


def _read_profile(path: Path) -> GpecProfileOutput:
    import xarray as xr

    from ._gpec_output import _attr_n_tor, _plain_attrs

    with xr.open_dataset(path) as ds:
        n_tor = _attr_n_tor(ds, path)
        attrs = _plain_attrs(ds)
        units = {
            name: str(ds[name].attrs["units"])
            for name in ds.variables
            if "units" in ds[name].attrs
        }
        named: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        dims: dict[str, tuple[str, ...]] = {}
        for name in ds.variables:
            if name == "i":
                continue
            variable = ds[name]
            dims[name] = tuple(dim for dim in variable.dims if dim != "i")
            # The file decides: complex when it carries the i axis, native
            # dtype otherwise. A static list would drop a variable written in
            # an unexpected shape, or keep its (2, N) layout undecoded.
            if "i" in variable.dims:
                value: Any = complex_var(ds, name)
            else:
                value = np.asarray(variable.values)
                if value.dtype.kind in ("S", "U"):
                    value = _decode_strings(value)
            if name in _NAMED_FIELDS:
                named[name] = value
            else:
                extras[name] = value
        return GpecProfileOutput(
            n_tor=n_tor,
            machine=str(scalar_attr(attrs.get("machine", "")) or ""),
            shot=int_attr(attrs.get("shot", 0)),
            time=float(float_attr(attrs.get("time", 0.0)) or 0.0),
            version=str(scalar_attr(attrs.get("version", "")) or ""),
            helicity=float_attr(attrs.get("helicity")),
            extras=extras,
            units=units,
            dims=dims,
            attrs=attrs,
            **named,
        )


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    """String array or character matrix to a tuple of stripped strings.

    GPEC writes names as a ``(index, strlen)`` matrix of single characters;
    xarray sometimes hands that back already joined as one padded string per
    entry, so both layouts are handled.
    """

    def _text(item: Any) -> str:
        return item.decode() if isinstance(item, bytes) else str(item)

    values = np.asarray(values)
    if values.ndim <= 1:
        out = [_text(item).strip() for item in np.atleast_1d(values)]
    else:
        out = ["".join(_text(item) for item in row).strip() for row in values]
    return tuple(name for name in out if name)


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


#: Every variable name that has a dataclass field of its own.
_NAMED_FIELDS = frozenset(_named_variables())


def read_resonant_table(path: str | Path) -> dict[str, np.ndarray]:
    """Just the rational-surface block of a profile file.

    Opens the dataset and pulls only the variables dimensioned on
    ``psi_n_rational``; xarray loads lazily, so this costs a few kilobytes
    where :func:`read_gpec_profile_output` costs the whole file.
    """
    import xarray as xr

    path = Path(path)
    with xr.open_dataset(path) as ds:
        n_tor = int(np.asarray(ds.attrs["n"]).reshape(-1)[0]) if "n" in ds.attrs else 0
        table: dict[str, np.ndarray] = {}
        for name in ds.variables:
            variable = ds[name]
            if tuple(dim for dim in variable.dims if dim != "i") != ("psi_n_rational",):
                continue
            table[name] = (
                complex_var(ds, name) if "i" in variable.dims else np.asarray(variable.values)
            )
        if "q_rational" in table:
            table["m_rational"] = n_tor * np.asarray(table["q_rational"], dtype=float)
    return table


def read_gpec_profile_output(path: str | Path) -> GpecProfileOutput:
    """Read one ``gpec_profile_output_n<mode>.nc`` file."""
    return _read_profile(Path(path))
