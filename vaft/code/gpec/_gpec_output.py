"""Ideal-GPEC's native output: typed containers over the ``gpec_*_n<mode>.nc`` files.

The reference native-output contract is the shot-48226 @ 300 ms VEST run
(``gpec_control_output_n1.nc`` + ``gpec_cylindrical_output_n1.nc``, GPEC
``v1.5.5``).  Conventions encoded here, verified against that file pair:

- **Complex values** are stored as a real array with a length-2 ``i``
  dimension (``i=0`` real, ``i=1`` imaginary).  Unlike the DCON-family files
  the ``i`` axis comes *first*; :func:`vaft.code.gpec._netcdf.complex_var`
  selects by dimension name, so both layouts decode identically.
- **``n_tor``** is read from the ``n`` global attribute, never inferred from
  the filename; a filename/attribute mismatch warns and trusts the attribute.
- **``shot``/``time``** global attributes are kept verbatim.  GPEC writes 0/0
  when the equilibrium header carries no shot/time (true for the reference
  run), so identifying the discharge is the caller's job -- see
  ``vaft.machine_mapping.gpec_ideal``.
- **Cylindrical grids** are ``(z, R)``-ordered in the file (``nr``/``nz``
  namelist intervals produce ``nr+1``/``nz+1`` points); arrays are kept in
  native order here and reoriented only in the IMAS mapping layer.
- **Vacuum fields** are not written separately by this run configuration:
  the cylindrical file holds the equilibrium field, the plasma response, and
  the total perturbed field, so the vacuum contribution is their difference
  (exact under GPEC's linear superposition).  The subtraction is left to the
  mapping layer so the native container stays a faithful transcript.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._netcdf import complex_var

__all__ = [
    "GpecControlOutput",
    "GpecCylindricalOutput",
    "GpecIdealResult",
    "read_gpec_netcdf",
]

#: Control-surface eigen-decomposition and coupling variables preserved
#: verbatim (complex, native dims) because they have no IMAS home.
_CONTROL_EXTRA_VARIABLES = (
    "L",
    "Lambda",
    "P",
    "rho",
    "W_e_eigenvalue",
    "W_e_eigenvector",
    "W_xe_eigenvalue",
    "W_xe_eigenvector",
    "rho_xe_eigenvalue",
    "rho_xe_eigenvector",
    "P_xe_eigenvalue",
    "P_xe_eigenvector",
    "X_eigenvalue",
    "X_eigenvector",
)


def _real_var(ds, name: str) -> Optional[np.ndarray]:
    if name not in ds.variables:
        return None
    return np.asarray(ds[name].values, dtype=float)


def _attr_n_tor(ds, path: Path) -> int:
    if "n" not in ds.attrs:
        raise ValueError(
            f"{path.name} carries no 'n' global attribute; cannot determine "
            "the toroidal mode number from metadata"
        )
    n_tor = int(np.asarray(ds.attrs["n"]).reshape(-1)[0])
    match = re.search(r"_n(\d+)\.nc$", path.name)
    if match and int(match.group(1)) != n_tor:
        warnings.warn(
            f"{path.name} names mode n={match.group(1)} but its 'n' attribute "
            f"is {n_tor}; trusting the attribute",
            stacklevel=3,
        )
    return n_tor


def _plain_attrs(ds) -> dict[str, Any]:
    plain = {}
    for key, value in ds.attrs.items():
        array = np.asarray(value)
        if array.ndim == 0:
            item = array.item()
            plain[key] = item.decode() if isinstance(item, bytes) else item
        else:
            plain[key] = array.tolist()
    return plain


@dataclass
class GpecControlOutput:
    """Control-surface output (``gpec_control_output_n<mode>.nc``).

    Field names mirror the netCDF variable names.  ``attrs`` is the complete
    global-attribute record (equilibrium summary, energies, provenance).
    """

    n_tor: int
    machine: str
    shot: int
    time: float
    version: str
    jacobian: str
    helicity: int
    energy_vacuum: float
    energy_surface: float
    energy_plasma: float
    m: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None
    psi_n_rational: Optional[np.ndarray] = None
    q_rational: Optional[np.ndarray] = None
    m_rational: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None  # boundary R(theta) [m]
    z: Optional[np.ndarray] = None  # boundary z(theta) [m]
    Phi: Optional[np.ndarray] = None  # complex (m,), total flux [Wb]
    Phi_x: Optional[np.ndarray] = None  # complex (m,), external flux [Wb]
    b_n: Optional[np.ndarray] = None  # complex (m,) [T]
    b_n_x: Optional[np.ndarray] = None  # complex (m,) [T]
    xi_n: Optional[np.ndarray] = None  # complex (m,) [m]
    xi_n_fun: Optional[np.ndarray] = None  # complex (theta,) [m]
    b_n_fun: Optional[np.ndarray] = None  # complex (theta,) [T]
    coil_names: tuple[str, ...] = ()
    Phi_coil: Optional[np.ndarray] = None  # complex (coil_index, m) [Wb]
    #: Eigen-decompositions and coupling matrices with no IMAS destination,
    #: keyed by their native variable names (complex, native dim order).
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def energy_total(self) -> float:
        """Sum of the vacuum/surface/plasma perturbed-energy attributes [J]."""
        return self.energy_vacuum + self.energy_surface + self.energy_plasma


@dataclass
class GpecCylindricalOutput:
    """Cylindrical-grid output (``gpec_cylindrical_output_n<mode>.nc``).

    Perturbed-field arrays are complex in the file's native ``(z, R)`` order;
    ``l`` is the interior mask (1 interior, 0 exterior, -1 boundary).
    """

    n_tor: int
    R: np.ndarray  # (R,) [m]
    z: np.ndarray  # (z,) [m]
    l: Optional[np.ndarray] = None
    b_r_equil: Optional[np.ndarray] = None  # real (z, R) [T]
    b_z_equil: Optional[np.ndarray] = None
    b_t_equil: Optional[np.ndarray] = None
    b_r: Optional[np.ndarray] = None  # complex (z, R), total perturbed [T]
    b_z: Optional[np.ndarray] = None
    b_t: Optional[np.ndarray] = None
    b_r_plasma: Optional[np.ndarray] = None  # complex (z, R), plasma response [T]
    b_z_plasma: Optional[np.ndarray] = None
    b_t_plasma: Optional[np.ndarray] = None
    A_r: Optional[np.ndarray] = None  # complex (z, R) [Vs/m]
    A_z: Optional[np.ndarray] = None
    A_t: Optional[np.ndarray] = None
    xi_r: Optional[np.ndarray] = None  # complex (z, R) [m]
    xi_z: Optional[np.ndarray] = None
    xi_t: Optional[np.ndarray] = None
    units: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GpecIdealResult:
    """One ideal-GPEC run's native output set for a single toroidal mode.

    ``profile`` (``gpec_profile_output_n<mode>.nc``, resonant/rational-surface
    quantities) is not parsed yet and stays ``None``; the source paths are
    recorded so nothing needed to reach it later is lost.
    """

    control: GpecControlOutput
    cylindrical: Optional[GpecCylindricalOutput] = None
    profile: Optional[Any] = None
    source_paths: dict[str, str] = field(default_factory=dict)

    @property
    def n_tor(self) -> int:
        return self.control.n_tor

    @classmethod
    def from_netcdf(
        cls,
        run_dir: str | Path,
        mode: int | None = None,
        *,
        control_path: str | Path | None = None,
        cylindrical_path: str | Path | None = None,
    ) -> "GpecIdealResult":
        """Read a run directory's control (required) and cylindrical (optional) files.

        ``mode`` selects ``gpec_*_output_n<mode>.nc``; when omitted, exactly
        one control file must be present.
        """
        run_dir = Path(run_dir)
        if control_path is None:
            control_path = _locate_control(run_dir, mode)
        control_path = Path(control_path)
        control = _read_control(control_path)

        if cylindrical_path is None:
            candidate = run_dir / f"gpec_cylindrical_output_n{control.n_tor}.nc"
            cylindrical_path = candidate if candidate.exists() else None
        cylindrical = (
            _read_cylindrical(Path(cylindrical_path))
            if cylindrical_path is not None
            else None
        )
        if cylindrical is not None and cylindrical.n_tor != control.n_tor:
            raise ValueError(
                f"control file is n={control.n_tor} but cylindrical file is "
                f"n={cylindrical.n_tor}"
            )

        source_paths = {"control": str(control_path)}
        if cylindrical_path is not None:
            source_paths["cylindrical"] = str(cylindrical_path)
        profile_path = run_dir / f"gpec_profile_output_n{control.n_tor}.nc"
        if profile_path.exists():
            source_paths["profile"] = str(profile_path)
        return cls(
            control=control, cylindrical=cylindrical, source_paths=source_paths
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe transcript of the control output plus source provenance.

        The cylindrical bulk arrays are deliberately *not* duplicated into
        JSON -- they are large, live losslessly in the source ``.nc`` named
        under ``source_paths``, and their IMAS mapping is the consumer.
        """
        control = {
            "n_tor": self.control.n_tor,
            "machine": self.control.machine,
            "shot": self.control.shot,
            "time": self.control.time,
            "version": self.control.version,
            "jacobian": self.control.jacobian,
            "helicity": self.control.helicity,
            "energy_vacuum": self.control.energy_vacuum,
            "energy_surface": self.control.energy_surface,
            "energy_plasma": self.control.energy_plasma,
            "coil_names": list(self.control.coil_names),
            "attrs": self.control.attrs,
        }
        for name in (
            "m",
            "theta",
            "psi_n_rational",
            "q_rational",
            "m_rational",
            "R",
            "z",
        ):
            control[name] = _array_to_json(getattr(self.control, name))
        for name in (
            "Phi",
            "Phi_x",
            "b_n",
            "b_n_x",
            "xi_n",
            "b_n_fun",
            "xi_n_fun",
            "Phi_coil",
        ):
            control[name] = _array_to_json(getattr(self.control, name))
        control["extras"] = {
            name: _array_to_json(value)
            for name, value in self.control.extras.items()
        }
        return {"control": control, "source_paths": dict(self.source_paths)}

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def _array_to_json(value: Optional[np.ndarray]) -> Any:
    if value is None:
        return None
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {"real": array.real.tolist(), "imag": array.imag.tolist()}
    return array.tolist()


def _locate_control(run_dir: Path, mode: int | None) -> Path:
    if mode is not None:
        path = run_dir / f"gpec_control_output_n{mode}.nc"
        if not path.exists():
            raise FileNotFoundError(f"missing ideal-GPEC control output: {path}")
        return path
    candidates = sorted(run_dir.glob("gpec_control_output_n*.nc"))
    if not candidates:
        raise FileNotFoundError(
            f"no gpec_control_output_n*.nc in {run_dir}; not a completed "
            "ideal-GPEC run directory"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"multiple control outputs in {run_dir} "
            f"({', '.join(p.name for p in candidates)}); pass mode= to select one"
        )
    return candidates[0]


def _decode_coil_names(ds) -> tuple[str, ...]:
    if "coil_name" not in ds.variables:
        return ()
    raw = ds["coil_name"].values
    names = []
    for row in np.atleast_2d(raw):
        chars = [
            item.decode() if isinstance(item, bytes) else str(item) for item in row
        ]
        names.append("".join(chars).strip())
    return tuple(name for name in names if name)


def _read_control(path: Path) -> GpecControlOutput:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        n_tor = _attr_n_tor(ds, path)
        attrs = _plain_attrs(ds)
        extras = {
            name: value
            for name in _CONTROL_EXTRA_VARIABLES
            if (value := complex_var(ds, name)) is not None
        }
        return GpecControlOutput(
            n_tor=n_tor,
            machine=str(attrs.get("machine", "")),
            shot=int(attrs.get("shot", 0)),
            time=float(attrs.get("time", 0.0)),
            version=str(attrs.get("version", "")),
            jacobian=str(attrs.get("jacobian", "")),
            helicity=int(attrs.get("helicity", 0)),
            energy_vacuum=float(attrs.get("energy_vacuum", 0.0)),
            energy_surface=float(attrs.get("energy_surface", 0.0)),
            energy_plasma=float(attrs.get("energy_plasma", 0.0)),
            m=_real_var(ds, "m"),
            theta=_real_var(ds, "theta"),
            psi_n_rational=_real_var(ds, "psi_n_rational"),
            q_rational=_real_var(ds, "q_rational"),
            m_rational=_real_var(ds, "m_rational"),
            R=_real_var(ds, "R"),
            z=_real_var(ds, "z"),
            Phi=complex_var(ds, "Phi"),
            Phi_x=complex_var(ds, "Phi_x"),
            b_n=complex_var(ds, "b_n"),
            b_n_x=complex_var(ds, "b_n_x"),
            xi_n=complex_var(ds, "xi_n"),
            b_n_fun=complex_var(ds, "b_n_fun"),
            xi_n_fun=complex_var(ds, "xi_n_fun"),
            coil_names=_decode_coil_names(ds),
            Phi_coil=complex_var(ds, "Phi_coil"),
            extras=extras,
            attrs=attrs,
        )


def _read_cylindrical(path: Path) -> GpecCylindricalOutput:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        n_tor = _attr_n_tor(ds, path)
        units = {
            name: str(ds[name].attrs["units"])
            for name in ds.variables
            if "units" in ds[name].attrs
        }
        if "R" not in ds.variables or "z" not in ds.variables:
            raise ValueError(f"{path.name} is missing the R/z grid coordinates")
        return GpecCylindricalOutput(
            n_tor=n_tor,
            R=np.asarray(ds["R"].values, dtype=float),
            z=np.asarray(ds["z"].values, dtype=float),
            l=_real_var(ds, "l"),
            b_r_equil=_real_var(ds, "b_r_equil"),
            b_z_equil=_real_var(ds, "b_z_equil"),
            b_t_equil=_real_var(ds, "b_t_equil"),
            b_r=complex_var(ds, "b_r"),
            b_z=complex_var(ds, "b_z"),
            b_t=complex_var(ds, "b_t"),
            b_r_plasma=complex_var(ds, "b_r_plasma"),
            b_z_plasma=complex_var(ds, "b_z_plasma"),
            b_t_plasma=complex_var(ds, "b_t_plasma"),
            A_r=complex_var(ds, "A_r"),
            A_z=complex_var(ds, "A_z"),
            A_t=complex_var(ds, "A_t"),
            xi_r=complex_var(ds, "xi_r"),
            xi_z=complex_var(ds, "xi_z"),
            xi_t=complex_var(ds, "xi_t"),
            units=units,
            attrs=_plain_attrs(ds),
        )


def read_gpec_netcdf(run_dir: str | Path, mode: int | None = None) -> GpecIdealResult:
    """Read one ideal-GPEC run directory into a :class:`GpecIdealResult`."""
    return GpecIdealResult.from_netcdf(run_dir, mode)
