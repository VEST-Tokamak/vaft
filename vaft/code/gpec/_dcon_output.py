"""DCON's own native output: a typed container over ``dcon_output_n<mode>.nc``.

DCON solves the ideal-MHD marginal-stability (delta-W) problem for a single
toroidal mode number. Several of its core outputs have no home in the IMAS
``mhd_linear`` IDS at all (the plasma/vacuum/total energy decomposition,
Mercier diagnostics, mode-range provenance, and the Fourier-space
eigenfunction reconstructed from ``solutions.bin``), so :class:`DconOutput`
is where those quantities live losslessly instead of being force-fit into
IMAS fields that don't mean the same thing (see
``vaft.machine_mapping.mhd_linear``, issue #170).

This is DCON's own schema, not a cross-solver abstraction: RDCON/STRIDE's
native output (:mod:`vaft.code.gpec._matching_output`) has a genuinely
different netCDF schema and is deliberately not folded in here.

Every field here is verified against the DCON Fortran source in GPEC
(``dcon/dcon.f``, ``dcon/dcon_netcdf.f``, ``dcon/free.f``,
``dcon/ode_output.f``, ``match/ideal.f``); see the module-level notes on
:class:`DconEigenfunction` for the one quantity (``v4``) whose physical
identity is not fully documented in the source and is kept only as a raw,
labeled value rather than interpreted.

``mlow``/``mhigh``/``mpert``/``mband`` are read from
``dcon_output_n<mode>.nc``'s global attributes, not by parsing DCON's ASCII
log: ``dcon/dcon_netcdf.f`` writes them (``nf90_put_att(ncid,nf90_global,
'mlow',mlow)`` etc.) directly onto the same netCDF file VAFT already reads
for ``W_t_eigenvalue``, which is both simpler and more robust than scraping
``dcon.out``'s free-format numeric row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
from typing import Any, Optional

import numpy as np


def _read_fortran_record_length(stream) -> Optional[int]:
    raw = stream.read(4)
    if len(raw) < 4:
        return None
    return struct.unpack("<i", raw)[0]


def _read_n_floats(stream, n: int) -> np.ndarray:
    raw = stream.read(n * 4)
    if len(raw) < n * 4:
        raise EOFError("Unexpected EOF while reading float data.")
    return np.frombuffer(raw, dtype="<f4")


def _read_solutions_bin_blocks(path: Path) -> list[list[np.ndarray]]:
    """Parse ``solutions.bin``'s Fortran unformatted records into per-``ipert`` blocks.

    Layout (``match/ideal.f:378-390``): outer loop over poloidal-harmonic
    blocks (``ipert=1,mpert``), inner loop over radial steps
    (``istep=0,mstep``), each record 7 ``REAL*4`` values -- ``psi, rho, q,
    Re(xi.grad(psi)), Im(xi.grad(psi)), Re(v4), Im(v4)`` -- with a blank
    (zero-length) record separating blocks.
    """
    blocks: list[list[np.ndarray]] = []
    with open(path, "rb") as stream:
        while True:
            length = _read_fortran_record_length(stream)
            if length is None:
                break
            if length == 0:
                continue
            vec = _read_n_floats(stream, length // 4)
            _read_fortran_record_length(stream)  # trailing record-length marker

            steps = [vec]
            while True:
                length2 = _read_fortran_record_length(stream)
                if length2 is None or length2 == 0:
                    break
                vec2 = _read_n_floats(stream, length2 // 4)
                _read_fortran_record_length(stream)
                steps.append(vec2)
            blocks.append(steps)
    return blocks


@dataclass
class DconEigenfunction:
    """Per-poloidal-harmonic normal-displacement eigenfunction from ``solutions.bin``.

    ``m`` is the true physical poloidal mode number for each block (``mlow +
    ipert``, 0-based ``ipert``), never a raw array index. ``xi_psi_real`` /
    ``xi_psi_imag`` are ``Re/Im(xi . grad(psi))`` -- confirmed both from
    ``match/ideal.f``'s ``b(:,istep)=ifac*singfac*v(:,1,istep)`` (the standard
    ``b_psi = i(m-nq).xi_psi`` relation) and independently from a
    physicist-authored reference reader that labels the same column
    ``r"Re$(\\xi \\cdot \\nabla \\psi)$"``.

    ``v4_real``/``v4_imag`` is the second Euler-Lagrange dependent variable
    (``v(:,4,istep)``, sourced from ``ud`` in the underlying ODE solution,
    i.e. plausibly ``d(pi)/dpsi``). Its physical identity was *not* found
    documented anywhere in the DCON/match source, so it is kept as a raw,
    labeled value rather than interpreted -- do not assume a meaning for it
    without independent confirmation.

    Rows are padded with NaN out to the longest block's step count, since
    integration step counts can differ slightly harmonic to harmonic.
    """

    m: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    q: np.ndarray
    xi_psi_real: np.ndarray
    xi_psi_imag: np.ndarray
    v4_real: np.ndarray
    v4_imag: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m.tolist(),
            "psi": self.psi.tolist(),
            "rho": self.rho.tolist(),
            "q": self.q.tolist(),
            "xi_psi_real": self.xi_psi_real.tolist(),
            "xi_psi_imag": self.xi_psi_imag.tolist(),
            "v4_real": self.v4_real.tolist(),
            "v4_imag": self.v4_imag.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DconEigenfunction":
        return cls(
            m=np.asarray(payload["m"], dtype=int),
            psi=np.asarray(payload["psi"], dtype=float),
            rho=np.asarray(payload["rho"], dtype=float),
            q=np.asarray(payload["q"], dtype=float),
            xi_psi_real=np.asarray(payload["xi_psi_real"], dtype=float),
            xi_psi_imag=np.asarray(payload["xi_psi_imag"], dtype=float),
            v4_real=np.asarray(payload["v4_real"], dtype=float),
            v4_imag=np.asarray(payload["v4_imag"], dtype=float),
        )


def read_solutions_bin(path: str | Path, *, mlow: int) -> DconEigenfunction:
    """Parse ``solutions.bin`` and label its blocks by true poloidal mode number.

    ``mlow`` must come from the companion run's ``dcon_output_n<mode>.nc``
    global attribute (see :func:`read_dcon_output`), not be guessed -- DCON
    computes it at runtime from the equilibrium (``mlow=MIN(nn*qmin,zero)-4-
    delta_mlow``, ``dcon/dcon.f:195``), so it is not a static default.
    """
    blocks = _read_solutions_bin_blocks(Path(path))
    n_ipert = len(blocks)
    if n_ipert == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return DconEigenfunction(
            m=np.zeros(0, dtype=int),
            psi=empty, rho=empty, q=empty,
            xi_psi_real=empty, xi_psi_imag=empty, v4_real=empty, v4_imag=empty,
        )

    max_steps = max(len(steps) for steps in blocks)
    arr = np.full((n_ipert, max_steps, 7), np.nan, dtype=np.float32)
    for i_block, steps in enumerate(blocks):
        for j_step, vec7 in enumerate(steps):
            arr[i_block, j_step, : vec7.size] = vec7

    m = mlow + np.arange(n_ipert, dtype=int)
    return DconEigenfunction(
        m=m,
        psi=arr[:, :, 0],
        rho=arr[:, :, 1],
        q=arr[:, :, 2],
        xi_psi_real=arr[:, :, 3],
        xi_psi_imag=arr[:, :, 4],
        v4_real=arr[:, :, 5],
        v4_imag=arr[:, :, 6],
    )


@dataclass
class DconOutput:
    """Typed container over one DCON run's native output (one ``n_tor``).

    Field names mirror ``dcon_output_n<mode>.nc``'s own variable/attribute
    names directly (``W_t_eigenvalue``, ``di``, ``dr``, ``ca1``, ``mlow``,
    ``mhigh``, ``mpert``, ``mband``) rather than inventing new terminology.
    """

    n_tor: int
    mlow: int
    mhigh: int
    mpert: int
    mband: int
    psi_n: Optional[np.ndarray] = None
    m: Optional[np.ndarray] = None
    W_p_eigenvalue: Optional[np.ndarray] = None  # complex, (mode,)
    W_v_eigenvalue: Optional[np.ndarray] = None
    W_t_eigenvalue: Optional[np.ndarray] = None
    di: Optional[np.ndarray] = None
    dr: Optional[np.ndarray] = None
    ca1: Optional[np.ndarray] = None
    eigenfunction: Optional[DconEigenfunction] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plasma1(self) -> Optional[complex]:
        """Least-stable plasma-response energy eigenvalue (``dcon/free.f``'s ``plasma1``)."""
        return None if self.W_p_eigenvalue is None or self.W_p_eigenvalue.size == 0 else complex(self.W_p_eigenvalue[0])

    @property
    def vacuum1(self) -> Optional[complex]:
        return None if self.W_v_eigenvalue is None or self.W_v_eigenvalue.size == 0 else complex(self.W_v_eigenvalue[0])

    @property
    def total1(self) -> Optional[complex]:
        """Least-stable total-energy eigenvalue -- ``dcon/free.f``'s ``total1``, numerically
        identical to ``W_t_eigenvalue`` at the least-stable mode index."""
        return None if self.W_t_eigenvalue is None or self.W_t_eigenvalue.size == 0 else complex(self.W_t_eigenvalue[0])

    @property
    def stable_free_boundary(self) -> Optional[bool]:
        """Free-boundary stability from ``sign(Re(total1))`` (``dcon/dcon.f:306-314``).

        ``None`` when no total-energy eigenvalue was computed (``vac_flag=false``
        runs). This does not cover fixed-boundary stability (``nzero``), which
        DCON only reports on stdout, not in any file VAFT retains today.
        """
        t1 = self.total1
        return None if t1 is None else bool(t1.real >= 0)

    def to_dict(self) -> dict[str, Any]:
        def _c(arr: Optional[np.ndarray]) -> Optional[dict[str, list[float]]]:
            if arr is None:
                return None
            arr = np.asarray(arr)
            return {"real": np.real(arr).tolist(), "imag": np.imag(arr).tolist()}

        return {
            "schema": "vaft.code.gpec.DconOutput",
            "schema_version": 1,
            "n_tor": self.n_tor,
            "mlow": self.mlow,
            "mhigh": self.mhigh,
            "mpert": self.mpert,
            "mband": self.mband,
            "psi_n": None if self.psi_n is None else np.asarray(self.psi_n).tolist(),
            "m": None if self.m is None else np.asarray(self.m).tolist(),
            "W_p_eigenvalue": _c(self.W_p_eigenvalue),
            "W_v_eigenvalue": _c(self.W_v_eigenvalue),
            "W_t_eigenvalue": _c(self.W_t_eigenvalue),
            "di": None if self.di is None else np.asarray(self.di).tolist(),
            "dr": None if self.dr is None else np.asarray(self.dr).tolist(),
            "ca1": None if self.ca1 is None else np.asarray(self.ca1).tolist(),
            "eigenfunction": None if self.eigenfunction is None else self.eigenfunction.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DconOutput":
        def _c(block: Optional[dict[str, list[float]]]) -> Optional[np.ndarray]:
            if block is None:
                return None
            return np.asarray(block["real"], dtype=float) + 1j * np.asarray(block["imag"], dtype=float)

        return cls(
            n_tor=payload["n_tor"],
            mlow=payload["mlow"],
            mhigh=payload["mhigh"],
            mpert=payload["mpert"],
            mband=payload["mband"],
            psi_n=None if payload.get("psi_n") is None else np.asarray(payload["psi_n"], dtype=float),
            m=None if payload.get("m") is None else np.asarray(payload["m"], dtype=int),
            W_p_eigenvalue=_c(payload.get("W_p_eigenvalue")),
            W_v_eigenvalue=_c(payload.get("W_v_eigenvalue")),
            W_t_eigenvalue=_c(payload.get("W_t_eigenvalue")),
            di=None if payload.get("di") is None else np.asarray(payload["di"], dtype=float),
            dr=None if payload.get("dr") is None else np.asarray(payload["dr"], dtype=float),
            ca1=None if payload.get("ca1") is None else np.asarray(payload["ca1"], dtype=float),
            eigenfunction=(
                None if payload.get("eigenfunction") is None else DconEigenfunction.from_dict(payload["eigenfunction"])
            ),
            metadata=payload.get("metadata", {}),
        )

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "DconOutput":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _complex_var(ds, name: str) -> Optional[np.ndarray]:
    """Read a GPEC-suite ``(..., i)`` netCDF variable as a complex array over its leading dim(s)."""
    if name not in ds.variables:
        return None
    var = ds[name]
    if "i" not in var.dims:
        return None
    real = var.isel(i=0).values
    imag = var.isel(i=1).values
    return np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)


def read_dcon_output(run_dir: str | Path, *, mode: int) -> DconOutput:
    """Build a :class:`DconOutput` from one completed DCON run directory.

    Reads ``dcon_output_n<mode>.nc`` for mode-range provenance, energy
    eigenvalues, and Mercier diagnostics; reads ``solutions.bin`` (if
    present -- it is written by the companion ``match`` tool, not DCON
    itself) for the Fourier-space eigenfunction. Never reads ``euler.bin``
    (out of scope: too large for routine conversion).
    """
    import xarray as xr

    run_dir = Path(run_dir)
    nc_path = run_dir / f"dcon_output_n{mode}.nc"
    with xr.open_dataset(nc_path) as ds:
        mlow = int(ds.attrs["mlow"])
        mhigh = int(ds.attrs["mhigh"])
        mpert = int(ds.attrs["mpert"])
        mband = int(ds.attrs["mband"])
        n_tor = int(ds.attrs.get("n", mode))

        psi_n = np.asarray(ds["psi_n"].values, dtype=float) if "psi_n" in ds.variables else None
        m = np.asarray(ds["m"].values, dtype=int) if "m" in ds.variables else None
        W_p_eigenvalue = _complex_var(ds, "W_p_eigenvalue")
        W_v_eigenvalue = _complex_var(ds, "W_v_eigenvalue")
        W_t_eigenvalue = _complex_var(ds, "W_t_eigenvalue")
        di = np.asarray(ds["di"].values, dtype=float) if "di" in ds.variables else None
        dr = np.asarray(ds["dr"].values, dtype=float) if "dr" in ds.variables else None
        ca1 = np.asarray(ds["ca1"].values, dtype=float) if "ca1" in ds.variables else None

    eigenfunction = None
    bin_path = run_dir / "solutions.bin"
    if bin_path.exists():
        eigenfunction = read_solutions_bin(bin_path, mlow=mlow)
        if eigenfunction.m.size and eigenfunction.m.size != mpert:
            # Not fatal -- match/ideal.f's ipert loop should span 1..mpert, but
            # this has not been independently verified for every DCON/match
            # version, so surface the mismatch rather than silently trust it.
            eigenfunction = DconEigenfunction(
                m=eigenfunction.m,
                psi=eigenfunction.psi,
                rho=eigenfunction.rho,
                q=eigenfunction.q,
                xi_psi_real=eigenfunction.xi_psi_real,
                xi_psi_imag=eigenfunction.xi_psi_imag,
                v4_real=eigenfunction.v4_real,
                v4_imag=eigenfunction.v4_imag,
            )

    metadata: dict[str, Any] = {"run_dir": str(run_dir), "nc_file": nc_path.name}
    if eigenfunction is not None and eigenfunction.m.size != mpert:
        metadata["eigenfunction_mpert_mismatch"] = {
            "solutions_bin_n_ipert": int(eigenfunction.m.size),
            "netcdf_mpert": mpert,
        }

    return DconOutput(
        n_tor=n_tor,
        mlow=mlow,
        mhigh=mhigh,
        mpert=mpert,
        mband=mband,
        psi_n=psi_n,
        m=m,
        W_p_eigenvalue=W_p_eigenvalue,
        W_v_eigenvalue=W_v_eigenvalue,
        W_t_eigenvalue=W_t_eigenvalue,
        di=di,
        dr=dr,
        ca1=ca1,
        eigenfunction=eigenfunction,
        metadata=metadata,
    )
