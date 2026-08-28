"""RDCON/STRIDE's shared native output: the PEST3 Galerkin matching-matrix schema.

RDCON and STRIDE both solve the same rational-surface matching problem via a
Galerkin method (``rdcon/gal.f``'s ``gal_write_pest3_data``) and both write an
identically-named, identically-shaped set of netCDF variables for it --
``rdcon/rdcon_netcdf.f`` and ``stride/stride_netcdf.f`` both define
``Delta_prime``/``A_prime``/``B_prime``/``Gamma_prime``/``Delta`` with the
same ``(r, r_prime, i)``/``(l, lp, i)`` dims and the identical ``"PEST3 Delta
Prime Matrix"`` long_name. That is a genuinely shared native schema, not a
premature cross-solver abstraction, so :class:`Pest3MatchingOutput` is the
one container shared by both solvers here; DCON's own output
(:mod:`vaft.code.gpec._dcon_output`) is not folded in, since its netCDF
schema is unrelated.

The IMAS ``mhd_linear`` IDS has no field for Delta-prime at all. The
``ntms`` (Neoclassical Tearing Modes) IDS has a ``deltaw[:]`` list of named
contributions to the Rutherford equation, and classical Delta-prime is
literally one such contribution -- so the *diagonal* (single-surface) values
of the matrices here are the ones VAFT's ``mhd_linear`` mapping layer
projects into ``ntms``. The full matrices, including off-diagonal
surface-surface coupling terms, have no IMAS home and stay here losslessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._netcdf import complex_scalar_attr, complex_var


@dataclass
class Pest3MatchingOutput:
    """Typed container over one RDCON or STRIDE run's PEST3 matching output (one ``n_tor``).

    Field names mirror the netCDF directly (``Delta_prime``, ``A_prime``,
    ``B_prime``, ``Gamma_prime``, ``Delta``, ``mlow``/``mhigh``/``mpert``/
    ``mband``), with one deliberate exception: the netCDF variable holding
    the poloidal mode number per rational surface is named ``r``/``r_prime``
    in the Fortran (``nf90_def_var(ncid,"r",...)``, long_name "Rational
    Surface Index") but its *values* are ``sing(i)%m`` -- the actual
    poloidal mode number, not a generic index. This container names the
    field ``m`` (the physically correct, unambiguous name) rather than
    reproducing the native variable's own misleading name.
    """

    solver: str  # "rdcon" or "stride" -- provenance only, not a schema difference
    n_tor: int
    mlow: int
    mhigh: int
    mpert: int
    mband: int
    m: Optional[np.ndarray] = None  # (msing,) poloidal mode number per rational surface
    psi_n_rational: Optional[np.ndarray] = None  # (msing,)
    q_rational: Optional[np.ndarray] = None  # (msing,)
    A_prime: Optional[np.ndarray] = None  # (msing, msing) complex, PEST3 matrix
    B_prime: Optional[np.ndarray] = None
    Gamma_prime: Optional[np.ndarray] = None
    Delta_prime: Optional[np.ndarray] = None
    Delta: Optional[np.ndarray] = None  # (2*msing, 2*msing) complex, raw Galerkin solution matrix
    nzero: Optional[int] = None
    plasma1: Optional[complex] = None
    vacuum1: Optional[complex] = None
    total1: Optional[complex] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def msing(self) -> int:
        return 0 if self.m is None else int(self.m.size)

    def delta_prime_diagonal(self) -> list[dict[str, Any]]:
        """The single-surface (diagonal) classical Delta-prime per rational surface.

        This is the value with a legitimate IMAS home (``ntms.deltaw``); the
        off-diagonal surface-surface coupling terms of the full matrix stay
        in :attr:`Delta_prime` only.
        """
        if self.Delta_prime is None or self.m is None:
            return []
        out = []
        for i in range(self.msing):
            value = self.Delta_prime[i, i]
            out.append(
                {
                    "m": int(self.m[i]),
                    "n": self.n_tor,
                    "psi_n": None if self.psi_n_rational is None else float(self.psi_n_rational[i]),
                    "q": None if self.q_rational is None else float(self.q_rational[i]),
                    "delta_prime_real": float(value.real),
                    "delta_prime_imag": float(value.imag),
                }
            )
        return out

    @property
    def stable(self) -> Optional[bool]:
        """Free-boundary-style stability from ``sign(Re(total1))``, when available.

        Mirrors ``dcon/dcon.f:306-314``'s convention; ``None`` when this run's
        netCDF carried no ``total1`` global attribute (DCON's own netCDF never
        does; RDCON/STRIDE do only when ``vac_flag``-equivalent output ran).
        """
        return None if self.total1 is None else bool(self.total1.real >= 0)

    def to_dict(self) -> dict[str, Any]:
        def _c(arr: Optional[np.ndarray]) -> Optional[dict[str, list]]:
            if arr is None:
                return None
            arr = np.asarray(arr)
            return {"real": np.real(arr).tolist(), "imag": np.imag(arr).tolist()}

        def _cs(value: Optional[complex]) -> Optional[dict[str, float]]:
            return None if value is None else {"real": float(value.real), "imag": float(value.imag)}

        return {
            "schema": "vaft.code.gpec.Pest3MatchingOutput",
            "schema_version": 1,
            "solver": self.solver,
            "n_tor": self.n_tor,
            "mlow": self.mlow,
            "mhigh": self.mhigh,
            "mpert": self.mpert,
            "mband": self.mband,
            "m": None if self.m is None else np.asarray(self.m).tolist(),
            "psi_n_rational": None if self.psi_n_rational is None else np.asarray(self.psi_n_rational).tolist(),
            "q_rational": None if self.q_rational is None else np.asarray(self.q_rational).tolist(),
            "A_prime": _c(self.A_prime),
            "B_prime": _c(self.B_prime),
            "Gamma_prime": _c(self.Gamma_prime),
            "Delta_prime": _c(self.Delta_prime),
            "Delta": _c(self.Delta),
            "nzero": self.nzero,
            "plasma1": _cs(self.plasma1),
            "vacuum1": _cs(self.vacuum1),
            "total1": _cs(self.total1),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Pest3MatchingOutput":
        def _c(block: Optional[dict[str, list]]) -> Optional[np.ndarray]:
            if block is None:
                return None
            return np.asarray(block["real"], dtype=float) + 1j * np.asarray(block["imag"], dtype=float)

        def _cs(block: Optional[dict[str, float]]) -> Optional[complex]:
            return None if block is None else complex(block["real"], block["imag"])

        return cls(
            solver=payload["solver"],
            n_tor=payload["n_tor"],
            mlow=payload["mlow"],
            mhigh=payload["mhigh"],
            mpert=payload["mpert"],
            mband=payload["mband"],
            m=None if payload.get("m") is None else np.asarray(payload["m"], dtype=int),
            psi_n_rational=(
                None if payload.get("psi_n_rational") is None else np.asarray(payload["psi_n_rational"], dtype=float)
            ),
            q_rational=None if payload.get("q_rational") is None else np.asarray(payload["q_rational"], dtype=float),
            A_prime=_c(payload.get("A_prime")),
            B_prime=_c(payload.get("B_prime")),
            Gamma_prime=_c(payload.get("Gamma_prime")),
            Delta_prime=_c(payload.get("Delta_prime")),
            Delta=_c(payload.get("Delta")),
            nzero=payload.get("nzero"),
            plasma1=_cs(payload.get("plasma1")),
            vacuum1=_cs(payload.get("vacuum1")),
            total1=_cs(payload.get("total1")),
            metadata=payload.get("metadata", {}),
        )

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "Pest3MatchingOutput":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def read_pest3_matching_output(run_dir: str | Path, *, solver: str, mode: int) -> Pest3MatchingOutput:
    """Build a :class:`Pest3MatchingOutput` from a completed RDCON or STRIDE run.

    ``solver`` must be ``"rdcon"`` or ``"stride"`` -- both read the same
    netCDF variable names (``rdcon/rdcon_netcdf.f`` and
    ``stride/stride_netcdf.f`` both write ``Delta_prime``/``A_prime``/
    ``B_prime``/``Gamma_prime``/``Delta`` with identical dims and the
    ``"PEST3 Delta Prime Matrix"`` long_name).
    """
    import xarray as xr

    if solver not in ("rdcon", "stride"):
        raise ValueError(f"Unsupported PEST3-matching solver: {solver!r}")

    run_dir = Path(run_dir)
    nc_path = run_dir / f"{solver}_output_n{mode}.nc"
    with xr.open_dataset(nc_path) as ds:
        mlow = int(ds.attrs["mlow"])
        mhigh = int(ds.attrs["mhigh"])
        mpert = int(ds.attrs["mpert"])
        mband = int(ds.attrs["mband"])
        n_tor = int(ds.attrs.get("n", mode))

        m = np.asarray(ds["r"].values, dtype=int) if "r" in ds.variables else None
        psi_n_rational = (
            np.asarray(ds["psi_n_rational"].values, dtype=float) if "psi_n_rational" in ds.variables else None
        )
        q_rational = np.asarray(ds["q_rational"].values, dtype=float) if "q_rational" in ds.variables else None

        A_prime = complex_var(ds, "A_prime")
        B_prime = complex_var(ds, "B_prime")
        Gamma_prime = complex_var(ds, "Gamma_prime")
        Delta_prime = complex_var(ds, "Delta_prime")
        Delta = complex_var(ds, "Delta")

        nzero_attr = ds.attrs.get("nzero")
        nzero = None if nzero_attr is None else int(nzero_attr)
        plasma1 = complex_scalar_attr(ds, "plasma1")
        vacuum1 = complex_scalar_attr(ds, "vacuum1")
        total1 = complex_scalar_attr(ds, "total1")

    return Pest3MatchingOutput(
        solver=solver,
        n_tor=n_tor,
        mlow=mlow,
        mhigh=mhigh,
        mpert=mpert,
        mband=mband,
        m=m,
        psi_n_rational=psi_n_rational,
        q_rational=q_rational,
        A_prime=A_prime,
        B_prime=B_prime,
        Gamma_prime=Gamma_prime,
        Delta_prime=Delta_prime,
        Delta=Delta,
        nzero=nzero,
        plasma1=plasma1,
        vacuum1=vacuum1,
        total1=total1,
        metadata={"run_dir": str(run_dir), "nc_file": nc_path.name},
    )
