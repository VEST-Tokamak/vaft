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

from ._dcon_output import _ballooning_evaluated
from ._netcdf import complex_scalar_attr, complex_var

#: Per-surface stability columns, in the order :meth:`Pest3MatchingOutput.rational_surface_stability` returns them.
_PROFILES = ("psi_n", "q", "di", "dr", "h", "ca1")
SURFACE_COLUMNS = ("m", "n", "psi_n", "q", "delta_prime_real", "delta_prime_imag", "di", "dr", "h", "ca1")


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
    #: Local stability profiles on the solver's own ``psi_n`` grid (issue #939).
    #: Both solvers write them from ``locstab%fs`` (``rdcon_netcdf.f:412-415``):
    #: ``di`` the ideal Mercier criterion D_I, ``dr`` the resistive
    #: interchange criterion D_R, ``ca1`` the high-n ballooning criterion C_A.
    #: ``dr`` is computed as ``di + (h - 1/2)**2`` (``rdcon/mercier.f:157``),
    #: so it is exactly D_R of Glasser, Greene & Johnson. ``h`` (Glasser's H)
    #: is written by RDCON only; it is ``None`` for STRIDE. ``ca1`` is NaN where
    #: the ballooning scan left the zero it was initialised with, the same rule
    #: as :attr:`DconOutput.ca1`: RDCON's own ``bal.f:53`` integrates only where
    #: ``di <= 0`` (and ``psi <= 1``), exactly like DCON's.
    psi_n: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    di: Optional[np.ndarray] = None
    dr: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None
    ca1: Optional[np.ndarray] = None

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

    def rational_surface_stability(self) -> list[dict[str, Any]]:
        """One row per rational surface: diagonal Delta-prime and the local criteria there.

        Columns are :data:`SURFACE_COLUMNS`. ``di``/``dr``/``h``/``ca1`` are
        interpolated linearly in ``psi_n`` from the solver's profiles to each
        surface's ``psi_n_rational``, and are ``None`` when the profile is
        absent. This is a table of values, not a verdict: neither a positive
        Delta-prime nor ``dr > 0`` alone decides tearing stability (#939).
        """
        rows = []
        for row in self.delta_prime_diagonal():
            for name in ("di", "dr", "h", "ca1"):
                row[name] = self._profile_at(getattr(self, name), row["psi_n"])
            rows.append({name: row[name] for name in SURFACE_COLUMNS})
        return rows

    def _profile_at(self, values: Optional[np.ndarray], psi_n: Optional[float]) -> Optional[float]:
        if values is None or self.psi_n is None or psi_n is None:
            return None
        grid = np.asarray(self.psi_n, dtype=float)
        values = np.asarray(values, dtype=float)
        # A surface beyond the solver's profile grid (it ends at psihigh, e.g.
        # 0.994) has no criteria rather than an extrapolated value.
        if values.shape != grid.shape or not grid[0] <= psi_n <= grid[-1]:
            return None
        value = float(np.interp(psi_n, grid, np.asarray(values, dtype=float)))
        return None if np.isnan(value) else value

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

        def _r(arr: Optional[np.ndarray]) -> Optional[list]:
            # NaN (unevaluated ca1) and +-inf are not valid JSON; they serialize as null.
            return None if arr is None else [float(v) if np.isfinite(v) else None for v in np.asarray(arr, dtype=float)]

        return {
            "schema": "vaft.code.gpec.Pest3MatchingOutput",
            "schema_version": 2,
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
            **{name: _r(getattr(self, name)) for name in _PROFILES},
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
            # Absent in schema_version 1 payloads.
            **{
                name: None if payload.get(name) is None else np.asarray(payload[name], dtype=float)
                for name in _PROFILES
            },
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

        profiles = {
            name: np.asarray(ds[name].values, dtype=float) if name in ds.variables else None for name in _PROFILES
        }
        if profiles["ca1"] is not None:
            profiles["ca1"] = np.where(_ballooning_evaluated(profiles["ca1"]), profiles["ca1"], np.nan)

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
        **profiles,
    )
