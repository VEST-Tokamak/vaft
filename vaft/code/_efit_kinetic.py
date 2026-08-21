"""Private kinetic-pressure implementation for :mod:`vaft.code.efit`.

This module layers a *kinetic pressure* constraint (``KPRFIT=1``) on top of the
untouched magnetic reconstruction implemented in :mod:`vaft.code.efit`.  It
follows the same Config / Inputs / Result + ``prepare`` / ``run`` / ``collect``
convention used by the other ``vaft.code`` adapters (``efit``, ``chease``,
``gpec``).

The physics logic and — critically — the exact namelist text formats are lifted
verbatim from the working ``ids_test`` batch scripts so that the emitted kfiles
are byte-compatible with what those scripts produce (EFIT changes behaviour if
the array formatting differs).  Sources:

* ``run_efit_raw5pts_batch.py``  : ``raw_points`` (p = e*ne*(Te+Ti), SIGPRE error
  propagation, error-weighted Ti fit vs psi_N), 5-point ``RPRESS=+R`` block.
* ``run_raw6pts_and_compare.py`` : 6th synthetic edge anchor
  (``RPRESS=-1.0``, ``PRESSR=0``, ``SIGPRE=0.05*max(p)``).
* ``run_kinetic_efit_batch.py``  : 129-pt psi-space spline block
  (``RPRESS=-psi_N``), ``scale_plasma`` PLASMA rescale, descending SCALES sweep,
  EFIT invocation, ``fmt`` array writer.
* ``run_kinetic_efit_test.py``   : half-cosine ``FWTPRE`` taper and the verified
  EFIT namelist semantics (``RPRESS<0`` => psi_N abscissa, effective weight =
  ``FWTPRE/SIGPRE``).

Design rules honoured here:

* :func:`build_kinetic_core_profiles` only *orchestrates* the existing
  :mod:`vaft.process.profile` fitters — it never re-fits anything.
* :func:`kinetic_pressure_points` implements all three encodings
  (``raw6`` default, ``raw5``, ``spline``).
* :func:`inject_pressure_constraint` inserts the ``KPRFIT`` block before the
  first IN1 ``/`` exactly like the batch scripts.
* :func:`run_kinetic_efit` reuses :func:`vaft.code.efit.run_efit` /
  :func:`collect_efit_outputs`; it does the descending PLASMA-scale sweep and
  keeps the *largest* converging scale.  When the EFIT executable cannot be
  resolved it returns a ``status='skipped'`` result instead of raising.
* Thomson-only shots (no IDS/charge_exchange ion data) fall back to the
  statistical ``Ti = TI_TE_RATIO_VEST * Te``
  (:data:`vaft.process.profile.TI_TE_RATIO_VEST`, fitted on the shots that
  carry both diagnostics), so the kinetic pressure becomes
  ``p = e*ne*(1+ratio)*Te`` with the ratio scatter propagated into SIGPRE.
  Control it via ``ti_te_ratio`` (``'auto'`` default / float / ``None`` =
  strict) on :func:`kinetic_pressure_points`,
  :func:`build_kinetic_core_profiles` and :class:`KineticEFITConfig`.

Heavy / cyclic dependencies (omas, vaft.process.profile, vaft.code.efit,
vaft.code.chease) are imported lazily inside the functions that need them. The
public symbols are re-exported by :mod:`vaft.code.efit`; this private module is
not a sibling external-code adapter.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Physical / formatting constants (lifted from the ids_test scripts)
# --------------------------------------------------------------------------- #

#: Elementary charge in J/eV (EFIT pressure is Pa = e * ne[m^-3] * T[eV]).
EQE = 1.602176634e-19

#: EFIT namelist array writer packs this many values per line.
#: (``VPL``/``VALS_PER_LINE`` in the scripts — always 5.)
VALS_PER_LINE = 5

#: Error-weighted polynomial degree for the Ti(R) / Ti(psi_N) fit.
TI_FIT_DEG = 3

#: CX channels beyond this psi_N (SOL) are excluded from the psi-space Ti fit.
TI_PSI_HI = 1.1

#: Spline (129-pt) encoding knobs — lifted from run_kinetic_efit_batch.py.
SPLINE_NPTS = 129
SPLINE_PSI_FULL = 0.40   # full FWTPRE weight for psi_N <= this
SPLINE_FWT_EDGE = 1.00   # FWTPRE floor at psi_N = 1 (1.0 == uniform, no taper)
SPLINE_SIG_FRAC = 0.15   # SIGPRE = SIG_FRAC * p_kin(axis), constant in Pa

#: Descending PLASMA-scale sweep; first (largest) converging scale wins.
DEFAULT_SCALES = (1.0, 0.94, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40)
def _resolve_ti_te_ratio(ti_te_ratio, ti_te_ratio_sigma=None):
    """Resolve the statistical Ti/Te fallback coefficient to (ratio, sigma).

    ``'auto'`` / ``'vest'`` -> :data:`vaft.process.profile.TI_TE_RATIO_VEST`
    (fitted on the VEST shots that carry BOTH Thomson and IDS/charge_exchange
    profiles; see ``vaft.process.profile.fit_ti_te_ratio`` for provenance and
    the estimator). A float is used as-is. ``ti_te_ratio_sigma=None`` falls
    back to ``TI_TE_RATIO_VEST_SIGMA`` (the slice-to-slice scatter).

    vaft.process.profile is imported lazily (module keeps heavy deps optional).
    """
    from vaft.process import profile as _profile

    if isinstance(ti_te_ratio, str):
        if ti_te_ratio.lower() not in ("auto", "vest"):
            raise ValueError(
                f"ti_te_ratio={ti_te_ratio!r}: expected 'auto'/'vest', a float, or None"
            )
        ratio = float(_profile.TI_TE_RATIO_VEST)
    else:
        ratio = float(ti_te_ratio)
    sigma = (
        float(_profile.TI_TE_RATIO_VEST_SIGMA)
        if ti_te_ratio_sigma is None
        else float(ti_te_ratio_sigma)
    )
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError(f"ti_te_ratio must be finite and non-negative, got {ratio!r}")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(
            f"ti_te_ratio_sigma must be finite and non-negative, got {sigma!r}"
        )
    return ratio, sigma


# --------------------------------------------------------------------------- #
# Pressure-point container
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PressurePoints:
    """Diagnostic pressure points fed to EFIT's ``KPRFIT=1`` constraint.

    Fields
    ------
    rpress : list[float]
        Abscissa per point.  ``> 0`` == real major radius R [m]; ``< 0`` ==
        ``-psi_N`` (EFIT reads the sign per element, ``pressure.F90:33-37``).
    zpress : list[float] | None
        Z [m] for the real-space (raw5/raw6) encodings (always ``0.0``); ``None``
        for the flux-space spline encoding (no ZPRESS line is emitted).
    pressr : list[float]
        Measured pressure [Pa].
    sigpre : list[float]
        Per-point sigma [Pa].
    fwtpre : list[float]
        Per-point weight; effective least-squares weight = ``FWTPRE/SIGPRE``.
    encoding : str
        One of ``{'raw6', 'raw5', 'spline'}``.
    """

    rpress: List[float]
    zpress: Optional[List[float]]
    pressr: List[float]
    sigpre: List[float]
    fwtpre: List[float]
    encoding: str = "raw6"

    def __len__(self) -> int:
        return len(self.rpress)


# --------------------------------------------------------------------------- #
# Config / Inputs / Result dataclasses (mirror vaft.code.efit shapes)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KineticEFITConfig:
    """Configuration for a kinetic-pressure EFIT run.

    ``executable`` resolves via ``config.executable`` or the ``$EFIT``
    environment variable (see :func:`find_efit_executable`).
    """

    executable: Optional[str] = None
    workdir: Path | str = "."
    shot: Optional[int] = None
    time_ms: Optional[float] = None
    encoding: str = "raw6"
    scales: Sequence[float] = DEFAULT_SCALES
    grid_size: int = 129  # EFIT grid dimension, passed as the `efit <n>` argument
    # Thomson-only fallback: Ti = ratio*Te when the ODS has no ion (IDS/CES)
    # data. 'auto' -> vaft.process.profile.TI_TE_RATIO_VEST; float -> forced;
    # None -> strict (raise). sigma None -> TI_TE_RATIO_VEST_SIGMA.
    ti_te_ratio: Any = "auto"
    ti_te_ratio_sigma: Optional[float] = None
    base_kfile: Optional[str] = None  # path to a base magnetic kfile (e.g. filedb);
    # when set, it is patched with the pressure block instead of building one from
    # the ODS via efit.generate_kfile (which needs magnetics the profile ODS lacks).
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None


@dataclass
class KineticEFITInputs:
    """Materialized inputs for a kinetic-pressure EFIT run.

    ``base_kfile_text`` is the *magnetic* kfile (from the filedb or from
    :func:`vaft.code.efit.generate_kfile`); this module text-patches it with the
    pressure constraint and the PLASMA rescale.
    """

    workdir: Path
    ods: Any = None
    geqdsk: Any = None
    base_kfile_text: str = ""
    points: Optional[PressurePoints] = None
    kfiles: tuple[Path, ...] = ()


@dataclass
class KineticEFITResult:
    """Result of the PLASMA-scale sweep."""

    returncode: Optional[int]
    workdir: Path
    gfile: Optional[Path] = None
    scale: Optional[float] = None
    converged: bool = False
    chi2: Optional[float] = None
    status: str = "ok"          # 'ok' | 'skipped' | 'error'
    reason: str = ""
    efit_result: Any = None

    @property
    def ok(self) -> bool:
        return self.converged and self.status == "ok"


# --------------------------------------------------------------------------- #
# Namelist text helpers (byte-compatible with the ids_test scripts)
# --------------------------------------------------------------------------- #

def _fmt_array(name: str, values: Sequence[float]) -> str:
    """Format a Fortran namelist float array, ``VALS_PER_LINE`` values per line.

    Byte-identical to ``fmt()`` in run_kinetic_efit_batch.py (VPL=5) and to the
    ``B.fmt`` used by the raw5 / raw6 builders.
    """
    out: List[str] = []
    line = f"{name}="
    for i, v in enumerate(values):
        if i and i % VALS_PER_LINE == 0:
            out.append(line)
            line = f" {float(v):.6e}"
        else:
            line += f" {float(v):.6e}"
    out.append(line)
    return "\n".join(out)


def _fwt(psi_n: float,
         psi_full: float = SPLINE_PSI_FULL,
         fwt_edge: float = SPLINE_FWT_EDGE) -> float:
    """Half-cosine FWTPRE taper (lifted from run_kinetic_efit_test.py:69-73).

    Full weight for ``psi_N <= psi_full`` then a half-cosine roll-off to
    ``fwt_edge`` at ``psi_N = 1``.  With ``fwt_edge == 1.0`` this is uniform.
    """
    if psi_n <= psi_full:
        return 1.0
    x = (psi_n - psi_full) / (1.0 - psi_full)
    return fwt_edge + (1.0 - fwt_edge) * 0.5 * (1.0 + math.cos(math.pi * x))


# --------------------------------------------------------------------------- #
# psi_N mapping + error-weighted Ti fits (lifted from run_efit_raw5pts_batch.py)
# --------------------------------------------------------------------------- #

def _psin_of_r(geq: Any, r_query, z: float = 0.0) -> np.ndarray:
    """Map ``R`` (at fixed ``z``) to *raw* psi_N through a VAFT GEQDSK.

    Mirrors ``psiN_of_R`` in run_efit_raw5pts_batch.py but reads the named
    GEQDSK fields (NW, NH, RDIM, RLEFT, ZDIM, ZMID, PSIRZ, SIMAG, SIBRY) instead
    of re-parsing a gfile.  Returns raw psi_N (may exceed 1 in the SOL, needed by
    the ``TI_PSI_HI`` filter) — it does NOT clip to [0, 1].
    """
    nw = int(geq["NW"])
    nh = int(geq["NH"])
    rdim = float(geq["RDIM"])
    zdim = float(geq["ZDIM"])
    rleft = float(geq["RLEFT"])
    zmid = float(geq["ZMID"])
    simag = float(geq["SIMAG"])
    sibry = float(geq["SIBRY"])
    # VAFT GEQDSK PSIRZ is stored as (nw, nh): psi[iR, iZ] (see
    # vaft.process.profile._rho_from_equilibrium_points).
    psirz = np.asarray(geq["PSIRZ"], dtype=float).reshape(nw, nh)
    r_grid = rleft + np.arange(nw) / (nw - 1) * rdim
    z_grid = (zmid - zdim / 2.0) + np.arange(nh) / (nh - 1) * zdim
    jz = int(np.argmin(np.abs(z_grid - z)))
    psi = np.interp(np.asarray(r_query, dtype=float), r_grid, psirz[:, jz])
    return (psi - simag) / (sibry - simag)


def _ti_weighted_fit(cR, cTi, csTi, R_eval):
    """Error-weighted poly fit of CX Ti(R); returns (Ti, sigma_Ti) at R_eval.

    Lifted from run_efit_raw5pts_batch.py:35-50.  Weights = 1/sigma; sigma_Ti is
    propagated from the unscaled fit covariance.  R_eval is clamped to the CX
    coverage (no blind extrapolation).
    """
    cR = np.asarray(cR, dtype=float)
    cTi = np.asarray(cTi, dtype=float)
    csTi = np.asarray(csTi, dtype=float)
    deg = min(TI_FIT_DEG, len(cR) - 1)
    w = 1.0 / np.clip(csTi, 1e-6, None)
    coef, cov = np.polyfit(cR, cTi, deg, w=w, cov="unscaled")
    x = np.clip(np.asarray(R_eval, dtype=float), cR.min(), cR.max())
    V = np.vander(x, deg + 1)
    ti = V @ coef
    sti = np.sqrt(np.einsum("ij,jk,ik->i", V, cov, V))
    return ti, sti


def _ti_weighted_fit_psin(cR, cTi, csTi, geq, R_eval, psi_hi: float = TI_PSI_HI):
    """Flux-function Ti fit: weighted poly fit of Ti vs psi_N, evaluated on curve.

    Lifted from run_efit_raw5pts_batch.py:56-81.  All CX channels are mapped
    R -> psi_N through ``geq``; Ti at ``R_eval`` is the fit evaluated at
    psi_N(R_eval) (used values lie on the fitted curve).  A channel-error floor
    is applied so SIGPRE never claims more than the data.
    """
    cR = np.asarray(cR, dtype=float)
    cTi = np.asarray(cTi, dtype=float)
    csTi = np.asarray(csTi, dtype=float)
    pcx = _psin_of_r(geq, cR)
    pev = _psin_of_r(geq, np.asarray(R_eval, dtype=float))
    keep = np.isfinite(pcx) & np.isfinite(cTi) & (pcx <= psi_hi)
    x, y, s = pcx[keep], cTi[keep], csTi[keep]
    if x.size == 0:
        raise RuntimeError(
            "no CX channels inside psi_N <= %.2f for the psi-space Ti fit" % psi_hi
        )
    deg = min(TI_FIT_DEG, x.size - 1)
    w = 1.0 / np.clip(s, 1e-6, None)
    coef, cov = np.polyfit(x, y, deg, w=w, cov="unscaled")
    xe = np.clip(pev, x.min(), min(x.max(), 1.0))
    V = np.vander(xe, deg + 1)
    ti = V @ coef
    sti = np.sqrt(np.einsum("ij,jk,ik->i", V, cov, V))
    o = np.argsort(x)
    sti = np.maximum(sti, np.interp(xe, x[o], s[o]))   # channel-error floor
    return np.maximum(ti, 0.0), sti


# --------------------------------------------------------------------------- #
# Raw TS/CX extraction (mirror raw_points(), reading from the ODS trees)
# --------------------------------------------------------------------------- #

def _raw_ne_te_ti(
    ods: Any,
    time_ms: float,
    geq: Any = None,
    *,
    ti_te_ratio: Any = "auto",
    ti_te_ratio_sigma: Optional[float] = None,
    return_fallback_meta: bool = False,
):
    """Return (R, ne, Te, sne, sTe, Ti, sTi) at the matched time.

    Mirrors ``raw_points`` in run_efit_raw5pts_batch.py:84-136 but pulls the data
    from the ODS ``thomson_scattering`` / ``charge_exchange`` trees instead of a
    JSON file.  Ti is mapped onto the TS radii: flux-function fit vs psi_N when
    ``geq`` is given, otherwise the legacy Ti(R) fit clamped to CX coverage.

    Thomson-only fallback: when the ODS carries NO usable ion data (no
    ``charge_exchange`` IDS at all, or no valid Ti channel at this time) and
    ``ti_te_ratio`` is not None, the ion temperature is estimated statistically
    as ``Ti = ratio * Te`` with
    ``sTi = sqrt((ratio*sTe)^2 + (sigma_ratio*Te)^2)`` so the ratio uncertainty
    propagates honestly into SIGPRE.  ``ti_te_ratio='auto'`` (default) resolves
    to :data:`vaft.process.profile.TI_TE_RATIO_VEST`; a float forces that
    coefficient; ``None`` restores the strict behaviour (raise when ion data is
    missing).  Slices WITH ion data are never affected.

    ``return_fallback_meta=True`` appends ``None`` for measured Ti or the
    ``(ratio, sigma_ratio)`` pair for statistical Ti. This lets the pressure
    caller account for the shared Te uncertainty without changing the legacy
    seven-value return shape.
    """
    target_s = time_ms / 1000.0

    # --- Thomson scattering ne / Te (nearest time) ---
    ts_time = np.asarray(ods["thomson_scattering.time"], dtype=float)
    it = int(np.argmin(np.abs(ts_time - target_s)))
    n_ts = len(ods["thomson_scattering.channel"])
    R, ne, Te, sne, sTe = [], [], [], [], []
    for i in range(n_ts):
        ch = ods[f"thomson_scattering.channel.{i}"]
        R.append(float(ch["position.r"]))
        ne.append(float(ch["n_e.data"][it]))
        sne.append(float(ch["n_e.data_error_upper"][it]))
        Te.append(float(ch["t_e.data"][it]))
        sTe.append(float(ch["t_e.data_error_upper"][it]))
    R = np.array(R, dtype=float)
    ne = np.array(ne, dtype=float)
    Te = np.array(Te, dtype=float)
    sne = np.array(sne, dtype=float)
    sTe = np.array(sTe, dtype=float)

    # Drop invalid Thomson channels (the shot-suffixed NeTe schema pads dead
    # polychromators with NaN) so they never reach the EFIT pressure constraint
    # -- an unfiltered NaN pressure point makes the reconstruction diverge.
    ts_valid = np.isfinite(R) & np.isfinite(ne) & np.isfinite(Te) & (ne > 0) & (Te > 0)
    if not np.any(ts_valid):
        raise RuntimeError(f"no valid Thomson channels in ODS at {time_ms} ms")
    R, ne, Te, sne, sTe = R[ts_valid], ne[ts_valid], Te[ts_valid], sne[ts_valid], sTe[ts_valid]
    # Preserve otherwise valid channels whose uncertainty is absent or invalid,
    # but assign a conservative 10% fallback so EFIT never receives SIGPRE=NaN.
    sne = np.where(np.isfinite(sne) & (sne > 0.0), sne, 0.1 * np.abs(ne))
    sTe = np.where(np.isfinite(sTe) & (sTe > 0.0), sTe, 0.1 * np.abs(Te))

    # --- Charge exchange Ti (nearest time per channel); statistical Ti = ratio*Te
    #     fallback when the shot has no usable ion data (Thomson-only) ---
    fallback_meta = None
    try:
        n_cx = len(ods["charge_exchange.channel"])
    except Exception:
        n_cx = 0

    cR, cTi, csTi = [], [], []
    for i in range(n_cx):
        try:
            ch = ods[f"charge_exchange.channel.{i}"]
            rt = np.asarray(ch["position.r.time"], dtype=float)
            if rt.size == 0:
                raise ValueError("empty position.r.time")
            jr = int(np.argmin(np.abs(rt - target_s)))
            ion = ch[f"ion.{0}"]
            ti = np.asarray(ion["t_i.data"], dtype=float)
            if ti.size == 0:
                raise ValueError("empty t_i.data")
            try:
                sti = np.asarray(ion["t_i.data_error_upper"], dtype=float)
            except Exception:
                sti = np.zeros_like(ti)
            j = min(jr, len(ti) - 1)
            v = ti[j]
            r = float(ch["position.r.data"][jr])
            if not np.isfinite(r) or not np.isfinite(v):
                raise ValueError("non-finite position or Ti")
            s = sti[j] if j < len(sti) else None
        except Exception as exc:  # noqa: BLE001 -- one dead channel must not hide good CX data
            print(
                f"[INFO] skipped invalid CX Ti channel {i} at {time_ms} ms ({exc})"
            )
            continue
        cR.append(r)
        cTi.append(float(v))
        csTi.append(
            float(s) if (s is not None and np.isfinite(s) and s > 0) else 0.1 * abs(v)
        )

    if cR:
        cR = np.array(cR, dtype=float)
        cTi = np.array(cTi, dtype=float)
        csTi = np.array(csTi, dtype=float)
        o = np.argsort(cR)

        if geq is not None:
            Ti, sTi = _ti_weighted_fit_psin(cR, cTi, csTi, geq, R)
        else:
            try:
                Ti, sTi = _ti_weighted_fit(cR[o], cTi[o], csTi[o], R)
                # fit-curve sigma can undershoot the channel errors; keep the local
                # measurement error as a floor so SIGPRE never claims more than data.
                sTi = np.maximum(
                    sTi, np.interp(np.clip(R, cR.min(), cR.max()), cR[o], csTi[o])
                )
            except np.linalg.LinAlgError:
                Ti = np.interp(R, cR[o], cTi[o])
                sTi = np.interp(R, cR[o], csTi[o])
    else:
        if ti_te_ratio is None:
            raise RuntimeError(
                f"no valid CX Ti channels in ODS at {time_ms} ms"
            )
        ratio, rsig = _resolve_ti_te_ratio(ti_te_ratio, ti_te_ratio_sigma)
        print(
            f"[INFO] kinetic pressure at {time_ms} ms: no usable ion "
            f"(IDS/charge_exchange) data; statistical fallback "
            f"Ti = {ratio:.3f}*Te (sigma_ratio {rsig:.3f})"
        )
        Ti = ratio * Te
        sTi = np.sqrt((ratio * sTe) ** 2 + (rsig * Te) ** 2)
        fallback_meta = (ratio, rsig)
    Ti = np.maximum(Ti, 0.0)
    values = (R, ne, Te, sne, sTe, Ti, sTi)
    return (*values, fallback_meta) if return_fallback_meta else values


def _spline_pressure(ods: Any, time_ms: float, npts: int = SPLINE_NPTS):
    """Return (psi_N grid [129], p_kin [Pa]) resampled from core_profiles.

    The 129-pt psi-space profile in the batch script comes from a saved
    ``kinetic_pressure_*.txt`` (col0 psi_N, col7 p_kin).  Here the equivalent
    profile is read from ``core_profiles.profiles_1d`` (``pressure_thermal`` =
    e*ne*(Te+Ti), written by :func:`build_kinetic_core_profiles`) and resampled
    onto a uniform 129-pt psi_N grid.
    """
    target_s = time_ms / 1000.0
    n = len(ods["core_profiles.profiles_1d"])
    if n == 0:
        raise RuntimeError("no core_profiles.profiles_1d in ODS for spline encoding")
    idx, best = 0, float("inf")
    for i in range(n):
        t = float(ods[f"core_profiles.profiles_1d.{i}.time"])
        d = abs(t - target_s)
        if d < best:
            best, idx = d, i
    base = f"core_profiles.profiles_1d.{idx}"

    if f"{base}.grid.psi" in ods:
        psi = np.asarray(ods[f"{base}.grid.psi"], dtype=float)
        psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
    else:
        rho_pol = np.asarray(ods[f"{base}.grid.rho_pol_norm"], dtype=float)
        psi_n = rho_pol ** 2

    p = np.asarray(ods[f"{base}.pressure_thermal"], dtype=float)
    grid = np.linspace(0.0, 1.0, npts)
    order = np.argsort(psi_n)
    p_grid = np.interp(grid, psi_n[order], p[order])
    return grid, np.maximum(p_grid, 0.0)


# --------------------------------------------------------------------------- #
# Public: kinetic pressure points
# --------------------------------------------------------------------------- #

def kinetic_pressure_points(
    ods: Any,
    time_ms: float,
    geq: Any = None,
    *,
    encoding: str = "raw6",
    sigma_floor: float = 0.05,
    ti_te_ratio: Any = "auto",
    ti_te_ratio_sigma: Optional[float] = None,
) -> PressurePoints:
    """Build the EFIT ``KPRFIT`` pressure points for the requested encoding.

    Encodings
    ---------
    ``raw6`` (default)
        5 real-space TS points (``RPRESS=+R``, ``ZPRESS=0``) plus a synthetic
        edge anchor (``RPRESS=-1.0`` i.e. psi_N=1, ``PRESSR=0``,
        ``SIGPRE=sigma_floor*max(p)``) pinning the separatrix pressure to 0.
    ``raw5``
        The 5 real-space TS points only.
    ``spline``
        129-pt psi-space profile (``RPRESS=-psi_N``) with the half-cosine
        ``FWTPRE`` taper and constant ``SIGPRE = SPLINE_SIG_FRAC * p(axis)``.

    For measured Ti in the raw encodings ``p = e*ne*(Te+Ti)`` and
    ``SIGPRE = e*sqrt(((Te+Ti)*sne)^2 + (ne*sTe)^2 + (ne*sTi)^2)`` floored at
    ``sigma_floor * p`` (run_efit_raw5pts_batch.py:129-131).

    Thomson-only shots (no IDS/charge_exchange ion data): with the default
    ``ti_te_ratio='auto'`` the raw encodings substitute the statistical
    ``Ti = TI_TE_RATIO_VEST * Te`` (see :func:`_raw_ne_te_ti`), i.e.
    ``p = e*ne*(1+ratio)*Te`` with the ratio uncertainty propagated into
    SIGPRE and the shared Te error counted once. Pass ``ti_te_ratio=None`` to
    restore the strict behaviour
    (raise when ion data is missing) or a float to force a coefficient.
    Shots with ion data are unaffected.
    """
    enc = encoding.lower()
    if enc in ("raw6", "raw5"):
        R, ne, Te, sne, sTe, Ti, sTi, fallback_meta = _raw_ne_te_ti(
            ods, time_ms, geq,
            ti_te_ratio=ti_te_ratio, ti_te_ratio_sigma=ti_te_ratio_sigma,
            return_fallback_meta=True,
        )
        p = ne * (Te + Ti) * EQE
        if fallback_meta is None:
            sig = EQE * np.sqrt(
                ((Te + Ti) * sne) ** 2 + (ne * sTe) ** 2 + (ne * sTi) ** 2
            )
        else:
            # Ti = ratio*Te shares the same Te measurement, so Te and Ti are
            # correlated. Propagate p=e*ne*(1+ratio)*Te directly instead of
            # treating their temperature uncertainties as independent.
            ratio, ratio_sigma = fallback_meta
            sig = EQE * np.sqrt(
                ((1.0 + ratio) * Te * sne) ** 2
                + (ne * (1.0 + ratio) * sTe) ** 2
                + (ne * Te * ratio_sigma) ** 2
            )
        sig = np.maximum(sig, sigma_floor * p)   # floor (default 5%)

        rpress = [float(x) for x in R]
        zpress = [0.0] * len(R)
        pressr = [float(x) for x in p]
        sigpre = [float(x) for x in sig]
        fwtpre = [1.0] * len(R)

        if enc == "raw6":
            # synthetic edge anchor: separatrix p=0 pinned (run_raw6pts_and_compare.py:33-35)
            s_edge = sigma_floor * float(np.max(p))
            rpress.append(-1.0)
            zpress.append(0.0)
            pressr.append(0.0)
            sigpre.append(s_edge)
            fwtpre.append(1.0)
        return PressurePoints(rpress, zpress, pressr, sigpre, fwtpre, enc)

    if enc == "spline":
        psi_n, p = _spline_pressure(ods, time_ms)
        s0 = SPLINE_SIG_FRAC * float(p[0])
        rpress = [float(-x) for x in psi_n]        # negative => psi_N abscissa
        pressr = [float(x) for x in p]
        sigpre = [s0] * len(psi_n)
        fwtpre = [float(_fwt(x)) for x in psi_n]
        return PressurePoints(rpress, None, pressr, sigpre, fwtpre, enc)

    raise ValueError(f"unknown encoding {encoding!r}; expected raw6, raw5 or spline")


# --------------------------------------------------------------------------- #
# Public: kfile text patching
# --------------------------------------------------------------------------- #

def inject_pressure_constraint(
    kfile_text: str,
    points: PressurePoints,
    *,
    encoding: str = "raw6",
) -> str:
    """Insert the ``KPRFIT=1`` pressure block before the first IN1 ``/``.

    The block layout matches the batch builders exactly:
    ``KPRFIT= 1``, ``NPRESS= N``, ``KPRESSB= 0``, then ``RPRESS`` (and
    ``ZPRESS`` for the real-space encodings), ``PRESSR``, ``SIGPRE``, ``FWTPRE``.
    ``ZPRESS`` is omitted when ``points.zpress is None`` (spline / flux-space),
    matching ``add_pressure`` in run_kinetic_efit_batch.py:73-83.
    """
    n = len(points.rpress)
    block = [
        "KPRFIT= 1",
        f"NPRESS= {n}",
        "KPRESSB= 0",
        _fmt_array("RPRESS", points.rpress),
    ]
    if points.zpress is not None:
        block.append(_fmt_array("ZPRESS", points.zpress))
    block.append(_fmt_array("PRESSR", points.pressr))
    block.append(_fmt_array("SIGPRE", points.sigpre))
    block.append(_fmt_array("FWTPRE", points.fwtpre))

    lines = kfile_text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == "/":
            lines[i:i] = block
            return "\n".join(lines) + "\n"
    raise RuntimeError("no IN1 terminator '/' found in kfile text")


def scale_plasma(kfile_text: str, scale: float) -> str:
    """Rescale the ``PLASMA=`` value by ``scale`` (returns patched text).

    Lifted from run_kinetic_efit_batch.py:86-89 (which also returned the original
    value; here we return only the patched text per the frozen signature).
    """
    m = re.search(r"(?m)^\s*PLASMA\s*=\s*([-+0-9.eE]+)", kfile_text)
    if m is None:
        raise RuntimeError("no PLASMA= line found in kfile text")
    orig = float(m.group(1))
    return re.sub(r"(?m)^\s*PLASMA\s*=.*$", f"PLASMA= {orig * scale}", kfile_text)


# --------------------------------------------------------------------------- #
# Executable resolution (canonical $EFITHOME with legacy $EFIT compatibility)
# --------------------------------------------------------------------------- #

def _resolve_efit_executable(config: KineticEFITConfig) -> Optional[Path]:
    """Resolve EFIT through the canonical adapter configuration."""
    from vaft.code import efit as _efit

    return _efit.find_efit_executable(
        _efit.EFITConfig(
            executable=config.executable,
            workdir=config.workdir,
            shot=config.shot,
            env=config.env,
            timeout=config.timeout,
        )
    )


def find_efit_executable(config: Optional[KineticEFITConfig] = None) -> Optional[Path]:
    """Public wrapper mirroring :func:`vaft.code.chease.find_chease_executable`."""
    return _resolve_efit_executable(config or KineticEFITConfig())


def _parse_chi2(logs: Sequence[Path]) -> Optional[float]:
    """Parse ``totalchi^2`` from EFIT logs (regex from run_raw6pts_and_compare.py:51)."""
    pat = re.compile(r"totalchi\^2 =\s*([0-9.Ee+-]+)")
    for lg in logs:
        try:
            text = Path(lg).read_text(errors="ignore")
        except Exception:
            continue
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Core-profiles orchestration (pure ODS -> ODS)
# --------------------------------------------------------------------------- #

def build_kinetic_core_profiles(
    ods: Any,
    geq: Any,
    time_ms: float,
    *,
    te_mode: str = "polynomial",
    ne_mode: str = "free_exponential",
    ti_mode: str = "polynomial",
    vtor_mode: str = "polynomial",
    ion_index: int = 0,
    time_tolerance_ms: float = 1.0,
    require_thomson: bool = False,
    require_ion: bool = False,
    ti_te_fallback: bool = True,
    ti_te_ratio: Any = "auto",
) -> Any:
    """Fit whatever kinetic diagnostics are present and write ``core_profiles.profiles_1d``.

    Orchestrates (does NOT re-implement) the :mod:`vaft.process.profile` pipeline::

        equilibrium_mapping_thomson_scattering -> profile_fitting_thomson_scattering
        equilibrium_mapping_charge_exchange   -> profile_fitting_charge_exchange
          -> core_profiles(...)

    Both diagnostics are OPTIONAL: Thomson scattering supplies ne/Te (electrons)
    and the charge_exchange (IDS/CES) supplies Ti/Vtor (ions). Whatever is present
    is written -- ion-only gives ion temperature/velocity, both give the full
    kinetic slice with ``pressure_thermal``. At least one usable diagnostic is
    required (or set ``require_thomson`` / ``require_ion`` to fail loudly when
    one is missing).

    Thomson-only slices (``ti_te_fallback=True`` and no usable ion fit): the ion
    temperature falls back to ``Ti = ti_te_ratio * Te``. The default
    ``ti_te_ratio='auto'`` resolves to the statistical
    :data:`vaft.process.profile.TI_TE_RATIO_VEST` (fitted on the shots carrying
    both diagnostics) and -- unlike the legacy ``Ti=Te`` fallback -- also writes
    ``pressure_thermal = e*ne*(1+ratio)*Te``, so the kinetic-EFIT spline
    encoding works for TS-only shots. Pass ``ti_te_ratio=None`` to restore the
    legacy ``Ti=Te`` fallback (no pressure written), or a float to force a
    coefficient. Slices with a real ion fit are unaffected.

    Pure ODS -> ODS: the passed ``ods`` is mutated in place and returned.

    ``time_tolerance_ms`` is forwarded to ``core_profiles`` for matching the
    equilibrium slice; widen it when the equilibrium and the diagnostic sample
    are at slightly different times.
    """
    from vaft.process import profile as _profile

    n_e_function = T_e_function = T_i_function = V_tor_function = None
    rho_ts = rho_cx = None

    # --- electrons: Thomson scattering (optional) ---
    if "thomson_scattering" in ods and len(ods["thomson_scattering.channel"]) > 0:
        try:
            rho_ts = _profile.equilibrium_mapping_thomson_scattering(ods, geq)
            n_e_function, T_e_function, *_ = _profile.profile_fitting_thomson_scattering(
                ods, time_ms, rho_ts,
                fitting_function_te=te_mode, fitting_function_ne=ne_mode,
            )
        except Exception as exc:  # noqa: BLE001
            if require_thomson:
                raise
            print(f"[INFO] core_profiles at {time_ms} ms: no Thomson fit ({exc})")
            n_e_function = T_e_function = rho_ts = None
    elif require_thomson:
        raise ValueError(f"no thomson_scattering in ODS at {time_ms} ms")

    # --- ions: charge_exchange / IDS / CES (optional) ---
    if "charge_exchange" in ods and len(ods["charge_exchange.channel"]) > 0:
        try:
            rho_cx = _profile.equilibrium_mapping_charge_exchange(ods, geq)
            V_tor_function, T_i_function, *_ = _profile.profile_fitting_charge_exchange(
                ods, time_ms, rho_cx,
                fitting_function_ti=ti_mode, fitting_function_vtor=vtor_mode,
                ion_index=ion_index,
            )
        except Exception as exc:  # noqa: BLE001
            if require_ion:
                raise
            print(f"[INFO] core_profiles at {time_ms} ms: no ion (charge_exchange) fit ({exc})")
            T_i_function = V_tor_function = rho_cx = None
    elif require_ion:
        raise ValueError(f"no charge_exchange in ODS at {time_ms} ms")

    if n_e_function is None and T_i_function is None:
        raise RuntimeError(
            f"no usable Thomson or charge_exchange fit at {time_ms} ms -- nothing to write"
        )

    # Resolve the statistical fallback coefficient only when it can be used
    # (Thomson-only slice with the fallback enabled); None keeps legacy Ti=Te.
    ratio = None
    if ti_te_ratio is not None and T_i_function is None and ti_te_fallback:
        ratio, _ = _resolve_ti_te_ratio(ti_te_ratio)

    _profile.core_profiles(
        ods,
        time_ms,
        rho_ts,
        n_e_function,
        T_e_function,
        T_i_function=T_i_function,
        V_tor_function=V_tor_function,
        ti_mapped_rho_position=rho_cx,
        time_tolerance_ms=time_tolerance_ms,
        geq=geq,
        ti_te_fallback=ti_te_fallback,
        ti_te_ratio=ratio,
    )
    return ods


# --------------------------------------------------------------------------- #
# prepare / run
# --------------------------------------------------------------------------- #

def _select_kfile(kfiles: Sequence[Path], time_ms: Optional[float]) -> Optional[Path]:
    """Pick the kfile whose ``.<time>`` suffix is closest to ``time_ms``."""
    if not kfiles:
        return None
    if time_ms is None:
        return kfiles[0]
    best, best_d = kfiles[0], float("inf")
    for k in kfiles:
        suffix = Path(k).name.split(".")[-1]
        try:
            t = float(suffix)
        except ValueError:
            continue
        d = abs(t - time_ms)
        if d < best_d:
            best_d, best = d, k
    return best


def prepare_kinetic_efit_inputs(
    ods: Any,
    geq: Any,
    config: KineticEFITConfig,
) -> KineticEFITInputs:
    """Generate the base magnetic kfile and build the kinetic pressure points.

    The base (magnetic) kfile text is produced with
    :func:`vaft.code.efit.generate_kfile` when the caller has not supplied one;
    the pressure points are built with :func:`kinetic_pressure_points` for the
    encoding in ``config``.
    """
    from vaft.code import efit as _efit

    workdir = Path(config.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    shot = config.shot
    if shot is None:
        shot = _efit._infer_shot(ods, None)

    if config.time_ms is None:
        raise ValueError("KineticEFITConfig.time_ms is required to build pressure points")

    if config.base_kfile is not None:
        # Patch a supplied base magnetic kfile (e.g. from the VEST filedb). The
        # profile-only ODS carries no magnetics, so generate_kfile cannot build one.
        base_kfile = Path(config.base_kfile).expanduser()
        base_text = base_kfile.read_text()
        kfiles = (base_kfile,)
    else:
        # Base magnetic kfile via efit.generate_kfile (needs magnetics in the ODS).
        _efit.generate_kfile(ods, shot, 2, 2, save_dir=str(workdir))
        kfiles = _efit._find_outputs(workdir, "k", shot)
        base_kfile = _select_kfile(kfiles, config.time_ms)
        base_text = Path(base_kfile).read_text() if base_kfile is not None else ""

    points = kinetic_pressure_points(
        ods, config.time_ms, geq, encoding=config.encoding,
        ti_te_ratio=config.ti_te_ratio,
        ti_te_ratio_sigma=config.ti_te_ratio_sigma,
    )

    ordered = ((base_kfile,) if base_kfile is not None else ()) + tuple(
        k for k in kfiles if k != base_kfile
    )
    return KineticEFITInputs(
        workdir=workdir,
        ods=ods,
        geqdsk=geq,
        base_kfile_text=base_text,
        points=points,
        kfiles=ordered,
    )


def run_kinetic_efit(
    inputs: KineticEFITInputs,
    config: KineticEFITConfig,
) -> KineticEFITResult:
    """Descending PLASMA-scale sweep; keep the largest converging scale.

    Reuses :func:`vaft.code.efit.run_efit` (which internally calls
    ``collect_efit_outputs``) for each scale.  Convergence is detected from a
    produced g-file; ``chi2`` is parsed from the EFIT log when available.  When
    the EFIT executable cannot be resolved (``$EFITHOME`` and ``$EFIT`` unset and
    ``config.executable`` None) a ``status='skipped'`` result is returned instead
    of raising.
    """
    from vaft.code import efit as _efit

    workdir = Path(config.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    exe = _resolve_efit_executable(config)
    if exe is None:
        return KineticEFITResult(
            returncode=None,
            workdir=workdir,
            status="skipped",
            reason=_efit._efit_unconfigured_reason(),
        )

    if not inputs.base_kfile_text:
        return KineticEFITResult(
            returncode=None,
            workdir=workdir,
            status="error",
            reason="no base kfile text to patch (inputs.base_kfile_text is empty)",
        )

    points = inputs.points
    if points is None:
        return KineticEFITResult(
            returncode=None,
            workdir=workdir,
            status="error",
            reason="no pressure points on inputs.points",
        )

    if inputs.kfiles:
        kname = Path(inputs.kfiles[0]).name
    elif config.shot is not None and config.time_ms is not None:
        kname = f"k0{config.shot}.{int(round(config.time_ms)):05d}"
    else:
        kname = "kfile.in"

    last: Any = None
    for scale in config.scales:      # descending: first success == max Ip
        scaled = scale_plasma(inputs.base_kfile_text, scale)
        patched = inject_pressure_constraint(scaled, points, encoding=config.encoding)

        scale_dir = workdir / f"s{int(round(scale * 100))}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        kpath = scale_dir / kname
        kpath.write_text(patched)

        efit_config = _efit.EFITConfig(
            executable=str(exe),
            workdir=scale_dir,
            shot=config.shot,
            args=(str(int(config.grid_size)),),  # EFIT needs the grid size: `efit <n>`
            env=config.env,
            timeout=config.timeout,
        )
        efit_inputs = _efit.EFITInputs(
            workdir=scale_dir, kfiles=(kpath,), files=(kpath,)
        )
        efit_result = _efit.run_efit(efit_inputs, efit_config)
        last = efit_result

        if efit_result.gfiles:
            gfile = Path(efit_result.gfiles[0])
            chi2 = _parse_chi2(efit_result.logs)
            return KineticEFITResult(
                returncode=efit_result.returncode,
                workdir=scale_dir,
                gfile=gfile,
                scale=float(scale),
                converged=True,
                chi2=chi2,
                status="ok",
                efit_result=efit_result,
            )

    return KineticEFITResult(
        returncode=(last.returncode if last is not None else None),
        workdir=workdir,
        converged=False,
        status="ok",
        reason="no PLASMA scale converged",
        efit_result=last,
    )


# --------------------------------------------------------------------------- #
# End-to-end chain
# --------------------------------------------------------------------------- #

def run_kinetic_chain(
    ods: Any,
    geq: Any,
    time_ms: float,
    *,
    efit_config: KineticEFITConfig,
    chease_config: Any = None,
    **profile_modes: Any,
) -> dict:
    """Full kinetic chain: profiles -> kinetic EFIT -> optional CHEASE refine.

    Steps:
      1. :func:`build_kinetic_core_profiles` (accepts ``te_mode`` / ``ne_mode`` /
         ``ti_mode`` / ``vtor_mode`` / ``ion_index`` via ``**profile_modes``).
      2. :func:`prepare_kinetic_efit_inputs` + :func:`run_kinetic_efit`.
      3. When ``chease_config`` is given and the EFIT step converged,
         :func:`vaft.code.chease.refine_equilibrium` on the produced g-file.

    Returns ``{'ods', 'kinetic_efit': KineticEFITResult, 'chease': CHEASEResult|None}``.
    """
    from dataclasses import replace

    ods = build_kinetic_core_profiles(ods, geq, time_ms, **profile_modes)

    if efit_config.time_ms is None:
        efit_config = replace(efit_config, time_ms=time_ms)

    inputs = prepare_kinetic_efit_inputs(ods, geq, efit_config)
    kinetic_result = run_kinetic_efit(inputs, efit_config)

    chease_result = None
    if chease_config is not None and kinetic_result.ok and kinetic_result.gfile is not None:
        from vaft.code import chease as _chease

        chease_result = _chease.refine_equilibrium(kinetic_result.gfile, chease_config)

    return {
        "ods": ods,
        "kinetic_efit": kinetic_result,
        "chease": chease_result,
    }


__all__ = [
    "EQE",
    "PressurePoints",
    "KineticEFITConfig",
    "KineticEFITInputs",
    "KineticEFITResult",
    "build_kinetic_core_profiles",
    "kinetic_pressure_points",
    "inject_pressure_constraint",
    "scale_plasma",
    "find_efit_executable",
    "prepare_kinetic_efit_inputs",
    "run_kinetic_efit",
    "run_kinetic_chain",
]
