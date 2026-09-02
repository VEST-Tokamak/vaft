"""Python-first CHEASE equilibrium-refinement workflow adapter."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vaft.formula.statistics import rms

from ._executables import executable_from_home, missing_home_message
from .base import CodeConfig, CodeInputs, CodeResult, CodeRunner

MU0 = 4.0e-7 * np.pi
NCHEASE = 401
CHEASE_HOME_ENV = "CHEASEHOME"
CHEASE_HOME_EXECUTABLE = Path("bin/chease")
CHEASE_COMPATIBILITY_ENVS = ("CHEASE", "CHEASE_EXEC_DIR")


@dataclass(frozen=True)
class CHEASEConfig:
    """Runtime and numerical configuration for CHEASE refinement."""

    executable: Optional[str] = None
    workdir: Path | str = Path(".")
    target_psin: float = 0.993
    relax: float = 0.5
    nideal: int = 11
    nw: int = 513
    # --- jsk95 parity knobs (defaults reproduce eqdsk.py run_mode='jsk95') ---
    epslon_exponent: int = 10  # emits EPSLON=1.0E-{exp}; jsk95 uses 10
    q95_constraint: bool = True  # drive QSPEC/CSSPEC from q at psi_N=sqrt(q95_psin)
    q95_psin: float = 0.95  # jsk95 constraint location (qloc = sqrt(this))
    ncscal: int = 1  # CHEASE current-scaling mode
    edge_zero: bool = True  # zero positive edge pprime, flatten FF' inward
    boundary_smoothing: str = "fft"  # "fft" (smooth_bnd) | "arclength" | "none"
    boundary_fft_num: int = 128  # smooth_bnd nf
    auto_cocos: bool = True
    output_cocos: str = "input"
    preserve_boundary_limiter: bool = True
    create_plot: bool = True
    cleanup: bool = False
    env: Mapping[str, str] = field(default_factory=dict)
    args: Sequence[str] = ()
    timeout: Optional[float] = None


@dataclass
class CHEASEInputs:
    """Materialized CHEASE input bundle."""

    workdir: Path
    source: Any = None
    input_geqdsk: Path | None = None
    expeq: Path | None = None
    namelist: Path | None = None
    geqdsk: Any = None
    files: tuple[Path, ...] = ()
    manifests: tuple[Path, ...] = ()


@dataclass
class CHEASEResult:
    """CHEASE run status, produced files, and parsed refined equilibrium."""

    returncode: Optional[int]
    workdir: Path
    input_geqdsk: Path | None = None
    refined_geqdsk: Path | None = None
    refined_ods: Any = None
    logs: tuple[Path, ...] = ()
    manifests: tuple[Path, ...] = ()
    figures: tuple[Path, ...] = ()
    outputs: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)
    comparison: Mapping[str, float] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class GeqdskSignInfo:
    """Compact sign-convention diagnostics for a GEQDSK object."""

    psi_axis: float
    psi_boundary: float
    dpsi: float
    bcentr: float
    current: float
    fpol_median: float
    q_median: float
    ffprim_median: float
    pprime_median: float

    @property
    def dpsi_sign(self) -> int:
        return _sign(self.dpsi)

    @property
    def bcentr_sign(self) -> int:
        return _sign(self.bcentr)

    @property
    def current_sign(self) -> int:
        return _sign(self.current)

    @property
    def fpol_sign(self) -> int:
        return _sign(self.fpol_median)

    @property
    def q_sign(self) -> int:
        return _sign(self.q_median)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "psi_axis": self.psi_axis,
            "psi_boundary": self.psi_boundary,
            "dpsi": self.dpsi,
            "dpsi_sign": self.dpsi_sign,
            "bcentr": self.bcentr,
            "bcentr_sign": self.bcentr_sign,
            "current": self.current,
            "current_sign": self.current_sign,
            "fpol_median": self.fpol_median,
            "fpol_sign": self.fpol_sign,
            "q_median": self.q_median,
            "q_sign": self.q_sign,
            "ffprim_median": self.ffprim_median,
            "pprime_median": self.pprime_median,
        }


#: The convention CHEASE expects, declared once in the shared registry.
#:
#: CHEASE works in normalised units with Ip and B0 positive (it is handed |Ip|
#: and |BCENTR| with SIGNIPXP = SIGNB0XP = 1), so its g-file input is oriented to
#: Ip < 0, B0 > 0 in the COCOS 2 system.  See Sauter Sect. IX for the index and
#: Eq. 22 for the input consistency conditions.
CHEASE_ORIENTATION = {"sigma_ip": -1, "sigma_b0": +1}


def _desired_signs_for_cocos(cocos: int, *, sigma_ip: int, sigma_b0: int) -> dict[str, int]:
    """The g-file sign pattern of an equilibrium in ``cocos`` with these orientations.

    Every entry follows from Sauter Eq. 23 once the index and the two
    orientation signs are fixed; none of them is an independent choice.
    """
    from vaft.data.cocos import cocos_spec

    spec = cocos_spec(cocos)
    return {
        "dpsi": spec.expected_sign("dpsi", sigma_ip=sigma_ip, sigma_b0=sigma_b0),
        "bcentr": sigma_b0,
        "current": sigma_ip,
        "fpol": spec.expected_sign("f", sigma_ip=sigma_ip, sigma_b0=sigma_b0),
        "q": spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=sigma_b0),
    }


#: Kept as a module-level name because the Snakemake workflow and the JSON
#: manifests refer to it; it is now derived rather than hand-maintained.
CHEASE_COCOS02_SIGNS = _desired_signs_for_cocos(2, **CHEASE_ORIENTATION)


def _sign(value: float, *, default: int = 1) -> int:
    if not np.isfinite(value) or abs(float(value)) < 1e-14:
        return default
    return 1 if float(value) > 0.0 else -1


def _finite_median(values: Any, *, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (np.abs(arr) > 1e-14)]
    if arr.size == 0:
        return float(default)
    return float(np.nanmedian(arr))


def _copy_geqdsk(geqdsk: Any):
    from vaft.data.eqdsk import GEQDSK

    mapping = {}
    for key, value in geqdsk.mapping.items():
        if isinstance(value, np.ndarray):
            mapping[key] = value.copy()
        else:
            mapping[key] = value
    return GEQDSK(mapping, source=geqdsk.source, metadata=dict(geqdsk.metadata))


def _coerce_geqdsk(source: Any):
    from vaft.data.eqdsk import GEQDSK, from_omas, read_geqdsk

    if isinstance(source, GEQDSK):
        return _copy_geqdsk(source)
    if isinstance(source, (str, Path)):
        return read_geqdsk(Path(source).expanduser())
    if isinstance(source, Mapping) and {"NW", "NH", "PSIRZ"}.issubset(source.keys()):
        return GEQDSK(dict(source), metadata={"parser": "mapping"})
    return from_omas(source)


def _geqdsk_sign_info(geqdsk: Any) -> GeqdskSignInfo:
    return GeqdskSignInfo(
        psi_axis=float(geqdsk["SIMAG"]),
        psi_boundary=float(geqdsk["SIBRY"]),
        dpsi=float(geqdsk["SIBRY"]) - float(geqdsk["SIMAG"]),
        bcentr=float(geqdsk["BCENTR"]),
        current=float(geqdsk["CURRENT"]),
        fpol_median=_finite_median(geqdsk["FPOL"], default=float(geqdsk["BCENTR"])),
        q_median=_finite_median(geqdsk["QPSI"]),
        ffprim_median=_finite_median(geqdsk["FFPRIM"]),
        pprime_median=_finite_median(geqdsk["PPRIME"]),
    )


def _desired_signs_from_info(info: GeqdskSignInfo) -> dict[str, int]:
    return {
        "desired_dpsi_sign": info.dpsi_sign,
        "desired_bcentr_sign": info.bcentr_sign,
        "desired_current_sign": info.current_sign,
        "desired_fpol_sign": info.fpol_sign,
        "desired_q_sign": info.q_sign,
    }


def _desired_signs_for_chease() -> dict[str, int]:
    """CHEASE's input sign pattern, derived from its declared COCOS."""
    from vaft.data.cocos import convention_for

    signs = _desired_signs_for_cocos(convention_for("chease").cocos, **CHEASE_ORIENTATION)
    return {f"desired_{name}_sign": value for name, value in signs.items()}


def _force_geqdsk_signs(
    geqdsk: Any,
    *,
    desired_dpsi_sign: int,
    desired_bcentr_sign: int,
    desired_current_sign: int,
    desired_fpol_sign: int,
    desired_q_sign: int,
):
    item = _copy_geqdsk(geqdsk)
    before = _geqdsk_sign_info(item)

    psi_factor = desired_dpsi_sign * before.dpsi_sign
    if psi_factor < 0:
        item["SIMAG"] = -float(item["SIMAG"])
        item["SIBRY"] = -float(item["SIBRY"])
        item["PSIRZ"] = -np.asarray(item["PSIRZ"], dtype=float)
        item["FFPRIM"] = -np.asarray(item["FFPRIM"], dtype=float)
        item["PPRIME"] = -np.asarray(item["PPRIME"], dtype=float)

    bcentr_factor = desired_bcentr_sign * before.bcentr_sign
    if bcentr_factor < 0:
        item["BCENTR"] = -float(item["BCENTR"])

    current_factor = desired_current_sign * before.current_sign
    if current_factor < 0:
        item["CURRENT"] = -float(item["CURRENT"])

    fpol_factor = desired_fpol_sign * before.fpol_sign
    if fpol_factor < 0:
        item["FPOL"] = -np.asarray(item["FPOL"], dtype=float)

    q_factor = desired_q_sign * before.q_sign
    if q_factor < 0:
        item["QPSI"] = -np.asarray(item["QPSI"], dtype=float)

    after = _geqdsk_sign_info(item)
    transform = {
        "psi": int(psi_factor),
        "bcentr": int(bcentr_factor),
        "current": int(current_factor),
        "fpol": int(fpol_factor),
        "q": int(q_factor),
    }
    return item, before, after, transform


def _profile(values: Any, size: int = NCHEASE) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot interpolate an empty GEQDSK profile")
    x = np.linspace(0.0, 1.0, arr.size)
    target = np.linspace(0.0, 1.0, size)
    return np.interp(target, x, arr)


def _grid_arrays(geqdsk: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nw = int(geqdsk["NW"])
    nh = int(geqdsk["NH"])
    r = np.linspace(float(geqdsk["RLEFT"]), float(geqdsk["RLEFT"]) + float(geqdsk["RDIM"]), nw)
    z = np.linspace(
        float(geqdsk["ZMID"]) - float(geqdsk["ZDIM"]) / 2.0,
        float(geqdsk["ZMID"]) + float(geqdsk["ZDIM"]) / 2.0,
        nh,
    )
    psi = np.asarray(geqdsk["PSIRZ"], dtype=float)
    if psi.shape != (nw, nh):
        psi = psi.reshape(nw, nh)
    return r, z, psi


def _polygon_area(rz: np.ndarray) -> float:
    if rz.shape[0] < 3:
        return 0.0
    return float(0.5 * abs(np.dot(rz[:, 0], np.roll(rz[:, 1], -1)) - np.dot(rz[:, 1], np.roll(rz[:, 0], -1))))


def _contains_axis(rz: np.ndarray, axis: tuple[float, float]) -> bool:
    try:
        from matplotlib.path import Path as MplPath

        return bool(MplPath(rz).contains_point(axis))
    except Exception:
        return False


def _target_boundary(geqdsk: Any, target_psin: float) -> np.ndarray:
    if target_psin <= 0.0 or target_psin >= 1.0:
        r = np.asarray(geqdsk["RBBBS"], dtype=float)
        z = np.asarray(geqdsk["ZBBBS"], dtype=float)
        if r.size and z.size:
            return np.column_stack([r[: min(r.size, z.size)], z[: min(r.size, z.size)]])

    try:
        from skimage import measure
    except Exception as exc:
        raise RuntimeError("skimage is required to generate CHEASE boundary contours") from exc

    r_grid, z_grid, psi = _grid_arrays(geqdsk)
    simag = float(geqdsk["SIMAG"])
    sibry = float(geqdsk["SIBRY"])
    target_psi = simag + float(target_psin) * (sibry - simag)

    # ``find_contours`` sees array rows first, so pass psi.T with rows=z and cols=r.
    contours = measure.find_contours(psi.T, target_psi)
    candidates = []
    axis = (float(geqdsk["RMAXIS"]), float(geqdsk["ZMAXIS"]))
    for contour in contours:
        if contour.shape[0] < 4:
            continue
        row = contour[:, 0]
        col = contour[:, 1]
        rz = np.column_stack(
            [
                np.interp(col, np.arange(r_grid.size), r_grid),
                np.interp(row, np.arange(z_grid.size), z_grid),
            ]
        )
        if not np.all(np.isfinite(rz)) or np.mean(rz[:, 0] > 0.0) < 0.95:
            continue
        span = max(np.ptp(rz[:, 0]), np.ptp(rz[:, 1]), 1.0e-12)
        closed = np.linalg.norm(rz[0] - rz[-1]) <= max(1.0e-3, 0.05 * span)
        area = _polygon_area(rz)
        score = area + (100.0 if _contains_axis(rz, axis) else 0.0) + (10.0 if closed else 0.0)
        candidates.append((score, area, rz))

    if not candidates:
        r = np.asarray(geqdsk["RBBBS"], dtype=float)
        z = np.asarray(geqdsk["ZBBBS"], dtype=float)
        if r.size and z.size:
            return np.column_stack([r[: min(r.size, z.size)], z[: min(r.size, z.size)]])
        raise ValueError(f"No usable target boundary contour found at psin={target_psin}")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _smooth_boundary(rz: np.ndarray, count: int = 256) -> np.ndarray:
    rz = np.asarray(rz, dtype=float)
    if rz.shape[0] < 4:
        return rz
    if np.linalg.norm(rz[0] - rz[-1]) > 1.0e-10:
        rz = np.vstack([rz, rz[0]])
    dr = np.diff(rz[:, 0])
    dz = np.diff(rz[:, 1])
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(dr * dr + dz * dz))])
    if s[-1] <= 0.0:
        return rz
    target = np.linspace(0.0, s[-1], count, endpoint=False)
    return np.column_stack([np.interp(target, s, rz[:, 0]), np.interp(target, s, rz[:, 1])])


def _smooth_boundary_fft(rz: np.ndarray, nf: int = 128) -> np.ndarray:
    """Uniform-theta cubic boundary resample matching eqdsk.py ``smooth_bnd``.

    eqdsk's fft/ifft roundtrip (same ``n=nf``) is an identity reconstruction, so
    the smoothing reduces to a periodic cubic interpolation of the normalized
    radius ``rad/amin`` versus geometric poloidal angle onto ``nf`` points.
    """
    from scipy.interpolate import interp1d

    rz = np.asarray(rz, dtype=float)
    if rz.shape[0] < 4:
        return rz
    if np.linalg.norm(rz[0] - rz[-1]) <= 1.0e-10:
        rz = rz[:-1]  # smooth_bnd operates on the open contour
    rcen = float((np.nanmax(rz[:, 0]) + np.nanmin(rz[:, 0])) * 0.5)
    zcen = float(rz[int(np.nanargmax(rz[:, 0])), 1])
    amin = float((np.nanmax(rz[:, 0]) - np.nanmin(rz[:, 0])) * 0.5)
    if amin <= 0.0:
        return rz
    rad = np.sqrt((rz[:, 0] - rcen) ** 2 + (rz[:, 1] - zcen) ** 2) / amin
    theta = np.arctan2(rz[:, 1] - zcen, rz[:, 0] - rcen)
    order = np.argsort(theta)
    theta = theta[order]
    rad = rad[order]
    # Wrap one full period so the spline is periodic across +/- pi.
    theta_ext = np.concatenate([theta, theta + 2.0 * np.pi])
    rad_ext = np.concatenate([rad, rad])
    keep = np.concatenate([[True], np.diff(theta_ext) > 0.0])  # strictly increasing
    theta_ext = theta_ext[keep]
    rad_ext = rad_ext[keep]
    theta_fine = np.linspace(0.0, 2.0 * np.pi, int(nf))
    rad_fine = interp1d(theta_ext, rad_ext, kind="cubic")(theta_fine)
    out = np.empty((int(nf), 2), dtype=float)
    out[:, 0] = rcen + rad_fine * amin * np.cos(theta_fine)
    out[:, 1] = zcen + rad_fine * amin * np.sin(theta_fine)
    return out


def _resolve_boundary(rz: np.ndarray, config: "CHEASEConfig") -> np.ndarray:
    """Dispatch boundary smoothing according to ``config.boundary_smoothing``."""
    method = str(config.boundary_smoothing).lower()
    if method in {"arclength", "arc", "arc_length"}:
        return _smooth_boundary(rz)
    if method in {"none", "off", "raw"}:
        return np.asarray(rz, dtype=float)
    # default / "fft" / "smooth_bnd" / "jsk95": eqdsk.py smooth_bnd parity
    return _smooth_boundary_fft(rz, int(config.boundary_fft_num))


def _edge_zero_profiles(
    psin: np.ndarray, pprime: np.ndarray, ffprim: np.ndarray, qloc: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Zero non-physical edge pprime reversals, flatten FF' inward (eqdsk.py:566-573).

    Walks from the separatrix toward the axis; every point whose ``pprime`` has
    the OPPOSITE sign to the bulk profile (a spline-undershoot reversal at the
    edge) is zeroed and ``ffprim`` there is held at its just-outside value.

    eqdsk.py tests a literal ``pprime > 0`` because it operates on the *raw* EFIT
    gEQDSK whose bulk ``pprime`` is negative. This adapter feeds ``_write_expeq``
    the COCOS-02-normalized gEQDSK, whose ``pprime`` sign may be flipped, so a
    literal ``> 0`` test would zero the ENTIRE profile. Testing against the bulk
    sign (median) reproduces eqdsk's intent regardless of the COCOS convention.

    ``qloc`` is nudged outward exactly as eqdsk.py does when a reversed point
    falls between ``qloc`` and 0.3 (inert for jsk95 since qloc = sqrt(0.95) > 0.3).
    """
    pp = np.array(pprime, dtype=float, copy=True)
    ff = np.array(ffprim, dtype=float, copy=True)
    bulk_sign = np.sign(np.median(pp))
    if bulk_sign == 0.0:
        bulk_sign = 1.0
    lenp = len(pp) - 1
    for i in range(lenp):
        ii = lenp - i - 1
        if pp[ii] * bulk_sign < 0.0:  # reversed relative to the bulk => non-physical
            if qloc < psin[ii] < 0.3:
                qloc = float(psin[ii] + 0.1)
            pp[ii] = 0.0
            ff[ii] = ff[ii + 1]
    return pp, ff, qloc


def _chease_mesh_params(nideal: int) -> dict[str, int]:
    if nideal == 8:
        return {"ns": 100, "nt": 100, "npsi": 300, "nchi": 512, "negp": 0, "ner": 2}
    if nideal == 10:
        return {"ns": 100, "nt": 100, "npsi": 200, "nchi": 200, "negp": 0, "ner": 0}
    return {"ns": 50, "nt": 50, "npsi": 200, "nchi": 100, "negp": 0, "ner": 2}


def _write_expeq(geqdsk: Any, path: Path, config: CHEASEConfig) -> dict[str, float]:
    for key in ("NW", "NH", "PSIRZ", "PPRIME", "FFPRIM", "PRES", "FPOL", "QPSI"):
        if key not in geqdsk:
            raise ValueError(f"GEQDSK is missing required CHEASE input field {key!r}")

    rz = _resolve_boundary(_target_boundary(geqdsk, config.target_psin), config)
    rcen = float((np.nanmax(rz[:, 0]) + np.nanmin(rz[:, 0])) * 0.5)
    zcen = float(rz[int(np.nanargmax(rz[:, 0])), 1])
    amin = float((np.nanmax(rz[:, 0]) - np.nanmin(rz[:, 0])) * 0.5)
    if rcen == 0.0 or amin <= 0.0:
        raise ValueError("Invalid CHEASE boundary geometry: zero major/minor radius")

    aspct = amin / rcen
    b0exp = float(geqdsk["RCENTR"]) * abs(float(geqdsk["BCENTR"])) / rcen
    if b0exp == 0.0:
        raise ValueError("Invalid CHEASE normalization: B0EXP is zero")

    psin = np.linspace(0.0, 1.0, NCHEASE)
    pprime = _profile(geqdsk["PPRIME"], NCHEASE)
    pressure = _profile(geqdsk["PRES"], NCHEASE)
    ffprim = _profile(geqdsk["FFPRIM"], NCHEASE)

    # jsk95 parity: constraint location qloc = sqrt(q95_psin); may be nudged
    # outward by edge-zeroing exactly as eqdsk.py chease_params/make_chease_expeq.
    qloc = float(np.sqrt(config.q95_psin)) if config.q95_constraint else 0.0
    if config.edge_zero:
        pprime, ffprim, qloc = _edge_zero_profiles(psin, pprime, ffprim, qloc)

    # CHEASE's inverse-equilibrium EXPEQ convention uses a negative-definite
    # pressure-gradient drive and the opposite FF' sign from EFIT/GEQDSK after
    # the input has been normalized to the COCOS-02 sign pattern.
    pp_input = -np.abs(pprime) * MU0 * (rcen**2) / b0exp
    ff_input = -ffprim / b0exp
    edge_pressure = MU0 * float(np.asarray(geqdsk["PRES"], dtype=float).reshape(-1)[-1]) / (b0exp**2)
    current = abs(MU0 * float(geqdsk["CURRENT"]) / rcen / b0exp)

    with path.open("w") as f:
        f.write(f"{aspct:.12g}\n")
        f.write(f"{zcen:.12g}\n")
        f.write(f"{edge_pressure:.12g}\n")
        f.write(f"{len(rz):d}\n")
        for r_val, z_val in rz:
            f.write(f"{r_val / rcen:.12g} {z_val / rcen:.12g}\n")
        f.write(f"{NCHEASE:d}\n")
        f.write("1 0\n")
        for value in np.sqrt(psin):
            f.write(f"{value:.12g}\n")
        for value in pp_input:
            f.write(f"{value:.12g}\n")
        for value in ff_input:
            f.write(f"{value:.12g}\n")
        f.write(f"{rcen:.12g}, R0 [M]\n")
        f.write(f"{b0exp:.12g}, B_T [T]\n")
        f.write(f"{current:.12g}, TOTAL CURRENT\n")

    q = np.asarray(geqdsk["QPSI"], dtype=float).reshape(-1)
    sign_q = np.sign(float(geqdsk["CURRENT"])) * np.sign(float(geqdsk["BCENTR"]))
    if config.q95_constraint and config.ncscal == 1 and q.size >= 4:
        # jsk95 double-sqrt: q is sampled at psi_N = sqrt(q95_psin) (NOT q95_psin),
        # and CSSPEC = sqrt(qloc) = q95_psin**0.25. See eqdsk.py:1717/763/698.
        from scipy.interpolate import interp1d

        x_q = np.linspace(0.0, 1.0, q.size)
        q_at = float(interp1d(x_q, q, kind="cubic", fill_value="extrapolate")(qloc))
        qval = q_at * sign_q
        csspec = float(np.sqrt(qloc))
    else:
        # Legacy adapter behavior: constrain on-axis q, CSSPEC disabled.
        qval = float(q[0] * sign_q) if q.size else 1.0
        csspec = 0.0
    return {
        "ASPCT": aspct,
        "R0EXP": rcen,
        "B0EXP": b0exp,
        "CURRT": current,
        "QSPEC": qval,
        "CSSPEC": csspec,
        "QLOC": qloc,
        # CHEASE sees already-normalized COCOS-02 inputs. Keeping these
        # experimental sign fields positive avoids a second sign flip inside
        # CHEASE's inverse solver.
        "SIGNB0XP": 1.0,
        "SIGNIPXP": 1.0,
    }


def _namelist_lines(config: CHEASEConfig, params: Mapping[str, float]) -> list[str]:
    mesh = _chease_mesh_params(config.nideal)
    width = 0.05
    lines = [
        "*************************\n",
        "***    CHEASE NAMELIST FILE\n",
        "***    namelist created by vaft.code.chease\n",
        "*************************\n",
        "&EQDATA\n",
        "! --- CHEASE input file option\n",
        "NOPT=0,\n",
        "NSURF=6,\n",
        "NEQDSK=0\n",
        "TENSBND=    -0.1,\n",
        "NSYM=0,\n",
        "NTCASE=0,\n",
        f"RELAX = {float(config.relax):.12g},\n",
        "NPLOT=0,\n",
        "NSMOOTH=1,\n",
        "NDIAGOP=1,\n",
        "NPROPT=2,\n",
        "TREEITM=\"euitm\", \"euitm\",\n",
        "XI=       0,\n",
        "NFUNRHO=0,\n",
        "TENSPROF=    -0.1,\n",
        "NPPFUN = 4, NPP=2,\n",
        "NFUNC=4, NIPR=2,\n",
        "CFNRESS=1.00,\n",
        "NRSCAL=0, NTMF0=0,\n",
        "NMESHPOL=0,SOLPDPOL=0.25,\n",
        "NMESHD=0, NPOIDD=2, SOLPDD=.60,DPLACE=-1.80,-1.80,\n",
        "                               DWIDTH=.18,.08,.05,\n",
        "NMESHE=0, NPOIDE=4, SOLPDE=.50,EPLACE=-1.70,-1.70,\n",
        "                               EWIDTH=.18,.08,.18,\n",
        "PSISCL= 1.0,\n",
        "NDIFPS=0,\n",
        "NDIFT=0,\n",
        "NINMAP=50,  NINSCA=50,\n",
        "NBSEXPQ=0,\n",
        "ETAEI= 0.1, RPEOP= 0.5,  RZION=1.5,\n",
        "NBAL=0, NBLOPT=0,\n",
        "NPPR=30,\n",
        "NTURN=20, NBLC0=16,\n",
        "NBSOPT=0,\n",
        "NBSTRP=1, BSFRAC=0, NBSFUN=1,\n",
        f"EPSLON=1.0E-{int(config.epslon_exponent)}, GAMMA=1.6666666667,\n",
        "CFBAL=10.00,\n",
        "COCOS_IN = 2,\n",
        "COCOS_OUT = 2,\n",
        f"NRBOX={int(config.nw)},\n",
        f"NZBOX={int(config.nw)},\n",
        "NEQDXTPO=2,\n",
        "NVEXP=1, REXT=10.0, R0W=1., RZ0W=0.,\n",
        "MSMAX=1,\n",
        "NTNOVA=12,\n",
        "NITMRUN=1, 99, NITMSHOT=10, 10,\n",
        "NITMOPT=0,\n",
        f"NCSCAL={int(config.ncscal)},\n",
        f"CSSPEC={float(params.get('CSSPEC', 0.0)):.6f},\n",
        f"NEGP={mesh['negp']}, NER={mesh['ner']},\n",
        f"NIDEAL={int(config.nideal)},\n",
        f"NMESHA=1, NPOIDA=2, SOLPDA=.20,APLACE= {1.0 - 0.5 * width:4.3f}, 1.000,\n",
        f"                               AWIDTH= {0.9 * width:4.3f}, 0.003,\n",
        f"NMESHC=1, NPOIDC=3, SOLPDC=.20,CPLACE=0.000, {1.0 - 0.5 * width:4.3f}, 1.00,\n",
        f"                               CWIDTH=0.005, {0.9 * width:4.3f}, 0.003,\n",
        f"NS={mesh['ns']}, NT={mesh['nt']},\n",
        f"NPSI={mesh['npsi']}, NCHI={mesh['nchi']},\n",
        f"ASPCT= {params['ASPCT']:.12g},\n",
        f"R0EXP= {params['R0EXP']:.12g}, B0EXP={params['B0EXP']:.12g},\n",
        f"SIGNB0XP=       {int(params['SIGNB0XP'])},\n",
        f"SIGNIPXP=       {int(params['SIGNIPXP'])},\n",
        f"CURRT={params['CURRT']:.12g}, QSPEC={params['QSPEC']:.12g},\n",
        "/\n",
    ]
    return lines


def _write_namelist(path: Path, config: CHEASEConfig, params: Mapping[str, float]) -> None:
    path.write_text("".join(_namelist_lines(config, params)))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _default_input_name(source: Any) -> str:
    if isinstance(source, (str, Path)):
        name = Path(source).expanduser().name
        return name or "input.geqdsk"
    return "input.geqdsk"


def _resolve_executable(config: CHEASEConfig) -> Path | None:
    candidates = []
    if config.executable:
        candidate = Path(config.executable).expanduser()
        return candidate if candidate.exists() and os.access(candidate, os.X_OK) else None
    environment = {**os.environ, **dict(config.env)}
    home_executable = executable_from_home(
        environment.get(CHEASE_HOME_ENV),
        home_variable=CHEASE_HOME_ENV,
        relative_path=CHEASE_HOME_EXECUTABLE,
        code_name="CHEASE",
    )
    if home_executable is not None:
        return home_executable
    chease_env = environment.get("CHEASE")
    if chease_env:
        chease_candidate = Path(chease_env).expanduser()
        candidates.append(chease_candidate / "chease" if chease_candidate.is_dir() else chease_candidate)
    env_path = environment.get("CHEASE_EXEC_DIR")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        candidates.append(env_candidate / "chease" if env_candidate.is_dir() else env_candidate)
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_chease_executable(config: CHEASEConfig | None = None) -> Path | None:
    """Resolve CHEASE from explicit config, ``$CHEASEHOME``, or legacy variables."""
    return _resolve_executable(config or CHEASEConfig())


def prepare_chease_inputs(source: Any, config: CHEASEConfig | None = None) -> CHEASEInputs:
    """Materialize GEQDSK/EXPEQ/namelist files needed by CHEASE."""
    config = config or CHEASEConfig()
    workdir = Path(config.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    original = _coerce_geqdsk(source)
    working = original
    input_info = _geqdsk_sign_info(original)
    manifests: list[Path] = []
    if config.auto_cocos:
        working, before, after, transform = _force_geqdsk_signs(working, **_desired_signs_for_chease())
        manifests.append(
            _write_json(
                workdir / "chease_cocos_transform.json",
                {
                    "auto_cocos": True,
                    "source_equilibrium": str(getattr(original, "source", "") or ""),
                    "chease_target": "cocos02_sign_pattern",
                    "input_signs": before.as_dict(),
                    "chease_input_signs": after.as_dict(),
                    "transform_to_chease_input": transform,
                },
            )
        )
    else:
        manifests.append(
            _write_json(
                workdir / "chease_cocos_transform.json",
                {
                    "auto_cocos": False,
                    "source_equilibrium": str(getattr(original, "source", "") or ""),
                    "input_signs": input_info.as_dict(),
                    "chease_input_signs": input_info.as_dict(),
                    "transform_to_chease_input": {"psi": 1, "bcentr": 1, "current": 1, "fpol": 1, "q": 1},
                },
            )
        )

    from vaft.data.eqdsk import write_geqdsk

    source_geqdsk = workdir / "source.geqdsk"
    write_geqdsk(original, source_geqdsk)
    input_geqdsk = workdir / _default_input_name(source)
    if input_geqdsk.name == source_geqdsk.name:
        input_geqdsk = workdir / "input.geqdsk"
    write_geqdsk(working, input_geqdsk)
    expeq = workdir / "EXPEQ"
    namelist = workdir / "chease_namelist"
    params = _write_expeq(working, expeq, config)
    _write_namelist(namelist, config, params)

    return CHEASEInputs(
        workdir=workdir,
        source=source,
        input_geqdsk=input_geqdsk,
        expeq=expeq,
        namelist=namelist,
        geqdsk=original,
        files=(source_geqdsk, input_geqdsk, expeq, namelist),
        manifests=tuple(manifests),
    )


def _refined_output_path(inputs: CHEASEInputs) -> Path:
    source = inputs.input_geqdsk or (inputs.workdir / "input.geqdsk")
    stem = source.stem
    suffix = source.suffix or ".geqdsk"
    return inputs.workdir / f"{stem}_chease{suffix}"


def _restore_boundary_limiter(refined: Any, source: Any):
    item = _copy_geqdsk(refined)
    for r_key, z_key, n_key in (("RBBBS", "ZBBBS", "NBBBS"), ("RLIM", "ZLIM", "LIMITR")):
        r = np.asarray(source.get(r_key, []), dtype=float)
        z = np.asarray(source.get(z_key, []), dtype=float)
        if r.size and z.size:
            count = min(r.size, z.size)
            item[r_key] = r[:count].copy()
            item[z_key] = z[:count].copy()
            item[n_key] = int(count)
    return item


def _preserve_source_wall(result: "CHEASEResult", source: Any) -> None:
    """Make an ODS-sourced run's `wall` provably identical to the input's.

    CHEASE refines the equilibrium only. For a g-file input, the wall never
    existed in the first place: RLIM/ZLIM is restored exactly by
    `_restore_boundary_limiter()`, and `geqdsk.to_omas()` derives `wall`
    from those unchanged values. For an ODS input, go one step further and
    skip that GEQDSK text round trip entirely -- EQDSK's E14.6 fixed-format
    only carries 6 significant digits -- by copying the input ODS's own
    `wall` IDS directly onto the result, so it is provably unchanged rather
    than merely numerically close.
    """
    from omas import ODS

    if result.refined_ods is None or not isinstance(source, ODS) or "wall" not in source:
        return
    result.refined_ods["wall"] = copy.deepcopy(source["wall"])


def comparison_metrics(original: Any, refined: Any) -> dict[str, float]:
    """Quantify how far CHEASE's refinement moved an EFIT equilibrium.

    ``original``/``refined`` are :class:`~vaft.data.eqdsk.GEQDSK`-like mappings
    (pre- and post-refinement g-files). Consumed both by the CHEASE run itself
    (``CHEASEResult.comparison``) and, embedded into the refined ODS's
    ``equilibrium.code.parameters``, by the ``chease`` validation stage
    (:mod:`vaft.validation`).
    """
    metrics = {}
    for name, key in (
        ("q_rms_rel", "QPSI"),
        ("pressure_rms_rel", "PRES"),
        ("pprime_rms_rel", "PPRIME"),
        ("ffprim_rms_rel", "FFPRIM"),
    ):
        a = np.asarray(original[key], dtype=float).reshape(-1)
        b = _profile(refined[key], a.size)
        # Normalized by the RMS of the original profile, not by a residual
        # baseline: this is a relative profile change, so a flat-zero
        # original falls back to an absolute RMS rather than to nan.
        denom = rms(a) or 1.0
        metrics[name] = float(rms(a - b) / denom)
    metrics["psi_axis_abs_diff"] = float(abs(float(original["SIMAG"]) - float(refined["SIMAG"])))
    metrics["psi_boundary_abs_diff"] = float(abs(float(original["SIBRY"]) - float(refined["SIBRY"])))
    current_original = float(original["CURRENT"])
    current_refined = float(refined["CURRENT"])
    metrics["current_abs_diff"] = float(abs(current_original - current_refined))
    metrics["current_rel_diff"] = float(
        abs(current_original - current_refined) / abs(current_original) if current_original else float("nan")
    )
    r0 = np.asarray(original.get("RBBBS", []), dtype=float).reshape(-1)
    z0 = np.asarray(original.get("ZBBBS", []), dtype=float).reshape(-1)
    r1 = np.asarray(refined.get("RBBBS", []), dtype=float).reshape(-1)
    z1 = np.asarray(refined.get("ZBBBS", []), dtype=float).reshape(-1)
    if r0.size and z0.size and r1.size and z1.size:
        b0 = _resample_closed_curve(np.column_stack([r0[: min(r0.size, z0.size)], z0[: min(r0.size, z0.size)]]), 256)
        b1 = _resample_closed_curve(np.column_stack([r1[: min(r1.size, z1.size)], z1[: min(r1.size, z1.size)]]), 256)
        delta = b0 - b1
        metrics["boundary_r_rms"] = rms(delta[:, 0])
        metrics["boundary_z_rms"] = rms(delta[:, 1])
        metrics["boundary_rz_rms"] = rms(np.hypot(delta[:, 0], delta[:, 1]))
    metrics["boundary_points"] = float(len(np.asarray(refined.get("RBBBS", []))))
    return metrics


def _psi_norm_for_plot(geqdsk: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, z, psi = _grid_arrays(geqdsk)
    denom = float(geqdsk["SIBRY"]) - float(geqdsk["SIMAG"])
    if abs(denom) < 1.0e-30:
        psi_norm = np.zeros_like(psi, dtype=float)
    else:
        psi_norm = (psi - float(geqdsk["SIMAG"])) / denom
    return r, z, psi_norm


def _resample_closed_curve(rz: np.ndarray, count: int = 256) -> np.ndarray:
    rz = np.asarray(rz, dtype=float)
    rz = rz[np.all(np.isfinite(rz), axis=1)]
    if rz.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    if rz.shape[0] == 1:
        return np.repeat(rz, count, axis=0)
    if np.linalg.norm(rz[0] - rz[-1]) > 1.0e-12:
        rz = np.vstack([rz, rz[0]])
    ds = np.sqrt(np.sum(np.diff(rz, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 0.0:
        return np.repeat(rz[:1], count, axis=0)
    target = np.linspace(0.0, s[-1], count, endpoint=False)
    return np.column_stack([np.interp(target, s, rz[:, 0]), np.interp(target, s, rz[:, 1])])


def _boundary_and_limiter_layers(geqdsk: Any, label: str, color: str, linestyle: str):
    """Boundary and limiter outlines for one equilibrium, as geometry layers."""
    from vaft.plot import GeometryLayer

    layers = []
    rb = np.asarray(geqdsk.get("RBBBS", []), dtype=float)
    zb = np.asarray(geqdsk.get("ZBBBS", []), dtype=float)
    if rb.size and zb.size:
        count = min(rb.size, zb.size)
        layers.append(
            GeometryLayer(
                r=rb[:count],
                z=zb[:count],
                kind="polyline",
                label=f"{label} boundary",
                style={"color": color, "linestyle": linestyle, "lw": 1.8},
            )
        )
    rl = np.asarray(geqdsk.get("RLIM", []), dtype=float)
    zl = np.asarray(geqdsk.get("ZLIM", []), dtype=float)
    if rl.size and zl.size:
        count = min(rl.size, zl.size)
        layers.append(
            GeometryLayer(
                r=rl[:count],
                z=zl[:count],
                kind="polyline",
                label="",
                style={"color": color, "linestyle": ":", "lw": 1.0, "alpha": 0.65},
            )
        )
    return layers


def _comparison_model(original: Any, refined: Any):
    """Build the CHEASE input-vs-refined comparison view model.

    Rendering stays in :mod:`vaft.plot` (issue #63): this only shapes the two
    EQDSKs into the typed view models the canonical panel renderer consumes.

    The four profile comparisons are drawn exactly as before.  The flux map is
    the *refined* psi_N field with the input boundary overlaid, rather than two
    overlaid contour sets: extracting contour polylines needs a live Axes, which
    only a renderer may own.  The dedicated boundary/limiter panel still
    compares both geometries directly, so nothing the comparison is for is lost.
    """
    from vaft.plot import Field2D, GeometryLayers, Panels, Profile1D, Series

    x0 = np.linspace(0.0, 1.0, int(original["NW"]))
    xr = np.linspace(0.0, 1.0, int(refined["NW"]))
    panels: list[Any] = []
    for key, title, ylabel in (
        ("QPSI", "Safety factor", "q"),
        ("PRES", "Pressure", "Pa"),
        ("PPRIME", "Pressure derivative", "dP/dpsi"),
        ("FFPRIM", "FF prime", "FF'"),
    ):
        panels.append(
            Profile1D(
                series=(
                    Series(
                        x=x0,
                        y=np.asarray(original[key], dtype=float),
                        label="input",
                        style={"lw": 2.0},
                    ),
                    Series(
                        x=xr,
                        y=np.asarray(refined[key], dtype=float),
                        label="CHEASE",
                        style={"lw": 1.6},
                    ),
                ),
                coordinate_label="Normalized flux",
                y_label=ylabel,
                title=title,
            )
        )

    r1, z1, psi1 = _psi_norm_for_plot(refined)
    input_layers = _boundary_and_limiter_layers(original, "input", "tab:blue", "--")
    refined_layers = _boundary_and_limiter_layers(refined, "CHEASE", "tab:orange", "-")
    panels.append(
        Field2D(
            r=r1,
            z=z1,
            values=psi1.T,
            value_label=r"$\psi_N$",
            title=r"$\psi_N(R,Z)$, CHEASE (input boundary overlaid)",
            contour_levels=np.linspace(0.1, 1.0, 10),
            filled=False,
            overlays=tuple(layer for layer in input_layers if layer.label),
        )
    )
    panels.append(
        GeometryLayers(
            layers=tuple(input_layers + refined_layers),
            title="Boundary and limiter",
        )
    )
    return Panels(models=tuple(panels), ncols=3, share_x=False)


def _create_comparison_plot(original: Any, refined: Any, target: Path) -> Path:
    from vaft.plot import render_panels, save_figure

    figure, _axes = render_panels(
        _comparison_model(original, refined), show=False, figsize=(15.0, 8.5)
    )
    return save_figure(figure, target, dpi=150)


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def _find_logs(workdir: Path) -> tuple[Path, ...]:
    names = {"chease.log", "log.chease"}
    logs = [path for path in workdir.iterdir() if path.is_file() and (path.name in names or path.suffix == ".log")]
    return tuple(sorted(logs))


def _find_manifests(workdir: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in workdir.glob("*.json") if path.is_file()))


def run_chease(inputs: CHEASEInputs, config: CHEASEConfig | None = None) -> CHEASEResult:
    """Run CHEASE with prepared inputs and collect the refined equilibrium."""
    config = config or CHEASEConfig(workdir=inputs.workdir)
    executable = _resolve_executable(config)
    if executable is None:
        raise FileNotFoundError(
            missing_home_message(
                home_variable=CHEASE_HOME_ENV,
                relative_path=CHEASE_HOME_EXECUTABLE,
                code_name="CHEASE",
                compatibility_variables=CHEASE_COMPATIBILITY_ENVS,
            )
        )
    if inputs.expeq is None or not inputs.expeq.exists():
        raise FileNotFoundError(f"Missing EXPEQ file in {inputs.workdir}")
    if inputs.namelist is None or not inputs.namelist.exists():
        raise FileNotFoundError(f"Missing chease_namelist file in {inputs.workdir}")

    env = os.environ.copy()
    env.update(dict(config.env))
    completed = subprocess.run(
        [str(executable), *config.args],
        cwd=str(inputs.workdir),
        env=env,
        text=True,
        capture_output=True,
        timeout=config.timeout,
        check=False,
    )
    log_path = inputs.workdir / "chease.log"
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")

    raw = inputs.workdir / "EQDSK_COCOS_02.OUT"
    refined_target = _refined_output_path(inputs)
    effective_returncode = completed.returncode
    missing_output_message = ""
    if completed.returncode == 0 and not raw.exists():
        effective_returncode = 1
        missing_output_message = (
            "CHEASE exited with code 0 but did not produce EQDSK_COCOS_02.OUT. "
            "Check chease.log/NOUT for CHEASE output_flag diagnostics."
        )

    if completed.returncode == 0 and raw.exists():
        shutil.copy2(raw, refined_target)
        from vaft.data.eqdsk import read_geqdsk, write_geqdsk

        refined = read_geqdsk(refined_target)
        if str(config.output_cocos).lower().replace("-", "_") in {"input", "preserve_input", "source"}:
            refined, before, after, transform = _force_geqdsk_signs(refined, **_desired_signs_from_info(_geqdsk_sign_info(inputs.geqdsk)))
            _write_json(
                refined_target.with_suffix(refined_target.suffix + ".cocos_export.json"),
                {
                    "output_cocos": config.output_cocos,
                    "before_export_signs": before.as_dict(),
                    "after_export_signs": after.as_dict(),
                    "transform": transform,
                },
            )
        if config.preserve_boundary_limiter:
            refined = _restore_boundary_limiter(refined, inputs.geqdsk)
            _write_json(
                refined_target.with_suffix(refined_target.suffix + ".geometry_preserve.json"),
                {"source_equilibrium": str(inputs.input_geqdsk), "refined_equilibrium": str(refined_target)},
            )
        write_geqdsk(refined, refined_target)

    result = collect_chease_outputs(inputs.workdir, config, source=inputs.source)
    result.returncode = effective_returncode
    result.stdout = completed.stdout
    result.stderr = (completed.stderr or "") + (("\n" + missing_output_message) if missing_output_message else "")
    if config.cleanup and result.ok:
        # Keep the returned paths meaningful only for non-cleanup workflows; cleanup is
        # intended for fire-and-forget integration jobs.
        shutil.rmtree(inputs.workdir, ignore_errors=True)
    return result


def collect_chease_outputs(workdir: str | Path, config: CHEASEConfig | None = None, source: Any = None) -> CHEASEResult:
    """Collect CHEASE files from a working directory and parse refined GEQDSK.

    `source` is the original input passed to `prepare_chease_inputs`/`refine_equilibrium`
    (an ODS or a GEQDSK). When it is an ODS, its `wall` IDS is copied onto the
    result verbatim so the invariant "CHEASE never invents or replaces limiter
    geometry" holds for every caller of this function, not only `run_chease()`.
    """
    base = Path(workdir).expanduser()
    config = config or CHEASEConfig(workdir=base)
    from vaft.data.eqdsk import read_geqdsk

    raw = base / "EQDSK_COCOS_02.OUT"
    # `_refined_output_path()` names the restored file `<stem>_chease<suffix>`,
    # where `<suffix>` is whatever Path.suffix finds after the last dot in the
    # *input* name -- for VEST's `g<shot>.<time>` gfiles (no .geqdsk/.gfile/.g
    # extension) that is the numeric time, e.g. `g039915_chease.00319`. A glob
    # restricted to the conventional GEQDSK extensions never matches that name
    # and silently falls back to the raw, pre-restore CHEASE output below,
    # discarding run_chease()'s boundary/limiter restoration entirely.
    refined_candidates = sorted(
        path
        for path in base.glob("*_chease*")
        if path.is_file()
        and not path.name.startswith(".")  # exclude AppleDouble/dotfile sidecars
        and path.suffix.lower() not in {".json", ".png", ".log"}
    )
    refined = refined_candidates[0] if refined_candidates else (raw if raw.exists() else None)
    preferred_inputs = [
        base / "source.geqdsk",
        base / "input.geqdsk",
        base / "input.gfile",
        base / "input.g",
    ]
    input_candidates = [path for path in preferred_inputs if path.exists()]
    input_candidates.extend(
        path
        for path in sorted(base.glob("*"))
        if path.is_file()
        and path not in input_candidates
        and not path.name.startswith(("EQDSK", "EXPEQ", "NOUT", "NJA", "log."))
        and path.name not in {"chease.log", "chease_namelist"}
        and path.suffix.lower() not in {".json", ".png", ".log"}
        and "_chease" not in path.stem
    )
    input_geqdsk = None
    for path in input_candidates:
        if path == refined or path.name.startswith("EQDSK"):
            continue
        try:
            read_geqdsk(path)
            input_geqdsk = path
            break
        except Exception:
            continue

    refined_ods = None
    comparison: dict[str, float] = {}
    figures: list[Path] = []
    if refined is not None and refined.exists():
        try:
            refined_obj = read_geqdsk(refined)
            refined_ods = refined_obj.to_omas()
            if input_geqdsk is not None:
                original_obj = read_geqdsk(input_geqdsk)
                comparison = comparison_metrics(original_obj, refined_obj)
                if config.create_plot:
                    figures.append(_create_comparison_plot(original_obj, refined_obj, base / "chease_comparison.png"))
        except Exception:
            refined_ods = None

    outputs = {
        "geqdsk": tuple(path for path in (refined,) if path is not None and path.exists()),
        "raw": tuple(path for path in (raw,) if path.exists()),
        "inputs": tuple(path for path in (base / "EXPEQ", base / "chease_namelist") if path.exists()),
    }
    result = CHEASEResult(
        returncode=None,
        workdir=base,
        input_geqdsk=input_geqdsk,
        refined_geqdsk=refined if refined is not None and refined.exists() else None,
        refined_ods=refined_ods,
        logs=_find_logs(base) if base.exists() else (),
        manifests=_find_manifests(base) if base.exists() else (),
        figures=tuple(figures),
        outputs=outputs,
        comparison=comparison,
        stdout=_read_optional(base / "chease.log") if base.exists() else "",
    )
    _preserve_source_wall(result, source)
    return result


def refine_equilibrium(source: Any, config: CHEASEConfig | None = None) -> CHEASEResult:
    """Prepare inputs, run CHEASE, and collect the refined equilibrium."""
    config = config or CHEASEConfig()
    inputs = prepare_chease_inputs(source, config)
    return run_chease(inputs, config)


__all__ = [
    "CHEASEConfig",
    "CHEASEInputs",
    "CHEASEResult",
    "CodeConfig",
    "CodeInputs",
    "CodeResult",
    "CodeRunner",
    "GeqdskSignInfo",
    "collect_chease_outputs",
    "comparison_metrics",
    "find_chease_executable",
    "prepare_chease_inputs",
    "refine_equilibrium",
    "run_chease",
]
