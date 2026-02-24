"""
Test Shafranov integrals (S_1, S_2, S_3, alpha) across three sources.

1. aeq file (a039915.00319): reference S_1, S_2, S_3, alpha from OMFITaeqdsk.
2. g-file (g039915.00319): load via omfit_classes -> ODS -> vaft Shafranov.
3. ODS from database: vaft.database.load(39915), equilibrium time index for the
   same time (0.319 s) -> compute_virial_equilibrium_quantities_ods -> S_1, S_2, S_3, alpha.

Compare CHEASE g-file, EFIT afile (aeq), and EFIT g-file by shot and time (no path args):
  python test_shafranov_integral.py <shot> <time_s> [base_dir]
  e.g. python test_shafranov_integral.py 41670 0.320
  Paths under base_dir (default /srv/vest.filedb/public):
    {base}/{shot}/chease/g0{shot}.00{time_ms}
    {base}/{shot}/efit/afile/a0{shot}.00{time_ms}
    {base}/{shot}/efit/gfile/g0{shot}.00{time_ms}
  If base_dir is omitted, PUBLIC_BASE is used so beta_p, li, W_mag, W_kin use that shot's ODS.

Requires: omfit_classes (OMFITgeqdsk/OMFITeqdsk, OMFITaeqdsk); vaft.database for ODS case.
"""
from __future__ import annotations

import os
import re
import numpy as np

# Data paths: vaft/vaft/data (relative to repo root; test lives in test/)
def _data_dir():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "vaft", "data")

GFILE_NAME = "g039915.00319"
AEQ_NAME = "a039915.00319"
# Shot and time [s] corresponding to g039915.00319 / a039915.00319
SHOT_FOR_GFILE = 39915
TIME_FOR_GFILE = 0.319  # 319 ms

PUBLIC_BASE = "/srv/vest.filedb/public"


def _diamag_meas_from_diagnostics_json(
    shot: int,
    t_eq: float,
    base: str | None = None,
    debug: bool | None = None,
) -> float:
    """
    Load ODS from {base}/{shot}/omas/{shot}_diagnostics.json (load_omas_json) and
    return diamagnetic flux [Wb] interpolated at t_eq [s], or np.nan if missing/fail.
    t_eq is in seconds. If magnetics.time in the file looks like ms (e.g. min > 10),
    it is converted to seconds so interpolation matches t_eq.

    If debug is True (or env DEBUG_DIAMAG_MEAS=1), print t_eq, time range, and result.
    """
    if debug is None:
        debug = os.environ.get("DEBUG_DIAMAG_MEAS", "").strip().lower() in ("1", "true", "yes")
    base = base or PUBLIC_BASE
    path = os.path.join(base, str(shot), "omas", f"{shot}_diagnostics.json")
    if not os.path.isfile(path):
        if debug:
            print(f"[DEBUG _diamag_meas] shot={shot} path not found: {path}")
        return np.nan
    try:
        from omas import load_omas_json
        ods = load_omas_json(path)
    except Exception as e:
        if debug:
            print(f"[DEBUG _diamag_meas] shot={shot} load_omas_json failed: {e}")
        return np.nan
    if "magnetics.time" not in ods or "magnetics.diamagnetic_flux.0.data" not in ods:
        if debug:
            print(f"[DEBUG _diamag_meas] shot={shot} missing magnetics.time or diamagnetic_flux.0.data")
        return np.nan
    if len(ods.get("magnetics.diamagnetic_flux", [])) == 0:
        if debug:
            print(f"[DEBUG _diamag_meas] shot={shot} magnetics.diamagnetic_flux empty")
        return np.nan
    t_m = np.asarray(ods["magnetics.time"], float).flatten()
    f_m = np.asarray(ods["magnetics.diamagnetic_flux.0.data"], float).flatten()
    if t_m.size < 1:
        if debug:
            print(f"[DEBUG _diamag_meas] shot={shot} t_m.size < 1")
        return np.nan
    t_eq_s = float(t_eq)
    t_m_converted = False
    # If file time looks like ms (e.g. 260–340), convert to seconds
    if 0.01 <= t_eq_s <= 2.0 and (np.nanmin(t_m) > 10.0 or np.nanmax(t_m) > 100.0):
        t_m = t_m / 1000.0
        t_m_converted = True
    # If t_eq looks like ms (e.g. 320), convert to seconds
    if t_eq_s > 10.0 and np.nanmax(t_m) <= 2.0:
        t_eq_s = t_eq_s / 1000.0
    result = float(np.interp(t_eq_s, t_m, f_m))
    if debug:
        print(
            f"[DEBUG _diamag_meas] shot={shot} t_eq_in={t_eq} t_eq_s={t_eq_s} "
            f"t_m range=[{np.nanmin(t_m):.6g}, {np.nanmax(t_m):.6g}] "
            f"t_m_converted={t_m_converted} result={result:.6g} (in_range={np.nanmin(t_m) <= t_eq_s <= np.nanmax(t_m)})"
        )
    return result


def paths_for_shot_time(
    shot: int,
    time_s: float,
    base: str | None = None,
) -> dict[str, str]:
    """
    Build paths for CHEASE g-file, EFIT afile (aeq), EFIT g-file, kfile, mfile under public tree.
    time_s is in seconds (e.g. 0.320); filename uses milliseconds as 00XXX.
    Returns dict with keys: "chease_gfile", "efit_afile", "efit_gfile", "efit_kfile", "efit_mfile".
    """
    base = base or PUBLIC_BASE
    time_ms = int(round(time_s * 1000))
    suffix = f"00{time_ms:03d}"
    root = os.path.join(base, str(shot))
    return {
        "chease_gfile": os.path.join(root, "chease", f"g0{shot}.{suffix}"),
        "efit_afile": os.path.join(root, "efit", "afile", f"a0{shot}.{suffix}"),
        "efit_gfile": os.path.join(root, "efit", "gfile", f"g0{shot}.{suffix}"),
        "efit_kfile": os.path.join(root, "efit", "kfile", f"k0{shot}.{suffix}"),
        "efit_mfile": os.path.join(root, "efit", "mfile", f"m0{shot}.{suffix}"),
    }


def _parse_kfile_dflux(k_path: str) -> tuple[float, float, float]:
    """
    Parse EFIT kfile for DFLUX (measured diamagnetic flux) and SIGDLC.
    DFLUX in file is in mV·s (millivolt-second). Returns (dflux_mVsec, dflux_Wb, sigdlc);
    1 mV·s = 1e-3 Wb. Returns (nan, nan, nan) if file missing or DFLUX not found.
    """
    if not os.path.isfile(k_path):
        return (np.nan, np.nan, np.nan)
    try:
        with open(k_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return (np.nan, np.nan, np.nan)
    m = re.search(r"DFLUX\s*=\s*([-\d.eE+]+)", text)
    dflux_mVsec = float(m.group(1)) if m else np.nan
    m = re.search(r"SIGDLC\s*=\s*([-\d.eE+]+)", text)
    sigdlc = float(m.group(1)) if m else np.nan
    dflux_Wb = dflux_mVsec * 1e-3 if np.isfinite(dflux_mVsec) else np.nan
    return (dflux_mVsec, dflux_Wb, sigdlc)


def _read_mfile_diamag(m_path: str) -> tuple[float, float]:
    """
    Read EFIT mfile (NetCDF) for CDFLUX (computed) and DIAMAG (measured), both in V·s (Wb).
    Returns (cdflux_Wb, diamag_Wb). Returns (np.nan, np.nan) if file missing or read fails.
    """
    if not os.path.isfile(m_path):
        return (np.nan, np.nan)
    try:
        import netCDF4
        with netCDF4.Dataset(m_path, "r") as ds:
            cdflux = float(ds.variables["cdflux"][0]) if "cdflux" in ds.variables else np.nan
            diamag = float(ds.variables["diamag"][0]) if "diamag" in ds.variables else np.nan
        return (cdflux, diamag)
    except Exception:
        return (np.nan, np.nan)


def _aeq_safe_get(aeq, key: str, scale: float = 1.0, default=np.nan):
    """Get scalar from aeq[key], handling array (take first element). Return default if missing/invalid."""
    try:
        if key not in aeq:
            return float(default) if np.isscalar(default) else default
        v = np.asarray(aeq[key], float).flatten()
        if v.size == 0:
            return float(default) if np.isscalar(default) else default
        x = float(v[0]) * scale
        return x if np.isfinite(x) else (float(default) if np.isscalar(default) else default)
    except (TypeError, KeyError, IndexError, ValueError):
        return float(default) if np.isscalar(default) else default


def _get_shafranov_from_aeq(aeq_path: str, skip_if_no_omfit: bool = False) -> dict:
    """
    Read aEQDSK file (OMFITaeqdsk) and return dict aligned with comparison table.
    Uses Table 20 / Fortran read order: S_1,S_2,S_3,alpha; R_0,Z_0 (from RM,ZM); B_pa (BPOLAV);
    V_p (VOLUME); beta_p (BETAP); li (LI); W_mag (WMHD); diamagnetic flux (CDFLUX, FLUXX/diamag); etc.
    """
    if skip_if_no_omfit:
        import pytest as _pytest
        _pytest.importorskip("omfit_classes")
    from omfit_classes.omfit_eqdsk import OMFITaeqdsk

    aeq = OMFITaeqdsk(aeq_path)
    # Magnetic axis in cm -> m
    R_0 = _aeq_safe_get(aeq, "rm", scale=1.0 / 100.0)
    Z_0 = _aeq_safe_get(aeq, "zm", scale=1.0 / 100.0)
    # If rm/zm missing, fallback to geometric center (rcntr, zcntr in cm)
    if not np.isfinite(R_0):
        R_0 = _aeq_safe_get(aeq, "rcntr", scale=1.0 / 100.0)
    if not np.isfinite(Z_0):
        Z_0 = _aeq_safe_get(aeq, "zcntr", scale=1.0 / 100.0)

    # VOLUME: aEQDSK in cm³ -> m³ (1e-6). BPOLAV T, BETAP, LI, WMHD J.
    V_p = _aeq_safe_get(aeq, "volume", scale=1.0e-6)
    B_pa = _aeq_safe_get(aeq, "bpolav")
    beta_p = _aeq_safe_get(aeq, "betap")
    li = _aeq_safe_get(aeq, "li")
    W_mag = _aeq_safe_get(aeq, "wmhd")
    # W_kin: use WDIA (diamagnetic stored energy in J) as proxy
    W_kin = _aeq_safe_get(aeq, "wdia")
    # Diamagnetic flux: CDFLUX = computed (Volt-sec = Wb), FLUXX/diamag = measured (Volt-sec)
    diamag_recon = _aeq_safe_get(aeq, "cdflux")
    diamag_meas = _aeq_safe_get(aeq, "fluxx")
    if not np.isfinite(diamag_meas):
        diamag_meas = _aeq_safe_get(aeq, "diamag")

    return {
        "S_1": _aeq_safe_get(aeq, "s1", scale=1.0),
        "S_2": _aeq_safe_get(aeq, "s2", scale=1.0),
        "S_3": _aeq_safe_get(aeq, "s3", scale=1.0),
        "alpha": _aeq_safe_get(aeq, "alpha", scale=1.0),
        "R_0": R_0,
        "Z_0": Z_0,
        "B_pa": B_pa,
        "V_p": V_p,
        "beta_p": beta_p,
        "li": li,
        "W_mag": W_mag,
        "W_kin": W_kin,
        "mui_hat": np.nan,   # aeq has no virial μ̂_i
        "mui_exact": np.nan,  # aeq has no 2D grid for volume-integral μ_i
        "diamagnetic_flux_measured": diamag_meas,
        "diamagnetic_flux_reconstructed": diamag_recon,
    }


def _get_shafranov_from_gfile_ods(
    g_path: str,
    skip_if_no_omfit: bool = False,
    load_magnetics_from_shot: int | None = None,
    diagnostics_base: str | None = None,
    debug: bool = False,
    t_eq_override: float | None = None,
) -> dict:
    """
    Load g-file -> ODS (gfile_to_omas), update boundary; then either:
    - If load_magnetics_from_shot is set: copy magnetics (diamagnetic flux) from that shot's
      ODS (vaft.database.load), run full virial -> return S_1..alpha, R_0, Z_0, B_pa, V_p,
      beta_p, li, W_mag, W_kin.
    - Else: compute only Shafranov (and R_0, Z_0, B_pa, V_p) without magnetics.

    If t_eq_override is set (e.g. user-specified time_s), it is used for Δφ meas interpolation
    from diagnostics JSON instead of equilibrium time from the g-file (which may be 0 or wrong).
    """
    if skip_if_no_omfit:
        import pytest as _pytest
        _pytest.importorskip("omfit_classes")
    from omfit_classes.omfit_eqdsk import OMFITeqdsk

    from vaft.code.efit import gfile_to_omas
    from vaft.omas.update import update_equilibrium_boundary
    from vaft.process.equilibrium import (
        poloidal_field_at_boundary,
        shafranov_integrals,
        calculate_average_boundary_poloidal_field,
    )

    eq = OMFITeqdsk(g_path)
    ods = gfile_to_omas(eq)
    update_equilibrium_boundary(ods)

    if load_magnetics_from_shot is not None:
        try:
            import vaft
            ods_mag = vaft.database.load(load_magnetics_from_shot)
            if "magnetics.time" in ods_mag and "magnetics.diamagnetic_flux.0.data" in ods_mag and len(ods_mag.get("magnetics.diamagnetic_flux", [])) > 0:
                ods["magnetics.time"] = np.asarray(ods_mag["magnetics.time"], float)
                ods["magnetics.diamagnetic_flux.0.data"] = np.asarray(ods_mag["magnetics.diamagnetic_flux.0.data"], float)
                from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods
                virial = compute_virial_equilibrium_quantities_ods(ods, time_slice=0)
                if 0 in virial:
                    v = virial[0]
                    ts = ods["equilibrium.time_slice"][0]
                    t_eq = ts.get("time")
                    if t_eq is None and "equilibrium.time" in ods and len(ods["equilibrium.time"]):
                        t_eq = float(ods["equilibrium.time"][0])
                    # Δφ meas: use t_eq_override (e.g. user time_s) if set, else ODS time
                    t_for_diamag = t_eq_override if t_eq_override is not None else t_eq
                    diamag_meas = np.nan
                    if t_for_diamag is not None and load_magnetics_from_shot is not None:
                        diamag_meas = _diamag_meas_from_diagnostics_json(
                            load_magnetics_from_shot, t_for_diamag, base=diagnostics_base or PUBLIC_BASE, debug=debug
                        )
                    diamag_recon = np.nan
                    try:
                        from vaft.omas.process_wrapper import compute_reconstructed_diamagnetic_flux
                        diamag_recon = float(compute_reconstructed_diamagnetic_flux(ods, time_index=0))
                    except Exception:
                        pass
                    mui = v.get("mui_hat", np.nan)
                    mui = float(mui) if np.isfinite(np.asarray(mui, float)) else np.nan
                    mui_exact = np.nan
                    try:
                        from vaft.omas.process_wrapper import compute_diamagnetism
                        mui_exact = float(compute_diamagnetism(ods, time_index=0))
                    except Exception:
                        pass
                    return {
                        "S_1": float(v["s_1"]),
                        "S_2": float(v["s_2"]),
                        "S_3": float(v["s_3"]),
                        "alpha": float(v["alpha"]),
                        "R_0": float(ts.get("boundary.geometric_axis.r", np.nan)),
                        "Z_0": float(ts.get("boundary.geometric_axis.z", np.nan)),
                        "B_pa": float(v["B_pa"]),
                        "V_p": float(v["V_p"]),
                        "beta_p": float(v["beta_p"]) if np.isfinite(v.get("beta_p", np.nan)) else np.nan,
                        "li": float(v["li"]) if np.isfinite(v.get("li", np.nan)) else np.nan,
                        "W_mag": float(v["W_mag"]) if np.isfinite(v.get("W_mag", np.nan)) else np.nan,
                        "W_kin": float(v["W_kin"]) if np.isfinite(v.get("W_kin", np.nan)) else np.nan,
                        "mui_hat": mui,
                        "mui_exact": mui_exact,
                        "diamagnetic_flux_measured": diamag_meas,
                        "diamagnetic_flux_reconstructed": diamag_recon,
                    }
        except Exception:
            pass
        # fall through to Shafranov-only if magnetics/virial failed

    ts = ods["equilibrium.time_slice"][0]
    R_grid_1d = np.asarray(ts["profiles_2d.0.grid.dim1"], float)
    Z_grid_1d = np.asarray(ts["profiles_2d.0.grid.dim2"], float)
    psi_RZ = np.asarray(ts["profiles_2d.0.psi"], float)
    nR, nZ = len(R_grid_1d), len(Z_grid_1d)
    if psi_RZ.shape != (nR, nZ):
        if psi_RZ.shape == (nZ, nR):
            psi_RZ = psi_RZ.T
        else:
            raise ValueError(f"psi shape {psi_RZ.shape} vs grid (nR={nR}, nZ={nZ})")

    R_bdry = np.asarray(ts["boundary.outline.r"], float)
    Z_bdry = np.asarray(ts["boundary.outline.z"], float)
    R_0 = float(ts["boundary.geometric_axis.r"])
    Z_0 = float(ts["boundary.geometric_axis.z"])

    B_p_bdry, _, _ = poloidal_field_at_boundary(
        R_grid_1d, Z_grid_1d, psi_RZ, R_bdry, Z_bdry
    )
    B_pa = float(calculate_average_boundary_poloidal_field(R_bdry, Z_bdry, B_p_bdry))
    # V_p from boundary (same as process_wrapper)
    R_bc = np.append(R_bdry, R_bdry[0]) if (R_bdry[0] != R_bdry[-1] or Z_bdry[0] != Z_bdry[-1]) else R_bdry
    Z_bc = np.append(Z_bdry, Z_bdry[0]) if (R_bdry[0] != R_bdry[-1] or Z_bdry[0] != Z_bdry[-1]) else Z_bdry
    dR_b = np.diff(R_bc)
    dZ_b = np.diff(Z_bc)
    R_mid_b = 0.5 * (R_bc[:-1] + R_bc[1:])
    V_p = float(np.abs(-np.sum(np.pi * (R_mid_b**2) * dZ_b)))

    if "profiles_2d.0.b_field_r" in ts and "profiles_2d.0.b_field_z" in ts:
        B_R_grid = np.asarray(ts["profiles_2d.0.b_field_r"], float)
        B_Z_grid = np.asarray(ts["profiles_2d.0.b_field_z"], float)
        if B_R_grid.shape == (nZ, nR):
            B_R_grid = B_R_grid.T
            B_Z_grid = B_Z_grid.T
    else:
        dpsi_dR, dpsi_dZ = np.gradient(psi_RZ, R_grid_1d, Z_grid_1d, edge_order=2)
        Rm, Zm = np.meshgrid(R_grid_1d, Z_grid_1d, indexing="ij")
        Rm_safe = np.where(Rm == 0.0, np.nan, Rm)
        B_R_grid = -(1.0 / Rm_safe) * dpsi_dZ
        B_Z_grid = (1.0 / Rm_safe) * dpsi_dR

    R_mesh, Z_mesh = np.meshgrid(R_grid_1d, Z_grid_1d, indexing="ij")
    S1, S2, S3, alpha = shafranov_integrals(
        R_bdry, Z_bdry, B_p_bdry, R_mesh, Z_mesh, B_R_grid, B_Z_grid, R_0=R_0, Z_0=Z_0
    )
    mui_exact = np.nan
    try:
        from vaft.omas.process_wrapper import compute_diamagnetism
        mui_exact = float(compute_diamagnetism(ods, time_index=0))
    except Exception:
        pass
    # Δφ meas: use t_eq_override (e.g. user time_s) if set, else ODS time
    diamag_meas = np.nan
    diamag_recon = np.nan
    if load_magnetics_from_shot is not None:
        t_eq = ts.get("time")
        if t_eq is None and "equilibrium.time" in ods and len(ods["equilibrium.time"]):
            t_eq = float(ods["equilibrium.time"][0])
        t_for_diamag = t_eq_override if t_eq_override is not None else t_eq
        if t_for_diamag is not None:
            diamag_meas = _diamag_meas_from_diagnostics_json(
                load_magnetics_from_shot, t_for_diamag, base=diagnostics_base or PUBLIC_BASE, debug=debug
            )
        try:
            from vaft.omas.process_wrapper import compute_reconstructed_diamagnetic_flux
            diamag_recon = float(compute_reconstructed_diamagnetic_flux(ods, time_index=0))
        except Exception:
            pass
    return {
        "S_1": float(S1),
        "S_2": float(S2),
        "S_3": float(S3),
        "alpha": float(alpha),
        "R_0": float(R_0),
        "Z_0": float(Z_0),
        "B_pa": B_pa,
        "V_p": V_p,
        "beta_p": np.nan,
        "li": np.nan,
        "W_mag": np.nan,
        "W_kin": np.nan,
        "mui_hat": np.nan,
        "mui_exact": mui_exact,
        "diamagnetic_flux_measured": diamag_meas,
        "diamagnetic_flux_reconstructed": diamag_recon,
    }


def _get_shafranov_from_ods(ods, target_time: float):
    """
    Get S_1, S_2, S_3, alpha from ODS at the equilibrium time slice closest to target_time.
    Uses compute_virial_equilibrium_quantities_ods (requires magnetics.diamagnetic_flux in ODS).
    Returns dict with S_1, S_2, S_3, alpha and "eq_idx" used, or None if missing data / KeyError.
    """
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods

    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        return None
    times = []
    for i in range(len(ods["equilibrium.time_slice"])):
        t = ods["equilibrium.time_slice"][i].get("time")
        if t is None:
            continue
        try:
            times.append((i, float(t)))
        except (TypeError, ValueError):
            continue
    if not times:
        return None
    eq_idx = min(times, key=lambda x: abs(x[1] - target_time))[0]
    try:
        virial = compute_virial_equilibrium_quantities_ods(ods, time_slice=eq_idx)
    except KeyError:
        return None
    if eq_idx not in virial:
        return None
    v = virial[eq_idx]
    eq_ts = ods["equilibrium.time_slice"][eq_idx]
    def _f(x):
        try:
            return float(x) if x is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    R_0 = _f(eq_ts.get("boundary.geometric_axis.r"))
    Z_0 = _f(eq_ts.get("boundary.geometric_axis.z"))
    def _v(k, default=np.nan):
        x = v.get(k, default)
        return float(x) if np.isfinite(np.asarray(x, float)) else np.nan
    mui = _v("mui_hat") if "mui_hat" in v else np.nan
    mui_exact = np.nan
    try:
        from vaft.omas.process_wrapper import compute_diamagnetism
        mui_exact = float(compute_diamagnetism(ods, time_index=eq_idx))
    except Exception:
        pass
    diamag_meas = np.nan
    diamag_recon = np.nan
    try:
        if "constraints.diamagnetic_flux.measured" in eq_ts:
            diamag_meas = _f(eq_ts["constraints.diamagnetic_flux.measured"])
            diamag_recon = _f(eq_ts["constraints.diamagnetic_flux.reconstructed"])
        else:
            c = eq_ts.get("constraints")
            if c is not None and hasattr(c, "get"):
                df = c.get("diamagnetic_flux")
                if df is not None and hasattr(df, "get"):
                    diamag_meas = _f(df.get("measured"))
                    diamag_recon = _f(df.get("reconstructed"))
    except Exception:
        pass

    return {
        "S_1": float(v["s_1"]),
        "S_2": float(v["s_2"]),
        "S_3": float(v["s_3"]),
        "alpha": float(v["alpha"]),
        "R_0": R_0,
        "Z_0": Z_0,
        "B_pa": _v("B_pa"),
        "V_p": _v("V_p"),
        "beta_p": _v("beta_p"),
        "li": _v("li"),
        "W_mag": _v("W_mag"),
        "W_kin": _v("W_kin"),
        "mui_hat": mui,
        "mui_exact": mui_exact,
        "diamagnetic_flux_measured": diamag_meas,
        "diamagnetic_flux_reconstructed": diamag_recon,
        "eq_idx": eq_idx,
        "time": float(eq_ts.get("time", np.nan)),
    }


def _fmt_table_cell(val) -> str:
    """
    Format a numeric value for the comparison table. Returns "N/A" for None/nan.
    Ensures negative zero (-0.0) is shown as "-0" so it is not displayed as "0".
    """
    if val is None:
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    # Preserve sign for zero: -0.0 must not display as "0"
    if v == 0 and np.signbit(v):
        return "-0.000000"
    return f"{v:12.6g}"


def _print_comparison_table(
    labels_and_rows: list[tuple[str, dict]],
    title: str = "Shafranov integral comparison",
) -> None:
    """Print a comparison table. labels_and_rows = [(label1, dict1), (label2, dict2), ...]."""
    labels = [x[0] for x in labels_and_rows]
    rows = [x[1] for x in labels_and_rows]
    width = 14
    key_width = 10
    ncol = len(labels)
    sep = "-" * (2 + key_width + ncol * (width + 2))
    print(f"\n{title}")
    print(sep)
    header = "  " + "  ".join(f"{lb:>{width}s}" for lb in labels)
    print(f"  {'':{key_width}s}" + header)
    print(sep)
    for key in ("S_1", "S_2", "S_3", "alpha"):
        cells = [_fmt_table_cell(r.get(key)) for r in rows]
        line = "  " + "  ".join(f"{c:>{width}s}" for c in cells)
        print(f"  {key:{key_width}s}" + line)
    extra_rows = (
        ("R_0", "R_0 [m]"),
        ("Z_0", "Z_0 [m]"),
        ("B_pa", "B_pa [T]"),
        ("V_p", "V_p [m³]"),
        ("beta_p", "beta_p"),
        ("li", "li"),
        ("W_mag", "W_mag [J]"),
        ("W_kin", "W_kin [J]"),
        ("mui_hat", "μ̂_i"),
        ("mui_exact", "μ_i (exact)"),
        ("diamagnetic_flux_measured", "Δφ meas [Wb]"),
        ("diamagnetic_flux_reconstructed", "Δφ recon [Wb]"),
    )
    for key, label in extra_rows:
        if not any(
            r.get(key) is not None
            and (not isinstance(r.get(key), float) or np.isfinite(r.get(key)))
            for r in rows
        ):
            continue
        cells = [_fmt_table_cell(r.get(key)) for r in rows]
        line = "  " + "  ".join(f"{c:>{width}s}" for c in cells)
        print(f"  {label:{key_width}s}" + line)
    print(sep)


def _print_comparison(aeq_vals: dict, gfile_vals: dict, ods_vals: dict | None = None) -> None:
    """Print comparison: aeq, gfile→vaft, and optionally ods→vaft (database)."""
    labels_and_rows = [("aeq", aeq_vals), ("gfile→vaft", gfile_vals)]
    if ods_vals is not None:
        row_ods = {k: ods_vals.get(k) for k in (
            "S_1", "S_2", "S_3", "alpha", "R_0", "Z_0", "B_pa", "V_p",
            "beta_p", "li", "W_mag", "W_kin",
            "mui_hat", "mui_exact", "diamagnetic_flux_measured", "diamagnetic_flux_reconstructed",
        )}
        labels_and_rows.append(("ods→vaft", row_ods))
    _print_comparison_table(labels_and_rows)
    if ods_vals is not None and "eq_idx" in ods_vals:
        print(f"  (ods time_slice index = {ods_vals['eq_idx']}, time = {ods_vals.get('time', 'N/A')} s)")
        if ods_vals.get("R_0") is not None and gfile_vals.get("R_0") is not None:
            r0g, r0o = gfile_vals["R_0"], ods_vals["R_0"]
            if abs(r0o - r0g) > 0.01:
                print(f"  → S_2 ∝ R_0: ods R_0 differs from gfile ({r0o:.4f} vs {r0g:.4f} m); consider update_equilibrium_boundary(ods) or fix ODS geometric_axis.")


def compare_two_gfiles(
    path1: str,
    path2: str,
    label1: str = "efit gfile",
    label2: str = "chease gfile",
    load_magnetics_from_shot: int | None = None,
) -> None:
    """
    Load two g-files, compute Shafranov and virial via vaft, and print comparison.
    If load_magnetics_from_shot is set, diamagnetic flux is taken from that shot's ODS
    so that beta_p, li, W_mag, W_kin are also computed and printed.
    Example: EFIT g-file vs CHEASE g-file for the same shot/time.
    """
    v1 = _get_shafranov_from_gfile_ods(path1, load_magnetics_from_shot=load_magnetics_from_shot)
    v2 = _get_shafranov_from_gfile_ods(path2, load_magnetics_from_shot=load_magnetics_from_shot)
    _print_comparison_table(
        [(label1, v1), (label2, v2)],
        title="Shafranov integral comparison (two g-files)",
    )


def compare_by_shot_time(
    shot: int,
    time_s: float,
    base: str | None = None,
    load_magnetics_from_shot: int | None = None,
    debug: bool = False,
) -> None:
    """
    Build paths for the given shot/time under base (default PUBLIC_BASE), then compare
    Shafranov (and virial) from: CHEASE g-file, EFIT afile (aeq), EFIT g-file.
    Any missing file is skipped and not shown in the table.
    """
    base = base or PUBLIC_BASE
    if load_magnetics_from_shot is None:
        load_magnetics_from_shot = shot
    paths = paths_for_shot_time(shot, time_s, base=base)
    labels_and_rows: list[tuple[str, dict]] = []

    if os.path.isfile(paths["efit_afile"]):
        try:
            aeq_vals = _get_shafranov_from_aeq(paths["efit_afile"])
            labels_and_rows.append(("efit afile (aeq)", aeq_vals))
        except Exception as e:
            print(f"efit afile skipped: {e}")
    else:
        print(f"efit afile not found: {paths['efit_afile']}")

    if os.path.isfile(paths["efit_gfile"]):
        try:
            efit_vals = _get_shafranov_from_gfile_ods(
                paths["efit_gfile"],
                load_magnetics_from_shot=load_magnetics_from_shot,
                diagnostics_base=base,
                debug=debug,
                t_eq_override=time_s,
            )
            labels_and_rows.append(("efit gfile", efit_vals))
        except Exception as e:
            print(f"efit gfile skipped: {e}")
    else:
        print(f"efit gfile not found: {paths['efit_gfile']}")

    if os.path.isfile(paths["chease_gfile"]):
        try:
            chease_vals = _get_shafranov_from_gfile_ods(
                paths["chease_gfile"],
                load_magnetics_from_shot=load_magnetics_from_shot,
                diagnostics_base=base,
                debug=debug,
                t_eq_override=time_s,
            )
            labels_and_rows.append(("chease gfile", chease_vals))
        except Exception as e:
            print(f"chease gfile skipped: {e}")
    else:
        print(f"chease gfile not found: {paths['chease_gfile']}")

    if not labels_and_rows:
        print("No files found; nothing to compare.")
        return
    _print_comparison_table(
        labels_and_rows,
        title=f"Shafranov integral comparison (shot={shot}, time={time_s} s)",
    )


def _ods_time_index_for(ods, target_time: float) -> int | None:
    """Return equilibrium time_slice index whose time is closest to target_time [s], or None."""
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        return None
    times = []
    for i in range(len(ods["equilibrium.time_slice"])):
        t = ods["equilibrium.time_slice"][i].get("time")
        if t is None:
            continue
        try:
            times.append((i, float(t)))
        except (TypeError, ValueError):
            continue
    if not times:
        return None
    return min(times, key=lambda x: abs(x[1] - target_time))[0]


def test_diamagnetic_signal(shot: int, time_s: float, base: str | None = None) -> None:
    """
    Review/compare all diamagnetic-flux-related values for a given shot and time.
    Prints a table with:
      - aeqdsk: CDFLUX (computed, V·s), FLUXX (measured, V·s)
      - keqdsk: DFLUX (measured, mV·s → shown in Wb)
      - meqdsk: CDFLUX (computed, V·s), DIAMAG (measured, V·s)
      - diagnostics JSON: Δφ meas interpolated at time_s [Wb]
      - ODS: Δφ recon (volume integral from equilibrium), μ_i (exact, dimensionless)
    Paths: {base}/{shot}/efit/afile|kfile|mfile, and {base}/{shot}/omas/{shot}_diagnostics.json.
    """
    base = base or PUBLIC_BASE
    paths = paths_for_shot_time(shot, time_s, base=base)

    def _fmt(x: float, default: str = "N/A") -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return default
        return f"{float(x):.6g}"

    rows: list[tuple[str, str, str, str]] = []  # (file/source, variable, unit, value)

    # ---- aeqdsk: CDFLUX, FLUXX ----
    aeq_cdflux, aeq_fluxx = np.nan, np.nan
    if os.path.isfile(paths["efit_afile"]):
        try:
            from omfit_classes.omfit_eqdsk import OMFITaeqdsk
            aeq = OMFITaeqdsk(paths["efit_afile"])
            aeq_cdflux = _aeq_safe_get(aeq, "cdflux")
            aeq_fluxx = _aeq_safe_get(aeq, "fluxx")
            if not np.isfinite(aeq_fluxx):
                aeq_fluxx = _aeq_safe_get(aeq, "diamag")
        except Exception:
            pass
    rows.append(("aeqdsk", "CDFLUX", "V·s (Wb)", _fmt(aeq_cdflux)))
    # FLUXX: read as-is from OMFITaeqdsk (no unit conversion). If 1000× smaller than kfile DFLUX→Wb, aeq may use µV·s or EFIT output may differ.
    rows.append(("aeqdsk", "FLUXX", "V·s (Wb)", _fmt(aeq_fluxx)))

    # ---- keqdsk: DFLUX in file is mV·s; we show raw and Wb (1 mV·s = 1e-3 Wb) ----
    dflux_mVsec, dflux_Wb, sigdlc = _parse_kfile_dflux(paths["efit_kfile"])
    rows.append(("keqdsk", "DFLUX (raw)", "mV·s", _fmt(dflux_mVsec)))
    rows.append(("keqdsk", "DFLUX (→Wb)", "Wb (= mV·s×1e-3)", _fmt(dflux_Wb)))
    if np.isfinite(sigdlc):
        rows.append(("keqdsk", "SIGDLC", "-", _fmt(sigdlc)))

    # ---- meqdsk: CDFLUX, DIAMAG ----
    meq_cdflux, meq_diamag = _read_mfile_diamag(paths["efit_mfile"])
    rows.append(("meqdsk", "CDFLUX", "V·s (Wb)", _fmt(meq_cdflux)))
    rows.append(("meqdsk", "DIAMAG", "V·s (Wb)", _fmt(meq_diamag)))

    # ---- diagnostics JSON: Δφ meas at time_s ----
    diag_meas = _diamag_meas_from_diagnostics_json(shot, time_s, base=base)
    rows.append(("diagnostics JSON", "Δφ meas", "Wb", _fmt(diag_meas)))

    # ---- ODS: Δφ recon (integral), μ_i (exact) ----
    ods_recon = np.nan
    ods_mui = np.nan
    try:
        import vaft
        ods = vaft.database.load(shot)
        eq_idx = _ods_time_index_for(ods, time_s)
        if eq_idx is not None:
            from vaft.omas.process_wrapper import (
                compute_reconstructed_diamagnetic_flux,
                compute_diamagnetism,
            )
            ods_recon = float(compute_reconstructed_diamagnetic_flux(ods, time_index=eq_idx))
            ods_mui = float(compute_diamagnetism(ods, time_index=eq_idx))
    except Exception:
        pass
    rows.append(("ODS (integral)", "Δφ recon", "Wb", _fmt(ods_recon)))
    rows.append(("ODS (integral)", "μ_i (exact)", "-", _fmt(ods_mui)))

    # Print table
    print(f"Diamagnetic signal comparison (shot={shot}, time={time_s} s)")
    print("-" * 72)
    print(f"{'파일/소스':<22} {'변수':<18} {'단위':<18} {'값':<14}")
    print("-" * 72)
    for src, var, unit, val in rows:
        print(f"{src:<22} {var:<18} {unit:<18} {val:<14}")
    print("-" * 72)


def test_shafranov_integral_vs_aeq():
    """Compare Shafranov and virial (S_1..alpha, R_0, Z_0, B_pa, V_p, beta_p, li, W_mag, W_kin):
    aeq, g-file→vaft (with diamagnetic flux from experiment), and ODS (database)→vaft."""
    import pytest
    data_dir = _data_dir()
    g_path = os.path.join(data_dir, GFILE_NAME)
    aeq_path = os.path.join(data_dir, AEQ_NAME)
    if not os.path.isfile(g_path):
        pytest.skip("g-file sample not found")
    if not os.path.isfile(aeq_path):
        pytest.skip("aeq file sample not found")

    aeq_vals = _get_shafranov_from_aeq(aeq_path, skip_if_no_omfit=True)
    gfile_vals = _get_shafranov_from_gfile_ods(
        g_path, skip_if_no_omfit=True, load_magnetics_from_shot=SHOT_FOR_GFILE
    )
    ods_vals = None
    try:
        import vaft
        ods = vaft.database.load(SHOT_FOR_GFILE)
        ods_vals = _get_shafranov_from_ods(ods, TIME_FOR_GFILE)
    except Exception:
        pass

    _print_comparison(aeq_vals, gfile_vals, ods_vals)

    # Assert aeq vs g-file→vaft (primary comparison)
    rtol = 0.15
    atol = 0.05
    for key in ("S_1", "S_2", "S_3", "alpha"):
        a, v = aeq_vals[key], gfile_vals[key]
        np.testing.assert_allclose(v, a, rtol=rtol, atol=atol, err_msg=f"{key} aeq vs gfile_vaft")


if __name__ == "__main__":
    import sys
    # Compare CHEASE g, EFIT afile, EFIT g by shot and time: python test_shafranov_integral.py <shot> <time_s> [base] [--debug]
    # Diamagnetic signal review only: add --diamag
    if len(sys.argv) >= 3:
        args = [a for a in sys.argv[1:] if a not in ("--debug", "--diamag")]
        debug = "--debug" in sys.argv
        diamag_only = "--diamag" in sys.argv
        try:
            shot = int(args[0])
            time_s = float(args[1])
            base = args[2] if len(args) >= 3 else None
        except (ValueError, IndexError):
            print("Usage: python test_shafranov_integral.py <shot> <time_s> [base_dir] [--debug] [--diamag]")
            print("  e.g. python test_shafranov_integral.py 41670 0.320")
            print("  e.g. python test_shafranov_integral.py 39915 0.32 --debug")
            print("  e.g. python test_shafranov_integral.py 39915 0.32 --diamag   # diamagnetic signal table only")
            sys.exit(1)
        try:
            if diamag_only:
                test_diamagnetic_signal(shot, time_s, base=base)
            else:
                compare_by_shot_time(shot, time_s, base=base, load_magnetics_from_shot=shot, debug=debug)
        except ImportError as e:
            print("Missing dependency:", e)
        except Exception as e:
            print("Comparison failed:", e)
            raise
        sys.exit(0)

    data_dir = _data_dir()
    g_path = os.path.join(data_dir, GFILE_NAME)
    aeq_path = os.path.join(data_dir, AEQ_NAME)

    if not os.path.isfile(g_path):
        print(f"g-file not found: {g_path}")
    elif not os.path.isfile(aeq_path):
        print(f"aeq file not found: {aeq_path}")
    else:
        try:
            aeq_vals = _get_shafranov_from_aeq(aeq_path)
            gfile_vals = _get_shafranov_from_gfile_ods(
                g_path, load_magnetics_from_shot=SHOT_FOR_GFILE
            )
            ods_vals = None
            try:
                import vaft
                ods = vaft.database.load(SHOT_FOR_GFILE)
                ods_vals = _get_shafranov_from_ods(ods, TIME_FOR_GFILE)
            except Exception as e:
                print("ODS (database) case skipped:", e)
            _print_comparison(aeq_vals, gfile_vals, ods_vals)
        except ImportError as e:
            print("Missing dependency:", e)
            print("Install omfit_classes and vaft dependencies (e.g. scikit-learn) and run from repo root with PYTHONPATH=.")
