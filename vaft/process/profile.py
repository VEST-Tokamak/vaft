import vaft
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from uncertainties import unumpy
from omas import *
from vaft.formula import fit_profile


def _outermost_surface_path(geq):
    """Return a matplotlib Path around the outermost traced flux surface.

    Used to detect measurement points outside the plasma; the nearest-surface
    search would otherwise silently pin them to the edge psi_N level.
    """
    from matplotlib.path import Path as _MplPath

    levels = np.asarray(geq['fluxSurfaces']['levels'], dtype=float)
    outer = int(np.argmax(levels))
    R = np.asarray(geq['fluxSurfaces']['flux'][outer]['R'], dtype=float)
    Z = np.asarray(geq['fluxSurfaces']['flux'][outer]['Z'], dtype=float)
    return _MplPath(np.column_stack([R, Z]))


def _rho_from_equilibrium_points(geq, r_points, z_points):
    """Map R/Z points to normalized poloidal flux using VAFT GEQDSK/ODS data.

    Points outside the last closed flux surface are returned as NaN rather than
    pinned to the edge psi_N level.
    """
    # Compatibility with legacy precomputed flux-surface dictionaries.
    try:
        flux_levels = geq['fluxSurfaces']['levels']
        boundary = _outermost_surface_path(geq)
        mapped = []
        for r_dot, z_dot in zip(r_points, z_points):
            if not boundary.contains_point((float(r_dot), float(z_dot))):
                mapped.append(np.nan)
                continue
            min_dist = float('inf')
            closest_rho = None
            for i in range(len(geq['fluxSurfaces']['flux'])):
                R = np.asarray(geq['fluxSurfaces']['flux'][i]['R'], dtype=float)
                Z = np.asarray(geq['fluxSurfaces']['flux'][i]['Z'], dtype=float)
                dists = np.sqrt((R - r_dot) ** 2 + (Z - z_dot) ** 2)
                min_flux_dist = np.min(dists)
                if min_flux_dist < min_dist:
                    min_dist = min_flux_dist
                    closest_rho = flux_levels[i]
            mapped.append(closest_rho)
        return np.clip(np.asarray(mapped, dtype=float), 0.0, 1.0)
    except Exception:
        pass

    try:
        nw = int(geq['NW'])
        nh = int(geq['NH'])
        r_grid = np.linspace(0.0, float(geq['RDIM']), nw) + float(geq['RLEFT'])
        z_grid = np.linspace(0.0, float(geq['ZDIM']), nh) - float(geq['ZDIM']) / 2.0 + float(geq['ZMID'])
        psi = np.asarray(geq['PSIRZ'], dtype=float).reshape(nw, nh)
        psi_axis = float(geq['SIMAG'])
        psi_boundary = float(geq['SIBRY'])
    except Exception:
        try:
            ts = geq['equilibrium.time_slice.0'] if 'equilibrium.time_slice.0' in geq else geq['equilibrium.time_slice'][0]
            prof2d = ts['profiles_2d.0']
            r_grid = np.asarray(prof2d['grid.dim1'], dtype=float)
            z_grid = np.asarray(prof2d['grid.dim2'], dtype=float)
            psi = np.asarray(prof2d['psi'], dtype=float)
            if psi.shape == (z_grid.size, r_grid.size):
                psi = psi.T
            psi_axis = float(ts['global_quantities.psi_axis'])
            psi_boundary = float(ts['global_quantities.psi_boundary'])
        except Exception as exc:
            raise ValueError("geq must be a VAFT GEQDSK, OMAS equilibrium ODS, or legacy fluxSurfaces mapping") from exc

    interp = RegularGridInterpolator((r_grid, z_grid), psi, bounds_error=False, fill_value=np.nan)
    points = np.column_stack([np.asarray(r_points, dtype=float), np.asarray(z_points, dtype=float)])
    psi_points = interp(points)
    rho = (psi_points - psi_axis) / (psi_boundary - psi_axis) if psi_boundary != psi_axis else np.zeros_like(psi_points)
    # Outside the LCFS (rho > 1) → NaN rather than pinning to the edge psi_N level.
    rho = np.where(rho > 1.0, np.nan, rho)
    return np.clip(rho, 0.0, 1.0)


def equilibrium_mapping_thomson_scattering(ods, geq):
    """
    Map Thomson scattering positions to normalized poloidal flux coordinates (rho).

    This function finds the closest flux surface for each Thomson scattering measurement point
    and maps it to the corresponding normalized poloidal flux value (rho) in the equilibrium data.

    Parameters:
        ods (ODS): The OMAS data structure containing Thomson scattering positions.
        geq (dict): The equilibrium data containing flux surfaces and levels.

    Returns:
        numpy.ndarray: An array of mapped rho positions for each Thomson scattering point.
    """
    # Extract Thomson scattering positions
    r_t = ods['thomson_scattering.channel.:.position.r']
    z_t = ods['thomson_scattering.channel.:.position.z']
    return _rho_from_equilibrium_points(geq, r_t, z_t)

def profile_fitting_thomson_scattering(
    ods,
    time_ms,
    mapped_rho_position,
    Te_order=3,
    Ne_order=3,
    uncertainty_option=1,
    rho_points=100,
    fitting_function_te='polynomial',
    fitting_function_ne='polynomial',
    time_tolerance_ms=1.0,
    ):
    """
    Fit Thomson scattering Te and ne profiles with selectable 1D methods.

    Supported modes:
    - fitting_function_te, fitting_function_ne ∈ {'gp', 'polynomial', 'exponential', 'linear', 'core_poly_edge_exp'}

    Behavior:
    - Extracts Thomson data (Te, ne) at the requested time.
    - Fits Te(ρ) and ne(ρ) on ρ_tor_norm ∈ [0, 1] using the selected model for each.
    - If uncertainty_option == 1 and per-channel uncertainties are available, they are used as weights.
    - Edge behavior:
      - Polynomial/exponential basis includes a (1 - ρ) factor so the fitted profile tends toward 0 at ρ=1.
      - core_poly_edge_exp blends a core polynomial with an edge exponential using tanh(ρ) transition.
    - Returns callable fit functions for evaluating Te and ne on arbitrary ρ in [0, 1],
      plus the fitted mean profiles on a uniform evaluation grid.

    ## Arguments:
    - ods: OMAS data structure
    - time_ms: time in milliseconds
    - mapped_rho_position: list/array of mapped rho_tor_norm positions for Thomson channels
    - Te_order, Ne_order: polynomial order (used for polynomial/exponential/core_poly_edge_exp)
    - uncertainty_option: use uncertainties when fitting (1 = enabled)
    - rho_points: number of evaluation points on ρ in [0, 1]
    - fitting_function_te, fitting_function_ne: fit method selection for Te and ne

    ## Returns:
    - n_e_function, T_e_function: callable fit functions
    - coeffs_ne, coeffs_te: fit coefficients (None for GP/linear)
    - n_e_rho, T_e_rho: fitted profiles evaluated on rho_eval

    ## Example:
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='gp', fitting_function_ne='gp')
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='polynomial', fitting_function_ne='exponential')
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='linear', fitting_function_ne='linear')
    - profile_fitting_thomson_scattering(ods, time_ms, mapped_rho_position, fitting_function_te='core_poly_edge_exp', fitting_function_ne='core_poly_edge_exp')
    """
    # --- Extract Thomson data (nearest time within tolerance) ---
    times = np.asarray(ods['thomson_scattering.time'], dtype=float)
    target_s = time_ms / 1e3
    time_index = int(np.argmin(np.abs(times - target_s)))
    if abs(times[time_index] - target_s) > time_tolerance_ms / 1e3:
        raise ValueError(
            f"No Thomson time within {time_tolerance_ms} ms of {time_ms} ms "
            f"(nearest: {times[time_index] * 1e3:.3f} ms)"
        )

    num_channels = len(ods['thomson_scattering.channel'])
    t_e, n_e, t_e_std, n_e_std = [], [], [], []

    for i in range(num_channels):
        ch = ods['thomson_scattering.channel'][i]
        t_e.append(ch['t_e.data'][time_index])
        n_e.append(ch['n_e.data'][time_index])
        t_e_std.append(ch['t_e.data_error_upper'][time_index])
        n_e_std.append(ch['n_e.data_error_upper'][time_index])

    t_e = np.array(t_e, dtype=float)
    n_e = np.array(n_e, dtype=float)
    t_e_std = np.array(t_e_std, dtype=float)
    n_e_std = np.array(n_e_std, dtype=float)
    rho_flat = np.asarray(mapped_rho_position, dtype=float).reshape(-1)

    # --- drop invalid channels: non-finite values/sigmas, zero sigma, unmapped rho ---
    valid = (
        np.isfinite(rho_flat)
        & np.isfinite(t_e) & np.isfinite(n_e)
        & np.isfinite(t_e_std) & (t_e_std > 0)
        & np.isfinite(n_e_std) & (n_e_std > 0)
    )
    if not np.all(valid):
        print(
            f"[INFO] dropped {int(np.sum(~valid))} invalid TS channel(s) "
            f"at {time_ms:.3f} ms (non-finite value/sigma or unmapped position)"
        )
    t_e, n_e = t_e[valid], n_e[valid]
    t_e_std, n_e_std = t_e_std[valid], n_e_std[valid]
    rho = np.clip(rho_flat[valid].reshape(-1, 1), 0, 1)

    # relative sigma floors (in fit space) so tiny-but-valid sigmas cannot
    # dominate the weighted fit
    t_e_std = np.maximum(t_e_std, 1e-3 * np.max(np.abs(t_e)))

    # density normalization (floor applied AFTER normalization)
    n_e_scale = 1e18
    n_e_norm = n_e / n_e_scale
    n_e_std_norm = np.maximum(n_e_std / n_e_scale, 1e-3 * np.max(np.abs(n_e_norm)))
    rho_eval = np.linspace(0, 1, rho_points)

    # --- Te / Ne FITS ---
    te_anchor_strength = None
    te_anchor = None
    if te_anchor_strength is not None:
        te_anchor = (np.array([1.0]), np.array([0.0]), np.array([te_anchor_strength]))

    T_e_rho, T_e_std, T_e_function_raw, coeffs_te = fit_profile(
        rho,
        t_e,
        t_e_std,
        rho_eval,
        order=Te_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_te,
        gp_anchor=te_anchor,
    )

    T_e_rho = np.maximum(np.asarray(T_e_rho, dtype=float), 0.0)

    def T_e_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0, 1)
        return np.maximum(T_e_function_raw(x), 0.0)

    ne_anchor = None
    if fitting_function_ne.lower() == 'gp':
        ne_typ = np.nanmedian(n_e_norm[n_e_norm > 0]) if np.any(n_e_norm > 0) else 1.0
        ne_anchor_sigma_norm = max(0.01 * ne_typ, 1e-4)
        ne_anchor = (np.array([1.0]), np.array([0.0]), np.array([ne_anchor_sigma_norm]))

    n_e_rho_norm, n_e_std_norm_fit, n_e_function_raw, coeffs_ne = fit_profile(
        rho,
        n_e_norm,
        n_e_std_norm,
        rho_eval,
        order=Ne_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_ne,
        gp_anchor=ne_anchor,
    )

    n_e_rho = np.maximum(n_e_rho_norm, 0.0) * n_e_scale
    n_e_std = n_e_std_norm_fit * n_e_scale

    def n_e_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0, 1)
        y_norm = n_e_function_raw(x)
        return np.maximum(y_norm, 0.0) * n_e_scale

    return n_e_function, T_e_function, coeffs_ne, coeffs_te, n_e_rho, T_e_rho


def equilibrium_mapping_charge_exchange(ods, geq):
    """
    Map charge_exchange (CES) channel positions to normalized poloidal flux (rho).

    This is analogous to equilibrium_mapping_thomson_scattering but uses
    charge_exchange.channel[:].position.(r,z) as the measurement points.

    Args:
        ods: OMAS data structure containing charge_exchange positions.
        geq: Equilibrium data (same structure as used for Thomson mapping).

    Returns:
        numpy.ndarray: mapped rho positions for each charge_exchange channel.
    """
    def _to_float_scalar(x):
        try:
            x = unumpy.nominal_values(x)
        except Exception:
            pass

        arr = np.asarray(x)
        if arr.size == 0:
            return float("nan")

        if arr.dtype.kind in {"U", "S", "O"}:
            try:
                arr = arr.astype(float)
            except Exception:
                return float(str(arr.reshape(-1)[0]))
        else:
            arr = arr.astype(float, copy=False)

        return float(arr.reshape(-1)[0])

    # Prefer OMAS `.data` leaves (what CES machine-mapping writes), fall back for compatibility.
    try:
        R_ce = ods["charge_exchange.channel.:.position.r.data"]
        Z_ce = ods["charge_exchange.channel.:.position.z.data"]
    except Exception:
        R_ce = ods["charge_exchange.channel.:.position.r"]
        Z_ce = ods["charge_exchange.channel.:.position.z"]

    r_vals = [_to_float_scalar(value) for value in R_ce]
    z_vals = [_to_float_scalar(value) for value in Z_ce]
    return _rho_from_equilibrium_points(geq, r_vals, z_vals)


def _filter_channels_for_fit(values, sigmas, rho, label, time_ms):
    """Drop channels with non-finite values/rho; handle invalid sigmas.

    Channels whose sigma is non-finite or <= 0 are dropped, unless NO channel
    has a valid sigma (e.g. data stored without uncertainties) — then all
    value-valid channels are kept with a uniform relative sigma. Surviving
    sigmas are floored at 1e-3 * max|value| so a tiny-but-valid sigma cannot
    dominate the weighted fit.
    """
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    rho = np.asarray(rho, dtype=float).reshape(-1)

    valid = np.isfinite(rho) & np.isfinite(values)
    sigma_ok = np.isfinite(sigmas) & (sigmas > 0)
    if np.any(valid & sigma_ok):
        valid &= sigma_ok
    else:
        sigmas = np.full_like(values, np.nan)  # replaced by the floor below

    dropped = int(np.sum(~valid))
    if dropped:
        print(
            f"[INFO] dropped {dropped} invalid {label} channel(s) at {time_ms:.3f} ms"
        )

    values, sigmas, rho = values[valid], sigmas[valid], rho[valid]
    floor = 1e-3 * float(np.max(np.abs(values))) if values.size else 1.0
    if not np.isfinite(floor) or floor <= 0:
        floor = 1.0
    sigmas = np.where(np.isfinite(sigmas) & (sigmas > 0), sigmas, floor)
    sigmas = np.maximum(sigmas, floor)
    return values, sigmas, rho


def _leaf_values_and_errors(node, time_index, clamp=False):
    """Return (nominal, sigma) scalars for an OMAS signal leaf at ``time_index``.

    OMAS (>=0.94.2) splits an assigned uarray immediately into ``<leaf>.data``
    (nominal) and ``<leaf>.data_error_upper``, so the re-read ``.data`` carries
    NO uncertainty and ``unumpy.std_devs(<leaf>.data)`` is ALL ZEROS. Read the
    stored ``.data_error_upper`` explicitly (mirroring the Thomson path) to
    recover the real per-channel sigma; fall back to any uncertainty still
    attached to ``.data``, else 0.

    ``node`` is the signal sub-node itself (e.g. ``ods[...ion.0.t_i]``).
    With ``clamp=True`` an out-of-range ``time_index`` is clamped to the last
    sample instead of raising (used for best-effort metadata extraction).
    """
    data = node['data']
    try:
        vals = np.asarray(unumpy.nominal_values(data), dtype=float)
    except Exception:
        vals = np.asarray(data, dtype=float)

    errs = None
    try:
        errs = np.asarray(unumpy.nominal_values(node['data_error_upper']), dtype=float)
    except Exception:
        try:
            errs = np.asarray(unumpy.std_devs(data), dtype=float)
        except Exception:
            errs = None
    if errs is None or errs.shape != vals.shape:
        errs = np.zeros_like(vals)

    if vals.ndim == 0:
        return float(vals), float(np.abs(errs))

    idx = int(time_index)
    if idx < 0 or idx >= vals.size:
        if clamp:
            idx = min(max(idx, 0), vals.size - 1)
        else:
            raise IndexError("charge_exchange signal .data shorter than time base")
    return float(vals[idx]), float(np.abs(errs[idx]))


def _sanitize_std(sigmas):
    """Replace 0 / NaN sigmas with the median of the valid ones.

    Even after :func:`_leaf_values_and_errors` recovers the real errors, a
    handful of channels can carry an exactly-zero or NaN sigma. A single ~0
    sigma explodes its ``1/sigma**2`` weight (~1e14) and lets one channel
    dominate the weighted fit. Substituting the median of the valid sigmas keeps
    every channel informative without a single-channel blow-up. If NO sigma is
    usable, fall back to a uniform sigma (an honestly unweighted fit).
    """
    sigmas = np.asarray(sigmas, dtype=float).copy()
    valid = np.isfinite(sigmas) & (sigmas > 0)
    if not np.any(valid):
        sigmas[:] = 1.0
        return sigmas
    med = float(np.median(sigmas[valid]))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    sigmas[~valid] = med
    return sigmas


def profile_fitting_charge_exchange(
    ods,
    time_ms,
    mapped_rho_position,
    Ti_order=3,
    Vtor_order=3,
    uncertainty_option=1,
    rho_points=100,
    fitting_function_ti='polynomial',
    fitting_function_vtor='polynomial',
    ion_index=0,
    time_tolerance_ms=1.0,
):
    """
    Fit charge_exchange (CES) ion temperature and toroidal velocity profiles.

    This mirrors profile_fitting_thomson_scattering, but for:
    - T_i(ρ): ion temperature from charge_exchange.channel[:].ion[ion_index].t_i.data
    - V_tor(ρ): toroidal ion velocity from charge_exchange.channel[:].ion[ion_index].velocity_tor.data

    Args:
        ods: OMAS data structure containing charge_exchange.
        time_ms: time in milliseconds (matched to charge_exchange.time).
        mapped_rho_position: mapped rho_tor_norm positions for CES channels.
        Ti_order, Vtor_order: polynomial order for respective fits.
        uncertainty_option: passed through to fit_profile.
        rho_points: number of evaluation points on ρ ∈ [0, 1].
        fitting_function_ti, fitting_function_vtor: fit methods for T_i and V_tor.
        ion_index: ion index within charge_exchange.channel[i].ion.

    Returns:
        (Vtor_function, Ti_function, coeffs_vtor, coeffs_ti, Vtor_rho, Ti_rho)
    """
    # --- time index ---
    times = np.asarray(ods['charge_exchange.time'], dtype=float)
    if times.ndim != 1:
        raise ValueError("charge_exchange.time must be 1D")

    target_s = time_ms / 1e3
    time_index = int(np.argmin(np.abs(times - target_s)))
    if abs(times[time_index] - target_s) > time_tolerance_ms / 1e3:
        raise ValueError(
            f"No charge_exchange time within {time_tolerance_ms} ms of {time_ms} ms "
            f"(nearest: {times[time_index] * 1e3:.3f} ms)"
        )

    num_channels = len(ods['charge_exchange.channel'])
    Ti, Vtor, Ti_std, Vtor_std = [], [], [], []

    for i in range(num_channels):
        ion = ods[f'charge_exchange.channel.{i}.ion.{ion_index}']

        # OMAS (>=0.94.2) stores an assigned uarray as `<leaf>.data` (nominal) +
        # `<leaf>.data_error_upper`; the re-read `.data` carries NO uncertainty,
        # so unumpy.std_devs(<leaf>.data) is ALL ZEROS. Read `.data_error_upper`
        # explicitly (mirrors the Thomson path) to recover the real sigma.
        ti_val, ti_err = _leaf_values_and_errors(ion['t_i'], time_index)
        v_val, v_err = _leaf_values_and_errors(ion['velocity_tor'], time_index)

        Ti.append(ti_val)
        Ti_std.append(ti_err)
        Vtor.append(v_val)
        Vtor_std.append(v_err)

    rho_flat = np.asarray(mapped_rho_position, dtype=float).reshape(-1)
    Ti, Ti_std, rho_ti = _filter_channels_for_fit(Ti, Ti_std, rho_flat, "CES T_i", time_ms)
    Vtor, Vtor_std, rho_v = _filter_channels_for_fit(Vtor, Vtor_std, rho_flat, "CES V_tor", time_ms)

    # Replace any residual 0/NaN sigma (from OMAS-split leaves) with the
    # valid-error median so a single near-zero sigma cannot blow up its
    # 1/sigma**2 weight and dominate the fit.
    Ti_std = _sanitize_std(Ti_std)
    Vtor_std = _sanitize_std(Vtor_std)

    rho = np.clip(rho_ti.reshape(-1, 1), 0.0, 1.0)
    rho_vtor = np.clip(rho_v.reshape(-1, 1), 0.0, 1.0)
    rho_eval = np.linspace(0.0, 1.0, rho_points)

    # --- Fit T_i(ρ) ---
    Ti_rho, Ti_std_fit, Ti_function_raw, coeffs_ti = fit_profile(
        rho,
        Ti,
        Ti_std,
        rho_eval,
        order=Ti_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_ti,
        gp_anchor=None,
    )

    Ti_rho = np.maximum(np.asarray(Ti_rho, dtype=float), 0.0)

    def Ti_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0.0, 1.0)
        return np.maximum(Ti_function_raw(x), 0.0)

    # --- Fit V_tor(ρ) ---
    Vtor_rho, Vtor_std_fit, Vtor_function_raw, coeffs_vtor = fit_profile(
        rho_vtor,
        Vtor,
        Vtor_std,
        rho_eval,
        order=Vtor_order,
        uncertainty_option=uncertainty_option,
        fitting_function=fitting_function_vtor,
        gp_anchor=None,
    )

    def Vtor_function(rho_input):
        x = np.clip(np.asarray(rho_input, float), 0.0, 1.0)
        return Vtor_function_raw(x)

    return Vtor_function, Ti_function, coeffs_vtor, coeffs_ti, Vtor_rho, Ti_rho
def _equilibrium_grid_at_time(ods, target_s, tol_s):
    """Return (rho_tor_norm, psi, psi_N) of the equilibrium slice at target_s, or None."""
    try:
        eq_times = np.asarray(ods['equilibrium.time'], dtype=float).reshape(-1)
    except Exception:
        return None
    if eq_times.size == 0:
        return None
    idx = int(np.argmin(np.abs(eq_times - target_s)))
    if abs(eq_times[idx] - target_s) > tol_s:
        return None
    try:
        rho = np.asarray(
            ods[f'equilibrium.time_slice.{idx}.profiles_1d.rho_tor_norm'], dtype=float
        )
        psi = np.asarray(
            ods[f'equilibrium.time_slice.{idx}.profiles_1d.psi'], dtype=float
        )
    except Exception:
        return None
    if rho.ndim != 1 or rho.shape != psi.shape or psi[-1] == psi[0]:
        return None
    psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
    return rho, psi, psi_n


def core_profiles(
    ods,
    time_ms,
    mapped_rho_position,
    n_e_function,
    T_e_function,
    tol_ms=0.1,
    T_i_function=None,
    V_tor_function=None,
    ti_mapped_rho_position=None,
    rho_points=100,
    time_tolerance_ms=1.0,
):
    """
    Construct and store core_profiles.profiles_1d from fitted kinetic profiles.

    The fit functions are functions of psi_N (normalized poloidal flux — the
    coordinate produced by the equilibrium mappers). When the ODS carries an
    equilibrium slice at the same time, profiles are evaluated at the psi_N of
    each equilibrium grid point and stored on the equilibrium's rho_tor_norm/psi
    grid; otherwise a uniform psi_N grid is stored honestly as
    grid.rho_pol_norm = sqrt(psi_N).

    If a profile already exists for the same time (within tol_ms), it is replaced.

    ## Arguments:
    - ods: OMAS data structure (mutable)
    - time_ms: time in milliseconds
    - mapped_rho_position: mapped psi_N positions of the Thomson channels
    - n_e_function, T_e_function: callable fit functions of psi_N for ne, Te
    - tol_ms: tolerance in milliseconds for duplicate time detection
    - T_i_function: optional callable Ti(psi_N) from the charge_exchange fit;
      when given, ion.0.temperature uses real Ti (instead of Ti=Te) and
      pressure_thermal = e*ne*(Te+Ti) is written (n_i ~= n_e)
    - V_tor_function: optional callable Vtor(psi_N) -> ion.0.velocity.toroidal
    - ti_mapped_rho_position: mapped psi_N positions of the charge_exchange
      channels (for the ion temperature_fit measurement metadata)
    - rho_points: evaluation points for the no-equilibrium fallback grid
    - time_tolerance_ms: tolerance for matching thomson/equilibrium times

    ## Returns:
    - Updated ods with replaced or appended core_profiles.profiles_1d entry.
    """
    e_J_per_eV = 1.602176634e-19
    target_s = time_ms / 1e3
    tol_s = time_tolerance_ms / 1e3

    # --- measured TS points (nearest time within tolerance) ---
    ts_times = np.asarray(ods['thomson_scattering.time'], dtype=float)
    t_idx = int(np.argmin(np.abs(ts_times - target_s)))
    if abs(ts_times[t_idx] - target_s) > tol_s:
        raise ValueError(
            f"No Thomson time within {time_tolerance_ms} ms of {time_ms} ms "
            f"(nearest: {ts_times[t_idx] * 1e3:.3f} ms)"
        )
    num_channels = len(ods['thomson_scattering.channel'])
    n_e_meas = np.array(
        [ods[f'thomson_scattering.channel.{i}.n_e.data'][t_idx] for i in range(num_channels)],
        dtype=float,
    )
    T_e_meas = np.array(
        [ods[f'thomson_scattering.channel.{i}.t_e.data'][t_idx] for i in range(num_channels)],
        dtype=float,
    )
    rho_meas = np.asarray(mapped_rho_position, dtype=float).reshape(-1)

    # --- evaluation grid: equilibrium grid when available, else uniform psi_N ---
    eq_grid = _equilibrium_grid_at_time(ods, target_s, tol_s)
    if eq_grid is not None:
        rho_tor_grid, psi_grid, psi_n_grid = eq_grid
    else:
        rho_tor_grid = psi_grid = None
        psi_n_grid = np.linspace(0.0, 1.0, rho_points)

    n_e_recon = np.asarray(n_e_function(psi_n_grid), dtype=float)
    T_e_recon = np.asarray(T_e_function(psi_n_grid), dtype=float)
    T_i_recon = (
        np.asarray(T_i_function(psi_n_grid), dtype=float)
        if T_i_function is not None
        else T_e_recon
    )
    V_tor_recon = (
        np.asarray(V_tor_function(psi_n_grid), dtype=float)
        if V_tor_function is not None
        else None
    )

    # --- check for duplicate time entries ---
    existing_times = []
    if 'core_profiles.profiles_1d' in ods:
        n_profiles = len(ods['core_profiles.profiles_1d'])
        for i in range(n_profiles):
            try:
                t_existing = ods[f'core_profiles.profiles_1d.{i}.time']
                if abs(t_existing * 1000 - time_ms) < tol_ms:
                    existing_times.append(i)
            except Exception:
                continue

    # --- remove duplicates before writing ---
    for i in sorted(existing_times, reverse=True):
        ods.pop(f'core_profiles.profiles_1d.{i}')
        print(f"[INFO] Removed duplicate core_profile at {time_ms:.3f} ms (index {i})")

    # --- Determine next available index after removal ---
    next_idx = len(ods['core_profiles.profiles_1d']) if 'core_profiles.profiles_1d' in ods else 0
    base = f'core_profiles.profiles_1d.{next_idx}'

    ods[f'{base}.time'] = target_s
    if eq_grid is not None:
        ods[f'{base}.grid.rho_tor_norm'] = rho_tor_grid
        ods[f'{base}.grid.psi'] = psi_grid
    else:
        # psi_N-based grid: rho_pol_norm = sqrt(psi_N); do NOT mislabel as rho_tor_norm
        ods[f'{base}.grid.rho_pol_norm'] = np.sqrt(psi_n_grid)

    # We assume all electrons are thermal electrons (because VEST is ohmically heated plasma)
    ods[f'{base}.electrons.density_thermal'] = n_e_recon
    ods[f'{base}.electrons.density'] = n_e_recon
    ods[f'{base}.electrons.temperature'] = T_e_recon

    # single main ion H+ with n_i ~= n_e (quasi-neutrality, no impurity dilution);
    # Ti(H+) taken equal to the measured impurity Ti when a fit is provided
    ods[f'{base}.ion.0.label'] = 'H+'
    ods[f'{base}.ion.0.z_ion'] = 1.0
    ods[f'{base}.ion.0.element.0.a'] = 1.00784
    ods[f'{base}.ion.0.element.0.z_n'] = 1.0
    ods[f'{base}.ion.0.element.0.atoms_n'] = 1
    ods[f'{base}.ion.0.density_thermal'] = n_e_recon
    ods[f'{base}.ion.0.density'] = n_e_recon
    ods[f'{base}.ion.0.temperature'] = T_i_recon
    if V_tor_recon is not None:
        ods[f'{base}.ion.0.velocity.toroidal'] = V_tor_recon
    if T_i_function is not None:
        ods[f'{base}.pressure_thermal'] = e_J_per_eV * n_e_recon * (T_e_recon + T_i_recon)

    # --- measurement/fit metadata (per IMAS DD, measured/reconstructed/rho all
    # have one entry per measurement point; reconstructed = fit AT those points) ---
    if eq_grid is not None:
        finite_meas = np.isfinite(rho_meas)
        rho_meas_psin = np.clip(rho_meas[finite_meas], 0, 1)
        rho_meas_tor = np.interp(rho_meas_psin, psi_n_grid, rho_tor_grid)

        fit_base_n = f'{base}.electrons.density_fit'
        fit_base_t = f'{base}.electrons.temperature_fit'
        ods[f'{fit_base_n}.rho_tor_norm'] = rho_meas_tor
        ods[f'{fit_base_n}.measured'] = n_e_meas[finite_meas]
        ods[f'{fit_base_n}.reconstructed'] = np.asarray(
            n_e_function(rho_meas_psin), dtype=float
        )
        ods[f'{fit_base_t}.rho_tor_norm'] = rho_meas_tor
        ods[f'{fit_base_t}.measured'] = T_e_meas[finite_meas]
        ods[f'{fit_base_t}.reconstructed'] = np.asarray(
            T_e_function(rho_meas_psin), dtype=float
        )

        if T_i_function is not None and ti_mapped_rho_position is not None:
            try:
                ce_times = np.asarray(ods['charge_exchange.time'], dtype=float)
                ce_idx = int(np.argmin(np.abs(ce_times - target_s)))
                n_ce = len(ods['charge_exchange.channel'])
                ti_meas, ti_meas_err = [], []
                for i in range(n_ce):
                    # OMAS split the assigned uarray -> read .data + .data_error_upper
                    val, err = _leaf_values_and_errors(
                        ods[f'charge_exchange.channel.{i}.ion.0.t_i'],
                        ce_idx,
                        clamp=True,
                    )
                    ti_meas.append(val)
                    ti_meas_err.append(err)
                ti_meas = np.asarray(ti_meas, dtype=float)
                ti_meas_err = np.asarray(ti_meas_err, dtype=float)
                ti_rho = np.asarray(ti_mapped_rho_position, dtype=float).reshape(-1)
                finite_ti = np.isfinite(ti_rho)
                ti_rho_psin = np.clip(ti_rho[finite_ti], 0, 1)
                fit_base_ti = f'{base}.ion.0.temperature_fit'
                ods[f'{fit_base_ti}.rho_tor_norm'] = np.interp(
                    ti_rho_psin, psi_n_grid, rho_tor_grid
                )
                ods[f'{fit_base_ti}.measured'] = ti_meas[finite_ti]
                ods[f'{fit_base_ti}.measured_error_upper'] = ti_meas_err[finite_ti]
                ods[f'{fit_base_ti}.reconstructed'] = np.asarray(
                    T_i_function(ti_rho_psin), dtype=float
                )
            except Exception as exc:
                print(f"[WARN] could not attach ion temperature_fit metadata: {exc}")

    # --- IDS bookkeeping: homogeneous time base matching profiles_1d order ---
    ods['core_profiles.ids_properties.homogeneous_time'] = 1
    n_profiles = len(ods['core_profiles.profiles_1d'])
    ods['core_profiles.time'] = np.array(
        [float(ods[f'core_profiles.profiles_1d.{i}.time']) for i in range(n_profiles)]
    )

    print(f"[UPDATED] core_profile at {time_ms:.3f} ms (index {next_idx})")
    return ods

def core_profiles_from_eq(
    ods,
    Te0_eV,
    rho_fit=None,
    tol_ms=0.1,
    eq_time_index=0,
    ):
    """
    Build synthetic core_profiles.profiles_1d from equilibrium pressure with:
        P(Pa) = 2 * ne(m^-3) * Te(eV) * e(J/eV)

    Assumptions:
      - same shape: ne = ne0*g, Te = Te0*g
      - g(rho) = sqrt(P(rho)/P(0))
      - Ti = Te is implicitly absorbed in factor 2
      - No Zeff / impurity modeling

    Reads:
      rho_src = ods['equilibrium.time_slice.<idx>.profiles_1d.rho_tor_norm']
      p_src   = ods['equilibrium.time_slice.<idx>.profiles_1d.pressure']   # Pa

    Writes:
      ods['core_profiles.profiles_1d.<next_idx>.*']
    """
    e_J_per_eV = 1.602176634e-19

    time_ms = ods['equilibrium.time'][eq_time_index] * 1e3

    if rho_fit is None:
        rho_fit = np.linspace(0.0, 1.0, 100)
    else:
        rho_fit = np.asarray(rho_fit, dtype=float)

    rho_src_path = f"equilibrium.time_slice.{eq_time_index}.profiles_1d.rho_tor_norm"
    p_src_path   = f"equilibrium.time_slice.{eq_time_index}.profiles_1d.pressure"

    rho_src = np.asarray(ods[rho_src_path], dtype=float)
    p_src   = np.asarray(ods[p_src_path], dtype=float)  # Pa

    if rho_src.ndim != 1 or p_src.ndim != 1:
        raise ValueError("Expected 1D rho_tor_norm and 1D pressure at the selected time_slice.")

    order = np.argsort(rho_src)
    rho_src = rho_src[order]
    p_src = p_src[order]

    P_fit = np.interp(rho_fit, rho_src, p_src)

    if np.any(~np.isfinite(P_fit)):
        raise ValueError("Pressure profile contains NaN/Inf.")
    if P_fit[0] <= 0:
        raise ValueError("Pressure at rho=0 must be > 0 to define sqrt shape.")

    g = np.sqrt(np.clip(P_fit / P_fit[0], 0.0, None))

    Te = Te0_eV * g  # eV

    # ---- FIX: include eV->J ----
    ne0 = P_fit[0] / (2.0 * Te0_eV * e_J_per_eV)  # m^-3
    ne = ne0 * g

    if np.any(ne < 0) or np.any(Te < 0):
        raise ValueError("Generated ne/Te has negative values (check pressure and Te0_eV).")

    # ---- remove duplicates ----
    existing_idxs = []
    if "core_profiles.profiles_1d" in ods:
        for i in range(len(ods["core_profiles.profiles_1d"])):
            try:
                t_existing = ods[f"core_profiles.profiles_1d.{i}.time"]  # s
                if abs(t_existing * 1000.0 - time_ms) < tol_ms:
                    existing_idxs.append(i)
            except Exception:
                continue

    for i in sorted(existing_idxs, reverse=True):
        ods.pop(f"core_profiles.profiles_1d.{i}")
        print(f"[INFO] Removed duplicate core_profile at {time_ms:.3f} ms (index {i})")

    next_idx = len(ods["core_profiles.profiles_1d"]) if "core_profiles.profiles_1d" in ods else 0
    base = f"core_profiles.profiles_1d.{next_idx}"

    ods[f"{base}.time"] = time_ms / 1000.0
    ods[f"{base}.grid.rho_tor_norm"] = rho_fit.tolist()

    ods[f"{base}.electrons.density_thermal"] = ne.tolist()
    ods[f"{base}.electrons.density"] = ne.tolist()
    ods[f"{base}.electrons.temperature"] = Te.tolist()

    ods[f"{base}.ion.0.label"] = "H+"
    ods[f"{base}.ion.0.density_thermal"] = ne.tolist()
    ods[f"{base}.ion.0.density"] = ne.tolist()
    ods[f"{base}.ion.0.temperature"] = Te.tolist()

    print(f"[UPDATED] core_profile from eq pressure (Pa) at {time_ms:.3f} ms "
          f"(index {next_idx}), eq_time_slice={eq_time_index}")
    return ods

def core_profiles_from_eq_ratio(
    ods,
    C_ne_over_Te,   # density / temperature ratio
    rho_fit=None,
    tol_ms=0.1,
    eq_time_index=0,
    ):
    """
    Build synthetic core_profiles from equilibrium pressure using:
        n_e = C * sqrt(f)
        T_e = sqrt(f)
        P(Pa) = 2 * n_e * T_e * e

    where f(rho) = P(rho) / P(0)

    Parameters
    ----------
    C_ne_over_Te : float
        Density / temperature ratio (m^-3 / eV)
    """

    e_J = 1.602176634e-19
    time_ms = ods['equilibrium.time'][eq_time_index] * 1e3

    if rho_fit is None:
        rho_fit = np.linspace(0, 1, 100)

    rho_src = np.asarray(
        ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.rho_tor_norm']
    )
    p_src = np.asarray(
        ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.pressure']
    )

    order = np.argsort(rho_src)
    rho_src, p_src = rho_src[order], p_src[order]

    P_fit = np.interp(rho_fit, rho_src, p_src)

    if P_fit[0] <= 0:
        raise ValueError("Invalid pressure profile")

    # --- shape ---
    f = P_fit / P_fit[0]

    # --- build profiles ---
    Te = np.sqrt(f)                 # eV (relative)
    ne = C_ne_over_Te * Te          # m^-3

    # --- scale to absolute pressure ---
    scale = P_fit[0] / (2 * C_ne_over_Te * e_J)
    Te *= np.sqrt(scale)
    ne *= np.sqrt(scale)

    # --- remove duplicates ---
    existing = []
    if 'core_profiles.profiles_1d' in ods:
        for i in range(len(ods['core_profiles.profiles_1d'])):
            t = ods[f'core_profiles.profiles_1d.{i}.time']
            if abs(t * 1000 - time_ms) < tol_ms:
                existing.append(i)
    for i in reversed(existing):
        ods.pop(f'core_profiles.profiles_1d.{i}')

    next_idx = len(ods['core_profiles.profiles_1d'])
    base = f'core_profiles.profiles_1d.{next_idx}'

    ods[f'{base}.time'] = time_ms / 1000
    ods[f'{base}.grid.rho_tor_norm'] = rho_fit.tolist()

    ods[f'{base}.electrons.density'] = ne.tolist()
    ods[f'{base}.electrons.density_thermal'] = ne.tolist()
    ods[f'{base}.electrons.temperature'] = Te.tolist()

    ods[f'{base}.ion.0.label'] = 'H+'
    ods[f'{base}.ion.0.density'] = ne.tolist()
    ods[f'{base}.ion.0.density_thermal'] = ne.tolist()
    ods[f'{base}.ion.0.temperature'] = Te.tolist()

    print(f"[UPDATED] core_profile from eq (ratio-fixed) at {time_ms:.2f} ms")
    return ods

def export_electron_profile_txt(
    n_e_function,
    T_e_function,
    n_e_coeff,
    T_e_coeff,
    rho_points=100,
    filename='electron_profiles.txt',
    ):
    """
    Export the fitted electron temperature and density profiles to a text file.

    This function evaluates the fitted electron temperature and density profiles at
    a specified number of rho points and exports the results to a text file.

    Parameters:
        n_e_function (callable): Function to compute fitted electron density at any rho.
        T_e_function (callable): Function to compute fitted electron temperature at any rho.
        n_e_coeff (numpy.ndarray): Coefficients of the fitted Ne function.
        T_e_coeff (numpy.ndarray): Coefficients of the fitted Te function.
        rho_points (int, optional): Number of rho points to evaluate the fitted profiles.
        filename (str, optional): The name of the text file to save the profiles to.
    """
    rho_eval = np.linspace(0, 1, rho_points)
    n_e_rho = n_e_function(rho_eval)
    T_e_rho = T_e_function(rho_eval)

    with open(filename, 'w') as f:
        f.write('psi_N, T_e [eV], n_e [m-3]\n')
        for rho, T_e, n_e in zip(rho_eval, T_e_rho, n_e_rho):
            f.write(f'{rho}, {T_e}, {n_e}\n')
