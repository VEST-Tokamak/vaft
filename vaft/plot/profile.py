import numpy as np
import matplotlib.pyplot as plt
import omas
from matplotlib.ticker import ScalarFormatter


def _extract_nominal_and_error(arr):
    """
    Utility to extract nominal values and 1-sigma errors from arrays that may
    contain uncertainties.unumpy objects. Falls back to zero error if the
    uncertainties package is unavailable.
    """
    try:
        from uncertainties import unumpy as unp

        np_arr = np.asarray(arr)
        if np_arr.dtype == object:
            vals = np.asarray(unp.nominal_values(np_arr), float)
            errs = np.asarray(unp.std_devs(np_arr), float)
        else:
            vals = np.asarray(np_arr, float)
            errs = np.zeros_like(vals)
    except Exception:
        vals = np.asarray(arr, float)
        errs = np.zeros_like(vals)
    vals_s = np.squeeze(vals)
    errs_s = np.squeeze(errs)

    return vals_s, errs_s


def plot_thomson_radial_position(ods, contour_quantity='psi_norm'):
    """
    Plot Thomson radial positions on the first available equilibrium boundary and wall.

    Automatically picks the first equilibrium time slice that actually contains data.
    """
    fig, ax = plt.subplots(figsize=(3, 4))

    # --- equilibrium slice with data 찾기 ---
    eq_slices = ods['equilibrium']['time_slice']
    time_index = None
    for i, eq in enumerate(eq_slices):
        try:
            if len(eq['boundary.outline.r']) > 0 and len(eq['boundary.outline.z']) > 0:
                time_index = i
                break
        except Exception:
            continue

    if time_index is None:
        time_index = 0
        flag = False
    else:
        flag = True

    eq = eq_slices[time_index]
    print(f"[INFO] Using equilibrium time slice {time_index}")

    # --- Wall geometry (optional) ---
    def _iter_values(container):
        """Yield values for dict/list/OMAS-like containers."""
        if isinstance(container, dict):
            for value in container.values():
                yield value
            return
        try:
            for idx in range(len(container)):
                yield container[idx]
            return
        except Exception:
            return

    if 'wall' in ods and 'description_2d' in ods['wall']:
        wall_desc = ods['wall']['description_2d']
        for desc in _iter_values(wall_desc):
            try:
                limiter = desc['limiter']
                units = limiter['unit']
            except Exception:
                continue
            for unit in _iter_values(units):
                try:
                    ax.plot(unit['outline.r'], unit['outline.z'], color='gray', alpha=0.4)
                except Exception:
                    continue

    # --- Thomson scattering data ---
    TS = ods['thomson_scattering']
    n_channels = len(TS['channel'])
    positions = np.array([[TS[f'channel.{i}.position.r'], TS[f'channel.{i}.position.z']] for i in range(n_channels)])
    names = [TS[f'channel.{i}.name'] for i in range(n_channels)]

    # --- Plotting ---
    if flag:
        ax.plot(eq['boundary.outline.r'], eq['boundary.outline.z'], color='#1f77b4', label='Boundary')
    for i, (r, z) in enumerate(positions):
        ax.scatter(r, z, s=40, label=names[i])

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=7, loc='best')
    ax.set_title(f'Thomson Scattering Positions (slice {time_index})')

    plt.tight_layout()
    plt.show()


def thomson_scattering_radial(ods, contour_quantity: str = 'psi_norm'):
    """
    Alias following the {ids}_{dimension} naming convention.
    """
    return plot_thomson_radial_position(ods, contour_quantity=contour_quantity)

def plot_electron_profile_with_thomson(ods):
    """
    Plot Te and ne profiles (measured vs reconstructed) from core_profiles.profiles_1d.
    Time info is taken from time_measurement field if available.
    """
    if 'core_profiles.profiles_1d' not in ods:
        print("No core_profiles data found.")
        return

    n_profiles = len(ods['core_profiles.profiles_1d'])

    for idx in range(n_profiles):
        base_te = f'core_profiles.profiles_1d.{idx}.electrons.temperature_fit'
        base_ne = f'core_profiles.profiles_1d.{idx}.electrons.density_fit'

        try:
            # 데이터 로드
            rho_te = np.array(ods[f'{base_te}.rho_tor_norm'])
            te_meas = np.array(ods[f'{base_te}.measured'])
            te_recon = np.array(ods[f'{base_te}.reconstructed'])

            rho_ne = np.array(ods[f'{base_ne}.rho_tor_norm'])
            ne_meas = np.array(ods[f'{base_ne}.measured']) / 1e19  # to 10^19 m^-3
            ne_recon = np.array(ods[f'{base_ne}.reconstructed']) / 1e19

            # 시간 정보
            time_list = ods.get(f'{base_te}.time_measurement', None)
            if isinstance(time_list, list) and len(time_list) > 0:
                time_ms = float(time_list[0]) * 1e3  # sec -> ms
            else:
                time_ms = np.nan

        except Exception as e:
            print(f"[SKIP] Failed to load profile {idx}: {e}")
            continue

        # Plot
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_title(f'Electron Profile #{idx} @ {time_ms:.1f} ms')
        ax1.set_xlabel('Normalized Poloidal Flux (ρ)')
        ax1.set_ylabel('Te [eV]', color='tab:red')
        ax1.plot(rho_te, te_recon, label='Te reconstructed', color='tab:red')
        ax1.scatter(rho_te, te_meas, label='Te measured', color='tab:red', facecolors='none', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('ne [$10^{19}$ m$^{-3}$]', color='tab:blue')
        ax2.plot(rho_ne, ne_recon, label='ne reconstructed', color='tab:blue')
        ax2.scatter(rho_ne, ne_meas, label='ne measured', color='tab:blue', marker='x')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        fig.tight_layout()
        plt.grid(True)
        plt.show()

def plot_thomson_time_series(ods):
    """
    Plot Thomson scattering time evolution (n_e and T_e) with error bars.

    ## Arguments:
    - `ods`: OMAS data structure containing thomson_scattering data.
    - `font_size`: Legend and tick font size.
    - `axis_size`: Axis label font size.
    """

    font_size=6
    axis_size=7.5

    TS = ods['thomson_scattering']
    n_channels = len(TS['channel'])
    colors = plt.cm.tab10.colors

    # --- 데이터 정리 ---
    n_e_data, n_e_data_error = [], []
    t_e_data, t_e_data_error = [], []
    names = []

    for i in range(n_channels):
        n_e_data.append(TS[f'channel.{i}.n_e.data'])
        n_e_data_error.append(TS[f'channel.{i}.n_e.data_error_upper'])
        t_e_data.append(TS[f'channel.{i}.t_e.data'])
        t_e_data_error.append(TS[f'channel.{i}.t_e.data_error_upper'])
        names.append(TS[f'channel.{i}.name'])

    # --- 그림 구성 ---
    fig = plt.figure(figsize=(5,5))

    # --- n_e plot ---
    ax1 = fig.add_subplot(211)
    for i in range(n_channels):
        ax1.errorbar(
            TS['time'] * 1e3,
            n_e_data[i],
            yerr=n_e_data_error[i],
            color=colors[i % len(colors)],
            capsize=1
        )

    ax1.set_ylabel(r'$n_e$ (m$^{-3}$)', fontsize=axis_size)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.grid(True)

    # --- T_e plot ---
    ax2 = fig.add_subplot(212)
    for i in range(n_channels):
        ax2.errorbar(
            TS['time'] * 1e3,
            t_e_data[i],
            yerr=t_e_data_error[i],
            label=names[i],
            color=colors[i % len(colors)],
            capsize=1
        )

    ax2.set_xlabel('Time (ms)', fontsize=axis_size)
    ax2.set_ylabel(r'$T_e$ (eV)', fontsize=axis_size)
    ax2.grid(True)

    # --- 전체 스타일 ---
    fig.tight_layout()
    fig.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=font_size,
    )

    plt.show()


def thomson_scattering_time(ods):
    """
    Alias following the {ids}_{dimension} naming convention.
    """
    return plot_thomson_time_series(ods)


def charge_exchange_radial(ods, ion_index: int = 0):
    """
    Plot Charge Exchange Spectroscopy (CES) radial profiles:
    - Toroidal ion velocity vs R
    - Ion temperature vs R

    Data are taken from the charge_exchange IDS produced by
    vaft.machine_mapping.charge_exchange.

    Args:
        ods: OMAS data structure containing charge_exchange data.
        ion_index: Index of ion within each channel (default: 0).
    """
    if 'charge_exchange' not in ods:
        print("No charge_exchange data found in ODS.")
        return

    CE = ods['charge_exchange']

    try:
        n_channels = len(CE['channel'])
    except Exception:
        print("charge_exchange.channel is missing or malformed.")
        return

    # Assume single time frame (index 0) as in current CES processing
    time_index = 0
    if 'time' in CE:
        try:
            time_s = float(np.asarray(CE['time'])[time_index])
        except Exception:
            time_s = np.nan
    else:
        time_s = np.nan

    R, R_err = [], []
    Ti, Ti_err = [], []
    Vtor_kms, Vtor_err_kms = [], []

    for ch in range(n_channels):
        try:
            r_vals, r_errs = _extract_nominal_and_error(
                CE[f'channel.{ch}.position.r.data']
            )
            ti_vals, ti_errs = _extract_nominal_and_error(
                CE[f'channel.{ch}.ion.{ion_index}.t_i.data']
            )
            v_vals, v_errs = _extract_nominal_and_error(
                CE[f'channel.{ch}.ion.{ion_index}.velocity_tor.data']
            )
        except Exception as e:
            print(f"[SKIP] Failed to read CES channel {ch}: {e}")
            continue

        # Guard against shorter time series
        if np.isscalar(r_vals) or np.ndim(r_vals) == 0:
            r_val = float(np.asarray(r_vals).reshape(-1)[0])
            r_err = float(np.asarray(r_errs).reshape(-1)[0])
        else:
            if time_index >= len(r_vals):
                continue
            r_val = float(r_vals[time_index])
            r_err = float(r_errs[time_index])

        if np.isscalar(ti_vals) or np.ndim(ti_vals) == 0:
            ti_val = float(np.asarray(ti_vals).reshape(-1)[0])
            ti_err = float(np.asarray(ti_errs).reshape(-1)[0])
        else:
            if time_index >= len(ti_vals):
                continue
            ti_val = float(ti_vals[time_index])
            ti_err = float(ti_errs[time_index])

        if np.isscalar(v_vals) or np.ndim(v_vals) == 0:
            v_val = float(np.asarray(v_vals).reshape(-1)[0])
            v_err = float(np.asarray(v_errs).reshape(-1)[0])
        else:
            if time_index >= len(v_vals):
                continue
            v_val = float(v_vals[time_index])
            v_err = float(v_errs[time_index])

        R.append(r_val)
        R_err.append(abs(r_err))
        Ti.append(ti_val)
        Ti_err.append(abs(ti_err))
        Vtor_kms.append(v_val / 1e3)
        Vtor_err_kms.append(abs(v_err) / 1e3)

    if len(R) == 0:
        print("No valid CES channels to plot.")
        return

    R = np.asarray(R)
    R_err = np.asarray(R_err)
    Ti = np.asarray(Ti)
    Ti_err = np.asarray(Ti_err)
    Vtor_kms = np.asarray(Vtor_kms)
    Vtor_err_kms = np.asarray(Vtor_err_kms)

    # Sort by radius for nicer profiles
    order = np.argsort(R)
    R = R[order]
    R_err = R_err[order]
    Ti = Ti[order]
    Ti_err = Ti_err[order]
    Vtor_kms = Vtor_kms[order]
    Vtor_err_kms = Vtor_err_kms[order]

    # Shot number (if available)
    try:
        shot = ods['dataset_description.data_entry.pulse']
    except Exception:
        shot = None

    time_ms = time_s * 1e3 if np.isfinite(time_s) else np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)

    # Toroidal velocity
    ax1.errorbar(
        R,
        Vtor_kms,
        yerr=Vtor_err_kms,
        xerr=R_err,
        fmt='-o',
        color='r',
        linewidth=1,
        markersize=4,
        capsize=2,
    )
    ax1.set_ylabel('Toroidal Velocity (km/s)', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Ion temperature
    ax2.errorbar(
        R,
        Ti,
        yerr=Ti_err,
        xerr=R_err,
        fmt='-o',
        color='b',
        linewidth=1,
        markersize=4,
        capsize=2,
    )
    ax2.set_xlabel('R (m)', fontsize=8)
    ax2.set_ylabel('Ion Temperature (eV)', fontsize=8)
    ax2.grid(True, alpha=0.3)

    if np.isfinite(time_ms):
        title = f'CES Profile @ {time_ms:.1f} ms'
    else:
        title = 'CES Profile'
    if shot is not None:
        title += f' (shot {shot})'
    fig.suptitle(title, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_ces_profile(ods, ion_index: int = 0):
    """
    Backward-compatible alias to keep the old naming.
    """
    return charge_exchange_radial(ods, ion_index=ion_index)


def charge_exchange_rho_profiles(
    ods,
    eq=None,
    time_ms: float = 300.0,
    ion_index: int = 0,
    rho_points: int = 100,
    fitting_function_ti: str = "polynomial",
    fitting_function_vtor: str = "polynomial",
    Ti_order: int = 3,
    Vtor_order: int = 3,
    uncertainty_option: int = 1,
    save_opt: int = 0,
    file_name: str | None = None,
):
    """
    Plot CES (charge_exchange) profiles vs normalized flux coordinate (rho).

    This is analogous in spirit to `plot_thomson_profiles`, but for:
    - V_tor(ρ): charge_exchange.channel[:].ion[ion_index].velocity_tor
    - T_i(ρ):   charge_exchange.channel[:].ion[ion_index].t_i

    Behavior
    --------
    - If a `core_profiles.profiles_1d` slice exists for the requested time, reuse it
      (skip mapping/fitting) when channel counts and temperature_fit lengths match.
    - Otherwise, requires `eq` and computes mapping → fit → writes into `core_profiles`.

    Storage / reuse
    ---------------
    This function is IMAS-schema compliant:
    - The fitted profiles are stored into `core_profiles.profiles_1d`
      (grid rho_tor_norm, ion.temperature, ion.velocity_tor).
    - Measured Ti points are stored into `ion.temperature_fit.*` (schema-supported).
    - IMAS 3.41.0 does NOT expose a `velocity_tor_fit.*` block, so measured Vtor
      points are plotted directly from `charge_exchange` (still schema-valid).

    If a matching `core_profiles.profiles_1d` time slice already exists, the
    function will reuse it and skip recomputing.
    """
    if "charge_exchange" not in ods:
        print("No charge_exchange data found in ODS.")
        return

    # #region agent log
    def _ces_rho_dbg(hypothesis_id, location, message, data):
        try:
            import json, time

            payload = {
                "sessionId": "bd9a35",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            with open(
                "/Users/yun/git/vaft/.cursor/debug-bd9a35.log", "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    # #endregion

    try:
        # Local import to avoid circulars at module import time.
        from vaft.process import profile as process_profile
        try:
            from uncertainties import unumpy
        except Exception:
            unumpy = None

        def _ces_slice_usable(idx: int) -> bool:
            """CES slice: channel-count match on temperature_fit + grid vs fitted ion profiles same length."""
            b = f"core_profiles.profiles_1d.{idx}"
            try:
                n_ch = len(ods["charge_exchange.channel"])
            except Exception:
                n_ch = 0
            if n_ch <= 0:
                return False
            try:
                r = np.asarray(ods[f"{b}.ion.{ion_index}.temperature_fit.rho_tor_norm"], dtype=float).reshape(-1)
                m = np.asarray(ods[f"{b}.ion.{ion_index}.temperature_fit.measured"], dtype=float).reshape(-1)
                if not (r.size == n_ch and m.size == n_ch and r.size > 0):
                    return False
                g = np.asarray(ods[f"{b}.grid.rho_tor_norm"], dtype=float).reshape(-1)
                t = np.asarray(ods[f"{b}.ion.{ion_index}.temperature"], dtype=float).reshape(-1)
                v = np.asarray(ods[f"{b}.ion.{ion_index}.velocity_tor"], dtype=float).reshape(-1)
                return g.size > 0 and g.size == t.size == v.size
            except Exception:
                return False

        # --- Reuse existing core_profiles slice if present ---
        rho_fit = Ti_fit = Vtor_fit = None
        rho_meas = Ti_meas = Ti_err = Vtor_meas = Vtor_err = np.asarray([])
        time_s = np.nan

        target_s = float(time_ms) / 1e3
        cp_index = None
        n_profiles_1d = 0
        if "core_profiles.profiles_1d" in ods:
            try:
                n_profiles_1d = len(ods["core_profiles.profiles_1d"])
                for i in range(n_profiles_1d):
                    t_existing = float(ods[f"core_profiles.profiles_1d.{i}.time"])
                    if np.isclose(t_existing, target_s) and _ces_slice_usable(i):
                        cp_index = i
                        break
            except Exception:
                cp_index = None

        _ces_rho_dbg(
            "H1_reuse_search",
            "charge_exchange_rho_profiles:reuse_search",
            "reuse_search_done",
            {
                "cp_index": cp_index,
                "n_profiles_1d": n_profiles_1d,
                "target_s": target_s,
                "eq_is_none": eq is None,
            },
        )

        if cp_index is not None:
            base = f"core_profiles.profiles_1d.{cp_index}"
            time_s = float(ods[f"{base}.time"])
            rho_fit = np.asarray(ods[f"{base}.grid.rho_tor_norm"], dtype=float)
            Ti_fit = np.asarray(ods[f"{base}.ion.{ion_index}.temperature"], dtype=float)
            Vtor_fit = np.asarray(ods[f"{base}.ion.{ion_index}.velocity_tor"], dtype=float)

            try:
                rho_meas = np.asarray(ods[f"{base}.ion.{ion_index}.temperature_fit.rho_tor_norm"], dtype=float)
                Ti_meas = np.asarray(ods[f"{base}.ion.{ion_index}.temperature_fit.measured"], dtype=float)
                Ti_err = np.asarray(ods[f"{base}.ion.{ion_index}.temperature_fit.measured_error_upper"], dtype=float)
            except Exception:
                rho_meas = np.asarray([])
                Ti_meas = np.asarray([])
                Ti_err = np.asarray([])

            try:
                times_ce = np.asarray(ods["charge_exchange.time"], dtype=float)
                tidx = int(np.argmin(np.abs(times_ce - float(time_s)))) if times_ce.size else 0
                n_ch = len(ods["charge_exchange.channel"])
                Vm, Ve = [], []
                for ch in range(n_ch):
                    v_u = ods[f"charge_exchange.channel.{ch}.ion.{ion_index}.velocity_tor.data"]
                    try:
                        if unumpy is None:
                            raise RuntimeError("uncertainties unavailable")
                        v_vals = unumpy.nominal_values(v_u)
                        v_stds = unumpy.std_devs(v_u)
                    except Exception:
                        v_vals = np.asarray(v_u, dtype=float)
                        v_stds = np.zeros_like(np.asarray(v_u, dtype=float), dtype=float)
                    v_vals = np.asarray(v_vals, dtype=float).reshape(-1)
                    v_stds = np.asarray(v_stds, dtype=float).reshape(-1)
                    v_val = float(v_vals[tidx] if v_vals.size > tidx else v_vals[0])
                    v_er = float(v_stds[tidx] if v_stds.size > tidx else v_stds[0] if v_stds.size else 0.0)
                    Vm.append(v_val)
                    Ve.append(abs(v_er))
                Vtor_meas = np.asarray(Vm, dtype=float)
                Vtor_err = np.asarray(Ve, dtype=float)
            except Exception:
                Vtor_meas = np.asarray([])
                Vtor_err = np.asarray([])

            _ces_rho_dbg(
                "H1_reuse",
                "charge_exchange_rho_profiles:reuse",
                "branch_reuse",
                {"cp_index": int(cp_index)},
            )
        else:
            _ces_rho_dbg(
                "H2_eq_required",
                "charge_exchange_rho_profiles:compute",
                "branch_compute",
                {"eq_is_none": eq is None},
            )
            if eq is None:
                raise ValueError(
                    "eq is required when CES rho-fit cache is missing. "
                    "Call charge_exchange_rho_profiles(ods, eq=..., time_ms=...)."
                )

            times = np.asarray(ods["charge_exchange.time"], dtype=float)
            target_s = float(time_ms) / 1e3
            idx_candidates = np.where(np.isclose(times, target_s))[0]
            if len(idx_candidates) == 0:
                time_index = int(np.argmin(np.abs(times - target_s)))
            else:
                time_index = int(idx_candidates[0])
            time_s = float(times[time_index]) if times.size else np.nan

            mapped_rho = process_profile.equilibrium_mapping_charge_exchange(ods, eq)

            (
                Vtor_func,
                Ti_func,
                _coeffs_vtor,
                _coeffs_ti,
                Vtor_fit,
                Ti_fit,
            ) = process_profile.profile_fitting_charge_exchange(
                ods,
                time_ms=float(time_ms),
                mapped_rho_position=mapped_rho,
                Ti_order=int(Ti_order),
                Vtor_order=int(Vtor_order),
                uncertainty_option=int(uncertainty_option),
                rho_points=int(rho_points),
                fitting_function_ti=str(fitting_function_ti),
                fitting_function_vtor=str(fitting_function_vtor),
                ion_index=int(ion_index),
            )

            rho_fit = np.linspace(0.0, 1.0, int(rho_points))

            n_channels = len(ods["charge_exchange.channel"])
            rho_meas = np.asarray(mapped_rho, dtype=float).reshape(-1)[:n_channels]

            Ti_meas, Ti_err = [], []
            Vtor_meas, Vtor_err = [], []

            for ch in range(n_channels):
                ion = ods[f"charge_exchange.channel.{ch}.ion.{ion_index}"]
                ti_u = ion["t_i.data"]
                v_u = ion["velocity_tor.data"]

                try:
                    if unumpy is None:
                        raise RuntimeError("uncertainties unavailable")
                    ti_vals = unumpy.nominal_values(ti_u)
                    ti_stds = unumpy.std_devs(ti_u)
                except Exception:
                    ti_vals = np.asarray(ti_u, dtype=float)
                    ti_stds = np.zeros_like(ti_vals, dtype=float)

                try:
                    if unumpy is None:
                        raise RuntimeError("uncertainties unavailable")
                    v_vals = unumpy.nominal_values(v_u)
                    v_stds = unumpy.std_devs(v_u)
                except Exception:
                    v_vals = np.asarray(v_u, dtype=float)
                    v_stds = np.zeros_like(v_vals, dtype=float)

                ti_vals = np.asarray(ti_vals, dtype=float)
                ti_stds = np.asarray(ti_stds, dtype=float)
                v_vals = np.asarray(v_vals, dtype=float)
                v_stds = np.asarray(v_stds, dtype=float)

                ti_val = float(ti_vals.reshape(-1)[time_index] if ti_vals.size > 1 else ti_vals.reshape(-1)[0])
                ti_er = float(ti_stds.reshape(-1)[time_index] if ti_stds.size > 1 else ti_stds.reshape(-1)[0])
                v_val = float(v_vals.reshape(-1)[time_index] if v_vals.size > 1 else v_vals.reshape(-1)[0])
                v_er = float(v_stds.reshape(-1)[time_index] if v_stds.size > 1 else v_stds.reshape(-1)[0])

                Ti_meas.append(ti_val)
                Ti_err.append(abs(ti_er))
                Vtor_meas.append(v_val)
                Vtor_err.append(abs(v_er))

            Ti_meas = np.asarray(Ti_meas, dtype=float)
            Ti_err = np.asarray(Ti_err, dtype=float)
            Vtor_meas = np.asarray(Vtor_meas, dtype=float)
            Vtor_err = np.asarray(Vtor_err, dtype=float)

            if "core_profiles.profiles_1d" in ods:
                to_remove = None
                for i in range(len(ods["core_profiles.profiles_1d"])):
                    try:
                        t_existing = float(ods[f"core_profiles.profiles_1d.{i}.time"])
                        if np.isclose(t_existing, time_s):
                            to_remove = i
                            break
                    except Exception:
                        continue
                if to_remove is not None:
                    ods.pop(f"core_profiles.profiles_1d.{to_remove}")
                next_idx = len(ods["core_profiles.profiles_1d"])
            else:
                next_idx = 0

            cp_base = f"core_profiles.profiles_1d.{next_idx}"
            ods[f"{cp_base}.time"] = float(time_s)
            ods[f"{cp_base}.grid.rho_tor_norm"] = np.asarray(rho_fit, dtype=float)

            try:
                ion_label = ods[f"charge_exchange.channel.0.ion.{ion_index}.label"]
            except Exception:
                ion_label = f"ion{int(ion_index)}"
            ods[f"{cp_base}.ion.{ion_index}.label"] = ion_label

            ods[f"{cp_base}.ion.{ion_index}.temperature"] = np.asarray(Ti_fit, dtype=float)
            ods[f"{cp_base}.ion.{ion_index}.velocity_tor"] = np.asarray(Vtor_fit, dtype=float)

            ods[f"{cp_base}.ion.{ion_index}.temperature_fit.rho_tor_norm"] = np.asarray(rho_meas, dtype=float)
            ods[f"{cp_base}.ion.{ion_index}.temperature_fit.measured"] = np.asarray(Ti_meas, dtype=float)
            ods[f"{cp_base}.ion.{ion_index}.temperature_fit.measured_error_upper"] = np.asarray(Ti_err, dtype=float)

            try:
                ods[f"{cp_base}.ids_properties.comment"] = (
                    "vaft: charge_exchange rho fit (Ti measured in temperature_fit; Vtor from charge_exchange)"
                )
            except Exception:
                pass

            _ces_rho_dbg(
                "H4_store",
                "charge_exchange_rho_profiles:store",
                "core_profiles_store_ok",
                {"next_idx": int(next_idx), "time_s": float(time_s), "ion_label": str(ion_label)},
            )

        if rho_fit is None or Ti_fit is None or Vtor_fit is None:
            raise RuntimeError(
                "charge_exchange_rho_profiles: missing rho_fit/Ti_fit/Vtor_fit after compute/reuse."
            )
        rho_fit = np.asarray(rho_fit, dtype=float).reshape(-1)
        Ti_fit = np.asarray(Ti_fit, dtype=float).reshape(-1)
        Vtor_fit = np.asarray(Vtor_fit, dtype=float).reshape(-1)
        if not (rho_fit.size == Ti_fit.size == Vtor_fit.size):
            raise ValueError(
                f"charge_exchange_rho_profiles: shape mismatch rho_fit={rho_fit.size} "
                f"Ti_fit={Ti_fit.size} Vtor_fit={Vtor_fit.size}"
            )

        _ces_rho_dbg(
            "H3_shapes",
            "charge_exchange_rho_profiles:before_plot",
            "before_plot",
            {
                "rho_fit_n": int(rho_fit.size),
                "Ti_fit_n": int(Ti_fit.size),
                "Vtor_fit_n": int(Vtor_fit.size),
                "rho_meas_n": int(np.asarray(rho_meas).size),
            },
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)

        if rho_meas.size:
            order = np.argsort(rho_meas)
            rho_m = rho_meas[order]
            Ti_m = Ti_meas[order] if Ti_meas.size else Ti_meas
            Ti_e = Ti_err[order] if Ti_err.size else Ti_err
            V_m = Vtor_meas[order] if Vtor_meas.size else Vtor_meas
            V_e = Vtor_err[order] if Vtor_err.size else Vtor_err
        else:
            rho_m, Ti_m, Ti_e, V_m, V_e = rho_meas, Ti_meas, Ti_err, Vtor_meas, Vtor_err

        if rho_m.size and V_m.size:
            ax1.errorbar(
                rho_m,
                V_m / 1e3,
                yerr=(V_e / 1e3) if V_e.size else None,
                fmt="o",
                markersize=4,
                capsize=2,
                color="tab:red",
                label="measured",
            )
        ax1.plot(rho_fit, Vtor_fit / 1e3, "-", color="tab:red", linewidth=1, label="fitted")
        ax1.set_ylabel("V_tor (km/s)", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=6)

        if rho_m.size and Ti_m.size:
            ax2.errorbar(
                rho_m,
                Ti_m,
                yerr=Ti_e if Ti_e.size else None,
                fmt="o",
                markersize=4,
                capsize=2,
                color="tab:blue",
                label="measured",
            )
        ax2.plot(rho_fit, Ti_fit, "-", color="tab:blue", linewidth=1, label="fitted")
        ax2.set_xlabel(r"$\rho_{tor,norm}$", fontsize=8)
        ax2.set_ylabel("T_i (eV)", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=6)

        try:
            shot = ods["dataset_description.data_entry.pulse"]
        except Exception:
            shot = None

        title = (
            f"CES profiles vs rho @ {float(time_s)*1e3:.1f} ms"
            if np.isfinite(time_s)
            else "CES profiles vs rho"
        )
        if shot is not None:
            title += f" (shot {shot})"
        fig.suptitle(title, fontsize=9)

        plt.tight_layout()
        if int(save_opt) == 1:
            plt.savefig(file_name if file_name else "charge_exchange_rho_profiles.png", dpi=150)
        else:
            plt.show()

        _ces_rho_dbg(
            "H5_plot_ok",
            "charge_exchange_rho_profiles:after_plot",
            "plot_complete",
            {},
        )

    except Exception as e:
        # #region agent log
        try:
            import traceback

            _ces_rho_dbg(
                "H_exc",
                "charge_exchange_rho_profiles:except",
                repr(e),
                {"tb": traceback.format_exc()},
            )
        except Exception:
            pass
        # #endregion
        raise


def charge_exchange_rho_profile(ods, **kwargs):
    """Alias for `charge_exchange_rho_profiles` (singular name)."""
    return charge_exchange_rho_profiles(ods, **kwargs)


def charge_exchange_time(ods, ion_index: int = 0):
    """
    Plot Charge Exchange Spectroscopy (CES) time evolution:
    - Toroidal ion velocity vs time
    - Ion temperature vs time

    Follows a similar style to plot_thomson_time_series.
    """
    if 'charge_exchange' not in ods:
        print("No charge_exchange data found in ODS.")
        return

    CE = ods['charge_exchange']

    try:
        times = np.asarray(CE['time'], float) * 1e3  # s -> ms
    except Exception as e:
        print(f"Failed to read charge_exchange.time: {e}")
        return

    try:
        n_channels = len(CE['channel'])
    except Exception:
        print("charge_exchange.channel is missing or malformed.")
        return

    font_size = 6
    axis_size = 7.5
    colors = plt.cm.tab10.colors

    v_data, v_err = [], []
    t_i_data, t_i_err = [], []
    names = []

    for i in range(n_channels):
        try:
            v_vals, v_errs = _extract_nominal_and_error(
                CE[f'channel.{i}.ion.{ion_index}.velocity_tor.data']
            )
            t_vals, t_errs = _extract_nominal_and_error(
                CE[f'channel.{i}.ion.{ion_index}.t_i.data']
            )
        except Exception as e:
            print(f"[SKIP] Failed to read CES channel {i}: {e}")
            continue

        v_data.append(np.asarray(v_vals, float) / 1e3)      # m/s -> km/s
        v_err.append(np.asarray(np.abs(v_errs), float) / 1e3)
        t_i_data.append(np.asarray(t_vals, float))
        t_i_err.append(np.asarray(np.abs(t_errs), float))

        try:
            label = CE[f'channel.{i}.identifier']
        except Exception:
            label = f'CES_{i}'
        names.append(label)

    if len(v_data) == 0:
        print("No valid CES channels to plot in time.")
        return

    fig = plt.figure(figsize=(5, 5))

    # Toroidal velocity
    ax1 = fig.add_subplot(211)
    for i, (v, verr) in enumerate(zip(v_data, v_err)):
        ax1.errorbar(
            times,
            v,
            yerr=verr,
            color=colors[i % len(colors)],
            capsize=1,
        )

    ax1.set_ylabel('V_tor (km/s)', fontsize=axis_size)
    ax1.grid(True)

    # Ion temperature
    ax2 = fig.add_subplot(212)
    for i, (ti, tier) in enumerate(zip(t_i_data, t_i_err)):
        ax2.errorbar(
            times,
            ti,
            yerr=tier,
            label=names[i],
            color=colors[i % len(colors)],
            capsize=1,
        )

    ax2.set_xlabel('Time (ms)', fontsize=axis_size)
    ax2.set_ylabel('T_i (eV)', fontsize=axis_size)
    ax2.grid(True)

    fig.tight_layout()
    fig.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=font_size,
    )

    plt.show()


def plot_pressure_profile_with_geqdsk(shot, time_ms, OMFITgeq, n_e_function, T_e_function, geqdsk, save_opt = 1):
    """
    Plot the core electron temperature and density profiles along with the pressure profile from GEQDSK.

    This function plots the fitted electron temperature and density profiles as functions
    of normalized poloidal flux (rho), and overlays the pressure profile from GEQDSK.

    Parameters:
        n_e_function (callable): Function to compute fitted electron density at any rho.
        T_e_function (callable): Function to compute fitted electron temperature at any rho.
        geqdsk (dict): The GEQDSK equilibrium data containing pressure profiles.
    """
    # Extract pressure profile from GEQDSK
    psi = np.zeros(len(geqdsk['fluxSurfaces']['flux']))
    psi_N = np.zeros(len(geqdsk['fluxSurfaces']['flux']))
    pressure = np.zeros(len(geqdsk['fluxSurfaces']['flux']))

    for i in range(len(OMFITgeq['fluxSurfaces']['flux'])):
        psi[i] = OMFITgeq['fluxSurfaces']['flux'][i]['psi']
        pressure[i] = OMFITgeq['fluxSurfaces']['flux'][i]['p']
    psi_N = (psi - psi[0]) / (psi[-1] - psi[0])

    # Evaluate the fitted profiles at the same rho points
    T_e_rho = T_e_function(psi_N)
    n_e_rho = n_e_function(psi_N)

    # Calculate electron pressure from fitted profiles
    p_e_rho = n_e_rho * T_e_rho * 1.602e-19  # Convert to Pa

    # Plot the core profiles
    plt.figure(figsize=(10, 6))
    plt.plot(psi_N, p_e_rho, label='Fitted $p_e$ from TS')
    plt.plot(psi_N, pressure, label='EFIT $p$')
    plt.xlabel(r'$\psi_N$')
    plt.ylabel(r'$p$ (Pa)')
    plt.title('Electron Pressure Profile vs EFIT Total Pressure Profile, shot = {}, time = {} ms'.format(shot, time_ms))
    plt.legend()

    if save_opt == 1:
        plt.savefig(f'pressure_profile_comparison_{shot}_{time_ms}.png')



def plot_thomson_profiles(ods, save_opt=0, file_name=None):
    """
    Plot fitted Thomson scattering electron temperature (Te)
    and electron density (Ne) profiles using core_profiles data.

    ## Arguments:
    - `ods`: OMAS data structure (must contain core_profiles and thomson_scattering)
    """

    # --- Configuration ---
    colors = plt.cm.tab10.colors
    title_fontsize = 6
    label_fontsize = 7
    legend_fontsize = 5

    # --- Prepare figure ---
    fig, axs = plt.subplots(2, 1, figsize=(5,4))  # axs[0] = Te, axs[1] = Ne

    n_profiles = len(ods['core_profiles']['profiles_1d'])
    num_channels = len(ods['thomson_scattering.channel'])

    print(f"[INFO] Found {n_profiles} fitted core_profiles")

    # --- Iterate over each fitted profile (time slice) ---
    for i in range(n_profiles):
        prof = ods['core_profiles']['profiles_1d'][i]
        time_s = prof['time']
        time_ms = time_s * 1e3

        # --- Fitted profiles ---
        rho_fit = prof['grid']['rho_tor_norm']
        T_e_fit = prof['electrons']['temperature']
        n_e_fit = np.asarray(prof['electrons']['density_thermal'], dtype=float) / 1e19

        # --- Thomson diagnostic reference (measured points) ---
        rho_meas = prof['electrons']['temperature_fit']['rho_tor_norm']
        T_e_meas = prof['electrons']['temperature_fit']['measured']
        n_e_meas = np.asarray(prof['electrons']['density_fit']['measured'], dtype=float) / 1e19

        # --- Error extraction from ODS (if exists) ---
        try:
            t_e_err = [ods[f'thomson_scattering.channel.{j}.t_e.data_error_upper'][i] for j in range(num_channels)]
            n_e_err = (
                np.asarray(
                    [ods[f'thomson_scattering.channel.{j}.n_e.data_error_upper'][i] for j in range(num_channels)],
                    dtype=float,
                )
                / 1e19
            )
        except Exception:
            t_e_err = np.zeros_like(T_e_meas)
            n_e_err = np.zeros_like(n_e_meas)

        # --- Color & Label ---
        color = colors[i % len(colors)]
        label = f"{time_ms:.1f} ms"

        # ---- Plot Te ----
        axs[0].errorbar(rho_meas, T_e_meas, yerr=t_e_err, fmt='.', color=color, label=label)
        axs[0].plot(rho_fit, T_e_fit, '-', color=color)

        # ---- Plot Ne ----
        axs[1].errorbar(rho_meas, n_e_meas, yerr=n_e_err, fmt='.', color=color, label=label)
        axs[1].plot(rho_fit, n_e_fit, '-', color=color)

    # --- Decorate plots ---
    axs[0].set_title('Electron Temperature Profile', fontsize=title_fontsize)
    axs[0].set_ylabel(r'$T_e$ (eV)', fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=6)
    axs[0].grid(True)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].set_xticklabels([])

    axs[1].set_title('Electron Density Profile', fontsize=title_fontsize)
    axs[1].set_xlabel(r'$\rho_{tor,norm}$', fontsize=label_fontsize)
    axs[1].set_ylabel(r'$N_e$ ($10^{19}$ m$^{-3}$)', fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=6)
    axs[1].grid(True)
    axs[1].legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if save_opt == 1:
        plt.savefig(file_name if file_name else 'thomson_profiles.png', dpi=150)
    else:
        plt.show()


def thomson_scattering_radial_profiles(ods, save_opt: int = 0, file_name: str | None = None):
    """
    Alias for plot_thomson_profiles following the {ids}_{dimension} naming convention.
    """
    return plot_thomson_profiles(ods, save_opt=save_opt, file_name=file_name)

def plot_TeNe_from_eq(
    ods,
    save_opt=0,
    file_name=None,
    only_synthetic=True,
    synthetic_tag=None,
    plot_pressure=False,
    eq_time_index=0,
):
    """
    Plot electron temperature (Te) and electron density (Ne) profiles from core_profiles,
    intended for profiles generated from equilibrium pressure (synthetic case).

    - Does NOT use thomson_scattering.
    - Plots core_profiles.profiles_1d[*].electrons.temperature and density_thermal.

    Parameters
    ----------
    ods : OMAS ODS
        Must contain core_profiles. If plot_pressure=True, must contain equilibrium pressure.
    save_opt : int
        If 1, save figure to file_name. Otherwise just show.
    file_name : str or None
        Output filename when save_opt=1. Default: "TeNe_from_eq.png"
    only_synthetic : bool
        If True, plot only profiles that look "synthetic" (no measured blocks available).
        Heuristic: skip profiles that contain electrons.temperature_fit.measured (TS-like).
        Set False to plot all core_profiles entries.
    synthetic_tag : str or None
        Optional: if you store any tag string in profile (not in your current writer),
        you could filter by it. Currently unused unless you add such metadata.
    plot_pressure : bool
        If True, also plot equilibrium pressure used (from equilibrium.time_slice.eq_time_index).
    eq_time_index : int
        Equilibrium time_slice index to use if plot_pressure=True.
    """

    # --- Configuration ---
    colors = plt.cm.tab10.colors
    title_fontsize = 6
    label_fontsize = 7
    legend_fontsize = 5

    # --- Prepare figure ---
    nrows = 3 if plot_pressure else 2
    fig, axs = plt.subplots(nrows, 1, figsize=(5, 4), sharex=True)
    if nrows == 2:
        axT, axN = axs
        axP = None
    else:
        axT, axN, axP = axs

    # --- core_profiles existence check ---
    if 'core_profiles' not in ods or 'profiles_1d' not in ods['core_profiles']:
        raise KeyError("ODS does not contain core_profiles.profiles_1d")

    n_profiles = len(ods['core_profiles']['profiles_1d'])
    print(f"[INFO] Found {n_profiles} core_profiles entries")

    plotted = 0

    # --- Iterate over each profile (time slice) ---
    for i in range(n_profiles):
        prof = ods['core_profiles']['profiles_1d'][i]

        # Heuristic filter: skip TS-like profiles if only_synthetic
        if only_synthetic:
            try:
                _ = prof['electrons']['temperature_fit']['measured']
                # if exists, it's likely TS-derived; skip
                continue
            except Exception:
                pass

        time_s = prof.get('time', np.nan)
        time_ms = time_s * 1e3 if np.isfinite(time_s) else np.nan

        # --- Fitted / stored profiles ---
        try:
            rho_fit = np.asarray(prof['grid']['rho_tor_norm'], dtype=float)
            T_e = np.asarray(prof['electrons']['temperature'], dtype=float)
            n_e = np.asarray(prof['electrons']['density_thermal'], dtype=float)
        except Exception as e:
            print(f"[WARN] Skip profile index {i} due to missing fields: {e}")
            continue

        color = colors[plotted % len(colors)]
        label = f"{time_ms:.1f} ms" if np.isfinite(time_ms) else f"idx {i}"

        # ---- Plot Te ----
        axT.plot(rho_fit, T_e, '-', color=color, label=label)

        # ---- Plot Ne ----
        axN.plot(rho_fit, n_e, '-', color=color, label=label)

        plotted += 1

    if plotted == 0:
        print("[WARN] No profiles plotted. (Check only_synthetic flag and core_profiles contents.)")

    # --- Decorate Te plot ---
    axT.set_title('Electron Temperature Profile (from core_profiles)', fontsize=title_fontsize)
    axT.set_ylabel(r'$T_e$ (eV)', fontsize=label_fontsize)
    axT.tick_params(axis='both', which='major', labelsize=6)
    axT.grid(True)
    axT.legend(fontsize=legend_fontsize)

    # --- Decorate Ne plot ---
    axN.set_title('Electron Density Profile (from core_profiles)', fontsize=title_fontsize)
    axN.set_ylabel(r'$N_e$ (m$^{-3}$)', fontsize=label_fontsize)
    axN.tick_params(axis='both', which='major', labelsize=6)
    axN.grid(True)
    axN.legend(fontsize=legend_fontsize)

    # --- Optional: plot equilibrium pressure for reference ---
    if plot_pressure:
        try:
            rho_eq = np.asarray(ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.rho_tor_norm'], dtype=float)
            p_eq = np.asarray(ods[f'equilibrium.time_slice.{eq_time_index}.profiles_1d.pressure'], dtype=float)

            order = np.argsort(rho_eq)
            rho_eq = rho_eq[order]
            p_eq = p_eq[order]

            axP.plot(rho_eq, p_eq, '-', label=f"eq pressure (slice {eq_time_index})")
            axP.set_title('Equilibrium Pressure Profile', fontsize=title_fontsize)
            axP.set_xlabel(r'$\rho_{tor,norm}$', fontsize=label_fontsize)
            axP.set_ylabel(r'$p$ (arb)', fontsize=label_fontsize)
            axP.tick_params(axis='both', which='major', labelsize=6)
            axP.grid(True)
            axP.legend(fontsize=legend_fontsize)
        except Exception as e:
            print(f"[WARN] plot_pressure=True but failed to read equilibrium pressure: {e}")
            axN.set_xlabel(r'$\rho_{tor,norm}$', fontsize=label_fontsize)
    else:
        axN.set_xlabel(r'$\rho_{tor,norm}$', fontsize=label_fontsize)

    plt.tight_layout()

    # --- Save option ---
    if save_opt == 1:
        if file_name is None:
            file_name = "TeNe_from_eq.png"
        plt.savefig(file_name, dpi=200)
        print(f"[SAVED] {file_name}")

    plt.show()


def plot_electron_psi_profile(ods, time_slice=None, figsize=(10, 6)):
    """
    Plot electron profiles (n_e, T_e) in psi_norm coordinate system.
    
    Args:
        ods: OMAS data structure
        time_slice: Time slice index (None = use first available)
        figsize: Figure size tuple
    """
    from vaft.omas.process_wrapper import compute_core_profile_psi
    
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Compute n_e profile
    try:
        psi_norm_n, n_e, time_n = compute_core_profile_psi(ods, option='n_e', time_slice=time_slice)
        axs[0].plot(psi_norm_n, n_e, 'b-', linewidth=2, label=r'$n_e$')
        axs[0].set_ylabel(r'$n_e$ (m$^{-3}$)', fontsize=12)
        axs[0].set_title(f'Electron Density Profile (t={time_n:.3f}s)', fontsize=14)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=10)
    except Exception as e:
        print(f"Warning: Could not plot n_e profile: {e}")
    
    # Compute T_e profile
    try:
        psi_norm_t, T_e, time_t = compute_core_profile_psi(ods, option='t_e', time_slice=time_slice)
        axs[1].plot(psi_norm_t, T_e, 'r-', linewidth=2, label=r'$T_e$')
        axs[1].set_ylabel(r'$T_e$ (eV)', fontsize=12)
        axs[1].set_xlabel(r'$\psi_N$', fontsize=12)
        axs[1].set_title(f'Electron Temperature Profile (t={time_t:.3f}s)', fontsize=14)
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=10)
    except Exception as e:
        print(f"Warning: Could not plot T_e profile: {e}")
    
    plt.tight_layout()
    plt.show()


def plot_electron_2d_profile(ods, time_slice=None, figsize=(20, 8)):
    """
    Plot electron profiles (n_e and T_e) in 2D (R,Z) coordinate system.
    
    Args:
        ods: OMAS data structure
        time_slice: Time slice index (None = use first available)
        figsize: Figure size tuple
    """
    from vaft.omas.process_wrapper import compute_core_profile_2d
    from matplotlib.colors import LogNorm
    
    try:
        # Compute both n_e and T_e profiles
        n_e_RZ, R_grid, Z_grid, psiN_RZ, time_val = compute_core_profile_2d(
            ods, option='n_e', time_slice=time_slice
        )
        T_e_RZ, _, _, _, _ = compute_core_profile_2d(
            ods, option='t_e', time_slice=time_slice
        )
        
        # Create 1x2 subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Create meshgrid for plotting (use 'ij' indexing to match shape convention)
        R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid, indexing='ij')
        
        # Plot n_e (left subplot)
        n_e_label = r'$n_e$ (m$^{-3}$)'
        n_e_levels = np.logspace(np.log10(n_e_RZ[n_e_RZ > 0].min()), 
                                 np.log10(n_e_RZ.max()), 20)
        cs1 = ax1.contourf(R_mesh, Z_mesh, n_e_RZ, levels=n_e_levels, cmap='plasma', extend='both')
        ax1.contour(R_mesh, Z_mesh, n_e_RZ, levels=n_e_levels[::3], colors='k', alpha=0.3, linewidths=0.5)
        cbar1 = plt.colorbar(cs1, ax=ax1)
        cbar1.set_label(n_e_label, fontsize=12)
        ax1.set_xlabel('R (m)', fontsize=12)
        ax1.set_ylabel('Z (m)', fontsize=12)
        ax1.set_title(f'Electron Density (t={time_val:.3f}s)', fontsize=14)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Plot T_e (right subplot)
        T_e_label = r'$T_e$ (eV)'
        T_e_levels = np.linspace(T_e_RZ.min(), T_e_RZ.max(), 20)
        cs2 = ax2.contourf(R_mesh, Z_mesh, T_e_RZ, levels=T_e_levels, cmap='hot', extend='both')
        ax2.contour(R_mesh, Z_mesh, T_e_RZ, levels=T_e_levels[::3], colors='k', alpha=0.3, linewidths=0.5)
        cbar2 = plt.colorbar(cs2, ax=ax2)
        cbar2.set_label(T_e_label, fontsize=12)
        ax2.set_xlabel('R (m)', fontsize=12)
        ax2.set_ylabel('Z (m)', fontsize=12)
        ax2.set_title(f'Electron Temperature (t={time_val:.3f}s)', fontsize=14)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Add flux surfaces to both subplots
        if psiN_RZ is not None:
            psi_levels = np.linspace(psiN_RZ.min(), psiN_RZ.max(), 10)
            ax1.contour(R_mesh, Z_mesh, psiN_RZ, levels=psi_levels, colors='w', alpha=0.5, linewidths=1)
            ax2.contour(R_mesh, Z_mesh, psiN_RZ, levels=psi_levels, colors='w', alpha=0.5, linewidths=1)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting 2D profile: {e}")
        raise


def plot_electron_time_volume_averaged(ods, figsize=(12, 6)):
    """
    Plot time evolution of volume-averaged electron quantities.
    
    Args:
        ods: OMAS data structure
        figsize: Figure size tuple
    """
    from vaft.omas.update import update_core_profiles_global_quantities_volume_average
    
    # Ensure volume averages are computed
    try:
        update_core_profiles_global_quantities_volume_average(ods)
    except Exception as e:
        print(f"Warning: Could not update volume averages: {e}")
    
    # Extract time and volume-averaged quantities using OMAS wildcard access
    try:
        times = np.asarray(ods['core_profiles.profiles_1d[:].time'], float)
    except (KeyError, ValueError):
        print("Warning: Could not extract time data")
        return
    
    try:
        n_e_vol = np.asarray(ods['core_profiles.global_quantities.n_e_volume_average'], float)
        T_e_vol = np.asarray(ods['core_profiles.global_quantities.t_e_volume_average'], float)
    except (KeyError, ValueError):
        print("Warning: Could not extract volume-averaged quantities")
        return
    
    # Check if arrays have same length
    if len(times) != len(n_e_vol) or len(times) != len(T_e_vol):
        print(f"Warning: Data length mismatch: times={len(times)}, n_e={len(n_e_vol)}, T_e={len(T_e_vol)}")
        min_len = min(len(times), len(n_e_vol), len(T_e_vol))
        times = times[:min_len]
        n_e_vol = n_e_vol[:min_len]
        T_e_vol = T_e_vol[:min_len]
    
    # Plot
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot n_e volume average
    valid_n = ~np.isnan(n_e_vol)
    if np.any(valid_n):
        axs[0].plot(times[valid_n], n_e_vol[valid_n], 'b-o', linewidth=2, markersize=4, label=r'$\langle n_e \rangle_V$')
        axs[0].set_ylabel(r'$\langle n_e \rangle_V$ (m$^{-3}$)', fontsize=12)
        axs[0].set_title('Volume-Averaged Electron Density', fontsize=14)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=10)
    else:
        axs[0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axs[0].transAxes)
    
    # Plot T_e volume average
    valid_t = ~np.isnan(T_e_vol)
    if np.any(valid_t):
        axs[1].plot(times[valid_t], T_e_vol[valid_t], 'r-o', linewidth=2, markersize=4, label=r'$\langle T_e \rangle_V$')
        axs[1].set_ylabel(r'$\langle T_e \rangle_V$ (eV)', fontsize=12)
        axs[1].set_xlabel('Time (s)', fontsize=12)
        axs[1].set_title('Volume-Averaged Electron Temperature', fontsize=14)
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=10)
    else:
        axs[1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axs[1].transAxes)
    
    plt.tight_layout()
    plt.show()


def plot_equilibrium_and_core_profiles_pressure(ods, figsize=(12, 6)):
    """
    Plot volume-averaged pressure from equilibrium and core_profiles on a single plot.
    
    - Equilibrium: volume-averaged pressure computed using compute_volume_averaged_pressure
    - Core profiles: electron pressure calculated as P_e = n_e * T_e * 1.6e-19
    
    Args:
        ods: OMAS data structure
        figsize: Figure size tuple
    """
    from vaft.omas.update import update_core_profiles_global_quantities_volume_average, update_equilibrium_profiles_1d_normalized_psi
    
    # Ensure core_profiles volume averages are computed
    try:
        update_core_profiles_global_quantities_volume_average(ods)
    except Exception as e:
        print(f"Warning: Could not update core_profiles volume averages: {e}")
    
    # Extract equilibrium time and compute volume-averaged pressure
    from vaft.omas.process_wrapper import compute_volume_averaged_pressure
    from vaft.formula.constants import MU0
    update_equilibrium_profiles_1d_normalized_psi(ods)
    eq_times = []
    eq_pressure = []
    eq_pressure_beta = []  # Pressure from beta_tor calculation
    
    try:
        # Get equilibrium time
        eq_times = np.asarray(ods['equilibrium.time_slice[:].time'], float)
        
        # Compute volume-averaged pressure using compute_volume_averaged_pressure
        eq_pressure = compute_volume_averaged_pressure(ods, time_slice=None)
        
        # Compute volume-averaged pressure from beta_tor: p = beta_tor / 2 / mu0 * B_0^2
        beta_tor = np.asarray(ods['equilibrium.time_slice[:].global_quantities.beta_tor'], float)
        
        # Handle B_0: it can be a scalar or an array
        b0_data = ods['equilibrium.vacuum_toroidal_field.b0']
        if isinstance(b0_data, (list, np.ndarray)):
            b0_arr = np.asarray(b0_data, float)
            if b0_arr.size == 1:
                B_0 = float(b0_arr.item())
            else:
                # If array, use first value or mean (assuming constant or using first)
                B_0 = float(b0_arr[0] if len(b0_arr) > 0 else np.mean(b0_arr))
        else:
            B_0 = float(b0_data)
        
        # p = beta_tor * B_0^2 / (2 * mu0)
        eq_pressure_beta = beta_tor * (B_0 ** 2) / (2.0 * MU0)
        
        # Ensure arrays have same length
        if len(eq_times) != len(beta_tor):
            print(f"Warning: Data length mismatch: times={len(eq_times)}, beta_tor={len(beta_tor)}")
            min_len = min(len(eq_times), len(beta_tor))
            eq_times = eq_times[:min_len]
            eq_pressure_beta = eq_pressure_beta[:min_len]
        
        # Ensure arrays have same length
        if len(eq_times) != len(eq_pressure):
            print(f"Warning: Data length mismatch: times={len(eq_times)}, pressure={len(eq_pressure)}")
            min_len = min(len(eq_times), len(eq_pressure))
            eq_times = eq_times[:min_len]
            eq_pressure = eq_pressure[:min_len]
            
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not extract equilibrium data: {e}")
        eq_times = []
        eq_pressure = []
        eq_pressure_beta = []
    except Exception as e:
        print(f"Warning: Could not compute volume-averaged pressure: {e}")
        eq_times = []
        eq_pressure = []
        eq_pressure_beta = []
    
    # Extract core_profiles time and calculate electron pressure
    cp_times = []
    cp_pressure = []
    
    try:
        cp_times = np.asarray(ods['core_profiles.profiles_1d[:].time'], float)
        n_e_vol = np.asarray(ods['core_profiles.global_quantities.n_e_volume_average'], float)
        T_e_vol = np.asarray(ods['core_profiles.global_quantities.t_e_volume_average'], float)
        
        # Calculate electron pressure: P_e = n_e * T_e * k_e (1.6e-19)
        # T_e is in eV, k_e = 1.6e-19 C (elementary charge)
        # For proper units: P (Pa) = n_e (m^-3) * T_e (eV) * e (C) * 1e3
        # where e = 1.6e-19 C, so: P = n_e * T_e * 1.6e-19 * 1e3
        k_e = 1.6e-19 
        cp_pressure = n_e_vol * T_e_vol * k_e
        
        # Check if arrays have same length
        if len(cp_times) != len(n_e_vol) or len(cp_times) != len(T_e_vol):
            print(f"Warning: Data length mismatch: times={len(cp_times)}, n_e={len(n_e_vol)}, T_e={len(T_e_vol)}")
            min_len = min(len(cp_times), len(n_e_vol), len(T_e_vol))
            cp_times = cp_times[:min_len]
            cp_pressure = cp_pressure[:min_len]
            
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not extract core_profiles data: {e}")
        cp_times = []
        cp_pressure = []
    
    # Plot on single figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot equilibrium pressure (from compute_volume_averaged_pressure)
    if len(eq_times) > 0 and len(eq_pressure) > 0:
        eq_times = np.asarray(eq_times)
        eq_pressure = np.asarray(eq_pressure)
        valid_eq = ~np.isnan(eq_pressure)
        if np.any(valid_eq):
            ax.plot(eq_times[valid_eq], eq_pressure[valid_eq], 'b-o', linewidth=2, 
                   markersize=4, label='Pressure (from equilibrium)', alpha=0.7)
    
    # # Plot equilibrium pressure (from beta_tor)
    # if len(eq_times) > 0 and len(eq_pressure_beta) > 0:
    #     eq_times_beta = np.asarray(eq_times)
    #     eq_pressure_beta = np.asarray(eq_pressure_beta)
    #     valid_beta = ~np.isnan(eq_pressure_beta)
    #     if np.any(valid_beta):
    #         ax.plot(eq_times_beta[valid_beta], eq_pressure_beta[valid_beta], 'c--^', linewidth=2, 
    #                markersize=4, label='Equilibrium Pressure (from beta_tor)', alpha=0.7)
    
    # Plot core_profiles electron pressure
    if len(cp_times) > 0 and len(cp_pressure) > 0:
        cp_times = np.asarray(cp_times)
        cp_pressure = np.asarray(cp_pressure)
        valid_cp = ~np.isnan(cp_pressure)
        if np.any(valid_cp):
            ax.plot(cp_times[valid_cp], cp_pressure[valid_cp], 'r-s', linewidth=2, 
                   markersize=4, label='Electron Pressure (n_e * T_e * k_e, from core_profiles)', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Pressure [Pa]', fontsize=12)
    ax.set_title('Volume-Averaged Pressure Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
