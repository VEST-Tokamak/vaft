import numpy as np
import matplotlib.pyplot as plt
import omas
from matplotlib.ticker import ScalarFormatter


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
    if 'wall' in ods and 'description_2d' in ods['wall']:
        wall_desc = ods['wall']['description_2d']
        wall_idx = time_index if time_index in wall_desc else 0
        if 'limiter' in wall_desc[wall_idx] and 'unit' in wall_desc[wall_idx]['limiter']:
            for unit in wall_desc[wall_idx]['limiter']['unit']:
                ax.plot(unit['outline.r'], unit['outline.z'], color='gray', alpha=0.4)

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
        n_e_fit = prof['electrons']['density_thermal']

        # --- Thomson diagnostic reference (measured points) ---
        rho_meas = prof['electrons']['temperature_fit']['rho_tor_norm']
        T_e_meas = prof['electrons']['temperature_fit']['measured']
        n_e_meas = prof['electrons']['density_fit']['measured']

        # --- Error extraction from ODS (if exists) ---
        try:
            t_e_err = [ods[f'thomson_scattering.channel.{j}.t_e.data_error_upper'][i] for j in range(num_channels)]
            n_e_err = [ods[f'thomson_scattering.channel.{j}.n_e.data_error_upper'][i] for j in range(num_channels)]
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
    axs[1].set_ylabel(r'$N_e$ (m$^{-3}$)', fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=6)
    axs[1].grid(True)
    axs[1].legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if save_opt == 1:
        plt.savefig(file_name if file_name else 'thomson_profiles.png', dpi=150)
    else:
        plt.show()

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
        axs[1].set_ylabel(r'$T_e$ (keV)', fontsize=12)
        axs[1].set_xlabel(r'$\psi_N$', fontsize=12)
        axs[1].set_title(f'Electron Temperature Profile (t={time_t:.3f}s)', fontsize=14)
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=10)
    except Exception as e:
        print(f"Warning: Could not plot T_e profile: {e}")
    
    plt.tight_layout()
    plt.show()


def plot_electron_2d_profile(ods, option='n_e', time_slice=None, figsize=(12, 10)):
    """
    Plot electron profiles in 2D (R,Z) coordinate system.
    
    Args:
        ods: OMAS data structure
        option: Profile option ('n_e' or 't_e')
        time_slice: Time slice index (None = use first available)
        figsize: Figure size tuple
    """
    from vaft.omas.process_wrapper import compute_core_profile_2d
    from matplotlib.colors import LogNorm
    
    if option not in ['n_e', 't_e']:
        raise ValueError(f"option must be 'n_e' or 't_e', got '{option}'")
    
    try:
        profile_RZ, R_grid, Z_grid, psiN_RZ, time_val = compute_core_profile_2d(
            ods, option=option, time_slice=time_slice
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create meshgrid for plotting (use 'ij' indexing to match shape convention)
        # profile_RZ has shape (len(R_grid), len(Z_grid))
        R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid, indexing='ij')
        
        # Plot contour
        if option == 'n_e':
            label = r'$n_e$ (m$^{-3}$)'
            # Use log scale for density
            levels = np.logspace(np.log10(profile_RZ[profile_RZ > 0].min()), 
                                np.log10(profile_RZ.max()), 20)
            cs = ax.contourf(R_mesh, Z_mesh, profile_RZ, levels=levels, cmap='plasma', extend='both')
            norm = LogNorm(vmin=profile_RZ[profile_RZ > 0].min(), vmax=profile_RZ.max())
        else:  # T_e
            label = r'$T_e$ (keV)'
            levels = np.linspace(profile_RZ.min(), profile_RZ.max(), 20)
            cs = ax.contourf(R_mesh, Z_mesh, profile_RZ, levels=levels, cmap='hot', extend='both')
            norm = None
        
        # Add contour lines
        ax.contour(R_mesh, Z_mesh, profile_RZ, levels=levels[::3], colors='k', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label(label, fontsize=12)
        
        # Add flux surfaces
        if psiN_RZ is not None:
            psi_levels = np.linspace(psiN_RZ.min(), psiN_RZ.max(), 10)
            ax.contour(R_mesh, Z_mesh, psiN_RZ, levels=psi_levels, colors='w', alpha=0.5, linewidths=1)
        
        ax.set_xlabel('R (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title(f'Electron {option.upper()} Profile in 2D (R,Z) (t={time_val:.3f}s)', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
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
    
    # Extract time and volume-averaged quantities
    times = []
    n_e_vol = []
    T_e_vol = []
    
    if 'core_profiles.profiles_1d' not in ods:
        print("Warning: core_profiles.profiles_1d not found in ODS")
        return
    
    for idx in range(len(ods['core_profiles.profiles_1d'])):
        cp_ts = ods['core_profiles.profiles_1d'][idx]
        
        # Get time
        if 'time' in cp_ts:
            times.append(float(cp_ts['time']))
        elif 'core_profiles.time' in ods and idx < len(ods['core_profiles.time']):
            times.append(float(ods['core_profiles.time'][idx]))
        else:
            times.append(float(idx))
        
        # Get volume-averaged quantities
        if 'core_profiles.global_quantities' in ods:
            if isinstance(ods['core_profiles.global_quantities'], list):
                if idx < len(ods['core_profiles.global_quantities']):
                    gq = ods['core_profiles.global_quantities'][idx]
                    if 'n_e_volume_average' in gq:
                        n_e_vol.append(float(gq['n_e_volume_average']))
                    else:
                        n_e_vol.append(np.nan)
                    if 't_e_volume_average' in gq:
                        T_e_vol.append(float(gq['t_e_volume_average']))
                    else:
                        T_e_vol.append(np.nan)
                else:
                    n_e_vol.append(np.nan)
                    T_e_vol.append(np.nan)
            else:
                gq = ods['core_profiles.global_quantities']
                if 'n_e_volume_average' in gq:
                    n_e_vol.append(float(gq['n_e_volume_average']))
                else:
                    n_e_vol.append(np.nan)
                if 't_e_volume_average' in gq:
                    T_e_vol.append(float(gq['t_e_volume_average']))
                else:
                    T_e_vol.append(np.nan)
        else:
            n_e_vol.append(np.nan)
            T_e_vol.append(np.nan)
    
    times = np.asarray(times)
    n_e_vol = np.asarray(n_e_vol)
    T_e_vol = np.asarray(T_e_vol)
    
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
        axs[1].set_ylabel(r'$\langle T_e \rangle_V$ (keV)', fontsize=12)
        axs[1].set_xlabel('Time (s)', fontsize=12)
        axs[1].set_title('Volume-Averaged Electron Temperature', fontsize=14)
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=10)
    else:
        axs[1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axs[1].transAxes)
    
    plt.tight_layout()
    plt.show()
