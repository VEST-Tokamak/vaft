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



def plot_thomson_profiles(ods):
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
    plt.show()
