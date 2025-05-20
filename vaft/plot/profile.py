import numpy as np
import matplotlib.pyplot as plt
import omas
import matplotlib.pyplot as plt
import numpy as np

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
    Plot time series of electron density (ne) and temperature (Te)
    from thomson_scattering data for each channel (with names).
    """
    if 'thomson_scattering.channel' not in ods or 'thomson_scattering.time' not in ods:
        print("Thomson data not found in ODS.")
        return

    time = np.array(ods['thomson_scattering.time']) * 1e3  # convert to ms
    num_channels = len(ods['thomson_scattering.channel'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i in range(num_channels):
        try:
            name = ods.get(f'thomson_scattering.channel.{i}.name', f'Channel {i}')
            ne = np.array(ods[f'thomson_scattering.channel.{i}.n_e.data']) / 1e19
            te = np.array(ods[f'thomson_scattering.channel.{i}.t_e.data'])

            axes[0].plot(time, ne, label=name)
            axes[1].plot(time, te, label=name)

        except Exception as e:
            print(f"[WARNING] Failed to plot channel {i}: {e}")
            continue

    # Plot settings
    axes[0].set_ylabel('ne [$10^{19}$ m$^{-3}$]')
    axes[0].legend(ncol=2, fontsize='small')
    axes[0].grid(True)

    axes[1].set_ylabel('Te [eV]')
    axes[1].set_xlabel('Time [ms]')
    axes[1].legend(ncol=2, fontsize='small')
    axes[1].grid(True)

    fig.suptitle('Thomson Scattering Time Series (per channel)')
    plt.tight_layout()
    plt.show()