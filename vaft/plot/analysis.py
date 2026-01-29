import os
import numpy as np
import matplotlib.pyplot as plt
import vaft
import matplotlib.patches as patches
from vaft.omas.process_wrapper import compute_point_vacuum_fields_ods


# matplotlib 설정 개선
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12

def analysis_diagnostics(ods):
    """
    Plot diagnostics
    """

    status=vaft.omas.find_shotclass(ods)
    shotnumber=vaft.omas.find_shotnumber(ods)

    tstart = vaft.omas.find_breakdown_onset(ods)
    pulse_length = vaft.omas.find_pulse_duration(ods)
    tend = tstart + pulse_length

    figsize=(24, 20)  # 더 큰 크기로 변경
    # Generate 5 x 2 subplots
    fig, ax = plt.subplots(5, 2, figsize=figsize, dpi=150, squeeze=False)

    # Set the title for the figure
    time_ip = ods['magnetics.ip.0.time']
    data_ip=ods['magnetics.ip.0.data']
    IpPeakTime = np.argmax(data_ip)
    Bt=ods['tf.b_field_tor_vacuum_r']['data']/0.4
    fig.suptitle(f'Diagnostics for {shotnumber} - {status} - Bt at Max Ip: {Bt[IpPeakTime]}', fontsize=16)
    # Plot Ip
    ax[0, 0].plot(ods['magnetics.time'], data_ip)
    # Plot Bt at R=0.4m => pressure
    #    ax[1, 0].plot(ods['tf.time'], ods['tf.b_field_tor_vacuum_r']['data']/0.4)
    data_pres=ods['barometry.gauge.0.pressure.data'] # pressure
    time_pres=ods['barometry.gauge.0.pressure.time'] # time
    ax[1, 0].plot(time_pres, data_pres)
    # Plot Line Radiation
    channel_idx = [0, 0, 1, 1, 1]
    line_idx = [0, 1, 0, 4, 5]
    line_label = ['Ha (show DaQ)', 'OI', 'Ha (Fast DaQ)', 'C-III', 'O-II']
    for i in range(5):
        ax[2, 0].plot(ods[f'spectrometer_uv.time'],\
                 ods[f'spectrometer_uv.channel.{channel_idx[i]}.processed_line.{line_idx[i]}.intensity.data'], label=line_label[i])
    ax[2, 0].legend()
    
    # Plot PF Coil
    for i in [1,5,6,9,10]:
        ax[3, 0].plot(ods['pf_active']['time'],ods['pf_active']['coil'][i-1]['current']['data']/1000,label=ods['pf_active']['coil'][i-1]['name'])
    ax[3, 0].legend()

    # Plot DiaFlux
    ax[4, 0].plot(ods['magnetics.time'], ods['magnetics.diamagnetic_flux.0.data'] * 1e3)

    # Find the index of the magnetic probes and flux loops
    Index_inFlux = np.where(ods['magnetics.flux_loop.:.position.0.r'] < 0.15)
    Index_OutFlux = np.where(ods['magnetics.flux_loop.:.position.0.r'] > 0.5)
    Index_inBz = np.where(ods['magnetics.b_field_pol_probe.:.position.r']<0.09)
    Index_outBz = np.where(ods['magnetics.b_field_pol_probe.:.position.r']>0.795)
    Index_sideBz = np.where(np.abs(ods['magnetics.b_field_pol_probe.:.position.z']) > 0.8)
    Index_inBz = Index_inBz[0]
    Index_sideBz = Index_sideBz[0]
    Index_outBz = Index_outBz[0]
    Index_inFlux = Index_inFlux[0]
    Index_OutFlux = Index_OutFlux[0]
    nb_inBz = len(Index_inBz)
    nb_sideBz = len(Index_sideBz)
    nb_outBz = len(Index_outBz)
    nb_Bz = nb_inBz + nb_sideBz + nb_outBz

    # Set the color map 
    cmap = plt.get_cmap('tab10')
    cmap = np.vstack((cmap(np.arange(10)), cmap(np.arange(10)), cmap(np.arange(10))))
    line_style = ['-'] * 10 + ['--'] * 10 + ['-.'] * 10
    
    # # Plot Inboard FL
    # for i, index in enumerate(Index_inFlux):
    #     if (index+nb_Bz) not in broken:
    #         ax[0, 1].plot(ods['magnetics.time'], ods[f'magnetics.flux_loop.{index}.flux.data'], label= f'{nb_Bz + index}', color=cmap[i], linestyle=line_style[i])
    # ax[0, 1].legend(fontsize = 'x-small', ncol=2)
    # # Plot Outboard FL
    # for i, index in enumerate(Index_OutFlux):
    #     if (index+nb_Bz) not in broken:
    #         ax[1, 1].plot(ods['magnetics.time'], ods[f'magnetics.flux_loop.{index}.flux.data'], label= f'{nb_Bz + index}', color=cmap[i], linestyle=line_style[i])
    # ax[1, 1].legend(fontsize = 'x-small')
    # # Plot Inboard Bz
    # for i, index in enumerate(Index_inBz):
    #     if index not in broken:
    #         ax[2, 1].plot(ods['magnetics.time'], ods[f'magnetics.b_field_pol_probe.{index}.field.data'], label= f'{index}', color=cmap[i], linestyle=line_style[i])
    # ax[2, 1].legend(ncol=4, fontsize = 'x-small')
    # # Plot Side Bz
    # for i, index in enumerate(Index_sideBz):
    #     if index not in broken:
    #         ax[3, 1].plot(ods['magnetics.time'], ods[f'magnetics.b_field_pol_probe.{index}.field.data'], label= f'{index}', color=cmap[i], linestyle=line_style[i])
    # ax[3, 1].legend(ncol=4, fontsize = 'x-small')
    # # Plot Outboard Bz
    # for i, index in enumerate(Index_outBz):
    #     if index not in broken:
    #         ax[4, 1].plot(ods['magnetics.time'], ods[f'magnetics.b_field_pol_probe.{index}.field.data'], label= f'{index}', color=cmap[i], linestyle=line_style[i])
    # ax[4, 1].legend(ncol=4, fontsize = 'x-small')

    # Set the title for each subplot
    ax[0, 0].set_title('Ip')
    #    ax[1, 0].set_title('Bt (R=0.4m)')
    ax[1, 0].set_title('Pressure')
    ax[2, 0].set_title('Line Radiation (Ha, CIII, OII)')
    ax[3, 0].set_title('PF Coil')
    ax[4, 0].set_title('DiaFlux') 
    ax[0, 1].set_title('Inboard FL')
    ax[1, 1].set_title('Outboard FL')
    ax[2, 1].set_title('Inboard Bz')
    ax[3, 1].set_title('Side Bz')
    ax[4, 1].set_title('Outboard Bz')

    # Set the xlim and xlabel for each subplot
    ax[0, 0].set_xlim([tstart, tend])
    ax[1, 0].set_xlim([tstart, tend])
    ax[2, 0].set_xlim([tstart, tend])
    ax[3, 0].set_xlim([tstart, tend])
    ax[4, 0].set_xlim([tstart, tend])
    ax[0, 1].set_xlim([tstart, tend])
    ax[1, 1].set_xlim([tstart, tend])
    ax[2, 1].set_xlim([tstart, tend])
    ax[3, 1].set_xlim([tstart, tend])
    ax[4, 1].set_xlim([tstart, tend])
    ax[4, 0].set_xlabel('Time (ms)')
    ax[4, 1].set_xlabel('Time (ms)')

    # Set the ylabel for each subplot
    ax[0, 0].set_ylabel('(kA)')
    ax[1, 0].set_ylabel('(Pa)')
    ax[2, 0].set_ylabel('(a.u.)')
    ax[3, 0].set_ylabel('(kA)')
    ax[4, 0].set_ylabel('(mWb)')
    ax[0, 1].set_ylabel('(Wb)')
    ax[1, 1].set_ylabel('(Wb)')
    ax[2, 1].set_ylabel('(T)')
    ax[3, 1].set_ylabel('(T)')
    ax[4, 1].set_ylabel('(T)')

    # Save the figure
    if not os.path.exists(os.path.join(save_dir, 'plots')):
        os.makedirs(os.path.join(save_dir, 'plots'))
    plt.savefig(os.path.join(save_dir, 'plots', f'{shotnumber}_diagnostics.png'))

    # Show the plot
    if show==1:
        plt.show()

def analysis_electromagnetics(ods):
    """
    Plot eddy
    """
    pass

def _plot_poloidal_geometry(ax, geometry, color):
    """
    Plot poloidal geometry data for coil and vessel structures.
    :param ax: Matplotlib axis to plot on
    :param geometry: Array of geometrical data
    :param color: Color for the geometrical shapes
    """
    for geom in geometry:
        rect = patches.Rectangle((geom[0] - geom[2] / 2, geom[1] - geom[3] / 2), geom[2], geom[3], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

# def analysis_operation(ods, xunit='ms', xlim='plasma'):
#     """
#     Generate a comprehensive time-dependent analysis of a single VEST discharge.

#     This multi-panel plot displays the evolution of primary signals, derived physics
#     parameters, and geometric properties throughout the shot.

#     - **Primary signals**: Measured Ip, diamagnetic flux, H-alpha emission, and
#       calculated on-axis loop voltage and vertical magnetic field.
#     - **Physics parameters**: Normalized beta (βN), internal inductance (li),
#       safety factors (q0, q95), and stored energy (WMHD) from EFIT.
#     - **Geometry**: Major radius (R), minor radius (a), elongation (κ),
#       triangularity (δ), and plasma volume.

#     Args:
#         ods (ODS): Input data.
#         xunit (str): Time unit for the x-axis ('s' or 'ms'). Default is 's'.
#         xlim (str or list): X-axis limits setting. Can be 'plasma', 'coil', 'none',
#                             or a list of two floats. Default is 'plasma'.
#     """

#     # Check for and calculate missing equilibrium data if equilibrium reconstruction exists
#     if 'equilibrium.time_slice' in ods and len(ods['equilibrium.time_slice']) > 0:
#         # Check for boundary parameters
#         if 'boundary' not in ods['equilibrium.time_slice.0']:
#             print("Equilibrium boundary parameters missing. Calculating...")
#             vaft.omas.update_equilibrium_boundary(ods)

#         # Check for MHD stored energy
#         if 'global_quantities.energy_mhd' not in ods['equilibrium.time_slice.0']:
#             print("Equilibrium stored energy (WMHD) missing. Calculating...")
#             vaft.omas.update_equilibrium_stored_energy(ods)

#     xlim_processed = handle_xlim(ods, xlim)
#     time_scale = 1000.0 if xunit == 'ms' else 1.0

#     # Pre-calculate vacuum fields for reuse
#     vacuum_time, vacuum_psi, _, vacuum_bz = compute_point_vacuum_fields_ods(ods, [(0.4, 0.0)], mode='vacuum')
#     vacuum_vloop = - np.gradient(vacuum_psi[:, 0], vacuum_time)

#     fig, axs = plt.subplots(
#         5, 3,
#         figsize=(20, 15),                 # 더 큰 크기로 변경 (width 20 : height 15)
#         dpi=150,                          # DPI를 높여서 더 선명하게
#         sharex=True,
#         gridspec_kw={'hspace': 0.1, 'wspace': 0.2}  # 열 간격을 0.3에서 0.2로 줄임
#     )

#     fig.subplots_adjust(
#         left=0.08, right=0.95,
#         top=0.90, bottom=0.08,
#         hspace=0.1, wspace=0.2            # 열 간격을 0.3에서 0.2로 줄임
#     )
#     def plot_quantity(ax, get_data, ylabel, style_key):
#         try:
#             time, y = get_data()
#             if xunit == 'ms':
#                 time = time * 1e3
#             style = PLOT_STYLES[style_key]
#             ax.plot(time, y, **style)
#             ax.set_ylabel(ylabel)
#         except Exception as e:
#             ax.text(0.5, 0.5, 'No data', ha='center', va='center')

#     # 스타일 사전
#     PLOT_STYLES = {
#         'diagnostic': dict(color='black', linestyle='-', label='Diagnostics'),
#         'vacuum': dict(color='tab:blue', linestyle='-', label='Vacuum'),
#         'equilibrium': dict(color='tab:red', linestyle='-', label='Equilibrium', marker='.'),
#     }

#     # --- Data Extraction Lambdas ---
#     # Column 1: Primary Signals
#     def get_ip():
#         return ods['magnetics.ip.0.time'], ods['magnetics.ip.0.data'] / 1e3
    
#     def get_ip_reconstructed():
#         return ods['magnetics.time'], ods['equilibrium.time_slice.:.global_quantities.ip'] / 1e3
    
#     def get_diamagnetic_flux():
#         return ods['magnetics.time'], ods['magnetics.diamagnetic_flux.0.data'] * 1e3 * (-1) # Wb -> mWb and negative sign
    
    
#     def get_h_alpha():
#         channel = 0
#         line_idx = 0
#         return ods['spectrometer_uv.time'], ods[f'spectrometer_uv.channel.{channel}.processed_line.{line_idx}.intensity.data']

#     def get_vloop():
#         return vacuum_time, vacuum_vloop
        
#     def get_bz_vacuum():
#         return vacuum_time, vacuum_bz[:, 0]

#     # Column 2: Physics Parameters
#     def get_wmhd():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.energy_mhd'] / 1e3 # kJ

#     def get_beta_n():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.beta_normal']

#     def get_li():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.li_3']

#     def get_q0():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.q_axis']

#     def get_q95():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.q_95']

#     # Column 3: Geometry Parameters
#     def get_rmajor():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.boundary.geometric_axis.r']

#     def get_aminor():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.boundary.minor_radius']

#     def get_elongation():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.boundary.elongation']

#     def get_triangularity():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.boundary.triangularity']

#     def get_volume():
#         return ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.volume']
    
#     # --- Plotting Calls ---
#     # Column 1: Primary Signals
#     plot_quantity(axs[0, 0], get_ip, r'$I_p$ [kA]', 'diagnostic')
#     plot_quantity(axs[1, 0], get_diamagnetic_flux, r'$\Delta \Phi_{\mathrm{D}}$ [mWb]', 'diagnostic')
#     plot_quantity(axs[2, 0], get_h_alpha, r'$\mathrm{H}_{\alpha}$ [a.u.]', 'diagnostic')
#     plot_quantity(axs[3, 0], get_vloop, r'$V_{\mathrm{loop}}$ [V]', 'vacuum')
#     plot_quantity(axs[4, 0], get_bz_vacuum, r'$B_{\mathrm{z}}$ [T]', 'vacuum')

#     # Column 2: Physics Parameters
#     plot_quantity(axs[0, 1], get_wmhd, r'$W_{\mathrm{MHD}}$ [kJ]', 'equilibrium')
#     plot_quantity(axs[1, 1], get_beta_n, r'$\beta_{\mathrm{N}}$', 'equilibrium')
#     plot_quantity(axs[2, 1], get_li, r'$l_{\mathrm{i}}$', 'equilibrium')
#     axs[2, 1].set_ylim(0, 1)
#     plot_quantity(axs[3, 1], get_q0, r'$q_0$', 'equilibrium')
#     # axs[3, 1].set_ylim(0, 4)
#     plot_quantity(axs[4, 1], get_q95, r'$q_{95}$', 'equilibrium')

#     # Column 3: Geometry Parameters
#     plot_quantity(axs[0, 2], get_rmajor, r'$R_{\mathrm{major}}$ [m]', 'equilibrium')
#     plot_quantity(axs[1, 2], get_aminor, r'$r_{\mathrm{minor}}$ [m]', 'equilibrium')
#     plot_quantity(axs[2, 2], get_elongation, r'$\kappa$', 'equilibrium')
#     plot_quantity(axs[3, 2], get_triangularity, r'$\delta$', 'equilibrium')
#     plot_quantity(axs[4, 2], get_volume, r'$V_{\mathrm{plasma}}$ [m$^3$]', 'equilibrium')

#     # --- Final Touches ---
#     for ax in axs.flat:
#         if not ax.lines: # Don't set xlim on empty plots with text
#             continue
#         if xlim_processed:
#             ax.set_xlim(xlim_processed)
#         ax.tick_params(axis='x', labelsize=10)
#         ax.tick_params(axis='y', labelsize=10)
#         ax.grid(True)
#         # Remove x-tick labels for non-bottom rows
#         if ax.get_subplotspec().rowspan.start < 4:
#             ax.tick_params(axis='x', labelbottom=False)

#     # Common x-axis label
#     for i in range(3):
#         # Only add xlabel if the plot is not off
#         if axs[4, i].axison:
#             axs[4, i].set_xlabel(f"Time [{xunit}]")

#     # --- Figure 전체 legend를 위 중앙에 ---
#     handles = [plt.Line2D([], [], color=style['color'], linestyle=style['linestyle'], label=style['label']) for style in PLOT_STYLES.values()]
#     fig.legend(
#         handles=handles,
#         loc='upper center',
#         bbox_to_anchor=(0.5, 0.98),  # 위치 조정
#         ncol=3,
#         fontsize=14,
#         frameon=False
#     )
#     # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
#     plt.show()



def time_equilibrium_analysis(ods, xunit='s', xlim='plasma'):
    """
    Generate a 3x2 analysis plot with vertically paired storylines.

    This plot provides a highly intuitive view by creating two parallel narratives:
    - Left Column (Core Performance): Ip, V_loop, beta_N
    - Right Column (Control & Edge): H_alpha, B_z, R_major

    Args:
        ods (ODS): Input data object from the omas library.
        xunit (str): Time unit for the x-axis ('s' or 'ms'). Default is 's'.
        xlim (str or list): X-axis limits setting. Can be 'plasma', 'coil', 'none',
                            or a list of two floats. Default is 'plasma'.
    """
    from .time import handle_xlim
    xlim_processed = handle_xlim(ods, xlim)
    vacuum_time, vacuum_vloop, _, vacuum_bz = compute_point_vacuum_fields_ods(ods, [(0.4, 0.0)], mode='vacuum')

    fig, axs = plt.subplots(
        3, 2,
        figsize=(14, 12),  # 3x2 레이아웃에 맞는 크기
        dpi=150,
        sharex=True,
        gridspec_kw={'hspace': 0.15, 'wspace': 0.25}
    )

    fig.subplots_adjust(
        left=0.1, right=0.9,
        top=0.9, bottom=0.1
    )

    PLOT_STYLES = {
        'diagnostic': dict(color='black', linestyle='-'),
        'vacuum': dict(color='tab:blue', linestyle='-'),
        'equilibrium': dict(color='tab:red', linestyle='-', marker='.'),
    }

    def plot_quantity(ax, get_data, ylabel, style_key):
        """데이터를 가져와 지정된 축에 플롯하는 헬퍼 함수"""
        try:
            time, y = get_data()
            if xunit == 'ms':
                time *= 1e3
            style = PLOT_STYLES[style_key]
            ax.plot(time, y, **style)
            ax.set_ylabel(ylabel, fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    def get_ip():
        return ods['magnetics.ip.0.time'], ods['magnetics.ip.0.data'] / 1e3
    def get_h_alpha():
        channel = 0
        line_idx = 0
        return ods['spectrometer_uv.time'], ods[f'spectrometer_uv.channel.{channel}.processed_line.{line_idx}.intensity.data']
    def get_vloop(): return vacuum_time, vacuum_vloop
    def get_bz_vacuum(): return vacuum_time, vacuum_bz[:, 0] if vacuum_bz.ndim > 1 else vacuum_bz
    def get_beta_n():
        return ods['equilibrium.time'], ods['equilibrium.time_slice.:.global_quantities.beta_normal']
    def get_rmajor():
        return ods['equilibrium.time'], ods['equilibrium.time_slice.:.boundary.geometric_axis.r']

    # Row 1
    plot_quantity(axs[0, 0], get_ip, r'$I_p$ [kA]', 'diagnostic')
    plot_quantity(axs[0, 1], get_h_alpha, r'$\mathrm{H}_{\alpha}$ [a.u.]', 'diagnostic')

    # Row 2
    plot_quantity(axs[1, 0], get_vloop, r'$V_{\mathrm{loop}}$ [V]', 'vacuum')
    plot_quantity(axs[1, 1], get_bz_vacuum, r'$B_{\mathrm{z}}$ [T]', 'vacuum')

    # Row 3
    plot_quantity(axs[2, 0], get_beta_n, r'$\beta_{\mathrm{N}}$', 'equilibrium')
    plot_quantity(axs[2, 1], get_rmajor, r'$R_{\mathrm{major}}$ [m]', 'equilibrium')

    for i in range(3):
        for j in range(2):
            ax = axs[i, j]
            if not ax.lines: continue
            if xlim_processed: ax.set_xlim(xlim_processed)
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            if i < 2:
                ax.tick_params(axis='x', labelbottom=False)

    for j in range(2):
        if axs[2, j].lines:
            axs[2, j].set_xlabel(f"Time [{xunit}]", fontsize=12)

    handles = [
        plt.Line2D([], [], **PLOT_STYLES['diagnostic'], label='Diagnostics'),
        plt.Line2D([], [], **PLOT_STYLES['vacuum'], label='Vacuum'),
        plt.Line2D([], [], **PLOT_STYLES['equilibrium'], label='Equilibrium'),
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        fontsize=14,
        frameon=False
    )
    plt.show()
