from pathlib import Path

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

def _value(ods, path, default=None):
    try:
        if path not in ods:
            return default
        return ods[path]
    except (KeyError, TypeError, ValueError, IndexError):
        return default


def _plot_series(axis, time, data, *, label=None, scale=1.0):
    """Draw one signal, skipping it when the ODS does not carry the data."""
    if time is None or data is None:
        return
    time = np.asarray(time, dtype=float).reshape(-1)
    data = np.asarray(data, dtype=float).reshape(-1)
    if time.size == 0 or time.size != data.size:
        return
    axis.plot(time, data * scale, label=label)


def _mark_missing(axis):
    if not axis.lines:
        axis.text(0.5, 0.5, "Data unavailable", ha="center", va="center", transform=axis.transAxes)


def analysis_diagnostics(
    ods,
    *,
    time_range=None,
    save_path=None,
    show=True,
    figsize=(16, 14),
):
    """Plot a ten-panel overview of available VEST diagnostic signals."""
    if time_range is None:
        try:
            start = float(vaft.omas.find_breakdown_onset(ods))
            duration = float(vaft.omas.find_pulse_duration(ods))
            if np.isfinite(start) and np.isfinite(duration) and duration > 0.0:
                time_range = (start, start + duration)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            pass
    fig, axes = plt.subplots(5, 2, figsize=figsize, dpi=150, squeeze=False, sharex=True)
    try:
        shot = vaft.omas.find_shotnumber(ods)
    except Exception:
        shot = _value(ods, "dataset_description.data_entry.pulse", "unknown")
    try:
        status = vaft.omas.find_shotclass(ods)
    except Exception:
        status = "unknown"
    fig.suptitle(f"Diagnostics for {shot} - {status}", fontsize=16)

    magnetics_time = _value(ods, "magnetics.time")
    _plot_series(axes[0, 0], _value(ods, "magnetics.ip.0.time", magnetics_time), _value(ods, "magnetics.ip.0.data"), scale=1e-3)
    _plot_series(axes[1, 0], _value(ods, "barometry.gauge.0.pressure.time"), _value(ods, "barometry.gauge.0.pressure.data"))

    uv_time = _value(ods, "spectrometer_uv.time")
    for channel, line, label in [(0, 0, "Ha (slow)"), (0, 1, "OI"), (1, 0, "Ha (fast)"), (1, 4, "C-III"), (1, 5, "O-II")]:
        _plot_series(axes[2, 0], uv_time, _value(ods, f"spectrometer_uv.channel.{channel}.processed_line.{line}.intensity.data"), label=label)

    pf_time = _value(ods, "pf_active.time")
    for index in (0, 4, 5, 8, 9):
        _plot_series(
            axes[3, 0], pf_time,
            _value(ods, f"pf_active.coil.{index}.current.data"),
            label=_value(ods, f"pf_active.coil.{index}.name", str(index + 1)), scale=1e-3,
        )
    _plot_series(axes[4, 0], _value(ods, "magnetics.diamagnetic_flux.0.time", magnetics_time), _value(ods, "magnetics.diamagnetic_flux.0.data"), scale=1e3)

    flux_groups = (axes[0, 1], axes[1, 1])
    probe_groups = (axes[2, 1], axes[3, 1], axes[4, 1])
    try:
        flux_count = len(ods["magnetics.flux_loop"])
    except (KeyError, TypeError):
        flux_count = 0
    for index in range(flux_count):
        radius = float(_value(ods, f"magnetics.flux_loop.{index}.position.0.r", np.nan))
        group = 0 if radius < 0.15 else 1 if radius > 0.5 else None
        if group is not None:
            _plot_series(flux_groups[group], magnetics_time, _value(ods, f"magnetics.flux_loop.{index}.flux.data"), label=str(index))

    try:
        probe_count = len(ods["magnetics.b_field_pol_probe"])
    except (KeyError, TypeError):
        probe_count = 0
    for index in range(probe_count):
        radius = float(_value(ods, f"magnetics.b_field_pol_probe.{index}.position.r", np.nan))
        z = float(_value(ods, f"magnetics.b_field_pol_probe.{index}.position.z", np.nan))
        group = 0 if radius < 0.09 else 1 if abs(z) > 0.8 else 2 if radius > 0.795 else None
        if group is not None:
            _plot_series(probe_groups[group], magnetics_time, _value(ods, f"magnetics.b_field_pol_probe.{index}.field.data"), label=str(index))

    titles = (("Ip", "Inboard FL"), ("Pressure", "Outboard FL"), ("Line Radiation", "Inboard Bz"), ("PF Coil", "Side Bz"), ("DiaFlux", "Outboard Bz"))
    ylabels = (("kA", "Wb"), ("Pa", "Wb"), ("a.u.", "T"), ("kA", "T"), ("mWb", "T"))
    for row in range(5):
        for column in range(2):
            axis = axes[row, column]
            axis.set_title(titles[row][column])
            axis.set_ylabel(ylabels[row][column])
            axis.grid(True, alpha=0.3)
            _mark_missing(axis)
            if axis.lines and any(line.get_label() and not line.get_label().startswith("_") for line in axis.lines):
                axis.legend(fontsize="x-small", ncol=2)
            if time_range is not None:
                axis.set_xlim(time_range)
    axes[4, 0].set_xlabel("Time [s]")
    axes[4, 1].set_xlabel("Time [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        target = Path(save_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
    if show:
        plt.show()
    return fig, axes

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
