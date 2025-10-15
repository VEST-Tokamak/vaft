from omas import *
import matplotlib.pyplot as plt
import vaft
import numpy as np
from matplotlib.path import Path
import logging
logger = logging.getLogger(__name__)


def pf_passive_overlay(ods, ax=None, colors=None, **kw):
    """
    Plot the passive loops (pf_passive) as polygons on a matplotlib axis.
    Each loop is drawn using its outline.r and outline.z coordinates.
    If ax is not provided, a new figure and axis are created.
    colors: list of colors or a matplotlib colormap (optional)
    """
    PFP = ods['pf_passive']
    nbloop = len(PFP['loop']) if isinstance(PFP['loop'], dict) else len(PFP['loop'])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    # Color handling
    if colors is None:
        from itertools import cycle
        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        get_color = lambda i: next(color_cycle)
    elif callable(colors):  # colormap
        get_color = lambda i: colors(i / max(nbloop - 1, 1))
    elif isinstance(colors, str):
        get_color = lambda i: colors
    elif isinstance(colors, (tuple, list)) and len(colors) == 4 and all(isinstance(c, (float, int)) for c in colors):
        # Single RGBA tuple
        get_color = lambda i: colors
    else:
        get_color = lambda i: colors[i % len(colors)]
    for iLoop in range(nbloop):
        r = PFP[f'loop.{iLoop}.element.0.geometry.outline.r']
        z = PFP[f'loop.{iLoop}.element.0.geometry.outline.z']
        # Close the polygon if not already closed
        if r[0] != r[-1] or z[0] != z[-1]:
            r = list(r) + [r[0]]
            z = list(z) + [z[0]]
        ax.plot(r, z, color=get_color(iLoop), **kw)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.set_title('PF Passive Loops Overlay')

def overlay_all(ods):
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [
        "#377eb8",  # Flux Loop
        "#ff7f00",  # B Field Pol Probe
        "#4daf4a",  # Coil (PF Active)
        "#e41a1c",  # Vessel (PF Passive)
        "#984ea3",  # Wall
        "#a65628",  # Thomson Scattering
        "#f781bf",  # 추가
        "#999999",  # 추가
    ]

    fig, ax = plt.subplots(figsize=(4, 8.4))

    handles = []
    labels = []

    # Magnetics overlay
    if hasattr(ods, 'plot_magnetics_overlay'):
        ods.plot_magnetics_overlay(ax=ax)
        handles.append(plt.Line2D([], [], color=colors[0]))
        labels.append('Flux Loop')
        handles.append(plt.Line2D([], [], color=colors[1]))
        labels.append('B Field Pol Probe')

    # PF Active overlay (Coil)
    if hasattr(ods, 'plot_pf_active_overlay'):
        ods.plot_pf_active_overlay(ax=ax, edgecolor=colors[2], facecolor='none')  # 원하는 색상 지정
        handles.append(plt.Line2D([], [], color=colors[2]))
        labels.append('Coil (PF Active)')

    # PF Passive overlay (Vessel)
    # pf_passive_overlay(ods, ax=ax, colors=colors[3])
    # make color 3 80% transparent
    from matplotlib.colors import to_rgba
    color = to_rgba(colors[3], alpha=0.6)  # 60% transparent
    vaft.plot.pf_passive_overlay(ods, ax=ax, colors=color)
    handles.append(plt.Line2D([], [], color=color))
    labels.append('Vessl (PF Passive)')

    # Wall overlay
    if hasattr(ods, 'plot_wall_overlay'):
        from matplotlib.colors import to_rgba
        color = to_rgba(colors[4], alpha=0.8)  # 80% transparent
        ods.plot_wall_overlay(ax=ax, color=color)
        handles.append(plt.Line2D([], [], color=color))
        labels.append('Limiter (Wall)')

    # Thomson Scattering overlay
    if 'thomson_scattering' in ods and hasattr(ods, 'plot_thomson_scattering_overlay'):
        ods.plot_thomson_scattering_overlay(ax=ax, colors=colors[5])
        handles.append(plt.Line2D([], [], color=colors[5]))
        labels.append('Thomson Scattering')
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.7)
    ax.legend(handles, labels, loc='upper center')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.set_title('VEST Overlay')
    ax.legend(handles, labels)
    plt.show()

def overlay_all_with_vacuum_psi_contour(ods, time=None, cmap='viridis', fontsize=12, savepath=None):
    """
    overlay_all과 vacuum_psi_contour를 결합하여,
    모든 오버레이(자기장, 코일, vessel, wall, thomson 등)와 vacuum psi contour를 한 플롯에 그린다.
    """
    colors = [
        "#377eb8",  # Flux Loop
        "#ff7f00",  # B Field Pol Probe
        "#4daf4a",  # Coil (PF Active)
        "#e41a1c",  # Vessel (PF Passive)
        "#984ea3",  # Wall
        "#a65628",  # Thomson Scattering
        "#f781bf",  # 추가
        "#999999",  # 추가
    ]

    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=(4, 8.4))

    handles = []
    labels = []

    # Magnetics overlay
    if hasattr(ods, 'plot_magnetics_overlay'):
        ods.plot_magnetics_overlay(ax=ax)
        handles.append(plt.Line2D([], [], color=colors[0]))
        labels.append('Flux Loop')
        handles.append(plt.Line2D([], [], color=colors[1]))
        labels.append('B Field Pol Probe')

    # PF Active overlay (Coil)
    if hasattr(ods, 'plot_pf_active_overlay'):
        ods.plot_pf_active_overlay(ax=ax, edgecolor=colors[2], facecolor='none')
        handles.append(plt.Line2D([], [], color=colors[2]))
        labels.append('Coil (PF Active)')

    # PF Passive overlay (Vessel)
    from matplotlib.colors import to_rgba
    color_vessel = to_rgba(colors[3], alpha=0.6)
    vaft.plot.pf_passive_overlay(ods, ax=ax, colors=color_vessel)
    handles.append(plt.Line2D([], [], color=color_vessel))
    labels.append('Vessl (PF Passive)')

    # Wall overlay
    if hasattr(ods, 'plot_wall_overlay'):
        color_wall = to_rgba(colors[4], alpha=0.8)
        ods.plot_wall_overlay(ax=ax, color=color_wall)
        handles.append(plt.Line2D([], [], color=color_wall))
        labels.append('Limiter (Wall)')

    # Thomson Scattering overlay
    if 'thomson_scattering' in ods and hasattr(ods, 'plot_thomson_scattering_overlay'):
        ods.plot_thomson_scattering_overlay(ax=ax, colors=colors[5])
        handles.append(plt.Line2D([], [], color=colors[5]))
        labels.append('Thomson Scattering')

    # --- vacuum psi contour 추가 ---
    if time is None:
        time = vaft.omas.find_breakdown_onset(ods)
    try:
        psi, R, Z = vaft.omas.compute_null_ods(ods, time)
    except Exception as e:
        logger.warning(f"Could not compute psi: {e}. Skipping psi contour.")
        psi, R, Z = None, None, None

    psi_to_plot = psi
    chamberboundary_data = None
    plot_chamber_boundary = False

    if psi is not None and R is not None and Z is not None:
        try:
            limiter_r, limiter_z = vaft.omas.find_chamber_boundary(ods)
            chamberboundary_data = np.vstack((limiter_r, limiter_z)).T
            path = Path(chamberboundary_data)
            points_for_mask = np.vstack((R.flatten(), Z.flatten()))
            mask = path.contains_points(points_for_mask.T).reshape(R.shape)
            psi_to_plot = np.where(mask, psi, np.nan)
            plot_chamber_boundary = True
        except Exception as e:
            logger.warning(f"Could not find or process chamber boundary: {e}. Plotting unmasked psi.")

        # chamber boundary
        if plot_chamber_boundary and chamberboundary_data is not None:
            ax.plot(chamberboundary_data[:,0], chamberboundary_data[:,1], 'k-')
        # psi contour
        cset = ax.contour(R, Z, psi_to_plot, 50, cmap=cmap)

    # 기타 설정
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.7)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_aspect('equal')
    ax.set_title(f'VEST Overlay with\nVacuum $\\psi$ contour at breakdown onset')
    ax.legend(handles, labels, loc='upper center')

    # psi contour의 경우, 외곽선/눈금/라벨 제거 옵션을 추가로 줄 수 있음 (원하면 추가)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    return fig, ax

def vacuum_psi_contour(ods, time=None, cmap='viridis', fontsize=12, savepath=None):
    """
    Plot the vacuum psi contour at a given time, focusing on the chamber content.
    No legend, no external box, only title and chamber boundary with psi contour.
    """
    if time is None:
        time = vaft.omas.find_breakdown_onset(ods)

    psi, R, Z = vaft.omas.compute_null_ods(ods, time)

    psi_to_plot = psi
    chamberboundary_data = None
    plot_chamber_boundary = False

    try:
        limiter_r, limiter_z = vaft.omas.find_chamber_boundary(ods)
        chamberboundary_data = np.vstack((limiter_r, limiter_z)).T
        
        path = Path(chamberboundary_data)
        points_for_mask = np.vstack((R.flatten(), Z.flatten()))
        mask = path.contains_points(points_for_mask.T).reshape(R.shape)
        psi_to_plot = np.where(mask, psi, np.nan)
        plot_chamber_boundary = True
    except Exception as e:
        logger.warning(f"Could not find or process chamber boundary: {e}. Plotting unmasked psi.")

    plt.rcParams.update({'font.size': fontsize})
    # plt.rcParams.update({'axes.labelsize': fontsize}) # Axis labels will be removed or kept based on later decision
    
    fig, ax = plt.subplots(figsize=(4, 6)) 
    
    if plot_chamber_boundary and chamberboundary_data is not None:
        ax.plot(chamberboundary_data[:,0], chamberboundary_data[:,1], 'k-') 
        
    cset = ax.contour(R, Z, psi_to_plot, 50, cmap=cmap)
    
    ax.set_aspect('equal')
    # ax.set_xlabel('R [m]') # Remove R label as per implied request for cleaner look
    # ax.set_ylabel('Z [m]') # Remove Z label
    
    # Remove legend by not calling ax.legend()

    # Remove external box (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove ticks and tick labels for a cleaner plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.title(f'Vacuum $\psi$ contour at\n t={time*1000:.1f}ms', fontsize=fontsize*1.1, y=1.02)
    
    # Determine plot limits based on chamber boundary if available, otherwise based on psi data extent
    if plot_chamber_boundary and chamberboundary_data is not None:
        r_min_plot, r_max_plot = chamberboundary_data[:,0].min(), chamberboundary_data[:,0].max()
        z_min_plot, z_max_plot = chamberboundary_data[:,1].min(), chamberboundary_data[:,1].max()
        # Add some padding
        padding_r = (r_max_plot - r_min_plot) * 0.05
        padding_z = (z_max_plot - z_min_plot) * 0.05
        ax.set_xlim(r_min_plot - padding_r, r_max_plot + padding_r)
        ax.set_ylim(z_min_plot - padding_z, z_max_plot + padding_z)
    else: # Fallback if no chamber boundary
        # Use valid psi data extent for limits
        valid_r = R[~np.isnan(psi_to_plot)]
        valid_z = Z[~np.isnan(psi_to_plot)]
        if len(valid_r) > 0 and len(valid_z) > 0:
            r_min_plot, r_max_plot = valid_r.min(), valid_r.max()
            z_min_plot, z_max_plot = valid_z.min(), valid_z.max()
            padding_r = (r_max_plot - r_min_plot) * 0.05
            padding_z = (z_max_plot - z_min_plot) * 0.05
            ax.set_xlim(r_min_plot - padding_r, r_max_plot + padding_r)
            ax.set_ylim(z_min_plot - padding_z, z_max_plot + padding_z)
        # else: no valid data to plot, limits will be default

    # No need for tight_layout if we manually control everything and remove spines/ticks
    # plt.tight_layout(rect=[0, 0, 1, 0.95]) 
                                        
    if savepath:
        # For a plot without axes, ensure background is not transparent if desired, or set facecolor
        fig.savefig(savepath, bbox_inches='tight', pad_inches=0.05) # pad_inches can help
    plt.show()
    return fig, ax



# def twodim_geometry_coil():
# def twodim_geometry_wall():
# def twodim_geometry_vessel():


# def twodim_equilibrium_boundary():
# def twodim_equilibrium_magnetic_axis():
# def twodim_equilibrium_j_tor():
# def twodim_equilibrium_psi():
# def twodim_equilibrium_q():
# def twodim_equilibrium_f():
# def twodim_equilibrium_ffprime():

