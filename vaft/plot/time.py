"""
This module contains functions for plotting time series data from OMAS ODS.
"""
import uncertainties.unumpy as unumpy 
from omas import *
import matplotlib.pyplot as plt
from vaft.process import signal_onoffset, is_signal_active
import matplotlib.pyplot as plt
import numpy as np
from vaft.omas import odc_or_ods_check
from vaft.plot.utils import get_from_path, extract_labels_from_odc
from vaft.omas.process_wrapper import compute_point_vacuum_fields_ods
import vaft.omas


"""
Fllowing functions are tools for plotting time series data.
"""

def handle_xlim(odc_or_ods, xlim_param='plasma'):
    """Helper function to handle xlim logic."""
    odc = odc_or_ods_check(odc_or_ods)
    if xlim_param == 'none':
        return None
    elif xlim_param == 'plasma':
        return set_xlim_time(odc, type='plasma')
    elif xlim_param == 'coil':
        return set_xlim_time(odc, type='coil')
    elif isinstance(xlim_param, list) and len(xlim_param) == 2:
        return xlim_param
    else:
        print(f"Invalid xlim: {xlim_param}, using default 'plasma'")
        return set_xlim_time(odc, type='plasma')

def handle_labels(odc, label_param, default_opt='key'):
    """Helper function to handle label logic."""
    if isinstance(label_param, list) and len(label_param) == len(odc.keys()):
        return label_param
    elif label_param in ['shot', 'pulse', 'run', 'key']:
        return extract_labels_from_odc(odc, opt=label_param)
    else:
        print(f"Invalid label: {label_param}, using {default_opt} as label.")
        return extract_labels_from_odc(odc, opt=default_opt)

def set_xlim_time(odc, type='plasma'):
    """
    Set time limits for x-axis of plot.
    
    Parameters:
    odc (ODC): ODC object to extract time limits from.
    type (str): Type of time limits to set. Options are 'plasma' or 'coil' or 'none'.
    """
    onsets = []
    offsets = []
    
    for key in odc.keys():
        ods = odc[key]
        try:
            if type == 'plasma' and 'magnetics.ip' in ods:
                time = ods['magnetics.ip.0.time']
                data = ods['magnetics.ip.0.data']
                onset, offset = signal_onoffset(time, data)
                onsets.append(onset)
                offsets.append(offset)
                
            elif type == 'coil' and 'pf_active.coil' in ods:
                num_coils = len(ods['pf_active.coil'])
                for i in range(num_coils):
                    time = ods['pf_active.time']
                    data = ods[f'pf_active.coil.{i}.current.data']
                    onset, offset = signal_onoffset(time, data)
                    onsets.append(onset)
                    offsets.append(offset)
                    
        except KeyError as e:
            print(f"Missing key {str(e)} in ODS {key}")
            continue

    if not onsets or not offsets:
        return None
        
    return [np.min(onsets), np.max(offsets)]

"""
Routinely available signals : pf_active, ip, flux_loop, bpol_probe, spectrometer_uv (filterscope), tf

Routinely available modelling : pf_passive, equilibrium
"""


"""
PF Active plotting functions
"""

def _determine_coil_indices(odc, indices_param):
    """Helper to determine coil indices to plot for pf_active functions."""
    if indices_param == 'used':
        coil_indices = set()
        for key in odc.keys():
            ods = odc[key]
            if 'pf_active.coil' in ods:
                num_coils = len(ods['pf_active.coil'])
                for i in range(num_coils):
                    if f'pf_active.coil.{i}.current.data' in ods and is_signal_active(ods[f'pf_active.coil.{i}.current.data']):
                        coil_indices.add(i)
        return sorted(list(coil_indices))
    elif indices_param == 'all':
        max_coils = 0
        if odc.values(): # Check if odc is not empty
            max_coils = max((len(ods.get('pf_active.coil', [])) for ods in odc.values()), default=0)
        return list(range(max_coils))
    elif isinstance(indices_param, int):
        return [indices_param]
    elif isinstance(indices_param, list):
        return indices_param
    else:
        raise ValueError("indices must be 'used', 'all', or a list of integers")

def _plot_pf_active_time_generic(odc_or_ods, indices_param, label_param, xunit, yunit_label, title_suffix, data_retrieval_func, xlim_param, y_value_multiplier_func=None):
    """Generic plotting function for pf_active coil data with subplots."""
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim_param)
    labels = handle_labels(odc, label_param)
    
    coil_indices = _determine_coil_indices(odc, indices_param)

    if not coil_indices:
        print("No valid coils found to plot")
        return

    nrows = len(coil_indices)
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 1.5 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    for ax, coil_idx in zip(axs, coil_indices):
        plot_successful_for_coil = False
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                time_data, current_data, coil_name = data_retrieval_func(ods, coil_idx, xunit)
                
                if y_value_multiplier_func:
                    current_data = y_value_multiplier_func(current_data, ods, coil_idx)

                ax.plot(time_data, current_data, label=lbl)
                plot_successful_for_coil = True
            except KeyError as e:
                # print(f"Missing data for coil {coil_idx} in ODS {key} for {title_suffix}: {e}") # Optional: more detailed logging
                continue
        
        if plot_successful_for_coil: # Only set labels if something was plotted
            ax.set_ylabel(f"{coil_name if 'coil_name' in locals() and coil_name else f'Coil {coil_idx}'} {yunit_label}")
            if coil_idx == coil_indices[0]: # Title and legend for the first subplot
                ax.set_title(f'PF Active Time - {title_suffix}')
                ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f'No data for coil {coil_idx}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_yticks([]) # Remove y-ticks if no data

    # Common X-axis label for the last subplot if plots were made
    if any(ax.lines for ax in axs): # Check if any axis has lines plotted
        axs[-1].set_xlabel(f'Time [{xunit}]')
        if xlim_processed is not None:
            plt.xlim(xlim_processed) # Apply xlim to the shared x-axis
    
    plt.tight_layout()
    plt.show()

def time_pf_active_current(odc_or_ods, indices='used', label='shot', xunit='s', yunit='kA', xlim='plasma'):
    """
    Plot PF coil currents in n x 1 subplots.
    """
    def data_retriever(ods, coil_idx, xunit_val):
        time = ods['pf_active.time']
        if xunit_val == 'ms':
            time = time * 1e3
        current = ods[f'pf_active.coil.{coil_idx}.current.data']
        name = ods[f'pf_active.coil.{coil_idx}.name']
        if yunit == 'MA':
            current = current / 1e6
        elif yunit == 'kA':
            current = current / 1e3
        return time, current, name

    _plot_pf_active_time_generic(odc_or_ods, indices, label, xunit, f'Current [{yunit}]', 'Current',
                            data_retriever, xlim)
pf_active_time_current = time_pf_active_current


def time_pf_active_current_turns(odc_or_ods, indices='used', label='shot', xunit='s', yunit='kA_T', xlim='plasma'):
    """
    Plot PF coil currents multiplied by turns in n x 1 subplots.
    """
    def data_retriever_turns(ods, coil_idx, xunit_val):
        time = ods['pf_active.time']
        if xunit_val == 'ms':
            time = time * 1e3
        
        current = ods[f'pf_active.coil.{coil_idx}.current.data']
        name = ods[f'pf_active.coil.{coil_idx}.name'] # Get name for ylabel consistency
        turns = np.sum(np.abs(ods[f'pf_active.coil.{coil_idx}.element.:.turns_with_sign']))
        
        val = current * turns
        if yunit == 'MA_T':
            val = val / 1e6
        elif yunit == 'kA_T':
            val = val / 1e3
        # Default is A_T, no conversion needed
        return time, val, name

    _plot_pf_active_time_generic(odc_or_ods, indices, label, xunit, f'[{yunit}]', 'Current-Turns',
                            data_retriever_turns, xlim)
pf_active_time_current_turns = time_pf_active_current_turns


"""
magnetics - ip, Rogowski coil[:Raw plasma current], diamagnetic_flux, Flux loop (flux, voltage), Bpol_probe (field, voltage, spectrogram)
"""
def time_magnetics_ip(odc_or_ods, label='shot', xunit='s', yunit='MA', xlim='plasma'):
    """
    Plot plasma current (Ip) time series.
    
    Parameters:
        odc_or_ods: ODS or ODC
            Input data containing plasma current measurements
        label: str
            Legend label option ('shot', 'key', 'run' or custom list)
        xunit: str
            Time unit ('s', 'ms')
        yunit: str
            Current unit ('A', 'kA', 'MA')
        xlim: str or list
            The x-axis limits. Can be 'plasma', 'coil', 'none', or a list of two floats.
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    xlim_processed = handle_xlim(odc_or_ods, xlim)

    # Handle labels
    labels = handle_labels(odc, label)

    plt.figure(figsize=(6, 2.5))
    
    for key, lbl in zip(odc.keys(), labels):
        try:
            # Get and convert time data
            time = odc[key]['magnetics.ip.0.time']
            if xunit == 'ms':
                time = time * 1e3
                
            # Get and convert current data
            current = odc[key]['magnetics.ip.0.data']
            if yunit == 'kA':
                current = current / 1e3
            elif yunit == 'MA':
                current = current / 1e6
                
            plt.plot(time, current, label=lbl)
            
        except KeyError as e:
            print(f"Missing IP data in {key}: {e}")
            continue

    plt.xlabel(f'Time [{xunit}]')
    plt.ylabel(f'Plasma Current [{yunit}]')
    plt.grid(True)
    plt.legend()

    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()
magnetics_time_ip = time_magnetics_ip


def time_magnetics_diamagnetic_flux(ods_or_odc, label='shot', xunit='s', yunit='Wb', xlim='plasma'):
    """
    Plot diamagnetic flux time series.
    
    Parameters:
        ods_or_odc: ODS or ODC
            The input data. Can be a single ODS or a collection of ODS objects (ODC).
        label: str
            The option for the legend. Can be 'shot', 'key', 'run', or a list of labels.
        xunit: str
            The unit of the x-axis. Can be 's', 'ms', or 'us'.
        yunit: str
            The unit of the y-axis. Typically 'Wb' for Weber.
        xlim: str or list
            The x-axis limits. Can be 'plasma', 'coil', 'none', or a list of two floats.
    """
    odc = odc_or_ods_check(ods_or_odc)
    
    # Handle xlim
    xlim_processed = handle_xlim(ods_or_odc, xlim)

    # Handle labels
    labels = handle_labels(odc, label)

    # Determine if multiple diamagnetic_flux entries exist
    # Assuming only one diamagnetic_flux entry per ODS

    # Create subplots (single plot)
    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Plot each ODS's diamagnetic_flux
    for key, lbl in zip(odc.keys(), labels):
        ods = odc[key]
        try:
            # Get time data and convert units
            time = ods['magnetics.time']
            if xunit == 'ms':
                time = time * 1e3

            # Get diamagnetic_flux data and convert units if necessary
            flux = ods['magnetics.diamagnetic_flux.0.data']
            data = flux  # Adjust if yunit requires conversion
            if abs(min(flux)) > abs(max(flux)):
                data = -data

            ax.plot(time, data, label=lbl)
        except KeyError as e:
            print(f"Missing diamagnetic_flux data in {key}: {e}")
            continue

    ax.set_ylabel(f'Diamagnetic Flux [{yunit}]')
    ax.set_xlabel(f'Time [{xunit}]')
    ax.legend()
    ax.grid(True)
    if xlim_processed is not None:
        ax.set_xlim(xlim_processed)

    plt.tight_layout()
    plt.show()


magnetics_time_diamagnetic_flux = time_magnetics_diamagnetic_flux


def time_diamagnetic_flux(ods_or_odc, label='shot', xunit='s', yunit='Wb', xlim='plasma'):
    """
    Plot diamagnetic flux time series: magnetics (raw), equilibrium measured, and equilibrium reconstructed.

    Overlays on one axes:
    - magnetics.diamagnetic_flux.0.data at magnetics.time (measured from diagnostic)
    - equilibrium.time_slice[:].constraints.diamagnetic_flux.measured (interpolated at eq times)
    - equilibrium.time_slice[:].constraints.diamagnetic_flux.reconstructed (from compute_reconstructed_diamagnetic_flux)

    Parameters
    ----------
    ods_or_odc : ODS or ODC
        Input data. Should have magnetics and equilibrium (with constraints updated by
        update_equilibrium_constraints_diamagnetic_flux) for full overlay.
    label : str
        Legend option: 'shot', 'key', 'run', or list of labels.
    xunit : str
        Time axis unit: 's', 'ms'.
    yunit : str
        Y-axis unit, e.g. 'Wb'.
    xlim : str or list
        X limits: 'plasma', 'coil', 'none', or [t_min, t_max].
    """
    odc = odc_or_ods_check(ods_or_odc)
    xlim_processed = handle_xlim(ods_or_odc, xlim)
    labels = handle_labels(odc, label)

    def _needs_diamagnetic_flux_update(ods):
        """True if equilibrium has time_slice but constraints.diamagnetic_flux is missing."""
        if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
            return False
        ts0 = ods['equilibrium.time_slice'][0]
        return (
            'constraints' not in ts0
            or 'diamagnetic_flux' not in ts0.get('constraints', {})
            or 'reconstructed' not in ts0.get('constraints', {}).get('diamagnetic_flux', {})
        )

    fig, ax = plt.subplots(figsize=(8, 3.5))

    for key, lbl in zip(odc.keys(), labels):
        ods = odc[key]
        if _needs_diamagnetic_flux_update(ods):
            try:
                from vaft.omas.update import update_equilibrium_constraints_diamagnetic_flux
                update_equilibrium_constraints_diamagnetic_flux(ods, time_slice=None)
            except Exception as e:
                pass  # continue plotting without eq constraints

        time_scale = 1e3 if xunit == 'ms' else 1.0

        # 1) Magnetics: raw diamagnetic flux
        try:
            t_mag = np.asarray(ods['magnetics.time'], float) * time_scale
            flux_mag = np.asarray(ods['magnetics.diamagnetic_flux.0.data'], float)
            if abs(np.nanmin(flux_mag)) > abs(np.nanmax(flux_mag)):
                flux_mag = -flux_mag
            ax.plot(t_mag, flux_mag, label=f'{lbl} (magnetics)', alpha=0.9)
        except (KeyError, TypeError) as e:
            pass  # magnetics optional

        # Helper: get equilibrium time array (from equilibrium.time or time_slice[*].time)
        def _eq_time(ods):
            if 'equilibrium.time' in ods:
                t = ods['equilibrium.time']
                t = np.atleast_1d(np.asarray(t, float))
                return t
            n = len(ods.get('equilibrium.time_slice', []))
            if n == 0:
                return np.array([])
            t = np.array([
                float(ods['equilibrium.time_slice'][i].get('time', np.nan))
                for i in range(n)
            ])
            return t

        # 2) Equilibrium: measured (interpolated at eq times)
        try:
            t_eq = _eq_time(ods)
            t_eq = t_eq * time_scale
            flux_meas = ods['equilibrium.time_slice.:.constraints.diamagnetic_flux.measured']
            flux_meas = np.atleast_1d(np.asarray(flux_meas, float))
            if t_eq.size and flux_meas.size and t_eq.size == flux_meas.size and np.any(np.isfinite(t_eq)):
                ax.plot(t_eq, flux_meas, 'o-', label=f'{lbl} (eq measured)', markersize=4)
        except (KeyError, TypeError):
            pass

        # 3) Equilibrium: reconstructed
        try:
            t_eq = _eq_time(ods)
            t_eq = t_eq * time_scale
            flux_recon = ods['equilibrium.time_slice.:.constraints.diamagnetic_flux.reconstructed']
            flux_recon = np.atleast_1d(np.asarray(flux_recon, float))
            if t_eq.size and flux_recon.size and t_eq.size == flux_recon.size and np.any(np.isfinite(t_eq)):
                ax.plot(t_eq, flux_recon, 's-', label=f'{lbl} (eq reconstructed)', markersize=4)
        except (KeyError, TypeError):
            pass

    ax.set_xlabel(f'Time [{xunit}]')
    ax.set_ylabel(f'Diamagnetic flux [{yunit}]')
    ax.set_title('Diamagnetic flux: magnetics, equilibrium measured, equilibrium reconstructed')
    ax.legend(loc='best')
    ax.grid(True)
    if xlim_processed is not None:
        ax.set_xlim(xlim_processed)
    plt.tight_layout()
    plt.show()
    return fig, ax


def time_impurity_effect(odc_or_ods, label='shot', xunit='s', xlim='plasma',
                         impurity_lines=None):
    """
    Combined 3x2 plot: Ip / Halpha, diamagnetic flux / CIII, inboard-midplane V_loop / OII.
    For checking plasma and impurity state in one view.
    Layout:
      Row 1: Ip, Halpha
      Row 2: Delta phi_dia, CIII
      Row 3: V_loop (inboard midplane flux loop), OII
    """
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim)
    labels = handle_labels(odc, label)

    # Fixed 3x2 layout: Ip, H_alpha | dia, CIII | vloop, OII
    fig, axs = plt.subplots(3, 2, figsize=(8, 4.5), sharex=True)

    # (0,0): Ip
    for key, lbl in zip(odc.keys(), labels):
        try:
            time = odc[key]['magnetics.ip.0.time'].copy()
            current = odc[key]['magnetics.ip.0.data'] / 1e3  # kA
            if xunit == 'ms':
                time = time * 1e3
            axs[0, 0].plot(time, current, label=lbl)
        except KeyError:
            continue
    axs[0, 0].set_ylabel(r'$I_p$ [kA]')
    axs[0, 0].grid(True)

    # (0,1): Halpha
    _plot_spectrometer_line_into_ax(odc, labels, axs[0, 1], 'H_alpha', xunit)
    axs[0, 1].set_ylabel(r'$\mathrm{H}_\alpha$ [a.u.]')
    axs[0, 1].grid(True)

    # (1,0): Diamagnetic flux
    for key, lbl in zip(odc.keys(), labels):
        try:
            time = odc[key]['magnetics.time'].copy()
            flux = odc[key]['magnetics.diamagnetic_flux.0.data']
            if np.abs(np.min(flux)) > np.abs(np.max(flux)):
                flux = -flux
            if xunit == 'ms':
                time = time * 1e3
            axs[1, 0].plot(time, flux, label=lbl)
        except KeyError:
            continue
    axs[1, 0].set_ylabel(r'$\Delta\phi_{\mathrm{dia}}$ [Wb]')
    axs[1, 0].grid(True)

    # (1,1): CIII
    _plot_spectrometer_line_into_ax(odc, labels, axs[1, 1], 'CIII', xunit)
    axs[1, 1].set_ylabel(r'$\mathrm{C}$-III [a.u.]')
    axs[1, 1].grid(True)

    # (2,0): V_loop (inboard midplane flux loop voltage)
    for key, lbl in zip(odc.keys(), labels):
        try:
            ods = odc[key]
            idx_arr = _find_flux_loop_inboard_midplane_indices(ods)
            if idx_arr is None or len(idx_arr[0]) == 0:
                continue
            item_idx = int(idx_arr[0][0])
            if 'magnetics.flux_loop.time' in ods:
                time = np.asarray(ods['magnetics.flux_loop.time']).copy()
            elif 'magnetics.time' in ods:
                time = np.asarray(ods['magnetics.time']).copy()
            else:
                continue
            flux = np.asarray(ods[f'magnetics.flux_loop.{item_idx}.flux.data']).copy()
            voltage = -np.gradient(flux, time)
            if xunit == 'ms':
                time = time * 1e3
            axs[2, 0].plot(time, voltage, label=lbl)
        except (KeyError, TypeError, IndexError):
            continue
        
    axs[2, 0].set_ylabel(r'$V_{\mathrm{loop}}^{\mathrm{FL,InMid}}$ [V]')
    axs[2, 0].grid(True)

    # (2,1): OII
    _plot_spectrometer_line_into_ax(odc, labels, axs[2, 1], 'OII', xunit)
    axs[2, 1].set_ylabel(r'$\mathrm{O}$-II [a.u.]')
    axs[2, 1].grid(True)

    # Single legend at top of figure (from first subplot that has lines)
    lines0 = axs[0, 0].get_lines()
    if lines0:
        handles = lines0
        leg_labels = [line.get_label() for line in lines0]
        fig.legend(handles, leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(handles), frameon=True)

    axs[2, 0].set_xlabel(f'Time [{xunit}]')
    axs[2, 1].set_xlabel(f'Time [{xunit}]')
    if xlim_processed is not None:
        for ax in axs.flat:
            ax.set_xlim(xlim_processed)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def _plot_spectrometer_line_into_ax(odc, labels, ax, line_name, xunit):
    """Plot a single spectrometer line into the given axes. Leaves ax unchanged if no data."""
    if line_name not in SPECTROMETER_LINE_MAP:
        ax.text(0.5, 0.5, f'No line {line_name}', ha='center', va='center', transform=ax.transAxes)
        ax.set_yticks([])
        return
    channel, line_idx = SPECTROMETER_LINE_MAP[line_name]
    plot_ok = False
    for key, lbl in zip(odc.keys(), labels):
        try:
            time_data = odc[key]['spectrometer_uv.time'].copy()
            data_val = odc[key][f'spectrometer_uv.channel.{channel}.processed_line.{line_idx}.intensity.data']
            if xunit == 'ms':
                time_data = time_data * 1e3
            ax.plot(time_data, data_val, label=lbl)
            plot_ok = True
        except KeyError:
            continue
    if not plot_ok:
        ax.text(0.5, 0.5, f'No data for {line_name}', ha='center', va='center', transform=ax.transAxes)
        ax.set_yticks([])


# In the VEST, the flux loop is classified as 'inboard', and 'outboard'
# 'inboard' is the flux loops located in the inboard (HF) side of vessel
# 'outboard' is the flux loops located in the outboard (LF) side of vessel

def _determine_magnetics_indices(odc, indices_param, ods_path_prefix, find_funcs):
    """Helper to determine indices for magnetics flux_loop and b_pol_probe plotting."""
    # `find_funcs` is a dict mapping index type (e.g., 'inboard') to a function
    # that returns indices for that type from an ODS.

    if indices_param == 'all':
        item_indices = set()
        for key in odc.keys():
            ods = odc[key]
            try:
                items_container = ods[ods_path_prefix]
                num_items = len(items_container)
                item_indices.update(range(num_items))
            except (KeyError, TypeError):
                continue
        return sorted(list(item_indices))
    elif indices_param in find_funcs: # e.g., 'inboard', 'outboard', 'side'
        item_indices = set()
        for key in odc.keys():
            ods = odc[key]
            try:
                found_indices_tuple = find_funcs[indices_param](ods)
                # find_funcs like _find_flux_loop_inboard_indices return a tuple (array([...]),)
                if found_indices_tuple is not None and len(found_indices_tuple) > 0 and hasattr(found_indices_tuple[0], '__iter__'):
                    item_indices.update(found_indices_tuple[0])
            except (KeyError, TypeError, AttributeError):
                continue
        return sorted(list(item_indices))
    elif isinstance(indices_param, int):
        return [indices_param]
    elif isinstance(indices_param, list):
        # Ensure all elements are integers if it's a list
        if all(isinstance(i, int) for i in indices_param):
            return sorted(list(set(indices_param))) # Deduplicate and sort
        else:
            raise ValueError("If indices is a list, it must contain only integers.")
    else:
        raise ValueError(
            f"indices must be 'all', one of {list(find_funcs.keys())}, an integer, or a list of integers; got {indices_param!r}"
        )

# (ods_path_prefix, indices_param) -> (nrows, ncols) for m x n grid
MAGNETICS_PLOT_LAYOUT = {
    ('magnetics.flux_loop', 'all'): (3, 4),
    ('magnetics.flux_loop', 'inboard'): (2, 4),
    ('magnetics.flux_loop', 'outboard'): (2, 2),
    ('magnetics.flux_loop', 'inboard_midplane'): (1, 1),
    ('magnetics.b_field_pol_probe', 'all'): (8, 8),
    ('magnetics.b_field_pol_probe', 'inboard'): (4, 7),
    ('magnetics.b_field_pol_probe', 'side'): (4, 4),
    ('magnetics.b_field_pol_probe', 'outboard'): (3, 7),
}

def _get_magnetics_position_r_z(ods, ods_path_prefix, item_idx):
    """Return (r, z) in meters for flux_loop or b_field_pol_probe item. Scalars for display."""
    try:
        if 'flux_loop' in ods_path_prefix:
            r_path = f'{ods_path_prefix}.{item_idx}.position.0.r'
            z_path = f'{ods_path_prefix}.{item_idx}.position.0.z'
        else:
            r_path = f'{ods_path_prefix}.{item_idx}.position.r'
            z_path = f'{ods_path_prefix}.{item_idx}.position.z'
        r = np.atleast_1d(ods[r_path]).flat[0]
        z = np.atleast_1d(ods[z_path]).flat[0]
        return float(r), float(z)
    except (KeyError, TypeError, IndexError):
        return None, None

def _plot_magnetics_time_subplot_generic(odc_or_ods, indices_param, label_param, xunit, yunit,
                                    ods_path_prefix, time_path_suffix, data_path_suffix,
                                    title_base, ylabel_base, find_funcs, xlim_param,
                                    data_transform=None):
    """Generic plotting function for magnetics data (flux_loop, b_pol_probe) with m x n subplots.
    data_transform: optional callable(time_1d, data_1d) -> transformed_data_1d (e.g. flux -> voltage).
    """
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim_param)
    labels = handle_labels(odc, label_param)

    item_indices = _determine_magnetics_indices(odc, indices_param, ods_path_prefix, find_funcs)

    if not item_indices:
        print(f"No valid {ylabel_base.lower()} found to plot for indices: {indices_param}")
        return

    layout_key = (ods_path_prefix, indices_param)
    nrows, ncols = MAGNETICS_PLOT_LAYOUT.get(layout_key, (len(item_indices), 1))
    n_axes = nrows * ncols
    figsize = (2.5 * ncols, 1.8 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, squeeze=False)
    axs_flat = axs.flatten()

    any_plot_made = False
    first_legend_handles, first_legend_labels = None, None
    for ax_idx, item_idx_val in enumerate(item_indices):
        if ax_idx >= n_axes:
            break
        ax = axs_flat[ax_idx]
        plot_successful_for_item = False
        legend_handles_labels_for_ax = {}
        r_display, z_display = None, None

        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                full_time_path = f'{ods_path_prefix}.{time_path_suffix}'
                full_data_path = f'{ods_path_prefix}.{item_idx_val}.{data_path_suffix}'

                if full_data_path not in ods:
                    continue
                if full_time_path in ods:
                    time_data = ods[full_time_path].copy()
                elif ods_path_prefix.startswith('magnetics.') and 'magnetics.time' in ods:
                    time_data = ods['magnetics.time'].copy()
                else:
                    continue
                data_val = np.asarray(ods[full_data_path]).copy()
                if data_transform is not None:
                    time_for_transform = np.asarray(time_data)
                    data_val = data_transform(time_for_transform, data_val)

                if r_display is None and z_display is None:
                    r_display, z_display = _get_magnetics_position_r_z(ods, ods_path_prefix, item_idx_val)

                if xunit == 'ms':
                    time_data = time_data * 1e3

                line, = ax.plot(time_data, data_val, label=lbl)
                if lbl not in legend_handles_labels_for_ax:
                    legend_handles_labels_for_ax[lbl] = line
                plot_successful_for_item = True
                any_plot_made = True
            except (KeyError, Exception):
                continue

        if ax_idx == 0 and legend_handles_labels_for_ax:
            first_legend_handles = list(legend_handles_labels_for_ax.values())
            first_legend_labels = list(legend_handles_labels_for_ax.keys())

        if plot_successful_for_item:
            r_str = f'{r_display:.3f}' if r_display is not None else '?'
            z_str = f'{z_display:.3f}' if z_display is not None else '?'
            if ax_idx % ncols == 0:
                ax.set_ylabel(f'[{yunit}]')
            ax.set_title(f'[{item_idx_val}] ({r_str}m, {z_str}m)')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f'No data for {ylabel_base.lower()} {item_idx_val}',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_yticks([])

    for j in range(len(item_indices), n_axes):
        axs_flat[j].set_visible(False)

    if any_plot_made:
        # xlabel on the bottom-most subplot in each column (so when below is empty, the one above shows it)
        for col in range(ncols):
            k_last = (len(item_indices) - 1 - col) // ncols
            idx_bottom = k_last * ncols + col
            if idx_bottom < len(item_indices):
                axs_flat[idx_bottom].set_xlabel(f'Time [{xunit}]')
        if xlim_processed is not None:
            axs_flat[0].set_xlim(xlim_processed)
        # Legend outside figure (right side) so it does not overlap subplots
        if first_legend_handles and first_legend_labels:
            fig.legend(first_legend_handles, first_legend_labels, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    else:
        fig.text(0.5, 0.5, 'No data found for any selected items.',
                 horizontalalignment='center', verticalalignment='center')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

def _find_flux_loop_all_indices(ods):
    # find the indices of all flux loop
    indices = np.arange(len(ods['magnetics.flux_loop']))
    return indices

def _find_flux_loop_inboard_indices(ods):
    # find the indices of inboard flux loop in VEST
    indices = np.where(ods['magnetics.flux_loop.:.position.0.r'] < 0.15)
    return indices

def _find_flux_loop_outboard_indices(ods):
    # find the indices of the flux loop outboard
    indices = np.where(ods['magnetics.flux_loop.:.position.0.r'] > 0.5)
    return indices

def _find_flux_loop_all_indices(ods):
    # find the indices of all flux loop
    return np.arange(len(ods['magnetics.flux_loop']))

def _find_flux_loop_inboard_midplane_indices(ods):
    # find the indices of the flux loop inboard midplane
    indices = np.where(ods['magnetics.flux_loop.:.position.0.r'] == 0.091)
    return indices

def time_magnetics_flux_loop_flux(ods_or_odc, indices='all', label='shot', xunit='s', yunit='Wb', xlim='plasma'):
    """
    Plot flux loop flux time series.
    """
    find_funcs = {
        'all': _find_flux_loop_all_indices,
        'inboard': _find_flux_loop_inboard_indices,
        'outboard': _find_flux_loop_outboard_indices,
        'inboard_midplane': _find_flux_loop_inboard_midplane_indices,
    }
    _plot_magnetics_time_subplot_generic(ods_or_odc, indices, label, xunit, yunit,
                                    ods_path_prefix='magnetics.flux_loop',
                                    time_path_suffix='time',
                                    data_path_suffix='flux.data',
                                    title_base='Flux Loop Flux',
                                    ylabel_base='Flux',
                                    find_funcs=find_funcs,
                                    xlim_param=xlim)
magnetics_time_flux_loop_flux = time_magnetics_flux_loop_flux

def time_magnetics_flux_loop_voltage(ods_or_odc, indices='all', label='shot', xunit='s', yunit='V', xlim='plasma'):
    """
    Plot flux loop voltage time series. Uses loop_voltage = -d(flux)/dt from flux loop flux data.
    """
    def _flux_to_voltage(time_s, flux):
        return -np.gradient(flux, time_s)

    find_funcs = {
        'all': _find_flux_loop_all_indices,
        'inboard': _find_flux_loop_inboard_indices,
        'outboard': _find_flux_loop_outboard_indices,
        'inboard_midplane': _find_flux_loop_inboard_midplane_indices,
    }
    _plot_magnetics_time_subplot_generic(ods_or_odc, indices, label, xunit, yunit,
                                    ods_path_prefix='magnetics.flux_loop',
                                    time_path_suffix='time',
                                    data_path_suffix='flux.data',
                                    title_base='Flux Loop Voltage',
                                    ylabel_base='Voltage',
                                    find_funcs=find_funcs,
                                    xlim_param=xlim,
                                    data_transform=_flux_to_voltage)
magnetics_time_flux_loop_voltage = time_magnetics_flux_loop_voltage


# bpol probe
# indices -> 'all', 'inboard', 'outboard', 'side'
# VEST classifies the bpol probe as 'inboard', 'side', and 'outboard'
# 'inboard' probes are located in the inboard (HF) midplane side of vessel
# 'side' probes are located in the inboard (HF) upper and lower coner side of vessel
# 'outboard' probes are located in the outboard (LF) side of vessel

def _find_bpol_probe_inboard_indices(ods):
    # find the indices of the bpol probe inboard
    indices = np.where(ods['magnetics.b_field_pol_probe.:.position.r'] < 0.09)
    return indices

def _find_bpol_probe_outboard_indices(ods):
    # find the indices of the bpol probe outboard
    indices = np.where(ods['magnetics.b_field_pol_probe.:.position.r'] > 0.795)
    return indices

def _find_bpol_probe_side_indices(ods):
    # find the indices of the bpol probe side
    indices = np.where(np.abs(ods['magnetics.b_field_pol_probe.:.position.z']) > 0.8)
    return indices

def _find_bpol_probe_all_indices(ods):
    # find the indices of all bpol probe
    return np.arange(len(ods['magnetics.b_field_pol_probe']))

def time_magnetics_b_field_pol_probe_field(ods_or_odc, indices='all', label='shot', xunit='s', yunit='T', xlim='plasma'):
    """
    Plot B-field time series from B-field poloidal probes.
    
    Parameters:
        ods_or_odc: ODS or ODC
            The input data. Can be a single ODS or a collection of ODS objects (ODC).
        indices: str or list of int
            The B-pol probe indices to plot. Can be 'all', 'inboard', 'outboard', 'side', or a list of indices.
        label: str
            The option for the legend. Can be 'shot', 'key', 'run', or a list of labels.
        xunit: str
            The unit of the x-axis. Can be 's', 'ms', or 'us'.
        yunit: str
            The unit of the y-axis. Typically 'T' for Tesla.
        xlim: str or list
            The x-axis limits. Can be 'plasma', 'coil', 'none', or a list of two floats.
    """
    find_funcs = {
        'inboard': _find_bpol_probe_inboard_indices,
        'outboard': _find_bpol_probe_outboard_indices,
        'side': _find_bpol_probe_side_indices
    }
    _plot_magnetics_time_subplot_generic(ods_or_odc, indices, label, xunit, yunit,
                                    ods_path_prefix='magnetics.b_field_pol_probe',
                                    time_path_suffix='time',
                                    data_path_suffix='field.data',
                                    title_base='B-field Poloidal Probe',
                                    ylabel_base='B-field',
                                    find_funcs=find_funcs,
                                    xlim_param=xlim)
magnetics_time_b_field_pol_probe_field = time_magnetics_b_field_pol_probe_field


"""
Equilibrium plotting functions
"""

def _plot_equilibrium_time_quantity(odc_or_ods, quantity_key, ylabel, title_suffix, label='shot', xunit='s', yunit=None, xlim='plasma'):
    """Helper function to plot generic equilibrium quantities."""
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim)
    labels = handle_labels(odc, label)

    plt.figure(figsize=(6, 2.5))
    for key, lbl in zip(odc.keys(), labels):
        ods = odc[key]
        try:
            time_data = ods['equilibrium.time']
            quantity_data = ods[f'equilibrium.time_slice.:.global_quantities.{quantity_key}']
            
            if xunit == 'ms':
                time_data = time_data * 1e3
            
            if yunit:
                if yunit == 'kA':
                    quantity_data = quantity_data / 1e3
                elif yunit == 'MA':
                    quantity_data = quantity_data / 1e6
            
            plt.plot(time_data, quantity_data, label=lbl)
        except Exception as e:
            print(f"Missing equilibrium {quantity_key} in {key}: {e}")
            continue
            
    plt.xlabel(f'Time [{xunit}]')
    plt.ylabel(ylabel)
    plt.title(f'Equilibrium {title_suffix}')
    plt.grid(True)
    plt.legend()
    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()


def time_equilibrium_plasma_current(odc_or_ods, label='shot', xunit='s', yunit='MA', xlim='plasma'):
    """
    Plot equilibrium plasma current (Ip) time series from equilibrium global_quantities.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'ip', f'Plasma Current [{yunit}]', 'Plasma Current', 
                             label=label, xunit=xunit, yunit=yunit, xlim=xlim)


def time_equilibrium_li(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium internal inductance (li_3) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'li_3', 'Internal Inductance [li_3]', 'Internal Inductance (li_3)',
                             label=label, xunit=xunit, yunit=None, xlim=xlim) # yunit is None as it's unitless


def time_equilibrium_beta_pol(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium poloidal beta (beta_pol) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_pol', 'Poloidal Beta [beta_pol]', 'Poloidal Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def time_equilibrium_beta_tor(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium toroidal beta (beta_tor) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_tor', 'Toroidal Beta [beta_tor]', 'Toroidal Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def time_equilibrium_beta_n(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium normalized beta (beta_n) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_normal', 'Normalized Beta [beta_n]', 'Normalized Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def time_equilibrium_w_mhd(odc_or_ods, label='shot', xunit='s', yunit='J', xlim='plasma'):
    """
    Plot equilibrium MHD stored energy (w_mhd) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'energy_mhd', 'MHD Stored Energy [J]', 'MHD Stored Energy',
                             label=label, xunit=xunit, yunit=None, xlim=xlim) # yunit conversion is not needed for J


def time_equilibrium_w_mag(odc_or_ods, label='shot', xunit='s', yunit='J', xlim='plasma'):
    """
    Plot equilibrium magnetic stored energy (w_mag) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'energy_mag', 'Magnetic Stored Energy [J]', 'Magnetic Stored Energy',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def time_equilibrium_w_tot(odc_or_ods, label='shot', xunit='s', yunit='J', xlim='plasma'):
    """
    Plot equilibrium total stored energy (w_tot) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'energy_total', 'Total Stored Energy [J]', 'Total Stored Energy',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def time_equilibrium_q0(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium q-axis (q0) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'q_axis', 'q-axis [q0]', 'q-axis (q0)',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def time_equilibrium_q95(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium q95 time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'q_95', 'q95', 'q95',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def time_equilibrium_qa(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium qa time series (if available).
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'qa', 'qa', 'qa',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)

# SHAPE QUANTITIES

def time_equilibrium_major_radius(odc_or_ods, label='shot', xunit='s', yunit='m', xlim='plasma'):
    """
    Plot equilibrium major radius (geometric_axis.r) time series.
    """
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim)
    labels = handle_labels(odc, label)
    plt.figure(figsize=(6, 2.5))
    for key, lbl in zip(odc.keys(), labels):
        ods = odc[key]
        try:
            time = ods['equilibrium.time']
            # Note: Path for major radius is different
            majr = ods['equilibrium.time_slice.:.boundary.geometric_axis.r'] 
            if xunit == 'ms':
                time = time * 1e3
            plt.plot(time, majr, label=lbl)
        except Exception as e:
            print(f"Missing equilibrium major radius in {key}: {e}")
            continue
    plt.xlabel(f'Time [{xunit}]')
    plt.ylabel(f'Major Radius [{yunit}]')
    plt.title('Equilibrium Major Radius')
    plt.grid(True)
    plt.legend()
    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()

equilibrium_time_plasma_current = time_equilibrium_plasma_current
equilibrium_time_li = time_equilibrium_li
equilibrium_time_beta_pol = time_equilibrium_beta_pol
equilibrium_time_beta_tor = time_equilibrium_beta_tor
equilibrium_time_beta_n = time_equilibrium_beta_n
equilibrium_time_w_mhd = time_equilibrium_w_mhd
equilibrium_time_w_mag = time_equilibrium_w_mag
equilibrium_time_w_tot = time_equilibrium_w_tot
equilibrium_time_q0 = time_equilibrium_q0
equilibrium_time_q95 = time_equilibrium_q95
equilibrium_time_qa = time_equilibrium_qa
equilibrium_time_major_radius = time_equilibrium_major_radius

"""
spectrometer_uv (filterscope)
VEST: channel 0 = Slow DaQ, channel 1 = Fast DaQ.
Slow: line 0 H-alpha_6563, line 1 OI_7770.
Fast: 0 H-alpha_6563, 1 H-beta_4861, 2 H-gamma_4340, 3 CII_3726, 4 CIII_1909, 5 OII_3726, 6 OV_629.
"""
# line_name -> (channel, line_idx). Order for 'all': slow then fast.
SPECTROMETER_LINE_MAP = {
    'Slow_H_alpha': (0, 0),
    'OI': (0, 1),
    'H_alpha': (1, 0),
    'H_beta': (1, 1),
    'H_gamma': (1, 2),
    'CII': (1, 3),
    'CIII': (1, 4),
    'OII': (1, 5),
    'OV': (1, 6),
}
SPECTROMETER_ALL_LINES_ORDER = [
    'Slow_H_alpha', 'OI', 'H_alpha', 'H_beta', 'H_gamma', 'CII', 'CIII', 'OII', 'OV',
]
SPECTROMETER_FAST_LINES = ['H_alpha', 'H_beta', 'H_gamma', 'CII', 'CIII', 'OII', 'OV']
SPECTROMETER_SLOW_LINES = ['Slow_H_alpha', 'OI']
SPECTROMETER_MAIN_LINES = ['H_alpha', 'CIII', 'OII']  # fast: H-alpha, C-III, OII


def _determine_spectrometer_lines(indices_param, line_map):
    """
    Return list of line names to plot.
    indices_param: 'all' | 'fast' | 'slow' | 'main' | 'H_alpha' | 'OI' | 'CII' | 'CIII' | 'OII' | 'OV' | 'H_beta' | 'H_gamma' | list
    """
    if indices_param == 'all':
        return [name for name in SPECTROMETER_ALL_LINES_ORDER if name in line_map]
    if indices_param == 'fast':
        return [name for name in SPECTROMETER_FAST_LINES if name in line_map]
    if indices_param == 'slow':
        return [name for name in SPECTROMETER_SLOW_LINES if name in line_map]
    if indices_param == 'main':
        return [name for name in SPECTROMETER_MAIN_LINES if name in line_map]
    if isinstance(indices_param, list):
        return [name for name in indices_param if name in line_map]
    if indices_param in line_map:
        return [indices_param]
    return []


def _plot_spectrometer_subplot_generic(odc_or_ods, indices_param, label_param, xunit, yunit, 
                                       line_map, xlim_param):
    """Generic plotting function for spectrometer data with subplots."""
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim_param)
    labels = handle_labels(odc, label_param)

    selected_lines = _determine_spectrometer_lines(indices_param, line_map)

    if not selected_lines:
        print("No valid spectral lines to plot")
        return

    nrows = len(selected_lines)
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 1.5 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    for ax, line_name in zip(axs, selected_lines):
        channel, line_idx = line_map[line_name]
        plot_successful_for_line = False
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                time_data = ods[f'spectrometer_uv.time']
                data_val = ods[f'spectrometer_uv.channel.{channel}.processed_line.{line_idx}.intensity.data']
                
                if xunit == 'ms':
                    time_data = time_data * 1e3
                
                ax.plot(time_data, data_val, label=lbl)
                plot_successful_for_line = True
            except KeyError as e:
                # print(f"Missing {line_name} data in ODS {key}: {e}")
                continue
        
        if plot_successful_for_line:
            ax.set_ylabel(f'[{yunit}]')
            ax.set_title(line_name.replace('_', '-'))
            ax.grid(True)
            if line_name == selected_lines[0]: # Legend for the first subplot
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {line_name}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_yticks([])

    if any(ax.lines for ax in axs):
        axs[-1].set_xlabel(f'Time [{xunit}]')
        if xlim_processed is not None:
            plt.xlim(xlim_processed)

    plt.tight_layout()
    plt.show()

def time_spectrometer_uv_intensity(odc_or_ods, indices='all', label='shot', xunit='s', yunit='a.u.', xlim='plasma'):
    """
    Plot UV spectrometer/filterscope intensity time series.

    indices: 'all' (all lines, slow then fast), 'fast' (fast channel only),
             'slow' (slow channel only), 'main' (fast: H-alpha, C-III, OII only),
             'H_alpha', 'OI', 'CII', 'CIII', 'OII', 'OV', 'H_beta', 'H_gamma', or list of names.
    """
    _plot_spectrometer_subplot_generic(odc_or_ods, indices, label, xunit, yunit, 
                                       SPECTROMETER_LINE_MAP, xlim)
spectrometer_uv_time_intensity = time_spectrometer_uv_intensity

"""
TF coil
"""
def time_tf_b_field_tor(odc_or_ods, label='shot', xunit='s', yunit='T', xlim='plasma'):
    """
    Plot vacuum toroidal magnetic field (B_tor) time series.
    
    Parameters:
        odc_or_ods: ODS or ODC
            Input data containing TF measurements
        label: str
            Legend label option ('shot', 'key', 'run' or custom list)
        xunit: str
            Time unit ('s', 'ms')
        yunit: str
            B-field unit ('T', 'mT')
        xlim: str or list
            X-axis limits setting
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    xlim_processed = handle_xlim(odc_or_ods, xlim)

    # Handle labels
    labels = handle_labels(odc, label)

    plt.figure(figsize=(6, 2.5))
    
    for key, lbl in zip(odc.keys(), labels):
        try:
            tf = odc[key]['tf']
            time = tf['time']
            
            if xunit == 'ms':
                time = time * 1e3
                
            b_field = tf['b_field_tor_vacuum_r.data'] / tf['r0']
            
            if yunit == 'mT':
                b_field *= 1e3
                
            plt.plot(time, b_field, label=lbl)
            
        except KeyError as e:
            print(f"Missing B_tor data in {key}: {e}")
            continue

    plt.xlabel(f'Time [{xunit}]')
    plt.ylabel(f'Toroidal Field [{yunit}]')
    plt.title('Vacuum Toroidal Magnetic Field')
    plt.grid(True)
    plt.legend()
    
    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()
tf_time_b_field_tor = time_tf_b_field_tor

def time_tf_b_field_tor_vacuum_r(odc_or_ods, label='shot', xunit='s', yunit='T·m', xlim='plasma'):
    """
    Plot vacuum R-component toroidal field (B_tor_vacuum_r) time series.
    
    Parameters:
        odc_or_ods: ODS or ODC
            Input data containing TF measurements
        label: str
            Legend label option
        xunit: str
            Time unit
        yunit: str
            Field unit ('T·m', 'mT·m')
        xlim: str/list
            X-axis limits
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    xlim_processed = handle_xlim(odc_or_ods, xlim)

    # Handle labels
    labels = handle_labels(odc, label)

    plt.figure(figsize=(6, 2.5))
    
    for key, lbl in zip(odc.keys(), labels):
        try:
            tf = odc[key]['tf']
            time = tf['time']
            
            if xunit == 'ms':
                time = time * 1e3
                
            b_field = tf['b_field_tor_vacuum_r.data']
            
            if yunit == 'mT·m':
                b_field *= 1e3
                
            plt.plot(time, b_field, label=lbl)
            
        except KeyError as e:
            print(f"Missing B_tor_vacuum_r data in {key}: {e}")
            continue

    plt.xlabel(f'Time [{xunit}]')
    plt.ylabel(f'R-component Field [{yunit}]')
    plt.title('Vacuum R-component Toroidal Field')
    plt.grid(True)
    plt.legend()
    
    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()
tf_time_b_field_tor_vacuum_r = time_tf_b_field_tor_vacuum_r

def time_tf_coil_current(odc_or_ods, label='shot', xunit='s', yunit='MA', xlim='plasma'):
    """
    Plot TF coil current time series.
    
    Parameters:
        odc_or_ods: ODS/ODC
            Input data
        label: str
            Legend labels
        xunit: str
            Time units
        yunit: str
            Current units ('A', 'kA', 'MA')
        xlim: str/list
            X-axis limits
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    xlim_processed = handle_xlim(odc_or_ods, xlim)
    # Handle labels
    labels = handle_labels(odc, label)

    plt.figure(figsize=(6, 2.5))
    
    for key, lbl in zip(odc.keys(), labels):
        try:
            tf = odc[key]['tf']
            time = tf['time']
            current = tf['coil.0.current.data']
            
            if xunit == 'ms':
                time = time * 1e3
                
            if yunit == 'kA':
                current /= 1e3
            elif yunit == 'MA':
                current /= 1e6
                
            plt.plot(time, current, label=lbl)
            
        except KeyError as e:
            print(f"Missing coil current in {key}: {e}")
            continue

    plt.xlabel(f'Time [{xunit}]')
    plt.ylabel(f'Coil Current [{yunit}]')
    plt.title('TF Coil Current')
    plt.grid(True)
    plt.legend()
    
    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()
tf_time_coil_current = time_tf_coil_current


"""
eddy_current (pf_passive)
"""

# def pf_passive_current




"""
barometry (Vacuum Gauge or Neutral Pressure Gauge)
"""

def time_barometry_pressure(odc_or_ods, label='shot', xunit='s', yunit='Pa', xlim='plasma'):
    """
    Plot neutral pressure time series from barometry gauges.
    
    Parameters:
        odc_or_ods: ODS or ODC
            Input data containing pressure measurements
        label: str
            Legend label option ('shot', 'key', 'run' or custom list)
        xunit: str
            Time unit ('s', 'ms')
        yunit: str
            Pressure unit ('Pa', 'kPa', 'mbar', 'Torr')
        xlim: str or list
            The x-axis limits. Can be 'plasma', 'coil', 'none', or a list of two floats.
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    xlim_processed = handle_xlim(odc_or_ods, xlim)

    # Handle labels
    labels = handle_labels(odc, label)

    plt.figure(figsize=(6, 2.5))
    
    for key, lbl in zip(odc.keys(), labels):
        try:
            # Get and convert time data
            time = odc[key]['barometry.gauge.0.pressure.time']
            if xunit == 'ms':
                time = time * 1e3
                
            # Get and convert pressure data
            pressure = odc[key]['barometry.gauge.0.pressure.data']
            if yunit == 'kPa':
                pressure = pressure / 1e3
            elif yunit == 'mbar':
                pressure = pressure / 100
            elif yunit == 'Torr':
                pressure = pressure / 133.322
                
            plt.plot(time, pressure, label=lbl)
            
        except KeyError as e:
            print(f"Missing pressure data in {key}: {e}")
            continue

    plt.xlabel(f'Time [{xunit}]')
    plt.ylabel(f'Neutral Pressure [{yunit}]')
    plt.title('Neutral Pressure Time Evolution')
    plt.grid(True)
    plt.legend()

    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()
barometry_time_pressure = time_barometry_pressure


"""
summary
"""


"""
global quantities
"""


"""
Not Routinely available signals
"""


"""
Thomson scattering
"""

# def time_thomson_scattering_density(odc_or_ods, label='shot', xunit='s', yunit='m^-3', xlim='plasma'):
#     """
#     Plot Thomson scattering electron density time series per channel.
    
#     Parameters:
#         odc_or_ods: ODS/ODC
#             Input data containing Thomson measurements
#         label: str
#             Legend labels option ('shot', 'key', 'run')
#         xunit: str
#             Time unit ('s', 'ms')
#         yunit: str
#             Density unit ('m^-3', 'cm^-3')
#         xlim: str/list
#             X-axis limits
#     """
#     odc = odc_or_ods_check(odc_or_ods)
    
#     # Handle xlim
#     if xlim == 'none':
#         xlim = None
#     elif xlim == 'plasma':
#         xlim = set_xlim_time(odc, type='plasma')
#     elif xlim == 'coil':
#         xlim = set_xlim_time(odc, type='coil')
#     elif isinstance(xlim, list) and len(xlim) == 2:
#         xlim = xlim
#     else:
#         print(f"Invalid xlim: {xlim}, using default 'plasma'")
#         xlim = set_xlim_time(odc, type='plasma')

#     # Handle labels
#     if isinstance(label, list) and len(label) == len(odc.keys()):
#         labels = label
#     else:
#         labels = extract_labels_from_odc(odc, opt=label)

#     # Determine channel count and radial positions from first ODS
#     first_key = next(iter(odc.keys()))
#     channels = list(odc[first_key]['thomson_scattering.channel'].keys())
#     n_channels = len(channels)
#     radial_positions = [odc[first_key][f'thomson_scattering.channel.{i}.position.r'] for i in range(n_channels)]

#     # Create subplots
#     fig, axs = plt.subplots(n_channels, 1, figsize=(10, 2.5*n_channels))
#     if n_channels == 1:
#         axs = [axs]
    
#     # Plot each channel in its own subplot
#     for ax, (channel, r_pos) in enumerate(zip(axs, radial_positions)):
#         for key, lbl in zip(odc.keys(), labels):
#             ods = odc[key]
#             try:
#                 time = ods['thomson_scattering.time']
#                 if xunit == 'ms':
#                     time = time * 1e3
                
#                 data = unumpy.nominal_values(ods[f'thomson_scattering.channel.{channel}.n_e.data'])
#                 err = unumpy.std_devs(ods[f'thomson_scattering.channel.{channel}.n_e.data'])
                
#                 if yunit == 'cm^-3':
#                     data = data / 1e6
#                     err = err / 1e6
                
#                 ax.errorbar(time, data, yerr=err, label=lbl)
                
#             except KeyError as e:
#                 print(f"Missing density data for channel {channel} in {key}: {e}")
#                 continue

#         ax.set_ylabel(f'n_e [{yunit}]')
#         ax.set_title(f'R = {r_pos:.3f} m')
#         ax.grid(True)
#         if channel == 0:
#             ax.legend()
#         if channel == n_channels-1:
#             ax.set_xlabel(f'Time [{xunit}]')
#         if xlim is not None:
#             ax.set_xlim(xlim)

#     plt.tight_layout()
#     plt.show()

# def time_thomson_scattering_temperature(odc_or_ods, label='shot', xunit='s', yunit='eV', xlim='plasma'):
#     """
#     Plot Thomson scattering electron temperature time series per channel.
    
#     Parameters:
#         odc_or_ods: ODS/ODC
#             Input data containing Thomson measurements
#         label: str
#             Legend labels option ('shot', 'key', 'run')
#         xunit: str
#             Time unit ('s', 'ms')
#         yunit: str
#             Temperature unit ('eV', 'keV')
#         xlim: str/list
#             X-axis limits
#     """
#     odc = odc_or_ods_check(odc_or_ods)
    
#     # Handle xlim
#     if xlim == 'none':
#         xlim = None
#     elif xlim == 'plasma':
#         xlim = set_xlim_time(odc, type='plasma')
#     elif xlim == 'coil':
#         xlim = set_xlim_time(odc, type='coil')
#     elif isinstance(xlim, list) and len(xlim) == 2:
#         xlim = xlim
#     else:
#         print(f"Invalid xlim: {xlim}, using default 'plasma'")
#         xlim = set_xlim_time(odc, type='plasma')

#     # Handle labels
#     if isinstance(label, list) and len(label) == len(odc.keys()):
#         labels = label
#     else:
#         labels = extract_labels_from_odc(odc, opt=label)

#     # Determine channels and radial positions
#     first_key = next(iter(odc.keys()))
#     channels = list(odc[first_key]['thomson_scattering.channel'].keys())
#     n_channels = len(channels)
#     radial_positions = [odc[first_key][f'thomson_scattering.channel.{i}.position.r'] for i in range(n_channels)]

#     # Create subplots
#     fig, axs = plt.subplots(n_channels, 1, figsize=(10, 2.5*n_channels))
#     if n_channels == 1:
#         axs = [axs]

#     # Plot each channel
#     for ax, (channel, r_pos) in enumerate(zip(axs, radial_positions)):
#         for key, lbl in zip(odc.keys(), labels):
#             ods = odc[key]
#             try:
#                 time = ods['thomson_scattering.time']
#                 if xunit == 'ms':
#                     time = time * 1e3
                
#                 data = unumpy.nominal_values(ods[f'thomson_scattering.channel.{channel}.t_e.data'])
#                 err = unumpy.std_devs(ods[f'thomson_scattering.channel.{channel}.t_e.data'])
                
#                 if yunit == 'keV':
#                     data = data / 1e3
#                     err = err / 1e3
                
#                 ax.errorbar(time, data, yerr=err, label=lbl)
                
#             except KeyError as e:
#                 print(f"Missing temperature data for channel {channel} in {key}: {e}")
#                 continue

#         ax.set_ylabel(f'T_e [{yunit}]')
#         ax.set_title(f'R = {r_pos:.3f} m')
#         ax.grid(True)
#         if channel == 0:
#             ax.legend()
#         if channel == n_channels-1:
#             ax.set_xlabel(f'Time [{xunit}]')
#         if xlim is not None:
#             ax.set_xlim(xlim)

#     plt.tight_layout()
#     plt.show()

"""
Ion Doppler Spectroscopy
"""

# def ion_doppler_spectroscopy_time_intensity
# def ion_doppler_spectroscopy_time_temperature
# def ion_doppler_spectroscopy_time_tor_velocity

"""
Interferometry
"""

# def interferometry_time_line_average_density

"""

"""

def time_electromagnetics_current(ods: ODS, label='shot', xunit='s', xlim='plasma', 
                               coil_indices='used', 
                               bpol_probes: dict = {'inboard_bz': {'idx': 4, 'coords': None}, 
                                                  'outboard_bz': {'idx': 39, 'coords': None}},
                               flux_loops: dict = {'fl_1': {'idx': 2, 'coords': None}},
                               onset: float = None,
                               time_of_interest: float = None):
    """Plot electromagnetic signals from a single ODS in a 2x3 subplot layout,
       similar to iFPC_preparation.ipynb.

    Args:
        ods: ODS object
            Input data containing electromagnetic measurements.
        label: str
            Base label for plots (e.g., shot number). Default is 'shot'.
        xunit: str
            Time unit ('s', 'ms'). Default is 's'.
        xlim: str or list or None
            X-axis limits. Can be 'plasma', 'coil', 'none', a list of two floats, or None.
            If 'plasma' or 'coil', limits are determined automatically.
            If None, matplotlib default is used. Default is 'plasma'.
        coil_indices: str or list
            Coil indices to plot for PF active currents. 
            Can be 'used', 'all', or a list of integer indices. Default is 'used'.
        bpol_probes: dict
            Dictionary defining B-poloidal probes to plot.
            Keys are descriptive names (e.g., 'inboard_bz').
            Values are dicts with 'idx' (int) and optional 'coords' (tuple for r,z).
            Default uses typical VEST inboard and outboard Bz probe indices.
        flux_loops: dict
            Dictionary defining flux loops to plot.
            Keys are descriptive names (e.g., 'fl_1').
            Values are dicts with 'idx' (int) and optional 'coords' (tuple for r,z).
            Default uses a typical VEST flux loop index.
        onset: float, optional
            Time offset to subtract from all time arrays (in seconds). Default is None.
        time_of_interest: float, optional
            Time (in seconds, relative to original time array before onset) to draw a vertical line.
            Default is None.
    """
    if not isinstance(ods, ODS):
        print("Error: This function expects a single ODS object.")
        return

    # Handle xlim
    # Simplified handle_xlim for single ODS context, or adapt existing one
    if xlim == 'plasma':
        # Attempt to get plasma time limits
        try:
            time_ip = ods['magnetics.ip.0.time']
            data_ip = ods['magnetics.ip.0.data']
            onset_ip, offset_ip = signal_onoffset(time_ip, data_ip)
            xlim_processed = [onset_ip, offset_ip]
            if onset is not None:
                 xlim_processed = [onset_ip - onset, offset_ip - onset]
            if xunit == 'ms':
                xlim_processed = [t * 1000 for t in xlim_processed]
        except KeyError:
            xlim_processed = None
            print("Warning: Could not determine xlim='plasma' due to missing IP data. Using default.")
    elif xlim == 'coil':
        # Attempt to get coil time limits (simplified)
        try:
            time_pf = ods['pf_active.time']
            # This requires checking all coils, simplified here
            # For a robust version, iterate coils like in set_xlim_time
            xlim_processed = [time_pf[0], time_pf[-1]] 
            if onset is not None:
                 xlim_processed = [t - onset for t in xlim_processed]
            if xunit == 'ms':
                xlim_processed = [t * 1000 for t in xlim_processed]
        except KeyError:
            xlim_processed = None
            print("Warning: Could not determine xlim='coil' due to missing pf_active.time. Using default.")
    elif xlim == 'none':
        xlim_processed = None
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim_processed = xlim # Assumed to be in the target xunit already if manually provided
    else:
        xlim_processed = None # Default to matplotlib auto-limit
        if xlim is not None:
            print(f"Invalid xlim: {xlim}, using default.")

    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=False) # sharex=False to allow different xlims if needed initially

    time_scale = 1.0
    if xunit == 'ms':
        time_scale = 1000.0

    # --- Row 0: Currents ---
    # Plasma Current (axs[0,0])
    try:
        time_ip = ods['magnetics.ip.0.time']
        current_ip = ods['magnetics.ip.0.data'] / 1e3  # kA
        plot_time_ip = (time_ip - (onset if onset is not None else 0)) * time_scale
        axs[0, 0].plot(plot_time_ip, current_ip, label=f'Shot {label}')
        axs[0, 0].set_ylabel('Plasma Current [kA]')
    except KeyError as e:
        print(f"Missing IP data: {e}")
        axs[0,0].text(0.5, 0.5, 'No IP data', ha='center', va='center')

    # Coil Currents (axs[0,1])
    try:
        time_pf = ods['pf_active.time']
        plot_time_pf = (time_pf - (onset if onset is not None else 0)) * time_scale
        
        odc = odc_or_ods_check(ods)
        actual_coil_indices = _determine_coil_indices(odc, coil_indices) # Use helper with temporary ODC
        coil_labels = []
        if 'pf_active.coil' in ods:
            for i_coil in actual_coil_indices:
                try:
                    coil_name = ods[f'pf_active.coil.{i_coil}.name']
                    current_coil = ods[f'pf_active.coil.{i_coil}.current.data'] / 1e3  # kA
                    n_turns_elements = ods[f'pf_active.coil.{i_coil}.element.:.turns_with_sign']
                    # Summing turns from all elements of the coil
                    total_turns = np.sum(np.abs(n_turns_elements)) if isinstance(n_turns_elements, np.ndarray) else np.abs(n_turns_elements) 
                    axs[0, 1].plot(plot_time_pf, current_coil, label=f'{coil_name} ({int(total_turns)} turns)')
                except KeyError as e:
                    print(f"Missing data for coil {i_coil}: {e}")
        axs[0, 1].set_ylabel('Coil Current [kA]')
        axs[0, 1].legend()
    except KeyError as e:
        print(f"Missing PF active data: {e}")
        axs[0,1].text(0.5, 0.5, 'No PF active data', ha='center', va='center')

    # Eddy Currents (axs[0,2])
    try:
        time_eddy = ods['pf_passive.time']
        plot_time_eddy = (time_eddy - (onset if onset is not None else 0)) * time_scale
        # Sum of all loop currents for a general view, or plot individually
        if 'pf_passive.loop' in ods and len(ods['pf_passive.loop']) > 0:
            all_eddy_currents = np.array([ods[f'pf_passive.loop.{i}.current'] 
                                          for i in range(len(ods['pf_passive.loop']))])
            for i_loop in range(all_eddy_currents.shape[0]):
                 axs[0, 2].plot(plot_time_eddy, all_eddy_currents[i_loop,:] / 1e3) #kA
            # axs[0, 2].plot(plot_time_eddy, np.sum(all_eddy_currents, axis=0) / 1e3, label=f'Total Eddy {label}')
        axs[0, 2].set_ylabel('Eddy Current [kA]')
    except KeyError as e:
        print(f"Missing PF passive data: {e}")
        axs[0,2].text(0.5, 0.5, 'No PF passive data', ha='center', va='center')

    # --- Row 1: Magnetic Field and Flux Contributions ---
    common_mag_time = None
    try:
        common_mag_time_orig = ods['magnetics.time']
        common_mag_time = (common_mag_time_orig - (onset if onset is not None else 0)) * time_scale
    except KeyError:
        print("Missing 'magnetics.time', cannot plot B-field or flux contributions.")
        for i in range(3): axs[1,i].text(0.5,0.5, 'No magnetics.time', ha='center',va='center')

    # Inboard Bz Probe (axs[1,0])
    probe_name_in = 'inboard_bz'
    if common_mag_time is not None and probe_name_in in bpol_probes:
        idx_in = bpol_probes[probe_name_in]['idx']
        rz_in = bpol_probes[probe_name_in].get('coords')
        if rz_in is None:
            try:
                r_in = ods[f'magnetics.b_field_pol_probe.{idx_in}.position.r']
                z_in = ods[f'magnetics.b_field_pol_probe.{idx_in}.position.z']
                rz_in = (r_in, z_in)
            except KeyError:
                print(f"Could not get coordinates for B-pol probe {idx_in}")
                rz_in = ('N/A', 'N/A') # Fallback if coords not in ODS or dict
        
        axs[1, 0].set_ylabel(f'Inboard Bz ({rz_in[0]:.3f}m, {rz_in[1]:.3f}m) [T]' if isinstance(rz_in[0], float) else f'Inboard Bz [T]')
        try:
            bz_measured_in = ods[f'magnetics.b_field_pol_probe.{idx_in}.field.data']
            axs[1, 0].plot(common_mag_time, bz_measured_in, label='Measured')

            if rz_in[0] != 'N/A': # Only calculate if coords are valid
                time_coil_in, _, _, bz_coil_in = compute_point_vacuum_fields_ods(ods, [rz_in], mode='pf_active')
                time_passive_in, _, _, bz_passive_in = compute_point_vacuum_fields_ods(ods, [rz_in], mode='pf_passive')
                time_coil_in = time_coil_in * time_scale
                time_passive_in = time_passive_in * time_scale
                # Align times if necessary, assuming compute_point_vacuum_fields_ods uses pf_active.time
                axs[1, 0].plot(time_coil_in, bz_coil_in[:,0], label='Coil')
                axs[1, 0].plot(time_passive_in, bz_coil_in[:,0] + bz_passive_in[:,0], label='Coil+Eddy')
        except KeyError as e:
            print(f"Error plotting inboard Bz probe {idx_in}: {e}")
        except Exception as e:
            print(f"Calculation error for inboard Bz probe {idx_in}: {e}")

    # Outboard Bz Probe (axs[1,1])
    probe_name_out = 'outboard_bz'
    if common_mag_time is not None and probe_name_out in bpol_probes:
        idx_out = bpol_probes[probe_name_out]['idx']
        rz_out = bpol_probes[probe_name_out].get('coords')
        if rz_out is None:
            try:
                r_out = ods[f'magnetics.b_field_pol_probe.{idx_out}.position.r']
                z_out = ods[f'magnetics.b_field_pol_probe.{idx_out}.position.z']
                rz_out = (r_out, z_out)
            except KeyError:
                print(f"Could not get coordinates for B-pol probe {idx_out}")
                rz_out = ('N/A', 'N/A')
        
        axs[1, 1].set_ylabel(f'Outboard Bz ({rz_out[0]:.3f}m, {rz_out[1]:.3f}m) [T]' if isinstance(rz_out[0], float) else 'Outboard Bz [T]')
        try:
            bz_measured_out = ods[f'magnetics.b_field_pol_probe.{idx_out}.field.data']
            axs[1, 1].plot(common_mag_time, bz_measured_out, label='Measured')
            if rz_out[0] != 'N/A': # Only calculate if coords are valid
                time_coil_out, _, _, bz_coil_out = compute_point_vacuum_fields_ods(ods, [rz_out], mode='pf_active')
                time_vacuum_out, _, _, bz_vacuum_out = compute_point_vacuum_fields_ods(ods, [rz_out], mode='vacuum')
                time_coil_out = time_coil_out * time_scale
                time_vacuum_out = time_vacuum_out * time_scale
                axs[1, 1].plot(time_coil_out, bz_coil_out[:,0], label='Coil')
                axs[1, 1].plot(time_vacuum_out, bz_vacuum_out[:,0], label='Vacuum')
        except KeyError as e:
            print(f"Error plotting outboard Bz probe {idx_out}: {e}")
        except Exception as e:
             print(f"Calculation error for outboard Bz probe {idx_out}: {e}")

    # Flux Loop (axs[1,2])
    fl_name = list(flux_loops.keys())[0] if flux_loops else None # Take the first flux loop defined
    if common_mag_time is not None and fl_name:
        idx_fl = flux_loops[fl_name]['idx']
        rz_fl = flux_loops[fl_name].get('coords') # For flux loops, position is often an outline.
                                                 # For simplicity, we might use a representative point if needed or just the index.
                                                 # The compute_point_vacuum_fields_ods needs a point.
                                                 # For now, we'll assume a point can be derived or is given if contributions are calculated.
        if rz_fl is None: # Attempt to get a representative R,Z if not given
            try:
                r_fl_pts = ods[f'magnetics.flux_loop.{idx_fl}.position.0.r'] # Assuming outline
                z_fl_pts = ods[f'magnetics.flux_loop.{idx_fl}.position.0.z']
                rz_fl = (np.mean(r_fl_pts), np.mean(z_fl_pts)) # Use centroid as representative point for calc
            except KeyError:
                 print(f"Could not get coordinates for Flux Loop {idx_fl}")
                 rz_fl = ('N/A', 'N/A')

        axs[1, 2].set_ylabel(f'Flux Loop ({rz_fl[0]:.3f}m, {rz_fl[1]:.3f}m) [Wb]' if isinstance(rz_fl[0], float) else 'Flux Loop [Wb]')
        try:
            psi_measured = ods[f'magnetics.flux_loop.{idx_fl}.flux.data']
            axs[1, 2].plot(common_mag_time, psi_measured, label='Measured')
            if rz_fl[0] != 'N/A': # Only calculate if coords are valid
                time_coil_fl, _, _, psi_coil_fl = compute_point_vacuum_fields_ods(ods, [rz_fl], mode='pf_active')
                time_passive_fl, _, _, psi_passive_fl = compute_point_vacuum_fields_ods(ods, [rz_fl], mode='pf_passive')
                time_coil_fl = time_coil_fl * time_scale
                time_passive_fl = time_passive_fl * time_scale
                axs[1, 2].plot(time_coil_fl, psi_coil_fl[:,0], label='Coil')
                axs[1, 2].plot(time_passive_fl, psi_coil_fl[:,0] + psi_passive_fl[:,0], label='Coil+Eddy')
        except KeyError as e:
            print(f"Error plotting flux loop {idx_fl}: {e}")
        except Exception as e:
            print(f"Calculation error for flux loop {idx_fl}: {e}")

    # --- General Plot Settings ---
    for i in range(3):
        axs[0, i].grid(True)
        axs[1, i].grid(True)
        axs[1, i].set_xlabel(f'Time [{xunit}]')
        axs[1, i].legend()
        if xlim_processed:
            axs[0,i].set_xlim(xlim_processed)
            axs[1,i].set_xlim(xlim_processed)
        else: # If no xlim_processed, try to share x-axis for each column if data exists
            if axs[0,i].lines and axs[1,i].lines: # Check if both subplots in a column have data
                 # Create a new twin Axes for axs[0,i] that shares the x-axis with axs[1,i]
                 # This is a bit tricky after plots are made. A simpler approach is to set xlims from data if not specified.
                 # For now, we rely on xlim_processed or individual plot auto-scaling.
                 pass 

    # Set xticklabels for top row to be invisible
    for i in range(3):
        axs[0, i].tick_params(axis='x', labelbottom=False)

    # Titles
    axs[0, 1].set_title('Full Circuit Current Waveform')
    axs[1, 1].set_title('Contributions to Measured Signal')

    # Draw vertical line for time_of_interest
    if time_of_interest is not None:
        time_line_val = (time_of_interest - (onset if onset is not None else 0)) * time_scale
        for i in range(2):
            for j in range(3):
                axs[i, j].axvline(time_line_val, color='grey', linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
    plt.show()

if __name__ == "__main__":
    from omas import ODS, ODC
    import numpy as np
    import inspect
    import sys
    import vaft

    # 1. Setup sample ODS and ODC
    ods = vaft.omas.sample_ods()
    odc = vaft.omas.sample_odc()

    # 2. List of all public plotting functions to test
    current_module = sys.modules[__name__]
    all_plot_functions_to_test = [
        'time_pf_active_current',
        'time_pf_active_current_turns',
        'time_magnetics_ip',
        'time_magnetics_diamagnetic_flux',
        'time_magnetics_flux_loop_flux',
        'time_magnetics_b_field_pol_probe_field',
        'time_equilibrium_plasma_current',
        'time_equilibrium_li',
        'time_equilibrium_beta_pol',
        'time_equilibrium_beta_tor',
        'time_equilibrium_beta_n',
        'time_equilibrium_w_mhd',
        'time_equilibrium_w_mag',
        'time_equilibrium_w_tot',
        'time_equilibrium_q0',
        'time_equilibrium_q95',
        'time_equilibrium_qa',
        'time_equilibrium_major_radius',
        'time_spectrometer_uv_intensity',
        'time_tf_b_field_tor',
        'time_tf_b_field_tor_vacuum_r',
        'time_tf_coil_current',
        'time_barometry_pressure',
        'time_electromagnetics_current',
    ]

    # 4. Define common test options
    test_labels_ods = ['shot'] 
    test_labels_odc = ['1', '2', '3'] # Test 'key' and a custom list for ODC
    test_xunits = ['s', 'ms']
    test_xlims_general = ['plasma', 'coil', 'none']
    test_xlims_short_pulse = [[0.0, 0.05], [0.001, 0.02]] # For typical short pulse data in samples

    for func_name in all_plot_functions_to_test:
        func_to_test = getattr(current_module, func_name, None)
        if func_to_test is None:
            print(f"ERROR: Function {func_name} not found in module. Skipping.")
            continue

        print(f"\\n--- Testing {func_name} ---")
        sig = inspect.signature(func_to_test)
        params = sig.parameters

        # --- ODS Test Call ---
        ods_call_args = {}
        if 'label' in params: ods_call_args['label'] = test_labels_ods[0]
        if 'xunit' in params: ods_call_args['xunit'] = test_xunits[0]
        if 'xlim' in params: ods_call_args['xlim'] = test_xlims_general[0]
        
        # Default yunits for ODS call
        yunit_map = {
            'time_pf_active_current': 'kA', 'time_pf_active_current_turns': 'kA_T',
            'time_magnetics_ip': 'MA', 'time_magnetics_diamagnetic_flux': 'Wb',
            'time_magnetics_flux_loop_flux': 'Wb', 'time_magnetics_b_field_pol_probe_field': 'T',
            'time_equilibrium_plasma_current': 'MA', 'time_equilibrium_w_mhd': 'J',
            'time_equilibrium_w_mag': 'J', 'time_equilibrium_w_tot': 'J',
            'time_equilibrium_major_radius': 'm', 'time_spectrometer_uv_intensity': 'a.u.',
            'time_tf_b_field_tor': 'T', 'time_tf_b_field_tor_vacuum_r': 'T·m',
            'time_tf_coil_current': 'kA', 'time_barometry_pressure': 'Pa',
        }
        if 'yunit' in params and func_name in yunit_map:
            ods_call_args['yunit'] = yunit_map[func_name]
        elif 'yunit' in params and params['yunit'].default is not inspect.Parameter.empty:
             ods_call_args['yunit'] = params['yunit'].default


        # Default indices for ODS call
        indices_map_ods = {
            'time_pf_active_current': 'used', 'time_pf_active_current_turns': 0,
            'time_magnetics_flux_loop_flux': 'all', 'time_magnetics_b_field_pol_probe_field': 0,
            'time_spectrometer_uv_intensity': 'H_alpha',
        }
        if 'indices' in params and func_name in indices_map_ods:
            ods_call_args['indices'] = indices_map_ods[func_name]
        elif 'indices' in params and params['indices'].default is not inspect.Parameter.empty:
             ods_call_args['indices'] = params['indices'].default

        print(f"  Calling with ODS and options: {ods_call_args}")
        try:
            func_to_test(ods, **ods_call_args)
            print(f"    {func_name}(ods) attempt finished.")
        except Exception as e:
            print(f"    ERROR during {func_name}(ods): {e}")
            import traceback
            traceback.print_exc()

        # --- ODC Test Call (with different options) ---
        odc_call_args = {}
        if 'label' in params: odc_call_args['label'] = test_labels_odc[0] # 'key'
        if 'xunit' in params: odc_call_args['xunit'] = test_xunits[1] if len(test_xunits)>1 else test_xunits[0]
        if 'xlim' in params: 
            odc_call_args['xlim'] = test_xlims_short_pulse[0] if 'pf_active' in func_name or 'ip' in func_name or 'spectrometer' in func_name else test_xlims_general[1]

        # Vary yunits for ODC call if possible
        yunit_map_odc_variant = {
            'time_pf_active_current': 'A', 'time_pf_active_current_turns': 'A_T',
            'time_magnetics_ip': 'kA', 
            'time_equilibrium_plasma_current': 'kA',
            'time_tf_b_field_tor': 'mT', 'time_tf_b_field_tor_vacuum_r': 'mT·m',
            'time_tf_coil_current': 'A', 'time_barometry_pressure': 'mbar',
        }
        if 'yunit' in params and func_name in yunit_map_odc_variant:
            odc_call_args['yunit'] = yunit_map_odc_variant[func_name]
        elif 'yunit' in params and func_name in yunit_map: # fallback to base yunit if no variant
             odc_call_args['yunit'] = yunit_map[func_name]
        elif 'yunit' in params and params['yunit'].default is not inspect.Parameter.empty: # fallback to default
             odc_call_args['yunit'] = params['yunit'].default


        # Vary indices for ODC call
        indices_map_odc = {
            'time_pf_active_current': 'all', 'time_pf_active_current_turns': [0, 1] if 'pf_active.coil.1.name' in ods else [0], # Check if coil 1 was added
            'time_magnetics_flux_loop_flux': 'inboard', 
            'time_magnetics_b_field_pol_probe_field': ['outboard', 0], # Test list of mixed types if applicable by func
            'time_spectrometer_uv_intensity': ['C_II', 'O_V', 'H_beta'],
        }
        if 'indices' in params and func_name in indices_map_odc:
            odc_call_args['indices'] = indices_map_odc[func_name]
        elif 'indices' in params and func_name in indices_map_ods: # fallback to ODS indices if no ODC variant
            odc_call_args['indices'] = indices_map_ods[func_name]
        elif 'indices' in params and params['indices'].default is not inspect.Parameter.empty: # fallback to default
             odc_call_args['indices'] = params['indices'].default
        
        # Special case for label list for ODC
        if 'label' in params and odc_call_args.get('label') == 'key': # If testing with odc and label can be a list
             if len(test_labels_odc) > 1 and isinstance(test_labels_odc[1], list):
                 if len(odc.keys()) == len(test_labels_odc[1]):
                    odc_call_args_custom_label = odc_call_args.copy()
                    odc_call_args_custom_label['label'] = test_labels_odc[1]
                    print(f"  Calling with ODC and custom list label, options: {odc_call_args_custom_label}")
                    try:
                        func_to_test(odc, **odc_call_args_custom_label)
                        print(f"    {func_name}(odc, custom label) attempt finished.")
                    except Exception as e:
                        print(f"    ERROR during {func_name}(odc, custom label): {e}")
                        traceback.print_exc()


        print(f"  Calling with ODC and options: {odc_call_args}")
        try:
            func_to_test(odc, **odc_call_args)
            print(f"    {func_name}(odc) attempt finished.")
        except Exception as e:
            print(f"    ERROR during {func_name}(odc): {e}")
            traceback.print_exc()

    print("\\nAll listed test plots attempted.")
    print("IMPORTANT: Ensure the sample ODS was populated with VEST-specific data paths for meaningful tests.")
    print("Many functions might show empty plots or errors if ods.sample() data is insufficient.")

def plot_core_profiles_time_volume_averaged(ods):
    """
    Plot volume-averaged core profile quantities as time series in N x 1 subplots.
    
    Parameters
    ----------
    ods : ODS
        OMAS data structure
        
    The function automatically checks for core_profiles.global_quantities and
    calls update_core_profiles_global_quantities_volume_average if needed.
    """
    from vaft.omas.update import update_core_profiles_global_quantities_volume_average
    from vaft.plot.utils import extract_labels_from_odc
    
    odc = odc_or_ods_check(ods)
    
    # Get shot number for title
    shot_labels = extract_labels_from_odc(odc, opt='shot')
    shot_label = shot_labels[0] if shot_labels else "Unknown"
    
    # Check if global_quantities exists and has volume_average data
    # If not, call update function
    needs_update = False
    for key in odc.keys():
        ods_item = odc[key]
        if 'core_profiles.global_quantities' not in ods_item:
            needs_update = True
            break
        gq = ods_item['core_profiles.global_quantities']
        if 'n_e_volume_average' not in gq or 't_e_volume_average' not in gq:
            needs_update = True
            break
    
    if needs_update:
        print("core_profiles.global_quantities volume_average data not found. Updating...")
        for key in odc.keys():
            update_core_profiles_global_quantities_volume_average(odc[key])
    
    # Collect all quantities to plot
    quantities = []
    
    # Electron quantities
    quantities.append({
        'path': 'core_profiles.global_quantities.n_e_volume_average',
        'label': r'$n_e$',
        'unit': r'm$^{-3}$'
    })
    quantities.append({
        'path': 'core_profiles.global_quantities.t_e_volume_average',
        'label': r'$T_e$',
        'unit': 'keV'
    })
    
    # Ion quantities (check first ODS to determine ion structure)
    first_ods = odc[list(odc.keys())[0]]
    if 'core_profiles.global_quantities' in first_ods:
        gq = first_ods['core_profiles.global_quantities']
        # Only add ion quantities if ion key exists and has data
        if 'ion' in gq and gq['ion']:
            # Get all ion indices
            if isinstance(gq['ion'], (list, tuple)):
                ion_indices = list(range(len(gq['ion'])))
            elif isinstance(gq['ion'], dict):
                ion_indices = list(gq['ion'].keys())
            else:
                ion_indices = []
            
            for ion_idx in ion_indices:
                # Check if this ion has volume_average data before adding
                try:
                    if isinstance(gq['ion'], (list, tuple)):
                        ion_item = gq['ion'][ion_idx]
                    else:
                        ion_item = gq['ion'][ion_idx]
                    
                    # Only add if both n_i and t_i exist
                    if 'n_i_volume_average' in ion_item and 't_i_volume_average' in ion_item:
                        quantities.append({
                            'path': f'core_profiles.global_quantities.ion[{ion_idx}].n_i_volume_average',
                            'label': f'$n_{{i,{ion_idx}}}$',
                            'unit': r'm$^{-3}$'
                        })
                        quantities.append({
                            'path': f'core_profiles.global_quantities.ion[{ion_idx}].t_i_volume_average',
                            'label': f'$T_{{i,{ion_idx}}}$',
                            'unit': 'keV'
                        })
                except (KeyError, IndexError, TypeError):
                    # Skip this ion if data is missing
                    continue
    
    n_quantities = len(quantities)
    if n_quantities == 0:
        print("No volume-averaged quantities found to plot.")
        return
    
    # Create N x 1 subplot layout
    fig, axs = plt.subplots(n_quantities, 1, figsize=(6, 1.5 * n_quantities), sharex=True)
    if n_quantities == 1:
        axs = [axs]
    
    # Get time array
    for idx, (ax, qty) in enumerate(zip(axs, quantities)):
        plot_successful = False
        
        for key, lbl in zip(odc.keys(), shot_labels):
            ods_item = odc[key]
            try:
                # Get time array
                if 'core_profiles.time' in ods_item:
                    time_data = np.asarray(ods_item['core_profiles.time'], float)
                else:
                    # Try to get from profiles_1d
                    if 'core_profiles.profiles_1d' in ods_item and len(ods_item['core_profiles.profiles_1d']) > 0:
                        time_data = []
                        for cp_idx in range(len(ods_item['core_profiles.profiles_1d'])):
                            cp_ts = ods_item['core_profiles.profiles_1d'][cp_idx]
                            if 'time' in cp_ts:
                                time_data.append(float(cp_ts['time']))
                            else:
                                time_data.append(float(cp_idx))
                        time_data = np.asarray(time_data)
                    else:
                        continue
                
                # Get quantity data
                # Handle OMAS path with array indices like 'ion[0]'
                try:
                    if '[' in qty['path'] and ']' in qty['path']:
                        # Split path and handle array indices
                        parts = qty['path'].split('.')
                        obj = ods_item
                        for part in parts:
                            if '[' in part and ']' in part:
                                # Extract key and index
                                key = part.split('[')[0]
                                idx_str = part.split('[')[1].split(']')[0]
                                idx = int(idx_str)
                                obj = obj[key][idx]
                            else:
                                obj = obj[part] if isinstance(obj, dict) else getattr(obj, part, None)
                            if obj is None:
                                break
                        qty_data = obj
                    else:
                        qty_data = get_from_path(ods_item, qty['path'])
                    
                    if qty_data is None:
                        continue
                    
                    qty_data = np.asarray(qty_data, float)
                except (KeyError, IndexError, AttributeError, ValueError):
                    continue
                
                # Check if data is valid
                if qty_data.size == 0 or qty_data.size != time_data.size:
                    continue
                
                ax.plot(time_data, qty_data, label=lbl)
                plot_successful = True
                
            except (KeyError, ValueError, TypeError) as e:
                continue
        
        if plot_successful:
            ax.set_ylabel(f"{qty['label']} [{qty['unit']}]")
            ax.grid(True)
            if idx == 0:
                ax.set_title(f'Shot {shot_label}')
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {qty["label"]}', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes)
            ax.set_yticks([])
    
    # Set xlabel only on last subplot
    if any(ax.lines for ax in axs):
        axs[-1].set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.show()
electromagnetics_time_current = time_electromagnetics_current


def time_voltage_consumption(ods, figsize=(4, 4)):
    """
    Plot time evolution of voltage consumption components.
    
    Plots V_loop, V_ind, and V_res on a single plot.
    
    Args:
        ods: OMAS data structure
        figsize: Figure size tuple (default: (4, 4))
    """
    from vaft.omas.formula_wrapper import compute_voltage_consumption
    
    try:
        t, V_loop, V_ind, V_res = compute_voltage_consumption(ods, time_slice=None)
    except Exception as e:
        print(f"Error computing voltage consumption: {e}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot all voltage components
    ax.plot(t, V_loop, 'b-o', linewidth=2, markersize=4, label='V_loop', alpha=0.7)
    ax.plot(t, V_ind, 'r-s', linewidth=2, markersize=4, label='V_ind', alpha=0.7)
    ax.plot(t, V_res, 'g-^', linewidth=2, markersize=4, label='V_res', alpha=0.7)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Voltage [V]', fontsize=12)
    ax.set_title('Voltage Consumption Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def time_virial_equilibrium_quantities(ods, figsize=(8, 10)):
    """
    Plot time evolution of virial equilibrium quantities.

    First row: diamagnetic flux (magnetics.diamagnetic_flux, interpolated at
    equilibrium time). Second row: diamagnetism μ_i (exact volume-integral) vs
    μ̂_i (approximated). Then s_1, s_2, s_3, alpha, B_pa; beta_p and li vs
    equilibrium; W_mag and W_kin with volume-integral W_mag and W_th.

    Args:
        ods: OMAS data structure
        figsize: Figure size tuple (default: (8, 10))
    """
    from vaft.omas.process_wrapper import (
        compute_virial_equilibrium_quantities_ods,
        compute_magnetic_energy,
        compute_volume_averaged_pressure,
        compute_diamagnetism,
    )
    from vaft.omas.update import update_equilibrium_global_quantities_volume

    if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
        print("Error: equilibrium.time_slice not found in ODS")
        return

    try:
        update_equilibrium_global_quantities_volume(ods)
    except Exception as e:
        print(f"Warning: Could not update volume: {e}")

    try:
        virial = compute_virial_equilibrium_quantities_ods(ods)
    except Exception as e:
        print(f"Error computing virial equilibrium quantities: {e}")
        return

    indices = sorted(virial.keys())
    if not indices:
        print("Error: no virial quantities computed")
        return
    n_plot = len(indices)
    t = np.array([float(ods['equilibrium.time_slice'][i].get('time', i)) for i in indices])

    diamagnetic_flux = np.full(n_plot, np.nan, dtype=float)
    if "magnetics.diamagnetic_flux.0.data" in ods and "magnetics.time" in ods and len(ods["magnetics.diamagnetic_flux"]) > 0:
        t_mag = np.asarray(ods["magnetics.time"], float)
        flux_mag = np.asarray(ods["magnetics.diamagnetic_flux.0.data"], float)
        if t_mag.size >= 2 and flux_mag.size == t_mag.size:
            diamagnetic_flux = np.abs(np.interp(t, t_mag, flux_mag))

    s_1 = np.array([virial[i]['s_1'] for i in indices], dtype=float)
    s_2 = np.array([virial[i]['s_2'] for i in indices], dtype=float)
    s_3 = np.array([virial[i]['s_3'] for i in indices], dtype=float)
    alpha = np.array([virial[i]['alpha'] for i in indices], dtype=float)
    B_pa = np.array([virial[i]['B_pa'] for i in indices], dtype=float)
    beta_p_virial = np.array([virial[i]['beta_p'] for i in indices], dtype=float)
    li_virial = np.array([virial[i]['li'] for i in indices], dtype=float)
    W_mag_virial = np.array([virial[i]['W_mag'] for i in indices], dtype=float)
    W_kin_virial = np.array([virial[i]['W_kin'] for i in indices], dtype=float)
    mui_hat = np.array([virial[i].get('mui_hat', np.nan) for i in indices], dtype=float)

    mui_exact = np.full(n_plot, np.nan, dtype=float)
    for k, i in enumerate(indices):
        try:
            mui_exact[k] = float(compute_diamagnetism(ods, time_index=i))
        except Exception:
            mui_exact[k] = np.nan

    beta_p_eq = np.full(n_plot, np.nan, dtype=float)
    li_eq = np.full(n_plot, np.nan, dtype=float)
    for k, i in enumerate(indices):
        eq_ts = ods['equilibrium.time_slice'][i]
        if 'global_quantities.beta_pol' in eq_ts:
            beta_p_eq[k] = float(eq_ts['global_quantities.beta_pol'])
        if 'global_quantities.li_3' in eq_ts:
            li_eq[k] = float(eq_ts['global_quantities.li_3'])

    n_slices = len(ods['equilibrium.time_slice'])
    W_mag_vol = np.zeros(n_plot, dtype=float)
    for k, i in enumerate(indices):
        try:
            W_mag_vol[k] = float(compute_magnetic_energy(ods, time_slice=i))
        except Exception:
            W_mag_vol[k] = np.nan

    p_vol_avg_cp = None
    p_vol_avg_eq = None
    try:
        p_vol_avg_cp = compute_volume_averaged_pressure(ods, time_slice=None, option='core_profiles')
    except Exception as e:
        print(f"Warning: Could not compute volume-averaged pressure (core_profiles): {e}")
    try:
        p_vol_avg_eq = compute_volume_averaged_pressure(ods, time_slice=None, option='equilibrium')
    except Exception as e:
        print(f"Warning: Could not compute volume-averaged pressure (equilibrium): {e}")

    W_th_cp = np.full(n_plot, np.nan, dtype=float)
    W_th_eq = np.full(n_plot, np.nan, dtype=float)
    for k, i in enumerate(indices):
        eq_ts = ods['equilibrium.time_slice'][i]
        try:
            volume = float(eq_ts['global_quantities.volume'])
        except (KeyError, ValueError):
            volume = np.nan
        if p_vol_avg_cp is not None and i < len(p_vol_avg_cp) and not np.isnan(p_vol_avg_cp[i]) and np.isfinite(volume):
            W_th_cp[k] = p_vol_avg_cp[i] * (3.0 / 2.0) * volume
        if p_vol_avg_eq is not None and i < len(p_vol_avg_eq) and not np.isnan(p_vol_avg_eq[i]) and np.isfinite(volume):
            W_th_eq[k] = p_vol_avg_eq[i] * (3.0 / 2.0) * volume

    fig, axes = plt.subplots(8, 1, figsize=(figsize[0], figsize[1]), sharex=True)

    axes[0].plot(t, diamagnetic_flux, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='diamagnetic flux')
    axes[0].set_ylabel('Δφ [Wb]', fontsize=12)
    axes[0].set_title('Diamagnetic flux', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, mui_exact, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='μ_i (exact)')
    axes[1].plot(t, mui_hat, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='μ̂_i (approximated)')
    axes[1].set_ylabel('μ_i', fontsize=12)
    axes[1].set_title('Diamagnetism', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, s_1, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='s_1')
    axes[2].plot(t, s_2, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='s_2')
    axes[2].plot(t, s_3, 'g-^', linewidth=2, markersize=4, alpha=0.7, label='s_3')
    axes[2].set_ylabel('Shafranov', fontsize=12)
    axes[2].set_title('Shafranov integrals', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, alpha, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='alpha')
    ax3_twin = axes[3].twinx()
    ax3_twin.plot(t, B_pa, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='B_pa')
    axes[3].set_ylabel('alpha', fontsize=12)
    ax3_twin.set_ylabel('B_pa [T]', fontsize=12)
    axes[3].set_title('alpha, B_pa', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper left', fontsize=10)
    ax3_twin.legend(loc='upper right', fontsize=10)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t, beta_p_virial, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='beta_p (virial)')
    axes[4].plot(t, beta_p_eq, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='beta_p (equilibrium)')
    axes[4].set_ylabel('beta_p', fontsize=12)
    axes[4].set_title('beta_poloidal', fontsize=12, fontweight='bold')
    axes[4].legend(fontsize=10)
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(t, li_virial, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='li (virial)')
    axes[5].plot(t, li_eq, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='li (equilibrium)')
    axes[5].set_ylabel('li', fontsize=12)
    axes[5].set_title('internal inductance', fontsize=12, fontweight='bold')
    axes[5].legend(fontsize=10)
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(t, W_mag_virial, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='W_mag (virial)')
    axes[6].plot(t, W_mag_vol, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='W_mag (volume-integral)')
    axes[6].set_ylabel('W_mag [J]', fontsize=12)
    axes[6].set_title('Magnetic Energy', fontsize=12, fontweight='bold')
    axes[6].legend(fontsize=10)
    axes[6].grid(True, alpha=0.3)

    axes[7].plot(t, W_kin_virial, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='W_kin (virial)')
    if p_vol_avg_cp is not None:
        axes[7].plot(t, W_th_cp, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='W_th (core_profiles)')
    if p_vol_avg_eq is not None:
        axes[7].plot(t, W_th_eq, 'm-^', linewidth=2, markersize=4, alpha=0.7, label='W_th (equilibrium)')
    axes[7].set_xlabel('Time [s]', fontsize=12)
    axes[7].set_ylabel('W [J]', fontsize=12)
    axes[7].set_title('Kinetic / Thermal Energy', fontsize=12, fontweight='bold')
    axes[7].legend(fontsize=10)
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def time_power_balance(ods, figsize=(4, 4)):
    """
    Plot time evolution of power balance energy terms.
    
    Plots -P_loss, dWdt, and P_heat on a single plot.
    
    Args:
        ods: OMAS data structure
        figsize: Figure size tuple (default: (4, 4))
    """
    from vaft.omas.formula_wrapper import compute_power_balance
    
    try:
        power_balance = compute_power_balance(ods)
        t = power_balance['time']
        P_loss = power_balance['P_loss']
        dWdt = power_balance['dWdt']
        P_heat = power_balance['P_heat']
    except Exception as e:
        print(f"Error computing power balance: {e}")
        return
    
    # Create 3x1 subplots
    fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1]), sharex=True)
    
    # Plot -P_loss
    axes[0].plot(t, P_loss, 'b-o', linewidth=2, markersize=4, alpha=0.7)
    axes[0].set_ylabel('P_loss [W]', fontsize=12)
    axes[0].set_title('P_loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot dWdt
    axes[1].plot(t, dWdt, 'r-s', linewidth=2, markersize=4, alpha=0.7)
    axes[1].set_ylabel('dW/dt [W]', fontsize=12)
    axes[1].set_title('dW/dt', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot P_heat
    axes[2].plot(t, P_heat, 'g-^', linewidth=2, markersize=4, alpha=0.7)
    axes[2].set_xlabel('Time [s]', fontsize=12)
    axes[2].set_ylabel('P_heat [W]', fontsize=12)
    axes[2].set_title('P_heat', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def time_energy(ods, figsize=(4, 4)):
    """
    Plot time evolution of magnetic energy and thermal energy.
    
    Plots W_mag and W_th (from core_profiles and equilibrium volume-averaged pressure).
    W_th = p_vol_average * 3/2 * plasma_volume
    
    Args:
        ods: OMAS data structure
        figsize: Figure size tuple (default: (4, 4))
    """
    from vaft.omas.process_wrapper import compute_magnetic_energy, compute_volume_averaged_pressure
    from vaft.omas.update import update_equilibrium_global_quantities_volume
    
    if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
        print("Error: equilibrium.time_slice not found in ODS")
        return
    
    # Ensure volume is computed
    try:
        update_equilibrium_global_quantities_volume(ods)
    except Exception as e:
        print(f"Warning: Could not update volume: {e}")
    
    # Compute volume-averaged pressure from core_profiles
    try:
        p_vol_avg_cp = compute_volume_averaged_pressure(ods, time_slice=None, option='core_profiles')
    except Exception as e:
        print(f"Error computing volume-averaged pressure (core_profiles): {e}")
        p_vol_avg_cp = None
    
    # Compute volume-averaged pressure from equilibrium
    try:
        p_vol_avg_eq = compute_volume_averaged_pressure(ods, time_slice=None, option='equilibrium')
    except Exception as e:
        print(f"Error computing volume-averaged pressure (equilibrium): {e}")
        p_vol_avg_eq = None
    
    if p_vol_avg_cp is None and p_vol_avg_eq is None:
        print("Error: Could not compute volume-averaged pressure from either option")
        return
    
    n_slices = len(ods['equilibrium.time_slice'])
    
    # Get time array
    t = np.zeros(n_slices, dtype=float)
    W_mag = np.zeros(n_slices, dtype=float)
    W_th_cp = np.zeros(n_slices, dtype=float)
    W_th_eq = np.zeros(n_slices, dtype=float)
    
    for i in range(n_slices):
        eq_ts = ods['equilibrium.time_slice'][i]
        
        # Get time
        t[i] = float(eq_ts.get('time', i))
        
        # Compute magnetic energy
        try:
            W_mag[i] = float(compute_magnetic_energy(ods, time_slice=i))
        except Exception as e:
            print(f"Warning: Could not compute W_mag for time_slice {i}: {e}")
            W_mag[i] = np.nan
        
        # Get plasma volume
        try:
            volume = float(eq_ts['global_quantities.volume'])
        except (KeyError, ValueError):
            print(f"Warning: volume not found for time_slice {i}")
            volume = np.nan
        
        # Calculate thermal energy from core_profiles: W_th = p_vol_average * 2/3 * volume
        if p_vol_avg_cp is not None and not np.isnan(p_vol_avg_cp[i]) and not np.isnan(volume):
            W_th_cp[i] = p_vol_avg_cp[i] * (3.0 / 2.0) * volume
        else:
            W_th_cp[i] = np.nan
        
        # Calculate thermal energy from equilibrium: W_th = p_vol_average * 3/2 * volume
        if p_vol_avg_eq is not None and not np.isnan(p_vol_avg_eq[i]) and not np.isnan(volume):
            W_th_eq[i] = p_vol_avg_eq[i] * (3.0 / 2.0) * volume
        else:
            W_th_eq[i] = np.nan
    
    # Create 2x1 subplots
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]), sharex=True)
    
    # Plot W_mag
    axes[0].plot(t, W_mag, 'b-o', linewidth=2, markersize=4, alpha=0.7)
    axes[0].set_ylabel('W_mag [J]', fontsize=12)
    axes[0].set_title('Magnetic Energy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot W_th from both options
    if p_vol_avg_cp is not None:
        axes[1].plot(t, W_th_cp, 'r-s', linewidth=2, markersize=4, alpha=0.7, label='from core_profiles')
    if p_vol_avg_eq is not None:
        axes[1].plot(t, W_th_eq, 'm-^', linewidth=2, markersize=4, alpha=0.7, label='from equilibrium')
    axes[1].set_xlabel('Time [s]', fontsize=12)
    axes[1].set_ylabel('W_th [J]', fontsize=12)
    axes[1].set_title('Thermal Energy', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def time_beta(ods, figsize=(4, 4)):
    """
    Plot time evolution of plasma beta parameters.
    
    Plots beta_t (toroidal beta), beta_p (poloidal beta), and beta_N (normalized beta)
    in a 3x1 subplot layout. Each subplot shows both equilibrium-based and core_profiles-based
    beta values.
    
    Args:
        ods: OMAS data structure
        figsize: Figure size tuple (default: (4, 4))
    """
    from vaft.omas.process_wrapper import compute_volume_averaged_pressure
    
    if 'equilibrium.time_slice' not in ods or len(ods['equilibrium.time_slice']) == 0:
        print("Error: equilibrium.time_slice not found in ODS")
        return
    
    n_slices = len(ods['equilibrium.time_slice'])
    
    # Initialize arrays
    t = np.zeros(n_slices, dtype=float)
    beta_t = np.zeros(n_slices, dtype=float)
    beta_p = np.zeros(n_slices, dtype=float)
    beta_N = np.zeros(n_slices, dtype=float)
    beta_t_core = np.zeros(n_slices, dtype=float)
    beta_p_core = np.zeros(n_slices, dtype=float)
    beta_N_core = np.zeros(n_slices, dtype=float)
    
    # Compute volume-averaged pressures for all time slices
    try:
        p_vol_avg_eq = compute_volume_averaged_pressure(ods, time_slice=None, option='equilibrium')
        p_vol_avg_core = compute_volume_averaged_pressure(ods, time_slice=None, option='core_profiles')
    except Exception as e:
        print(f"Warning: Could not compute volume-averaged pressure: {e}")
        p_vol_avg_eq = np.full(n_slices, np.nan)
        p_vol_avg_core = np.full(n_slices, np.nan)
    
    # Extract beta values from each time slice
    for i in range(n_slices):
        eq_ts = ods['equilibrium.time_slice'][i]
        
        # Get time
        t[i] = float(eq_ts.get('time', i))
        
        # Get beta_tor (beta_t)
        try:
            beta_t[i] = float(eq_ts['global_quantities.beta_tor'])
        except (KeyError, ValueError):
            print(f"Warning: beta_tor not found for time_slice {i}")
            beta_t[i] = np.nan
        
        # Get beta_pol (beta_p)
        try:
            beta_p[i] = float(eq_ts['global_quantities.beta_pol'])
        except (KeyError, ValueError):
            print(f"Warning: beta_pol not found for time_slice {i}")
            beta_p[i] = np.nan
        
        # Get beta_normal (beta_N)
        try:
            beta_N[i] = float(eq_ts['global_quantities.beta_normal'])
        except (KeyError, ValueError):
            print(f"Warning: beta_normal not found for time_slice {i}")
            beta_N[i] = np.nan
        
        # Compute core_profiles-based beta values
        # beta_core = beta_eq * (p_core / p_eq)
        if not np.isnan(p_vol_avg_eq[i]) and p_vol_avg_eq[i] != 0:
            ratio = p_vol_avg_core[i] / p_vol_avg_eq[i]
            beta_t_core[i] = beta_t[i] * ratio if not np.isnan(beta_t[i]) else np.nan
            beta_p_core[i] = beta_p[i] * ratio if not np.isnan(beta_p[i]) else np.nan
            beta_N_core[i] = beta_N[i] * ratio if not np.isnan(beta_N[i]) else np.nan
        else:
            beta_t_core[i] = np.nan
            beta_p_core[i] = np.nan
            beta_N_core[i] = np.nan
    
    # Create 3x1 subplots
    fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1]), sharex=True)
    
    # Plot beta_t (equilibrium and core_profiles)
    axes[0].plot(t, beta_t, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Equilibrium')
    axes[0].plot(t, beta_t_core, 'b--s', linewidth=2, markersize=4, alpha=0.7, label='Core Profiles')
    axes[0].set_ylabel(r'$\beta_t$', fontsize=12)
    axes[0].set_title('Toroidal Beta', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot beta_p (equilibrium and core_profiles)
    axes[1].plot(t, beta_p, 'r-o', linewidth=2, markersize=4, alpha=0.7, label='Equilibrium')
    axes[1].plot(t, beta_p_core, 'r--s', linewidth=2, markersize=4, alpha=0.7, label='Core Profiles')
    axes[1].set_ylabel(r'$\beta_p$', fontsize=12)
    axes[1].set_title('Poloidal Beta', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot beta_N (equilibrium and core_profiles)
    axes[2].plot(t, beta_N, 'g-^', linewidth=2, markersize=4, alpha=0.7, label='Equilibrium')
    axes[2].plot(t, beta_N_core, 'g--v', linewidth=2, markersize=4, alpha=0.7, label='Core Profiles')
    axes[2].set_xlabel('Time [s]', fontsize=12)
    axes[2].set_ylabel(r'$\beta_N$', fontsize=12)
    axes[2].set_title('Normalized Beta', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
