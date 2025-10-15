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
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5 * nrows), sharex=True)
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


def pf_active_time_current(odc_or_ods, indices='used', label='shot', xunit='s', yunit='kA', xlim='plasma'):
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


def pf_active_time_current_turns(odc_or_ods, indices='used', label='shot', xunit='s', yunit='kA_T', xlim='plasma'):
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

    _plot_pf_active_time_generic(odc_or_ods, indices, label, xunit, f'Current-Turns [{yunit}]', 'Current-Turns',
                            data_retriever_turns, xlim)

"""
magnetics - ip, Rogowski coil[:Raw plasma current], diamagnetic_flux, Flux loop (flux, voltage), Bpol_probe (field, voltage, spectrogram)
"""
def magnetics_time_ip(odc_or_ods, label='shot', xunit='s', yunit='MA', xlim='plasma'):
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

    plt.figure(figsize=(10, 4))
    
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
    plt.title('Plasma Current Time Evolution')
    plt.grid(True)
    plt.legend()

    if xlim_processed is not None:
        plt.xlim(xlim_processed)
    plt.tight_layout()
    plt.show()


def magnetics_time_diamagnetic_flux(ods_or_odc, label='shot', xunit='s', yunit='Wb', xlim='plasma'):
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
    fig, ax = plt.subplots(figsize=(10, 4))

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
    ax.set_title('Diamagnetic Flux Time Series')
    ax.legend()
    ax.grid(True)
    if xlim_processed is not None:
        ax.set_xlim(xlim_processed)

    plt.tight_layout()
    plt.show()


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
            if f'{ods_path_prefix}' in ods: # Check if the base path exists
                # Ensure what we are getting length of is indeed a list-like structure or dict for num_items
                items_container = ods[f'{ods_path_prefix}']
                if isinstance(items_container, (list, dict)):
                    num_items = len(items_container)
                    item_indices.update(range(num_items))
        return sorted(list(item_indices))
    elif indices_param in find_funcs: # e.g., 'inboard', 'outboard', 'side'
        item_indices = set()
        for key in odc.keys():
            ods = odc[key]
            try:
                # Ensure ods has the necessary structure for find_funcs
                if ods_path_prefix not in ods or not all(p in ods for p in [f'{ods_path_prefix}.:.position.0.r', f'{ods_path_prefix}.:.position.z']): # Basic check
                     # Fallback if specific structure for find_funcs is missing, to avoid KeyError in find_funcs
                     if f'{ods_path_prefix}' in ods and isinstance(ods[f'{ods_path_prefix}'], (list,dict)): # check if base path exists
                        pass # let find_funcs try, or it might fail gracefully
                     else: # if base path doesn't exist, skip
                        continue


                found_indices_tuple = find_funcs[indices_param](ods)

                # find_funcs like _find_flux_loop_inboard_indices return a tuple (array([...]),)
                if found_indices_tuple is not None and len(found_indices_tuple) > 0 and hasattr(found_indices_tuple[0], '__iter__'):
                    item_indices.update(found_indices_tuple[0])
            except KeyError as e:
                # print(f"KeyError while trying to find indices for {indices_param} in {key}: {e}")
                continue # ODS might not have the necessary structure
            except Exception as e:
                # print(f"Unexpected error while trying to find indices for {indices_param} in {key}: {e}")
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
        print(f"Invalid label: {label}, using key as label.")
        labels = extract_labels_from_odc(odc, opt='key')

    # Determine flux loop indices to plot
    if indices == 'all':
        flux_indices = []
        for key in odc.keys():
            ods = odc[key]
            if 'magnetics.flux_loop' in ods:
                num_flux = len(ods['magnetics.flux_loop'])
                for i in range(num_flux):
                    flux_indices.append(i)
        flux_indices = sorted(set(flux_indices))
    elif indices == 'inboard':
        flux_indices = []
        for key in odc.keys():
            ods = odc[key]
            inboard = _find_flux_loop_inboard_indices(ods)
            flux_indices.extend(inboard[0])
        flux_indices = sorted(set(flux_indices))
    elif indices == 'outboard':
        flux_indices = []
        for key in odc.keys():
            ods = odc[key]
            outboard = _find_flux_loop_outboard_indices(ods)
            flux_indices.extend(outboard[0])
        flux_indices = sorted(set(flux_indices))
    elif isinstance(indices, int):
        flux_indices = [indices]
    elif isinstance(indices, list):
        flux_indices = indices
    else:
        raise ValueError(f"indices must be 'all', one of {list(find_funcs.keys())}, an integer, or a list of integers.")

def _plot_magnetics_time_subplot_generic(odc_or_ods, indices_param, label_param, xunit, yunit,
                                    ods_path_prefix, time_path_suffix, data_path_suffix,
                                    title_base, ylabel_base, find_funcs, xlim_param):
    """Generic plotting function for magnetics data (flux_loop, b_pol_probe) with subplots."""
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim_param)
    labels = handle_labels(odc, label_param)

    item_indices = _determine_magnetics_indices(odc, indices_param, ods_path_prefix, find_funcs)

    if not item_indices:
        print(f"No valid {ylabel_base.lower()} found to plot for indices: {indices_param}")
        return

    nrows = len(item_indices)
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    any_plot_made = False
    for ax_idx, item_idx_val in enumerate(item_indices): # Use enumerate for axs index
        ax = axs[ax_idx]
        plot_successful_for_item = False
        legend_handles_labels_for_ax = {} # To avoid duplicate legend entries per subplot

        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                # Construct full paths carefully
                # e.g. magnetics.flux_loop.time or magnetics.b_field_pol_probe.time
                full_time_path = f'{ods_path_prefix}.{time_path_suffix}'
                # e.g. magnetics.flux_loop.0.flux.data or magnetics.b_field_pol_probe.0.field.data
                full_data_path = f'{ods_path_prefix}.{item_idx_val}.{data_path_suffix}'

                if full_time_path not in ods or full_data_path not in ods:
                    # print(f"Missing time or data path in ODS {key} for item {item_idx_val}: {full_time_path} or {full_data_path}")
                    continue

                time_data = ods[full_time_path]
                data_val = ods[full_data_path]

                if xunit == 'ms':
                    time_data = time_data * 1e3
                # Add yunit conversions if necessary, similar to pf_active or equilibrium helpers

                line, = ax.plot(time_data, data_val, label=lbl)
                if lbl not in legend_handles_labels_for_ax: # Store unique handles for legend
                    legend_handles_labels_for_ax[lbl] = line
                plot_successful_for_item = True
                any_plot_made = True
            except KeyError as e:
                # print(f"Missing data for {ylabel_base} {item_idx_val} in ODS {key}: {e}")
                continue
            except Exception as e:
                # print(f"Error plotting {ylabel_base} {item_idx_val} in ODS {key}: {e}")
                continue

        if plot_successful_for_item:
            ax.set_ylabel(f'{ylabel_base} [{yunit}]')
            if ax_idx == 0: # Title for the first subplot
                ax.set_title(f'{title_base} Time Series')
            # Use unique handles for legend for this specific subplot
            if legend_handles_labels_for_ax:
                 ax.legend(legend_handles_labels_for_ax.values(), legend_handles_labels_for_ax.keys())
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f'No data for {ylabel_base.lower()} {item_idx_val}',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_yticks([]) # Remove y-ticks if no data

    if any_plot_made: # Common X-axis label for the last subplot if plots were made
        axs[-1].set_xlabel(f'Time [{xunit}]')
        if xlim_processed is not None:
            # Apply xlim to the shared x-axis by setting it on one of the axes (e.g., the first one)
            # since they are shared. Or plt.xlim() if figure is current.
            axs[0].set_xlim(xlim_processed)
    else: # If no plots were made at all across all subplots
        # Optionally, remove the entire figure or display a message
        fig.text(0.5, 0.5, 'No data found for any selected items.',
                 horizontalalignment='center', verticalalignment='center')


    plt.tight_layout()
    plt.show()

def _find_flux_loop_inboard_indices(ods):
    # find the indices of inboard flux loop in VEST
    indices = np.where(ods['magnetics.flux_loop.:.position.0.r'] < 0.15)
    return indices

def _find_flux_loop_outboard_indices(ods):
    # find the indices of the flux loop outboard
    indices = np.where(ods['magnetics.flux_loop.:.position.0.r'] > 0.5)
    return indices

def magnetics_time_flux_loop_flux(ods_or_odc, indices='all', label='shot', xunit='s', yunit='Wb', xlim='plasma'):
    """
    Plot flux loop flux time series.
    """
    find_funcs = {
        'inboard': _find_flux_loop_inboard_indices,
        'outboard': _find_flux_loop_outboard_indices
    }
    _plot_magnetics_time_subplot_generic(ods_or_odc, indices, label, xunit, yunit,
                                    ods_path_prefix='magnetics.flux_loop',
                                    time_path_suffix='time',
                                    data_path_suffix='flux.data',
                                    title_base='Flux Loop Flux',
                                    ylabel_base='Flux',
                                    find_funcs=find_funcs,
                                    xlim_param=xlim)

# def magnetics_time_flux_loop_voltage


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

def magnetics_time_b_field_pol_probe_field(ods_or_odc, indices='all', label='shot', xunit='s', yunit='T', xlim='plasma'):
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


"""
Equilibrium plotting functions
"""

def _plot_equilibrium_time_quantity(odc_or_ods, quantity_key, ylabel, title_suffix, label='shot', xunit='s', yunit=None, xlim='plasma'):
    """Helper function to plot generic equilibrium quantities."""
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim)
    labels = handle_labels(odc, label)

    plt.figure(figsize=(10, 4))
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


def equilibrium_time_plasma_current(odc_or_ods, label='shot', xunit='s', yunit='MA', xlim='plasma'):
    """
    Plot equilibrium plasma current (Ip) time series from equilibrium global_quantities.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'ip', f'Plasma Current [{yunit}]', 'Plasma Current', 
                             label=label, xunit=xunit, yunit=yunit, xlim=xlim)


def equilibrium_time_li(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium internal inductance (li_3) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'li_3', 'Internal Inductance [li_3]', 'Internal Inductance (li_3)',
                             label=label, xunit=xunit, yunit=None, xlim=xlim) # yunit is None as it's unitless


def equilibrium_time_beta_pol(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium poloidal beta (beta_pol) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_pol', 'Poloidal Beta [beta_pol]', 'Poloidal Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_beta_tor(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium toroidal beta (beta_tor) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_tor', 'Toroidal Beta [beta_tor]', 'Toroidal Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_beta_n(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium normalized beta (beta_n) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_normal', 'Normalized Beta [beta_n]', 'Normalized Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    elif label in ['shot', 'pulse', 'run', 'key']:
        labels = extract_labels_from_odc(odc, opt=label)
    else:
        print(f"Invalid label: {label}, using key as label.")
        labels = extract_labels_from_odc(odc, opt='key')

    # Determine B-pol probe indices to plot
    if indices == 'all':
        probe_indices = []
        for key in odc.keys():
            ods = odc[key]
            if 'magnetics.b_field_pol_probe' in ods:
                num_probes = len(ods['magnetics.b_field_pol_probe'])
                for i in range(num_probes):
                    probe_indices.append(i)
        probe_indices = sorted(set(probe_indices))
    elif indices == 'inboard':
        probe_indices = []
        for key in odc.keys():
            ods = odc[key]
            inboard = _find_bpol_probe_inboard_indices(ods)
            probe_indices.extend(inboard[0])
        probe_indices = sorted(set(probe_indices))
    elif indices == 'outboard':
        probe_indices = []
        for key in odc.keys():
            ods = odc[key]
            outboard = _find_bpol_probe_outboard_indices(ods)
            probe_indices.extend(outboard[0])
        probe_indices = sorted(set(probe_indices))
    elif indices == 'side':
        probe_indices = []
        for key in odc.keys():
            ods = odc[key]
            side = _find_bpol_probe_side_indices(ods)
            probe_indices.extend(side[0])
        probe_indices = sorted(set(probe_indices))
    elif isinstance(indices, int):
        probe_indices = [indices]
    elif isinstance(indices, list):
        probe_indices = indices
    else:
        raise ValueError("indices must be 'all', 'inboard', 'outboard', 'side', or a list of integers")

    if not probe_indices:
        print("No valid B-pol probes found to plot")
        return

    # Create subplots
    nrows = len(probe_indices)
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5*nrows))
    if nrows == 1:
        axs = [axs]

    # Plot each B-pol probe in its own subplot
    for ax, probe_idx in zip(axs, probe_indices):
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                # Get time data and convert units
                time = ods['magnetics.b_field_pol_probe.time']
                if xunit == 'ms':
                    time = time * 1e3

                # Get B-field data and convert units if necessary
                field = ods[f'magnetics.b_field_pol_probe.{probe_idx}.field.data']
                data = field  # Adjust if yunit requires conversion

                ax.plot(time, data, label=lbl)
            except KeyError as e:
                print(f"Missing data for B-pol probe {probe_idx} in {key}: {e}")
                continue

        ax.set_ylabel(f'B-field [{yunit}]')
        # only show xlabel for the last subplot
        if probe_idx == probe_indices[-1]:
            ax.set_xlabel(f'Time [{xunit}]')
        if probe_idx == probe_indices[0]:
            ax.set_title('B-field Poloidal Probe Time Series')
            ax.legend()
        ax.grid(True)
        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def equilibrium_time_plasma_current(odc_or_ods, label='shot', xunit='s', yunit='MA', xlim='plasma'):
    """
    Plot equilibrium plasma current (Ip) time series from equilibrium global_quantities.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'ip', f'Plasma Current [{yunit}]', 'Plasma Current', 
                             label=label, xunit=xunit, yunit=yunit, xlim=xlim)


def equilibrium_time_li(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium internal inductance (li_3) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'li_3', 'Internal Inductance [li_3]', 'Internal Inductance (li_3)',
                             label=label, xunit=xunit, yunit=None, xlim=xlim) # yunit is None as it's unitless


def equilibrium_time_beta_pol(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium poloidal beta (beta_pol) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_pol', 'Poloidal Beta [beta_pol]', 'Poloidal Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_beta_tor(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium toroidal beta (beta_tor) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_tor', 'Toroidal Beta [beta_tor]', 'Toroidal Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_beta_n(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium normalized beta (beta_n) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'beta_normal', 'Normalized Beta [beta_n]', 'Normalized Beta',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_w_mhd(odc_or_ods, label='shot', xunit='s', yunit='J', xlim='plasma'):
    """
    Plot equilibrium MHD stored energy (w_mhd) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'energy_mhd', 'MHD Stored Energy [J]', 'MHD Stored Energy',
                             label=label, xunit=xunit, yunit=None, xlim=xlim) # yunit conversion is not needed for J


def equilibrium_time_w_mag(odc_or_ods, label='shot', xunit='s', yunit='J', xlim='plasma'):
    """
    Plot equilibrium magnetic stored energy (w_mag) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'energy_mag', 'Magnetic Stored Energy [J]', 'Magnetic Stored Energy',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_w_tot(odc_or_ods, label='shot', xunit='s', yunit='J', xlim='plasma'):
    """
    Plot equilibrium total stored energy (w_tot) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'energy_total', 'Total Stored Energy [J]', 'Total Stored Energy',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_q0(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium q-axis (q0) time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'q_axis', 'q-axis [q0]', 'q-axis (q0)',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_q95(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium q95 time series.
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'q_95', 'q95', 'q95',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)


def equilibrium_time_qa(odc_or_ods, label='shot', xunit='s', yunit='', xlim='plasma'):
    """
    Plot equilibrium qa time series (if available).
    """
    _plot_equilibrium_time_quantity(odc_or_ods, 'qa', 'qa', 'qa',
                             label=label, xunit=xunit, yunit=None, xlim=xlim)

# SHAPE QUANTITIES

def equilibrium_time_major_radius(odc_or_ods, label='shot', xunit='s', yunit='m', xlim='plasma'):
    """
    Plot equilibrium major radius (geometric_axis.r) time series.
    """
    odc = odc_or_ods_check(odc_or_ods)
    xlim_processed = handle_xlim(odc_or_ods, xlim)
    labels = handle_labels(odc, label)
    plt.figure(figsize=(10, 4))
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

"""
spectrometer_uv (filterscope)
"""


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
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5 * nrows), sharex=True)
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
            ax.set_ylabel(f'Intensity [{yunit}]')
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

def spectrometer_uv_time_intensity(odc_or_ods, indices='all', label='shot', xunit='s', yunit='a.u.', xlim='plasma'):
    """
    Plot UV spectrometer/filterscope intensity time series.
    """
    _plot_spectrometer_subplot_generic(odc_or_ods, indices, label, xunit, yunit, 
                                       SPECTROMETER_LINE_MAP, xlim)

"""
TF coil
"""
def tf_time_b_field_tor(odc_or_ods, label='shot', xunit='s', yunit='T', xlim='plasma'):
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

    plt.figure(figsize=(10, 4))
    
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

def tf_time_b_field_tor_vacuum_r(odc_or_ods, label='shot', xunit='s', yunit='T·m', xlim='plasma'):
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

    plt.figure(figsize=(10, 4))
    
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

def tf_time_coil_current(odc_or_ods, label='shot', xunit='s', yunit='MA', xlim='plasma'):
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

    plt.figure(figsize=(10, 4))
    
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


"""
eddy_current (pf_passive)
"""

# def pf_passive_current




"""
barometry (Vacuum Gauge or Neutral Pressure Gauge)
"""

def barometry_time_pressure(odc_or_ods, label='shot', xunit='s', yunit='Pa', xlim='plasma'):
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

    plt.figure(figsize=(10, 4))
    
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

def electromagnetics_time_current(ods: ODS, label='shot', xunit='s', xlim='plasma', 
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

    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharex=False) # sharex=False to allow different xlims if needed initially

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
        'pf_active_time_current',
        'pf_active_time_current_turns',
        'magnetics_time_ip',
        'magnetics_time_diamagnetic_flux',
        'magnetics_time_flux_loop_flux',
        'magnetics_time_b_field_pol_probe_field',
        'equilibrium_time_plasma_current',
        'equilibrium_time_li',
        'equilibrium_time_beta_pol',
        'equilibrium_time_beta_tor',
        'equilibrium_time_beta_n',
        'equilibrium_time_w_mhd',
        'equilibrium_time_w_mag',
        'equilibrium_time_w_tot',
        'equilibrium_time_q0',
        'equilibrium_time_q95',
        'equilibrium_time_qa',
        'equilibrium_time_major_radius',
        'spectrometer_uv_time_intensity',
        'tf_time_b_field_tor',
        'tf_time_b_field_tor_vacuum_r',
        'tf_time_coil_current',
        'barometry_time_pressure',
        'electromagnetics_time_current',
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
            'pf_active_time_current': 'kA', 'pf_active_time_current_turns': 'kA_T',
            'magnetics_time_ip': 'MA', 'magnetics_time_diamagnetic_flux': 'Wb',
            'magnetics_time_flux_loop_flux': 'Wb', 'magnetics_time_b_field_pol_probe_field': 'T',
            'equilibrium_time_plasma_current': 'MA', 'equilibrium_time_w_mhd': 'J',
            'equilibrium_time_w_mag': 'J', 'equilibrium_time_w_tot': 'J',
            'equilibrium_time_major_radius': 'm', 'spectrometer_uv_time_intensity': 'a.u.',
            'tf_time_b_field_tor': 'T', 'tf_time_b_field_tor_vacuum_r': 'T·m',
            'tf_time_coil_current': 'kA', 'barometry_time_pressure': 'Pa',
        }
        if 'yunit' in params and func_name in yunit_map:
            ods_call_args['yunit'] = yunit_map[func_name]
        elif 'yunit' in params and params['yunit'].default is not inspect.Parameter.empty:
             ods_call_args['yunit'] = params['yunit'].default


        # Default indices for ODS call
        indices_map_ods = {
            'pf_active_time_current': 'used', 'pf_active_time_current_turns': 0,
            'magnetics_time_flux_loop_flux': 'all', 'magnetics_time_b_field_pol_probe_field': 0,
            'spectrometer_uv_time_intensity': 'H_alpha',
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
            'pf_active_time_current': 'A', 'pf_active_time_current_turns': 'A_T',
            'magnetics_time_ip': 'kA', 
            'equilibrium_time_plasma_current': 'kA',
            'tf_time_b_field_tor': 'mT', 'tf_time_b_field_tor_vacuum_r': 'mT·m',
            'tf_time_coil_current': 'A', 'barometry_time_pressure': 'mbar',
        }
        if 'yunit' in params and func_name in yunit_map_odc_variant:
            odc_call_args['yunit'] = yunit_map_odc_variant[func_name]
        elif 'yunit' in params and func_name in yunit_map: # fallback to base yunit if no variant
             odc_call_args['yunit'] = yunit_map[func_name]
        elif 'yunit' in params and params['yunit'].default is not inspect.Parameter.empty: # fallback to default
             odc_call_args['yunit'] = params['yunit'].default


        # Vary indices for ODC call
        indices_map_odc = {
            'pf_active_time_current': 'all', 'pf_active_time_current_turns': [0, 1] if 'pf_active.coil.1.name' in ods else [0], # Check if coil 1 was added
            'magnetics_time_flux_loop_flux': 'inboard', 
            'magnetics_time_b_field_pol_probe_field': ['outboard', 0], # Test list of mixed types if applicable by func
            'spectrometer_uv_time_intensity': ['C_II', 'O_V', 'H_beta'],
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
