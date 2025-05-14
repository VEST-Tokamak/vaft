"""
This module contains functions for plotting time series data from OMAS ODS.
"""

from omas import *
import matplotlib.pyplot as plt
from vaft.process import signal_onoffset, is_signal_active
import matplotlib.pyplot as plt
import numpy as np

"""
Fllowing functions are tools for plotting time series data.
"""

def odc_or_ods_check(odc_or_ods):
    """
    Check input type and initialize ODC if necessary.
    
    Parameters:
    odc_or_ods (ODC or ODS): Input object to check.
    
    Returns:
    ODC: Initialized ODC object.
    
    Raises:
    TypeError: If input is not of type ODS or ODC.
    """
    if isinstance(odc_or_ods, ODC):
        return odc_or_ods
    elif isinstance(odc_or_ods, ODS):
        odc = ODC()
        odc['0'] = odc_or_ods
        return odc
    else:
        raise TypeError("Input must be of type ODS or ODC")

def extract_labels_from_odc(odc, opt = 'shot'):
    """
    Extract list from ODC object. 
    
    Parameters:
    odc (ODC): ODC object to extract labels from.
    opt (str): The option for the list. Can be 'shot'/'pulse' or 'key'
    Returns:
    list: List of labels extracted from ODC.
    """
    labels = []
    for key in odc.keys():
        if opt == 'key':
            labels.append(key)
        elif opt == 'shot' or opt == 'pulse':
            try:
                data_entry = odc[key].get('dataset_description.data_entry', {})
                labels.append(data_entry.get('pulse'))
            except:
                print(f"Key {key} does not have a dataset_description.data_entry.")
                labels.append(key)
        elif opt == 'run':
            try:
                data_entry = odc[key].get('dataset_description.data_entry', {})
                labels.append(data_entry.get('run'))
            except:
                print(f"Key {key} does not have a dataset_description.data_entry.")
                labels.append(key)
        else:
            print(f"Invalid option: {opt}, using key as label.")
            labels.append(key)
    return labels

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
pf_active (pf coil)
"""
def pf_active_time_current(odc_or_ods, indices='used', label='shot', xunit='s', yunit='kA', xlim='plasma'):
    """
    Plot PF coil currents in n x 1 subplots.

    Parameters:
        odc_or_ods: ODS or ODC
            The input data. Can be a single ODS or a collection of ODS objects (ODC).
        indices: str or list of int
            The indices of the coils to plot. Can be 'used', 'all', or a list of indices.
        label: str
            The option for the legend. Can be 'shot', 'key', 'run', or a list of labels.
        xunit: str
            The unit of the x-axis. Can be 's', 'ms', or 'us'.
        yunit: str
            The unit of the y-axis. Can be 'kA', 'MA', or 'A'.
        xlim: str or list
            The x-axis limits. Can be 'plasma', 'coil', 'none', or a list of two floats.
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    elif label in ['shot', 'pulse', 'run', 'key']:
        labels = extract_labels_from_odc(odc, opt=label)
    else:
        print(f"Invalid label: {label}, using key as label.")
        labels = extract_labels_from_odc(odc, opt='key')

    # Determine coil indices to plot
    if indices == 'used':
        coil_indices = set()
        for key in odc.keys():
            ods = odc[key]
            if 'pf_active.coil' in ods:
                num_coils = len(ods['pf_active.coil'])
                for i in range(num_coils):
                    if f'pf_active.coil.{i}.current.data' in ods and is_signal_active(ods[f'pf_active.coil.{i}.current.data']):
                        coil_indices.add(i)
        coil_indices = sorted(coil_indices)
    elif indices == 'all':
        max_coils = max((len(ods.get('pf_active.coil', [])) for ods in odc.values()), default=0)
        coil_indices = list(range(max_coils))
    elif isinstance(indices, int):
        coil_indices = [indices]
    elif isinstance(indices, list):
        coil_indices = indices
    else:
        raise ValueError("indices must be 'used', 'all', or a list of integers")

    if not coil_indices:
        print("No valid coils found to plot")
        return

    # Create subplots
    nrows = len(coil_indices)
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5*nrows))
    if nrows == 1:
        axs = [axs]

    # Plot each coil in its own subplot
    for ax, coil_idx in zip(axs, coil_indices):
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                # Handle time unit conversion
                time = ods['pf_active.time']
                if xunit == 'ms':
                    time = time * 1e3

                # Handle current data and unit conversion
                data = ods[f'pf_active.coil.{coil_idx}.current.data']
                name = ods[f'pf_active.coil.{coil_idx}.name']
                if yunit == 'MA':
                    data = data / 1e6
                elif yunit == 'kA':
                    data = data / 1e3
                ax.plot(time, data, label=lbl)
            except KeyError:
                continue  # Skip if coil doesn't exist in this ODS
        ax.set_ylabel(f'{name} Current [{yunit}]')
        # only show xlabel for the last subplot
        if coil_idx == len(coil_indices) - 1:
            ax.set_xlabel(f'Time [{xunit}]')
        # only show legend for the first subplot
        if coil_idx == 0:
            ax.set_title(f'pf active time-current')
            ax.legend()
        if xlim is not None:
            ax.set_xlim(xlim)
    plt.tight_layout()
    plt.show()

def pf_active_time_current_turns(odc_or_ods, indices='used', label='shot', xunit='s', yunit='kA_T', xlim='plasma'):
    """
    Plot PF coil currents multiplied by turns in n x 1 subplots.

    Parameters:
        odc_or_ods: ODS or ODC
            The input data. Can be a single ODS or a collection of ODS objects (ODC).
        indices: str or list of int
            The indices of the coils to plot. Can be 'used', 'all', or a list of indices.
        label: str
            The option for the legend. Can be 'shot', 'key', 'run', or a list of labels.
        xunit: str
            The unit of the x-axis. Can be 's', 'ms', or 'us'.
        yunit: str
            The unit of the y-axis. Can be 'kA_T', 'MA_T', or 'A_T'.
        xlim: str or list
            The x-axis limits. Can be 'plasma', 'coil', 'none', or a list of two floats.
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    elif label in ['shot', 'pulse', 'run', 'key']:
        labels = extract_labels_from_odc(odc, opt=label)
    else:
        print(f"Invalid label: {label}, using key as label.")
        labels = extract_labels_from_odc(odc, opt='key')

    # Determine coil indices to plot (same logic as pf_active_time_current)
    if indices == 'used':
        coil_indices = set()
        for key in odc.keys():
            ods = odc[key]
            if 'pf_active.coil' in ods:
                num_coils = len(ods['pf_active.coil'])
                for i in range(num_coils):
                    if f'pf_active.coil.{i}.current.data' in ods and is_signal_active(ods[f'pf_active.coil.{i}.current.data']):
                        coil_indices.add(i)
        coil_indices = sorted(coil_indices)
    elif indices == 'all':
        max_coils = max((len(ods.get('pf_active.coil', [])) for ods in odc.values()), default=0)
        coil_indices = list(range(max_coils))
    elif isinstance(indices, int):
        coil_indices = [indices]
    elif isinstance(indices, list):
        coil_indices = indices
    else:
        raise ValueError("indices must be 'used', 'all', or a list of integers")

    if not coil_indices:
        print("No valid coils found to plot")
        return

    # Create subplots
    nrows = len(coil_indices)
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5*nrows))
    if nrows == 1:
        axs = [axs]

    # Plot each coil in its own subplot
    for ax, coil_idx in zip(axs, coil_indices):
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                # Get time data and convert units
                time = ods['pf_active.time']
                if xunit == 'ms':
                    time = time * 1e3
                
                # Get current data and calculate turns
                current = ods[f'pf_active.coil.{coil_idx}.current.data']
                turns = np.sum(np.abs(ods[f'pf_active.coil.{coil_idx}.element.:.turns_with_sign']))
                
                # Convert units
                if yunit == 'MA_T':
                    data = current * turns / 1e6
                elif yunit == 'kA_T':
                    data = current * turns / 1e3
                else:  # A_T
                    data = current * turns

                ax.plot(time, data, label=lbl)
            except KeyError as e:
                print(f"Missing data for coil {coil_idx} in {key}: {e}")
                continue

        ax.set_ylabel(f'Current-Turns [{yunit}]')
        # only show xlabel for the last subplot
        if coil_idx == coil_indices[-1]:
            ax.set_xlabel(f'Time [{xunit}]')
        if coil_idx == 0:
            ax.set_title(f'pf active time-current turns')
            ax.legend()
        ax.grid(True)
        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

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
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    else:
        labels = extract_labels_from_odc(odc, opt=label)

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

    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()


# def magnetics_time_rogowski_coil_current(ods_or_odc, labels=None):
#     odc = odc_or_ods_check(ods_or_odc)

#     if labels is None or len(labels) != len(odc.keys()):
#         labels = extract_labels_from_odc(odc)

#     for key, label in zip(odc.keys(), labels):
#         time = odc[key]['magnetics.rogowski_coil.0.time']
#         current = odc[key]['magnetics.rogowski_coil.0.data']
#         plt.plot(time, current, label=label)

#     plt.xlabel("Time [s]")
#     plt.ylabel("Rogowski Coil Current [A]")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


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
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    elif label in ['shot', 'pulse', 'run', 'key']:
        labels = extract_labels_from_odc(odc, opt=label)
    else:
        print(f"Invalid label: {label}, using key as label.")
        labels = extract_labels_from_odc(odc, opt='key')

    # Determine if multiple diamagnetic_flux entries exist
    # Assuming only one diamagnetic_flux entry per ODS

    # Create subplots (single plot)
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot each ODS's diamagnetic_flux
    for key, lbl in zip(odc.keys(), labels):
        ods = odc[key]
        try:
            # Get time data and convert units
            time = ods['magnetics.diamagnetic_flux.time']
            if xunit == 'ms':
                time = time * 1e3

            # Get diamagnetic_flux data and convert units if necessary
            flux = ods['magnetics.diamagnetic_flux.0.data']
            data = flux  # Adjust if yunit requires conversion

            ax.plot(time, data, label=lbl)
        except KeyError as e:
            print(f"Missing diamagnetic_flux data in {key}: {e}")
            continue

    ax.set_ylabel(f'Diamagnetic Flux [{yunit}]')
    ax.set_xlabel(f'Time [{xunit}]')
    ax.set_title('Diamagnetic Flux Time Series')
    ax.legend()
    ax.grid(True)
    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


# In the VEST, the flux loop is classified as 'inboard', and 'outboard'
# 'inboard' is the flux loops located in the inboard (HF) side of vessel
# 'outboard' is the flux loops located in the outboard (LF) side of vessel

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
    
    Parameters:
        ods_or_odc: ODS or ODC
            The input data. Can be a single ODS or a collection of ODS objects (ODC).
        indices: str or list of int
            The flux loop indices to plot. Can be 'all', 'inboard', 'outboard', or a list of indices.
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
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    elif label in ['shot', 'pulse', 'run', 'key']:
        labels = extract_labels_from_odc(odc, opt=label)
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
        raise ValueError("indices must be 'all', 'inboard', 'outboard', or a list of integers")

    if not flux_indices:
        print("No valid flux loops found to plot")
        return

    # Create subplots
    nrows = len(flux_indices)
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5*nrows))
    if nrows == 1:
        axs = [axs]

    # Plot each flux loop in its own subplot
    for ax, flux_idx in zip(axs, flux_indices):
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                # Get time data and convert units
                time = ods['magnetics.flux_loop.time']
                if xunit == 'ms':
                    time = time * 1e3

                # Get flux data and convert units if necessary
                flux = ods[f'magnetics.flux_loop.{flux_idx}.flux.data']
                data = flux  # Adjust if yunit requires conversion

                ax.plot(time, data, label=lbl)
            except KeyError as e:
                print(f"Missing data for flux loop {flux_idx} in {key}: {e}")
                continue

        ax.set_ylabel(f'Flux [{yunit}]')
        # only show xlabel for the last subplot
        if flux_idx == flux_indices[-1]:
            ax.set_xlabel(f'Time [{xunit}]')
        if flux_idx == flux_indices[0]:
            ax.set_title('Flux Loop Flux Time Series')
            ax.legend()
        ax.grid(True)
        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

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
    odc = odc_or_ods_check(ods_or_odc)
    
    # Handle xlim
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

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


"""
equilibrium
"""

# def equilibrium_time_global_quantities

# shape quantities (major_radius, minor_radius, elongation, triangularity, etc.)
# def equilibrium_time_shape_quantities
# def equilibrium_time_major_radius
# def equilibrium_time_minor_radius
# def equilibrium_time_elongation
# def equilibrium_time_triangularity
# def equilibrium_time_upper_triangularity
# def equilibrium_time_lower_triangularity
# def equilibrium_time_magnetic_axis_r
# def equilibrium_time_magnetic_axis_z
# def equilibrium_time_current_centre_r
# def equilibrium_time_current_centre_z


# mhd quantities (plasma_current, plasma_current_density, etc.)
# def equilibrium_time_mhd_quantities
# def equilibrium_time_pressure
# def equilibrium_time_plasma_current
# def equilibrium_time_f
# def equilibrium_time_ffprime
# def equilibrium_time_q0
# def equilibrium_time_q95
# def equilibrium_time_qa
# def equilibrium_time_li
# def equilibrium_time_beta_pol
# def equilibrium_time_beta_tor
# def equilibrium_time_beta_n
# def equilibrium_time_w_mhd
# def equilibrium_time_w_mag
# def equilibrium_time_w_tot

"""
spectrometer_uv (filterscope)
"""

def spectrometer_uv_time_intensity(odc_or_ods, indices='all', label='shot', xunit='s', yunit='a.u.', xlim='plasma'):
    """
    Plot UV spectrometer/filterscope intensity time series.
    
    Parameters:
        odc_or_ods: ODS/ODC
            Input data containing spectrometer measurements
        indices: str or list
            Line indices to plot: 'all', 'H_alpha', 'H_alpha_fast', 
            'C_II', 'C_III', 'O_I', 'O_II', 'O_V' or list of these
        label: str
            Legend labels option ('shot', 'key', 'run')
        xunit: str
            Time unit ('s', 'ms')
        yunit: str
            Intensity unit (typically 'a.u.')
        xlim: str/list
            X-axis limits
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Line mapping configuration
    LINE_MAP = {
        'H_alpha': (0, 0),
        'O_I': (0, 1),
        'H_alpha_fast': (1, 0),
        'H_beta': (1, 1),
        'H_gamma': (1, 2),
        'C_II': (1, 3),
        'C_III': (1, 4),
        'O_II': (1, 5),
        'O_V': (1, 6)
    }

    # Handle indices selection
    if indices == 'all':
        selected_lines = list(LINE_MAP.keys())
    elif isinstance(indices, str):
        selected_lines = [indices]
    else:
        selected_lines = indices

    # Verify valid lines
    valid_lines = []
    for line in selected_lines:
        if line in LINE_MAP:
            valid_lines.append(line)
        else:
            print(f"Warning: Invalid line index {line} ignored")

    if not valid_lines:
        print("No valid spectral lines to plot")
        return

    # Handle xlim
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    else:
        labels = extract_labels_from_odc(odc, opt=label)

    # Create subplots
    nrows = len(valid_lines)
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5*nrows))
    if nrows == 1:
        axs = [axs]

    # Plot each spectral line
    for ax, line in zip(axs, valid_lines):
        channel, line_idx = LINE_MAP[line]
        
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                time = ods[f'spectrometer_uv.channel.{channel}.processed_line.{line_idx}.intensity.time']
                data = ods[f'spectrometer_uv.channel.{channel}.processed_line.{line_idx}.intensity.data']
                
                if xunit == 'ms':
                    time = time * 1e3
                
                ax.plot(time, data, label=lbl)
                
            except KeyError as e:
                print(f"Missing {line} data in {key}: {e}")
                continue

        ax.set_ylabel(f'Intensity [{yunit}]')
        ax.set_title(line.replace('_', '-'))
        ax.grid(True)
        if line == valid_lines[0]:
            ax.legend()
        if line == valid_lines[-1]:
            ax.set_xlabel(f'Time [{xunit}]')
        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

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
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    else:
        labels = extract_labels_from_odc(odc, opt=label)

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
    
    if xlim is not None:
        plt.xlim(xlim)
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
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    else:
        labels = extract_labels_from_odc(odc, opt=label)

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
    
    if xlim is not None:
        plt.xlim(xlim)
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
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

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
    
    if xlim is not None:
        plt.xlim(xlim)
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
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    else:
        labels = extract_labels_from_odc(odc, opt=label)

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

    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()



"""
summary
"""

"""
global quantities
"""
# def summary_time_global_quantities
# def summary_time_global_quantities_beta_pol
# def summary_time_global_quantities_beta_tor
# def summary_time_global_quantities_beta_n
# def summary_time_global_quantities_w_mhd
# def summary_time_global_quantities_w_mag
# def summary_time_global_quantities_w_tot
# def summary_time_global_quantities_greenwald_density


"""
Not Routinely available signals
"""


"""
Thomson scattering
"""

def time_thomson_scattering_density(odc_or_ods, label='shot', xunit='s', yunit='m^-3', xlim='plasma'):
    """
    Plot Thomson scattering electron density time series per channel.
    
    Parameters:
        odc_or_ods: ODS/ODC
            Input data containing Thomson measurements
        label: str
            Legend labels option ('shot', 'key', 'run')
        xunit: str
            Time unit ('s', 'ms')
        yunit: str
            Density unit ('m^-3', 'cm^-3')
        xlim: str/list
            X-axis limits
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    else:
        labels = extract_labels_from_odc(odc, opt=label)

    # Determine channel count and radial positions from first ODS
    first_key = next(iter(odc.keys()))
    channels = list(odc[first_key]['thomson_scattering.channel'].keys())
    n_channels = len(channels)
    radial_positions = [odc[first_key][f'thomson_scattering.channel.{i}.position.r'] for i in range(n_channels)]

    # Create subplots
    fig, axs = plt.subplots(n_channels, 1, figsize=(10, 2.5*n_channels))
    if n_channels == 1:
        axs = [axs]
    
    # Plot each channel in its own subplot
    for ax, (channel, r_pos) in enumerate(zip(axs, radial_positions)):
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                time = ods['thomson_scattering.time']
                if xunit == 'ms':
                    time = time * 1e3
                
                data = unumpy.nominal_values(ods[f'thomson_scattering.channel.{channel}.n_e.data'])
                err = unumpy.std_devs(ods[f'thomson_scattering.channel.{channel}.n_e.data'])
                
                if yunit == 'cm^-3':
                    data = data / 1e6
                    err = err / 1e6
                
                ax.errorbar(time, data, yerr=err, label=lbl)
                
            except KeyError as e:
                print(f"Missing density data for channel {channel} in {key}: {e}")
                continue

        ax.set_ylabel(f'n_e [{yunit}]')
        ax.set_title(f'R = {r_pos:.3f} m')
        ax.grid(True)
        if channel == 0:
            ax.legend()
        if channel == n_channels-1:
            ax.set_xlabel(f'Time [{xunit}]')
        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

def time_thomson_scattering_temperature(odc_or_ods, label='shot', xunit='s', yunit='eV', xlim='plasma'):
    """
    Plot Thomson scattering electron temperature time series per channel.
    
    Parameters:
        odc_or_ods: ODS/ODC
            Input data containing Thomson measurements
        label: str
            Legend labels option ('shot', 'key', 'run')
        xunit: str
            Time unit ('s', 'ms')
        yunit: str
            Temperature unit ('eV', 'keV')
        xlim: str/list
            X-axis limits
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    # Handle xlim
    if xlim == 'none':
        xlim = None
    elif xlim == 'plasma':
        xlim = set_xlim_time(odc, type='plasma')
    elif xlim == 'coil':
        xlim = set_xlim_time(odc, type='coil')
    elif isinstance(xlim, list) and len(xlim) == 2:
        xlim = xlim
    else:
        print(f"Invalid xlim: {xlim}, using default 'plasma'")
        xlim = set_xlim_time(odc, type='plasma')

    # Handle labels
    if isinstance(label, list) and len(label) == len(odc.keys()):
        labels = label
    else:
        labels = extract_labels_from_odc(odc, opt=label)

    # Determine channels and radial positions
    first_key = next(iter(odc.keys()))
    channels = list(odc[first_key]['thomson_scattering.channel'].keys())
    n_channels = len(channels)
    radial_positions = [odc[first_key][f'thomson_scattering.channel.{i}.position.r'] for i in range(n_channels)]

    # Create subplots
    fig, axs = plt.subplots(n_channels, 1, figsize=(10, 2.5*n_channels))
    if n_channels == 1:
        axs = [axs]

    # Plot each channel
    for ax, (channel, r_pos) in enumerate(zip(axs, radial_positions)):
        for key, lbl in zip(odc.keys(), labels):
            ods = odc[key]
            try:
                time = ods['thomson_scattering.time']
                if xunit == 'ms':
                    time = time * 1e3
                
                data = unumpy.nominal_values(ods[f'thomson_scattering.channel.{channel}.t_e.data'])
                err = unumpy.std_devs(ods[f'thomson_scattering.channel.{channel}.t_e.data'])
                
                if yunit == 'keV':
                    data = data / 1e3
                    err = err / 1e3
                
                ax.errorbar(time, data, yerr=err, label=lbl)
                
            except KeyError as e:
                print(f"Missing temperature data for channel {channel} in {key}: {e}")
                continue

        ax.set_ylabel(f'T_e [{yunit}]')
        ax.set_title(f'R = {r_pos:.3f} m')
        ax.grid(True)
        if channel == 0:
            ax.legend()
        if channel == n_channels-1:
            ax.set_xlabel(f'Time [{xunit}]')
        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

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


