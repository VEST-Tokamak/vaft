"""
This module contains functions for plotting time series data from OMAS ODS.
"""

from omas import *
import matplotlib.pyplot as plt
from vest.process import find_signal_onoffset, is_signal_active
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
                onset, offset = find_signal_onoffset(time, data)
                onsets.append(onset)
                offsets.append(offset)
                
            elif type == 'coil' and 'pf_active.coil' in ods:
                num_coils = len(ods['pf_active.coil'])
                for i in range(num_coils):
                    time = ods['pf_active.time']
                    data = ods[f'pf_active.coil.{i}.current.data']
                    onset, offset = find_signal_onoffset(time, data)
                    onsets.append(onset)
                    offsets.append(offset)
                    
        except KeyError as e:
            print(f"Missing key {str(e)} in ODS {key}")
            continue

    if not onsets or not offsets:
        return None
        
    return [np.min(onsets), np.max(offsets)]

"""
Routinely available signals : pf_active, ip, flux_loop, bpol_probe, filterscope, tf

Routinely available modelling : pf_passive, equilibrium
"""


"""
pf_coil
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
                    if f'pf_active.coil.{i}.current.data' in ods:
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
                ax.set_title(f'Coil {coil_idx} Current-Turns')
                ax.set_xlabel(f'Time [{xunit}]')
                ax.set_ylabel(f'Current-Turns [{yunit}]')
                ax.grid(True)
                ax.legend()
                
            except KeyError as e:
                print(f"Missing data for coil {coil_idx} in {key}: {e}")
                continue

        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

# turns = ods['pf_active.coil.0.element.:.turns_with_sign']
# total_turns = int(np.sum(np.abs(turns)))




"""
magnetics - Rogowski coil, Flux loop, Bpol_probe
"""


# def time_magnetics

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

# def _find_flux_loop_inboard_indices(ods):
#     # find the indices of the flux loop inboard
#     indices = np.where(ods['magnetics.flux_loop.:.position.0.r'] < 0.15)
#     return indices

# def _find_flux_loop_outboard_indices(ods):
#     # find the indices of the flux loop outboard
#     indices = np.where(ods['magnetics.flux_loop.:.position.0.r'] > 0.5)
#     return indices

# def magnetics_time_flux_loop_flux(ods_or_odc, labels=None):
#     odc = odc_or_ods_check(ods_or_odc)
#     if labels is None or len(labels) != len(odc.keys()):
#         labels = extract_labels_from_odc(odc)
#     fig, axs = plt.subplots(3, 4, figsize=(12, 8))
#     for key, label in zip(odc.keys(), labels):
#         time = odc[key]['magnetics.flux_loop.:.time']
#         flux = odc[key]['magnetics.flux_loop.:.flux.data']
#         axs[0, 0].plot(time, flux, label=label)
#         axs[0, 0].set_title("Flux")
#         axs[0, 0].legend()
#         axs[0, 0].grid(True)
#     plt.show()

# def magnetics_time_flux_loop_{voltage, flux}
# indices -> 'all', 'inboard', 'outboard'

# def _find_bpol_probe_inboard_indices(ods):
#     # find the indices of the bpol probe inboard
#     indices = np.where(ods['magnetics.b_field_pol_probe.:.position.r'] < 0.09)
#     return indices

# def _find_bpol_probe_outboard_indices(ods):
#     # find the indices of the bpol probe outboard
#     indices = np.where(ods['magnetics.b_field_pol_probe.:.position.r'] > 0.795)
#     return indices

# def _find_bpol_probe_side_indices(ods):
#     # find the indices of the bpol probe side
#     indices = np.where(np.abs(ods['magnetics.b_field_pol_probe.:.position.z']) > 0.8)
#     return indices

# def time_bpol_probe_all_voltage(ods_or_odc, labels=None):
#     odc = odc_or_ods_check(ods_or_odc)
#     if labels is None or len(labels) != len(odc.keys()):
#         labels = extract_labels_from_odc(odc)
#     fig, axs = plt.subplots(8, 8, figsize=(12, 8))
#     for key, label in zip(odc.keys(), labels):
#         time = odc[key]['magnetics.b_field_pol_probe.:.time']
#         voltage = odc[key]['magnetics.b_field_pol_probe.:.voltage.data']
#         axs[0, 0].plot(time, voltage, label=label)
#         axs[0, 0].set_title("Voltage")
#         axs[0, 0].legend()
#         axs[0, 0].grid(True)
#     plt.show()

# def magnetics_time_b_field_pol_probe_{voltage, flux, spectrogram}
# indices -> 'all', 'inboard', 'outboard', 'side'


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
filterscope
"""

# def filterscope_time_intensity
# def filterscope_time_spectrogram

# indices -> 'all', 'H_alpha', 'H_alpha_fast', 'C_II', 'C_III', 'O_I', 'O_II'



"""
TF coil
"""
# def time_tf_b_field_tor(ods):
#     ods = odc_or_ods_check(ods)
#     TF = ods['tf']
#     R0=TF['r0']
#     myy=TF['b_field_tor_vacuum_r.data']/R0
#     myx=TF['time']
#     myy2=TF['coil.0.current.data']
    
#     fig1=plt.figure(facecolor='white')
#     plt.plot(myx,myy,label='b_field_tor')

#     fig2=plt.figure(facecolor='white')
#     plt.plot(myx,myy2,label='current')

#     mystring="Shot: {} Run:{}".format(shot,run)
#     plt.title(mystring)
#     plt.legend()

#     plt.show()

# def time_tf_current(ods):
#     TF = ods['tf']
#     R0=TF['r0']
#     myy=TF['b_field_tor_vacuum_r.data']/R0
#     myx=TF['time']
#     myy2=TF['coil.0.current.data']
    
#     fig1=plt.figure(facecolor='white')
#     plt.plot(myx,myy,label='b_field_tor')

#     fig2=plt.figure(facecolor='white')
#     plt.plot(myx,myy2,label='current')

#     mystring="Shot: {} Run:{}".format(shot,run)
#     plt.title(mystring)
#     plt.legend()

#     plt.show()

# def time_tf_b_field_tor_vacuum_r

"""
eddy_current (pf_passive)
"""

# def time_pf_passive_total_current
# def time_pf_passive_duplicated_current

"""
Barometer (Vacuum Gauge or Neutral Pressure Gauge)
"""

# def time_barometer_pressure

"""
Not Routinely available signals
"""


"""
Thomson scattering
"""

# def time_thomson_scattering_all_density
# def time_thomson_scattering_all_temperature

# def time_thomson_scattering_indices_density
# def time_thomson_scattering_indices_temperature

"""
Ion Doppler Spectroscopy
"""

# def time_ion_doppler_spectroscopy_CIII_intensity
# def time_ion_doppler_spectroscopy_CIII_spectrogram
# def time_ion_doppler_spectroscopy_CIII_tor_velocity

"""
Interferometry
"""

"""

"""

