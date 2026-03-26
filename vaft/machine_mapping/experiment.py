"""
Experimental data processing module for VEST database.

This module provides functions for processing raw diagnostic data
and converting it to OMAS/IMAS data structure format.
Each function processes both static (geometry) and dynamic (measurement) data.
"""

import os
import vaft
import yaml
import numpy as np
from omas import ODS
from typing import Tuple, Dict, Any, Optional
import scipy.signal as signal
from datetime import datetime
from scipy.io import loadmat
from uncertainties import unumpy

# Centralize shared helpers in utils.py (IDS modules import from there).
from . import utils as _utils

def load_raw_data(source: str, field: str, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw data from either a shot or file source.
    
    Args:
        source: Source for data (file path or shot number)
        field: Field name to load
        options: Optional dictionary containing loading options
            - source_type: Optional str ('file' or 'shot')
            - file_format: Optional str for file format if source_type is 'file'
    
    Returns:
        Tuple of (time, data) arrays
    """
    # Legacy wrapper (kept for compatibility within this module).
    return _utils.load_raw_data(source, field, options)

def process_signal(time: np.ndarray, data: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process signal data with various options.
    
    Args:
        time: Time array
        data: Data array
        options: Optional dictionary containing processing options
            - time_range: Optional tuple (tstart, tend) to limit time range
            - resample: Optional bool to control resampling
            - filter_params: Optional dict for signal filtering parameters
                - type: str ('lowpass', 'highpass', 'bandpass')
                - cutoff: float or tuple of floats
                - order: int
    
    Returns:
        Tuple of (processed_time, processed_data) arrays
    """
    # Legacy wrapper (kept for compatibility within this module).
    return _utils.process_signal(time, data, options)

def get_diagnostic_info(source: str, diagnostic_type: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get diagnostic information from the source.
    
    Args:
        source: Source for data (file path or shot number)
        diagnostic_type: Type of diagnostic
        options: Optional dictionary containing options
            - source_type: Optional str ('file' or 'shot')
            - info_file: Optional str for info file path if source_type is 'file'
    
    Returns:
        Dictionary containing diagnostic information
    """
    # Legacy wrapper (kept for compatibility within this module).
    return _utils.get_diagnostic_info(source, diagnostic_type, options)

def get_static_info(source: str, diagnostic_type: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get static information for a diagnostic from the VEST configuration.
    
    Args:
        source: Source for data (file path or shot number)
        diagnostic_type: Type of diagnostic
        options: Optional dictionary containing options
            - source_type: Optional str ('file' or 'shot')
            - info_file: Optional str for info file path if source_type is 'file'
    
    Returns:
        Dictionary containing static information
    """
    # Legacy wrapper (kept for compatibility within this module).
    return _utils.get_static_info(source, diagnostic_type, options)

def process_static_geometry(ods: ODS, diagnostic_type: str, static_info: Dict[str, Any]) -> None:
    """
    Process static geometry information for a diagnostic.
    
    Args:
        ods: OMAS data structure
        diagnostic_type: Type of diagnostic
        static_info: Dictionary containing static information
    """
    # Legacy wrapper (kept for compatibility within this module).
    _utils.process_static_geometry(ods, diagnostic_type, static_info)

def process_static_channels(ods: ODS, diagnostic_type: str, static_info: Dict[str, Any]) -> None:
    """
    Process static channel information for a diagnostic.
    
    Args:
        ods: OMAS data structure
        diagnostic_type: Type of diagnostic
        static_info: Dictionary containing static information
    """
    # Legacy wrapper (kept for compatibility within this module).
    _utils.process_static_channels(ods, diagnostic_type, static_info)

def pf_active(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    from .pf_active import pf_active as _pf_active

    _pf_active(ods, static_source, dynamic_source, options)

def filterscope(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    from .filterscope import filterscope as _filterscope

    _filterscope(ods, static_source, dynamic_source, options)

def barometry(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    from .barometry import barometry as _barometry

    _barometry(ods, static_source, dynamic_source, options)

def tf(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    """
    Process TF coil data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Optional dictionary containing processing options
            - time_range: Optional tuple (tstart, tend) to limit time range
            - resample: Optional bool to control resampling
            - filter_params: Optional dict for signal filtering parameters
            - static_source_type: Optional str ('file' or 'shot')
            - dynamic_source_type: Optional str ('file' or 'shot')
    """
    from .tf import tf as _tf

    _tf(ods, static_source, dynamic_source, options)

def magnetics_flux_loop(ods: ODS, static_source: str, dynamic_source: str, options: dict) -> None:
    """
    Process flux loop data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Dictionary containing processing options
    """
    # Process static data
    static_info = get_static_info(static_source, 'flux_loop', options)
    process_static_geometry(ods, 'flux_loop', static_info)
    process_static_channels(ods, 'flux_loop', static_info)
    
    # Process dynamic data
    info = get_diagnostic_info(dynamic_source, 'flux_loop', {
        'source_type': options.get('dynamic_source_type', 'shot'),
        'info_file': options.get('info_file', 'vest.yaml')
    })
    
    # Collect all flux loop data
    time = None
    raw_data = []
    gains = []
    labels = []
    
    for loop_idx in info['labels'].keys():
        label = info['labels'][loop_idx]
        field = info['fields'][loop_idx]
        gain = info['gains'][loop_idx]
        
        # Load data
        t, data = load_raw_data(dynamic_source, field, {
            'source_type': options.get('dynamic_source_type', 'shot'),
            'file_format': options.get('file_format', 'mat')
        })
        
        if time is None:
            time = t
        raw_data.append(data)
        gains.append(gain)
        labels.append(label)
    
    # Convert to numpy arrays
    raw_data = np.array(raw_data).T  # Shape: (time, channels)
    gains = np.array(gains)
    
    # Process flux loop data using vaft.process
    from vaft.process.magnetics import flux_loop_flux
    time, processed_data, baselines = flux_loop_flux(
        time=time,
        raw=raw_data,
        gain=gains,
        baseline_onset=options.get('baseline_onset', 0.27),
        baseline_offset=options.get('baseline_offset', 0.28),
        baseline_type=options.get('baseline_type', 'linear'),
        baseline_onset_window=options.get('baseline_onset_window', 500),
        baseline_offset_window=options.get('baseline_offset_window', 100)
    )
    
    # Store in ODS
    for i, (label, flux) in enumerate(zip(labels, processed_data.T)):
        if i == 0:
            ods[f'flux_loop.loop.{i}.flux.time'] = time
        ods[f'flux_loop.loop.{i}.flux.data'] = flux
        ods[f'flux_loop.loop.{i}.voltage.data'] = raw_data[:, i] * gains[i]
        ods[f'flux_loop.loop.{i}.name'] = label

def magnetics_b_field_pol_probe(ods: ODS, static_source: str, dynamic_source: str, options: dict) -> None:
    """
    Process poloidal field probe data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Dictionary containing processing options
    """
    # Process static data
    static_info = get_static_info(static_source, 'b_field_pol_probe', options)
    process_static_geometry(ods, 'b_field_pol_probe', static_info)
    process_static_channels(ods, 'b_field_pol_probe', static_info)
    
    # Process dynamic data
    info = get_diagnostic_info(dynamic_source, 'b_field_pol_probe', {
        'source_type': options.get('dynamic_source_type', 'shot'),
        'info_file': options.get('info_file', 'vest.yaml')
    })
    
    # Collect all probe data
    raw_data = []
    gains = []
    labels = []
    
    for probe_idx in info['labels'].keys():
        label = info['labels'][probe_idx]
        field = info['fields'][probe_idx]
        gain = info['gains'][probe_idx]
        
        # Load data
        t, data = load_raw_data(dynamic_source, field, {
            'source_type': options.get('dynamic_source_type', 'shot'),
            'file_format': options.get('file_format', 'mat')
        })
        
        raw_data.append(data)
        gains.append(gain)
        labels.append(label)
    
    # Convert to numpy arrays
    raw_data = np.array(raw_data).T  # Shape: (time, channels)
    gains = np.array(gains)
    
    # Process B-field probe data using vaft.process
    from vaft.process.magnetics import b_field_pol_probe_field
    lowpass_param = options.get('lowpass_param', 0.01)
    raw, filtered_raw, integrated_flux, field, baselines = b_field_pol_probe_field(
        time=t,  # Use the last loaded time array
        raw=raw_data,
        gain=gains,
        lowpass_param=lowpass_param,
        baseline_onset=options.get('baseline_onset', 0.27),
        baseline_offset=options.get('baseline_offset', 0.28),
        baseline_type=options.get('baseline_type', 'linear'),
        baseline_onset_window=options.get('baseline_onset_window', 500),
        baseline_offset_window=options.get('baseline_offset_window', 100)
    )
    
    # Store in ODS
    for i, (label, field_data) in enumerate(zip(labels, field.T)):
        if i == 0:
            ods[f'b_field_pol_probe.probe.{i}.field.time'] = t
        ods[f'b_field_pol_probe.probe.{i}.field.data'] = field_data
        ods[f'b_field_pol_probe.probe.{i}.voltage.data'] = raw_data[:, i] * gains[i]
        ods[f'b_field_pol_probe.probe.{i}.name'] = label

def magnetics_ip(ods: ODS, static_source: str, dynamic_source: str, options: dict) -> None:
    """
    Process Rogowski coil and plasma current data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Dictionary containing processing options
    """
    # Process static data
    static_info = get_static_info(static_source, 'rogowski_coil', options)
    process_static_geometry(ods, 'rogowski_coil', static_info)
    process_static_channels(ods, 'rogowski_coil', static_info)
    
    # Process dynamic data
    info = get_diagnostic_info(dynamic_source, 'rogowski_coil', {
        'source_type': options.get('dynamic_source_type', 'shot'),
        'info_file': options.get('info_file', 'vest.yaml')
    })
    
    # Get Rogowski coil and flux loop data
    rogowski_label = info['labels']['0']
    rogowski_field = info['fields']['0']
    rogowski_gain = info['gains']['0']
    flux_loop_field = info.get('fl_field', {}).get('0')
    effective_res = info.get('effective_res', {}).get('0', 5.8e-4)
    
    # Load Rogowski coil data
    time, rogowski_raw = load_raw_data(dynamic_source, rogowski_field, {
        'source_type': options.get('dynamic_source_type', 'shot'),
        'file_format': options.get('file_format', 'mat')
    })
    
    # Load flux loop data if available
    if flux_loop_field:
        _, flux_loop_raw = load_raw_data(dynamic_source, flux_loop_field, {
            'source_type': options.get('dynamic_source_type', 'shot'),
            'file_format': options.get('file_format', 'mat')
        })
    else:
        flux_loop_raw = None
    
    # Process plasma current using vaft.process
    from vaft.process.magnetics import rogowski_coil_ip
    time, ip = rogowski_coil_ip(
        time=time,
        rogowski_raw=rogowski_raw,
        flux_loop_raw=flux_loop_raw,
        flux_loop_gain=11,  # Default value, should be configurable
        effective_vessel_res=effective_res,
        baseline_onset=options.get('baseline_onset', 0.27),
        baseline_offset=options.get('baseline_offset', 0.28),
        baseline_type=options.get('baseline_type', 'linear'),
        baseline_onset_window=options.get('baseline_onset_window', 500),
        baseline_offset_window=options.get('baseline_offset_window', 100)
    )
    
    # Store in ODS
    ods['rogowski_coil.coil.0.current.time'] = time
    ods['rogowski_coil.coil.0.current.data'] = ip
    ods['rogowski_coil.coil.0.voltage.data'] = rogowski_raw * rogowski_gain
    ods['rogowski_coil.coil.0.name'] = rogowski_label

def magnetics(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    """
    Process all magnetic diagnostics data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Optional dictionary containing processing options
            - time_range: Optional tuple (tstart, tend) to limit time range
            - resample: Optional bool to control resampling
            - filter_params: Optional dict for signal filtering parameters
            - static_source_type: Optional str ('file' or 'shot')
            - dynamic_source_type: Optional str ('file' or 'shot')
            - baseline_onset: Optional float for baseline onset time (default: 0.27)
            - baseline_offset: Optional float for baseline offset time (default: 0.28)
            - baseline_type: Optional str for baseline fitting type (default: 'linear')
            - baseline_onset_window: Optional int for baseline onset window (default: 500)
            - baseline_offset_window: Optional int for baseline offset window (default: 100)
    """
    from .magnetics import magnetics as _magnetics

    _magnetics(ods, static_source, dynamic_source, options)

def ion_doppler_spectroscopy(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    """
    Process ion Doppler spectroscopy data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Optional dictionary containing processing options
            - time_range: Optional tuple (tstart, tend) to limit time range
            - resample: Optional bool to control resampling
            - filter_params: Optional dict for signal filtering parameters
            - static_source_type: Optional str ('file' or 'shot')
            - dynamic_source_type: Optional str ('file' or 'shot')
    """
    # Implementation will be added
    pass

def spectrometer_uv(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    """
    Process UV spectrometer data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Optional dictionary containing processing options
            - time_range: Optional tuple (tstart, tend) to limit time range
            - resample: Optional bool to control resampling
            - filter_params: Optional dict for signal filtering parameters
            - static_source_type: Optional str ('file' or 'shot')
            - dynamic_source_type: Optional str ('file' or 'shot')
    """
    # Implementation will be added
    pass

def camera_visible(ods: ODS, static_source: str, dynamic_source: str, options: dict = None) -> None:
    """
    Process visible camera data.
    
    Args:
        ods: OMAS data structure
        static_source: Source for static/geometry data (file path or shot number)
        dynamic_source: Source for dynamic/measurement data (file path or shot number)
        options: Optional dictionary containing processing options
            - time_range: Optional tuple (tstart, tend) to limit time range
            - resample: Optional bool to control resampling
            - filter_params: Optional dict for signal filtering parameters
            - static_source_type: Optional str ('file' or 'shot')
            - dynamic_source_type: Optional str ('file' or 'shot')
    """
    # Implementation will be added
    pass

# # =============================================================================
# # Thomson Scattering
# # =============================================================================

def thomson_scattering(ods, shotnumber, filepath=None):
    """
    Set up static properties for Thomson scattering in the ODS object for VEST tokamak.

    This function populates the 'thomson_scattering' section of the given ODS object
    with static (time-independent) data such as positions and names of the channels.
    
    Load dynamic Thomson scattering data from a .mat file into the ODS object for VEST tokamak.

    This function reads electron temperature and density data from a MATLAB .mat file
    for a given shot number and populates the 'thomson_scattering' section of the ODS.

    Parameters:
        ods (ODS): The OMAS data structure to populate.
        shotnumber (int): The shot number to load data for.
        base_path (str, optional): The base directory containing the data files.
            Defaults to the current working directory.

    Parameters:
        ods (ODS): The OMAS data structure to populate.
    """
    from .thomson_scattering import thomson_scattering as _thomson_scattering

    _thomson_scattering(ods, shotnumber, filepath)


def charge_exchange(ods: ODS, filepath: Optional[str] = None) -> None:
    """
    Load sample Charge Exchange Spectroscopy (CES) data into the charge_exchange IDS.

    This implementation is a minimal example based on a single CES file (CES_47514.mat)
    containing one time frame and 10 LOS points. It maps LOS position, ion temperature,
    and toroidal velocity (with 1-sigma uncertainties) into the OMAS charge_exchange IDS.

    Args:
        ods: OMAS data structure to populate.
        filepath: Optional explicit path to a `CES_<shot>.mat` file. If omitted,
            `vaft/data/CES_47514.mat` is used. The shot number is inferred from
            `ods['dataset_description.data_entry.pulse']` when available, otherwise
            from the filename pattern.
    """
    from .charge_exchange import charge_exchange as _charge_exchange

    _charge_exchange(ods, filepath)

def raw_database_info(file: str, shot: int, key: str) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve data for all channels of a system from a YAML file in dictionary format.

    Args:
        file (str): Path to the YAML file.
        shot (int): Shot number to retrieve the data for.
        key (str): Key to retrieve the data for (e.g., 'tf').

    Returns:
        dict: A dictionary containing labels, fields, and gains indexed by channel.
              Example: {'labels': {'0': 'TF Coil 1', ...},
                        'fields': {'0': 1, ...},
                        'gains': {'0': -30000, ...}}
    """
    # Legacy wrapper (kept for compatibility within this module).
    return _utils.raw_database_info(file, shot, key)

def tf_from_raw_database(ods: ODS, shot: int, tstart: float, tend: float, dt: float) -> None:
    """
    Process TF coil data from the VEST raw database.
    
    Args:
        ods: ODS structure
        shot: Shot number
        tstart: Start time
        tend: End time
        dt: Time step
    """
    from .tf import tf_from_raw_database as _tf_from_raw_database

    _tf_from_raw_database(ods, shot, tstart, tend, dt)

def pf_active_from_raw_database(ods: ODS, shot: int, tstart: float, tend: float, dt: float) -> None:
    """
    Process PF active coil data from the VEST raw database.
    
    Args:
        ods: ODS structure
        shot: Shot number
        tstart: Start time
        tend: End time
        dt: Time step
    """
    # Update static ODS structure
    vaft.machine_mapping.pf_active_static(ods)
    
    info = raw_database_info('raw_database.yaml', shot, 'pf_active')
    
    # Process each PF coil
    for coil_idx in info['labels'].keys():
        label = info['labels'][coil_idx]
        field = info['fields'][coil_idx]
        gain = info['gains'][coil_idx]
        
        time, data = vaft.database.raw.load(shot, field)
        current = data * gain
        
        # Store in ODS
        ods[f'pf_active.coil.{coil_idx}.current.time'] = time
        ods[f'pf_active.coil.{coil_idx}.current.data'] = current
        ods[f'pf_active.coil.{coil_idx}.name'] = label

def filterscope_from_raw_database(ods: ODS, shot: int, tstart: float, tend: float, dt: float) -> None:
    """
    Process filterscope data from the VEST raw database.
    
    Args:
        ods: ODS structure
        shot: Shot number
        tstart: Start time
        tend: End time
        dt: Time step
    """
    # Update static ODS structure
    vaft.machine_mapping.filterscope_static(ods)
    
    info = raw_database_info('raw_database.yaml', shot, 'filterscope')
    
    # Process each filterscope channel
    for ch_idx in info['labels'].keys():
        label = info['labels'][ch_idx]
        field = info['fields'][ch_idx]
        gain = info['gains'][ch_idx]
        
        time, data = vaft.database.raw.load(shot, field)
        signal = data * gain
        
        # Store in ODS
        ods[f'filterscope.channel.{ch_idx}.signal.time'] = time
        ods[f'filterscope.channel.{ch_idx}.signal.data'] = signal
        ods[f'filterscope.channel.{ch_idx}.name'] = label

def barometry_from_raw_database(ods: ODS, shot: int, tstart: float, tend: float, dt: float) -> None:
    """
    Process barometry data from the VEST raw database.
    
    Args:
        ods: ODS structure
        shot: Shot number
        tstart: Start time
        tend: End time
        dt: Time step
    """
    # Update static ODS structure
    vaft.machine_mapping.barometry_static(ods)
    
    info = raw_database_info('raw_database.yaml', shot, 'barometry')
    
    # Process each pressure gauge
    for gauge_idx in info['labels'].keys():
        label = info['labels'][gauge_idx]
        field = info['fields'][gauge_idx]
        gain = info['gains'][gauge_idx]
        
        time, data = vaft.database.raw.load(shot, field)
        pressure = data * gain  # Convert to Pa if needed
        
        # Store in ODS
        ods[f'barometry.gauge.{gauge_idx}.pressure.time'] = time
        ods[f'barometry.gauge.{gauge_idx}.pressure.data'] = pressure
        ods[f'barometry.gauge.{gauge_idx}.name'] = label


def flux_loop_from_raw_database(ods: ODS, shot: int) -> None:
    raise NotImplementedError(
        "flux_loop_from_raw_database is not implemented in this legacy module. "
        "Use IDS-based machine mapping modules instead."
    )


def b_field_pol_probe_from_raw_database(ods: ODS, shot: int) -> None:
    raise NotImplementedError(
        "b_field_pol_probe_from_raw_database is not implemented in this legacy module. "
        "Use IDS-based machine mapping modules instead."
    )


def rogowski_coil_and_ip_from_raw_database(ods: ODS, shot: int) -> None:
    raise NotImplementedError(
        "rogowski_coil_and_ip_from_raw_database is not implemented in this legacy module. "
        "Use IDS-based machine mapping modules instead."
    )


def magnetics_from_raw_database(ods: ODS, shot: int, tstart: float, tend: float, dt: float) -> None:
    """
    Process all magnetic diagnostics data from the VEST raw database.
    
    Args:
        ods: ODS structure
        shot: Shot number
        tstart: Start time
        tend: End time
        dt: Time step
    """
    # Update static ODS structure
    vaft.machine_mapping.magnetics_static(ods)
    
    # Process each type of magnetic diagnostic
    flux_loop_from_raw_database(ods, shot)
    b_field_pol_probe_from_raw_database(ods, shot)
    rogowski_coil_and_ip_from_raw_database(ods, shot)

def calculate_em_coupling_from_raw_database(ods: ODS) -> None:
    """
    Calculate electromagnetic coupling from raw database data.
    
    Args:
        ods: ODS structure
    """
    # Update static ODS structure
    vaft.machine_mapping.em_coupling_static(ods)
    
    # Calculate mutual inductances and resistances
    # This is a placeholder - actual implementation would depend on the specific requirements
    # and available data in the raw database
    pass

def dataset_description_from_raw_database(ods: ODS, shot: int, options: dict | None = None) -> None:
    """
    Legacy helper to populate the ``dataset_description`` IDS for data that
    originates from the VEST raw database.

    This is now a thin wrapper around :func:`vaft.machine_mapping.dataset_description`
    to ensure there is a single, schema-consistent implementation.

    Parameters
    ----------
    ods:
        OMAS data structure.
    shot:
        Shot number associated with this dataset.
    options:
        Optional dictionary forwarded to ``dataset_description``. Any keys
        accepted there (e.g. ``run``, ``description``, ``user``, ``machine``,
        ``dd_version``, ``imas_version``) may be provided here.
    """
    if options is None:
        options = {}

    # Default description specific to raw-database imports while still allowing
    # callers to override it.
    if 'description' not in options:
        options['description'] = 'VEST dataset imported from raw database'

    # Ensure we tag this as a shot-based entry.
    options.setdefault('source_type', 'shot')

    # Delegate to the canonical implementation in vaft.machine_mapping.meta.
    vaft.machine_mapping.dataset_description(ods, shot, options)

# """
# Mangetics
# """

# def flux_loop_from_raw_database(ods, shot):
#     """
#     Process flux loop data from the VEST raw database.
#     """
#     # Update static ODS structure

#     info = raw_database_info(file = 'raw_database.yaml', shot, 'flux_loop')
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']

#     (time, data) = vaft.database.raw.load(shot,field)
#     (time, BtR, BtZ, BtPhi) = vaft.process.flux_loop(time,data,gain)

#     FL=ods['flux_loop']
#     FL['ids_properties.comment'] = 'Flux loop data from VEST raw database'
#     FL['ids_properties.homogeneous_time'] = 1
#     FL['time']=time
#     FL['b_field_tor_vacuum_r.data']=BtR
#     FL['b_field_tor_vacuum_z.data']=BtZ
#     FL['b_field_tor_vacuum_phi.data']=BtPhi

# def b_field_pol_probe_from_raw_database(ods, shot):
#     """
#     Process B-field poloidal probe data from the VEST raw database.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.b_field_pol_probe_static(ods)

#     info = raw_database_info(file = 'raw_database_info.yaml', shot, 'b_field_pol_probe')
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']

#     # setting
#     lowpass_param = 0.01
#     baseline_onset, _ = vaft.omas.general.find_pf_active_onset(ods)
#     baseline_offset = 0.28
#     baseline_type = 'linear'
#     baseline_onset_window = 500
#     baseline_offset_window = 100
#     plot_opt = False

#     (time, data) = vaft.database.raw.load(shot,field)
#     (time, BtR, BtZ, BtPhi) = vaft.process.b_field_pol_probe_field(
#         time, data, gain, lowpass_param, baseline_onset, baseline_offset, baseline_type, baseline_onset_window, baseline_offset_window, plot_opt)
#     BP=ods['b_field_pol_probe']
#     BP['ids_properties.comment'] = 'B-field poloidal probe data from VEST raw database'
#     BP['ids_properties.homogeneous_time'] = 1
#     BP['time']=time
#     BP['b_field_tor_vacuum_r.data']=BtR
#     BP['b_field_tor_vacuum_z.data']=BtZ
#     BP['b_field_tor_vacuum_phi.data']=BtPhi

# def rogowski_coil_and_ip_from_raw_database(ods, shot):
#     """
#     Process Rogowski coil and Ip data from the VEST raw database.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.rogowski_coil_and_ip_static(ods)

#     info = raw_database_info(file = 'raw_database_info.yaml', shot, 'rogowski_coil')
#     label = info['labels']['0']
#     field = info['fields']['0']
#     gain = info['gains']['0']
#     fl_field = info['fl_field']['0']
#     effective_res = info['effective_res']['0']

#     (rogowski_time, rogowski_raw) = vaft.database.raw.load(shot,field)
#     (time_fl, fl_law) = vaft.database.raw.load(shot,fl_field)

#     (time, ip) = vaft.process.ip(time,rogowski_raw,gain,fl_law,effective_res)

#     RC=ods['rogowski_coil']
#     RC['ids_properties.comment'] = 'Rogowski coil and Ip data from VEST raw database'
#     RC['ids_properties.homogeneous_time'] = 1
#     RC['time']=time
#     RC['current.data']=Ip

# def diamagnetic_flux_from_raw_database(ods, shot):

# # def internal_magnetic_probe_array_dynamic(ods, shot):


# def magnetics_from_raw_database(ods, shot):
#     """
#     Process magnetics data from the VEST raw database.
#     """
#     # Update static ODS structure
#     vaft.machine_mapping.magnetics_static(ods)

#     # Load raw data from database, post-process, and map routinely available magnetic diagnostics to ODS

    

# # =============================================================================
# # Ion Doppler Spectroscopy
# # =============================================================================

# def ion_doppler_spectroscopy_from_file(
#     ods: Dict[str, Any],
#     shotnumber: int,
#     options: str = 'single'
# ) -> None:
#     """
#     Load ion Doppler spectroscopy data into ODS['charge_exchange'].

#     :param ods: ODS structure.
#     :param shotnumber: Shot number.
#     :param options: 'single' or 'profile'.
#     """
#     ods['charge_exchange.ids_properties.homogeneous_time'] = 1
#     if options == 'single':
#         print("read_doppler_single(ods, shotnumber) stub")
#     elif options == 'profile':
#         print("read_doppler_profile(ods, shotnumber) stub")


# # =============================================================================
# # Fast Camera
# # =============================================================================

# def vfit_fastcamera_from_file(ods: Dict[str, Any], shotnumber: int) -> None:
#     """
#     Load fast camera frames from local .bmp for ODS['camera_visible'].

#     :param ods: ODS structure.
#     :param shotnumber: Shot number.
#     """
#     vfit_camera_visible(ods, shotnumber)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Refactored EFIT Workflow for VEST Data

# This script automates VEST diagnostic data retrieval (poloidal/toroidal fields,
# flux loops, etc.), computes eddy currents, generates EFIT constraints (k-files),
# and merges EFIT results back into ODS for further analysis and plotting.
# """


