from vaft.process import psi_to_radial
import matplotlib.pyplot as plt
import numpy as np
from omas import ODS, ODC
from .utils import get_from_path, extract_labels_from_odc
from vaft.omas import odc_or_ods_check
# legend -> time_points sec (time_slice_index)
# axis -> radial, rho_tor_norm, psi_n, vertical...

# def {ods}_{axis}_{quantity}(ods, time_slice):

# def radial_equilibrium_summary
# def radial_equilibrium_q(ods, time_slice=None):
#     """Plot safety factor q profile from equilibrium data.
    
#     Args:
#         ods: OMAS data structure containing equilibrium data
#         time_slice: Time slice index to plot. If None, uses first time slice.
#     """
#     ods = ods_or_odc_check(ods)
#     if time_slice is None:
#         time_slice = ods['equilibrium.time_slice'].keys()[0]
        
#     # Get q profile data
#     q = ods[f'equilibrium.time_slice.{time_slice}.profiles_1d.q']

#     # Get rho coordinate and convert to radial coordinate
#     rho = ods[f'equilibrium.time_slice.{time_slice}.profiles_1d.rho_tor'] # Rho coordinate
#     # r = rho_to_radial(
    

# def radial_equilibrium_pressure
# def radial_equilibrium_j_tor
# def radial_equilibrium_pprime
# def onedim_radial_equilibrium_f
# def onedim_radial_equilibrium_ffprime

# def onedim_psi_plot(
# def onedim_psi_equilibrium_summary
# def onedim_psi_equilibrium_q
# def onedim_psi_equilibrium_pressure
# def onedim_psi_equilibrium_j_tor
# def onedim_psi_equilibrium_pprime
# def onedim_psi_equilibrium_f
# def onedim_psi_equilibrium_ffprime

# def onedim_rho_plot(
# def onedim_rho_equilibrium_summary
# def onedim_rho_equilibrium_q
# def onedim_rho_equilibrium_pressure
# def onedim_rho_equilibrium_j_tor
# def onedim_rho_equilibrium_pprime
# def onedim_rho_equilibrium_f
# def onedim_rho_equilibrium_ffprime



# def onedim_vertical_plot
# def onedim_vertical_magnetics_flux_loop_inboard_voltage
# def onedim_vertical_magnetics_flux_loop_inboard_flux
# def onedim_vertical_magnetics_flux_loop_outboard_voltage
# def onedim_vertical_magnetics_flux_loop_outboard_flux

# def onedim_vertical_magnetics_bpol_probe_inboard_voltage
# def onedim_vertical_magnetics_bpol_probe_inboard_flux
# def onedim_vertical_magnetics_bpol_probe_side_voltage
# def onedim_vertical_magnetics_bpol_probe_side_flux
# def onedim_vertical_magnetics_bpol_probe_outboard_voltage
# def onedim_vertical_magnetics_bpol_probe_outboard_flux

ONEDIM_PLOT_CONFIGS = {
    'equilibrium': [
        {
            'name': 'pressure',
            'path': 'equilibrium.time_slice.0.profiles_1d.pressure',
            'ylabel': 'Pressure [Pa]',
            'yunit': 'Pa',
            'coordinate': 'rho_tor_norm'
        },
        {
            'name': 'q',
            'path': 'equilibrium.time_slice.0.profiles_1d.q',
            'ylabel': 'Safety Factor q',
            'yunit': '',
            'coordinate': 'rho_tor_norm'
        },
        # ... 다른 equilibrium profiles
    ],
    'core_profiles': [
        {
            'name': 'ne',
            'path': 'core_profiles.profiles_1d.0.electrons.density',
            'ylabel': 'Electron Density [m^-3]',
            'yunit': 'm^-3',
            'coordinate': 'rho_tor_norm'
        },
        {
            'name': 'te',
            'path': 'core_profiles.profiles_1d.0.electrons.temperature',
            'ylabel': 'Electron Temperature [keV]',
            'yunit': 'keV',
            'coordinate': 'rho_tor_norm'
        },
        # ... 다른 core profiles
    ],
    # ... 다른 신호군들
}

# 3. 제너릭 1D 플로팅 함수
def plot_onedim_profile(odc_or_ods, data_path, ylabel, coordinate='rho_tor_norm', 
                       xlabel='Normalized Toroidal Flux', yunit='', label='shot', 
                       time_slice=0, **kwargs):
    """
    Generic 1D profile plotting function
    
    Parameters:
    -----------
    odc_or_ods : ODC or ODS
        ODC or ODS object containing the data
    data_path : str
        Path to the data in ODS format
    ylabel : str
        Label for y-axis
    coordinate : str
        Radial coordinate to use ('rho_tor_norm', 'psi_n', etc.)
    xlabel : str
        Label for x-axis
    yunit : str
        Unit for y-axis
    label : str
        Label for the plot
    time_slice : int
        Time slice to plot (for time-dependent profiles)
    **kwargs : dict
        Additional plotting parameters
    """
    odc = odc_or_ods_check(odc_or_ods)
    
    plt.figure(figsize=(10, 6))
    
    for key, lbl in zip(odc.keys(), [f"{label} {k}" for k in odc.keys()]):
        ods = odc[key]
        try:
            # Get radial coordinate
            rho = get_radial_coordinate(ods, coordinate)
            
            # Get profile data
            data = get_from_path(ods, data_path)
            
            if data is None or rho is None:
                print(f"Missing data for {key}")
                continue
                
            # Plot
            plt.plot(rho, data, label=lbl, **kwargs)
            
        except Exception as e:
            print(f"Error plotting {key}: {e}")
            continue
    
    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} [{yunit}]" if yunit else ylabel)
    plt.title(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. 자동 래퍼 함수 생성
for group, configs in ONEDIM_PLOT_CONFIGS.items():
    for cfg in configs:
        def make_plot_func(data_path, ylabel, coordinate, yunit):
            def plot_func(odc_or_ods, **kwargs):
                plot_onedim_profile(
                    odc_or_ods,
                    data_path,
                    ylabel,
                    coordinate=coordinate,
                    yunit=yunit,
                    **kwargs
                )
            return plot_func
        
        # Create function name
        func_name = f'plot_{group}_{cfg["name"]}'
        # Create and register the function
        globals()[func_name] = make_plot_func(
            cfg['path'],
            cfg['ylabel'],
            cfg['coordinate'],
            cfg['yunit']
        )

# 5. 테스트 코드
if __name__ == "__main__":
    ods = ODS()
    ods.sample()
    
    # Test all plotting functions
    for group, configs in ONEDIM_PLOT_CONFIGS.items():
        for cfg in configs:
            func_name = f'plot_{group}_{cfg["name"]}'
            func = globals().get(func_name)
            if func is not None:
                print(f"Testing {func_name}...")
                try:
                    func(ods)
                except Exception as e:
                    print(f"{func_name} failed: {e}")

# --- Placeholder/Simplified Utils ---
# In a package, these would likely come from a shared utils module (e.g., from .utils import ...)

# --- Configuration for 1D Plotting ---
COORDINATE_DEFINITIONS = {
    'psi_n': {
        'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.psi_n',
        'label': 'Normalized Poloidal Flux (ψ_N)'
    },
    'rho_tor_norm': {
        'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.rho_tor_norm',
        'label': 'Normalized Toroidal Flux (ρ_N)'
    },
    'r_major': {
        'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.r_major',
        'label': 'Major Radius (R) [m]'
    },
    'r_minor': {
        'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.r_minor',
        'label': 'Minor Radius (r) [m]'
    },
    # Future coordinates (r_major, r_minor, etc.) can be added here.
    # They might require a conversion function if not direct paths.
    # Example:
    # 'r_major': {
    #     'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.r', # Hypothetical direct path
    #     'label': 'Major Radius (R) [m]',
    #     # 'conversion_func': get_r_major_from_psi # If conversion is needed
    # },
}

TARGET_QUANTITIES = {
    'equilibrium': { # ODS group name
        'j_tor': { # Quantity name
            'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.j_tor',
            'ylabel': 'Toroidal Current Density (j_tor) [A/m^2]'
        },
        'pressure': {
             'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.pressure',
             'ylabel': 'Pressure [Pa]'
        },
        'q': {
            'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.q',
            'ylabel': 'Safety Factor (q)'
        },
        'pprime': {
            'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.pprime',
            'ylabel': "Pressure Gradient (P') [Pa/Wb]"
        },
        'f': {
            'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.f',
            'ylabel': 'Poloidal Current Function (F) [T*m]'
        },
        'ffprime': {
            'path_template': 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.ffprime',
            'ylabel': "FF' [T^2*m^2/Wb]"
        }
    }
}

# --- Data Retrieval ---
def _ensure_1d_numpy_array(data_in):
    """
    Ensures the input data is a 1D numpy array, suitable for plotting.
    Handles None, scalars, 1D arrays, column vectors (N,1), row vectors (1,N),
    and 2D matrices (N,M, taking the first column).
    Returns a contiguous 1D numpy array, a scalar, or None.
    """
    if data_in is None:
        return None
    
    # Convert to numpy array if it's not already one (e.g., list, OMAS data object)
    # Using np.asanyarray to avoid copying if already an ndarray.
    data = np.asanyarray(data_in)

    if data.ndim == 0: # Scalar value
        return data # Return scalar as is. Plotting functions usually handle scalars.
    elif data.ndim == 1: # Already 1D
        return np.ascontiguousarray(data)
    elif data.ndim == 2:
        # Case 1: Row vector (e.g., shape (1, N))
        if data.shape[0] == 1 and data.shape[1] >= 1:
            print(f"Info: Input data for plotting is a row vector (shape {data.shape}). Flattening to 1D.")
            return np.ascontiguousarray(data.flatten())
        # Case 2: Column vector (e.g., shape (N, 1))
        elif data.shape[1] == 1 and data.shape[0] >= 1:
            print(f"Info: Input data for plotting is a column vector (shape {data.shape}). Flattening to 1D.")
            return np.ascontiguousarray(data.flatten())
        # Case 3: True 2D matrix (N, M) where M > 1
        elif data.shape[1] > 1 : 
            print(f"Warning: Input data for plotting is 2D (shape {data.shape}). Using first column.")
            return np.ascontiguousarray(data[:, 0])
        # Case 4: Fallback for other 2D shapes e.g. (N,0), (0,M), (0,0) or (N,1) if not caught by column vector.
        # For (N,0) or (0,M) or (0,0) arrays, flatten produces an empty 1D array.
        # For (N,1) if it somehow reaches here, flatten is fine.
        else: 
            print(f"Info: Input data for plotting is 2D (shape {data.shape}) with an unusual structure or is empty. Attempting to flatten. Plot may be empty.")
            return np.ascontiguousarray(data.flatten())
    else: # More than 2 dimensions
        print(f"Error: Input data for plotting has {data.ndim} dimensions (shape {data.shape}). Cannot process as 1D.")
        return None

def get_1d_profile_data(ods, time_slice_idx, quantity_path_template, coord_path_template):
    """
    Fetches 1D profile data (quantity and coordinate) for a given ODS and time slice index.
    Handles string vs. int time_slice_idx for path formatting.
    """
    try:
        ts = ods['equilibrium.time_slice'][time_slice_idx]
        profiles_1d = ts['profiles_1d']

        if coord_path_template.endswith('r_major') or coord_path_template.endswith('r_minor'):
            r_in = np.ascontiguousarray(profiles_1d['r_inboard'])
            r_out = np.ascontiguousarray(profiles_1d['r_outboard'])
            
            # Assuming the quantity to plot (e.g., j_tor) is stored under its own name in profiles_1d
            # And that it corresponds to the full radial profile (like psi_n or rho_tor_norm)
            # We need to map this quantity to r_in and r_out. This is tricky without more info.
            # For now, let's assume the quantity path gives us the full profile, and we need to select parts.
            # This part likely needs ODS structure expertise if j_tor isn't directly split for r_in/r_out.

            # Let's re-fetch the target quantity using its direct path template for r_major/r_minor cases
            # This ensures we get the quantity defined by TARGET_QUANTITIES, not just hardcoded 'j_tor'
            quantity_path_for_rmaj_rmin = quantity_path_template.format(time_slice_idx=str(time_slice_idx))
            quantity_keys_for_rmaj_rmin = quantity_path_for_rmaj_rmin.split('.')
            _quantity_data_full_profile_raw = ods # temp var for raw data
            for key in quantity_keys_for_rmaj_rmin:
                _quantity_data_full_profile_raw = _quantity_data_full_profile_raw[key]

            quantity_data_full_profile = _ensure_1d_numpy_array(_quantity_data_full_profile_raw)
            
            if quantity_data_full_profile is None:
                 print(f"Error: Could not process quantity data from path '{quantity_path_for_rmaj_rmin}' for r_major/r_minor as 1D. It might be >2D or of an unsupported type.")
                 return None, None
            # quantity_data_full_profile is now guaranteed to be 1D, scalar or None

            # This assumes quantity_data_full_profile has same length as a standard radial coordinate like psi_n
            # And that r_in and r_out map to the first len(r_in) and next len(r_out) points of this full profile
            # This is a strong assumption. OMAS structure might have specific j_tor for inboard/outboard.
            # If profiles_1d.j_tor has len(r_in) + len(r_out) points, it is simpler.
            # Assuming profiles_1d.j_tor is a single array corresponding to r_in and r_out concatenated
            # Let's assume quantity_data_full_profile from TARGET_QUANTITIES is THE array to use and split.
            # This is often the case if j_tor path points to a single j_tor array.

            num_r_in = len(r_in)
            num_r_out = len(r_out)

            # If j_tor is structured to have inboard and outboard parts directly:
            # quantity_in = quantity_data_full_profile[:num_r_in] # Or some other OMAS specific path
            # quantity_out = quantity_data_full_profile[num_r_in:num_r_in+num_r_out]
            # For now, assume quantity_data_full_profile covers a grid that r_in and r_out also cover.
            # A common scenario is that r_in and r_out are halves of a full radial grid, and j_tor corresponds to that full grid.
            # If j_tor in ODS is already split, the path should reflect that. 
            # Given current `TARGET_QUANTITIES` points to a single `j_tor` path, we assume it's a single profile.
            # The most robust way would be to interpolate j_tor onto the r_in/r_out grid if they are arbitrary.
            # Simpler: If j_tor has num_r_in + num_r_out points. Assume first num_r_in are for inboard, next for outboard.
            
            if len(quantity_data_full_profile) == num_r_in + num_r_out:
                quantity_in_part = quantity_data_full_profile[:num_r_in]
                quantity_out_part = quantity_data_full_profile[num_r_in:]
            elif len(quantity_data_full_profile) == num_r_in and len(quantity_data_full_profile) == num_r_out: # Should not happen if r_in/r_out are parts
                 # This case implies j_tor might be defined on a grid that r_in and r_out fully map to individually
                 # which means j_tor should be used twice, once for r_in, once for r_out.
                quantity_in_part = quantity_data_full_profile
                quantity_out_part = quantity_data_full_profile
            else:
                # Fallback or error: we don't know how to map j_tor to r_in/r_out
                # This might happen if j_tor is on a different grid (e.g. psi_n) and r_in/r_out are arbitrary.
                print(f"Error: Cannot map quantity (len {len(quantity_data_full_profile)}) to r_in (len {num_r_in}) and r_out (len {num_r_out}). Interpolation might be needed.")
                return None, None

            if coord_path_template.endswith('r_major'):
                coordinate_data = np.concatenate([r_in[::-1], r_out])
                quantity_data = np.concatenate([quantity_in_part[::-1], quantity_out_part])
            else:  # r_minor
                r_axis = ts['global_quantities.magnetic_axis.r']
                r_minor_in = r_axis - r_in[::-1]
                r_minor_out = r_out - r_axis
                coordinate_data = np.concatenate([r_minor_in, r_minor_out])
                quantity_data = np.concatenate([quantity_in_part[::-1], quantity_out_part])
        else:
            coord_path = coord_path_template.format(time_slice_idx=str(time_slice_idx))
            quantity_path = quantity_path_template.format(time_slice_idx=str(time_slice_idx))
            
            coord_keys = coord_path.split('.')
            quantity_keys = quantity_path.split('.')
            
            coordinate_data = ods
            _quantity_data_raw_val_temp = ods # Use a temporary variable for raw data before processing
            
            for key in coord_keys:
                coordinate_data = coordinate_data[key]
            for key in quantity_keys:
                _quantity_data_raw_val_temp = _quantity_data_raw_val_temp[key]

            # Process the raw quantity data to ensure it's 1D
            quantity_data = _ensure_1d_numpy_array(_quantity_data_raw_val_temp)
            
            if quantity_data is None: # _ensure_1d_numpy_array handles logging for multi-dim or processing errors
                # Additional context for this specific failure point if needed
                print(f"Error: Failed to process quantity data from path '{quantity_path}' into a usable 1D array for plotting.")
                return None, None
            # quantity_data is now guaranteed to be 1D, scalar, or None. If None, we've already returned.
        
        if coordinate_data is not None:
            coordinate_data = np.ascontiguousarray(coordinate_data)
        # quantity_data is already a contiguous numpy array (or scalar/None)
        # from _ensure_1d_numpy_array, so no need to call np.ascontiguousarray again here.

        if coordinate_data is None or quantity_data is None: # quantity_data could be None from _ensure_1d_numpy_array
            return None, None        
        
        if not (hasattr(coordinate_data, '__len__') and hasattr(quantity_data, '__len__')):
            print(f"Data is not array-like (coord type: {type(coordinate_data)}, quant type: {type(quantity_data)}).")
            return None, None
            
        if len(coordinate_data) != len(quantity_data):
            print(f"Data length mismatch: coord={len(coordinate_data)}, quant={len(quantity_data)} (Path: {quantity_path if 'quantity_path' in locals() else 'N/A'}).")
            return None, None
            
        return coordinate_data, quantity_data
    except KeyError as e:
        print(f"Key not found during data retrieval: {str(e)} (Path: {coord_path_template if 'coord_path_template' in locals() else 'N/A'} or {quantity_path_template if 'quantity_path_template' in locals() else 'N/A'}).")
        return None, None
    except Exception as e:
        print(f"Error fetching 1D profile data: {str(e)}.")
        return None, None

# --- Generic Plotting Function ---
def check_and_update_equilibrium_data(ods, coordinate_name):
    """
    Check if required data exists and update if needed.
    
    Parameters:
    -----------
    ods : ODS
        Input OMAS data structure
    coordinate_name : str
        Name of the coordinate to check ('psi_n', 'rho_tor_norm', 'r_major', 'r_minor')
    
    Returns:
    --------
    bool
        True if data is available after check/update
    """
    from vaft.omas.update import update_equilibrium_coordinates
    
    # Check if data exists
    has_data = False
    try:
        if coordinate_name == 'psi_n':
            # Check if psi_n exists in any time slice
            for ts in ods['equilibrium.time_slice']:
                if 'psi_n' in ods['equilibrium.time_slice'][ts]['profiles_1d']:
                    has_data = True
                    break
        elif coordinate_name == 'rho_tor_norm':
            # Check if rho_tor_norm exists in any time slice
            for ts in ods['equilibrium.time_slice']:
                if 'rho_tor_norm' in ods['equilibrium.time_slice'][ts]['profiles_1d']:
                    has_data = True
                    break
        elif coordinate_name in ['r_major', 'r_minor']:
            # Check if r_inboard/r_outboard exists in any time slice
            for ts in ods['equilibrium.time_slice']:
                profiles_1d = ods['equilibrium.time_slice'][ts]['profiles_1d']
                if ('r_inboard' in profiles_1d and 'r_outboard' in profiles_1d and
                    'j_tor' in profiles_1d and 'global_quantities.magnetic_axis.r' in ods['equilibrium.time_slice'][ts]):
                    has_data = True
                    break
    except (KeyError, AttributeError):
        has_data = False
    
    # Update if data is missing
    if not has_data:
        print(f"Required data for {coordinate_name} not found. Updating equilibrium coordinates...")
        update_equilibrium_coordinates(ods)
        
        # Verify update was successful
        try:
            if coordinate_name == 'psi_n':
                for ts in ods['equilibrium.time_slice']:
                    if 'psi_n' in ods['equilibrium.time_slice'][ts]['profiles_1d']:
                        return True
            elif coordinate_name == 'rho_tor_norm':
                for ts in ods['equilibrium.time_slice']:
                    if 'rho_tor_norm' in ods['equilibrium.time_slice'][ts]['profiles_1d']:
                        return True
            elif coordinate_name in ['r_major', 'r_minor']:
                for ts in ods['equilibrium.time_slice']:
                    profiles_1d = ods['equilibrium.time_slice'][ts]['profiles_1d']
                    if ('r_inboard' in profiles_1d and 'r_outboard' in profiles_1d and
                        'j_tor' in profiles_1d and 'global_quantities.magnetic_axis.r' in ods['equilibrium.time_slice'][ts]):
                        return True
        except (KeyError, AttributeError):
            pass
        return False
    
    return True

def get_equilibrium_parameters(ods, time_slice_idx):
    """
    Get equilibrium parameters for a given time slice.
    
    Parameters:
    -----------
    ods : ODS
        Input OMAS data structure
    time_slice_idx : str
        Time slice index
    
    Returns:
    --------
    dict
        Dictionary containing time, Ip, li, and q0 values
    """
    try:
        ts = ods['equilibrium.time_slice'][time_slice_idx]
        params = {
            'time': float(ts.get('time', 0.0)),
            'ip': float(ts.get('global_quantities.ip', 0.0)),
            'li': float(ts.get('global_quantities.li', 0.0)),
            'q0': float(ts.get('global_quantities.q0', 0.0))
        }
        return params
    except (KeyError, AttributeError, ValueError, TypeError):
        return {'time': 0.0, 'ip': 0.0, 'li': 0.0, 'q0': 0.0}

def format_equilibrium_title(quantity_name, coordinate_name, params):
    """
    Format title with equilibrium parameters.
    
    Parameters:
    -----------
    quantity_name : str
        Name of the quantity being plotted
    coordinate_name : str
        Name of the coordinate being plotted
    params : dict
        Dictionary containing time, Ip, li, and q0 values
    
    Returns:
    --------
    str
        Formatted title string
    """
    title = f"{str(quantity_name).replace('_', ' ').title()} vs {str(coordinate_name).replace('_', ' ').title()}"
    
    # Add time if available and non-zero
    if params['time'] is not None and params['time'] != 0.0:
        title += f" at t = {params['time']:.4g}s"
    
    # Add other parameters if available and non-zero
    params_str = []
    if params['ip'] is not None and params['ip'] != 0.0:
        params_str.append(f"Ip = {params['ip']/1e6:.2f} MA")
    if params['li'] is not None and params['li'] != 0.0:
        params_str.append(f"li = {params['li']:.2f}")
    if params['q0'] is not None and params['q0'] != 0.0:
        params_str.append(f"q0 = {params['q0']:.2f}")
    
    if params_str:
        title += f" ({', '.join(params_str)})"
    
    return title

def format_legend_label(label, params):
    """
    Format legend label with time and Ip information.
    
    Parameters:
    -----------
    label : str or int
        Base label for the plot
    params : dict
        Dictionary containing time and Ip values
    
    Returns:
    --------
    str
        Formatted legend label
    """
    # Convert label to string if it's not already
    legend_parts = [str(label)]
    
    if params['time'] is not None:
        legend_parts.append(f"t = {params['time']:.4g}s")
    if params['ip'] is not None:
        legend_parts.append(f"Ip = {params['ip']/1e6:.2f} MA")
    
    return " | ".join(legend_parts)

def find_time_slice_for_max_ip(ods, time_slice_base_path):
    """
    Finds the time slice key corresponding to the maximum Ip in an ODS.
    Uses direct OMAS path access with ':' wildcard.

    Parameters:
    -----------
    ods : ODS
        Input OMAS data structure.
    time_slice_base_path : str
        Base path to the time slices collection (e.g., 'equilibrium.time_slice').

    Returns:
    --------
    str
        The time slice key for max Ip, or first available key if not found.
    """
    try:
        # Get available time slice keys first
        if time_slice_base_path not in ods:
            return "0"  # Default to "0" if path doesn't exist
            
        available_keys = list(ods[time_slice_base_path].keys())
        if not available_keys:
            return "0"  # Default to "0" if no time slices
            
        # Try to get IP values using wildcard path
        try:
            ip_path = f"{time_slice_base_path}.:.global_quantities.ip"
            ip_values = ods[ip_path]
            
            if ip_values:
                # Find time slice with maximum IP
                max_ip = max(ip_values)
                max_ip_key = str(list(ip_values.keys())[list(ip_values.values()).index(max_ip)])
                return max_ip_key
        except Exception:
            pass  # If IP path access fails, fall back to first available key
            
        # Fallback to first available time slice
        return str(available_keys[0])

    except Exception:
        # Ultimate fallback
        return "0"

def plot_onedim_profile_interactive(
    odc_or_ods,
    ods_group_name, 
    quantity_name,  
    coordinate_name, 
    time_slices=None, 
    labels_opt='shot',
    **plot_kwargs
):
    """
    Plots a 1D profile from ODS/ODC. 
    Interactive for single ODS. Specific comparison modes for ODC.
    Automatically updates equilibrium data if needed.
    """
    odc = odc_or_ods_check(odc_or_ods)
    labels_map = {key: label for key, label in zip(list(odc.keys()), extract_labels_from_odc(odc, opt=labels_opt))}

    if not odc:
        print("Error: ODC is empty or invalid. Cannot plot.")
        return

    if ods_group_name == 'equilibrium':
        first_ods_key_for_check = list(odc.keys())[0]
        if not check_and_update_equilibrium_data(odc[first_ods_key_for_check], coordinate_name):
            print(f"Error: Failed to update required data for {coordinate_name} on ODS '{first_ods_key_for_check}'. Plotting may fail.")

    try:
        quantity_info = TARGET_QUANTITIES[ods_group_name][quantity_name]
        coord_info = COORDINATE_DEFINITIONS[coordinate_name]
    except KeyError:
        print(f"Error: Plotting configuration not found for group '{ods_group_name}', "
              f"quantity '{quantity_name}', or coordinate '{coordinate_name}'.")
        return

    quantity_path_template = quantity_info['path_template']
    coord_path_template = coord_info['path_template']
    default_ylabel = quantity_info['ylabel']
    default_xlabel = coord_info['label']
    time_slice_base_path = '.'.join(quantity_path_template.split('.')[:2])

    is_odc_input = len(odc.keys()) > 1
    
    def _draw_plot_for_single_timeslice_key(
        current_ts_key_str, plot_title_override=None, target_ax=None, specific_ods_key_to_plot=None, legend_base_label_override=None, show_legend=True
    ):
        # Only plot a single ODS, always require specific_ods_key_to_plot
        if specific_ods_key_to_plot is None:
            print("Error: specific_ods_key_to_plot must be provided for _draw_plot_for_single_timeslice_key.")
            return False

        ods_item = odc[specific_ods_key_to_plot]
        base_label_str = labels_map.get(specific_ods_key_to_plot, str(specific_ods_key_to_plot))
        ods_params = get_equilibrium_parameters(ods_item, current_ts_key_str)
        actual_legend_base = legend_base_label_override if legend_base_label_override is not None else base_label_str
        legend_label = format_legend_label(actual_legend_base, ods_params)
        coord_data, quant_data = get_1d_profile_data(
            ods_item, current_ts_key_str,
            quantity_path_template, coord_path_template
        )
        if coord_data is not None and quant_data is not None:
            if not target_ax:
                fig, ax_to_use = plt.subplots(figsize=plot_kwargs.get('figsize', (10, 6)))
                ax_to_use.set_title(plot_title_override or format_equilibrium_title(quantity_name, coordinate_name, ods_params))
                ax_to_use.set_xlabel(default_xlabel)
                ax_to_use.set_ylabel(default_ylabel)
            else:
                ax_to_use = target_ax
            try:
                ax_to_use.plot(coord_data, quant_data, label=legend_label, **{k: v for k, v in plot_kwargs.items() if k != 'figsize'})
                ax_to_use.grid(True)
                if not target_ax and show_legend:
                    ax_to_use.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    plt.show()
                elif not target_ax:
                    plt.tight_layout()
                    plt.show()
                return True
            except Exception as e:
                print(f"Error plotting data for ODS '{specific_ods_key_to_plot}' at slice key '{current_ts_key_str}': {e}")
        else:
            print(f"No data available for ODS '{specific_ods_key_to_plot}' at slice key '{current_ts_key_str}'.")
        return False

    if is_odc_input:
        fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (10, 6)))
        current_plot_args = {k: v for k, v in plot_kwargs.items() if k != 'figsize'}
        plot_successful_for_any_ods = False

        title_text = f"{quantity_name.replace('_', ' ').title()} vs {coordinate_name.replace('_', ' ').title()} (Max Ip Slices)"
        ax.set_title(current_plot_args.pop('title', title_text))
        ax.set_xlabel(current_plot_args.pop('xlabel', default_xlabel))
        ax.set_ylabel(current_plot_args.pop('ylabel', default_ylabel))
        print('ax.lines before:', len(ax.lines)) 

        print('number of ODC items:', len(odc.items()))
        for ods_key, ods_item in odc.items():
            max_ip_ts_key = find_time_slice_for_max_ip(ods_item, time_slice_base_path)
            print(f'plotting: ods_key={ods_key}, id(ods_item)={id(ods_item)}, max_ip_ts_key={max_ip_ts_key}')
            if max_ip_ts_key is None or (time_slice_base_path in ods_item and max_ip_ts_key not in ods_item[time_slice_base_path]):
                print(f"Warning: Could not determine valid max Ip time slice for ODS '{ods_key}' (key: {max_ip_ts_key}). Skipping.")
                continue
            
            # Plot this ODS's max IP slice on the shared ax. 
            # _draw_plot_for_single_timeslice_key will use the label for the line, 
            # but won't call ax.legend() itself because target_ax is provided.
            print('ax.lines before:', len(ax.lines))
            if _draw_plot_for_single_timeslice_key(
                    max_ip_ts_key, 
                    target_ax=ax, 
                    specific_ods_key_to_plot=ods_key):  
                plot_successful_for_any_ods = True
                print('ax.lines after:', len(ax.lines))
        if plot_successful_for_any_ods:
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Single call to legend() for the ODC plot
        else:
            ax.text(0.5, 0.5, "No data available for max Ip slices in any ODS",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.show()

    else: # Single ODS input
        single_ods_item = odc[list(odc.keys())[0]]
        available_time_slice_keys_for_single_ods = []
        if time_slice_base_path in single_ods_item and single_ods_item[time_slice_base_path]:
            available_time_slice_keys_for_single_ods = sorted(list(single_ods_item[time_slice_base_path].keys()))

        if not available_time_slice_keys_for_single_ods and "{time_slice_idx}" in quantity_path_template:
            print(f"Error: No time slices found at '{time_slice_base_path}' in the ODS.")
            return

        selected_keys_for_ods_plotting = []
        is_interactive_attempt_for_single_ods = False

        if time_slices is None or str(time_slices).lower() == 'all':
            if len(available_time_slice_keys_for_single_ods) > 1 and "{time_slice_idx}" in quantity_path_template:
                is_interactive_attempt_for_single_ods = True 
            elif available_time_slice_keys_for_single_ods:
                selected_keys_for_ods_plotting = [str(available_time_slice_keys_for_single_ods[0])]
            elif "{time_slice_idx}" not in quantity_path_template:
                selected_keys_for_ods_plotting = ["0"]
            else:
                print("Error: No time slices available to plot for the ODS.")
                return
        elif isinstance(time_slices, (int, str)):
            selected_keys_for_ods_plotting = [str(time_slices)]
        elif isinstance(time_slices, list):
            selected_keys_for_ods_plotting = [str(s) for s in time_slices]
        else:
            print(f"Warning: Invalid 'time_slices' argument for ODS: {time_slices}. Plotting first available slice.")
            if available_time_slice_keys_for_single_ods:
                selected_keys_for_ods_plotting = [str(available_time_slice_keys_for_single_ods[0])]
            elif "{time_slice_idx}" not in quantity_path_template:
                selected_keys_for_ods_plotting = ["0"]
            else:
                print("Error: No time slices to plot for the ODS.")
                return

        if is_interactive_attempt_for_single_ods:
            try:
                from ipywidgets import interactive, Dropdown
                import IPython.display
                
                if not available_time_slice_keys_for_single_ods or not all(isinstance(k, (str, int)) for k in available_time_slice_keys_for_single_ods):
                    print("Error: No valid time slices for interactive ODS plot.")
                    if available_time_slice_keys_for_single_ods and isinstance(available_time_slice_keys_for_single_ods[0], (str, int)):
                        # Fallback to plotting the first slice if available_time_slice_keys_for_single_ods is not empty but failed other checks somehow
                        _draw_plot_for_single_timeslice_key(str(available_time_slice_keys_for_single_ods[0]), specific_ods_key_to_plot=list(odc.keys())[0])
                    return

                # Generate richer dropdown options: (display_label, value_key)
                dropdown_options_with_details = []
                for ts_key_str_val in available_time_slice_keys_for_single_ods:
                    params = get_equilibrium_parameters(single_ods_item, ts_key_str_val)
                    # Format Ip to two decimal places for MA, time to 3 significant figures
                    ip_ma_str = f"{params['ip']/1e6:.2f}MA" if params['ip'] is not None else "N/A"
                    time_str = f"{params['time']:.3g}s" if params['time'] is not None else "N/A"
                    label_str = f"Idx: {ts_key_str_val} | t={time_str} | Ip={ip_ma_str}"
                    dropdown_options_with_details.append((label_str, str(ts_key_str_val)))
                
                default_value_key = None
                if dropdown_options_with_details: # If there are actual slices
                    default_value_key = dropdown_options_with_details[0][1] # Use the key of the first option
                    # Add "Plot All Slices" option only if there are slices to plot
                    dropdown_options_with_details.append(("Plot All Slices", "ALL_SLICES_KEY"))
                else: # Should ideally be caught by earlier checks, but as a safeguard
                    print("Error: No time slices available to create dropdown options.")
                    return

                slice_widget = None
                try:
                    slice_widget = Dropdown(
                        options=dropdown_options_with_details,
                        value=default_value_key, # This must be one of the 'value_key' parts
                        description='Time Slice:',
                        disabled=False,
                        style={'description_width': 'initial'} # Ensure full description is visible
                    )
                except Exception as e:
                    print(f"Error creating Dropdown widget: {e}")
                    if default_value_key and default_value_key != "ALL_SLICES_KEY":
                       _draw_plot_for_single_timeslice_key(default_value_key, specific_ods_key_to_plot=list(odc.keys())[0])
                    return

                if slice_widget is None: # Should not happen if try block succeeded
                    if default_value_key and default_value_key != "ALL_SLICES_KEY":
                        _draw_plot_for_single_timeslice_key(default_value_key, specific_ods_key_to_plot=list(odc.keys())[0])
                    return
                
                # Get the single ODS key name for passing to specific_ods_key_to_plot
                single_ods_key_name = list(odc.keys())[0]

                def _interactive_plot_wrapper(selected_value_from_dropdown):
                    # Prevent empty plot on initial dummy call
                    if selected_value_from_dropdown is None or selected_value_from_dropdown == '':
                        return
                    if selected_value_from_dropdown == "ALL_SLICES_KEY":
                        # Plot all time slices on one graph
                        fig_all, ax_all = plt.subplots(figsize=plot_kwargs.get('figsize', (10, 6)))
                        
                        # Construct a title for the "All Slices" plot
                        # Use first ODS params for general title parts if needed, though time/Ip specific parts are per-slice
                        # For now, a generic title indicating all slices.
                        all_slices_title = f"{str(quantity_name).replace('_', ' ').title()} vs {str(coordinate_name).replace('_', ' ').title()} - All Time Slices"
                        
                        # Extract common plot args, removing figsize if present
                        common_plot_args_all_slices = {k: v for k, v in plot_kwargs.items() if k != 'figsize'}

                        ax_all.set_title(common_plot_args_all_slices.pop('title', all_slices_title))
                        ax_all.set_xlabel(common_plot_args_all_slices.pop('xlabel', default_xlabel))
                        ax_all.set_ylabel(common_plot_args_all_slices.pop('ylabel', default_ylabel))
                        
                        plot_successful_at_least_once = False
                        for ts_key_individual in available_time_slice_keys_for_single_ods:
                            if _draw_plot_for_single_timeslice_key(current_ts_key_str=ts_key_individual, 
                                                                 plot_title_override="", # Suppress individual titles
                                                                 target_ax=ax_all, 
                                                                 specific_ods_key_to_plot=single_ods_key_name, 
                                                                 legend_base_label_override=str(ts_key_individual),
                                                                 show_legend=True):
                                plot_successful_at_least_once = True
                        
                        if plot_successful_at_least_once:
                            ax_all.grid(True)
                            ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        else:
                            ax_all.text(0.5, 0.5, "No data available for any time slice.",
                                     horizontalalignment='center', verticalalignment='center', transform=ax_all.transAxes)
                        plt.tight_layout()
                        plt.show()
                    else:
                        # Plot a single selected time slice (no legend)
                        _draw_plot_for_single_timeslice_key(current_ts_key_str=selected_value_from_dropdown, 
                                                          plot_title_override=None, # Let it generate its default title
                                                          specific_ods_key_to_plot=single_ods_key_name,
                                                          legend_base_label_override=None,
                                                          show_legend=False) # No legend for single slice

                interactive_plot_object = interactive(
                    _interactive_plot_wrapper,
                    selected_value_from_dropdown=slice_widget
                )
                IPython.display.display(interactive_plot_object)

            except ImportError:
                print("Warning: ipywidgets not found. Plotting first available time slice for ODS. "
                      "Install ipywidgets and run in Jupyter for interactive plots.")
                if available_time_slice_keys_for_single_ods:
                    first_key_str = str(available_time_slice_keys_for_single_ods[0])
                    _draw_plot_for_single_timeslice_key(first_key_str)
                else:
                    print("Error: No time slices available for fallback non-interactive plot.")
            except Exception as e:
                print(f"An unexpected error occurred during interactive plot setup: {e}")
                if available_time_slice_keys_for_single_ods:
                    first_key_str = str(available_time_slice_keys_for_single_ods[0])
                    _draw_plot_for_single_timeslice_key(first_key_str)
                else:
                    print("Error: No time slices available for fallback after unexpected error.")
        else: 
            for ts_key in selected_keys_for_ods_plotting:
                if "{time_slice_idx}" in quantity_path_template and ts_key not in available_time_slice_keys_for_single_ods:
                    print(f"Warning: Time slice key '{ts_key}' not available in ODS. "
                          f"Available keys: {available_time_slice_keys_for_single_ods}. Skipping.")
                    continue
                _draw_plot_for_single_timeslice_key(ts_key)

# Dynamically create plotting functions for all (quantity, coordinate) pairs in equilibrium
_equilibrium_quantities = ['j_tor', 'q', 'pressure', 'pprime', 'f', 'ffprime']
_coordinates = ['psi_n', 'rho_tor_norm', 'r_major', 'r_minor']

for qty in _equilibrium_quantities:
    for coord in _coordinates:
        func_name = f"equilibrium_{coord}_{qty}"
        
        # Create docstring
        readable_qty = qty.replace('_', ' ').title()
        readable_coord = coord.replace('_', ' ').title()
        if coord == 'psi_n':
            readable_coord = "Normalized Poloidal Flux (ψ_N)"
        elif coord == 'rho_tor_norm':
            readable_coord = "Normalized Toroidal Flux (ρ_N)"
        elif coord == 'r_major':
            readable_coord = "Major Radius (R)"
        elif coord == 'r_minor':
            readable_coord = "Minor Radius (r)"

        docstring = f"""
    Plots Equilibrium {readable_qty} vs {readable_coord}.
    Interactive time slice selection if 'time_slices' is None/'all' for single ODS in Jupyter.

    Parameters:
    -----------
    odc_or_ods : ODC or ODS
        Input OMAS data structure.
    time_slices : str, int, list, or None, optional
        Specifies which time slices to plot.
        - None or 'all': Interactive selection or all slices for single ODS; max Ip for ODC.
        - int or str: A single time slice index/key.
        - list: A list of time slice indices/keys.
    labels_opt : str, optional
        Option for generating legend labels from ODC keys (e.g., 'shot', 'key', 'index').
        Default is 'shot'.
    **plot_kwargs : dict
        Additional keyword arguments passed to matplotlib.pyplot.plot().
        Example: linestyle='--', marker='o', color='red', figsize=(10,6)
        Note: 'title', 'xlabel', 'ylabel' can be overridden via plot_kwargs.
    """

        def _create_plot_function(q, c):
            # Closure to capture q and c
            def generated_plot_func(odc_or_ods, time_slices=None, labels_opt='shot', **plot_kwargs):
                plot_onedim_profile_interactive(
                    odc_or_ods,
                    ods_group_name='equilibrium',
                    quantity_name=q,
                    coordinate_name=c,
                    time_slices=time_slices,
                    labels_opt=labels_opt,
                    **plot_kwargs
                )
            generated_plot_func.__name__ = func_name
            generated_plot_func.__doc__ = docstring
            return generated_plot_func

        globals()[func_name] = _create_plot_function(qty, coord)

# --- Main for Testing ---
if __name__ == "__main__":
    # Setup a sample ODS with necessary data for testing
    ods_sample = ODS()
    n_points = 50
    psi_n_sample = np.linspace(0, 1, n_points)
    rho_tor_norm_sample = np.sqrt(psi_n_sample)
    
    # Define time slices (using string keys as ODS does)
    time_slice_keys_main = ['0', '1', '2'] # ODS dictionary keys are strings
    times_main = [0.1, 0.2, 0.3]

    for i, ts_key in enumerate(time_slice_keys_main):
        time_val = times_main[i]
        # Example j_tor profile that varies with psi_n and time slice
        j_tor_sample = (1 - psi_n_sample**2) * (1 + 0.1 * i) * 1e6 
        pressure_sample = (1 - psi_n_sample) * (1 + 0.05 * i) * 1e5
        q_sample = 1 + 2 * psi_n_sample**2 + 0.1 * i
        pprime_sample = -2 * psi_n_sample * (1 + 0.05 * i) * 1e5 # d(pressure)/d(psi_n) assuming psi_n is poloidal flux / some_const
        f_sample = (2.0 - psi_n_sample) * (1 - 0.05*i)
        ffprime_sample = -(2.0 - psi_n_sample)*(1 - 0.05*i)


        ods_sample[f'equilibrium.time_slice.{ts_key}.time'] = time_val
        ods_sample[f'equilibrium.time_slice.{ts_key}.global_quantities.ip'] = (2.0 + 0.1*i) * 1e6 # Sample Ip
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.psi_n'] = np.copy(psi_n_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.rho_tor_norm'] = np.copy(rho_tor_norm_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.j_tor'] = np.copy(j_tor_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.pressure'] = np.copy(pressure_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.q'] = np.copy(q_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.pprime'] = np.copy(pprime_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.f'] = np.copy(f_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.ffprime'] = np.copy(ffprime_sample)
        
        # For r_major, r_minor tests
        # Let's assume magnetic axis at R0=1.0m
        # r_minor = rho_tor_norm * a_minor (e.g. a_minor = 0.3)
        # r_inboard = R0 - r_minor, r_outboard = R0 + r_minor
        # But profiles_1d.r_inboard and r_outboard are usually specific grid points on LFS/HFS midplane
        # For simplicity, let's make them up based on rho_tor_norm assuming a circular flux surface representation
        # This is not how ODS typically stores r_inboard/r_outboard for general equilibria
        # but will allow testing the r_major/r_minor plotting logic.
        R0 = 1.0
        a_minor_effective = 0.3 # An effective minor radius for this example
        
        # Create r_inboard and r_outboard.
        # For a proper test, these should be distinct, e.g., points on HFS and LFS midplane.
        # We'll make r_inboard go from R0-a_minor_effective to R0 (HFS)
        # and r_outboard go from R0 to R0+a_minor_effective (LFS)
        # The number of points for r_in/r_out might be different from n_points for psi_n grid.
        # Let's use fewer points for r_in/r_out for demonstration
        n_r_points = n_points // 2 
        r_in_sample = np.linspace(R0 - a_minor_effective, R0 - 0.01, n_r_points) # up to near axis
        r_out_sample = np.linspace(R0 + 0.01, R0 + a_minor_effective, n_r_points) # from near axis

        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.r_inboard'] = np.copy(r_in_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.profiles_1d.r_outboard'] = np.copy(r_out_sample)
        ods_sample[f'equilibrium.time_slice.{ts_key}.global_quantities.magnetic_axis.r'] = R0
        
        # We need to ensure j_tor (and other quantities) are also defined in a way that can be mapped
        # to r_in/r_out by get_1d_profile_data.
        # The current get_1d_profile_data assumes that if the quantity (e.g. j_tor) has length
        # len(r_in) + len(r_out), it splits it.
        # Let's make a simplified j_tor for this specific r_in/r_out grid.
        # This is a bit artificial as j_tor is usually on psi_n or rho_tor_norm grid.
        # For testing, we'll provide a j_tor_for_rmaj_rmin that has the correct length.
        j_tor_for_rmaj_rmin_in = (1 - ((R0 - r_in_sample)/a_minor_effective)**2) * (1 + 0.1 * i) * 1e6
        j_tor_for_rmaj_rmin_out = (1 - ((r_out_sample - R0)/a_minor_effective)**2) * (1 + 0.1 * i) * 1e6
        j_tor_combined_for_r = np.concatenate([j_tor_for_rmaj_rmin_in, j_tor_for_rmaj_rmin_out])
        
        # The TARGET_QUANTITIES specifies 'equilibrium.time_slice.{time_slice_idx}.profiles_1d.j_tor'
        # The get_1d_profile_data function, when handling r_major/r_minor, re-fetches the quantity
        # using this path. So, the main j_tor should ideally be this combined one if we want r_major/r_minor
        # plots of j_tor to work with this splitting logic.
        # This means the j_tor for psi_n/rho_tor_norm plots and r_major/r_minor plots might need
        # different handling or the ODS needs to be structured very carefully.
        # For now, the test for r_major_j_tor will likely use the full j_tor (len n_points) and might
        # hit the "Cannot map quantity" error if its length doesn't match n_r_points*2, or it will
        # use the first column if j_tor were 2D.
        # Let's overwrite the j_tor with the one suitable for r_major/r_minor split if it's of the correct length
        # or just rely on the original j_tor and see how get_1d_profile_data handles it.
        # The current get_1d_profile_data's r_major/minor logic seems to expect the quantity (e.g. j_tor)
        # to be on a grid that has len(r_in) + len(r_out) points.
        # The original j_tor_sample has n_points. n_r_points*2 = n_points. So it should work.

    print(f"Sample ODS created with time slices: {ods_sample['equilibrium.time_slice'].keys()}")

    print("\n--- Testing plot_equilibrium_psi_n_j_tor (single ODS) ---")
    print("Test 1: Default time_slices (interactive in Jupyter, else first slice)")
    plot_equilibrium_psi_n_j_tor(ods_sample)
    
    print("\nTest 2: Specific time_slice_key='1'")
    plot_equilibrium_psi_n_j_tor(ods_sample, time_slices='1') # Pass key as string

    print("\nTest 3: List of time_slice_keys=['0', '2']")
    plot_equilibrium_psi_n_j_tor(ods_sample, time_slices=['0', '2'])

    print("\nTest 4: time_slices='all' (interactive in Jupyter, else first slice)")
    plot_equilibrium_psi_n_j_tor(ods_sample, time_slices='all')

    print("\n--- Testing plot_equilibrium_rho_tor_norm_j_tor (single ODS) ---")
    print("Test 5: Specific time_slice_key='0'")
    plot_equilibrium_rho_tor_norm_j_tor(ods_sample, time_slices='0')

    # Create a sample ODC for further testing
    odc_sample = ODC()
    odc_sample['shot_A'] = ods_sample # Reuse the same ODS structure

    ods_sample_b = ODS() # Create a slightly different ODS for the ODC
    for i, ts_key in enumerate(time_slice_keys_main):
        time_val = times_main[i]
        j_tor_sample_b = (1 - psi_n_sample**1.5) * (1 + 0.2 * i) * 1e6 # Different profile
        ods_sample_b[f'equilibrium.time_slice.{ts_key}.time'] = time_val
        ods_sample_b[f'equilibrium.time_slice.{ts_key}.profiles_1d.psi_n'] = np.copy(psi_n_sample)
        ods_sample_b[f'equilibrium.time_slice.{ts_key}.profiles_1d.rho_tor_norm'] = np.copy(rho_tor_norm_sample)
        ods_sample_b[f'equilibrium.time_slice.{ts_key}.profiles_1d.j_tor'] = np.copy(j_tor_sample_b)
    odc_sample['shot_B'] = ods_sample_b
    
    print("\n--- Testing with ODC ---")
    print("Test 6: ODC, default time_slices (interactive or first slice)")
    plot_equilibrium_psi_n_j_tor(odc_sample, labels_opt='shot')

    print("\nTest 7: ODC, specific time_slice_key='0', custom plot kwargs")
    plot_equilibrium_psi_n_j_tor(odc_sample, time_slices='0', labels_opt='shot', 
                                 linestyle='--', marker='x', figsize=(8,5))

    print("\n--- Testing Edge Cases ---")
    ods_missing_data = ODS()
    ods_missing_data['equilibrium.time_slice.0.time'] = 0.1
    ods_missing_data['equilibrium.time_slice.0.profiles_1d.psi_n'] = psi_n_sample
    # j_tor is intentionally missing for this slice
    print("Test 8: ODS with missing j_tor for a slice (should show 'No data' or error message)")
    plot_equilibrium_psi_n_j_tor(ods_missing_data, time_slices='0')
    
    ods_no_slices_at_all = ODS()
    # This ODS has no 'equilibrium.time_slice' structure
    print("\nTest 9: ODS with no 'equilibrium.time_slice' structure at all")
    plot_equilibrium_psi_n_j_tor(ods_no_slices_at_all)

    print("\nTest 10: Invalid quantity name")
    plot_onedim_profile_interactive(ods_sample, 'equilibrium', 'non_existent_quantity', 'psi_n')

    print("\nAll planned tests executed. If in Jupyter, please check interactive plot behavior.")
    print("Note: For non-Jupyter environments, 'interactive' mode falls back to plotting the first slice.")

