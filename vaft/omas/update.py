"""
Update derived quantities for the ods data structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from vaft.formula import normalize_psi
from vaft.process.equilibrium import psi_to_RZ, volume_average
from omas import *
from omfit_classes.omfit_eqdsk import OMFITeqdsk
from omfit_classes.fluxSurface import fluxSurfaces
# update_diagnostics_file(ods, filename)

# print_available_ids(ods)

def update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=None):
    """
    Update normalized poloidal flux (psi_norm) for all time slices.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    # Process all time slices if not specified
    time_slices = range(len(ods['equilibrium.time_slice'])) if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    
    for idx in time_slices:
        ts = ods['equilibrium.time_slice'][idx]
        # Get psi values
        psi = ts['profiles_1d.psi']
        psi_axis = ts['global_quantities.psi_axis']
        psi_bdry = ts['global_quantities.psi_boundary']
        
        # Calculate normalized psi
        psi_n = (psi - psi_axis) / (psi_bdry - psi_axis)
        
        # Store in ODS
        ts['profiles_1d.psi_norm'] = psi_n

def update_equilibrium_profiles_1d_radial_coordinates(ods, time_slice=None, plot_opt=0):
    """
    Update radial coordinates for all time slices using 2D psi mapping.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
        plot_opt (int): 0=no plot, 1=plot validation, 2=plot derivatives
    """
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    
    # Process all time slices if not specified
    time_slices = ods['equilibrium.time_slice'] if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    
    for idx in time_slices:
        ts = ods['equilibrium.time_slice'][idx]
        # Extract 2D grid data at magnetic axis
        grid_r = ts['profiles_2d.0.grid.dim1']
        grid_z = ts['profiles_2d.0.grid.dim2']
        psi_2d = ts['profiles_2d.0.psi']
        z_axis = ts['global_quantities.magnetic_axis.z']
        z_idx = np.argmin(np.abs(grid_z - z_axis))
        psi_2d_slice = psi_2d[:, z_idx]

        # Boundary processing
        boundary_r = ts['boundary.outline.r']
        r_min, r_max = np.min(boundary_r), np.max(boundary_r)
        r_axis = ts['global_quantities.magnetic_axis.r']
        
        # Split into inboard/outboard regions
        mask_in = (grid_r >= r_min) & (grid_r <= r_axis)
        mask_out = (grid_r >= r_axis) & (grid_r <= r_max)
        psi_in, r_in = psi_2d_slice[mask_in], grid_r[mask_in]
        psi_out, r_out = psi_2d_slice[mask_out], grid_r[mask_out]

        # Create interpolation functions
        f_in = interp1d(psi_in[::-1], r_in[::-1], 
                       kind='cubic', fill_value='extrapolate')
        f_out = interp1d(psi_out, r_out, 
                        kind='cubic', fill_value='extrapolate')

        # Map 1D profiles
        psi_1d = ts['profiles_1d.psi']
        ts['profiles_1d.r_inboard'] = f_in(psi_1d)
        ts['profiles_1d.r_outboard'] = f_out(psi_1d)

        # Generate validation plots
        if plot_opt >= 1:
            plt.figure(figsize=[15,10])
            plt.suptitle(f'Time Slice {idx} Validation')
            
            # Primary mapping validation
            plt.subplot(221)
            plt.plot(r_in, psi_in, 'r-', label='2D Inboard')
            plt.plot(ts['profiles_1d.r_inboard'], psi_1d, 'b--', label='Inboard')
            plt.plot(ts['global_quantities.magnetic_axis.r'], ts['global_quantities.psi_axis'], 'k.', label='Magnetic Axis')
            plt.plot(r_out, psi_out, 'g-', label='2D Outboard')
            plt.plot(ts['profiles_1d.r_outboard'], psi_1d, 'm--', label='Outboard')
            plt.xlabel('R [m]'), plt.ylabel('Psi'), plt.legend(loc='upper right')
            
            plt.subplot(222)
            plt.plot(ts['profiles_1d.r_inboard'], ts['profiles_1d.j_tor'], 'r-', label='Inboard')
            plt.plot(ts['profiles_1d.r_outboard'], ts['profiles_1d.j_tor'], 'g-', label='Outboard')
            plt.xlabel('R [m]'), plt.ylabel('J_tor [A/m2]')

            plt.subplot(223)
            plt.plot(ts['profiles_1d.r_inboard'], ts['profiles_1d.pressure'], 'r-', label='Inboard')
            plt.plot(ts['profiles_1d.r_outboard'], ts['profiles_1d.pressure'], 'g-', label='Outboard')
            plt.xlabel('R [m]'), plt.ylabel('Pressure [Pa]')

            plt.subplot(224)
            plt.plot(ts['profiles_1d.r_inboard'], ts['profiles_1d.q'], 'r-', label='Inboard')
            plt.plot(ts['global_quantities.magnetic_axis.r'], ts['global_quantities.q_axis'], 'k.', label='Magnetic Axis')
            plt.plot(ts['profiles_1d.r_outboard'], ts['profiles_1d.q'], 'g-', label='Outboard')
            plt.xlabel('R [m]'), plt.ylabel('safety factor')

            plt.legend()
            plt.tight_layout()
            plt.show()

def update_equilibrium_boundary(ods, time_slice=None):
    """
    Update geometric axis for all time slices.
    """
    time_slices = ods['equilibrium.time_slice'] if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    
    for idx in time_slices:
        ts = ods['equilibrium']['time_slice'][idx]
        # check if boundary.outline exists
        if 'boundary.outline' not in ts:
            print(f"Warning: boundary.outline not found for time slice {idx}")
            continue
        r_min = ts['boundary.outline']['r'].min()
        r_max = ts['boundary.outline']['r'].max()
        z_min = ts['boundary.outline']['z'].min()
        z_max = ts['boundary.outline']['z'].max()
        ts['boundary.geometric_axis.r'] = (r_max + r_min) / 2
        ts['boundary.geometric_axis.z'] = (z_max + z_min) / 2
        ts['boundary.minor_radius'] = (r_max - r_min) / 2
        ts['boundary.triangularity_lower'] = ts['profiles_1d.triangularity_lower'][-1]
        ts['boundary.triangularity_upper'] = ts['profiles_1d.triangularity_upper'][-1]
        ts['boundary.triangularity'] = (ts['boundary.triangularity_lower'] + ts['boundary.triangularity_upper']) / 2
        # elongation
        ts['boundary.elongation'] = ts['profiles_1d.elongation'][-1]

def update_equilibrium_coordinates(ods, time_slice=None, plot_opt=0):
    """
    Main entry point for updating all equilibrium coordinates.
    Updates normalized psi and radial coordinates for all time slices.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
        plot_opt (int): 0=no plot, 1=plot validation
    """
    # Update normalized psi
    update_equilibrium_profiles_1d_normalized_psi(ods, time_slice)
    
    # Update radial coordinates
    update_equilibrium_profiles_1d_radial_coordinates(ods, time_slice, plot_opt)

def update_equilibrium_global_quantities_q_min(ods, time_slice=None):
    """
    Update q_min for all time slices using min() of profiles_1d.q
    """
    # Process all time slices if not specified
    time_slices = range(len(ods['equilibrium.time_slice'])) if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    for idx in time_slices:
        ts = ods['equilibrium.time_slice'][idx]
        ts['global_quantities.q_min'] = ts['profiles_1d.q'].min()

def update_equilibrium_global_quantities_volume(ods, time_slice=None):
    """
    Update volume for all time slices using profiles_1d.volume
    """
    # check if profiles_1d.volume exists for each time slice
    if time_slice is None:
        if 'equilibrium.time_slice' in ods and len(ods['equilibrium.time_slice']):
            time_slice = range(len(ods['equilibrium.time_slice']))
        else:
            print("Warning: No time slices found in ODS. Cannot update stored energy.")
            return
    # Convert single integer to list for iteration
    if isinstance(time_slice, (int, np.integer)):
        time_slice = [time_slice]
    for idx in time_slice:
        ts = ods['equilibrium.time_slice'][idx]
        if 'profiles_1d.volume' not in ts:
            print(f"Warning: profiles_1d.volume not found for time slice {idx}")
            continue
        ts['global_quantities.volume'] = ts['profiles_1d.volume'][-1]

def update_equilibrium_profiles_2d_j_tor(ods, time_slice=None):
    """
    Update 2D toroidal current density (j_tor) by mapping 1D j_tor profile to 2D (R,Z) grid.
    
    This function maps profiles_1d.j_tor onto the 2D equilibrium grid using psi_norm
    coordinate mapping, similar to how pressure is mapped.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    from vaft.process.equilibrium import psi_to_RZ
    
    # Process all time slices if not specified
    time_slices = range(len(ods['equilibrium.time_slice'])) if time_slice is None else (
        [time_slice] if isinstance(time_slice, (int, np.integer)) else time_slice)
    
    for idx in time_slices:
        try:
            eq_ts = ods['equilibrium.time_slice'][idx]
        except (IndexError, KeyError):
            print(f"Warning: Time slice index {idx} is out of bounds. Skipping.")
            continue
        
        # Check if 1D j_tor exists
        if 'profiles_1d.j_tor' not in eq_ts:
            print(f"Warning: profiles_1d.j_tor not found for time slice {idx}. Skipping.")
            continue
        
        # Check if 2D grid and psi exist
        if 'profiles_2d.0.grid.dim1' not in eq_ts or 'profiles_2d.0.grid.dim2' not in eq_ts:
            print(f"Warning: profiles_2d.0.grid not found for time slice {idx}. Skipping.")
            continue
        
        if 'profiles_2d.0.psi' not in eq_ts:
            print(f"Warning: profiles_2d.0.psi not found for time slice {idx}. Skipping.")
            continue
        
        # Get 1D j_tor profile
        j_tor_1d = np.asarray(eq_ts['profiles_1d.j_tor'], float)
        
        # Ensure psi_norm exists
        eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
        if 'psi_norm' not in eq_profiles_1d:
            update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=idx)
            eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
            if 'psi_norm' not in eq_profiles_1d:
                print(f"Warning: Failed to create psi_norm for time slice {idx}. Skipping.")
                continue
        
        # Get psi_norm grid (typically uniform for 1D profiles)
        psi_norm_1d = np.asarray(eq_profiles_1d['psi_norm'], float)
        
        # Ensure psi_norm_1d and j_tor_1d have the same length
        if len(psi_norm_1d) != len(j_tor_1d):
            # If lengths don't match, create uniform psi_norm grid
            psi_norm_1d = np.linspace(0.0, 1.0, len(j_tor_1d))
        
        # Get 2D grid and psi
        R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
        Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
        psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
        
        # Get psi normalization constants
        psi_axis = float(eq_ts.get('global_quantities.psi_axis', np.nan))
        psi_lcfs = float(eq_ts.get('global_quantities.psi_boundary', np.nan))
        
        if not np.isfinite(psi_axis) or not np.isfinite(psi_lcfs) or psi_lcfs == psi_axis:
            # Fallback: normalize by min/max of psi_RZ
            psi_axis = float(np.nanmin(psi_RZ))
            psi_lcfs = float(np.nanmax(psi_RZ))
        
        # Map 1D j_tor to 2D (R,Z)
        try:
            j_tor_RZ, _psiN_RZ = psi_to_RZ(psi_norm_1d, j_tor_1d, psi_RZ, psi_axis, psi_lcfs)
            
            # Store in profiles_2d.0.j_tor
            eq_ts['profiles_2d.0.j_tor'] = j_tor_RZ
        except Exception as e:
            print(f"Warning: Could not map j_tor to 2D for time slice {idx}: {e}")
            continue

def update_equilibrium_profiles_2d_sfl_coordinates(ods, time_slice=None, profiles_2d_idx=1, convention='sfl', n_theta=129, plot_opt=0):
    """
    Update Straight Field Line (SFL) coordinates for `profiles_2d` entries.

    This function computes SFL coordinates (like PEST, equal arc-length, etc.)
    for specified time slices in an OMAS data structure. It populates a
    `profiles_2d` entry with the SFL grid (R, Z, ψ, θ_SFL).

    The `profiles_2d.grid.dim1` will store normalized poloidal flux (ψ_norm).
    The `profiles_2d.grid.dim2` will store the SFL poloidal angle (θ_SFL).
    The `profiles_2d.psi` will store the non-normalized poloidal flux (ψ).
    The `profiles_2d.r`, `profiles_2d.z`, `profiles_2d.theta` will store the
    corresponding R, Z, and SFL poloidal angle values on this grid.

    Parameters:
        ods (OMAS ODS): Input OMAS data structure.
        time_slice (int, list of int, None): Specifies which time slice(s) to process.
            If None, all time slices are processed.
            If int or list of int, these are treated as direct indices into
            `ods['equilibrium.time_slice']`.
        profiles_2d_idx (int): Index of the `profiles_2d` entry to update
            (e.g., `ods['equilibrium.time_slice'][t_idx]['profiles_2d'][profiles_2d_idx]`).
            Default is 1.
        convention (str): Poloidal angle convention for SFL coordinates.
            Supported values: 'sfl' (maps to 'straight_line'), 'straight_line' (PEST-like),
            'equal_arc', 'hamada', 'boozer'. Default is 'sfl'.
        n_theta (int): Number of poloidal points for the SFL grid. Default is 129.
        plot_opt (int): Plotting option:
            0: No plots.
            1: Show interactive plots of the SFL grid (ψ-θ_SFL) and the R-Z mesh.
            2: Same as 1, plus save plots to PNG files.
    """
    time_idx_list = []
    if time_slice is None:
        if 'equilibrium.time_slice' in ods and len(ods['equilibrium.time_slice']):
            time_idx_list = range(len(ods['equilibrium.time_slice']))
        else:
            print("Warning: No time slices found in ODS. Cannot update SFL coordinates.")
            return
    elif isinstance(time_slice, (int, np.integer)):
        time_idx_list = [time_slice]
    elif isinstance(time_slice, (list, np.ndarray)):
        time_idx_list = time_slice
    else:
        raise ValueError(f"time_slice must be an int, list of ints, or None. Got {type(time_slice)}")

    method_map = {
        'sfl': 'straight_line',
        'straight_line': 'straight_line',
        'equal_arc': 'equal_arc',
        'hamada': 'hamada',
        'boozer': 'boozer'
    }
    actual_method = method_map.get(convention.lower(), 'straight_line')

    for idx in time_idx_list:
        try:
            ts = ods['equilibrium.time_slice'][idx]
            time_val = ts.get('time', float(idx)) # Use actual time if available, else index
        except IndexError:
            print(f"Warning: Time slice index {idx} is out of bounds. Skipping.")
            continue

        eq_obj = OMFITeqdsk()
        try:
            # OMFITeqdsk.from_omas needs the parent ODS and the time_idx within that ODS
            eq_obj.from_omas(ods, time_idx=idx)
        except Exception as e:
            print(f"Error creating OMFITeqdsk from ODS for time slice index {idx} (time {time_val}): {e}")
            print("Please ensure the ODS time slice contains necessary equilibrium data (psi_axis, psi_boundary, R/Z grid, 2D psi, etc.)")
            continue

        fs = fluxSurfaces(gEQDSK=eq_obj, quiet=True)
        try:
            fs.findSurfaces()
            fs.calc_poloidal_angle(method=actual_method, npts=n_theta)
        except Exception as e:
            print(f"Error in fluxSurfaces processing for time slice index {idx} (time {time_val}): {e}")
            continue

        prof2d = ts['profiles_2d'][profiles_2d_idx]

        dim1_vals = fs['levels']  # Normalized psi
        if fs['flux'] and len(fs['flux']) > 0 and 'theta' in fs['flux'][0]:
             dim2_vals = fs['flux'][0]['theta']  # SFL theta coordinates
        else:
            print(f"Warning: SFL theta coordinates not found in fluxSurfaces output for time slice {idx}. Skipping SFL update for this slice.")
            continue


        prof2d['grid.dim1'] = dim1_vals
        prof2d['grid.dim2'] = dim2_vals
        prof2d['grid_type.index'] = 11
        # Name from IMAS data dictionary for grid_type 11: 'psi_norm_straight_field_line_theta'
        # The test script used 'inverse_psi_straight_field_line'.
        # Let's use the IMAS standard name if possible, but ensure consistency with project.
        # For now, sticking to what test script provides as it might be specific to vaft's OMAS usage.
        prof2d['grid_type.name'] = 'inverse_psi_straight_field_line'


        nr = len(dim1_vals)
        nt = len(dim2_vals)

        R_2d = np.zeros((nr, nt))
        Z_2d = np.zeros((nr, nt))
        Psi_2d_values = np.zeros((nr, nt)) # Non-normalized psi
        Theta_2d_values = np.zeros((nr, nt)) # SFL theta

        for k_idx in range(nr):
            if k_idx < len(fs['flux']):
                flux_surface_data = fs['flux'][k_idx]
                R_2d[k_idx, :] = flux_surface_data.get('R', np.zeros(nt))
                Z_2d[k_idx, :] = flux_surface_data.get('Z', np.zeros(nt))
                Psi_2d_values[k_idx, :] = flux_surface_data.get('psi', np.zeros(nt)) # Should be a scalar, broadcast if so.
                # Ensure psi is broadcast if it's scalar from fs['flux'][k]['psi']
                if np.isscalar(flux_surface_data.get('psi')):
                     Psi_2d_values[k_idx, :] = flux_surface_data.get('psi') # Broadcasts
                else:
                     Psi_2d_values[k_idx, :] = flux_surface_data.get('psi', np.zeros(nt))

                Theta_2d_values[k_idx, :] = flux_surface_data.get('theta', np.zeros(nt))
            else:
                print(f"Warning: Not enough flux surface data in fs['flux'] for surface {k_idx}. Expected {nr} surfaces.")

        prof2d['r'] = R_2d
        prof2d['z'] = Z_2d
        prof2d['psi'] = Psi_2d_values
        prof2d['theta'] = Theta_2d_values

        if plot_opt >= 1:
            plt.figure(figsize=(10, 8))
            # Using prof2d.psi which stores non-normalized psi, as per test script's data population
            for i_surf in range(nr):
                # prof2d.psi should be (nr, nt), so prof2d.psi[i_surf,0] is one value for the surface
                plt.plot(prof2d['grid.dim2'], np.full_like(prof2d['grid.dim2'], prof2d['psi'][i_surf, 0]), 'k-', lw=0.5)

            plt.xlabel(r'$\theta_{\rm SFL}$ [rad]')
            plt.ylabel(r'$\psi$ [Wb]') # Label reflects non-normalized psi
            plt.title(f'Time: {time_val:.4f}s - ψ–θ SFL Grid (prof2d[{profiles_2d_idx}], {convention})')
            plt.ylim(min(prof2d['psi'][:,0]), max(prof2d['psi'][:,0])) # Adjust y-limits
            if plot_opt >= 2:
                plt.savefig(f'sfl_psi_theta_t{time_val:.3f}_idx{profiles_2d_idx}_{convention}.png')
            plt.show()

            fig_rz, ax_rz = plt.subplots(figsize=(10, 10))
            # Plot flux surfaces (constant psi_norm)
            for i_surf in range(nr):
                ax_rz.plot(prof2d['r'][i_surf, :], prof2d['z'][i_surf, :], 'b-', lw=0.7, label='Flux Surface' if i_surf == 0 else None)
            # Plot field lines (constant SFL theta)
            for j_theta in range(0, nt, max(1, nt // 16)): # Plot a subset of theta lines for clarity
                ax_rz.plot(prof2d['r'][:, j_theta], prof2d['z'][:, j_theta], 'r--', lw=0.5, label='SFL $\\theta$ line' if j_theta == 0 else None)
            
            ax_rz.set_aspect('equal')
            ax_rz.set_xlabel('R [m]')
            ax_rz.set_ylabel('Z [m]')
            ax_rz.set_title(f'Time: {time_val:.4f}s - SFL R-Z Mesh (prof2d[{profiles_2d_idx}], {convention})')
            
            # Add magnetic axis from global_quantities if available
            gq = ts.get('global_quantities', {})
            if 'magnetic_axis.r' in gq and 'magnetic_axis.z' in gq:
                ax_rz.plot(gq['magnetic_axis.r'], gq['magnetic_axis.z'], 'kx', markersize=10, mew=2, label='Mag. Axis')
            
            if nr > 0 or nt > 0 : # only add legend if something was plotted.
                handles, labels = ax_rz.get_legend_handles_labels()
                by_label = dict(zip(labels, handles)) # remove duplicate labels
                ax_rz.legend(by_label.values(), by_label.keys())

            if plot_opt >= 2:
                plt.savefig(f'sfl_rz_mesh_t{time_val:.3f}_idx{profiles_2d_idx}_{convention}.png')
            plt.show()

def update_equilibrium_stored_energy(ods, time_slice=None):
    """
    Update stored energy for all time slices. [ref. omas.physics_equilibrium_stored_energy]
    """
    if time_slice is None:
        if 'equilibrium.time_slice' in ods and len(ods['equilibrium.time_slice']):
            time_slice = range(len(ods['equilibrium.time_slice']))
        else:
            print("Warning: No time slices found in ODS. Cannot update stored energy.")
            return
    for idx in time_slice:
        ts = ods['equilibrium.time_slice'][idx]
        # check if profiles_1d.pressure and profiles_1d.volume exist
        if 'profiles_1d.pressure' not in ts or 'profiles_1d.volume' not in ts:
            print(f"Warning: profiles_1d.pressure or profiles_1d.volume not found for time slice {idx}")
            continue
        pressure_equil = ts['profiles_1d.pressure']
        volume_equil = ts['profiles_1d.volume']
        ts['global_quantities.energy_mhd'] = 3.0 / 2.0 * np.trapz(pressure_equil, x=volume_equil)
        
def update_core_profiles_global_quantities_volume_average(ods, time_slice=None):
    """
    Update volume-averaged core profile quantities using equilibrium geometry.

    This function maps 1D core profiles (in flux space) onto the 2D (R,Z)
    equilibrium grid via ψ_N and computes volume-averaged quantities:

    - core_profiles.global_quantities.n_e_volume_average
    - core_profiles.global_quantities.t_e_volume_average
    - core_profiles.global_quantities.n_i_volume_average
    - core_profiles.global_quantities.t_i_volume_average

    The function matches core_profiles time indices to equilibrium time slices
    by finding the closest matching time values.
    """
    from vaft.process.equilibrium import psi_to_RZ, volume_average
    
    # Basic availability checks
    if 'core_profiles.profiles_1d' not in ods:
        print("Warning: core_profiles.profiles_1d not found in ODS.")
        return
    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        print("Warning: equilibrium.time_slice not found in ODS.")
        return

    n_core_slices = len(ods['core_profiles.profiles_1d'])
    n_equil_slices = len(ods['equilibrium.time_slice'])

    # Extract time arrays for matching
    core_times = []
    for idx in range(n_core_slices):
        cp_ts = ods['core_profiles.profiles_1d'][idx]
        if 'time' in cp_ts:
            core_times.append(float(cp_ts['time']))
        elif 'core_profiles.time' in ods and idx < len(ods['core_profiles.time']):
            core_times.append(float(ods['core_profiles.time'][idx]))
        else:
            print(f"Warning: time not found for core_profiles.profiles_1d[{idx}], using index as time")
            core_times.append(float(idx))
    
    equil_times = []
    for idx in range(n_equil_slices):
        eq_ts = ods['equilibrium.time_slice'][idx]
        if 'time' in eq_ts:
            equil_times.append(float(eq_ts['time']))
        elif 'equilibrium.time' in ods and idx < len(ods['equilibrium.time']):
            equil_times.append(float(ods['equilibrium.time'][idx]))
        else:
            print(f"Warning: time not found for equilibrium.time_slice[{idx}], using index as time")
            equil_times.append(float(idx))
    
    core_times = np.asarray(core_times)
    equil_times = np.asarray(equil_times)

    # Build list of core profile indices to process
    if time_slice is None:
        core_indices = range(n_core_slices)
    elif isinstance(time_slice, (int, np.integer)):
        core_indices = [time_slice] if time_slice < n_core_slices else []
    else:
        core_indices = [idx for idx in time_slice if idx < n_core_slices]

    # Initialize result lists for all time slices
    n_e_vol_list = []
    T_e_vol_list = []
    ion_vol_dict = {}  # {ion_idx: {'n_i': [], 'T_i': []}}

    # Step 1 & 2: Process each core profile time slice
    for cp_idx in core_indices:
        cp_time = core_times[cp_idx]
        
        # Find closest equilibrium time
        equil_idx = np.argmin(np.abs(equil_times - cp_time))
        time_diff = abs(equil_times[equil_idx] - cp_time)
        
        if time_diff > 0.1:  # Warn if time difference is large (> 100ms)
            print(f"Warning: Large time difference ({time_diff:.3f}s) between core_profiles[{cp_idx}] (t={cp_time:.3f}s) and equilibrium[{equil_idx}] (t={equil_times[equil_idx]:.3f}s)")
        
        cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
        eq_ts = ods['equilibrium.time_slice'][equil_idx]

        # Get 1D flux coordinate for core profiles (always rho_tor_norm)
        grid = cp_ts.get('grid', ods['core_profiles'].get('grid', ODS()))
        if 'rho_tor_norm' not in grid:
            print(f"Warning: rho_tor_norm grid missing for core_profiles.profiles_1d[{cp_idx}], skipping")
            n_e_vol_list.append(np.nan)
            T_e_vol_list.append(np.nan)
            # Append NaN for all existing ions
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)
            continue
        
        rho_tor_norm_cp = np.asarray(grid['rho_tor_norm'], float)

        # Get equilibrium profiles_1d for coordinate conversion
        eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
        
        # Ensure equilibrium has psi_norm (create if missing)
        if 'psi_norm' not in eq_profiles_1d:
            update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=equil_idx)
            eq_profiles_1d = eq_ts.get('profiles_1d', ODS())
            if 'psi_norm' not in eq_profiles_1d:
                print(f"Warning: failed to create psi_norm for equilibrium.time_slice[{equil_idx}], skipping")
                n_e_vol_list.append(np.nan)
                T_e_vol_list.append(np.nan)
                for ion_idx in ion_vol_dict.keys():
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                continue
        
        # Get equilibrium rho_tor_norm and psi_norm for coordinate mapping
        if 'rho_tor_norm' not in eq_profiles_1d:
            print(f"Warning: rho_tor_norm missing in equilibrium.profiles_1d for time_slice[{equil_idx}], skipping")
            n_e_vol_list.append(np.nan)
            T_e_vol_list.append(np.nan)
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)
            continue
        
        rho_tor_norm_eq = np.asarray(eq_profiles_1d['rho_tor_norm'], float)
        psi_norm_eq = np.asarray(eq_profiles_1d['psi_norm'], float)
        
        # Ensure monotonicity for equilibrium rho_tor_norm and psi_norm mapping
        if not np.all(np.diff(rho_tor_norm_eq) > 0):
            sort_idx = np.argsort(rho_tor_norm_eq)
            rho_tor_norm_eq_sorted = rho_tor_norm_eq[sort_idx]
            psi_norm_eq_sorted = psi_norm_eq[sort_idx]
            unique_mask = np.concatenate(([True], np.diff(rho_tor_norm_eq_sorted) > 1e-10))
            rho_tor_norm_eq_sorted = rho_tor_norm_eq_sorted[unique_mask]
            psi_norm_eq_sorted = psi_norm_eq_sorted[unique_mask]
        else:
            rho_tor_norm_eq_sorted = rho_tor_norm_eq
            psi_norm_eq_sorted = psi_norm_eq
        
        # Use equilibrium psi_norm grid as target coordinate system
        psiN_1d = psi_norm_eq_sorted
        
        # Create inverse mapping: psi_norm -> rho_tor_norm
        interp_psi_to_rho = interp1d(psi_norm_eq_sorted, rho_tor_norm_eq_sorted,
                                     kind='linear',
                                     bounds_error=False,
                                     fill_value=(rho_tor_norm_eq_sorted[0], rho_tor_norm_eq_sorted[-1]))
        rho_tor_norm_at_psiN = interp_psi_to_rho(psiN_1d)

        # Equilibrium 2D grid and ψ(R,Z)
        try:
            R_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim1'], float)
            Z_grid = np.asarray(eq_ts['profiles_2d.0.grid.dim2'], float)
            psi_RZ = np.asarray(eq_ts['profiles_2d.0.psi'], float)
            psi_axis = float(eq_ts['global_quantities.psi_axis'])
            psi_lcfs = float(eq_ts['global_quantities.psi_boundary'])
        except KeyError:
            print(f"Warning: missing profiles_2d.0 or global_quantities.psi_* for equilibrium.time_slice[{equil_idx}], skipping")
            n_e_vol_list.append(np.nan)
            T_e_vol_list.append(np.nan)
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)
            continue

        # Helper function: rho_tor_norm -> psi_norm -> 2D RZ
        def convert_to_2d(profile_1d_rho):
            # Interpolate profile from rho_tor_norm_cp to rho_tor_norm_at_psiN
            interp_func = interp1d(rho_tor_norm_cp, profile_1d_rho,
                                  kind='linear',
                                  bounds_error=False,
                                  fill_value=(profile_1d_rho[0], profile_1d_rho[-1]))
            profile_1d = interp_func(rho_tor_norm_at_psiN)
            profile_RZ, psiN_RZ = psi_to_RZ(psiN_1d, profile_1d, psi_RZ, psi_axis, psi_lcfs)
            return profile_RZ, psiN_RZ

        # Step 2: Process electron profiles
        psiN_RZ = None
        n_e_vol = np.nan
        T_e_vol = np.nan
        
        if 'electrons.density' in cp_ts and 'electrons.temperature' in cp_ts:
            try:
                # Process n_e
                n_e_1d_rho = np.asarray(cp_ts['electrons.density'], float)
                n_e_RZ, psiN_RZ = convert_to_2d(n_e_1d_rho)
                n_e_vol, _ = volume_average(n_e_RZ, psiN_RZ, R_grid, Z_grid)
                
                # Process T_e
                T_e_1d_rho = np.asarray(cp_ts['electrons.temperature'], float)
                T_e_RZ, _ = convert_to_2d(T_e_1d_rho)
                T_e_vol, _ = volume_average(T_e_RZ, psiN_RZ, R_grid, Z_grid)
            except Exception as e:
                print(f"Warning: Error processing electron profiles for core_profiles[{cp_idx}]: {e}")
        else:
            print(f"Warning: electrons density/temperature missing in core_profiles.profiles_1d[{cp_idx}]")
        
        n_e_vol_list.append(n_e_vol)
        T_e_vol_list.append(T_e_vol)

        # Step 2: Process ion profiles (each ion individually)
        if 'ion' in cp_ts and cp_ts['ion']:
            # Get list of ion indices
            ion_indices = []
            if isinstance(cp_ts['ion'], dict):
                ion_indices = list(cp_ts['ion'].keys())
            elif isinstance(cp_ts['ion'], (list, tuple)):
                ion_indices = list(range(len(cp_ts['ion'])))
            
            for ion_idx in ion_indices:
                # Initialize ion result lists if not exists
                if ion_idx not in ion_vol_dict:
                    ion_vol_dict[ion_idx] = {'n_i': [], 'T_i': []}
                    # Fill with NaN for previous time slices
                    for _ in range(len(n_e_vol_list) - 1):
                        ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                        ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                
                # Get ion data
                if isinstance(cp_ts['ion'], dict):
                    ion_ts = cp_ts['ion'][ion_idx]
                else:
                    ion_ts = cp_ts['ion'][ion_idx]
                
                # Check if ion_ts is valid and has required keys
                if not isinstance(ion_ts, dict) or ion_ts is None:
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                    continue
                
                if 'density' not in ion_ts or 'temperature' not in ion_ts:
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                    continue
                
                # Process ion profiles
                try:
                    n_i_1d_rho = np.asarray(ion_ts['density'], float)
                    T_i_1d_rho = np.asarray(ion_ts['temperature'], float)
                    
                    # Check if arrays are valid
                    if n_i_1d_rho.size == 0 or T_i_1d_rho.size == 0:
                        ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                        ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                        continue
                    if np.all(np.isnan(n_i_1d_rho)) or np.all(np.isnan(T_i_1d_rho)):
                        ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                        ion_vol_dict[ion_idx]['T_i'].append(np.nan)
                        continue
                    
                    # Convert to 2D RZ and compute volume average
                    n_i_RZ, _ = convert_to_2d(n_i_1d_rho)
                    T_i_RZ, _ = convert_to_2d(T_i_1d_rho)
                    
                    n_i_vol, _ = volume_average(n_i_RZ, psiN_RZ, R_grid, Z_grid)
                    T_i_vol, _ = volume_average(T_i_RZ, psiN_RZ, R_grid, Z_grid)
                    
                    ion_vol_dict[ion_idx]['n_i'].append(n_i_vol)
                    ion_vol_dict[ion_idx]['T_i'].append(T_i_vol)
                    
                except Exception as e:
                    print(f"Warning: Error processing ion[{ion_idx}] for core_profiles[{cp_idx}]: {e}")
                    ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                    ion_vol_dict[ion_idx]['T_i'].append(np.nan)
        else:
            # No ions for this time slice, append NaN for all existing ions
            for ion_idx in ion_vol_dict.keys():
                ion_vol_dict[ion_idx]['n_i'].append(np.nan)
                ion_vol_dict[ion_idx]['T_i'].append(np.nan)

    # Step 3: Store results in core_profiles.global_quantities
    if 'core_profiles.global_quantities' not in ods:
        ods['core_profiles.global_quantities'] = ODS()
    
    gq = ods['core_profiles.global_quantities']
    gq['n_e_volume_average'] = n_e_vol_list
    gq['t_e_volume_average'] = T_e_vol_list
    
    # Store ion results only if ion data exists
    if ion_vol_dict:
        # Store ion results
        if 'ion' not in gq:
            gq['ion'] = []
        
        # Ensure ion array has enough elements
        max_ion_idx = max(ion_vol_dict.keys())
        while len(gq['ion']) <= max_ion_idx:
            gq['ion'].append(ODS())
        
        for ion_idx, results in ion_vol_dict.items():
            gq['ion'][ion_idx]['n_i_volume_average'] = results['n_i']
            gq['ion'][ion_idx]['t_i_volume_average'] = results['T_i']