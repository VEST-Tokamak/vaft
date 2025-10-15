# update_summary()
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from vaft.formula import normalize_psi
from omas import *
from omfit_classes.omfit_eqdsk import OMFITeqdsk
from omfit_classes.fluxSurface import fluxSurfaces
# update_diagnostics_file(ods, filename)

# print_available_ids(ods)

<<<<<<< Updated upstream
=======
def update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=None):
    """
    Update normalized poloidal flux (psi_norm) for all time slices.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    # Process all time slices if not specified
    time_slices = ods['equilibrium.time_slice'] if time_slice is None else (
        [time_slice] if isinstance(time_slice, int) else time_slice)
    
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
>>>>>>> Stashed changes
def update_equilibrium_profiles_1d_normalized_psi(ods, time_slice=None):
    """
    Update normalized poloidal flux (psi_n) for all time slices.
    
    Parameters:
        ods (OMAS structure): Input OMAS data structure
        time_slice (int/list/None): Specific time slice(s) to process. None=all
    """
    # Process all time slices if not specified
    time_slices = ods['equilibrium.time_slice'] if time_slice is None else (
        [time_slice] if isinstance(time_slice, int) else time_slice)
    
    for idx in time_slices:
        ts = ods['equilibrium.time_slice'][idx]
        # Get psi values
        psi = ts['profiles_1d.psi']
        psi_axis = ts['global_quantities.psi_axis']
        psi_bdry = ts['global_quantities.psi_boundary']
        
        # Calculate normalized psi
        psi_n = (psi - psi_axis) / (psi_bdry - psi_axis)
        
        # Store in ODS
        ts['profiles_1d.psi_n'] = psi_n

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
        [time_slice] if isinstance(time_slice, int) else time_slices)
    
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
    for idx in time_slice:
        ts = ods['equilibrium.time_slice'][idx]
        ts['global_quantities.q_min'] = ts['profiles_1d.q'].min()


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
    elif isinstance(time_slice, int):
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

if __name__ == '__main__':
    ods = ODS()
