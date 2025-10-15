# update_summary()
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from vaft.formula import psi_norm
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
        [time_slice] if isinstance(time_slice, int) else time_slice)
    
    for idx, ts in enumerate(time_slices):
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
            plt.figure(figsize=[15,5])
            plt.suptitle(f'Time Slice {idx} Validation')
            
            # Primary mapping validation
            plt.subplot(131)
            plt.plot(r_in, psi_in, 'r-', label='2D Inboard')
            plt.plot(ts['profiles_1d.r_inboard'], psi_1d, 'b--', label='1D Mapped')
            plt.xlabel('R [m]'), plt.ylabel('Psi'), plt.legend()
            
            plt.subplot(132)
            plt.plot(r_out, psi_out, 'g-', label='2D Outboard')
            plt.plot(ts['profiles_1d.r_outboard'], psi_1d, 'm--', label='1D Mapped')
            plt.xlabel('R [m]'), plt.ylabel('Psi'), plt.legend()

            # Derivative validation
            if plot_opt >= 2:
                plt.subplot(133)
                dr_in = np.gradient(ts['profiles_1d.r_inboard'], psi_1d)
                dr_out = np.gradient(ts['profiles_1d.r_outboard'], psi_1d)
                plt.plot(psi_1d, dr_in, label='Inboard dR/dψ')
                plt.plot(psi_1d, dr_out, label='Outboard dR/dψ')
                plt.xlabel('Psi'), plt.ylabel('Derivative')
                plt.legend()

            plt.tight_layout()
            plt.show()

<<<<<<< Updated upstream
def update_equilibrium_coordinates(ods):
    """Main entry point for updating all equilibrium coordinates"""
    # Update normalized psi for all time slices
    for ts in ods['equilibrium.time_slice']:

        psi = ods[f'equilibrium.time_slice.{ts}.profiles_1d.psi']
        psi_axis = ods[f'equilibrium.time_slice.{ts}.global_quantities.psi_axis']
        psi_bdry = ods[f'equilibrium.time_slice.{ts}.global_quantities.psi_boundary']
=======
def update_equilibrium_boundary(ods, time_slice=None):
    """
    Update geometric axis for all time slices.
    """
    time_slices = ods['equilibrium.time_slice'] if time_slice is None else (
        [time_slice] if isinstance(time_slice, int) else time_slices)
    
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
    for idx in time_slice:
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
    for idx in time_slice:
        ts = ods['equilibrium.time_slice'][idx]
        if 'profiles_1d.volume' not in ts:
            print(f"Warning: profiles_1d.volume not found for time slice {idx}")
            continue
        ts['global_quantities.volume'] = ts['profiles_1d.volume'][-1]

>>>>>>> Stashed changes

        ods[f'equilibrium.time_slice.{ts}.profiles_1d.psi_norm'] = psi_norm(psi, psi_axis, psi_bdry)

        ods[f'equilibrium.time_slice.{ts}.profiles_1d.psi_norm'] = (
            ods[f'equilibrium.time_slice.{ts}.profiles_1d.psi'][0] - ods[f'equilibrium.time_slice.{ts}.profiles_1d.psi_magnetic_axis']
        ) / (
            ods['equilibrium.time_slice.0.profiles_1d.psi'][-1] - ods['equilibrium.time_slice.0.profiles_1d.psi_magnetic_axis']
        )
    
    # Update radial coordinates with validation plots for first slice
    update_equilibrium_profiles_1d_radial_coordinates(ods, time_slice=0, plot_opt=2)


def update_equilibrium_coordinates(ods):
    """
    Update the psi_norm, r_inboard, r_outboard information in the equilibrium ODS.
    """

    # psi_norm
    ods['equilibrium.time_slice.0.profiles_1d.psi_norm'] = (
        ods['equilibrium.time_slice.0.profiles_1d.psi'][0] - ods['equilibrium.time_slice.0.profiles_1d.psi_magnetic_axis']
    ) / (
        ods['equilibrium.time_slice.0.profiles_1d.psi'][-1] - ods['equilibrium.time_slice.0.profiles_1d.psi_magnetic_axis']
    )

    # r_inboard, r_outboard
    update_equilibrium_profiles_1d_radial_coordinates(ods, time_slice=0, plot_opt=0)

<<<<<<< Updated upstream
=======

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
        

if __name__ == '__main__':
    ods = ODS()
>>>>>>> Stashed changes
