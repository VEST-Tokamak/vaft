"""
Convert geqdsk to IMAS profiles_2d grid using FluxSurface straight_field_line coordinates
--------------------------------------------------------------------------------
This script reads a geqdsk file, builds a FluxSurface, computes straight-field-line coordinates (PEST),
and populates an OMAS ODS with profiles_2d (inverse_psi_straight_field_line).

Dependencies:
    pip install omfit_eqdsk omfit_classes omas numpy matplotlib

Usage:
    ods = geqdsk_to_imas_2d('g12345.00999', n_theta=129, out_nc='eq_imas.nc')
"""
import numpy as np
import matplotlib.pyplot as plt
from omas import ODS, save_omas_nc
from vaft.code.omfit_eqdsk import OMFITeqdsk
from vaft.code.fluxSurface import fluxSurfaces


def plot_sfl_grid(ods: ODS, save_plots: bool = True, method: str = 'straight_line') -> None:
    """
    Plot the straight-field-line (SFL) grid and flux surface mesh.

    Parameters
    ----------
    ods        : OMAS ODS containing the profiles_2d data
    save_plots : if True, save plots to files
    """
    ts = ods['equilibrium.time_slice[0]']
    prof2d = ts['profiles_2d[1]']
    
    # Plot 1: ψ-θ grid lines
    plt.figure(figsize=(10, 10))
    dim2 = prof2d['grid.dim2']
    nr = len(prof2d['grid.dim1'])
    
    # Draw constant ψ lines
    for i in range(nr):
        psi_val = prof2d['psi'][i, 0]
        plt.plot(dim2, np.full_like(dim2, psi_val), 'k-', lw=0.5)

    plt.xlabel(r'$\theta_{\rm SFL}$ [rad]')
    plt.ylabel(r'$\psi_{\rm norm}$')
    plt.title(f'ψ–θ grid lines (SFL mapping) {method}')
    if save_plots:
        plt.savefig(f'sfl_psi_theta_{method}.png')
    plt.show()

    # Plot 2: Flux surface mesh in R-Z plane
    R2D = prof2d['r'][:]
    Z2D = prof2d['z'][:]

    fig, ax = plt.subplots(figsize=(15, 15))
    nr, nt = R2D.shape
    
    # Draw flux surfaces
    for i in range(nr):
        ax.plot(R2D[i], Z2D[i], 'k-', lw=0.5)
    # Draw field lines
    for j in range(nt):
        ax.plot(R2D[:, j], Z2D[:, j], 'k-', lw=0.5)
        
    ax.set_aspect('equal')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(f'Flux Surface Mesh in R-Z Plane {method}')
    if save_plots:
        plt.savefig(f'flux_surface_mesh_{method}.png')
    plt.show()


def geqdsk_to_imas_2d(gfile: str,
                      n_theta: int = 129,
                      out_nc: str | None = None,
                      method: str = 'straight_line',
                      plot: bool = True) -> ODS:
    """
    Convert geqdsk → IMAS profiles_2d grid using FluxSurface straight_field_line.

    Parameters
    ----------
    gfile    : path to EQDSK file
    n_theta  : number of poloidal points
    out_nc   : if provided, save ODS to this netCDF file
    plot     : if True, generate plots of the SFL grid and flux surfaces
    method   : 'equal_arc', 'straight_line', 'hamada', 'boozer'

    Returns
    -------
    ods      : OMAS ODS with filled equilibrium/time_slice[0]/profiles_2d[0]
    """
    # Read equilibrium
    eq = OMFITeqdsk(gfile)

    # Build FluxSurface and compute PEST (straight_field_line)
    fs = fluxSurfaces(gEQDSK=eq, quiet=True)
    fs.findSurfaces()
    fs.calc_poloidal_angle(method=method, npts=n_theta)

    # Prepare OMAS container
    ods = eq.to_omas()
    ts = ods['equilibrium.time_slice[0]']

    # Set up grid dimensions
    dim1 = fs['levels']            # normalized psi levels (nr,)
    dim2 = fs['flux'][0]['theta']  # pest theta (n_theta,)

    # Configure grid type
    ts['profiles_2d[1].grid.dim1'] = dim1
    ts['profiles_2d[1].grid.dim2'] = dim2
    ts['profiles_2d[1].grid_type.index'] = 11
    ts['profiles_2d[1].grid_type.name'] = 'inverse_psi_straight_field_line'

    # Fill r, z, psi, theta arrays
    nr = len(dim1)
    nt = len(dim2)
    R2D = np.zeros((nr, nt))
    Z2D = np.zeros_like(R2D)
    psi2D = np.zeros_like(R2D)
    theta2D = np.zeros_like(R2D)

    for k in range(nr):
        R2D[k, :] = fs['flux'][k]['R']
        Z2D[k, :] = fs['flux'][k]['Z']
        psi2D[k, :] = fs['flux'][k]['psi']
        theta2D[k, :] = fs['flux'][k]['theta']

    # Store results in ODS
    ts['profiles_2d[1].r'] = R2D
    ts['profiles_2d[1].theta'] = theta2D
    ts['profiles_2d[1].z'] = Z2D
    ts['profiles_2d[1].psi'] = psi2D

    # Generate plots if requested
    if plot:
        plot_sfl_grid(ods, method=method)

    # Save to netCDF if requested
    if out_nc:
        save_omas_nc(ods, out_nc)

    return ods


if __name__ == '__main__':
    # Example usage
    gfile = '/home/user1/h5pyd/vaft/vaft/data/g039020.031180'
    n_theta = 129
    out_nc = 'ods_sfl.nc'

    ods = geqdsk_to_imas_2d(gfile, n_theta=n_theta, out_nc=out_nc, method='equal_arc', plot=True)
    ods = geqdsk_to_imas_2d(gfile, n_theta=n_theta, out_nc=out_nc, method='straight_line', plot=True)
    # ods = geqdsk_to_imas_2d(gfile, n_theta=n_theta, out_nc=out_nc, method='hamada', plot=True)
    ods = geqdsk_to_imas_2d(gfile, n_theta=n_theta, out_nc=out_nc, method='boozer', plot=True)
    print('Conversion completed successfully')
