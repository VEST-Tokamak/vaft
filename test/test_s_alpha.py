import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import brentq
from omas import *

# =============================================
# Formulas
# =============================================
def magnetic_shear_1d(r, q):
    """Calculate magnetic shear from q profile"""
    q = np.where(q == 0, 1e-10, q)
    dqdr = np.gradient(q, r, edge_order=2)
    return (r / q) * dqdr

def ballooning_alpha_1d(mu0, V, R, p, psi):
    """Calculate ballooning alpha parameter"""
    dVdpsi = np.gradient(V, psi)
    dpdpsi = np.gradient(p, psi)
    sqrt_term = np.sqrt(V / (2 * np.pi**2 * R))
    return (mu0 / (2 * np.pi**2)) * dVdpsi * sqrt_term * dpdpsi

# =============================================
# ODS Mapping Functions
# =============================================
def get_profile_data(ts, time_slice=0, grid_index=0):
    """Extract equilibrium data from ODS"""    
    # 1D profiles
    profiles_1d = {
        'q': np.asarray(ts['profiles_1d']['q']),
        'p': np.asarray(ts['profiles_1d']['pressure']),
        'psi': np.asarray(ts['profiles_1d']['psi']),
        'V': np.asarray(ts['profiles_1d']['volume'])
    }
    
    # 2D profiles
    prof2d = ts['profiles_2d'][grid_index]
    profiles_2d = {
        'R_grid': np.asarray(prof2d['grid.dim1']),
        'Z_grid': np.asarray(prof2d['grid.dim2']),
        'psi': np.asarray(prof2d['psi'])
    }
    
    # Geometry
    geometry = {
        'R0': ts['global_quantities']['magnetic_axis']['r'],
        'Z0': ts['global_quantities']['magnetic_axis']['z'],
        'Rb': np.asarray(ts['boundary']['outline']['r']),
        'Zb': np.asarray(ts['boundary']['outline']['z'])
    }
    
    return profiles_1d, profiles_2d, geometry

# =============================================
# Core Functions
# =============================================
def ray_boundary_length(R0, Z0, Rb, Zb, theta):
    """
    Find first intersection length ℓ with LCFS for a given polar angle θ.
    """
    cosT, sinT = np.cos(theta), np.sin(theta)
    ell_min = np.inf
    
    # Debug information
    print(f"\nDebug for theta = {theta} radians ({theta*180/np.pi} degrees):")
    print(f"R0 = {R0}, Z0 = {Z0}")
    print(f"Boundary points: {len(Rb)} points")
    print(f"R range: [{min(Rb)}, {max(Rb)}]")
    print(f"Z range: [{min(Zb)}, {max(Zb)}]")
    
    for i in range(len(Rb) - 1):
        x1, y1 = Rb[i] - R0, Zb[i] - Z0
        x2, y2 = Rb[i+1] - R0, Zb[i+1] - Z0
        dx, dy = x2 - x1, y2 - y1
        denom = dx * sinT - dy * cosT
        if np.abs(denom) < 1e-12:  # parallel
            continue
        u = (x1 * sinT - y1 * cosT) / denom
        t = (x1 + u*dx) / cosT if np.abs(cosT) > np.abs(sinT) else (y1 + u*dy) / sinT
        if 0.0 <= u <= 1.0 and t > 0 and t < ell_min:
            ell_min = t
            print(f"Found intersection at segment {i}: u={u}, t={t}")
    
    if np.isinf(ell_min):
        print("No intersection found. Possible issues:")
        print(f"1. Ray direction: cos(theta)={cosT}, sin(theta)={sinT}")
        print(f"2. Boundary might be too far from axis or not properly defined")
        print(f"3. Theta angle might be pointing away from the plasma")
        raise RuntimeError("ray does not hit LCFS – check θ or boundary definition")
    
    return ell_min

def minor_radius_1d(ts, theta, grid_index=0):
    """
    Return ℓ_k (axis→psi_k surface) for every psi_k in profiles_1d
    along the ray defined by poloidal angle θ.
    """
    # Get data from ODS
    profiles_1d, profiles_2d, geometry = get_profile_data(ts, 0, grid_index)
    
    # Extract data
    psi_k = profiles_1d['psi']
    N = len(psi_k)
    R0, Z0 = geometry['R0'], geometry['Z0']
    Rb, Zb = geometry['Rb'], geometry['Zb']
    
    # Length to LCFS in this theta
    ell_LCFS = ray_boundary_length(R0, Z0, Rb, Zb, theta)
    
    # Create psi(R,Z) interpolator
    fpsi = RegularGridInterpolator((profiles_2d['Z_grid'], profiles_2d['R_grid']), 
                                 profiles_2d['psi'],
                                 bounds_error=False, fill_value=np.nan)
    
    # Helper function for psi on ray
    def psi_on_ray(l):
        return fpsi([[Z0 + l*np.sin(theta), R0 + l*np.cos(theta)]])[0]
    
    # Check monotonicity
    increasing = psi_on_ray(0) < psi_on_ray(ell_LCFS)
    
    r_minor = np.empty(N)
    eps = 1e-6  # minimum increment [m]
    lo = 0.0
    
    for i, psi_target in enumerate(psi_k):
        try:
            r_minor[i] = brentq(lambda l: psi_on_ray(l) - psi_target,
                              lo, ell_LCFS, xtol=1e-10, rtol=1e-10)
            # Ensure minimum increment
            if i > 0 and r_minor[i] <= r_minor[i-1]:
                r_minor[i] = r_minor[i-1] + eps
            lo = r_minor[i]
        except ValueError:
            # If root not found, ensure minimum increment
            r_minor[i] = r_minor[i-1] + eps if i > 0 else eps
            lo = r_minor[i]
    
    return r_minor

def line_s_alpha(ods, theta_target=0.0, time_slice=0, grid_index=0):
    """Calculate s and alpha along a line at given theta"""
    ts = ods['equilibrium']['time_slice'][time_slice]
    
    # Get data from ODS
    profiles_1d, profiles_2d, geometry = get_profile_data(ts, time_slice, grid_index)
    
    # Extract geometry
    R0, Z0 = geometry['R0'], geometry['Z0']
    Rb, Zb = geometry['Rb'], geometry['Zb']
    
    # Calculate LCFS length
    ell_LCFS = ray_boundary_length(R0, Z0, Rb, Zb, theta_target)
    
    # Create psi(R,Z) interpolator
    fpsi = RegularGridInterpolator((profiles_2d['Z_grid'], profiles_2d['R_grid']), 
                                 profiles_2d['psi'],
                                 bounds_error=False, fill_value=np.nan)
    
    # Helper function for psi on ray
    psi_on_ray = lambda l: fpsi([[Z0 + l*np.sin(theta_target),
                                R0 + l*np.cos(theta_target)]])[0]
    
    print("ψ(0) =", psi_on_ray(0))
    print("ψ(ell_LCFS) =", psi_on_ray(ell_LCFS))
    
    # Calculate minor radius
    r1d = minor_radius_1d(ts, theta_target, grid_index)
    
    # Calculate s and alpha
    s1d = magnetic_shear_1d(r1d, profiles_1d['q'])
    alpha1d = ballooning_alpha_1d(MU0, profiles_1d['V'], r1d, 
                                profiles_1d['p'], profiles_1d['psi'])
    
    # Interpolate for local values
    r_local = r1d.copy()
    s_of_r = interp1d(r1d, s1d, kind='cubic',
                     bounds_error=False, fill_value=np.nan)
    s_local = s_of_r(r_local)
    
    alpha_of_psi = interp1d(profiles_1d['psi'], alpha1d, kind='cubic',
                           bounds_error=False, fill_value=np.nan)
    alpha_local = alpha_of_psi(psi_on_ray(r_local))
    
    return r_local, s_local, alpha_local, r1d, s1d, alpha1d

def compute_s_alpha_2d_ods(ods, theta_target, grid_index=0, grid_kind='total'):
    """
    Compute magnetic shear and ballooning alpha parameter for all time slices in an ODS.
    """
    if 'equilibrium' not in ods or 'time_slice' not in ods['equilibrium']:
        raise ValueError("ODS does not contain equilibrium data")
    
    n_slices = len(ods['equilibrium']['time_slice'])
    if n_slices == 0:
        raise ValueError("No time slices found in equilibrium data")
    
    times = np.array(ods['equilibrium']['time'])
    
    # Single time slice case
    if n_slices == 1:
        r_local, s_local, alpha_local, _, _, _ = line_s_alpha(ods, theta_target, 0)
        return r_local, s_local, alpha_local, times
    
    # Multiple time slices case
    results = []
    for i in range(n_slices):
        try:
            result = line_s_alpha(ods, theta_target, i)
            results.append(result[:3])  # Only keep r_local, s_local, alpha_local
        except Exception as e:
            print(f"Warning: Error processing time slice {i}: {e}")
    
    if not results:
        raise ValueError("No valid results from any time slice")
    
    # Stack results
    stacked_results = np.array(results).transpose(1, 0, 2)
    r_local = stacked_results[0]
    s_local = stacked_results[1]
    alpha_local = stacked_results[2]
    
    return r_local, s_local, alpha_local, times

def plot_s_alpha_diagram(paths, theta_list, save_path='s_alpha_diagram.png'):
    """
    Create s-alpha diagram for various poloidal paths.
    
    Parameters:
    -----------
    paths : list
        List of tuples containing (r_local, s_local, alpha_local, r1d, s1d, alpha1d)
    theta_list : list
        List of poloidal angles corresponding to each path
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot each path
    for (r_local, s_local, alpha_local, _, _, _), theta in zip(paths, theta_list):
        # Convert theta to degrees for the label
        theta_deg = theta * 180 / np.pi
        plt.plot(s_local, alpha_local, '-', label=f'θ = {theta_deg:.1f}°')
    
    # Add stability boundaries (example)
    s = np.linspace(-1, 2, 100)
    alpha_ideal = 0.6 * s  # Ideal ballooning stability boundary
    plt.plot(s, alpha_ideal, 'k--', label='Ideal Ballooning')
    
    plt.xlabel('Magnetic Shear (s)')
    plt.ylabel('Ballooning Alpha (α)')
    plt.title('s-α Diagram for Various Poloidal Angles')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_radial_profiles(paths, theta_list, save_path='radial_profiles.png'):
    """
    Create radial profiles of s and alpha for various poloidal paths.
    
    Parameters:
    -----------
    paths : list
        List of tuples containing (r_local, s_local, alpha_local, r1d, s1d, alpha1d)
    theta_list : list
        List of poloidal angles corresponding to each path
    save_path : str
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Plot s profiles
    for (r_local, s_local, _, _, _, _), theta in zip(paths, theta_list):
        theta_deg = theta * 180 / np.pi
        ax1.plot(r_local, s_local, '-', label=f'θ = {theta_deg:.1f}°')
    
    ax1.set_ylabel('Magnetic Shear (s)')
    ax1.set_title('Radial Profiles of Magnetic Shear')
    ax1.grid(True)
    ax1.legend()
    
    # Plot alpha profiles
    for (r_local, _, alpha_local, _, _, _), theta in zip(paths, theta_list):
        theta_deg = theta * 180 / np.pi
        ax2.plot(r_local, alpha_local, '-', label=f'θ = {theta_deg:.1f}°')
    
    ax2.set_xlabel('Minor Radius (m)')
    ax2.set_ylabel('Ballooning Alpha (α)')
    ax2.set_title('Radial Profiles of Ballooning Alpha')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================
# Main Execution
# =============================================
if __name__ == '__main__':
    ods = ODS()
    ods = ods.sample()
    MU0 = 4 * np.pi * 1e-7
    
    # Define poloidal angles to analyze
    theta_list = [0.0+1e-2, np.pi/3, np.pi/3*2, np.pi+1e-2, -np.pi/3*2, -np.pi/3]
    paths = [line_s_alpha(ods, th) for th in theta_list]
    
    # Create plots
    plot_s_alpha_diagram(paths, theta_list)
    plot_radial_profiles(paths, theta_list)
    
    # Original plots
    # -----------------------------------------------------------------
    # Plot 1 : boundary + theta paths
    # -----------------------------------------------------------------
    plt.figure()
    boundary_R = ods['equilibrium']['time_slice'][0]['boundary']['outline']['r']
    boundary_Z = ods['equilibrium']['time_slice'][0]['boundary']['outline']['z']
    plt.plot(boundary_R, boundary_Z, 'k-', linewidth=2, label='Plasma Boundary')

    for (r_local, _, _, _, _, _), th in zip(paths, theta_list):
        # Get the current time slice profiles
        prof2d = ods['equilibrium']['time_slice'][0]['profiles_2d'][0]
        R_axis = ods['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['r']
        Z_axis = ods['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['z']
        
        # Get the path coordinates from the already computed paths
        path_R = r_local * np.cos(th) + R_axis
        path_Z = r_local * np.sin(th) + Z_axis
        
        # Plot the path
        plt.plot(path_R, path_Z, '-', label=f'θ = {th:.2f}')

    plt.gca().set_aspect('equal')
    plt.title('Plasma Boundary and θ-paths')
    plt.xlabel('R [m]'); plt.ylabel('Z [m]')
    plt.grid(True)
    plt.legend()
    plt.savefig('boundary_theta_paths.png')
    plt.close()

    # -----------------------------------------------------------------
    # Plot 2 : s(r) for each θ
    # -----------------------------------------------------------------
    psi_1d = ods['equilibrium']['time_slice'][0]['profiles_1d']['psi']
    psi_1d_norm = (psi_1d - psi_1d[0]) / (psi_1d[-1] - psi_1d[0])

    plt.figure()
    for (r_l, s_l, _, _, _, _), th in zip(paths, theta_list):
        plt.plot(psi_1d_norm, s_l, label=f'θ={th:.2f}')
    plt.title('s(r) along θ paths')
    plt.xlabel('minor radius [m]'); plt.ylabel('s')
    plt.legend(); plt.grid()
    plt.savefig('s_r_theta_paths.png')
    plt.close()

    # -----------------------------------------------------------------
    # Plot 3 : α(r) local vs 1‑D
    # -----------------------------------------------------------------
    plt.figure()
    for (r_l, _, a_l, _, _, _), th in zip(paths, theta_list):
        plt.plot(psi_1d_norm, a_l, label=f'local θ={th:.2f}')
    # 1‑D α
    plt.plot(paths[0][3], paths[0][5], linestyle='--', label='α 1‑D')
    plt.title('α(r): local vs 1‑D')
    plt.xlabel('minor radius [m]'); plt.ylabel('α')
    plt.legend(); plt.grid()
    plt.savefig('alpha_r_theta_paths.png')
    plt.close()

    # -----------------------------------------------------------------
    # Plot 4 : s(r) comparison at inboard/outboard
    # -----------------------------------------------------------------
    r_inboard, s_pi, _, _, _, _  = paths[theta_list.index(np.pi+1e-2)]
    r_outboard, s_0, _, R1d, s1d_full, _ = paths[theta_list.index(0.0+1e-2)]

    s_inb_1d = np.interp(r_inboard, ods['equilibrium']['time_slice'][0]['profiles_1d']['r_inboard'], s1d_full)
    s_out_1d = np.interp(r_outboard, ods['equilibrium']['time_slice'][0]['profiles_1d']['r_outboard'], s1d_full)

    plt.figure()
    plt.plot(psi_1d_norm, s_pi, label='s local θ=π')
    plt.plot(psi_1d_norm, s_inb_1d, linestyle='--', label='s 1‑D inboard')
    plt.plot(psi_1d_norm, s_0, label='s local θ=0')
    plt.plot(psi_1d_norm, s_out_1d, linestyle='--', label='s 1‑D outboard')
    plt.title('s(r): local vs 1‑D (inboard/outboard)')
    plt.xlabel('minor radius [m]'); plt.ylabel('s')
    plt.legend(); plt.grid()
    plt.savefig('s_r_inboard_outboard.png')
    plt.close()

    # -----------------------------------------------------------------
    # Plot 5 : ψ(r) vs r_minor
    # -----------------------------------------------------------------
    plt.figure()
    plt.plot(r_inboard, psi_1d, '.-', label='inboard')
    plt.plot(r_outboard, psi_1d, '.-', label='outboard')
    plt.title(f'r_minor vs ψ_k')    
    plt.xlabel('r_minor [m]'); plt.ylabel('ψ_k [Wb]')
    plt.legend(); plt.grid()
    plt.savefig('psi_r_minor.png')
    plt.close()

    print('done')

    # Example usage with real IMAS data:
    """
    import imas
    ids = imas.DB('shot', run).open()
    ods = ids.to_ods()

    # For a single time slice:
    r, s, alpha, R1d, s1d, alpha1d = line_s_alpha(ods, theta=0)

    # For all time slices:
    r_local, s_local, alpha_local, times = compute_s_alpha_2d_ods(ods, theta_target=0)
    """
