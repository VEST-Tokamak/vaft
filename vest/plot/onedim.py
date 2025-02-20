from vest.process import psi_to_radial
import matplotlib.pyplot as plt

# legend -> time_points sec (time_slice_index)
# axis -> radial, psiN, rhoN, vertical...

# def {ods}_{axis}_{quantity}(ods, time_slice):

# def onedim_radial_equilibrium_summary
# def onedim_radial_equilibrium_q(ods, time_slice=None):
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
    

# def onedim_radial_equilibrium_pressure
# def onedim_radial_equilibrium_j_tor
# def onedim_radial_equilibrium_pprime
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

