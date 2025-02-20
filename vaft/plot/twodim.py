from omas import *


def twodim_geometry_all(ods):
    ods.plot_wall_overlay(color='lightgray')
    ods.plot_bolometer_overlay()
    ods.plot_charge_exchange_overlay()
    ods.plot_thomson_scattering_overlay()
    ods.plot_interferometer_overlay()
    ods.plot_magnetics_overlay( flux_loop_style={'marker': 's'},
    pol_probe_style={'marker': 'x'},
    tor_probe_style={'marker': 'o'}
    )
    ods.plot_pf_active_overlay(edgecolor='red')
    ods.plot_gas_injection_overlay()
    ods.plot_position_control_overlay()
    ods.plot_langmuir_probes_overlay()


# def twodim_geometry_coil():
# def twodim_geometry_wall():
# def twodim_geometry_vessel():


# def twodim_equilibrium_boundary():
# def twodim_equilibrium_magnetic_axis():
# def twodim_equilibrium_j_tor():
# def twodim_equilibrium_psi():
# def twodim_equilibrium_q():
# def twodim_equilibrium_f():
# def twodim_equilibrium_ffprime():

