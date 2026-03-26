"""Lazy process namespace for reusable numerical kernels."""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "numerical": ".numerical",
    "signal_processing": ".signal_processing",
    "electromagnetics": ".electromagnetics",
    "magnetics": ".magnetics",
    "equilibrium": ".equilibrium",
    "profile": ".profile",
}

_EXPORT_MAP = {
    "time_derivative": ".numerical",
    "smooth": ".signal_processing",
    "vest_coil_current_noise_reduction": ".signal_processing",
    "VEST_CoilCurrentNoiseReduction": ".signal_processing",
    "vfit_signal_start_end": ".signal_processing",
    "vfit_signal_startend": ".signal_processing",
    "process_signal": ".signal_processing",
    "define_baseline": ".signal_processing",
    "subtract_baseline": ".signal_processing",
    "signal_on_offset": ".signal_processing",
    "signal_onoffset": ".signal_processing",
    "is_signal_active": ".signal_processing",
    "dist": ".electromagnetics",
    "self_inductance_new": ".electromagnetics",
    "self_induM_new": ".electromagnetics",
    "compute_br_bz_phi": ".electromagnetics",
    "calc_grid": ".electromagnetics",
    "compute_response_matrix": ".electromagnetics",
    "compute_response_vector": ".electromagnetics",
    "compute_impedance_matrices": ".electromagnetics",
    "solve_eddy_currents": ".electromagnetics",
    "compute_vacuum_fields_1d": ".electromagnetics",
    "radial_to_psi": ".equilibrium",
    "psi_to_rho": ".equilibrium",
    "rho_to_psi": ".equilibrium",
    "psi_to_rz": ".equilibrium",
    "psi_to_RZ": ".equilibrium",
    "calculate_reconstructed_diamagnetic_flux": ".equilibrium",
    "calculate_diamagnetism": ".equilibrium",
    "volume_average": ".equilibrium",
    "psi_to_radial": ".equilibrium",
    "poloidal_field_at_boundary": ".equilibrium",
    "calculate_average_boundary_poloidal_field": ".equilibrium",
    "shafranov_integrals": ".equilibrium",
}

__all__ = sorted(list(_SUBMODULES.keys()) + list(_EXPORT_MAP.keys()))


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name], __name__)
        globals()[name] = module
        return module

    module_name = _EXPORT_MAP.get(name)
    if module_name is not None:
        module = import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
