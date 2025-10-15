"""
Machine mapping module for VEST database.

This module provides functions for mapping raw diagnostic data to OMAS/IMAS data structures.
The module is organized into three main components:

- meta: Metadata and dataset description functions
- experiment: Raw diagnostic data processing functions with static and dynamic sources
- model: Model-based and derived physical quantities calculation functions
"""

from .meta import (
    dataset_description,
    summary
)

from .experiment import (
    pf_active,
    filterscope,
    barometry,
    tf,
    magnetics,
    ion_doppler_spectroscopy,
    spectrometer_uv,
    camera_visible,
    thomson_scattering
)

from .model import (
    em_coupling,
    equilibrium,
    mhd_linear,
    pf_passive,
    pf_plasma
)

__all__ = [
    'dataset_description',
    'summary',
    'pf_active',
    'filterscope',
    'barometry',
    'tf',
    'magnetics',
    'ion_doppler_spectroscopy',
    'spectrometer_uv',
    'camera_visible',
    'thomson_scattering',
    'em_coupling',
    'equilibrium',
    'mhd_linear',
    'pf_passive',
    'pf_plasma'
]
