"""
Model-based and derived physical quantities calculation module for VEST database.

This module provides functions for calculating model-based and derived physical quantities
from raw diagnostic data and storing them in OMAS/IMAS data structure format.
"""

import os
import yaml
import numpy as np
from omas import ODS
import xarray as xr
import re
import struct

def em_coupling(ods: ODS, options: dict = None) -> None:
    """
    Calculate electromagnetic coupling.
    
    Args:
        ods: OMAS data structure
        options: Optional dictionary containing calculation options
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
    """
    from .em_coupling import em_coupling as _em_coupling

    _em_coupling(ods, options)

def equilibrium(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate equilibrium quantities.
    
    Args:
        ods: OMAS data structure
        source: Source for equilibrium data (file path or shot number)
        options: Optional dictionary containing calculation options
            - time_slice: Optional float for specific time point
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    from .equilibrium import equilibrium as _equilibrium

    _equilibrium(ods, source, options)



# # =============================================================================
# # Equilibrium from External Analysis (Element/Profiles)
# # =============================================================================

# def vfit_element_analysis_from_file(ods: Dict[str, Any], shot: int) -> None:
#     """
#     Populate ODS equilibrium from external element analysis mat file.

#     :param ods: ODS containing 'equilibrium'.
#     :param shot: Shot number.
#     """
#     print("Reading element analysis data from .mat (stub).")
#     # mat = _read_mat_from_ElementAnalysis(shot)
#     # parse mat and store in ods accordingly
#     print("vfit_equilibrium_from_element_analysis done (stub).")


# def vfit_from_profile_fitting_from_file(ods: Dict[str, Any], shot: int) -> None:
#     """
#     Populate ODS equilibrium from external profile fitting mat file.

#     :param ods: ODS containing 'equilibrium'.
#     :param shot: Shot number.
#     """
#     print("Reading profile fitting data from .mat (stub).")
#     # mat = _read_mat_from_ProfileFitting(shot)
#     # parse mat and store in ods
#     print("vfit_equilibrium_from_profile_fitting done (stub).")


def mhd_linear(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate linear MHD stability.
    
    Args:
        ods: OMAS data structure
        source: Source for MHD data (file path or shot number)
            "example : /srv/vest.filedb/public/39915" -> read multiple mode for multiple time slice in stability analysis such as dcon, rdcon and stride
        options: Optional dictionary containing calculation options
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    from .mhd_linear import mhd_linear as _mhd_linear

    _mhd_linear(ods, source, options)



def pf_passive(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate passive PF coil effects.
    
    Args:
        ods: OMAS data structure
        source: Source for passive coil data (file path or shot number)
        options: Optional dictionary containing calculation options
            - time_slice: Optional float for specific time point
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    from .pf_passive import pf_passive as _pf_passive

    _pf_passive(ods, source, options)

def pf_plasma(ods: ODS, source: str, options: dict = None) -> None:
    """
    Calculate plasma current distribution effects.
    
    Args:
        ods: OMAS data structure
        source: Source for plasma current data (file path or shot number)
        options: Optional dictionary containing calculation options
            - time_slice: Optional float for specific time point
            - method: Optional str to specify calculation method
            - params: Optional dict for calculation parameters
            - source_type: Optional str ('file' or 'shot')
    """
    from .pf_plasma import pf_plasma as _pf_plasma

    _pf_plasma(ods, source, options)
