from typing import List, Tuple, Dict, Any
from numpy import ndarray
from omas import *
from vaft.process import compute_response_matrix, compute_response_vector
from vaft.formula import magnetic_shear, ballooning_alpha



def compute_magnetic_shear(ods, time_slice: slice) -> ndarray:
    """
    Compute magnetic shear from ODS
    """
    r = ods['equilibrium']['time_slice'][time_slice]['profiles_1d']['r']
    q = ods['equilibrium']['time_slice'][time_slice]['profiles_1d']['q']
    return magnetic_shear(r, q)


def compute_ballooning_alpha(ods, time_slice: slice) -> ndarray:
    """
    Compute ballooning alpha from ODS
    """
    V = ods['equilibrium']['profiles_1d']['V']
    R = ods['equilibrium']['profiles_1d']['R']
    p = ods['equilibrium']['profiles_1d']['p']
    psi = ods['equilibrium']['profiles_1d']['psi']
    return ballooning_alpha(V, R, p, psi)

