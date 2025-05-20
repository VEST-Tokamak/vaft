"""
Utility functions for plasma physics calculations.

This module provides common utility functions used throughout the formula module.
"""

import numpy as np
from typing import Union, Tuple, List, Dict

def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate gradient with proper handling of array dimensions.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
        
    Returns
    -------
    np.ndarray
        Gradient dy/dx
    """
    return np.gradient(y, x)

def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate definite integral using trapezoidal rule.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
        
    Returns
    -------
    float
        Definite integral ∫y dx
    """
    return np.trapz(y, x)

def normalize_profile(x: Union[float, np.ndarray],
                     x_axis: float,
                     x_boundary: float) -> Union[float, np.ndarray]:
    """
    Normalize a profile to range [0,1].
    
    Parameters
    ----------
    x : Union[float, np.ndarray]
        Values to normalize
    x_axis : float
        Value at axis
    x_boundary : float
        Value at boundary
        
    Returns
    -------
    Union[float, np.ndarray]
        Normalized values: (x - x_axis) / (x_boundary - x_axis)
    """
    return (x - x_axis) / (x_boundary - x_axis)

def calculate_peaking_factor(central: float,
                           volume_avg: float) -> float:
    """
    Calculate peaking factor PF = X(0) / ⟨X⟩.
    
    Parameters
    ----------
    central : float
        Central value
    volume_avg : float
        Volume-averaged value
        
    Returns
    -------
    float
        Peaking factor
    """
    return central / volume_avg

def calculate_volume_weighted_average(x: np.ndarray,
                                    V: np.ndarray) -> float:
    """
    Calculate volume-weighted average of a profile.
    
    Parameters
    ----------
    x : np.ndarray
        Profile values
    V : np.ndarray
        Volume elements
        
    Returns
    -------
    float
        Volume-weighted average
    """
    return np.sum(x * V) / np.sum(V)

def calculate_poloidal_flux(R: np.ndarray,
                          B_theta: np.ndarray,
                          l: np.ndarray,
                          psi_axis: float = 0.0) -> float:
    """
    Calculate poloidal flux from line integral of R*B_theta.
    
    Parameters
    ----------
    R : np.ndarray
        Major radius values
    B_theta : np.ndarray
        Poloidal magnetic field values
    l : np.ndarray
        Path length values
    psi_axis : float, optional
        Poloidal flux at magnetic axis, by default 0.0
        
    Returns
    -------
    float
        Poloidal flux value
    """
    return trapz_integral(l, R * B_theta) + psi_axis

def calculate_toroidal_flux(B_phi: np.ndarray,
                          dA: np.ndarray) -> float:
    """
    Calculate toroidal flux through surface.
    
    Parameters
    ----------
    B_phi : np.ndarray
        Toroidal magnetic field values
    dA : np.ndarray
        Area elements
        
    Returns
    -------
    float
        Toroidal flux value
    """
    return np.sum(B_phi * dA) 