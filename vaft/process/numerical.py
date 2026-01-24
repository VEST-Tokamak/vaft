import numpy as np
from typing import Union


def time_derivative(time: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Compute weighted derivative for non-uniform time-data arrays.
    
    Uses a weighted central difference scheme that accounts for non-uniform
    time spacing. For interior points, it uses a weighted average of forward
    and backward differences, with weights proportional to the time intervals.
    
    Parameters
    ----------
    time : np.ndarray
        Time array (non-uniform spacing allowed)
    data : np.ndarray
        Data array corresponding to time points
        
    Returns
    -------
    np.ndarray
        Derivative of data with respect to time, same shape as input data
        
    Notes
    -----
    - First point uses forward difference
    - Last point uses backward difference
    - Interior points use weighted central difference
    - Handles edge cases (single point, two points, etc.)
    """
    time = np.asarray(time)
    data = np.asarray(data)
    
    if time.shape != data.shape:
        raise ValueError(f"Time and data arrays must have the same shape. "
                        f"Got time.shape={time.shape}, data.shape={data.shape}")
    
    if len(time) < 2:
        raise ValueError("At least 2 time points are required for derivative calculation")
    
    # Initialize output array
    derivative = np.zeros_like(data, dtype=float)
    
    # Single point case (shouldn't happen due to check above, but handle gracefully)
    if len(time) == 1:
        return derivative
    
    # Two points case: use simple forward difference
    if len(time) == 2:
        dt = time[1] - time[0]
        if dt == 0:
            derivative[:] = 0.0
        else:
            derivative[0] = (data[1] - data[0]) / dt
            derivative[1] = derivative[0]  # Same for both points
        return derivative
    
    # For arrays with 3+ points:
    # First point: forward difference
    dt_forward = time[1] - time[0]
    if dt_forward != 0:
        derivative[0] = (data[1] - data[0]) / dt_forward
    else:
        derivative[0] = 0.0
    
    # Interior points: weighted central difference
    for i in range(1, len(time) - 1):
        dt_backward = time[i] - time[i-1]
        dt_forward = time[i+1] - time[i]
        
        # Avoid division by zero
        if dt_backward == 0 and dt_forward == 0:
            derivative[i] = 0.0
        elif dt_backward == 0:
            derivative[i] = (data[i+1] - data[i]) / dt_forward
        elif dt_forward == 0:
            derivative[i] = (data[i] - data[i-1]) / dt_backward
        else:
            # Weighted average: weight is proportional to the opposite time interval
            # This gives more weight to the closer neighbor
            w_backward = dt_forward  # Weight for backward difference
            w_forward = dt_backward  # Weight for forward difference
            
            backward_diff = (data[i] - data[i-1]) / dt_backward
            forward_diff = (data[i+1] - data[i]) / dt_forward
            
            derivative[i] = (w_backward * backward_diff + w_forward * forward_diff) / (w_backward + w_forward)
    
    # Last point: backward difference
    dt_backward = time[-1] - time[-2]
    if dt_backward != 0:
        derivative[-1] = (data[-1] - data[-2]) / dt_backward
    else:
        derivative[-1] = 0.0
    
    return derivative

