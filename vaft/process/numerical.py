"""Numerical primitives on plain arrays.

One function so far.  Anything that is a physics formula belongs in
:mod:`vaft.formula`; anything that knows about a diagnostic belongs in a
diagnostic module.  This is for the operations that are neither.

Notation
--------
t   : time                          [s]
x   : the sampled quantity          [any]
"""

import numpy as np
from typing import Union


__all__ = ["time_derivative"]


def time_derivative(time: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Time derivative of a sampled quantity on a non-uniform time axis.

    A weighted central difference: at each interior point the forward and
    backward differences are averaged with weights proportional to the
    *opposite* interval, so the closer neighbour counts more.  On a uniform
    grid this reduces to the ordinary central difference; the ends use a
    one-sided difference.

    Parameters
    ----------
    time : np.ndarray
        Sample times; spacing may vary [s].
    data : np.ndarray
        The sampled quantity, same shape as ``time`` [any].

    Returns
    -------
    np.ndarray
        ``d(data)/dt`` at each sample, same shape as ``data`` [any/s].

    Raises
    ------
    ValueError
        The arrays differ in shape, or hold fewer than two samples.

    Processing steps
    ----------------
    1. First sample: forward difference.  Last sample: backward difference.
    2. Interior sample ``i``: with ``h_b = t[i] - t[i-1]`` and
       ``h_f = t[i+1] - t[i]``, the derivative is
       ``(h_f * backward + h_b * forward) / (h_b + h_f)``.
    3. A zero interval on either side falls back to the difference across the
       other; zero on both sides yields ``0``.

    Convention
    ----------
    The weighting is the simple interval-weighted average of the two one-sided
    differences, not the second-order non-uniform stencil; it is what was
    written and is kept so that results do not move.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A zero time interval produces a ``0`` derivative silently rather than an
    error, so a duplicated timestamp hides as a flat spot.  The scheme is
    first-order accurate on a non-uniform grid and amplifies sample noise as
    any finite difference does; smooth or filter first if the derivative is
    to be read quantitatively.  The loop over interior points is pure Python
    and slow on long records.
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

