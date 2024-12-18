import numpy as np
from scipy import signal, integrate

# Naming convention for function name: {diagnostics_name}_{processing_quantity}
def b_field_pol_probe_field(
    time,
    raw,
    gain,
    coil_onset_time,
    lowpass_coeff
):
    """
    Process B-field poloidal probe data for multiple signals.

    This function takes time and raw data arrays from multiple magnetic probes that
    measure the poloidal component of the magnetic field. Each column of `raw`
    is considered a separate signal from a different probe.

    Steps:
    1. Apply a low-pass filter.
    2. Integrate the signal to obtain magnetic flux.
    3. Remove a linear offset determined by fitting a polynomial (usually linear)
       to data before the coil onset time.

    Parameters
    ----------
    time : np.ndarray, shape [m]
        Time array for the signals.
    raw : np.ndarray, shape [m x n]
        Measured raw data from the B-field poloidal probes. Each column is a 
        separate signal.
    gain : np.ndarray, shape [n]
        Gain factor for each probe signal. Must match the number of columns in `raw`.
    coil_onset_time : float
        Time index (in seconds) up to which we consider the pre-onset data for
        baseline correction.
    lowpass_coeff : np.ndarray
        Coefficients for the low-pass filter.

    Returns
    -------
    processed_time : np.ndarray, shape [m]
        Time array after processing (same as input).
    field : np.ndarray, shape [m x n]
        Integrated, offset-corrected B-field data for each probe.
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # Check dimensions
    m, n = data.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match number of signals (n).")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples (m).")
    
    # Determine offset fitting range
    integration_index = np.argmin(np.abs(time - coil_onset_time))
    offset_fit_range = (0, integration_index)
    fit_start, fit_end = offset_fit_range

    # Apply low-pass filter to each signal
    filtered_raw = signal.lfilter(lowpass_coeff, 1, raw, axis=0)

    # Integrate to get flux for each signal
    # int_flux: [m x n]
    # Negative sign as per original code
    int_flux = -integrate.cumtrapz(filtered_raw / gain, time, initial=0, axis=0)

    # Remove offset by fitting a line to the pre-onset region for each signal
    flux = np.empty_like(int_flux)
    for i in range(n):
        coeff = np.polyfit(time[fit_start:fit_end], int_flux[fit_start:fit_end, i], 1)
        int_flux_corrected = int_flux[:, i] - np.polyval(coeff, time)
        flux[:, i] = int_flux_corrected

    return time, flux

def flux_loop_flux(
    time,
    raw,
    gain,
    coil_onset_time
):
    """
    Process flux loop data for multiple signals.

    This function integrates the flux loop raw data (which typically measures total
    flux) for multiple signals at once. Each column of `raw` is a separate 
    flux loop signal. The procedure:
    1. Integrate the raw data to obtain flux (dividing by gain and applying a 
       negative sign and 1/(2*pi) factor).
    2. Remove linear offsets using data before the coil onset time.

    Parameters
    ----------
    time : np.ndarray, shape [m]
        Time array for the flux loop signals.
    raw : np.ndarray, shape [m x n]
        Measured raw data from multiple flux loops. Each column is a separate
        signal.
    gain : np.ndarray, shape [n]
        Gain factor for each flux loop signal. Must match number of columns in `raw`.
    coil_onset_time : float
        Time index used to determine offset fitting region.

    Returns
    -------
    processed_time : np.ndarray, shape [m]
        Time array after processing (same as input).
    processed_data : np.ndarray, shape [m x n]
        Integrated, offset-corrected flux data for each loop.
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # Check dimensions
    m, n = data.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match number of signals (n).")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples (m).")

    integration_index = np.argmin(np.abs(time - coil_onset_time))
    offset_fit_range = (0, integration_index)
    fit_start, fit_end = offset_fit_range

    # Integrate flux loop data for each signal
    # The original code uses '-integrate.cumtrapz(...) / (2*pi)'
    int_flux = -integrate.cumtrapz(data / gain, time, initial=0, axis=0) / (2 * np.pi)

    # Remove offset for each signal
    flux = np.empty_like(int_flux)
    for i in range(n):
        coeff = np.polyfit(time[fit_start:fit_end], int_flux[fit_start:fit_end, i], 1)
        int_flux_corrected = int_flux[:, i] - np.polyval(coeff, time)
        flux[:, i] = int_flux_corrected

    return time, flux