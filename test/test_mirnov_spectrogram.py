import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import vaft  # Assuming 'vaft' is your custom data loading module

# --- Core Processing Functions ---

def design_fir_filter(sampling_rate: float, cutoff_freq, filter_type: str) -> np.ndarray:
    """Designs FIR filter coefficients using the firwin method."""
    filter_order = int(sampling_rate * 1e-3)
    if filter_order % 2 == 0:
        filter_order += 1  # Ensure odd order for Type I filter

    if filter_type not in ['lowpass', 'highpass', 'bandpass']:
        raise ValueError("Filter type must be 'lowpass', 'highpass', or 'bandpass'")

    pass_zero_mapping = {'lowpass': True, 'highpass': False, 'bandpass': False}
    return signal.firwin(
        filter_order,
        cutoff_freq,
        fs=sampling_rate,
        pass_zero=pass_zero_mapping[filter_type],
        window='hann'
    )

def load_signal(shot_config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Loads raw signal data from the database."""
    shot = shot_config["shot_number"]
    field = shot_config["field_id"]
    print(f"--- Loading data for Shot #{shot} (Field: {field}) ---")
    time_raw, data_raw = vaft.database.load_raw(shot, field)
    return time_raw, data_raw

def process_signal(data_raw: np.ndarray, proc_config: dict) -> np.ndarray:
    """Filters and calibrates the raw signal."""
    fs = proc_config["sampling_rate"]
    
    # Design band-pass filter by applying high-pass and low-pass sequentially
    b_high = design_fir_filter(fs, proc_config["high_pass_cutoff"], 'highpass')
    b_low = design_fir_filter(fs, proc_config["low_pass_cutoff"], 'lowpass')
    
    # Remove DC offset from the entire signal
    processed = data_raw - np.mean(data_raw)
    
    # Apply filters with zero-phase filtering
    processed = signal.filtfilt(b_high, 1.0, processed)
    processed = signal.filtfilt(b_low, 1.0, processed)
    
    # Apply amplifier gain calibration
    data_processed = processed / proc_config["amplifier_gain"]
    print("--- Signal processing complete ---")
    return data_processed

def calculate_stft_manual(
    time_raw: np.ndarray, 
    data_processed: np.ndarray, 
    shot_config: dict, 
    stft_config: dict,
    proc_config: dict
) -> dict:
    """
    Runs a manual STFT, replicating the original MATLAB script's algorithm.
    """
    fs = proc_config["sampling_rate"]
    window_size = stft_config["window_size"]
    shot_number = shot_config["shot_number"]
    time_range = stft_config.get("time_range")
    
    # Select time range for analysis (replicating MATLAB's index logic)
    if time_range:
        start_idx = np.argmax(time_raw >= time_range[0])
        end_idx = np.argmax(time_raw >= time_range[1])
        xtime_indices = np.arange(start_idx, end_idx)
    else: # Use hardcoded default index ranges from the original script
        if shot_number < 16566:
            xtime_indices = np.arange(len(time_raw) - window_size)
        elif shot_number < 41660:
            xtime_indices = np.arange(15000, 25000 - window_size)
        else:
            xtime_indices = np.arange(10000, 25000 - window_size)
    
    # Manual STFT implementation
    num_time_bins = len(xtime_indices)
    window_data_matrix = np.zeros((window_size, num_time_bins))
    
    half_window = window_size // 2
    for i, center_idx in enumerate(xtime_indices):
        start_slice, end_slice = center_idx - half_window, center_idx + half_window
        if start_slice >= 0 and end_slice <= len(data_processed):
            window_data_matrix[:, i] = data_processed[start_slice:end_slice]

    # Apply window function and perform FFT
    hann_window = signal.windows.hann(window_size)
    windowed_data = window_data_matrix * hann_window[:, np.newaxis]
    Y = np.fft.fft(windowed_data, axis=0)
    
    # Scale the spectrum (replicating MATLAB's P1/P2 logic)
    P2 = np.abs(Y / window_size)
    P1 = P2[0:half_window + 1, :]
    P1[1:-1, :] *= 2

    stft_result = {
        'frequencies': np.fft.rfftfreq(window_size, d=1/fs),
        'time_bins': time_raw[xtime_indices],
        'magnitude': P1
    }
    print("--- Manual STFT calculation complete ---")
    return stft_result

# --- Plotting Functions ---

def plot_spectrogram(stft_result: dict, shot_config: dict, plot_config: dict):
    """
    Visualizes the STFT result, plotting linear magnitude.
    """
    t = stft_result['time_bins']
    f = stft_result['frequencies']
    Sxx_mag = stft_result['magnitude']

    fig, ax = plt.subplots(figsize=(12, 6))
    
    c = ax.pcolormesh(
        t * 1000,
        f / 1e3,
        Sxx_mag,
        shading='gouraud',
        cmap=plot_config["colormap"],
        # Use vmin/vmax from config for linear scale if provided
        vmin=plot_config.get("color_lim", None),
        vmax=plot_config.get("color_lim_max_fraction", 1.0) * np.max(Sxx_mag) if plot_config.get("color_lim_max_fraction") else None
    )

    if "freq_lim_khz" in plot_config:
        ax.set_ylim(plot_config["freq_lim_khz"])

    ax.set_title(f'Spectrogram for Shot #{shot_config["shot_number"]} (Field: {shot_config["field_id"]})')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (kHz)')
    fig.colorbar(c, ax=ax, label='Magnitude (a.u.)')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Central Configuration ---
    # All parameters for an analysis run are defined in this single dictionary.
    CONFIG = {
        "shot_info": {
            "shot_number": 39915,
            "field_id": 171
        },
        "processing": {
            "sampling_rate": 250e3,
            "high_pass_cutoff": 2e3,
            "low_pass_cutoff": 90e3,
            "amplifier_gain": -0.05
        },
        "stft": {
            "time_range": [0.31, 0.32],
            "window_size": 500,
        },
        "plotting": {
            "freq_lim_khz": [0, 80],
            "colormap": "viridis",
            # Optional: for linear scale, set max color to a fraction of the data's peak
            # "color_lim_max_fraction": 0.5 
        }
    }

    # --- Functional Workflow Execution ---
    # 1. Load data
    time_raw, data_raw = load_signal(CONFIG["shot_info"])
    
    # 2. Process the signal
    data_processed = process_signal(data_raw, CONFIG["processing"])

    # 3. Calculate STFT
    stft_result = calculate_stft_manual(
        time_raw,
        data_processed,
        CONFIG["shot_info"],
        CONFIG["stft"],
        CONFIG["processing"]
    )

    # 4. Generate the plot
    plot_spectrogram(stft_result, CONFIG["shot_info"], CONFIG["plotting"])
