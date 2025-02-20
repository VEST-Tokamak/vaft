import numpy as np
from scipy import signal, integrate
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from numpy.polynomial.polynomial import polyfit, polyval
import statistics
import scipy
from ipywidgets import interact, IntSlider
import numpy as np
from scipy.signal import csd, coherence, find_peaks
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from vaft.process import define_baseline, subtract_baseline

# Naming convention for function name: {diagnostics_name}_{processing_quantity}

def rogowski_coil_ip(
    time,
    rogowski_raw,
    flux_loop_raw,
    flux_loop_gain=11,
    effective_vessel_res=5.8e-4,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
    smooth_window=10
):
    """
    Compute the plasma current from Rogowski coil and flux loop signals.

    The function:
    1. Defines baseline indices near baseline_onset and baseline_offset.
    2. Fits a baseline (using baseline_type) and subtracts it from both rogowski_raw and flux_loop_raw.
    3. Converts the flux_loop_raw to a "reference current" by multiplying by flux_loop_gain.
    4. Optionally smooths the flux loop reference with a Savitzky-Golay filter (smooth_window).
    5. Subtracts flux_loop_ref from rogowski_raw to get the final Ip.
    6. If the resulting signal is predominantly negative, flips its sign.

    Parameters
    ----------
    time : np.ndarray
        Time array for the signals.
    rogowski_raw : np.ndarray
        Raw Rogowski coil data array (plasma current sensor).
    flux_loop_raw : np.ndarray
        Raw flux loop array (used as a reference).
    flux_loop_gain : float, optional
        Gain factor/multiplier for flux loop data. Default is 11.
    effective_vessel_res : float, optional
        Effective vessel resistance or mutual inductance factor. (Currently not used in baseline.)
    baseline_onset : float, optional
        Time (in seconds) at which signals begin to deviate; used for baseline definition.
    baseline_offset : float, optional
        Time (in seconds) at which signals return to baseline; used for baseline definition.
    baseline_type : {'linear','quadratic','spline','exp'}, optional
        Type of baseline fitting to use. Default 'linear'.
    baseline_onset_window : int, optional
        Number of samples before baseline_onset to include in the baseline fit. Default 500.
    baseline_offset_window : int, optional
        Number of samples after baseline_offset to include in the baseline fit. Default 100.
    smooth_window : int, optional
        Window size for Savitzky-Golay smoothing of flux_loop reference. Default 10 (must be odd).

    Returns
    -------
    time : np.ndarray
        Same time array as input.
    ip : np.ndarray
        Final processed plasma current signal.
    """
    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)

    # Define baseline indices
    baseline_indices_rogowski = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )
    baseline_indices_flux = baseline_indices_rogowski  # same region, typically

    # Subtract baseline from rogowski
    rogowski_corr, rogowski_baseline = subtract_baseline(
        time, rogowski_raw, baseline_indices_rogowski, fitting_opt=baseline_type
    )

    # Subtract baseline from flux loop
    flux_corr, flux_baseline = subtract_baseline(
        time, flux_loop_raw, baseline_indices_flux, fitting_opt=baseline_type
    )

    # Convert flux loop signal to current reference
    # For example: flux_ref = flux_corr * (flux_loop_gain / mutual_inductance)
    # We'll do the simplest version: flux_corr * flux_loop_gain
    flux_ref = flux_corr * flux_loop_gain

    # Smooth the flux loop reference (Savitzky-Golay) if smooth_window > 2
    if smooth_window < 3:
        smooth_window = 3
    if smooth_window % 2 == 0:
        smooth_window += 1

    flux_ref_smooth = savgol_filter(flux_ref, smooth_window, polyorder=1)

    # Final plasma current
    ip = rogowski_corr - flux_ref_smooth

    # If absolute negative peak is larger than the positive peak, invert
    if abs(np.min(ip)) > abs(np.max(ip)):
        ip = -ip

    return time, ip


def b_field_pol_probe_field(
    time,
    raw,
    gain,
    lowpass_param,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
    plot_opt=False,
):
    """
    Process B-field poloidal probe data with gain applied first.
    """
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    m, n = raw.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match the number of signals (columns in raw).")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples (rows in raw).")

    # Apply gain at the start
    raw = raw * gain

    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)

    baseline_indices = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )

    # Apply low-pass filter
    filtered_raw = signal.lfilter(lowpass_param, [1.0], raw, axis=0)

    # Integrate to get flux (negative sign if your system defines it so)
    integrated_flux = -integrate.cumtrapz(filtered_raw, time, initial=0, axis=0)

    # Subtract baseline for each column
    field = np.empty_like(integrated_flux)
    baselines = np.empty_like(integrated_flux)
    for i in range(n):
        flux_corrected, baseline = subtract_baseline(
            time, integrated_flux[:, i], baseline_indices, fitting_opt=baseline_type
        )
        field[:, i] = flux_corrected
        baselines[:, i] = baseline

    if plot_opt:
        def interactive_plot(index):
            plt.figure(figsize=(10, 8))

            # Plot 1: Raw and filtered signals
            plt.subplot(2, 1, 1)
            plt.plot(time, raw[:, index], label="Raw (gain applied)", alpha=0.7)
            plt.plot(time, filtered_raw[:, index], label="Filtered Signal", alpha=0.7)
            plt.title(f"B-field Signal Processing: Index {index}\nBaseline: {baseline_type}")
            plt.xlabel("Time (s)")
            plt.ylabel("Signal")
            plt.legend()
            plt.grid()

            # Plot 2: Integrated signal and baseline-corrected signal
            plt.subplot(2, 1, 2)
            plt.plot(time, integrated_flux[:, index], label="Integrated Signal", alpha=0.7)
            plt.plot(time, baselines[:, index], label="Baseline", alpha=0.7)
            plt.plot(time, field[:, index], label="Baseline-Corrected Signal", alpha=0.7)
            plt.xlabel("Time (s)")
            plt.ylabel("Flux")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

        interact(interactive_plot, index=IntSlider(min=0, max=n-1, step=1, value=0))

    return raw, filtered_raw, integrated_flux, field, baselines

def flux_loop_flux(
    time,
    raw,
    gain,
    baseline_onset=0.27,
    baseline_offset=0.28,
    baseline_type='linear',
    baseline_onset_window=500,
    baseline_offset_window=100,
    plot_opt=False,
):
    """
    Process flux loop data for multiple signals.

    Steps:
    1. Integrate the raw data (dividing by gain) to obtain flux, 
       including a negative sign and 1/(2*pi) factor if desired.
    2. Remove baseline offsets using define_baseline + subtract_baseline.

    Parameters
    ----------
    time : np.ndarray, shape [m]
        Time array for the flux loop signals.
    raw : np.ndarray, shape [m x n]
        Measured raw data from multiple flux loops. Each column is a separate signal.
    gain : np.ndarray, shape [n]
        Gain factor for each flux loop signal. Must match number of columns in `raw`.
    baseline_onset : float
        Time in seconds to define the start of the baseline region.
    baseline_offset : float
        Time in seconds to define the end of the baseline region.
    baseline_type : {'linear','quadratic','spline','exp'}
        Type of baseline fitting to use.
    baseline_onset_window : int
        Number of samples before baseline_onset to include in the baseline fit.
    baseline_offset_window : int
        Number of samples after baseline_offset to include in the baseline fit.
    plot_opt : bool
        Whether to plot the results interactively.

    Returns
    -------
    time : np.ndarray, shape [m]
        Same as input.
    processed_data : np.ndarray, shape [m x n]
        Integrated, baseline-corrected flux data for each loop.
    baselines : np.ndarray, shape [m x n]
        Baseline values for each signal.
    """
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    m, n = raw.shape
    if gain.shape[0] != n:
        raise ValueError("Length of gain must match number of signals.")
    if time.shape[0] != m:
        raise ValueError("Length of time must match number of samples.")

    # Apply gain at the start
    raw = raw * gain

    # Convert baseline onset/offset in seconds to integer indices
    onset_idx = np.searchsorted(time, baseline_onset)
    offset_idx = np.searchsorted(time, baseline_offset)
    baseline_indices = define_baseline(
        time, onset_idx, baseline_onset_window, offset_idx, baseline_offset_window
    )

    # Integrate flux loop data for each signal
    # - sign if that is convention, also / (2*pi)
    integrated_data = -integrate.cumtrapz(raw, time, initial=0, axis=0) / (2 * np.pi)

    # Remove offset for each signal
    processed_data = np.empty_like(integrated_data)
    baselines = np.empty_like(integrated_data)
    for i in range(n):
        flux_corrected, baseline = subtract_baseline(
            time, integrated_data[:, i], baseline_indices, fitting_opt=baseline_type
        )
        processed_data[:, i] = flux_corrected
        baselines[:, i] = baseline

    if plot_opt:
        def interactive_plot(index):
            plt.figure(figsize=(10, 8))

            # Plot 1: Raw signal
            plt.subplot(2, 1, 1)
            plt.plot(time, raw[:, index], label="Raw (gain applied)", alpha=0.7)
            plt.title(f"Flux Loop Signal Processing: Index {index}\nBaseline: {baseline_type}")
            plt.xlabel("Time (s)")
            plt.ylabel("Signal")
            plt.legend()
            plt.grid()

            # Plot 2: Integrated signal and baseline-corrected signal
            plt.subplot(2, 1, 2)
            plt.plot(time, integrated_data[:, index], label="Integrated Signal", alpha=0.7)
            plt.plot(time, baselines[:, index], label="Baseline", alpha=0.7)
            plt.plot(time, processed_data[:, index], label="Baseline-Corrected Signal", alpha=0.7)
            plt.xlabel("Time (s)")
            plt.ylabel("Flux")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

        interact(interactive_plot, index=IntSlider(min=0, max=n-1, step=1, value=0))

    return time, processed_data, baselines


# def toroidal_mode_analysis(
#     time_vector, 
#     signal_matrix, 
#     toroidal_angles, 
#     time_points, 
#     window_size=1000, 
#     thres_peak=0.1, 
#     plot_opt=False,
#     nperseg=256,
#     coherence_q=4
# ):
#     """
#     Compute coherence, phase, toroidal mode number, and relative power using the first signal as reference.
    
#     Parameters
#     ----------
#     time_vector : np.ndarray
#         Time axis vector (e.g., 0~1s, 250kHz sampling -> length 250000)
#     signal_matrix : np.ndarray
#         2D array of shape (num_signals x num_samples).
#         Each row represents a different probe(channel), each column represents a time sample.
#     toroidal_angles : np.ndarray
#         Toroidal angles (in radians) corresponding to each probe(row). Length num_signals.
#     time_points : list or np.ndarray
#         Time indices at which to perform analysis (e.g., [1000, 2000, 3000, ...])
#     window_size : int
#         Window size determining how many samples to analyze around each time_point.
#         (default 1000 -> ±500 points)
#     thres_peak : float
#         Minimum height ratio for peak detection relative to maximum spectrum value (default 0.1).
#     plot_opt : bool
#         If True, displays a simple phase plot with slider.
#     nperseg : int
#         nperseg value to use for csd, coherence calculations (default 256).
#     coherence_q : int
#         q value used for coherence threshold calculation. (default 4)
#         Generalizes the original tanh(1.96 / sqrt(2*q-2)) formula.
    
#     Returns
#     -------
#     results : dict
#         {
#           "time": [t1, t2, ...],             # Actual analysis times (seconds)
#           "coherence": [...],               # Array of coherence values for valid peaks for [num_signals-1] channels at each time_point
#           "phase": [...],                   # Array of phase values
#           "mode_number": [...],             # Array of mode numbers
#           "frequencies": [...],             # Array of peak frequencies
#           "power": [...]                    # Array of relative peak powers
#         }
#     """

#     num_signals, num_samples = signal_matrix.shape
#     if len(toroidal_angles) != num_signals:
#         raise ValueError("The number of toroidal angles must match the number of signals.")
    
#     # 샘플링 주파수(Hz)
#     f_sample = 1.0 / np.mean(np.diff(time_vector))
    
#     # 코히런스 임계값(원본 코드 아이디어)
#     coherence_threshold = np.tanh(1.96 / np.sqrt(2 * coherence_q - 2))
    
#     results = {
#         "time": [],
#         "coherence": [],
#         "phase": [],
#         "mode_number": [],
#         "frequencies": [],
#         "power": []
#     }
    
#     all_time_results = []  # 플롯에서 슬라이더로 접근 가능하도록 저장
    
#     half_win = window_size // 2
    
#     for t_idx in time_points:
#         # 창 범위 확인
#         if t_idx < half_win or t_idx >= num_samples - half_win:
#             continue
        
#         window_start = t_idx - half_win
#         window_end   = t_idx + half_win
        
#         ref_signal = signal_matrix[0, window_start:window_end]
#         ref_angle  = toroidal_angles[0]
        
#         time_results = {
#             "coherence": [],
#             "phase": [],
#             "mode_number": [],
#             "frequencies": [],
#             "power": []
#         }
        
#         # 각 프로브(i=1~num_signals-1)에 대해
#         for i in range(1, num_signals):
#             signal_i = signal_matrix[i, window_start:window_end]
            
#             # Cross-spectral density
#             f, pxy = csd(ref_signal, signal_i, fs=f_sample, nperseg=nperseg)
#             magnitude = np.abs(pxy)
            
#             # Coherence
#             _, cxy = coherence(ref_signal, signal_i, fs=f_sample, nperseg=nperseg)
            
#             # 피크 찾기: 크기가 thres_peak * max(magnitude) 이상인 피크
#             peaks, peak_props = find_peaks(
#                 magnitude, 
#                 height=thres_peak * np.max(magnitude)
#             )
#             # 크기 기준 내림차순 정렬
#             peak_heights = peak_props["peak_heights"]
#             desc_order = np.argsort(peak_heights)[::-1]
#             peaks = peaks[desc_order]
            
#             # 코히런스 필터
#             valid_peaks = []
#             for pk in peaks:
#                 if cxy[pk] > coherence_threshold:
#                     valid_peaks.append(pk)
#             valid_peaks = np.array(valid_peaks, dtype=int)
            
#             if len(valid_peaks) > 0:
#                 # 위상(각 유효 피크에서)
#                 phase_vals = np.angle(pxy[valid_peaks])
                
#                 # 모드 번호: (phase / Δphi)
#                 delta_phi = toroidal_angles[i] - ref_angle
#                 n_raw = phase_vals / delta_phi
#                 n_rounded = np.round(n_raw).astype(int)
                
#                 # 상대 파워: 각 피크의 |pxy| / 전체 스펙트럼 |pxy| 합
#                 total_power = np.sum(magnitude)
#                 power_vals  = magnitude[valid_peaks] / total_power
                
#                 time_results["coherence"].append(cxy[valid_peaks])
#                 time_results["phase"].append(phase_vals)
#                 time_results["mode_number"].append(n_rounded)
#                 time_results["frequencies"].append(f[valid_peaks])
#                 time_results["power"].append(power_vals)
#             else:
#                 # 유효 피크가 없으면 빈 배열 저장
#                 time_results["coherence"].append(np.array([]))
#                 time_results["phase"].append(np.array([]))
#                 time_results["mode_number"].append(np.array([]))
#                 time_results["frequencies"].append(np.array([]))
#                 time_results["power"].append(np.array([]))
        
#         # 전체 결과에 추가
#         results["time"].append(time_vector[t_idx])
#         results["coherence"].append(time_results["coherence"])
#         results["phase"].append(time_results["phase"])
#         results["mode_number"].append(time_results["mode_number"])
#         results["frequencies"].append(time_results["frequencies"])
#         results["power"].append(time_results["power"])
        
#         all_time_results.append(time_results)
    
#     if plot_opt:
#         fig, ax = plt.subplots()
#         plt.subplots_adjust(bottom=0.25)

#         def update_plot(idx):
#             ax.clear()
#             time_idx = time_points[idx]
#             time_result = all_time_results[idx]

#             # Plot reference point
#             ax.scatter([toroidal_angles[0]], [0], marker='o', color='red',
#                        label='Reference (0°)', s=100)

#             # Plot phase differences relative to reference
#             for i in range(num_signals - 1):
#                 phases = time_result["phase"][i]
#                 if len(phases) > 0:
#                     # 여러 피크가 있을 수 있으나 여기서는 평균값만 예시로 표시
#                     ax.scatter([toroidal_angles[i+1]], [np.mean(phases)], 
#                                marker='o', label=f'Probe {i+1}')

#             # 대략적 모드번호 피팅 예시 (평균 모드 사용)
#             if any(len(mn) > 0 for mn in time_result["mode_number"]):
#                 valid_modes = [np.mean(mn) for mn in time_result["mode_number"] if len(mn) > 0]
#                 if valid_modes:
#                     mean_mode = np.mean(valid_modes)
#                     theta = np.linspace(0, 2*np.pi, 100)
#                     ax.plot(theta, mean_mode * theta, 'r--', label=f'n={mean_mode:.1f}')

#             ax.set_title(f'Time: {time_vector[time_idx]:.5f} s')
#             ax.set_xlabel('Toroidal Angle (rad)')
#             ax.set_ylabel('Phase Difference (rad)')
#             ax.set_ylim(-np.pi, np.pi)
#             ax.set_xlim(0, 2*np.pi)
#             ax.legend()
#             ax.grid(True)
#             plt.draw()

#         ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
#         slider = Slider(ax_slider, "Time Index", 0, len(time_points) - 1, 
#                         valinit=0, valstep=1)
#         slider.on_changed(lambda val: update_plot(int(val)))
#         update_plot(0)
#         plt.show()

#     return results

