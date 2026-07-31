import vaft
from omas import *
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings

# ----------------------------------------------------------------------
# Find information from ODS
# ----------------------------------------------------------------------
def find_shotnumber(ods):
    """Find the shot number from the ODS."""
    return ods['dataset_description.data_entry.pulse']

def find_shotclass(ods,plot_opt=0):
    """
    !!! Obsolete function:
    it is not successfully find the shot class. Need to be improved.

    Find shot class from ODS
    # 'Plasma': Plasma discharge is stable and Halpha is detected
    # 'Vacuum': Vacuum Test Shot
    # 'Breakdown failure': Try to discharge but failed (BD Test Shot)

    """
    # check existence of barometry and spectrometer
    if 'barometry' not in ods:
        print('Barometry not found in ODS')
        return
    if 'spectrometer_uv' not in ods:
        print('Spectrometer not found in ODS')
        return

    # Check status: Vacuum, BD failure, Plasma
    time_pres=ods['barometry.gauge.0.pressure.time'] # time
    data_pres=ods['barometry.gauge.0.pressure.data'] # pressure
    data_alpha=ods['spectrometer_uv.channel.0.processed_line.0.intensity.data'] # Halpha
    time_alpha=ods['spectrometer_uv.time'] # Halpha

    if vaft.process.is_signal_active(data_alpha, 0.01):
        status = 'Plasma'
    else:
        if not vaft.process.is_signal_active(data_pres): # no pressure?
            status='Vacuum'
        else:
            status='BD failure'
    return status

def find_chamber_boundary(ods):
    """Find the chamber boundary from the ODS."""
    return ods['wall.description_2d.0.limiter.unit.0.outline.r'], ods['wall.description_2d.0.limiter.unit.0.outline.z']

def _find_signal_onset(ods, time_key, data_key):
    """Helper to find signal onset using vaft.process.signal_on_offset."""
    time = ods.time(time_key)
    data = ods[data_key]
    onset, _ = vaft.process.signal_on_offset(time, data, threshold = 0.05)
    return onset

def find_breakdown_onset(ods):
    """Find the onset of the breakdown using spectrometer_uv signal."""
    return _find_signal_onset(ods, 'spectrometer_uv', 'spectrometer_uv.channel.0.processed_line.0.intensity.data')

def find_vloop_onset(ods):
    """Find the onset of the loop voltage signal (maximum of flux loop signal)."""
    time = ods.time('magnetics')
    flux = ods['magnetics.flux_loop.0.flux.data']
    return time[np.argmax(flux)]

def find_ip_onset(ods):
    """Find the onset of the plasma current signal."""
    return _find_signal_onset(ods, 'magnetics', 'magnetics.ip.0.data')

def find_pf_active_onset(ods):
    """Find the onset for each pf_active channel current signal."""
    time = ods.time('pf_active')
    onsets = []
    for i in range(len(ods['pf_active.channel'])):
        current = ods[f'pf_active.channel.{i}.current.data']
        onset, _ = vaft.process.signal_on_offset(time, current)
        onsets.append(onset)
    return onsets

def find_pulse_duration(ods):
    """Find the duration of the pulse using spectrometer_uv signal."""
    time = ods.time('spectrometer_uv')
    data = ods['spectrometer_uv.channel.0.processed_line.0.intensity.data']
    onset, offset = vaft.process.signal_on_offset(time, data, threshold=0.05)
    return offset - onset

def find_max_ip(ods):
    """Find the maximum plasma current."""
    current = ods['magnetics.ip.0.data']
    from scipy.signal import medfilt
    data_filtered = medfilt(current, kernel_size=15)
    
    max_org = np.max(current)
    max_filtered = np.max(data_filtered)

    print(f"Original max IP: {max_org}, Filtered max IP: {max_filtered}")

    if ods['dataset_description']['data_entry']['pulse'] == 40919 or ods['dataset_description']['data_entry']['pulse'] == '40919':
        if max_filtered > 100:
            raise RuntimeError("조건을 만족하지 않아 종료합니다.")
    
    return np.max(data_filtered)

def find_bt(ods):
    """Find the mean toroidal field at R=0.4m during plasma."""
    time = ods.time('tf.time')
    bt = ods['tf.b_field_tor_vacuum_r.data'] / ods['tf.r0']
    plasma_onset = find_breakdown_onset(ods)
    pulse_duration = find_pulse_duration(ods)
    plasma_offset = plasma_onset + pulse_duration
    onset_idx = np.searchsorted(time, plasma_onset)
    offset_idx = np.searchsorted(time, plasma_offset)
    bt = bt[onset_idx:offset_idx]
    return np.mean(bt)

def find_major_radius(ods):
    """Placeholder for finding major radius."""
    print('to do')

# ----------------------------------------------------------------------
# Common Helper: ODS ↔ ODC distinction
# ----------------------------------------------------------------------
def odc_or_ods_check(odc_or_ods):
    """
    Check input type and initialize ODC if necessary.
    
    Parameters:
    odc_or_ods (ODC or ODS): Input object to check.
    
    Returns:
    ODC: Initialized ODC object.
    
    Raises:
    TypeError: If input is not of type ODS or ODC.
    """
    if isinstance(odc_or_ods, ODC):            # already ODC
        return odc_or_ods
    elif isinstance(odc_or_ods, ODS):          # single ODS → wrap in ODC
        odc = ODC()
        odc['0'] = odc_or_ods
        return odc
    else:
        raise TypeError("Input must be an ODS or an ODC")
    
# ----------------------------------------------------------------------
# Time convention conversion (ODS × N → applied to entire ODC)
# ----------------------------------------------------------------------
def shift_time(one_ods, time_shift):
    """
    Shifts ONLY a narrow, explicitly defined set of time-like fields.

    1. Uses .paths() to ensure it works in all environments.
    2. Protects reference times under 'summary.code.parameters'.
    3. **Crucially, checks if the LEAF node's name itself is 'time', 'onset', or 'offset'.
       This is the definitive fix for the data corruption issue (e.g., 'ip' being shifted).**
    """
    protected_path_str = 'summary.code.parameters'

    # 1. 필수적인 .paths() 순회 사용
    for path in one_ods.paths():
        # path가 비어있는 엣지 케이스 방지
        if not path:
            continue
            
        # 2. 기준 시간 경로 보호
        path_str = '.'.join(map(str, path))
        if path_str.startswith(protected_path_str):
            continue
            
        # 3. ✨ 최종 핵심 로직: 경로의 '마지막 이름'이 정확히 일치하는지 확인
        if path[-1] in ('time', 'onset', 'offset'):
            try:
                val = one_ods[path_str]
                if isinstance(val, (np.ndarray, float, int)):
                    # 값을 변경
                    one_ods[path] = val + time_shift
            except (LookupError, TypeError, ValueError):
                # 값을 읽을 수 없는 중간 노드는 정상적으로 무시
                pass

def change_time_convention(odc_or_ods, convention='vloop'):
    """
    Convert time convention of ODS or ODC. (Improved Version)
    
    Features:
    - Fixes the critical state corruption bug by using the improved shift_time_v2.
    """
    # This part remains the same as the original code
    odc = odc_or_ods_check(odc_or_ods)

    for shot_key, ods in odc.items():
        params = ods.setdefault('summary.code.parameters', CodeParameters())

        if 'vloop_onset' not in params:
            params['time_convention'] = 'daq'
            params['vloop_onset']      = find_vloop_onset(ods)
            params['ip_onset']         = find_ip_onset(ods)
            params['breakdown_onset']  = find_breakdown_onset(ods)
        original = params.get('time_convention', 'daq')
        if original == convention:
            continue

        onsets = {
            'daq':        0,
            'vloop':      params['vloop_onset'],
            'ip':         params['ip_onset'],
            'breakdown':  params['breakdown_onset'],
        }
        if original not in onsets or convention not in onsets:
            raise ValueError(f"[{shot_key}] Unknown convention: {original} → {convention}")

        time_shift = onsets[original] - onsets[convention]
        print(f"[{shot_key}] shift {time_shift:+.6g} s  ({original} → {convention})")

        shift_time(ods, time_shift)
        params['time_convention'] = convention

    return odc

# ----------------------------------------------------------------------
# Print info
# ----------------------------------------------------------------------
def print_info(ods, key_name=None):
    """Print summary information and key structure of ODS."""
    if key_name is None:
        print("{:<20} : {}".format("Machine_name", ods['dataset_description.data_entry.machine']))
        print("{:<20} : {}".format("Shot_number", ods['dataset_description.data_entry.pulse']))
        print("{:<20} : {}".format("Operation_type", ods['dataset_description.data_entry.pulse_type']))
        print("{:<20} : {}".format("Run", ods['dataset_description.data_entry.run']))
        print("{:<20} : {}".format("User_name", ods['dataset_description.data_entry.user']))
        print(" {:<20} : {}\n".format("KEY", "VALUES"))
        for key in ods.keys():
            print(" {:<20}".format(key), ':', ','.join(ods[key].keys()))
    else:
        if key_name in ods.keys():
            print(f"\n Number of {key_name} Data set \n")
            for key in ods[key_name]:
                if key in ("time", "ids_properties"):
                    continue
                print("  {:<17} : {}".format(key, len(ods[key_name][key])))
        else:
            print("key_name value Error!")

def classify_shot(ods, pressure_threshold=0.01, halpha_threshold=0.01):
    """Determine the classification of a shot based on pressure and H-alpha signals."""
    try:
        data_pres = ods['barometry.gauge.0.pressure.data']
        if not vaft.process.is_signal_active(data_pres, threshold=pressure_threshold):
            return 'Vacuum'
        data_alpha = ods['spectrometer_uv.channel.0.processed_line.0.intensity.data']
        if not vaft.process.is_signal_active(data_alpha, threshold=halpha_threshold):
            return 'BD failure'
        try:
            ip = ods['magnetics.ip.0.data']
            if np.max(ip) > 0:
                return 'Plasma'
            else:
                return 'BD failure'
        except Exception:
            return 'Plasma'
    except Exception as e:
        print(f"Error in find_shotclass: {str(e)}")
        return 'Vacuum'

# ----------------------------------------------------------------------
# Combine ODS
# ----------------------------------------------------------------------
def find_matching_time_indices(ods, time_slice=None, atol: float = 1.0e-6):
    """
    Find matching time indices between core_profiles and equilibrium time slices.
    
    This function determines the core profile time slice index and finds the corresponding
    equilibrium time slice index by matching times. The closest equilibrium
    time slice is accepted when it is within ``atol`` of the core-profile time.
    
    Parameters
    ----------
    ods : ODS
        OMAS data structure
    time_slice : int, optional
        Desired time slice index for core profile. If None, uses index 0.
        If provided index is out of range, defaults to 0.
    
    Returns
    -------
    tuple
        (cp_idx, equil_idx, time) where:
        - cp_idx: Core profile time slice index
        - equil_idx: Matching equilibrium time slice index
        - time: Time value (must be identical for both core_profiles and equilibrium)
    
    Raises
    ------
    KeyError
        If required data structures are missing in ODS
    ValueError
        If no equilibrium time is within ``atol`` of the selected core-profile time
    """
    # Basic availability checks
    if 'core_profiles.profiles_1d' not in ods:
        raise KeyError("core_profiles.profiles_1d not found in ODS")
    if 'equilibrium.time_slice' not in ods or not len(ods['equilibrium.time_slice']):
        raise KeyError("equilibrium.time_slice not found in ODS")
    
    # Determine time slice for core profile
    if time_slice is None:
        cp_idx = 0
    else:
        cp_idx = time_slice if time_slice < len(ods['core_profiles.profiles_1d']) else 0
    
    cp_ts = ods['core_profiles.profiles_1d'][cp_idx]
    
    # Get core profile time
    if 'time' in cp_ts:
        cp_time = float(cp_ts['time'])
    elif 'core_profiles.time' in ods and cp_idx < len(ods['core_profiles.time']):
        cp_time = float(ods['core_profiles.time'][cp_idx])
    else:
        cp_time = float(cp_idx)
    
    # Find matching equilibrium time slice
    equil_times = []
    for idx in range(len(ods['equilibrium.time_slice'])):
        eq_ts = ods['equilibrium.time_slice'][idx]
        if 'time' in eq_ts:
            equil_times.append(float(eq_ts['time']))
        elif 'equilibrium.time' in ods and idx < len(ods['equilibrium.time']):
            equil_times.append(float(ods['equilibrium.time'][idx]))
        else:
            equil_times.append(float(idx))
    
    equil_times = np.asarray(equil_times)
    equil_idx = np.argmin(np.abs(equil_times - cp_time))
    equil_time = float(equil_times[equil_idx])
    
    # Verify that the closest time is within tolerance.
    if not np.isclose(cp_time, equil_time, rtol=0.0, atol=float(atol)):
        raise ValueError(
            f"Time mismatch: cp_time={cp_time:.6f}s, equil_time={equil_time:.6f}s "
            f"(abs diff={abs(cp_time - equil_time):.3e}s > atol={float(atol):.3e}s). "
            f"(cp_idx={cp_idx}, equil_idx={equil_idx})"
        )
    
    return cp_idx, equil_idx, cp_time

def _invalid_imas_path(error):
    """Return the invalid ODS path reported by OMAS, if available."""
    error_msg = str(error)
    if not re.search(
        r"(?:invalid IMAS|not a valid IMAS|does not satisfy IMAS)",
        error_msg,
        flags=re.IGNORECASE,
    ):
        return None

    match = re.search(
        r"location:\s*['\"]?([A-Za-z0-9_.:\[\]-]+)",
        error_msg,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def combine_ods(ods_list):
    """
    Merge multiple ODS objects while automatically handling invalid IMAS structures.

    Parameters
    ----------
    ods_list : list of ODS
        List of ODS objects to merge

    Returns
    -------
    ODS
        Merged ODS object
    """
    ods_list = list(ods_list)
    imas_version = ods_list[0].imas_version if ods_list else ODS().imas_version
    combined_ods = ODS(imas_version=imas_version)

    for index, ods in enumerate(ods_list):
        sanitized_ods = ods.copy()
        removed_paths = set()

        while True:
            # Merge into a trial copy so a failed update cannot leave a partial
            # version of this ODS in the result.
            trial_ods = combined_ods.copy()
            try:
                trial_ods.update(sanitized_ods)
            except Exception as error:
                invalid_path = _invalid_imas_path(error)
                if invalid_path is None or invalid_path in removed_paths:
                    raise

                try:
                    del sanitized_ods[invalid_path]
                except Exception:
                    raise error

                removed_paths.add(invalid_path)
                first_error_line = str(error).splitlines()[0]
                warnings.warn(
                    f"Skipping invalid IMAS location {invalid_path!r} from "
                    f"ODS #{index + 1}: {first_error_line}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            combined_ods = trial_ods
            break

    return combined_ods
