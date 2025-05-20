import vaft
from omas import *
import numpy as np

def _find_signal_onset(ods, time_key, data_key):
    """Helper to find signal onset using vaft.process.signal_onoffset."""
    time = ods.time(time_key)
    data = ods[data_key]
    onset, _ = vaft.process.signal_onoffset(time, data)
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
        onset, _ = vaft.process.signal_onoffset(time, current)
        onsets.append(onset)
    return onsets

def find_pulse_duration(ods):
    """Find the duration of the pulse using spectrometer_uv signal."""
    time = ods.time('spectrometer_uv')
    data = ods['spectrometer_uv.channel.0.processed_line.0.intensity.data']
    onset, offset = vaft.process.signal_onoffset(time, data)
    return offset - onset

def find_max_ip(ods):
    """Find the maximum plasma current."""
    current = ods['magnetics.ip.0.data']
    return np.max(current)

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
    Check ODS/ODC input and always return in ODC format.
    (ODC → as is, ODS → wrapped in ODC at index 0)
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
# Batch time field shift
# ----------------------------------------------------------------------
def shift_time(one_ods, time_shift):
    """
    Shift all 'time / onset / offset' type fields in a single ODS by time_shift.
    """
    for path in one_ods.paths():                    # ('equilibrium', 'time_slice', 0, 'time') ...
        path_str = '.'.join(map(str, path))
        if any(tag in path_str for tag in ('time', 'onset', 'offset')):
            try:
                val = one_ods[path_str]
                if isinstance(val, (np.ndarray, float, int)):
                    one_ods[path_str] = val + time_shift
            except (TypeError, ValueError):
                # Skip if not numeric
                pass

# ----------------------------------------------------------------------
# Time convention conversion (ODS × N → applied to entire ODC)
# ----------------------------------------------------------------------
def change_time_convention(odc_or_ods, convention='vloop'):
    """
    Convert time convention of ODS or ODC to specified convention('daq'|'vloop'|'ip'|'breakdown').
    For multiple shots (ODC), convert each shot independently.
    """
    odc = odc_or_ods_check(odc_or_ods)       # ensure ODC format

    for shot_key, ods in odc.items():        # iterate through ODS in ODC
        params = ods.setdefault('summary.code.parameters', CodeParameters())

        # ---------------- Initial calculation ----------------
        if not hasattr(params, 'vloop_onset'):
            params.time_convention = 'daq'
            params.vloop_onset      = find_vloop_onset(ods)
            params.ip_onset         = find_ip_onset(ods)
            params.breakdown_onset  = find_breakdown_onset(ods)

        original = getattr(params, 'time_convention', 'daq')
        if original == convention:
            # Skip if already using same convention
            continue

        # ---------------- Calculate shift amount ----------------
        onsets = {
            'daq':        0,
            'vloop':      params.vloop_onset,
            'ip':         params.ip_onset,
            'breakdown':  params.breakdown_onset,
        }
        if original not in onsets or convention not in onsets:
            raise ValueError(f"[{shot_key}] Unknown convention: {original} → {convention}")

        time_shift = onsets[original] - onsets[convention]
        print(f"[{shot_key}] shift {time_shift:+.6g} s  ({original} → {convention})")

        # ---------------- Apply actual shift ----------------
        shift_time(ods, time_shift)
        params.time_convention = convention

    return odc        # return for potential use

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
