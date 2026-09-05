import logging
import re
import warnings

import numpy as np
from omas import *

import vaft

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

def signal_time(ods, data_path):
    """Return the time axis that actually belongs to ``data_path``.

    IMAS lets an IDS store time either way, flagged by
    ``ids_properties.homogeneous_time``: homogeneous (1) means every dynamic
    node shares the IDS-level ``<ids>.time``, heterogeneous (0) means each node
    carries its own ``time`` sibling.  The node's own axis wins when it is
    populated and the IDS-level one is the fallback -- the one rule
    :func:`vaft.validation.imas.resolve_signal_time` applies everywhere; this
    wrapper only turns its ``None`` into a :class:`KeyError` that names both
    paths it tried.
    """
    from vaft.validation.imas import resolve_signal_time

    base = data_path[:-5] if data_path.endswith(".data") else data_path
    time = resolve_signal_time(ods, base)
    if time is not None:
        return time
    node_time = f"{base}.time"
    ids_time = f"{data_path.split('.', 1)[0]}.time"
    raise KeyError(
        f"no time axis for {data_path!r}: tried {node_time!r} and {ids_time!r}. "
        "A heterogeneous IDS must give the node its own time sibling; a "
        "homogeneous one must populate the IDS-level time."
    )


# The onset finders answer on the DAQ clock: the shared plasma-analysis span
# they search inside is configured in DAQ seconds, so they are meaningful on a
# product in the 'daq' time convention only. change_time_convention derives
# them once and keeps the result in summary.code.parameters for that reason.
# Imports stay function-local: vaft.omas star-imports this module, and a
# module-level `plasma_timing` name would shadow the submodule.

def _plasma_timing_or_raise(ods):
    """The shared :class:`~vaft.omas.plasma_timing.PlasmaTiming`, or ``ValueError`` when no plasma was found."""
    from .plasma_timing import plasma_timing

    timing = plasma_timing(ods)
    if not timing.found:
        raise ValueError(f"no plasma timing: {timing.fallback_reason}")
    return timing


def find_breakdown_onset(ods):
    """The plasma onset from the shared timing (``vaft.omas.plasma_timing``).

    H-alpha by label is authoritative, the plasma-current principal pulse is
    the fallback; the name keeps the historical convention key.  Raises
    ``ValueError`` naming the reason when neither source shows a plasma, and
    :class:`~vaft.omas.plasma_timing.PlasmaTimingError` when the product has
    no plasma current at all.
    """
    return float(_plasma_timing_or_raise(ods).onset)


def find_pulse_duration(ods):
    """``offset - onset`` of the shared plasma window."""
    timing = _plasma_timing_or_raise(ods)
    return float(timing.offset - timing.onset)


def find_ip_onset(ods):
    """The start of the plasma-current principal pulse, from the shared timing."""
    from .plasma_timing import plasma_timing

    timing = plasma_timing(ods)
    if timing.ip is None or not timing.ip.found:
        reason = ", ".join(timing.ip.onset.flags) if timing.ip is not None else "ip_unusable"
        raise ValueError(f"no plasma-current pulse: {reason}")
    return float(timing.ip.start)


def find_vloop_onset(ods):
    """The loop-voltage zero crossing after the solenoid-driven excursion.

    :func:`vaft.omas.discharge_timing.discharge_timing` on the
    inboard-midplane flux loop; raises ``ValueError`` naming the flags when
    the event was not found.
    """
    from .discharge_timing import discharge_timing

    event = discharge_timing(ods).vloop
    if not event.found:
        raise ValueError(f"no loop-voltage zero crossing: {', '.join(event.flags) or 'not found'}")
    return float(event.zero_crossing)


def find_pf_active_onset(ods):
    """The onset of every ``pf_active`` coil current, in coil order.

    One entry per coil; ``nan`` for a coil that did not fire (an idle coil
    is flat).  :func:`vaft.omas.discharge_timing.discharge_timing` carries
    the evidence when the provenance matters.
    """
    from .discharge_timing import discharge_timing

    return [float(c.time) if c.found else float("nan") for c in discharge_timing(ods).pf_onsets]


def find_bt(ods):
    """The mean toroidal field at ``tf.r0`` over the shared plasma window."""
    timing = _plasma_timing_or_raise(ods)
    time = np.asarray(signal_time(ods, 'tf.b_field_tor_vacuum_r.data'), dtype=float)
    bt = np.asarray(ods['tf.b_field_tor_vacuum_r.data'], dtype=float) / ods['tf.r0']
    inside = (time >= timing.onset) & (time <= timing.offset)
    if not np.any(inside):
        raise ValueError("no toroidal-field samples fall inside the plasma window")
    return float(np.mean(bt[inside]))

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

#: Written to ``summary.code.parameters.onset_method`` by the current derivation.
ONSET_METHOD = "plasma_timing/discharge_timing"
ONSET_METHOD_LEGACY = "legacy"
_ORIGIN_KEYS = ("vloop_onset", "ip_onset", "breakdown_onset")
_logger = logging.getLogger(__name__)


def _derive_onsets(ods):
    """The three convention origins with their sources and flags, from the shared timings."""
    from .discharge_timing import discharge_timing
    from .plasma_timing import SOURCE_IP, plasma_timing

    values, sources, flags = {}, {}, []
    timing = plasma_timing(ods)
    if timing.found:
        values["breakdown_onset"] = float(timing.onset)
        sources["breakdown_onset_source"] = str(timing.source)
    else:
        flags.append(f"breakdown_onset:not_found:{timing.fallback_reason}")
    if timing.ip is not None and timing.ip.found:
        values["ip_onset"] = float(timing.ip.start)
        sources["ip_onset_source"] = SOURCE_IP
    else:
        reason = ", ".join(timing.ip.onset.flags) if timing.ip is not None else "ip_unusable"
        flags.append(f"ip_onset:not_found:{reason}")
    event = discharge_timing(ods).vloop
    if event.found:
        values["vloop_onset"] = float(event.zero_crossing)
        sources["vloop_onset_source"] = (
            f"{event.base} {event.voltage_source} zero crossing after the ohmic excursion"
        )
    else:
        flags.append(f"vloop_onset:not_found:{', '.join(event.flags) or 'not found'}")
    flags.extend(f"vloop:{flag}" for flag in event.flags if flag == "approached_without_crossing")
    return values, sources, flags


def _record_onsets(params, ods, extra_flags=()):
    for key in _ORIGIN_KEYS:
        if key in params:
            del params[key]
    values, sources, flags = _derive_onsets(ods)
    for key, value in values.items():
        params[key] = value
    for key, value in sources.items():
        params[key] = value
    params["onset_method"] = ONSET_METHOD
    params["onset_flags"] = ";".join([*flags, *extra_flags])


def _prepare_onset_memo(params, ods, shot_key):
    """Bring ``summary.code.parameters`` to a state the conversion can read.

    Four states: no memo (derive); a memo written by the current method
    (never recompute); a legacy memo on an unshifted product (re-derive, the
    stored origins were the argmax-of-flux and 5 % rules); a legacy memo on
    a product already shifted with those origins (keep them -- they are what
    makes the shift reversible -- and say so).
    """
    has_legacy = any(key in params for key in _ORIGIN_KEYS) and "onset_method" not in params
    if "onset_method" in params:
        return
    if not has_legacy:
        params["time_convention"] = "daq"
        _record_onsets(params, ods)
        return
    if params.get("time_convention", "daq") == "daq":
        _record_onsets(params, ods, extra_flags=("legacy_memo_rederived",))
        _logger.info("[%s] legacy onset memo re-derived with %s", shot_key, ONSET_METHOD)
        return
    params["onset_method"] = ONSET_METHOD_LEGACY
    params["onset_flags"] = ";".join(
        [flag for flag in str(params.get("onset_flags", "")).split(";") if flag]
        + ["legacy_origins_retained"]
    )
    _logger.warning(
        "[%s] product already shifted to %r with legacy onset origins; keeping them "
        "(flag legacy_origins_retained)",
        shot_key, params.get("time_convention"),
    )


def change_time_convention(odc_or_ods, convention='vloop'):
    """Shift every time-like leaf so that ``convention``'s origin is zero.

    Conventions: ``'daq'`` (the acquisition clock), ``'vloop'`` (the
    loop-voltage zero crossing after the solenoid excursion,
    :func:`find_vloop_onset`), ``'ip'`` (the plasma-current pulse start,
    :func:`find_ip_onset`) and ``'breakdown'`` (the plasma onset,
    :func:`find_breakdown_onset`).  The origins are derived once, on the
    product's DAQ clock, and kept in ``summary.code.parameters`` with their
    sources (``*_onset_source``), the method (``onset_method``) and the flags
    (``onset_flags``); a later call never recomputes them, which is what
    keeps a shift reversible.  An origin that was not found is absent, and
    asking for its convention raises ``ValueError`` with the reason.  Works
    on an ODS or an ODC and returns the ODC (the ODS is shifted in place).
    """
    odc = odc_or_ods_check(odc_or_ods)

    for shot_key, ods in odc.items():
        params = ods.setdefault('summary.code.parameters', CodeParameters())
        _prepare_onset_memo(params, ods, shot_key)
        original = params.get('time_convention', 'daq')
        if original == convention:
            continue

        onsets = {'daq': 0.0}
        for key in _ORIGIN_KEYS:
            if key in params:
                onsets[key[: -len("_onset")]] = float(params[key])
        known = ('daq', 'vloop', 'ip', 'breakdown')
        if original not in known or convention not in known:
            raise ValueError(f"[{shot_key}] Unknown convention: {original} -> {convention}")
        for name in (original, convention):
            if name not in onsets:
                reason = next(
                    (flag for flag in str(params.get("onset_flags", "")).split(";")
                     if flag.startswith(f"{name}_onset:")),
                    "origin not recorded",
                )
                raise ValueError(f"[{shot_key}] no {name!r} origin: {reason}")

        time_shift = onsets[original] - onsets[convention]
        _logger.info("[%s] shift %+.6g s  (%s -> %s)", shot_key, time_shift, original, convention)

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
        if not vaft.process.is_signal_active(data_pres, var_ratio_thresh=pressure_threshold):
            return 'Vacuum'
        data_alpha = ods['spectrometer_uv.channel.0.processed_line.0.intensity.data']
        if not vaft.process.is_signal_active(data_alpha, var_ratio_thresh=halpha_threshold):
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


#: Locations under this suffix are exempt from IMAS structure validation
#: (OMAS treats ``*.code.parameters.*`` as free-form user metadata); see
#: ``omas_core.ODS.__setitem__``. The fast pre-pass below must honor the same
#: exemption or it would prune legitimate metadata the real merge accepts.
_CODE_PARAMETERS_INFIX = ".code.parameters."


def _prune_invalid_imas_paths(ods, imas_version):
    """Remove every schema-invalid leaf from ``ods`` in a single pass.

    ``combine_ods`` discovers invalid locations by attempting a real merge and
    parsing the exception OMAS raises -- correct, but each discovery costs a
    full ``combined_ods.copy()`` and a full ``trial_ods.update(sanitized_ods)``
    retried from scratch. Once ``combined_ods`` has grown large across a
    backfill, paying that per invalid leaf turns one contaminated input with
    N invalid locations into N+1 full-tree merges.

    A location's validity does not depend on what is already in
    ``combined_ods`` (OMAS's structure lookup is a pure function of the
    location string and the IMAS version), so it can be checked directly with
    the same cached lookup OMAS itself uses internally, without touching
    ``combined_ods`` at all. This turns discovery into one walk over
    ``ods.paths()`` plus O(n) cached, no-copy schema lookups.

    Returns the set of removed locations (as dotted strings), so the caller
    can warn about them and the exception-driven retry loop in
    :func:`combine_ods` never rediscovers them.
    """
    try:
        from omas.omas_utils import imas_structure, l2o
    except ImportError:
        # Defensive: if a future OMAS release moves this private helper,
        # skip the fast path. combine_ods still works via the exception-driven
        # retry loop below, just without this speedup.
        return set()

    removed = set()
    for path in ods.paths():
        location = l2o(path)
        if _CODE_PARAMETERS_INFIX in location:
            continue
        try:
            imas_structure(imas_version, location)
            continue
        except (LookupError, TypeError):
            pass

        if location in removed:
            continue
        try:
            del ods[location]
        except Exception:
            # Leave it for the exception-driven retry loop to sort out.
            continue
        removed.add(location)

    return removed


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

        # Fast pre-pass: find and remove every schema-invalid leaf up front, so
        # the retry loop below normally runs once regardless of how many
        # invalid locations this input has. See _prune_invalid_imas_paths.
        removed_paths = _prune_invalid_imas_paths(sanitized_ods, imas_version)
        for location in sorted(removed_paths):
            warnings.warn(
                f"Skipping invalid IMAS location {location!r} from "
                f"ODS #{index + 1}: Not a valid IMAS {imas_version} location: "
                f"{location}",
                RuntimeWarning,
                stacklevel=2,
            )

        while True:
            # Merge into a trial copy so a failed update cannot leave a partial
            # version of this ODS in the result. The pre-pass above means this
            # normally succeeds on the first attempt; it remains as a
            # defensive fallback for any invalid location the static
            # structure lookup does not catch.
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


#: Where the convention is recorded on an ODS.
#:
#: IMAS DD 4 adds ``ids_properties.cocos``, but DD 3.x -- which OMAS defaults to
#: and VAFT targets -- has no such field, which is why nothing has ever written
#: the path VAFT reads.  The convention therefore lives on
#: ``equilibrium.code.parameters``, the same place VAFT already keeps CHEASE's
#: comparison metrics and EFIT's auxiliary quantities, and the standard field is
#: written too whenever the data dictionary in use accepts it.
COCOS_PARAMETER_PATH = "equilibrium.code.parameters.cocos"


def ods_cocos(ods, *, default=None):
    """The COCOS index an ODS declares, or ``default`` when it declares none.

    Reads the DD 4 ``ids_properties.cocos`` field first, then VAFT's
    ``equilibrium.code.parameters.cocos`` block.  An ODS written before VAFT
    labelled its output declares nothing, so the caller keeps whatever legacy
    assumption it had rather than guessing.
    """
    from vaft.data.cocos import COCOS_INDICES

    for path in ("equilibrium.ids_properties.cocos", COCOS_PARAMETER_PATH):
        try:
            value = ods[path]
        except Exception:
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index in COCOS_INDICES:
            return index
    return default


def set_ods_cocos(ods, index, *, source=None):
    """Record ``index`` on ``ods`` so a consumer need not guess the convention.

    Every VAFT path that produces an equilibrium ODS should call this.  Writes
    ``equilibrium.code.parameters.cocos`` always, and the standard
    ``ids_properties.cocos`` as well when the data dictionary defines it, so an
    ODS is correctly labelled under both DD 3 and DD 4.
    """
    from vaft.data.cocos import COCOS_INDICES

    index = int(index)
    if index not in COCOS_INDICES:
        raise ValueError(
            f"COCOS index {index!r} is not defined; expected one of {COCOS_INDICES}"
        )
    ods[COCOS_PARAMETER_PATH] = index
    if source is not None:
        ods["equilibrium.code.parameters.cocos_source"] = str(source)
    try:  # DD 4 only; harmless where the field does not exist.
        ods["equilibrium.ids_properties.cocos"] = index
    except Exception:
        pass
    return index
