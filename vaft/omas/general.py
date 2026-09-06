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

def find_shotclass(ods, plot_opt=0):
    """The lenient form of :func:`classify_shot`: ``None`` when the ODS cannot be classified.

    A product without the plasma current the shared timing needs, or one
    that raises on read, answers ``None`` here rather than raising; the
    classification itself is :func:`classify_shot`, and the two never
    disagree on a classifiable shot.
    """
    try:
        return classify_shot(ods)
    except (KeyError, ValueError):   # PlasmaTimingError is a ValueError
        return None

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


# The onset finders answer on the product's own clock: the shared
# plasma-analysis span is configured in DAQ seconds and the timing modules
# displace it by the origin change_time_convention recorded, so a product
# re-referenced to an event is searched where its plasma is. The origins
# themselves are derived once, on the DAQ clock, and kept in
# summary.code.parameters. Imports stay function-local: vaft.omas star-imports
# this module, and a module-level `plasma_timing` name would shadow the submodule.

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

    The event of :mod:`vaft.omas.discharge_timing` on the inboard-midplane
    flux loop, anchored on the ohmic coil alone (the other coils are not
    detected); raises ``ValueError`` naming the flags when it was not found.
    """
    from .discharge_timing import loop_voltage_event, oh_coil_onset

    event = loop_voltage_event(ods, anchor=oh_coil_onset(ods))
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
    """The representative peak of the plasma current inside the plasma window.

    ``vaft.omas.plasma_features``: the largest sustained excursion of the
    median-filtered, zero-phase low-passed current between the plasma onset
    and offset -- a coil-firing impulse is never the answer.  Raises
    ``ValueError`` naming the reason when the timing found no plasma or the
    current carries no qualifying peak.
    """
    from .plasma_features import ip_peak

    feature = ip_peak(ods)
    if not feature.found:
        raise ValueError(f"no plasma-current peak: {feature.reason or ', '.join(feature.flags)}")
    return float(feature.value)


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
    """The three convention origins with their sources, reasons and flags, from the shared timings."""
    from .discharge_timing import discharge_timing
    from .plasma_timing import SOURCE_IP, plasma_timing

    values, notes, flags = {}, {}, []
    timing = plasma_timing(ods)
    if timing.found:
        values["breakdown_onset"] = float(timing.onset)
        notes["breakdown_onset_source"] = str(timing.source)
    else:
        notes["breakdown_onset_reason"] = str(timing.fallback_reason)
    if timing.ip is not None and timing.ip.found:
        values["ip_onset"] = float(timing.ip.start)
        notes["ip_onset_source"] = SOURCE_IP
    else:
        notes["ip_onset_reason"] = (
            ", ".join(timing.ip.onset.flags) if timing.ip is not None else "ip_unusable"
        )
    event = discharge_timing(ods).vloop
    if event.found:
        values["vloop_onset"] = float(event.zero_crossing)
        notes["vloop_onset_source"] = (
            f"{event.base} {event.voltage_source} zero crossing after the ohmic excursion"
        )
    else:
        notes["vloop_onset_reason"] = ", ".join(event.flags) or "not found"
    if "approached_without_crossing" in event.flags:
        flags.append("vloop:approached_without_crossing")
    return values, notes, flags


def _record_onsets(params, ods, shot_key, extra_flags=()):
    """Derive the origins and, only when at least one exists, write the memo.

    Nothing is written before the derivation succeeds: a product that yields
    no origin at all (typically one whose axes are not on the DAQ clock but
    whose memo was lost) must not be stamped ``daq`` with frozen not-found
    origins, which no later call could repair.
    """
    values, notes, flags = _derive_onsets(ods)
    if not values:
        reasons = "; ".join(f"{key[:-len('_reason')]}: {value}" for key, value in notes.items())
        raise ValueError(
            f"[{shot_key}] no time-convention origin could be derived on the DAQ clock "
            f"({reasons}); the memo is left untouched"
        )
    for key in _ORIGIN_KEYS:
        params.pop(key, None)
        params.pop(f"{key}_source", None)
        params.pop(f"{key}_reason", None)
    params.update(values)
    params.update(notes)
    params["time_convention"] = "daq"
    params["onset_method"] = ONSET_METHOD
    params["onset_flags"] = ";".join([*flags, *extra_flags])


def _prepare_onset_memo(params, ods, shot_key):
    """Bring ``summary.code.parameters`` to a state the conversion can read.

    Four states: a memo written by the current method (never recompute); no
    memo (derive); a legacy memo on an unshifted product (re-derive, the
    stored origins were the argmax-of-flux and 5 % rules); a legacy memo on
    a product already shifted with those origins (keep them -- they are what
    makes the shift reversible -- and say so).
    """
    if "onset_method" in params:
        return
    has_legacy = any(key in params for key in _ORIGIN_KEYS)
    if not has_legacy:
        _record_onsets(params, ods, shot_key)
        return
    if params.get("time_convention", "daq") == "daq":
        _record_onsets(params, ods, shot_key, extra_flags=("legacy_memo_rederived",))
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
    sources (``*_onset_source``), the reason an origin is missing
    (``*_onset_reason``), the method (``onset_method``) and the flags
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
                reason = params.get(f"{name}_onset_reason", "origin not recorded")
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

def classify_shot(ods, pressure_threshold=0.01, halpha_threshold=None):
    """The class of a shot -- ``'Plasma'``, ``'BD failure'`` or ``'Vacuum'`` -- as a string.

    :func:`vaft.omas.shot_class.shot_class` decides it from the shared plasma
    timing and the barometry pressure response, and carries which check
    decided; this is that record's label.  ``halpha_threshold`` is no longer
    used: the light is judged by the plasma timing, not by a variance ratio.
    """
    from .shot_class import shot_class

    if halpha_threshold is not None:
        warnings.warn(
            "classify_shot: halpha_threshold is ignored; the light is judged by vaft.omas.plasma_timing",
            DeprecationWarning, stacklevel=2,
        )
    return shot_class(ods, pressure_threshold=pressure_threshold).label

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

    from vaft.ods_access import path_value

    for path in ("equilibrium.ids_properties.cocos", COCOS_PARAMETER_PATH):
        # A non-mutating read: a plain ``ods[path]`` of a missing path creates
        # an empty node that ``flat()`` hides but ``save`` writes (issue #478).
        value = path_value(ods, path)
        if value is None:
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
    if _dd_defines(ods, "equilibrium.ids_properties.cocos"):  # DD 4 only
        ods["equilibrium.ids_properties.cocos"] = index
    return index


def _dd_defines(ods, path):
    """Whether the data dictionary version ``ods`` targets defines ``path``.

    An ODS with consistency checks off accepts any path, so the DD itself has
    to be asked; otherwise a DD 3 artifact ends up carrying a DD 4 leaf that
    its native serializers cannot write (issue #478).
    """
    from omas.omas_utils import omas_info_node

    try:
        info = omas_info_node(path, imas_version=getattr(ods, "imas_version", None))
    except Exception:
        return False
    return bool(info) and "data_type" in info


#: Equilibrium leaves that scale with psi (times 2*pi from Wb/rad to Wb) and
#: the psi-derivative leaves that scale inversely.  Only leaves the slice holds
#: are touched.  The list covers what VAFT's own writers (eqdsk, vfit, the
#: update helpers) produce; DD leaves no VAFT path writes
#: (``q_min.psi``, ``psi_external_average``, ``constraints.*.position.psi``)
#: are not on it, so point this at externally produced ODSs with care.
_PSI_LIKE_LEAVES = (
    "global_quantities.psi_axis",
    "global_quantities.psi_boundary",
    "boundary.psi",
    "boundary_separatrix.psi",
    "profiles_1d.psi",
    "profiles_1d.dpsi_drho_tor",
)
_PER_PSI_LEAVES = (
    "profiles_1d.dpressure_dpsi",
    "profiles_1d.f_df_dpsi",
    "profiles_1d.dvolume_dpsi",
    "profiles_1d.darea_dpsi",
)


def equilibrium_psi_to_weber(ods, *, source=None):
    """Rewrite a legacy Wb/rad equilibrium in place to the DD's full weber.

    The IMAS Data Dictionary stores ``equilibrium.*.psi`` in Wb (COCOS 11);
    artifacts written before issue #236 hold the g-file's Wb/rad verbatim and
    declare nothing.  This multiplies every psi-like leaf by 2*pi, divides the
    psi-derivative profiles by it, does the same to ``core_profiles`` psi grids,
    and then declares COCOS 11 through :func:`set_ods_cocos` so no reader has
    to guess again.

    Returns ``True`` when a conversion was applied and ``False`` when the ODS
    already declares a weber convention or holds no equilibrium.  A declared
    COCOS 1-8 is trusted as provenance and converted without probing the
    data.  An undeclared ODS is probed slice by slice, and ``ValueError`` is
    raised when the slices disagree about their storage family or when no
    slice can settle it -- guessing would silently corrupt the flux by 2*pi,
    which is what this function exists to end (issue #478).
    """
    import numpy as np

    from vaft.data.cocos import VAFT_INTERNAL_COCOS
    from vaft.data.eqdsk import TWO_PI, slice_flux_exponent

    if "equilibrium" not in ods or "equilibrium.time_slice" not in ods:
        return False
    declared = ods_cocos(ods)
    if declared is not None:
        if declared >= 11:
            return False
        exponents = {0}
    else:
        exponents = set()
        undecided = []
        for index in range(len(ods["equilibrium.time_slice"])):
            exponent = slice_flux_exponent(ods[f"equilibrium.time_slice.{index}"])
            if exponent is None:
                undecided.append(index)
            else:
                exponents.add(exponent)
        if not exponents:
            raise ValueError(
                "No equilibrium time slice carries enough data (profiles_1d.phi, "
                "or a boundary outline with psi and ip) to settle whether psi is "
                "stored in Wb or Wb/rad; declare the COCOS index instead"
            )
        if len(exponents) > 1:
            raise ValueError(
                "Equilibrium time slices disagree about their psi convention; "
                "refusing to convert an inconsistent ODS"
            )
        if exponents == {1}:
            set_ods_cocos(ods, VAFT_INTERNAL_COCOS, source=source)
            return False

    for index in range(len(ods["equilibrium.time_slice"])):
        ts = ods[f"equilibrium.time_slice.{index}"]
        for leaf in _PSI_LIKE_LEAVES:
            if leaf in ts:
                scaled = np.asarray(ts[leaf], float) * TWO_PI
                ts[leaf] = float(scaled) if scaled.ndim == 0 else scaled
        for leaf in _PER_PSI_LEAVES:
            if leaf in ts:
                ts[leaf] = np.asarray(ts[leaf], float) / TWO_PI
        if "profiles_2d" in ts:
            for grid_index in range(len(ts["profiles_2d"])):
                if f"profiles_2d.{grid_index}.psi" in ts:
                    ts[f"profiles_2d.{grid_index}.psi"] = (
                        np.asarray(ts[f"profiles_2d.{grid_index}.psi"], float) * TWO_PI
                    )
    if "core_profiles.profiles_1d" in ods:
        for index in range(len(ods["core_profiles.profiles_1d"])):
            leaf = f"core_profiles.profiles_1d.{index}.grid.psi"
            if leaf in ods:
                ods[leaf] = np.asarray(ods[leaf], float) * TWO_PI
    set_ods_cocos(ods, VAFT_INTERNAL_COCOS, source=source)
    return True
