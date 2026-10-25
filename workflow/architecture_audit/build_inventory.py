#!/usr/bin/env python3
"""Regenerate the Phase A preprocessing and magnetics inventories (issue #2148).

Read-only with respect to the package: it parses source files with :mod:`ast`
and reads the three retired magnetics YAML assets (kept as the test fixture
``test/data/legacy_magnetics`` since #2158) and ``vest.yaml``; it never imports
``vaft``.

Two products, both deterministic (sorted keys, sorted rows, repository-relative
paths, no timestamps):

``docs/architecture/consolidation/inventory.json``
    One row per *implementation* -- a function, method or module body in a
    git-tracked file under ``vaft/`` or ``workflow/`` -- that calls a filtering, smoothing, baseline,
    resampling, integration, differentiation, fitting or calibration primitive
    (SciPy/NumPy or one of VAFT's own preprocessing helpers), or that reads one
    of the three external magnetics YAML assets. The mechanically derived
    fields (call sites, static callers, static production reachability) are
    merged with the hand-curated classification table :data:`CURATED` below,
    keyed by ``(source_file, implementation)``. Rows without a curated entry
    carry ``classification = "unverified"`` and say so in ``curated = false``.

``docs/architecture/consolidation/magnetics_inventory.json``
    The 75 equilibrium magnetic sensors joined from ``MD.yaml``,
    ``VEST_MagneticsGeometry_Full_ver_2302.yaml`` and ``table.yaml`` with the
    ``vest.yaml`` wiring revisions and known faults, plus the proposed stable
    identifiers of the Phase B1 registry.

Static reachability is an approximation and the JSON says so: calls through
``self``, ``getattr``, registries and string dispatch are not followed, and a
class counts as reached when its name is called from a reached function.
``docs/architecture/consolidation/callgraph.md`` records the dispatch that has
to be verified at run time.

Usage::

    python workflow/architecture_audit/build_inventory.py          # write
    python workflow/architecture_audit/build_inventory.py --check  # exit 1 on drift
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path("docs/architecture/consolidation")
INVENTORY_NAME = "inventory.json"
MAGNETICS_NAME = "magnetics_inventory.json"
SCAN_ROOTS = ("vaft", "workflow")
#: The generator itself names every primitive and asset as a string.
EXCLUDED_PREFIXES = ("workflow/architecture_audit/",)
SCHEMA_VERSION = 1

#: The three assets left the package for the vest.yaml sensor registry (#2158);
#: the audit keeps joining them from their test-fixture home.
GEOMETRY_DIR = Path("test/data/legacy_magnetics")
MD_YAML = "MD.yaml"
GEOMETRY_YAML = "VEST_MagneticsGeometry_Full_ver_2302.yaml"
TABLE_YAML = "table.yaml"
VEST_YAML = Path("vaft/machine_mapping/vest.yaml")
MAGNETICS_ASSETS = (MD_YAML, GEOMETRY_YAML, TABLE_YAML)
#: A second field-code <-> label table (all DAQ fields, magnetics included).
SQL_TABLE = "sql_table.txt"
ASSET_LITERALS = (*MAGNETICS_ASSETS, SQL_TABLE)

#: Production entry points: the Snakefiles whose ``python <script>.py`` shell
#: commands start a stage. Pipeline 3 calls its scripts by relative path.
PIPELINES = {
    "pipeline1": "workflow/automatic_pipeline_1_routine_data_processing",
    "pipeline2": "workflow/automatic_pipeline_2_corrective_data_update",
    "pipeline3": "workflow/automatic_pipeline_3_data_summary",
}

# --------------------------------------------------------------------------
# Primitive table: fully qualified callee -> operation category.
# --------------------------------------------------------------------------

_SCIPY_SIGNAL_FILTER_DESIGN = (
    "firwin", "firwin2", "remez", "butter", "cheby1", "cheby2", "ellip", "bessel",
    "iirfilter", "iirnotch", "kaiserord",
)
_SCIPY_SIGNAL_FILTER_APPLY = (
    "lfilter", "filtfilt", "sosfilt", "sosfiltfilt", "lfilter_zi", "fftconvolve",
    "convolve", "oaconvolve", "wiener",
)
PRIMITIVES: dict[str, str] = {}
for _name in _SCIPY_SIGNAL_FILTER_DESIGN:
    PRIMITIVES[f"scipy.signal.{_name}"] = "filter_design"
for _name in _SCIPY_SIGNAL_FILTER_APPLY:
    PRIMITIVES[f"scipy.signal.{_name}"] = "filter_application"
PRIMITIVES.update(
    {
        "scipy.signal.savgol_filter": "smoothing",
        "scipy.signal.medfilt": "smoothing",
        "scipy.signal.decimate": "resampling",
        "scipy.signal.resample": "resampling",
        "scipy.signal.resample_poly": "resampling",
        "scipy.signal.detrend": "baseline",
        "scipy.ndimage.uniform_filter1d": "smoothing",
        "scipy.ndimage.uniform_filter": "smoothing",
        "scipy.ndimage.median_filter": "smoothing",
        "scipy.ndimage.gaussian_filter1d": "smoothing",
        "scipy.ndimage.gaussian_filter": "smoothing",
        "numpy.convolve": "smoothing",
        "numpy.interp": "resampling",
        "scipy.interpolate.interp1d": "resampling",
        "scipy.interpolate.CubicSpline": "resampling",
        "scipy.interpolate.PchipInterpolator": "resampling",
        "scipy.interpolate.Akima1DInterpolator": "resampling",
        "scipy.interpolate.UnivariateSpline": "resampling",
        "scipy.interpolate.make_interp_spline": "resampling",
        "scipy.interpolate.RegularGridInterpolator": "resampling",
        "scipy.interpolate.RectBivariateSpline": "resampling",
        "scipy.interpolate.griddata": "resampling",
        "scipy.integrate.cumulative_trapezoid": "integration",
        "scipy.integrate.cumtrapz": "integration",
        "scipy.integrate.trapezoid": "integration",
        "scipy.integrate.trapz": "integration",
        "scipy.integrate.simpson": "integration",
        "scipy.integrate.simps": "integration",
        "numpy.trapz": "integration",
        "numpy.trapezoid": "integration",
        "numpy.gradient": "differentiation",
        "numpy.polyfit": "polynomial_fit",
        "numpy.linalg.lstsq": "least_squares_fit",
        "scipy.optimize.least_squares": "least_squares_fit",
        "numpy.cumsum": "cumulative_sum",
        # ``<frame>.rolling(...)``: pandas windows; the receiver is not resolved.
        "<attribute>.rolling": "smoothing",
        "numpy.polynomial.polynomial.polyfit": "polynomial_fit",
        "scipy.optimize.curve_fit": "curve_fit",
        # VAFT's own helpers: a call to one of them is a delegation, recorded so
        # the inventory shows who already reuses the canonical candidate.
        "vaft.compat.cumtrapz_compat": "integration",
        "vaft.process.numerical.time_derivative": "differentiation",
        "vaft.process.signal_processing.smooth": "smoothing",
        "vaft.process.signal_processing.fir_filter": "filter_application",
        "vaft.process.signal_processing.fir_filter_coefficients": "filter_design",
        "vaft.process.signal_processing.apply_fir_filter": "filter_application",
        "vaft.process.signal_processing.butterworth_lowpass": "filter_application",
        "vaft.process.signal_processing.butterworth_bandpass": "filter_application",
        "vaft.process.signal_processing.butterworth_highpass": "filter_application",
        "vaft.process.signal_processing.pf_current_filter": "filter_application",
        "vaft.process.signal_processing.anti_alias_filter": "filter_application",
        "vaft.process.signal_processing.resample_to_time": "resampling",
        "vaft.process.signal_processing.detrend_moving_average": "baseline",
        "vaft.process.signal_processing.subtract_baseline": "baseline",
        "vaft.process.signal_processing._polyfit_linear_baseline": "baseline",
        "vaft.process.signal_processing.define_baseline": "baseline",
        "vaft.process.signal_processing.repair_clipped_interval": "signal_repair",
        "vaft.process.signal_processing.vest_coil_current_noise_reduction": "smoothing",
        "vaft.process.signal_processing.process_signal": "filter_application",
        "vaft.process.magnetics._linear_baseline": "baseline",
        "vaft.process.magnetics._calibrated_signal": "calibration",
        "vaft.machine_mapping.magnetics._linear_baseline_subtract": "baseline",
        "vaft.machine_mapping.utils.calibrate_vest_signal": "calibration",
        "vaft.code.efit.legacy.smooth": "smoothing",
        "vaft.process.soft_x_rays.sxr_baseline_correction": "baseline",
    }
)

#: Calls that make a function's configuration come from ``vest.yaml``.
CONFIG_RESOLVERS = {
    "resolve_vest_diagnostic",
    "resolve_shot_revisions",
    "resolve_shot_revisions_with_provenance",
    "resolve_diagnostics_time_policies",
    "resolve_plasma_timing_policy",
    "resolve_discharge_timing_policy",
    "resolve_impa_config",
    "equilibrium_magnetics_processing_config",
    "magnetics_wiring_for_shot",
    "known_magnetics_faults",
    "_equilibrium_magnetics_revision",
    "vest_core_profiles_policy",
}

#: Loader APIs over the external magnetics YAMLs; calling one makes the caller
#: a consumer for the migration map. Since #2158 the private loaders are gone
#: and these public APIs read the vest.yaml registry instead; the asset each
#: one replaced is kept so the map still says which YAML a consumer used to read.
MAGNETICS_LOADERS = {
    "vaft.machine_mapping.magnetics.vest_equilibrium_magnetics_channel_definitions": (MD_YAML,),
    "vaft.machine_mapping.magnetics.vest_md_channel_definitions": (MD_YAML,),
    "vaft.machine_mapping.magnetics.magnetics_wiring_for_shot": (MD_YAML,),
    "vaft.machine_mapping.magnetics._equilibrium_probe_field_codes": (MD_YAML, GEOMETRY_YAML),
    "vaft.machine_mapping.magnetics.equilibrium_probe_count": (MD_YAML,),
    "vaft.machine_mapping.magnetics._pinned_probe_index": (GEOMETRY_YAML,),
    "vaft.machine_mapping.magnetics._pinned_channel_index": (GEOMETRY_YAML,),
    "vaft.machine_mapping.utils._load_magnetics_channel_groups": (GEOMETRY_YAML,),
    "vaft.machine_mapping.magnetics.known_magnetics_faults": (GEOMETRY_YAML,),
    "vaft.machine_mapping.magnetics._equilibrium_magnetics_revision": (GEOMETRY_YAML,),
}

#: VAFT helpers that implement a preprocessing operation without calling a
#: listed primitive themselves (pure-NumPy loops, means); each gets a row at
#: its definition. Every ``vaft.*`` key of :data:`PRIMITIVES` is added too.
DEFINITION_OPERATIONS = {
    "vaft.plot.soft_x_rays._baseline_correct": "baseline",
    "vaft.machine_mapping.pf_active._baseline_mean": "baseline",
    "vaft.machine_mapping.magnetics._baseline_correct_limiter_voltage": "baseline",
    "vaft.code.efit.legacy.box_average": "smoothing",
    "vaft.code.efit.legacy.box_average_series": "smoothing",
    "vaft.machine_mapping.soft_x_rays.vfit_soft_x_rays_dynamic": "baseline",
}

_SNIPPET_LIMIT = 160

# --------------------------------------------------------------------------
# Hand-curated classification table (issue #2148 section 3).
#
# Key: (source_file, implementation). Values override the defaults; every key
# must match a scanned row, which the build checks so a rename cannot leave a
# stale classification behind.
# --------------------------------------------------------------------------

_PR2135 = (
    "Since #2135 (merged, 75b154e31) this is applied through "
    "vaft.process.signal_processing.apply_fir_filter(short_record='causal'): "
    "a refactor, the numerics are unchanged."
)

CURATED: dict[tuple[str, str], dict[str, Any]] = {
    # ---- canonical candidates in vaft.process.signal_processing -------------
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.smooth"): {
        "classification": "canonical",
        "algorithm_semantics": "MATLAB smooth(span): centred moving average, span forced odd, shrinking window at the ends.",
        "scientific_risk": "low",
        "recommended_action": "keep as the single moving-average owner; legacy.smooth delegates (B3).",
        "related_issue": "#2162",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.butterworth_highpass"): {
        "classification": "canonical",
        "algorithm_semantics": "scipy butter(order, highpass) + filtfilt (zero_phase=True) or lfilter; cutoff must lie in (0, Nyquist).",
        "scientific_risk": "low",
        "recommended_action": "keep beside butterworth_lowpass/bandpass.",
        "related_issue": "#2162",
        "resolved_by": "#2162 (B3.6)",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.fir_filter_coefficients"): {
        "classification": "canonical",
        "algorithm_semantics": "firwin window-method design, default window='hann', numtaps=fs*1e-3 forced odd (>=3); raises on edges outside (0, Nyquist).",
        "scientific_risk": "medium",
        "recommended_action": "keep; NOT a drop-in for the VEST Hamming designs (26 taps at 25 kHz there vs 25 here, #2135 row). Expose window/numtaps explicitly when B3 composes the PF filter.",
        "related_issue": "#2162",
        "evidence": "#2135 inventory row: fir_filter_coefficients (hann, forced odd) is not numerically equivalent to the donor vest_filter (hamming, round(fs*1e-3)).",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.fir_filter"): {
        "classification": "canonical",
        "algorithm_semantics": "designs with fir_filter_coefficients; zero_phase=True applies through apply_fir_filter (since #2135), zero_phase=False still calls lfilter directly.",
        "scientific_risk": "low",
        "recommended_action": "keep.",
        "related_issue": "#2135",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.apply_fir_filter"): {
        "classification": "canonical",
        "algorithm_semantics": "zero-phase FIR application (filtfilt); a record shorter than 3*len(taps)+1 raises (short_record='raise', default) or falls back to causal lfilter (short_record='causal').",
        "scientific_risk": "low",
        "recommended_action": "keep as the single FIR application (#2135 row).",
        "related_issue": "#2135",
        "resolved_by": "#2135",
    },
    ("vaft/process/magnetics.py", "vaft.process.magnetics.vest_probe_lowpass_taps"): {
        "classification": "intentional_specialization",
        "algorithm_semantics": "firwin(lowpass_taps=251, lowpass_cutoff=2.5 kHz, fs=250 kHz), Hamming window: the donor vest_filter design; not equal to fir_filter_coefficients (hann).",
        "scientific_risk": "low",
        "recommended_action": "keep (#2135 row: FIR designs per diagnostic are intentional specializations).",
        "related_issue": "#2135",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.butterworth_lowpass"): {
        "classification": "canonical",
        "algorithm_semantics": "Butterworth IIR (default order 2), zero_phase -> filtfilt else lfilter.",
        "scientific_risk": "low",
        "recommended_action": "keep; add butterworth_highpass(zero_phase=True) for the plot recipe (B3).",
        "related_issue": "#2162",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.butterworth_bandpass"): {
        "classification": "canonical",
        "algorithm_semantics": "Butterworth IIR band pass (default order 2), zero_phase -> filtfilt.",
        "scientific_risk": "low",
        "recommended_action": "keep.",
        "related_issue": "#2162",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.anti_alias_filter"): {
        "classification": "canonical",
        "algorithm_semantics": "Kaiser-sized FIR anti-alias before downsampling, zero-phase per finite run, filtfilt length rule _filtfilt_min_length (#893).",
        "scientific_risk": "low",
        "recommended_action": "keep as the resampling anti-alias owner.",
        "related_issue": "#425",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.resample_to_time"): {
        "classification": "canonical",
        "algorithm_semantics": "anti-alias (when downsampling) then linear np.interp per finite run; explicit extrapolation policy (#425/#626); unfiltered='nan' (default, B3.12) makes every target sample drawn from a run too short to filter NaN instead of a possibly aliased value, 'keep' leaves it for a mask-reading caller.",
        "scientific_risk": "low",
        "recommended_action": "keep as the resampling owner; direct np.interp in mappers stays where the source is already band-limited (documented per site).",
        "related_issue": "#425, #2162",
        "resolved_by": "#2162 (B3.12)",
        "evidence": "test/test_resample_short_run_alias.py: 20 kHz tone and 1-100 kHz chirp on a 900-sample run (kept: folded at full amplitude; default: NaN); AST guard that only a mask-reading caller passes 'keep'; 4-shot capture leaf-identical.",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.subtract_baseline"): {
        "classification": "canonical",
        "algorithm_semantics": "fitting_opt='polyfit' (B3.1): np.polyfit deg 1 over in-bounds indices, finite-only by default (nan_policy='omit') or NaN-propagating, <2 valid -> DegenerateBaselineWindowError or zero with warning (#639). Other options: curve_fit of linear/quadratic/exp model or exact spline, no finite-only filtering, no degenerate check; 'linear' is NOT routed to polyfit (agrees only to ~7.5e-13 relative).",
        "scientific_risk": "low",
        "recommended_action": "keep as the single linear-baseline owner.",
        "related_issue": "#2162, #639",
        "resolved_by": "#2162 (B3.1)",
        "evidence": "test/test_baseline_consolidation.py: bit-identical to frozen develop copies of _linear_baseline and _linear_baseline_subtract; 4-shot capture (39915/41524/41672/42962) leaf-identical.",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing._polyfit_linear_baseline"): {
        "classification": "canonical",
        "algorithm_semantics": "the fitting_opt='polyfit' kernel of subtract_baseline: index guard, nan_policy, #639 degenerate error/fallback.",
        "scientific_risk": "low",
        "recommended_action": "keep (private kernel; process.magnetics._linear_baseline and MM._linear_baseline_subtract call it).",
        "related_issue": "#2162, #639",
        "resolved_by": "#2162 (B3.1)",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.detrend_moving_average"): {
        "classification": "canonical",
        "algorithm_semantics": "subtract centred moving average of window_samples.",
        "scientific_risk": "low",
        "recommended_action": "keep.",
        "related_issue": "",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.repair_clipped_interval"): {
        "classification": "canonical",
        "algorithm_semantics": "detect samples at the DAQ clip values and bridge the clipped interval (linear/cubic per config).",
        "scientific_risk": "low",
        "recommended_action": "keep; PF6 and Rogowski/diamagnetic callers already delegate.",
        "related_issue": "",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.vest_coil_current_noise_reduction"): {
        "classification": "legacy",
        "algorithm_semantics": "VFIT coil-current noise reduction (Hamming FIR + filtfilt); VEST_CoilCurrentNoiseReduction is its camel-case alias.",
        "scientific_risk": "low",
        "recommended_action": "E: confirm no production caller, then deprecate in favour of the PF filter composition (B3).",
        "related_issue": "#2168",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.process_signal"): {
        "classification": "legacy",
        "algorithm_semantics": "option-driven generic chain (Butterworth filtfilt / resample_to_time); no production caller.",
        "scientific_risk": "low",
        "recommended_action": "E: retire with machine_mapping.utils.process_signal once no caller remains.",
        "related_issue": "#2168",
    },
    # ---- B-pol probe / magnetics chain --------------------------------------
    ("vaft/process/magnetics.py", "vaft.process.magnetics.b_field_pol_probe_field"): {
        "classification": "canonical",
        "production_reachability_note": "pipeline1 generate_diagnostics_ods (every probe) since #2135, via vest_equilibrium_magnetics_detailed.",
        "algorithm_semantics": "gain -> zero-phase apply_fir_filter -> -cumtrapz_compat -> linear baseline (_linear_baseline over baseline_indices, #639 degenerate handling). Replaced vest_b_field_pol_probe_legacy (causal lfilter, 0.5 ms lag), which #2135 removed.",
        "scientific_risk": "low",
        "recommended_action": "keep as the single B-pol probe chain.",
        "related_issue": "#2135, #1541",
        "resolved_by": "#2135",
        "evidence": "#2147 comment: on 39915 main = public(t - 0.50 ms) on all 63 probes before #2135; vacuum residual improves on 38030/39915/41660/44136/45001.",
    },
    ("vaft/process/magnetics.py", "vaft.process.magnetics._linear_baseline"): {
        "classification": "legacy",
        "algorithm_semantics": "delegates to signal_processing._polyfit_linear_baseline (nan_policy='omit'); keeps the caller-facing stacklevel.",
        "scientific_risk": "low",
        "recommended_action": "E: retire the private alias; callers use subtract_baseline(fitting_opt='polyfit').",
        "related_issue": "#2162, #639",
        "resolved_by": "#2162 (B3.1)",
        "evidence": "bit-identical to the frozen develop copy (values, error and warning text, stacklevel).",
    },
    ("vaft/process/magnetics.py", "vaft.process.magnetics.vest_flux_loop_flux_from_voltage"): {
        "classification": "intentional_specialization",
        "production_reachability_note": "pipeline1 generate_diagnostics_ods, all 11 loops.",
        "configuration_source": "vest.yaml equilibrium_magnetics.processing.window (flux_baseline_window / flux_baseline_samples / daq_mode) else VestMagneticsProcessingConfig legacy index ladder.",
        "algorithm_semantics": "-cumtrapz of the calibrated loop voltage / 2pi, then polyfit baseline chosen by era: leading samples (native DAQ), [0.24, 0.26] s window (slow DAQ 43685-46403) or legacy sample ranges (3499:5000 / 11999:15000, or 5999:7000 for flux_loop_number in (9, 10, 11)). flux_loop_number is the 1-based POSITIONAL counter, so 9-11 are fl_13/fl_14/fl_15 (canonical indices 72-74), not physical loops #9-#11 -- donor intent unverified (magnetics_reconciliation.md Q7). NO low pass.",
        "scientific_risk": "medium",
        "recommended_action": "keep the three baseline rules (donor VEST_MagneticSignalProcessing[2].m); fix the comments that claim an 'already low-passed loop voltage' (process/magnetics.py around the flux branch, machine_mapping/magnetics.py _interpolate_signal) in B3.",
        "related_issue": "#2162",
        "evidence": "Pipeline 1 trace: no filter call on the flux-loop path; comments contradict the code.",
    },
    ("vaft/process/magnetics.py", "vaft.process.magnetics.mirnov_preprocess_signal"): {
        "classification": "intentional_specialization",
        "production_reachability_note": "not called by Pipeline 1 (fluctuation Mirnov voltages are published raw); research callers only.",
        "algorithm_semantics": "mean removal -> fir_filter_coefficients(window='hann', numtaps=filter_order or fs*1e-3 forced odd) high/low pass -> apply_fir_filter(short_record='causal') (since #2135; same length rule as before) -> / amplifier gain; a cutoff <= 0 or >= Nyquist skips its stage and is recorded in the report (or warned without one, #2159); an even explicit filter_order raises.",
        "scientific_risk": "medium",
        "recommended_action": "none; the Nyquist skip is an explicit status since #2159.",
        "related_issue": "#2159, #2162",
        "resolved_by": "#2159 (explicit skip status); #2162 (B3.4: _firwin_order -> fir_filter_coefficients, coefficients identical at 25 k/250 k/1 M/2 MHz)",
    },
    ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics.vest_plasma_rogowski_current"): {
        "classification": "canonical",
        "algorithm_semantics": "calibration only (vest.yaml plasma_current.calibration).",
        "scientific_risk": "low",
        "recommended_action": "keep.",
        "related_issue": "",
    },
    ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics.vest_diamagnetic_rogowski_current"): {
        "classification": "canonical",
        "algorithm_semantics": "repair_clipped_interval then cumtrapz_compat * (-1/shunt).",
        "scientific_risk": "low",
        "recommended_action": "keep.",
        "related_issue": "#2162",
        "resolved_by": "#2162 (B3.3)",
    },
    # ---- machine_mapping magnetics ------------------------------------------
    ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics._linear_baseline_subtract"): {
        "classification": "legacy",
        "algorithm_semantics": "delegates to signal_processing._polyfit_linear_baseline with nan_policy='propagate' (a NaN in the window keeps Ip all-NaN, as before) (production Ip baseline).",
        "scientific_risk": "low",
        "recommended_action": "E: retire the private alias. MM._polyfit_baseline (dead) was deleted in B3.1.",
        "related_issue": "#2162",
        "resolved_by": "#2162 (B3.1)",
    },
    ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics.vfit_plasma_current"): {
        "classification": "intentional_specialization",
        "production_reachability_note": "pipeline1 generate_diagnostics_ods (Ip, ref = -1) and generate_impa_ods.",
        "configuration_source": "vest.yaml plasma_current.{calibration, processing.baseline, processing.reference} with shot revisions.",
        "algorithm_semantics": "calibrate -> polyfit baseline -> FL10 reference subtraction (subtract | subtract_fl10_windowed: decimate + smooth + interp | disabled >= 47117) -> sign flip; the ref != -1 low pass (causal lfilter before #2135, zero-phase apply_fir_filter since, which raises on records under 79 samples) has no production caller.",
        "scientific_risk": "medium",
        "recommended_action": "keep the FL10 decimate chain as a named specialization; baselines delegate (B3).",
        "related_issue": "#2135, #2162",
        "resolved_by": "#2135 (Ip-reference low pass only)",
    },
    # ---- other machine_mapping chains ---------------------------------------
    ("vaft/machine_mapping/pf_active.py", "vaft.machine_mapping.pf_active._pf_filter"): {
        "classification": "intentional_specialization",
        "production_reachability_note": "pipeline1 generate_diagnostics_ods (all PF circuits) and IMPA strict vfit_pf.",
        "configuration_source": "vest.yaml 0.diagnostics.pf_active.processing.filter (251 taps, 2.5 kHz, fs 25 kHz, smoothing 10).",
        "algorithm_semantics": "reads vest.yaml and calls signal_processing.pf_current_filter (was _legacy_pf_filter). " + _PR2135,
        "scientific_risk": "medium",
        "recommended_action": "keep (config adapter).",
        "related_issue": "#2162, #2135",
        "resolved_by": "#2162 (B3.5)",
        "evidence": "test/test_pf_current_filter.py: bit-identical to the frozen develop _legacy_pf_filter at 20000/754/753/200/5 samples; 4-shot capture leaf-identical.",
    },
    ("vaft/process/signal_processing.py", "vaft.process.signal_processing.pf_current_filter"): {
        "classification": "canonical",
        "configuration_source": "caller arguments (vest.yaml values via pf_active._pf_filter).",
        "algorithm_semantics": "fir_filter_coefficients(lowpass, window='hamming', numtaps odd >= 3) -> apply_fir_filter(short_record='causal': zero-phase from 3*numtaps+1 samples, forward-only below, 125-sample / 5 ms delay) -> uniform_filter1d(min(window, len)) (half-sample lag for an even width).",
        "scientific_risk": "low",
        "recommended_action": "keep as the named PF composition.",
        "related_issue": "#2162",
        "resolved_by": "#2162 (B3.5)",
    },
    ("vaft/machine_mapping/tf.py", "vaft.machine_mapping.tf.vfit_tf_current_detailed"): {
        "classification": "intentional_specialization",
        "production_reachability_note": "pipeline1 generate_diagnostics_ods (TF) and IMPA (vfit_tf_current).",
        "configuration_source": "vest.yaml 0.diagnostics.tf (calibration -3e4, 26 taps, 2.5 kHz, fs 25 kHz, baseline 1000, smoothing 50, excursion_repair).",
        "algorithm_semantics": "calibrate -> mean baseline (1000) -> repair_tf_excursions -> firwin(26) -> zero-phase apply_fir_filter (causal lfilter, 0.5 ms lag, before #2135) -> smooth(50). apply_fir_filter uses the default short_record='raise', so a TF record under 3*26+1 = 79 samples now raises (failing TF and IMPA via vfit_tf_current) where it was filtered causally before.",
        "scientific_risk": "low",
        "recommended_action": "keep excursion repair + smooth(50) as TF specialization; products made before #2135 carry the lag (regeneration needs owner approval).",
        "related_issue": "#2135",
        "resolved_by": "#2135",
        "evidence": "#2135: donor VEST_TFwaveform.m uses filtfilt; causal 26 taps at 25 kHz = 0.5 ms lag; IMPA TF coupling 1.01-1.05 -> 1.02-1.07.",
    },
    ("vaft/machine_mapping/tf.py", "vaft.machine_mapping.tf.repair_tf_excursions"): {
        "classification": "intentional_specialization",
        "algorithm_semantics": "uniform_filter1d + median_filter detection, local polyfit bridge of TF excursions.",
        "scientific_risk": "low",
        "recommended_action": "keep (TF-specific).",
        "related_issue": "",
    },
    ("vaft/machine_mapping/barometry.py", "vaft.machine_mapping.barometry.vfit_barometry_dynamic"): {
        "classification": "intentional_specialization",
        "algorithm_semantics": "medfilt despike then resample_to_time (a run too short to anti-alias is stored NaN since B3.12).",
        "scientific_risk": "low",
        "recommended_action": "keep.",
        "related_issue": "#2162",
    },
    ("vaft/machine_mapping/langmuir_probes.py", "vaft.machine_mapping.langmuir_probes.vfit_langmuir_probes_dynamic"): {
        "classification": "intentional_specialization",
        "algorithm_semantics": "resample_to_time with its anti-alias filter, return_filter_mask=True and unfiltered='keep': target samples touched by a run too short to filter keep their value and get validity -1 (#915).",
        "scientific_risk": "low",
        "recommended_action": "keep: resampling already delegates to resample_to_time (anti-alias inside).",
        "related_issue": "#2162",
    },
    # ---- process: IMPA, onset, SXR -----------------------------------------
    ("vaft/process/impa.py", "vaft.process.impa.impa_lowpass"): {
        "classification": "intentional_specialization",
        "production_reachability_note": "pipeline1 generate_impa_ods (opt-in, impa.enable false in config.yaml).",
        "configuration_source": "vest.yaml impa.processing.signal_lowpass_hz (250 Hz) and sample_rate (default 25 kHz; compared with the acquired time vector by check_declared_sample_rate since #2159, report-only).",
        "algorithm_semantics": "firwin(_FIR_TAPS) -> filtfilt, lfilter fallback. " + _PR2135,
        "scientific_risk": "medium",
        "recommended_action": "none for the rate check (done, #2159); the 26-tap design's effective band edge is #2175.",
        "related_issue": "#432, #2159, #2175",
        "resolved_by": "#2159 (declared-vs-observed sample-rate status)",
    },
    ("vaft/process/soft_x_rays.py", "vaft.process.soft_x_rays.sxr_baseline_correction"): {
        "classification": "canonical",
        "algorithm_semantics": "SXR reference-window baseline subtraction.",
        "scientific_risk": "low",
        "recommended_action": "keep; plot/soft_x_rays._baseline_correct delegates (B3).",
        "related_issue": "#2162",
    },
    # ---- code / plot duplicates ---------------------------------------------
    ("vaft/code/efit/legacy.py", "vaft.code.efit.legacy.smooth"): {
        "classification": "legacy",
        "algorithm_semantics": "MATLAB smooth(span) re-implemented with explicit loops; agrees with signal_processing.smooth to ~1e-12 on finite data, but keeps a NaN local where the cumulative-sum version carries it into every later sample, and does not clip a span longer than the record.",
        "scientific_risk": "low",
        "recommended_action": "kept in B3.2 (not delegated, the NaN behaviour differs); retire together with its only caller, the deprecated H-alpha detector (E).",
        "related_issue": "#2162",
        "evidence": "test/test_preprocessing_duplicates.py pins the NaN difference.",
    },
}

# Remaining curated rows: production chains and the duplicates #2147 names.
CURATED.update(
    {
        ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics._apply_fl10_windowed_compensation"): {
            "classification": "intentional_specialization",
            "production_reachability_note": "pipeline1 Ip for reference mode subtract_fl10_windowed (vest.yaml plasma_current, 46403-47116).",
            "configuration_source": "vest.yaml 0.diagnostics.plasma_current.processing.reference revisions.",
            "algorithm_semantics": "scipy.signal.decimate (Chebyshev IIR, zero-phase) of FL10 voltage, offset sample, smooth(), np.interp back onto the Ip grid.",
            "scientific_risk": "medium",
            "recommended_action": "keep as named FL10 compensation; record the decimation factor in provenance (B2).",
            "related_issue": "#2162",
        },
        ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics.vest_diamagnetic_flux_detailed"): {
            "classification": "intentional_specialization",
            "production_reachability_note": "pipeline1 generate_diagnostics_ods (diamagnetic flux).",
            "configuration_source": "vest.yaml 0.diagnostics.diamagnetic_flux (TF circuit L/R/C, shunt, saturation limits) + plasma window from vest.yaml plasma_timing.",
            "algorithm_semantics": "repair_clipped_interval -> cumtrapz_compat -> gap-bridged reference by np.interp -> two more cumtrapz_compat passes -> two-point straight-line baseline written by hand -> x1000.",
            "scientific_risk": "medium",
            "recommended_action": "keep the two-point baseline (documented semantics differ from a least-squares fit).",
            "related_issue": "#2162",
            "resolved_by": "#2162 (B3.3: cumulative_trapezoid -> cumtrapz_compat)",
        },
        ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics._interpolate_signal"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "np.interp onto the policy grid for flux loops and probes (already on magnetics_time), Ip (_map_ip) and diamagnetic flux (native grid, unfiltered); the comment now says so (B3.8).",
            "scientific_risk": "medium",
            "recommended_action": "D: check the spectra of the unfiltered callers (loops upstream, Ip, diamagnetic flux) above the target Nyquist before deciding on resample_to_time.",
            "related_issue": "#2162, #425",
            "resolved_by": "#2162 (B3.8 comment only)",
        },
        ("vaft/machine_mapping/magnetics.py", "vaft.machine_mapping.magnetics._baseline_correct_limiter_voltage"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "median of the 0-0.2 s window subtracted from limiter shunt voltages.",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/machine_mapping/pf_active.py", "vaft.machine_mapping.pf_active._baseline_mean"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "mean of the first baseline_samples (5000) samples.",
            "scientific_risk": "low",
            "recommended_action": "keep (offset, not drift, removal).",
            "related_issue": "",
        },
        ("vaft/machine_mapping/impa.py", "vaft.machine_mapping.impa._resample"): {
            "classification": "unverified",
            "algorithm_semantics": "np.interp onto the first-loaded channel's grid; no-op when the grids already agree.",
            "scientific_risk": "medium",
            "recommended_action": "B2: check_declared_sample_rate on the reference grid (#425/#432).",
            "related_issue": "#425, #432",
        },
        ("vaft/process/magnetics.py", "vaft.process.magnetics.rogowski_coil_ip"): {
            "classification": "intentional_specialization",
            "production_reachability_note": "not reached by the Snakefiles; generic (non-VEST) Rogowski chain.",
            "algorithm_semantics": "define_baseline/subtract_baseline and savgol_filter(polyorder=1) of the flux reference.",
            "scientific_risk": "low",
            "recommended_action": "keep; covered by the subtract_baseline owner change (B3).",
            "related_issue": "#2162",
        },
        ("vaft/process/magnetics.py", "vaft.process.magnetics.flux_loop_flux"): {
            "classification": "canonical",
            "production_reachability_note": "not reached by the Snakefiles (VEST uses vest_flux_loop_flux_from_voltage).",
            "algorithm_semantics": "generic flux-loop chain: cumtrapz_compat and define/subtract_baseline.",
            "scientific_risk": "low",
            "recommended_action": "keep as the machine-independent loop chain; the VEST chain stays a specialization.",
            "related_issue": "",
        },
        ("vaft/process/onset.py", "vaft.process.onset.zero_phase_lowpass"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "butterworth_lowpass(order 4, zero_phase) for onset detection only.",
            "scientific_risk": "low",
            "recommended_action": "keep; already delegates to the canonical Butterworth.",
            "related_issue": "",
        },
        ("vaft/process/onset.py", "vaft.process.onset.median_smooth"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "scipy medfilt despike for onset detection (vest.yaml plasma_timing prefilter).",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/process/langmuir.py", "vaft.process.langmuir.median_filter_signal"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "scipy medfilt of Langmuir probe currents.",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/process/signal_processing.py", "vaft.process.signal_processing.signal_on_offset"): {
            "classification": "legacy",
            "algorithm_semantics": "savgol smoothing then threshold for on/offset (signal_onoffset alias).",
            "scientific_risk": "low",
            "recommended_action": "E: retire once onset.* replaces every caller.",
            "related_issue": "#2168",
        },
        ("vaft/code/efit/legacy.py", "vaft.code.efit.legacy.vest_signal_onoffsetpeak"): {
            "classification": "legacy",
            "production_reachability_note": "statically reached through correct_flux_loop, which generate_constraints_ods runs only when fl_correct_option != 0 (0 in config.yaml).",
            "algorithm_semantics": "savgol smoothing + normalised threshold onset/peak/offset.",
            "scientific_risk": "low",
            "recommended_action": "E: retire with correct_flux_loop if the option stays 0.",
            "related_issue": "#2168",
        },
        ("vaft/code/efit/legacy.py", "vaft.code.efit.legacy.correct_flux_loop"): {
            "classification": "legacy",
            "production_reachability_note": "config-gated off (constraints.fl_correct_option = 0).",
            "algorithm_semantics": "VFIT flux-loop correction with np.interp.",
            "scientific_risk": "low",
            "recommended_action": "E: retire or keep as explicit compatibility option.",
            "related_issue": "#2168",
        },
        ("vaft/code/efit/legacy.py", "vaft.code.efit.legacy.box_average"): {
            "classification": "intentional_specialization",
            "production_reachability_note": "pipeline1 generate_constraints_ods (vacuum-time drop and constraint averaging).",
            "algorithm_semantics": "mean over a closed window [center-half_width, center+half_width].",
            "scientific_risk": "low",
            "recommended_action": "keep (EFIT constraint averaging contract).",
            "related_issue": "",
        },
        ("vaft/code/efit/kfile.py", "vaft.code.efit.kfile.build_efit_coil_currents"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "np.interp of PF circuit currents onto EFIT coil groups (vest.yaml efit_coilset).",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/code/efit/kfile.py", "vaft.code.efit.kfile.generate_constraints_ods"): {
            "classification": "intentional_specialization",
            "production_reachability_note": "pipeline1 generate_constraints_ods; probe/loop grouping by INBOARD_PROBE_MAX_R etc. at kfile.py ~578-624.",
            "algorithm_semantics": "box-averaged constraints, |relative error x data| uncertainty, np.interp of eddy loop currents (VCURRT).",
            "scientific_risk": "medium",
            "recommended_action": "C2: grouping from the registry 'family' instead of r/z thresholds.",
            "related_issue": "#2164",
        },
        ("vaft/plot/backend/recipes.py", "vaft.plot.backend.recipes._build_interferometer_spectrogram"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "butterworth_highpass(order 4, 200 Hz, zero_phase=True), sample rate inferred from median dt.",
            "scientific_risk": "low",
            "recommended_action": "keep (display recipe; the filter is the canonical one).",
            "related_issue": "#2162",
            "resolved_by": "#2162 (B3.6)",
        },
        ("vaft/plot/soft_x_rays.py", "vaft.plot.soft_x_rays._baseline_correct"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "subtract nanmean over a caller time range (no fallback): a display offset.",
            "scientific_risk": "low",
            "recommended_action": "kept in B3.7 with the reason at the definition (sxr_baseline_correction is a tail window with a last-1000-samples fallback, not a drop-in); B3.11 decides the single SXR baseline owner.",
            "related_issue": "#2162",
            "evidence": "semantics differ: arbitrary range mean vs tail window with fallback.",
        },
        ("vaft/compat.py", "vaft.compat.cumtrapz_compat"): {
            "classification": "canonical",
            "algorithm_semantics": "cumulative trapezoid, scipy cumulative_trapezoid or the old cumtrapz name chosen by getattr; numerically identical.",
            "scientific_risk": "low",
            "recommended_action": "keep as the single cumulative-integration owner; direct cumulative_trapezoid calls delegate (B3).",
            "related_issue": "#2162",
        },
        ("vaft/machine_mapping/utils.py", "vaft.machine_mapping.utils.calibrate_vest_signal"): {
            "classification": "canonical",
            "algorithm_semantics": "named declarative calibration (linear multiply/divide, exponential_pressure, logarithmic_power) validated by validate_calibration.",
            "scientific_risk": "low",
            "recommended_action": "keep as the calibration owner.",
            "related_issue": "",
        },
        ("vaft/process/magnetics.py", "vaft.process.magnetics._calibrated_signal"): {
            "classification": "duplicate",
            "algorithm_semantics": "divide|multiply by the MD.yaml calibration: the linear case of calibrate_vest_signal without its validation (zero divisor not rejected).",
            "scientific_risk": "low",
            "recommended_action": "B1/B3: registry validation rejects zero calibration; keep the private helper or route through calibrate_vest_signal (equivalent for the linear case).",
            "related_issue": "#2158, #2162",
        },
        ("vaft/process/magnetics.py", "vaft.process.magnetics.vest_flux_loop_voltage"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "calibration only; no low pass anywhere on the loop-voltage path.",
            "scientific_risk": "medium",
            "recommended_action": "keep; see vest_equilibrium_magnetics_detailed for the anti-alias question.",
            "related_issue": "#2162",
        },
        ("vaft/process/magnetics.py", "vaft.process.magnetics.vest_equilibrium_magnetics_detailed"): {
            "classification": "intentional_specialization",
            "production_reachability_note": "pipeline1 generate_diagnostics_ods (every equilibrium probe and loop).",
            "configuration_source": "VestMagneticsProcessingConfig resolved from vest.yaml equilibrium_magnetics.processing.window (+ optional workflow override via magnetics_processing_for_shot).",
            "algorithm_semantics": "per channel: probe chain or flux chain on the native grid, then np.interp onto timebase()[index_start:index_end+1]. The flux-loop branch comment claims 'an already low-passed loop voltage'; the code applies no low pass, so in the native-DAQ era (>= 46404, 250 kHz loops) the decimating np.interp onto 25 kHz has no anti-alias filter beyond the integration's 1/f roll-off.",
            "scientific_risk": "medium",
            "recommended_action": "D: measure the flux-loop spectrum above 12.5 kHz on a native-DAQ shot before deciding; B3: correct the comment either way.",
            "related_issue": "#425, #2162, #2166",
            "evidence": "process/magnetics.py flux branch (comment above np.interp) vs vest_flux_loop_voltage (calibration only).",
        },
        ("vaft/process/numerical.py", "vaft.process.numerical.time_derivative"): {
            "classification": "canonical",
            "algorithm_semantics": "non-uniform 3-point weighted difference.",
            "scientific_risk": "low",
            "recommended_action": "keep as the time-derivative owner (np.gradient callers are mostly spatial).",
            "related_issue": "",
        },
        ("vaft/machine_mapping/pf_active.py", "vaft.machine_mapping.pf_active._coerce_signal_to_reference"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "np.interp of each PF circuit onto the first circuit's grid (same DAQ, no rate change).",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/machine_mapping/pf_active.py", "vaft.machine_mapping.pf_active.vfit_pf_active_dynamic"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "np.interp of the filtered PF currents (2.5 kHz low pass at 25 kHz) onto the analysis grid.",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/machine_mapping/tf.py", "vaft.machine_mapping.tf.vfit_tf_dynamic"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "Bt*R from the filtered TF current, np.interp onto build_window_time_axis.",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/machine_mapping/impa.py", "vaft.machine_mapping.impa.impa"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "np.interp of IMPA results onto the target grid.",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "#432",
        },
        ("vaft/process/impa.py", "vaft.process.impa.fit_impa_tf_coupling"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "lstsq TF-coupling fit with a polyfit trend.",
            "scientific_risk": "medium",
            "recommended_action": "keep; #2135 moves the fitted coupling from 1.01-1.05 to 1.02-1.07 (TF zero-phase) -- record in D.",
            "related_issue": "#2135, #2166",
        },
        ("vaft/validation/magnetics.py", "vaft.validation.magnetics.spike_intervals"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "median_filter residual for QA spike detection; report-only, does not alter data.",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/formula/utils.py", "vaft.formula.utils.fit_profile"): {
            "classification": "canonical",
            "production_reachability_note": "pipeline2 core_profiles (Thomson/CES fits).",
            "algorithm_semantics": "curve_fit(absolute_sigma=True) of string-dispatched profile models; parametric modes return zero y_std (no propagated profile uncertainty).",
            "scientific_risk": "medium",
            "recommended_action": "keep as fit owner; the missing profile sigma is a separate uncertainty finding (Pipeline 2 trace), not a consolidation item.",
            "related_issue": "",
        },
        ("vaft/code/efit/kinetic.py", "vaft.code.efit.kinetic._ti_weighted_fit"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "weighted np.polyfit T_i fit with covariance, floored at the channel error (kinetic EFIT).",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/process/resistive_zeff.py", "vaft.process.resistive_zeff.smooth_local_polynomial"): {
            "classification": "intentional_specialization",
            "algorithm_semantics": "local polyfit smoothing, off by default (resistive_zeff.smoothing.window_s null).",
            "scientific_risk": "low",
            "recommended_action": "keep.",
            "related_issue": "",
        },
        ("vaft/code/efit/legacy.py", "vaft.code.efit.legacy.box_average_series"): {
            "classification": "intentional_specialization",
            "production_reachability_note": "pipeline1 generate_constraints_ods (EFIT applicability: the box-averaged Ip at every sample).",
            "algorithm_semantics": "cumsum-difference box average at every sample over [t-h, t+h] (searchsorted left/right): legacy.box_average evaluated at every sample; moved from the workflow's _box_averaged (B3.10).",
            "scientific_risk": "low",
            "recommended_action": "keep beside box_average (vectorised form of the same contract).",
            "related_issue": "#2162, #2164",
            "resolved_by": "#2162 (B3.10)",
            "evidence": "test/test_box_average_series.py: bit-identical to the frozen workflow helper; per-sample box_average to 1e-12 relative.",
        },
        ("vaft/machine_mapping/soft_x_rays.py", "vaft.machine_mapping.soft_x_rays.vfit_soft_x_rays_dynamic"): {
            "classification": "duplicate",
            "algorithm_semantics": "np.mean baseline over an index range (baseline_range) per channel: a fourth SXR baseline variant beside process.soft_x_rays.sxr_baseline_correction, plot.soft_x_rays._baseline_correct and the rolling detrend.",
            "scientific_risk": "low",
            "recommended_action": "B3: one SXR baseline owner with explicit window semantics (index range, time range, tail with fallback).",
            "related_issue": "#2162",
        },
        ("workflow/resistive_zeff/build_zeff.py", "workflow.resistive_zeff.build_zeff._ip_trace"): {
            "classification": "unverified",
            "algorithm_semantics": "savgol_filter of Ip in a study script.",
            "scientific_risk": "low",
            "recommended_action": "study only; C5 replaces the hard-coded 'Flux Loop - #10' name lookup with the registry id fl_10.",
            "related_issue": "#2164",
        },
    }
)


#: Rows whose curated data names a different implementation but whose
#: classification is mechanical: every function under these files whose only
#: preprocessing is plotting-time smoothing is a plot specialization.
LAYER_DEFAULTS: dict[str, dict[str, str]] = {
    "plot": {
        "classification": "unverified",
        "scientific_risk": "low",
        "recommended_action": "plot-time transformation; does not feed stored products. Triage in B3 only where it restates a process helper.",
    },
    "workflow_study": {
        "classification": "unverified",
        "scientific_risk": "low",
        "recommended_action": "research/study script outside the production Snakefiles; not migrated unless promoted.",
    },
}

#: Uncurated readers of the external magnetics YAMLs (or of their loaders).
ASSET_CONSUMER_DEFAULTS = {
    "classification": "legacy",
    "scientific_risk": "medium",
    "recommended_action": "B1/C: read the registry resolver instead; see the consumer migration map in magnetics_reconciliation.md.",
}

#: Uncurated, statically unreached numerics outside machine_mapping: geometry,
#: profile and equilibrium post-processing on reconstructed quantities rather
#: than measured-signal preprocessing. Listed for completeness, not triaged.
DERIVED_DEFAULTS = {
    "classification": "unverified",
    "scientific_risk": "unassessed",
    "recommended_action": "outside the #2147 measured-signal consolidation; not triaged in Phase A.",
}


# --------------------------------------------------------------------------
# AST scanning
# --------------------------------------------------------------------------


def _module_name(relpath: str) -> str:
    parts = Path(relpath).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _owning_layer(relpath: str) -> str:
    parts = Path(relpath).parts
    if parts[0] == "workflow":
        if len(parts) > 1 and parts[1].startswith("automatic_pipeline_"):
            return "workflow_production"
        return "workflow_study"
    if len(parts) > 1 and parts[1] in {"omas", "imas"}:
        return "omas"
    if len(parts) > 2:
        return parts[1]
    return "vaft"


def _tracked_files(root: Path) -> set[str] | None:
    """Git-tracked paths, so untracked local scripts cannot change the products."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", *SCAN_ROOTS],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {entry for entry in listing.decode("utf-8").split("\0") if entry}


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    tracked = _tracked_files(root)
    for scan_root in SCAN_ROOTS:
        for path in sorted((root / scan_root).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if tracked is not None and rel not in tracked:
                continue
            if "/__pycache__/" in rel or "/.ipynb_checkpoints/" in rel:
                continue
            if rel.startswith(EXCLUDED_PREFIXES):
                continue
            files.append(path)
    return files


class _ModuleScan(ast.NodeVisitor):
    """Collect imports, definitions and calls of one module."""

    def __init__(self, root: Path, path: Path):
        self.root = root
        self.path = path
        self.relpath = path.relative_to(root).as_posix()
        self.module = _module_name(self.relpath)
        self.package = self.module if path.name == "__init__.py" else self.module.rpartition(".")[0]
        self.aliases: dict[str, str] = {}
        self.defs: dict[str, int] = {}  # qualified name -> line
        self.classes: set[str] = set()
        self.calls: list[tuple[str, str, int, str]] = []  # (scope, callee, line, snippet)
        self.asset_refs: list[tuple[str, str, int]] = []  # (scope, asset, line)
        self.module_level_names: set[str] = set()
        self._scope: list[str] = []

    # -- imports ------------------------------------------------------------
    def _resolve_relative(self, level: int, module: str | None) -> str:
        base = self.package.split(".") if self.package else []
        if level > 1:
            base = base[: len(base) - (level - 1)]
        tail = module.split(".") if module else []
        return ".".join(base + tail)

    def _sibling_module(self, name: str) -> str | None:
        head = name.split(".")[0]
        if (self.path.parent / f"{head}.py").exists() and not self.module.startswith("vaft"):
            return _module_name((self.path.parent / f"{head}.py").relative_to(self.root).as_posix())
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            target = alias.name
            sibling = self._sibling_module(target)
            if sibling is not None:
                target = sibling
            if alias.asname:
                self.aliases[alias.asname] = target
            else:
                head = alias.name.split(".")[0]
                self.aliases.setdefault(head, sibling or head)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = self._resolve_relative(node.level, node.module)
        else:
            base = node.module or ""
            sibling = self._sibling_module(base) if base else None
            if sibling is not None:
                base = sibling
        for alias in node.names:
            if alias.name == "*":
                continue
            self.aliases[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name

    # -- scopes -------------------------------------------------------------
    def _qualified(self, name: str) -> str:
        return ".".join([self.module, *self._scope, name])

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = self._qualified(node.name)
        self.defs[qualified] = node.lineno
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = self._qualified(node.name)
        self.defs[qualified] = node.lineno
        self.classes.add(qualified)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _current_scope(self) -> str:
        return ".".join([self.module, *self._scope]) if self._scope else f"{self.module}.<module>"

    # -- calls and asset references ----------------------------------------
    def _dotted(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._dotted(node.value)
            return None if base is None else f"{base}.{node.attr}"
        return None

    def resolve(self, dotted: str) -> str:
        if dotted.startswith("<attribute>."):
            return dotted
        head, _, rest = dotted.partition(".")
        if head in self.aliases:
            target = self.aliases[head]
        elif head in self.module_level_names:
            target = f"{self.module}.{head}"
        else:
            target = head
        return f"{target}.{rest}" if rest else target

    def visit_Expr(self, node: ast.Expr) -> None:
        # A bare string statement is a docstring or comment-like literal: it
        # names an asset without reading it.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._dotted(node.func)
        if dotted is None and isinstance(node.func, ast.Attribute) and node.func.attr == "rolling":
            dotted = "<attribute>.rolling"
        if dotted is not None:
            try:
                snippet = ast.unparse(node)
            except Exception:  # pragma: no cover - unparse is total on parsed trees
                snippet = dotted
            snippet = " ".join(snippet.split())
            if len(snippet) > _SNIPPET_LIMIT:
                snippet = snippet[: _SNIPPET_LIMIT - 3] + "..."
            self.calls.append((self._current_scope(), dotted, node.lineno, snippet))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            for asset in ASSET_LITERALS:
                if asset in node.value:
                    self.asset_refs.append((self._current_scope(), asset, node.lineno))
        self.generic_visit(node)


def _numpy_scipy_normalize(name: str) -> str:
    """Map ``np.``-style and submodule spellings onto the PRIMITIVES keys."""
    for prefix, canonical in (
        ("scipy.signal.signaltools.", "scipy.signal."),
        ("scipy.signal._signaltools.", "scipy.signal."),
        ("scipy.integrate._quadrature.", "scipy.integrate."),
        ("scipy.ndimage.filters.", "scipy.ndimage."),
        ("numpy.lib.", "numpy."),
    ):
        if name.startswith(prefix):
            return canonical + name[len(prefix):]
    return name


def scan(root: Path) -> dict[str, Any]:
    modules: list[_ModuleScan] = []
    for path in _python_files(root):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module = _ModuleScan(root, path)
        module.module_level_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        module.visit(tree)
        modules.append(module)
    return {"modules": modules}


def _definition_index(modules: Iterable[_ModuleScan]) -> tuple[dict[str, tuple[str, int]], dict[str, list[str]]]:
    defs: dict[str, tuple[str, int]] = {}
    by_short: dict[str, list[str]] = defaultdict(list)
    for module in modules:
        for qualified, line in module.defs.items():
            defs[qualified] = (module.relpath, line)
            by_short[qualified.rsplit(".", 1)[-1]].append(qualified)
    return defs, by_short


def _targets(callee: str, defs: dict[str, tuple[str, int]], by_short: dict[str, list[str]]) -> list[str]:
    """Definitions a resolved callee may name (exact, or re-exported by a parent package)."""
    if callee in defs:
        return [callee]
    if not callee.startswith("vaft"):
        return []
    prefix, _, short = callee.rpartition(".")
    return sorted(q for q in by_short.get(short, ()) if q.startswith(prefix + ".") and q.count(".") > prefix.count(".") + 1)


def _entry_modules(root: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    for pipeline, directory in PIPELINES.items():
        snakefile = root / directory / "Snakefile"
        text = snakefile.read_text(encoding="utf-8")
        scripts = set(re.findall(r"python\s+\{SCRIPT_DIR\}/([\w./-]+\.py)", text))
        scripts |= set(re.findall(r"python\s+(gen_[\w]+\.py)", text))
        names = [_module_name(f"{directory}/{script}") for script in sorted(scripts)]
        if "{PLOT_SCRIPT}" in text:
            names.append(_module_name(f"{PIPELINES['pipeline1']}/generate_stage_plots.py"))
        if "{REPLICATE_SCRIPT}" in text:
            names.append(_module_name(f"{PIPELINES['pipeline1']}/replicate_to_hsds.py"))
        entries[pipeline] = sorted(set(names))
    return entries


def _reachability(
    modules: list[_ModuleScan],
    defs: dict[str, tuple[str, int]],
    by_short: dict[str, list[str]],
    entries: dict[str, list[str]],
) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    module_defs: dict[str, list[str]] = defaultdict(list)
    for module in modules:
        for qualified in module.defs:
            module_defs[module.module].append(qualified)
            # A nested scope is reached with its parent; a class with its methods.
            parent = qualified.rpartition(".")[0]
            if parent in module.defs:
                edges[parent].add(qualified)
        for scope, dotted, _line, _snippet in module.calls:
            callee = _numpy_scipy_normalize(module.resolve(dotted))
            for target in _targets(callee, defs, by_short):
                edges[scope].add(target)
    class_names = {q for m in modules for q in m.classes}
    for qualified in class_names:
        for child in list(defs):
            if child.startswith(qualified + "."):
                edges[qualified].add(child)
    reached: dict[str, set[str]] = {}
    for pipeline, names in entries.items():
        seen: set[str] = set()
        queue: deque[str] = deque()
        for name in names:
            queue.append(f"{name}.<module>")
            queue.extend(module_defs.get(name, ()))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(edges.get(node, ()))
        reached[pipeline] = seen
    return reached


def _row_key(relpath: str, implementation: str) -> tuple[str, str]:
    return relpath, implementation


def build_inventory(root: Path = REPO_ROOT) -> dict[str, Any]:
    modules = scan(root)["modules"]
    defs, by_short = _definition_index(modules)
    entries = _entry_modules(root)
    reached = _reachability(modules, defs, by_short, entries)

    callers: dict[str, set[str]] = defaultdict(set)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    config_calls: dict[str, set[str]] = defaultdict(set)

    for module in modules:
        for scope, dotted, line, snippet in module.calls:
            callee = _numpy_scipy_normalize(module.resolve(dotted))
            for target in _targets(callee, defs, by_short):
                callers[target].add(scope)
            short = callee.rsplit(".", 1)[-1]
            if short in CONFIG_RESOLVERS:
                config_calls[scope].add(short)
            operation = PRIMITIVES.get(callee)
            assets: tuple[str, ...] = ()
            if operation is None:
                for loader, loader_assets in MAGNETICS_LOADERS.items():
                    if loader in _targets(callee, defs, by_short) or callee == loader:
                        operation, assets = "magnetics_asset_consumer", loader_assets
                        break
            if operation is None:
                continue
            implementation = scope
            key = _row_key(module.relpath, implementation)
            row = rows.setdefault(key, _new_row(module, implementation, defs))
            row["operation"].add(operation)
            row["primitives"].add(callee)
            row["assets"].update(assets)
            row["call_sites"].add((callee, snippet))
        for scope, asset, line in module.asset_refs:
            key = _row_key(module.relpath, scope)
            row = rows.setdefault(key, _new_row(module, scope, defs))
            row["operation"].add("magnetics_asset_read")
            row["assets"].add(asset)
            row["call_sites"].add((f"<string literal {asset}>", asset))

    definition_ops = dict(DEFINITION_OPERATIONS)
    definition_ops.update({k: v for k, v in PRIMITIVES.items() if k.startswith("vaft.")})
    for module in modules:
        for qualified, line in module.defs.items():
            operation = definition_ops.get(qualified)
            if operation is None:
                continue
            row = rows.setdefault(_row_key(module.relpath, qualified), _new_row(module, qualified, defs))
            row["operation"].add(operation)
            row["call_sites"].add(("<definition>", f"def {qualified.rsplit('.', 1)[-1]}"))

    unmatched = sorted(set(CURATED) - set(rows))
    if unmatched:
        raise SystemExit(f"curated entries without a scanned row: {unmatched}")

    output = []
    for key in sorted(rows):
        row = rows[key]
        implementation = row["implementation"]
        pipelines = sorted(p for p, seen in reached.items() if implementation in seen)
        row_callers = sorted(callers.get(implementation, ()))
        resolvers = sorted(config_calls.get(implementation, ()))
        layer = row["owning_layer"]
        curated = CURATED.get(key, {})
        asset_only = row["operation"] <= {"magnetics_asset_consumer", "magnetics_asset_read"}
        if asset_only:
            defaults = ASSET_CONSUMER_DEFAULTS
            scope = "magnetics_asset_consumer"
        else:
            defaults = LAYER_DEFAULTS.get(layer, {})
            in_scope = bool(curated) or layer in {"machine_mapping", "workflow_production"} or bool(pipelines)
            scope = "preprocessing" if in_scope else "derived_numerics_not_triaged"
            if not in_scope and not defaults:
                defaults = DERIVED_DEFAULTS
        entry = {
            "implementation": implementation,
            "source_file": row["source_file"],
            "owning_layer": layer,
            "operation": sorted(row["operation"]),
            "primitives": sorted(row["primitives"]),
            "assets": sorted(row["assets"]),
            "call_sites": [{"call": call, "snippet": snippet} for call, snippet in sorted(row["call_sites"])],
            "callers": row_callers,
            "production_reachability": (
                ",".join(pipelines) if pipelines else "not_statically_reached"
            ),
            "production_reachability_note": curated.get("production_reachability_note", ""),
            "configuration_source": curated.get(
                "configuration_source",
                f"vest.yaml via {', '.join(resolvers)}" if resolvers else "caller arguments / module literals (not curated)",
            ),
            "algorithm_semantics": curated.get(
                "algorithm_semantics", "calls " + ", ".join(sorted(row["primitives"])) + " (not curated)"
            ),
            "classification": curated.get("classification", defaults.get("classification", "unverified")),
            "scientific_risk": curated.get("scientific_risk", defaults.get("scientific_risk", "unassessed")),
            "recommended_action": curated.get(
                "recommended_action", defaults.get("recommended_action", "triage in the owning B3/C slice")
            ),
            "related_issue": curated.get("related_issue", ""),
            "evidence": curated.get("evidence", ""),
            "resolved_by": curated.get("resolved_by", ""),
            "curated": bool(curated),
            "audit_scope": scope,
        }
        output.append(entry)

    summary = defaultdict(int)
    for entry in output:
        summary[entry["classification"]] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "workflow/architecture_audit/build_inventory.py",
        "issue": "#2148",
        "scan_roots": list(SCAN_ROOTS),
        "entry_modules": entries,
        "reachability_method": (
            "static: Snakefile 'python <script>.py' entry modules, then resolved calls "
            "(imports, relative imports, parent-package re-exports); self/getattr/registry "
            "dispatch not followed; a called class reaches its methods"
        ),
        "classification_counts": dict(sorted(summary.items())),
        "rows": output,
    }


def _new_row(module: _ModuleScan, implementation: str, defs: dict[str, tuple[str, int]]) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "source_file": module.relpath,
        "owning_layer": _owning_layer(module.relpath),
        "operation": set(),
        "primitives": set(),
        "assets": set(),
        "call_sites": set(),
    }


# --------------------------------------------------------------------------
# Magnetics inventory
# --------------------------------------------------------------------------

#: Proposed stable B1 identifiers. ``table`` = the table.yaml DAQ label agrees
#: with the geometric ladder of its column; ``position`` = the label of the field
#: the 2409 layout wires there does NOT agree with the ladder, so the id is the
#: column and z, and the ladder name is only a candidate for the B1 owner to
#: confirm (magnetics_reconciliation.md Q4).
_LADDER_IDS = {
    # Position-shaped on purpose: a label-shaped id ('bpol_C4-05') would reuse
    # a string another position carries as its DAQ label. The ladder candidate
    # is recorded alongside for the B1 owner (Q4).
    35: ("bpol_C_z+0.060", "C2-04"),   # 2409 feeds field 225 'C4-05', pre-39438 field 170 'C2-04'
    46: ("bpol_C_z-0.380", "C4-05"),   # field 226 labelled 'C4-06'
    47: ("bpol_C_z-0.420", "C4-06"),   # 2409 feeds field 170 'C2-04', pre-39438 field 225 'C4-05'
    61: ("bpol_L_z-1.0328", "L-06"),   # field 72 labelled 'L-07'
    62: ("bpol_L_z-1.0728", "L-07"),   # field 71 labelled 'L-06'
}


def _proposed_id(index: int, kind: str, name: str) -> tuple[str, str]:
    if index in _LADDER_IDS:
        return _LADDER_IDS[index][0], "position"
    if kind == "flux_loop":
        match = re.fullmatch(r"Flux Loop - #(\d+)", name)
        return f"fl_{int(match.group(1)):02d}", "table"
    match = re.fullmatch(r"MagneticFieldProbe_([A-Z]\d?-\d\d)(_Bz)?", name)
    return f"bpol_{match.group(1)}", "table"


def _load(root: Path, relative: Path) -> Any:
    with open(root / relative, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _entry_canonical_index(
    entry: dict[str, Any], kind: str, records: list[dict[str, Any]], *, index_scope: str
) -> int:
    """The canonical registry index a ``wiring``/``known_faults`` entry names.

    The two forms the runtime accepts (``vaft.machine_mapping.magnetics_registry.
    _entry_record``), with its rules restated so the audit does not import the
    code it audits: ``{id: ...}`` (#2158), or the legacy ``{index, r, z}`` pinned
    by position, where ``index`` is canonical for probes and counts within the
    kind for flux loops (``index_scope="kind"``). A pin that misses its record
    raises, as it does at runtime.
    """
    if ("id" in entry) == ("index" in entry):
        raise ValueError(f"{entry!r}: name the sensor by id or by index, exactly one")
    if ("r" in entry) != ("z" in entry):
        raise ValueError(f"{entry!r}: give both r and z, or neither")

    def pinned(record: dict[str, Any]) -> bool:
        return all(
            key not in entry or abs(float(record[key]) - float(entry[key])) <= 1e-9 for key in ("r", "z")
        )

    if "id" in entry:
        matches = [i for i, r in enumerate(records) if str(r["id"]) == str(entry["id"])]
        if len(matches) != 1 or str(records[matches[0]]["kind"]) != kind:
            raise ValueError(f"{entry!r}: no {kind} with that id")
        # An id entry's r/z are optional, but checked when given.
        if not pinned(records[matches[0]]):
            raise ValueError(f"{entry!r}: the r/z it states are not those of its id")
        return matches[0]
    if "r" not in entry or "z" not in entry:
        raise ValueError(f"{entry!r}: a legacy index entry must pin r and z")
    index = int(entry["index"])
    if index_scope == "kind":
        rows = [i for i, r in enumerate(records) if str(r["kind"]) == kind]
        canonical = rows[index] if 0 <= index < len(rows) else None
    else:
        canonical = index if 0 <= index < len(records) else None
    record = records[canonical] if canonical is not None else None
    if record is None or str(record["kind"]) != kind or not pinned(record):
        raise ValueError(f"{entry!r}: the pinned r/z is not the {kind} at that index")
    return canonical


def build_magnetics_inventory(root: Path = REPO_ROOT) -> dict[str, Any]:
    md = _load(root, GEOMETRY_DIR / MD_YAML)
    geometry = _load(root, GEOMETRY_DIR / GEOMETRY_YAML)
    table = _load(root, GEOMETRY_DIR / TABLE_YAML)
    vest = _load(root, VEST_YAML)
    names = {int(e["field_code"]): str(e["name"]) for e in table["entries"]}
    processing = vest[0]["diagnostics"]["equilibrium_magnetics"]["processing"]
    wiring = processing["wiring"]
    faults = processing["known_faults"]
    # Since #2158 wiring overrides and faults name a sensor by its registry id;
    # the legacy {index, r, z} form the runtime still accepts is resolved too.
    records = vest[0]["diagnostics"]["equilibrium_magnetics"]["sensors"]["records"]
    if [int(r["index"]) for r in records] != list(range(len(records))):
        raise ValueError("vest.yaml sensor records are not in canonical index order")
    kind_index_by_canonical = {}
    _seen: dict[str, int] = defaultdict(int)
    for canonical, r in enumerate(records):
        kind_index_by_canonical[canonical] = _seen[str(r["kind"])]
        _seen[str(r["kind"])] += 1

    overrides_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for revision in wiring.get("revisions") or ():
        for entry in revision.get("probe_overrides") or ():
            overrides_by_index[
                _entry_canonical_index(entry, "b_field_pol_probe", records, index_scope="canonical")
            ].append(
                {
                    "from_shot": revision.get("from_shot"),
                    "to_shot": revision.get("to_shot"),
                    "layout": revision.get("layout"),
                    "field_code": int(entry["field_code"]),
                    "calibration": float(entry["calibration"]),
                    "name": names.get(int(entry["field_code"])),
                }
            )

    fault_windows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for block, bounds in [(faults, {})] + [
        (rev, {"from_shot": rev.get("from_shot"), "to_shot": rev.get("to_shot")})
        for rev in faults.get("revisions") or ()
    ]:
        for kind, key, scope in (
            ("b_field_pol_probe", "probes", "canonical"),
            ("flux_loop", "flux_loops", "kind"),
        ):
            for entry in block.get(key) or ():
                canonical = _entry_canonical_index(entry, kind, records, index_scope=scope)
                fault_windows[(kind, kind_index_by_canonical[canonical])].append(
                    {
                        "from_shot": bounds.get("from_shot"),
                        "to_shot": bounds.get("to_shot"),
                        "scope": "base (shots outside every revision)" if not bounds else "revision",
                        "reason": str(entry["reason"]),
                    }
                )

    sensors = []
    kind_counter: dict[str, int] = defaultdict(int)
    for index, (cal, geo) in enumerate(zip(md["channels"], geometry["channels"])):
        kind = str(geo["kind"])
        kind_index = kind_counter[kind]
        kind_counter[kind] += 1
        field_code = int(geo["field_code"])
        name = names[field_code]
        stable_id, id_basis = _proposed_id(index, kind, name)
        key = (kind, kind_index)
        sensors.append(
            {
                "proposed_id": stable_id,
                "id_basis": id_basis,
                "kind": kind,
                "canonical_index": index,
                "kind_index": kind_index,
                "name": name,
                "ladder_candidate": _LADDER_IDS[index][1] if index in _LADDER_IDS else None,
                # DAQ labels this position carries under another era's wiring.
                "aliases": sorted({o["name"] for o in overrides_by_index.get(index, ()) if o["name"] and o["name"] != name}),
                "r": float(geo["r"]),
                "z": float(geo["z"]),
                "field_code": field_code,
                "calibration": float(cal["calibration"]),
                "calibration_units": "V per (T/s), applied as divide (VestMagneticsProcessingConfig.calibration_mode)"
                if kind == "b_field_pol_probe"
                else "V per (Wb/s), applied as divide",
                "md_field_code": int(cal["field_code"]),
                "md_kind": str(cal["kind"]),
                "wiring_revisions": overrides_by_index.get(index, []),
                "known_faults": fault_windows.get(key, []),
                "family": _family(kind, float(geo["r"]), float(geo["z"])),
                "provenance": {
                    "calibration": f"{GEOMETRY_DIR.as_posix()}/{MD_YAML} (source: {md.get('source')})",
                    "geometry": f"{GEOMETRY_DIR.as_posix()}/{GEOMETRY_YAML} (source: {geometry.get('source')})",
                    "name": f"{GEOMETRY_DIR.as_posix()}/{TABLE_YAML} (source: {table.get('source')})",
                },
            }
        )
    counts = dict(sorted(kind_counter.items()))
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "workflow/architecture_audit/build_inventory.py",
        "issue": "#2148",
        "sources": {
            MD_YAML: {"declared_source": md.get("source"), "description": md.get("description")},
            GEOMETRY_YAML: {"declared_source": geometry.get("source"), "description": geometry.get("description")},
            TABLE_YAML: {"declared_source": table.get("source"), "description": table.get("description")},
            "vest.yaml": {"wiring_layout": wiring.get("layout"), "base_probe_overrides": list(wiring.get("probe_overrides") or [])},
        },
        "kind_counts": counts,
        "sensors": sensors,
    }


def _family(kind: str, r: float, z: float) -> str:
    # Same thresholds as vaft.machine_mapping.magnetics (INBOARD_PROBE_MAX_R etc.).
    if kind == "flux_loop":
        return "inboard" if r < 0.15 else ("outboard" if r > 0.5 else "unclassified")
    if abs(z) > 0.8:
        return "side"
    if r < 0.09:
        return "inboard"
    if r > 0.795:
        return "outboard"
    return "unclassified"


# --------------------------------------------------------------------------


def render(root: Path = REPO_ROOT) -> dict[str, str]:
    """``{relative output path: file text}`` for both products."""
    def dump(payload: Any) -> str:
        return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"

    return {
        (OUTPUT_DIR / INVENTORY_NAME).as_posix(): dump(build_inventory(root)),
        (OUTPUT_DIR / MAGNETICS_NAME).as_posix(): dump(build_magnetics_inventory(root)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--check", action="store_true", help="exit 1 if a committed product differs")
    args = parser.parse_args(argv)
    products = render(args.root)
    stale = []
    for relative, text in products.items():
        target = args.root / relative
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != text:
                stale.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if stale:
        print("stale: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
