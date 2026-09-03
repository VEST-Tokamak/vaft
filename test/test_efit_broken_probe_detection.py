"""Offline automatic Bpol probe-quality detection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from omas import ODS


SCRIPT = (
    Path(__file__).parents[1]
    / "workflow/automatic_pipeline_1_routine_data_processing/generate_constraints_ods.py"
)
SPEC = importlib.util.spec_from_file_location("generate_constraints_ods", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_detects_gross_integrator_drift_without_flagging_peer_variation():
    ods = ODS(consistency_check=False)
    time = np.linspace(0.26, 0.36, 2500)
    for index in range(64):
        amplitude = 0.05 * (1.0 + 0.1 * np.sin(index))
        values = amplitude * np.sin(2 * np.pi * 20 * time)
        if index == 25:
            values = values + np.linspace(0.0, 2.7, time.size)
        ods[f"magnetics.b_field_pol_probe.{index}.field.data"] = values

    assert MODULE._detect_broken_bpol_probes(ods) == [26]


def _array(n: int = 64, bad: int = 25) -> ODS:
    ods = ODS(consistency_check=False)
    time = np.linspace(0.26, 0.36, 2500)
    for index in range(n):
        values = 0.05 * (1.0 + 0.1 * np.sin(index)) * np.sin(2 * np.pi * 20 * time)
        if index == bad:
            values = values + np.linspace(0.0, 2.7, time.size)
        ods[f"magnetics.b_field_pol_probe.{index}.field.data"] = values
        ods[f"magnetics.b_field_pol_probe.{index}.field.time"] = time
    return ods


def test_projected_validity_wins_over_the_amplitude_detector():
    """Once the diagnostics stage has assessed the magnetics (#189/#343), the
    script's own detector must not vote: kfile already folds condemned
    channels in, and two detectors disagreeing on one probe is worse than one.
    Here the stage condemned probe 10 and cleared the drifting probe 25."""
    from vaft.validation.imas import write_validity

    ods = _array()
    for index in range(64):
        verdict = -2 if index == 9 else 0
        write_validity(
            ods, f"magnetics.b_field_pol_probe.{index}.field", [verdict] * 2500, scalar=verdict
        )
    assert MODULE._condemned_by_diagnostics_stage(ods) == [10]
    assert MODULE._resolve_broken(ods, [3], detect=True) == [3]  # kfile adds 10 itself
    assert MODULE._resolve_broken(ods, [], detect=False) == []


def test_products_without_an_assessment_fall_back_to_the_amplitude_detector():
    ods = _array()
    assert MODULE._condemned_by_diagnostics_stage(ods) is None
    assert MODULE._resolve_broken(ods, [3], detect=True) == [3, 26]


def test_condemned_flux_loops_are_listed_at_the_legacy_offset():
    from vaft.validation.imas import write_validity

    ods = _array(n=4)
    for index in range(4):
        write_validity(ods, f"magnetics.b_field_pol_probe.{index}.field", [0] * 2500, scalar=0)
    time = np.linspace(0.26, 0.36, 2500)
    for index in range(2):
        ods[f"magnetics.flux_loop.{index}.flux.data"] = np.zeros(2500)
        ods[f"magnetics.flux_loop.{index}.flux.time"] = time
    write_validity(ods, "magnetics.flux_loop.0.flux", [0] * 2500, scalar=0)
    write_validity(ods, "magnetics.flux_loop.1.flux", [-2] * 2500, scalar=-2)
    # flux loop 1 -> legacy index 1 + nbprobe(4) = 5, one-based 6
    assert MODULE._condemned_by_diagnostics_stage(ods) == [6]
