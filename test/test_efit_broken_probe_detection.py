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
