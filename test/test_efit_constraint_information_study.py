import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "workflow"
    / "efit_constraint_information"
    / "constraint_information_study.py"
)
FIXTURE = ROOT / "test" / "data" / "efit_constraint_study_fixture.json"


def _study():
    spec = importlib.util.spec_from_file_location(
        "efit_constraint_information_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Variable:
    def __init__(self, values):
        self.data = np.asarray(values, dtype=float)


def test_complete_and_confirmation_matrices_cover_the_declared_experiment():
    study = _study()
    complete = {item.name for item in study.selected_variants("complete")}
    confirmation = {item.name for item in study.selected_variants("confirmation")}

    assert len(complete) == 36
    assert confirmation == study.CONFIRMATION_NAMES
    assert {
        "zero_vcurrt",
        "no_passive_response",
        "no_pf_relations",
        "no_pf_penalty_or_relations",
        "flux_loops_x100",
        "bpol_probes_x100",
        "diamagnetic_flux_x100",
        "diamagnetic_flux_x1000",
        "diamagnetic_flux_x10000",
        "ip_only",
        "flux_loops_plus_bpol",
    } <= complete
    assert {
        f"diamagnetic_flux_x{scale}" for scale in range(1_000, 10_001, 1_000)
    } <= complete


def test_family_chi_squared_sum_is_taken_directly_from_mfile_channels():
    study = _study()
    data = {
        "silopt": _Variable([1.0, 3.0, 9.0]),
        "csilop": _Variable([2.0, 1.0, 4.0]),
        "sigsil": _Variable([0.5, 0.5, 0.5]),
        "fwtsi": _Variable([2.0, 0.0, 0.25]),
        "saisil": _Variable([4.0, 99.0, 1.5625]),
    }

    result = study._family_record(data, "flux_loop", 10.0)

    assert result["active_channels"] == 2
    assert result["physical_residual"] == [1.0, -2.0, -5.0]
    assert result["chi2_sum"] == 104.5625
    assert result["chi2_recomputed_sum"] == 5.5625
    assert result["objective_scale"] == 10.0


def test_stored_ip_vcurrt_fixture_reproduces_reported_chipasma():
    study = _study()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    for record in fixture["ip_accounting"].values():
        result = study.ip_accounting_record(
            ip_measured=record["ip_measured"],
            ipmhd=record["ipmhd"],
            vessel_current_sum=record["vessel_current_sum"],
            sigma_ip=record["sigma_ip"],
            chipasma_reported=record["chipasma"],
        )
        assert result["prediction_relative_error"] < 5.0e-8


def test_stored_profile_study_pf_verification_has_only_roundoff_chifcc():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pf = fixture["pf_profile_model_verification"]

    assert pf["bit_identical_arrays"] == ["fccurt", "ccbrsp", "sigfcc", "fwtfc"]
    assert pf["41672"]["max_abs_chifcc_difference"] < 1.0e-17
    assert pf["39915"]["max_abs_chifcc_difference"] < 3.0e-21
    assert pf["41524"]["max_abs_chifcc_difference"] < 1.0e-21


def test_material_response_uses_the_documented_thresholds():
    study = _study()
    comparison = {
        "geometry": {
            "lcfs_rms_mm": {"median": 5.0},
            "absolute_relative_change": {
                "area": {"median": 0.0},
                "volume": {"median": 0.0},
            },
        },
        "acceptance_change_percentage_points": 0.0,
        "family_chi2_absolute_relative_change": {},
        "family_residual_rms_absolute_relative_change": {},
    }
    assert study.material_response(comparison)
