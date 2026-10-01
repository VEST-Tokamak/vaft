"""EFIT's reported Ip chi-square carries the prescribed vessel current.

EFIT fits the Ip row as a plasma-only current, then reports a chi-square whose
model adds ``sum(VCURRT)`` for any ``IVESEL > 0``.  VEST's plasma current is the
inner Rogowski, which links no vessel current, so the reported value is
``(sum(VCURRT)/sigma_Ip)**2``.  These tests pin the recomputation and the rule
that a chi-square normalized by a legacy sigma is not graded.

Every fixture has two slices with different vessel currents: a single slice
cannot tell a per-slice correction from a constant one.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from omas import ODS

from vaft.omas.efit_quality import (
    IP_CHI_SQUARED_CONVENTION,
    constraint_uncertainty_model,
    convergence_metrics,
    fit_quality_metrics,
    ip_chi_squared_record,
)

SIGMA_IP = 1.0e4
# (measured Ip, reconstructed plasma-only Ip, prescribed sum(VCURRT)) per slice.
SLICES = (
    (125_830.6, 125_830.6001, 131_574.9),
    (79_931.1, 79_931.0999, 61_858.5),
)


def _reported(measured: float, reconstructed: float, vessel: float) -> float:
    """What EFIT's chisqr writes: the vessel current added to the model."""
    return ((measured - reconstructed - vessel) / SIGMA_IP) ** 2


def _ods(*, uncertainty_model: str | None = None, with_afile: bool = True) -> ODS:
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41672
    ods["equilibrium.time"] = [0.331, 0.342]
    for index, (measured, reconstructed, vessel) in enumerate(SLICES):
        root = f"equilibrium.time_slice.{index}"
        ods[f"{root}.time"] = ods["equilibrium.time"][index]
        ods[f"{root}.global_quantities.ip"] = reconstructed
        ip = f"{root}.constraints.ip"
        ods[f"{ip}.measured"] = measured
        ods[f"{ip}.reconstructed"] = reconstructed
        ods[f"{ip}.measured_error_upper"] = SIGMA_IP
        ods[f"{ip}.weight"] = 1.0 / SIGMA_IP
        ods[f"{ip}.chi_squared"] = _reported(measured, reconstructed, vessel)
        # One magnetic channel, fitted and tiny, as under legacy weighting.
        probe = f"{root}.constraints.bpol_probe.0"
        ods[f"{probe}.measured"] = 0.05
        ods[f"{probe}.reconstructed"] = 0.035
        ods[f"{probe}.weight"] = 1.0e-3
        ods[f"{probe}.chi_squared"] = (0.015 * 1.0e-3) ** 2
        ods[f"{probe}.source"] = "bpol_probe_ch0"
        ods[f"{root}.convergence.iterations_n"] = 11
        parameters = f"equilibrium.code.parameters.time_slice.{index}"
        ods[f"{parameters}.in1.ivesel"] = 1
        ods[f"{parameters}.in1.mxiter"] = -100
        ods[f"{parameters}.in1.error"] = 1e-5
        if with_afile:
            aeqdsk = f"{parameters}.aeqdsk"
            ods[f"{aeqdsk}.jflag"] = 0 if _reported(measured, reconstructed, vessel) >= 80 else 1
            ods[f"{aeqdsk}.lflag"] = 1 if _reported(measured, reconstructed, vessel) >= 80 else 0
            ods[f"{aeqdsk}.chisq"] = _reported(measured, reconstructed, vessel) + 1e-9
            ods[f"{aeqdsk}.ipmhd"] = reconstructed
            ods[f"{aeqdsk}.terror"] = 5e-3
    if uncertainty_model is not None:
        ods["equilibrium.code.parameters.uncertainty_model"] = uncertainty_model
    return ods


def test_the_recomputed_value_is_the_plasma_only_residual_and_keeps_efits():
    measured, reconstructed, vessel = SLICES[0]
    reported = _reported(measured, reconstructed, vessel)

    record = ip_chi_squared_record(measured, reconstructed, SIGMA_IP, reported)

    expected_z = (measured - reconstructed) / SIGMA_IP
    assert record["z"] == expected_z
    assert record["chi_squared"] == expected_z**2
    assert record["chi_squared"] < 1e-12
    assert record["chi_squared_efit_reported"] == reported
    # The term is what EFIT added: (sum VCURRT/sigma)^2 up to the cross term
    # 2*sum(VCURRT)*(m-r)/sigma^2, which the ~1e-4 A residual makes ~1e-9 relative.
    assert math.isclose(
        record["vessel_accounting_term"], (vessel / SIGMA_IP) ** 2, rel_tol=1e-6
    )
    assert math.isclose(
        record["vessel_accounting_term"] + record["chi_squared"], reported, rel_tol=1e-12
    )
    assert record["convention"] == IP_CHI_SQUARED_CONVENTION
    assert record["source"] == "recomputed_plasma_only"


def test_a_missing_sigma_is_unavailable_not_zero():
    record = ip_chi_squared_record(1.0e5, 1.0e5, float("nan"), 100.0)

    assert record["source"] == "unavailable"
    assert math.isnan(record["chi_squared"])
    assert math.isnan(record["vessel_accounting_term"])


def test_fit_quality_uses_the_recomputed_value_on_every_slice():
    ods = _ods()

    for index, (measured, reconstructed, vessel) in enumerate(SLICES):
        metrics = fit_quality_metrics(ods, time_slice=index)
        ip = metrics["scalars"]["ip"]
        assert ip["chi_squared"] < 1e-12
        # Sign and size come from the same residual.
        assert ip["z"] == (measured - reconstructed) / SIGMA_IP
        assert ip["chi_squared_efit_reported"] == _reported(measured, reconstructed, vessel)
        # The total no longer carries the vessel term, so Ip no longer owns it.
        assert metrics["chi_squared_total"] < 1e-9
        assert metrics["chi_squared_share"]["ip"] < 1e-3


def test_criterion_one_is_restated_beside_efits_verdict_not_instead_of_it():
    ods = _ods()

    first = convergence_metrics(ods, time_slice=0)
    second = convergence_metrics(ods, time_slice=1)

    # Slice 0: 131.6 kA of vessel current -> EFIT rejects on #1.
    assert first["verdict"]["accepted"] is False
    assert first["error"]["chi_squared_total"] > 80.0
    assert first["error"]["efit_chi_squared_includes_prescribed_vessel_current"] is True
    assert first["error"]["chi_squared_total_vessel_corrected"] < 1e-6
    assert first["error"]["criterion_1_vessel_corrected_passed"] is True
    # Slice 1: 61.9 kA -> EFIT already accepts; the correction agrees.
    assert second["verdict"]["accepted"] is True
    assert second["error"]["criterion_1_vessel_corrected_passed"] is True
    # The two slices differ, so the correction is per slice.
    assert (
        first["error"]["ip_chi_squared"]["vessel_accounting_term"]
        != second["error"]["ip_chi_squared"]["vessel_accounting_term"]
    )


def test_the_ip_self_consistency_reads_the_afile_plasma_only_current():
    metrics = convergence_metrics(_ods(), time_slice=0)

    sources = metrics["self_consistency"]["ip_sources"]
    assert "aeqdsk_ipmhd" in sources
    assert np.isfinite(sources["aeqdsk_ipmhd"])


def test_the_uncertainty_model_is_read_only_from_an_explicit_record():
    assert constraint_uncertainty_model(_ods()) == "unknown"
    assert constraint_uncertainty_model(_ods(uncertainty_model="legacy_weight")) == "legacy_weight"

    product = _ods()
    product["equilibrium.code.parameters"] = json.dumps(
        {"efit_collection": {}, "uncertainty_model": "standard_deviation"}
    )
    assert constraint_uncertainty_model(product) == "standard_deviation"

    unrecorded_product = _ods()
    unrecorded_product["equilibrium.code.parameters"] = json.dumps({"efit_collection": {}})
    assert constraint_uncertainty_model(unrecorded_product) == "unknown"


def test_sigma_normalized_grades_are_not_available_unless_sigma_is_statistical():
    from vaft.validation.equilibrium import validate_magnetic_fit

    for model in (None, "legacy_weight"):
        results = validate_magnetic_fit(_ods(uncertainty_model=model), time_slice=0)
        for check in ("bpol_probe", "ip", "global"):
            assert results[check]["status"] == "not_available", (model, check)
        # The physical-unit residual is still reported.
        assert math.isclose(
            results["ip"]["residual"], SLICES[0][0] - SLICES[0][1], rel_tol=1e-9
        )

    graded = validate_magnetic_fit(_ods(uncertainty_model="standard_deviation"), time_slice=0)
    assert graded["ip"]["status"] != "not_available"
    assert graded["ip"]["chi_squared"] < 1e-12


def test_reliability_rows_carry_the_recomputed_ip_chi_squared():
    from vaft.database._summary import extract_efit_magnetic_reliability

    ods = _ods()
    for index in range(len(SLICES)):
        ods[f"equilibrium.time_slice.{index}.constraints.freedom_degrees_n"] = 1
        ods[f"equilibrium.time_slice.{index}.constraints.chi_squared_reduced"] = (
            _reported(*SLICES[index]) + 1e-9
        )

    rows = pd.DataFrame(extract_efit_magnetic_reliability(ods, 41672))
    ip_rows = rows[rows["measurement_type"] == "ip"].sort_values("eq_index")

    assert len(ip_rows) == 2
    assert (ip_rows["chi_squared"] < 1e-12).all()
    # The slice aggregate loses the same vessel term, per slice.
    assert (ip_rows["chi_squared_reduced"] < 1e-6).all()


# --------------------------------------------------------------------------- #
# The writers say what the reader reads (cold review 0.8.0 efit-and-magnetics F1)
# --------------------------------------------------------------------------- #
def _efit_collection_parameters():
    """The pipeline-1 EFIT collection stage's payload builder, imported by path."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "workflow" / "automatic_pipeline_1_routine_data_processing" / "generate_efit_ods.py"
    )
    spec = importlib.util.spec_from_file_location("_generate_efit_ods_for_uncertainty_model", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.efit_collection_parameters


def _pipeline1_product(preset_name):
    from vaft.code.efit.presets import efit_preset

    build = _efit_collection_parameters()
    common = dict(status="success", slice_statuses=[], mapping_diagnostics=[],
                  artifact_hashes={}, artifact_manifest={})
    if preset_name is None:
        payload = build(**common)
    else:
        # What the k-file stage leaves beside its manifest: the record plus the floor changes.
        record = {**efit_preset(preset_name).record(), "sigma_floor_changes": []}
        payload = build(**common, efit_preset=record)
    ods = _ods()
    ods["equilibrium.code.parameters"] = payload
    return ods, json.loads(payload)


def test_a_pipeline_1_statistical_product_is_graded():
    """The preset stated `standard_deviation`; the reader used to answer "unknown"."""
    from vaft.validation.equilibrium import validate_magnetic_fit

    ods, decoded = _pipeline1_product("statistical_891")
    assert decoded["uncertainty_model"] == "standard_deviation"
    assert constraint_uncertainty_model(ods) == "standard_deviation"
    results = validate_magnetic_fit(ods, time_slice=0)
    for check in ("bpol_probe", "ip", "global"):
        assert results[check]["status"] != "not_available", check
        assert results[check]["uncertainty_model"] == "standard_deviation"


def test_a_pipeline_1_routine_preset_product_says_legacy_weight_and_is_not_graded():
    from vaft.validation.equilibrium import validate_magnetic_fit

    ods, decoded = _pipeline1_product("routine")
    assert decoded["uncertainty_model"] == "legacy_weight"
    assert constraint_uncertainty_model(ods) == "legacy_weight"
    assert validate_magnetic_fit(ods, time_slice=0)["ip"]["status"] == "not_available"


def test_a_pipeline_1_product_without_a_preset_record_stays_unknown():
    """No preset, no statement: the routine payload is what it was before presets."""
    ods, decoded = _pipeline1_product(None)
    assert "uncertainty_model" not in decoded and "efit_preset" not in decoded["efit_collection"]
    assert constraint_uncertainty_model(ods) == "unknown"


def test_a_product_written_before_the_key_existed_is_read_from_its_preset_record():
    """0.8.0-era payloads carry the preset record but no top-level key."""
    ods, decoded = _pipeline1_product("statistical_891")
    del decoded["uncertainty_model"]
    ods["equilibrium.code.parameters"] = json.dumps(decoded, sort_keys=True)
    assert constraint_uncertainty_model(ods) == "standard_deviation"

    # A record that does not carry the mode is not a statement either.
    decoded["efit_collection"]["efit_preset"]["scientific"]["constraints"].pop("uncertainty_mode")
    ods["equilibrium.code.parameters"] = json.dumps(decoded, sort_keys=True)
    assert constraint_uncertainty_model(ods) == "unknown"


def test_a_pipeline_2_kinetic_product_states_its_uncertainty_model():
    """Pipeline 2 put the preset record in the manifest only; the product said nothing."""
    from vaft.omas.vest_upstream import kinetic_efit_parameters

    # No preset named is the default, statistical_891, since 2026-10-01.
    for preset, expected in ((None, "standard_deviation"), ("routine", "legacy_weight"),
                             ("statistical_891", "standard_deviation")):
        ods = _ods()
        ods["equilibrium.code.parameters"] = kinetic_efit_parameters("kinetic_efit", preset)
        assert constraint_uncertainty_model(ods) == expected, preset
        decoded = json.loads(ods["equilibrium.code.parameters"])
        assert decoded["kinetic_efit"]["stage"] == "kinetic_efit"
        assert (decoded["kinetic_efit"]["efit_preset"] or {}).get("name") == (preset or "statistical_891")
