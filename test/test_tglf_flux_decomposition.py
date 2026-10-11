"""TGLF flux = QL weight x saturated intensity x SAT3 amplitude factor, gated by a rebuild (#1900).

Fixtures are four configurations of one real state and surface from the #1482 sensitivity
campaign (39915 @ 0.317 s, magnetics lineage, r/a 0.7, GACODE b493397): SAT0 and SAT1
electrostatic, SAT2 and SAT3 with A_parallel. The decomposition is accepted only when
it rebuilds TGLF's own ``sum_flux_spectrum``. The tests below show that the gate passes
on all four and that it refuses the factorisation once the SAT3 factor is left out.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.tglf.outputs import INTENSITY_MOMENTS, TglfOutputs, collect_tglf_outputs
from vaft.code.gacode.tglf.spectral import (
    RECONSTRUCTION_RTOL,
    closure_split,
    flux_decomposition,
    ky_integration_weights,
    sat3_qla_factors,
)

DATA = Path(__file__).parent / "data" / "gacode"
CONFIGS = ("sat0-es", "sat1-es", "sat2-em-bper", "sat3-em-bper")


@pytest.fixture(scope="module")
def runs():
    return {c: collect_tglf_outputs(DATA / f"tglf_vest_39915_r0.70_{c}") for c in CONFIGS}


@pytest.mark.parametrize("config", CONFIGS)
def test_the_factors_rebuild_tglfs_flux_spectrum(runs, config):
    found = flux_decomposition(runs[config])
    assert found.validated, found.reason
    assert found.relative_error < 1e-12          # exact up to TGLF's printed digits
    # The ky-integrated, mode-summed totals are TGLF's gbflux (printed to 5 digits).
    totals = np.nansum(found.integrated(), axis=1)
    for index, quantity in ((0, "particle"), (1, "energy")):
        assert np.allclose(totals[:, index], runs[config].gbflux[quantity], rtol=1e-4, atol=1e-6)


def test_without_the_sat3_amplitude_factor_the_rebuild_is_refused(runs):
    """The 0.67-0.83 ratio reported on #1900 was this factor missing, and the gate catches it."""
    sat3 = runs["sat3-em-bper"]
    as_sat2 = dataclasses.replace(sat3, saturation_parameters={**sat3.saturation_parameters, "SAT_RULE": 2})
    found = flux_decomposition(as_sat2)
    assert found.status == "reconstruction_mismatch"
    assert found.weight is None and found.flux is None
    assert found.relative_error > 0.1


def test_a_wrong_ky_grid_model_is_refused_not_absorbed(runs):
    found = flux_decomposition(runs["sat2-em-bper"], kygrid_model=0)
    assert found.status == "reconstruction_mismatch"


def test_sat3_amplitude_factors_follow_the_mode_transition_function(runs):
    qla, x = sat3_qla_factors(runs["sat3-em-bper"])
    assert qla.shape == (5, 2) and x.shape == (2,)
    assert np.all((qla[0] >= 1.2 - 1e-12) & (qla[0] <= 2.2 + 1e-12))     # particle: 2 x [0.6, 1.1]
    assert np.all((qla[1] >= 1.2 - 1e-12) & (qla[1] <= 1.5 + 1e-12))     # energy: 2 x [0.6, 0.75]
    assert np.allclose(qla[2:], 1.6)
    assert sat3_qla_factors(runs["sat2-em-bper"]) is not None             # formable, not applied
    assert np.allclose(flux_decomposition(runs["sat2-em-bper"]).qla, 1.0)


def test_ky_weights_integrate_like_the_writer():
    ky = np.array([0.1, 0.2, 0.4, 0.8])
    w = ky_integration_weights(ky)
    assert w[1, 0] == pytest.approx(0.1) and w[0, 0] == 0.0
    assert np.allclose(w.sum(axis=0)[1:], np.diff(ky))      # dky0 + dky1 = interval width
    linear = ky_integration_weights(ky, kygrid_model=0)
    assert np.all(linear[0] == 0.0) and np.allclose(linear[1], 0.1)


def test_missing_inputs_are_reported_as_unavailable(runs):
    stripped = dataclasses.replace(runs["sat0-es"], ql_flux_spectrum=None)
    found = flux_decomposition(stripped)
    assert found.status == "unavailable" and "ql_flux_spectrum" in found.reason


def test_closure_split_is_exact_for_runs_with_shared_linear_physics(runs):
    a, b = runs["sat2-em-bper"], runs["sat3-em-bper"]
    split = closure_split(a, b)
    assert split.status == "split" and split.linear_max_rel_diff == 0.0
    native = np.nansum(b.sum_flux_spectrum, axis=2) - np.nansum(a.sum_flux_spectrum, axis=2)
    assert np.allclose(split.total[..., :native.shape[-1]], native, rtol=1e-10, atol=1e-12)
    assert np.allclose(split.intensity_term + split.qla_term, split.total, rtol=1e-12, atol=1e-12)
    assert np.any(np.abs(split.qla_term[..., 1]) > 0)      # SAT3's factor does contribute


def test_sat0_to_sat1_at_gyro_units_changes_only_the_intensity(runs):
    split = closure_split(runs["sat0-es"], runs["sat1-es"])
    assert split.status == "split"
    assert np.all(split.qla_term == 0.0)
    assert np.allclose(split.intensity_term, split.total)


def test_sat1_to_sat2_is_not_split(runs):
    """The presets change XNU_MODEL, WDIA_TRAPPED and the ky grid: the linear physics differs."""
    split = closure_split(runs["sat1-es"], runs["sat2-em-bper"])
    assert split.status == "linear_differs" and split.total is None


def test_ql_intensity_times_the_potential_intensity_is_the_written_intensity(runs):
    for run in runs.values():
        assert run.ql_intensity_spectrum.shape == (3, 21, 2, len(INTENSITY_MOMENTS))
        rebuilt = run.ql_intensity_spectrum * run.field_spectrum[None, :, :, 1, None]
        assert np.allclose(rebuilt, run.intensity_spectrum, rtol=1e-12, atol=0.0)


def test_ql_intensity_round_trips(runs, tmp_path):
    back = TglfOutputs.read_json(runs["sat3-em-bper"].write_json(tmp_path / "out.json"))
    assert np.array_equal(back.ql_intensity_spectrum, runs["sat3-em-bper"].ql_intensity_spectrum)
    assert RECONSTRUCTION_RTOL == 1e-6


# --------------------------------------------------------------------------- the pair product
def _sensitivity_module():
    import importlib.util
    import sys

    path = Path(__file__).parents[1] / "workflow" / "transport_atlas" / "build_sensitivity.py"
    spec = importlib.util.spec_from_file_location("transport_atlas_sensitivity_1900", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree(root, config, run):
    """A run_tglf.py-shaped tree whose outputs.json predates the spectra (pre-#1591)."""
    import json
    import shutil

    sat, field = int(config[3]), config.split("-", 1)[1]
    params = {"sat_rule": sat, "use_bper": field != "es", "use_bpar": False}
    tree = root / f"tglf-{config}" / "39915" / "magnetics" / "00317"
    surface = tree / "r0.70"
    shutil.copytree(DATA / f"tglf_vest_39915_r0.70_{config}", surface)
    legacy = run.to_dict()
    for key in ("field_spectrum", "intensity_spectrum", "ql_flux_spectrum", "ql_intensity_spectrum",
                "saturation_parameters"):
        legacy[key] = None
    legacy["schema_version"] = 2
    (surface / "outputs.json").write_text(json.dumps(legacy))
    (tree / "state.json").write_text(json.dumps({
        "shot": 39915, "time_efit_s": 0.317, "efit_lineage": "magnetics", "efit_quality": "good",
        "ti_lineage": "ti_eq_te_assumed", "state_identity": "s39915", "tglf_config": f"tglf-{config}",
        "tglf_parameters": params,
        "surfaces": [{"r_over_a": 0.7, "status": "solved", "run_identity": config}]}))


def test_the_pair_product_splits_only_pairs_with_shared_linear_physics(runs, tmp_path):
    import csv
    import json

    module = _sensitivity_module()
    for config, run in runs.items():
        _tree(tmp_path / "runs", config, run)
    assert module.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")]) == 0
    with open(tmp_path / "out" / "sensitivity.csv", newline="") as handle:
        rows = {r["tglf_config"]: r for r in csv.DictReader(handle)}
    assert {r["native_source"] for r in rows.values()} == {"native_files"}
    assert {r["flux_reconstruction"] for r in rows.values()} == {"validated"}
    assert rows["tglf-sat2-em-bper"]["wdia_trapped_effective"] == "1.0"
    assert rows["tglf-sat1-es"]["xnu_model_effective"] == "2"
    with open(tmp_path / "out" / "sat_pairs.csv", newline="") as handle:
        pairs = {(r["pair"], r["field_model"]): r for r in csv.DictReader(handle)}
    # The fixtures hold SAT0/1 ES and SAT2/3 EM, so only these two pairs can form.
    assert set(pairs) == {("sat0-sat1", "es"), ("sat2-sat3", "em-bper")}
    closure = pairs[("sat2-sat3", "em-bper")]
    assert closure["comparison"] == "quasilinear_saturation_closure"
    assert closure["split_status"] == "split" and closure["linear_identical"] == "True"
    change = float(closure["qe_gb_change"])
    assert change == pytest.approx(float(closure["qe_gb_b"]) - float(closure["qe_gb_a"]), rel=1e-4)
    assert float(closure["qe_gb_intensity_term"]) + float(closure["qe_gb_qla_term"]) == pytest.approx(change)
    assert float(pairs[("sat0-sat1", "es")]["qe_gb_qla_term"]) == 0.0
    schema = json.loads((tmp_path / "out" / "schema.json").read_text())
    assert "not a pure" in schema["sat_pair_interpretation"]["quasilinear_saturation_closure"]
    with open(tmp_path / "out" / "linear_spectra.csv", newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 4 * 21 * 2


def test_a_combined_pair_is_reported_but_not_split(runs, tmp_path):
    module = _sensitivity_module()
    sat1 = runs["sat1-es"]
    # SAT2 ES is not a fixture; the SAT2 EM run relabelled ES has SAT2's linear physics,
    # which is all the combined pair needs to show it is never split.
    _tree(tmp_path, "sat1-es", sat1)
    _tree(tmp_path, "sat2-em-bper", runs["sat2-em-bper"])
    rows = module.build_rows(tmp_path)
    for row in rows:
        row["field_model"] = "es"
    (pair,) = module.build_sat_pairs(rows)
    assert pair["pair"] == "sat1-sat2" and pair["comparison"] == "combined_model"
    assert pair["split_status"] == "linear_differs" and pair["linear_identical"] is False
    assert "qe_gb_change" not in pair
    assert pair["gamma_max_a"] is not None and pair["gamma_max_b"] is not None


def _synthetic_sat3(x_target, *, zero_ions=False):
    """One mode, kinetic electrons + one ion, weights chosen so x hits ``x_target`` at kymax."""
    from types import SimpleNamespace

    ky = np.array([0.2, 0.4, 0.6, 0.8])
    w = np.zeros((2, 1, 1, ky.size, 5))
    w[1, 0, 0, :, :] = 0.0 if zero_ions else 1.0
    # Electron/ion energy-weight ratio linear in ky, so TGLF's interpolation is exact.
    w[0, 0, 0, :, 1] = x_target + (ky - 0.5)
    w[0, 0, 0, :, 0] = 1.0
    return SimpleNamespace(ql_flux_spectrum=w, ky_spectrum=ky, spectral_species=(1, 2),
                           saturation_parameters={"SAT_RULE": 3, "kymax_out": 0.5})


@pytest.mark.parametrize("x", [0.7, 0.85, 0.9, 0.95, 1.1])
def test_sat3_factor_interpolates_x_at_kymax_and_follows_the_transition(x):
    qla, xs = sat3_qla_factors(_synthetic_sat3(x))
    assert xs[0] == pytest.approx(x, abs=1e-12)
    blend = min(max((x - 0.8) / 0.2, 0.0), 1.0)
    assert qla[0, 0] == pytest.approx(2 * (1.1 * (1 - blend) + 0.6 * blend))
    assert qla[1, 0] == pytest.approx(2 * (0.75 * (1 - blend) + 0.6 * blend))


def test_sat3_factor_falls_back_to_x_half_when_the_ion_sum_vanishes():
    qla, xs = sat3_qla_factors(_synthetic_sat3(0.9, zero_ions=True))
    assert xs[0] == 0.5 and qla[1, 0] == pytest.approx(1.5) and qla[0, 0] == pytest.approx(2.2)


def test_a_missing_or_zero_native_flux_is_not_validated(runs):
    run = runs["sat0-es"]
    missing = flux_decomposition(dataclasses.replace(run, sum_flux_spectrum=run.sum_flux_spectrum * np.nan))
    assert missing.status == "unavailable"
    zeroed = run.sum_flux_spectrum.copy()
    zeroed[..., 1] = 0.0                 # TGLF "says" zero energy flux; the rebuild does not
    assert flux_decomposition(dataclasses.replace(run, sum_flux_spectrum=zeroed)).status == "reconstruction_mismatch"
