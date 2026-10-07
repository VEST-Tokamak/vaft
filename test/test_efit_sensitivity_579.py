"""The #579 sensitivity-study scripts: report, model spread, products, Thomson peaking band."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "workflow" / "efit_uncertainty_calibration"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"sens579_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stub_criteria(verdicts):
    """A criteria module whose ``evaluate`` returns the verdict prepared for each setting."""
    module = types.SimpleNamespace(PASS="pass")

    def evaluate(record):
        good, admissible, consistent = verdicts[record["setting"]]
        return {"good": good, "physically_consistent": consistent,
                "verdicts": {"admissible": {"status": "pass" if admissible else "fail"},
                             "thomson": {"log_ratio": record.get("log_ratio")}}}

    module.evaluate = evaluate
    return module


WORKING = "p2f1_probe_x3.62_loop_x2.15_dia_x16_ip_x4_floor2pct_psiexit"
BASIS_12 = "p1f2_probe_x3.62_loop_x2.15_dia_x16_ip_x4_floor2pct_psiexit"
TWO_AXES = "p1f2_probe_x7.24_loop_x2.15_dia_x16_ip_x4_floor2pct_psiexit"


def _record(setting, wmhd, *, time_ms=318):
    return {"shot": 39916, "time_ms": time_ms, "setting": setting, "converged": True, "log_ratio": -0.5,
            "scalars": {"wmhd": wmhd, "betap": 0.5, "li": 1.0, "q95": 6.0}, "fit": {}}


def test_setting_names_parse_into_the_ensemble_axes():
    report = _load("sensitivity_report")
    assert report.axes(WORKING) == {"basis": (2, 1), "dia": 16.0, "probe": 3.62, "loop": 2.15}
    assert report.axes("p1f1_probe_x1.81_loop_x2.15_dia_off_ip_x4")["dia"] is None
    assert report.axes("routine") == {}
    assert report.is_working(report.axes(WORKING)) and not report.is_working({})


def test_the_marginal_effect_moves_one_axis_at_a_time(monkeypatch):
    report = _load("sensitivity_report")
    verdicts = {WORKING: (True, True, True), BASIS_12: (True, True, False), TWO_AXES: (True, True, True)}
    monkeypatch.setattr(report, "_criteria", lambda: _stub_criteria(verdicts))
    out = report.report([_record(WORKING, 1000.0), _record(BASIS_12, 1100.0), _record(TWO_AXES, 5000.0)])
    basis = out["marginal"]["basis"]
    assert basis["(1, 2)"]["wmhd"]["median"] == pytest.approx(1.1)
    # TWO_AXES changes basis *and* probe sigma, so it enters neither marginal.
    assert basis["(1, 2)"]["wmhd"]["n"] == 1
    assert "7.24" not in out["marginal"]["probe"]
    configuration = {c["setting"]: c for c in out["configurations"]}
    assert configuration[BASIS_12]["good_and_consistent"] == 0
    assert configuration[WORKING]["good_and_consistent"] == 1
    assert out["slices"][0]["working"]["p_over_p_e"] == pytest.approx(math.exp(0.5))


def test_model_spread_sigma_log_and_the_bimodal_flag(tmp_path, monkeypatch):
    spread = _load("model_spread")
    stats = spread._stats([1.0, 2.0, 4.0])
    assert stats["n"] == 3 and stats["sigma_log"] > 0
    assert math.isnan(spread._stats([-1.0, 1.0])["sigma_log"])  # a non-positive bound has no log width
    verdicts = {WORKING: (True, True, None), BASIS_12: (True, True, None)}
    stub = _stub_criteria(verdicts)
    monkeypatch.setattr(spread, "_criteria", lambda: stub)
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"records": [
        {**_record(WORKING, 1000.0), "scalars": {"wmhd": 1000.0, "betap": 0.5, "li": 1.0}},
        {**_record(BASIS_12, 4000.0), "scalars": {"wmhd": 4000.0, "betap": 0.5, "li": 1.0}},
    ]}), encoding="utf-8")
    (row,) = spread.build([table], bimodal_ratio=2.0)
    assert row["time_efit_s"] == 0.318 and row["efit_lineage"] == "magnetics"
    assert row["w_mhd_J_viable_n"] == 2
    assert row["viable_bimodal"]  # q84/q16 of {1000, 4000} is about 3, well above 2


def test_profile_peaking_is_the_axis_value_over_the_integrated_mean(monkeypatch):
    """Same definition as sensitivity_pressure's peaking_e: p(0) / (W / 1.5 / V)."""
    band = _load("thomson_peaking_band")
    import vaft.validation.kinetic_state as kinetic_state

    seen = {}

    def fake_ratio(equilibrium, index, rho, p_e):
        seen.update(rho=np.asarray(rho), p_e=np.asarray(p_e))
        return {"available": True, "w_e_j": 1.5 * 0.5 * 2.0, "volume_m3": 2.0}  # <p>_V = 0.5

    monkeypatch.setattr(kinetic_state, "integrated_pressure_ratio", fake_ratio)
    assert band.profile_peaking(object(), 0, lambda x: 1.0 - x ** 2) == pytest.approx(2.0)
    assert seen["rho"][0] == 0.0 and seen["rho"][-1] == 1.0
    assert math.isnan(band.profile_peaking(object(), 0, lambda x: np.full_like(x, np.nan)))
    monkeypatch.setattr(kinetic_state, "integrated_pressure_ratio", lambda *a, **k: {"available": False})
    assert math.isnan(band.profile_peaking(object(), 0, lambda x: 1.0 - x ** 2))


def test_the_band_reports_a_missing_equilibrium_slice_instead_of_failing(monkeypatch):
    band = _load("thomson_peaking_band")
    import vaft.validation.kinetic_state as kinetic_state

    def missing(*_args, **_kwargs):
        raise LookupError("no slice within 0.0005 s")

    monkeypatch.setattr(kinetic_state, "slice_at_time", missing)
    out = band.peaking_band(object(), object(), 0.318, draws=3)
    assert out["n_draws"] == 0 and math.isnan(out["central"])
    assert out["reason"].startswith("no equilibrium slice")


def test_compose_products_records_its_sources(tmp_path, monkeypatch):
    compose = _load("compose_products")
    from omas import ODS
    import vaft.database.composition as composition

    shot = 39916
    for stage, name in (("diagnostics", "diagnostics"), ("eddy", "eddy")):
        directory = tmp_path / "filedb" / "omas" / stage / str(shot) / "output"
        directory.mkdir(parents=True)
        (directory / f"{name}.json.gz").write_bytes(b"stand-in")

    def fake_compose(*, diagnostics, eddy, eddy_manifest):
        ods = ODS()
        ods["dataset_description.data_entry.pulse"] = shot
        return ods, {"diagnostics": str(diagnostics), "eddy": str(eddy), "manifest": eddy_manifest}

    monkeypatch.setattr(composition, "compose_stage_products", fake_compose)
    entry = compose.compose(tmp_path / "filedb", shot, tmp_path / "products")
    assert Path(entry["product"]).is_file() and len(entry["sha256"]) == 64
    assert entry["diagnostics"]["path"].endswith("diagnostics.json.gz")
    assert entry["composition"]["manifest"] is None  # no eddy manifest on disk
    # The record says the diagnostics-hash check could not run, rather than
    # silently looking like a composed pair (cold review 0.8.0 delta-absorb-17 F6).
    assert entry["eddy_manifest"]["present"] is False
    assert entry["eddy_manifest"]["diagnostics_hash_check"].startswith("skipped")
    assert entry["eddy_manifest"]["path"].endswith("manifest.json")

    # Paths come from the FileDB resolvers, not a hand-built layout.
    from vaft.database.filedb import FileDB

    fdb = FileDB(tmp_path / "filedb")
    assert Path(entry["diagnostics"]["path"]) == fdb.omas_product("diagnostics", shot=shot)
    assert Path(entry["eddy"]["path"]) == fdb.omas_product("eddy", shot=shot)
    assert Path(entry["eddy_manifest"]["path"]) == fdb.omas_manifest("eddy", shot=shot)

    manifest = fdb.omas_manifest("eddy", shot=shot)
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    entry = compose.compose(tmp_path / "filedb", shot, tmp_path / "products")
    assert entry["composition"]["manifest"] == str(manifest)  # json round-trip
    assert entry["eddy_manifest"] == {"path": str(manifest), "present": True,
                                      "diagnostics_hash_check": "performed"}


def test_the_bootstrap_perturbs_what_the_fit_reads_and_restores_it(monkeypatch):
    """Each draw reaches the fit's channel values; the ODS is unchanged afterwards (cold review)."""
    band = _load("thomson_peaking_band")
    from omas import ODS
    import vaft.process.profile as profile

    thomson = ODS()
    thomson["thomson_scattering.time"] = np.array([0.317, 0.318])
    for i in range(3):
        channel = thomson[f"thomson_scattering.channel.{i}"]
        channel["t_e.data"] = np.array([10.0, 100.0 + i])
        channel["n_e.data"] = np.array([1e18, 1e19])
        channel["t_e.data_error_upper"] = np.array([1.0, 10.0])
        channel["n_e.data_error_upper"] = np.array([1e17, 1e18])
    before = [(np.array(thomson[f"thomson_scattering.channel.{i}.t_e.data"]),
               np.array(thomson[f"thomson_scattering.channel.{i}.n_e.data"])) for i in range(3)]
    seen, calls = [], {"n": 0}

    def fake_fit(ods, time_ms, mapped, **_kwargs):
        calls["n"] += 1
        seen.append(profile._thomson_points(ods, time_ms, 1.0)["t_e"].copy())
        if calls["n"] == 3:
            raise RuntimeError("Te fit failed at every order")  # a refused draw
        return (lambda x: np.ones_like(x), lambda x: np.ones_like(x))

    monkeypatch.setattr(profile, "equilibrium_mapping_thomson_scattering", lambda *a, **k: None)
    monkeypatch.setattr(profile, "profile_fitting_thomson_scattering", fake_fit)
    monkeypatch.setattr(band, "profile_peaking", lambda equilibrium, index, pressure: 1.0)
    out = band.peaking_band(thomson, object(), 0.318, draws=4, equilibrium_index=0)
    assert out["channels"] == 3 and out["central"] == 1.0
    assert out["n_draws"] == 3  # one of the four draws was refused, and is counted out
    np.testing.assert_array_equal(seen[0], [100.0, 101.0, 102.0])  # the central fit sees the measurement
    assert not np.allclose(seen[1], seen[0])  # a draw sees perturbed values
    for i, (t_e, n_e) in enumerate(before):
        np.testing.assert_array_equal(thomson[f"thomson_scattering.channel.{i}.t_e.data"], t_e)
        np.testing.assert_array_equal(thomson[f"thomson_scattering.channel.{i}.n_e.data"], n_e)

    def failing(*_a, **_k):
        raise RuntimeError("no fit")

    monkeypatch.setattr(profile, "profile_fitting_thomson_scattering", failing)
    out = band.peaking_band(thomson, object(), 0.318, draws=2, equilibrium_index=0)
    assert out["n_draws"] == 0 and "no fit" in out["reason"]
    np.testing.assert_array_equal(thomson["thomson_scattering.channel.0.t_e.data"], before[0][0])
