"""Thomson against EFIT pressure per slice, the atlas state key (#1430, #1454).

The equilibria here have several slices stored out of time order, because the
bug this layer exists to avoid -- pairing by index -- passes every one-slice test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omas import ODS

from vaft.validation import kinetic_state as ks
from vaft.validation.equilibrium import _thomson_pressure, thomson_pressure_samples

E = 1.602176634e-19
R0, A, KAPPA = 0.40, 0.25, 1.6
TIMES = (0.314, 0.312, 0.316, 0.313, 0.315)  # stored out of order on purpose


def _p0(time: float) -> float:
    """Each slice gets its own axis pressure so a wrong slice shows up."""
    return 1000.0 * (1.0 + 100.0 * (time - 0.312))


def _equilibrium(times=TIMES, *, phi=True, proxy=False, outline=True) -> ODS:
    ods = ODS(consistency_check=False)
    r = np.linspace(0.1, 0.8, 71)
    z = np.linspace(-0.6, 0.6, 81)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    psi_n = ((rm - R0) ** 2 + (zm / KAPPA) ** 2) / A ** 2
    level = np.linspace(0.0, 1.0, 41)
    ods["equilibrium.time"] = np.asarray(times)
    for i, t in enumerate(times):
        root = f"equilibrium.time_slice.{i}"
        ods[f"{root}.time"] = t
        ods[f"{root}.global_quantities.psi_axis"] = -0.02
        ods[f"{root}.global_quantities.psi_boundary"] = 0.03
        ods[f"{root}.profiles_1d.psi"] = -0.02 + 0.05 * level
        ods[f"{root}.profiles_1d.pressure"] = _p0(t) * (1.0 - level)
        if phi:
            ods[f"{root}.profiles_1d.phi"] = 0.01 * level ** 1.3
        if proxy:
            ods[f"{root}.profiles_1d.rho_tor_norm"] = np.sqrt(level)
        ods[f"{root}.profiles_2d.0.grid.dim1"] = r
        ods[f"{root}.profiles_2d.0.grid.dim2"] = z
        ods[f"{root}.profiles_2d.0.psi"] = -0.02 + 0.05 * psi_n
        if outline:
            theta = np.linspace(0, 2 * np.pi, 200)
            ods[f"{root}.boundary.outline.r"] = R0 + A * np.cos(theta)
            ods[f"{root}.boundary.outline.z"] = KAPPA * A * np.sin(theta)
    return ods


def _thomson(times=(0.310, 0.311, 0.312, 0.313, 0.314, 0.315, 0.316, 0.317), radii=(0.30, 0.36, 0.42, 0.48)) -> ODS:
    ods = ODS(consistency_check=False)
    ods["thomson_scattering.time"] = np.asarray(times)
    for c, radius in enumerate(radii):
        base = f"thomson_scattering.channel.{c}"
        ods[f"{base}.position.r"] = radius
        ods[f"{base}.position.z"] = 0.0
        # n_e rises with time so each sample is distinguishable
        ods[f"{base}.n_e.data"] = 1e18 * (1.0 + np.arange(len(times)))
        ods[f"{base}.n_e.data_error_upper"] = 1e17 * np.ones(len(times))
        ods[f"{base}.t_e.data"] = 50.0 * np.ones(len(times))
        ods[f"{base}.t_e.data_error_upper"] = 5.0 * np.ones(len(times))
    return ods


def test_state_time_rounds_to_the_contract_precision():
    assert ks.state_time(0.31600000001) == 0.316
    assert ks.LINEAGES == ("magnetics", "electron_kinetic")
    assert ks.QUALITIES == ("good", "admissible")


def test_slice_is_found_by_time_not_index():
    eq = _equilibrium()
    for t in TIMES:
        index, offset = ks.slice_at_time(eq, t)
        assert TIMES[index] == t and offset == 0.0
    index, offset = ks.slice_at_time(eq, 0.3154)
    assert TIMES[index] == 0.315 and offset == pytest.approx(-0.0004)


def test_slice_beyond_tolerance_is_refused():
    eq = _equilibrium()
    # cadence 1 ms -> tolerance max(0.5 ms, 1 ms) = 1 ms
    assert ks.default_tolerance(np.asarray(TIMES)) == pytest.approx(1e-3)
    ks.slice_at_time(eq, 0.3169)
    with pytest.raises(LookupError, match="beyond"):
        ks.slice_at_time(eq, 0.3171)
    with pytest.raises(LookupError):
        ks.slice_at_time(eq, 0.3165, tolerance_s=1e-4)


def test_matched_ratios_use_the_slice_and_sample_at_that_time():
    eq, ts = _equilibrium(), _thomson()
    for t in TIMES:
        row = ks.match_thomson_pressure(eq, ts, time_s=t)
        assert row["ts_status"] == "matched"
        assert row["time_efit_s"] == t and row["time_ts_s"] == pytest.approx(t)
        assert row["dt_ts_efit_s"] == pytest.approx(0.0, abs=1e-12)
        sample = int(round((t - 0.310) * 1e3))
        for ch in row["channels"]:
            psi_n = ((ch["r"] - R0) ** 2) / A ** 2
            assert ch["psi_norm"] == pytest.approx(psi_n, abs=2e-3)
            assert ch["p_recon"] == pytest.approx(_p0(t) * (1 - psi_n), rel=5e-3)
            assert ch["p_e"] == pytest.approx(E * 1e18 * (1 + sample) * 50.0)
            assert ch["r_p"] == pytest.approx(ch["p_recon"] / ch["p_e"])
            assert ch["n_e_error"] == 1e17 and ch["t_e_error"] == 5.0


def test_r_sum_is_the_quantity_the_criteria_band_grades():
    eq, ts = _equilibrium(), _thomson()
    for t in TIMES:
        row = ks.match_thomson_pressure(eq, ts, time_s=t)
        index = TIMES.index(t)
        graded = _thomson_pressure(eq, index, ts)
        assert row["r_sum"] == pytest.approx(1.0 / graded["sum_ratio"], rel=1e-12)
        assert row["log_ratio"] == pytest.approx(graded["log_ratio"], rel=1e-12)
        assert graded["channels_inside"] == row["points"]


def test_the_split_sampler_keeps_the_check_result():
    eq, ts = _equilibrium(), _thomson()
    graded = _thomson_pressure(eq, 2, ts)
    sampled = thomson_pressure_samples(eq, 2, ts)
    p_e = sum(c["p_e"] for c in sampled["channels"])
    p_r = sum(c["p_recon"] for c in sampled["channels"])
    assert graded["log_ratio"] == pytest.approx(math.log(p_e / p_r))
    assert graded["status"] in {"pass", "warn", "fail"}
    assert graded["time_offset"] == 0.0


def test_thomson_outside_tolerance_stays_as_unmatched():
    eq = _equilibrium()
    ts = _thomson(times=(0.300, 0.301))
    row = ks.match_thomson_pressure(eq, ts, time_s=0.314)
    assert row["ts_status"] == "unmatched"
    assert "beyond" in row["reason"]
    assert "r_sum" not in row
    assert row["dt_ts_efit_s"] == pytest.approx(-0.013)


def test_thomson_outside_the_plasma_is_invalid_not_unmatched():
    eq = _equilibrium()
    ts = _thomson(radii=(0.75, 0.78))
    row = ks.match_thomson_pressure(eq, ts, time_s=0.314)
    assert row["ts_status"] == "invalid"
    assert "LCFS" in row["reason"]


def test_explicit_ts_tolerance_applies_to_a_one_slice_equilibrium():
    eq = _equilibrium(times=(0.3145,))
    ts = _thomson()
    assert ks.match_thomson_pressure(eq, ts, time_s=0.3145)["ts_status"] == "matched"
    row = ks.match_thomson_pressure(eq, ts, time_s=0.3145, ts_tolerance_s=1e-4)
    assert row["ts_status"] == "unmatched"


def test_rho_comes_from_phi_and_never_from_the_proxy():
    eq = _equilibrium()
    coordinate = ks.rho_tor_norm_of(eq, 0)
    assert coordinate["coordinate"] == "rho_tor_norm" and coordinate["source"] == "phi"
    level = np.linspace(0.0, 1.0, 41)
    np.testing.assert_allclose(coordinate["rho_tor_norm"], level ** 0.65)

    proxied = ks.rho_tor_norm_of(_equilibrium(phi=False, proxy=True), 0)
    assert proxied["coordinate"] == "unavailable" and "proxy" in proxied["reason"]
    row = ks.match_thomson_pressure(_equilibrium(phi=False, proxy=True), _thomson(), time_s=0.314)
    assert row["rho_coordinate"] == "unavailable"
    assert all(math.isnan(c["rho_tor_norm"]) for c in row["channels"])
    assert all(np.isfinite(c["psi_norm"]) for c in row["channels"])


def test_integrated_ratio_recovers_a_uniform_ratio():
    eq = _equilibrium()
    index = TIMES.index(0.315)
    coordinate = ks.rho_tor_norm_of(eq, index)
    p_eq = _p0(0.315) * (1.0 - coordinate["psi_norm"])
    full = ks.integrated_pressure_ratio(eq, index, coordinate["rho_tor_norm"], p_eq / 2.0)
    assert full["available"] and full["cell_weights"] == "outline"
    assert full["ratio"] == pytest.approx(2.0, rel=1e-3)
    assert full["volume_m3"] == pytest.approx(2 * np.pi * R0 * np.pi * A * A * KAPPA, rel=0.03)
    span = ks.integrated_pressure_ratio(eq, index, coordinate["rho_tor_norm"], p_eq / 2.0,
                                        psi_norm_span=(0.2, 0.6))
    assert span["ratio"] == pytest.approx(2.0, rel=1e-3)
    assert span["volume_m3"] < full["volume_m3"]


def test_integrated_ratio_does_not_extrapolate_p_e():
    eq = _equilibrium()
    coordinate = ks.rho_tor_norm_of(eq, 0)
    p_eq = _p0(TIMES[0]) * (1.0 - coordinate["psi_norm"])
    keep = coordinate["rho_tor_norm"] <= 0.5
    inner = ks.integrated_pressure_ratio(eq, 0, coordinate["rho_tor_norm"][keep], p_eq[keep] / 3.0)
    full = ks.integrated_pressure_ratio(eq, 0, coordinate["rho_tor_norm"], p_eq / 3.0)
    assert inner["ratio"] == pytest.approx(3.0, rel=1e-3)
    assert inner["volume_m3"] < full["volume_m3"]


def test_core_profiles_pressure_is_matched_by_time():
    cp = ODS(consistency_check=False)
    cp["core_profiles.time"] = np.array([0.316, 0.312, 0.314])
    for j, t in enumerate((0.316, 0.312, 0.314)):
        root = f"core_profiles.profiles_1d.{j}"
        cp[f"{root}.time"] = t
        cp[f"{root}.grid.rho_tor_norm"] = np.linspace(0, 1, 5)
        cp[f"{root}.electrons.density_thermal"] = np.full(5, 1e18 * (j + 1))
        cp[f"{root}.electrons.temperature"] = np.full(5, 10.0)
    got = ks.core_profiles_electron_pressure(cp, time_s=0.312)
    assert got["available"] and got["time_s"] == 0.312
    np.testing.assert_allclose(got["p_e"], E * 2e18 * 10.0)
    missing = ks.core_profiles_electron_pressure(cp, time_s=0.320)
    assert not missing["available"] and "beyond" in missing["reason"]


def test_inputs_are_not_mutated():
    eq, ts = _equilibrium(), _thomson()
    before_eq, before_ts = set(eq.flat()), set(ts.flat())
    ks.match_thomson_pressure(eq, ts, time_s=0.313)
    coordinate = ks.rho_tor_norm_of(eq, 1)
    ks.integrated_pressure_ratio(eq, 1, coordinate["rho_tor_norm"], np.ones(41))
    ks.core_profiles_electron_pressure(eq, time_s=0.313)
    assert set(eq.flat()) == before_eq and set(ts.flat()) == before_ts


def test_per_slice_time_wins_over_a_stale_shared_time():
    eq = _equilibrium()
    eq["equilibrium.time"] = np.asarray(sorted(TIMES))  # stale: no longer the stored order
    for t in TIMES:
        index, _ = ks.slice_at_time(eq, t)
        assert TIMES[index] == t
    row = ks.match_thomson_pressure(eq, _thomson(), time_s=0.316)
    assert row["time_efit_s"] == 0.316 and row["dt_ts_efit_s"] == pytest.approx(0.0, abs=1e-12)


def test_phi_stored_edge_to_axis_keeps_rho_zero_on_axis():
    eq = _equilibrium(times=(0.314,))
    level = np.linspace(0.0, 1.0, 41)
    root = "equilibrium.time_slice.0.profiles_1d"
    eq[f"{root}.psi"] = (-0.02 + 0.05 * level)[::-1]
    eq[f"{root}.pressure"] = (_p0(0.314) * (1.0 - level))[::-1]
    eq[f"{root}.phi"] = (-0.01 * level ** 1.3)[::-1]
    coordinate = ks.rho_tor_norm_of(eq, 0)
    assert coordinate["source"] == "phi"
    np.testing.assert_allclose(coordinate["rho_tor_norm"], (level ** 0.65)[::-1], atol=1e-12)


def test_an_implausible_stored_rho_is_not_used():
    eq = _equilibrium(times=(0.314,), phi=False)
    eq["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = np.zeros(41)
    coordinate = ks.rho_tor_norm_of(eq, 0)
    assert coordinate["coordinate"] == "unavailable" and "monotonic" in coordinate["reason"]
    eq["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = np.linspace(0, 1, 41) ** 0.6
    assert ks.rho_tor_norm_of(eq, 0)["source"] == "stored"


def test_a_scalar_error_node_does_not_break_the_sampler():
    ts = _thomson()
    ts["thomson_scattering.channel.0.n_e.data_error_upper"] = 1e17
    sampled = thomson_pressure_samples(_equilibrium(), 0, ts)
    assert sampled["available"] and math.isnan(sampled["channels"][0]["n_e_error"])


def test_mismatched_pressure_length_is_reported_not_raised():
    eq = _equilibrium(times=(0.314,))
    eq["equilibrium.time_slice.0.profiles_1d.pressure"] = np.ones(10)
    coordinate = ks.rho_tor_norm_of(eq, 0)
    got = ks.integrated_pressure_ratio(eq, 0, coordinate["rho_tor_norm"], np.ones(41))
    assert not got["available"] and "length" in got["reason"]


def test_unavailable_samples_carry_a_code():
    eq = _equilibrium()
    assert thomson_pressure_samples(eq, 0, ODS(consistency_check=False))["code"] == "no_thomson"
    assert thomson_pressure_samples(eq, 0, _thomson(times=(0.2, 0.21)))["code"] == "beyond_tolerance"
    assert thomson_pressure_samples(eq, 0, _thomson(radii=(0.78,)))["code"] == "no_channel_inside"


# ---------------------------------------------------------------------------
# the Tier A build, end to end on a synthetic FileDB
# ---------------------------------------------------------------------------

def _plain(ods: ODS, path) -> None:
    import gzip
    import json
    from omas import save_omas_json

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix("")
    save_omas_json(ods, str(raw))
    with open(raw) as source, gzip.open(path, "wt") as target:
        target.write(source.read())
    raw.unlink()


def _record(shot, time_ms, *, setting="statistical_891", betap=0.2, converged=True):
    return {"setting": setting, "shot": shot, "time_ms": time_ms, "converged": converged,
            "scalars": {"betap": betap, "wmhd": 100.0, "q95": 5.0, "ipmhd": 9.5e4, "li": 0.8},
            "pressure_min": 0.0, "ip_measured": 1e5, "fit": {"probe_reduced_chi2": 1.0, "loop_reduced_chi2": 1.0,
                                                              "ip_sigma_median": 1e4},
            "gs": {}, "virial": None, "thomson": None}


def _build_state_module():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "workflow" / "kinetic_state" / "build_state.py"
    spec = importlib.util.spec_from_file_location("lane_k_build_state", root)
    build_state = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_state)
    return build_state


def _text_mode_calls_without_encoding(source: Path) -> list[str]:
    """``open``/``gzip.open`` in text mode, ``read_text`` and ``write_text`` calls lacking ``encoding=``."""
    import ast

    missing = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        called = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
        if called in {"open", "opener"}:
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
            text_mode = "b" not in str(mode)
        else:
            text_mode = called in {"read_text", "write_text"}
        if text_mode and "encoding" not in {keyword.arg for keyword in node.keywords}:
            missing.append(f"{source.name}:{node.lineno}: {called}()")
    return missing


def test_the_atlas_is_written_and_read_as_utf8_not_at_the_locale():
    """CSV reason cells may carry non-ASCII; the repository rule is UTF-8 everywhere
    (vaft/version.py; cold review 0.8.0 delta-absorb-5 F6)."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "workflow" / "kinetic_state" / "build_state.py"
    assert _text_mode_calls_without_encoding(source) == []


def test_the_kinetic_veto_grades_the_slice_at_the_row_time_not_slice_zero(tmp_path):
    """The electron-EFIT row is keyed by time; its veto and scalars must come from
    the slice at that time, as its Thomson match does (cold review 0.8.0 delta-absorb-5 F4)."""
    import json
    from omas import save_omas_json

    build_state = _build_state_module()
    times = (0.316, 0.312)  # the row's slice (0.312) stored second, on purpose
    ods = _equilibrium(times)
    for i, q95 in enumerate((3.0, 7.0)):
        ods[f"equilibrium.time_slice.{i}.global_quantities.q_95"] = q95
        ods[f"equilibrium.time_slice.{i}.global_quantities.ip"] = 1.0e5 * (i + 1)
    save_omas_json(ods, str(tmp_path / "kin.json"))
    product = json.loads((tmp_path / "kin.json").read_text(encoding="utf-8"))
    seen = []

    class Criteria:
        def admissible(self, record):
            seen.append(record)
            return {"status": "admissible", "reasons": []}

    veto, scalars = build_state._kinetic_veto(Criteria(), product, {}, {"ip_measured": 2.0e5}, 0.312)
    assert veto["status"] == "admissible"
    assert scalars["q95"] == 7.0 and scalars["ipmhd"] == 2.0e5
    assert seen[0]["scalars"]["q95"] == 7.0
    assert math.isclose(seen[0]["pressure_min"], 0.0, abs_tol=0.0)
    with pytest.raises(LookupError):
        build_state._kinetic_veto(Criteria(), product, {}, {}, 0.320)


def test_build_gates_on_rederived_labels_and_uses_the_earning_setting(tmp_path):
    import json

    build_state = _build_state_module()

    shot = 99001
    filedb = tmp_path / "filedb"
    _plain(_equilibrium(), filedb / "omas/efit/magnetic" / str(shot) / "output/efit.json.gz")
    _plain(_thomson(), filedb / "omas/thomson" / str(shot) / "output/thomson.json.gz")
    records = [
        _record(shot, 312),                                         # admissible
        _record(shot, 313, betap=-1.0),                             # vetoed: never a row
        _record(shot, 314, setting="routine"),                      # routine only: never counts
        _record(shot, 315, setting="a_scan", betap=-1.0),           # a setting that failed...
        _record(shot, 315, setting="b_scan"),                       # ...and the one that earned it
        {"setting": "c_scan", "shot": shot, "time_ms": 315, "error": "boom"},
    ]
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"records": records, "labels": [
        {"shot": shot, "time_ms": 313, "label": "admissible"}]}))  # stale: criteria say otherwise
    out = tmp_path / "atlas"
    result = build_state.build(filedb, analysis, out, ti_te_ratio=1.0)

    rows = {r["time_efit_s"]: r for r in result["state"]}
    assert sorted(rows) == [0.312, 0.315]
    assert rows[0.315]["efit_setting"] == "b_scan"
    assert all(r["efit_lineage"] == "magnetics" and r["efit_status"] == "valid" for r in rows.values())
    assert rows[0.312]["ts_status"] == "matched"
    assert result["manifest"]["labels"]["differ_from_stored"] == 1
    assert (out / "state.csv").is_file() and (out / "schema/state.schema.json").is_file()


def test_magnetics_rows_carry_thomson_consistency_regraded_not_stored():
    """criteria v2: Thomson is a column beside efit_quality, re-graded with this
    checkout's criteria rather than read from the analysis JSON's stored verdicts."""
    import math

    build_state = _build_state_module()
    criteria = build_state._criteria()
    assert build_state.STATE_COLUMNS["thomson_consistent"][0] == "boolean"
    # A record whose stored evaluation is stale (graded under the old [1, 3] band).
    record = {"thomson": {"log_ratio": math.log(1.0 / 2.7)},
              "evaluation": {"verdicts": {"thomson": {"status": "pass"}}}}
    evaluation = criteria.evaluate(record)
    assert build_state._status(evaluation, "thomson") == "fail"
    assert evaluation["physically_consistent"] is False
