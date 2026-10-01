"""The Tier A stability atlas: slice selection and the atlas builder's rules (#1429)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "workflow" / "stability_atlas"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def select():
    yield _load("select_slices")
    sys.modules.pop("select_slices", None)


@pytest.fixture(scope="module")
def build(select):
    yield _load("build_atlas")
    sys.modules.pop("build_atlas", None)


def _label(shot, time_ms, label):
    return {"shot": shot, "time_ms": time_ms, "label": label, "good": [], "admissible": ["statistical_891"] * (label != "unreconstructible")}


ANALYSIS = {
    "labels": [
        _label(39915, 316, "admissible"),
        _label(39915, 317, "good"),
        _label(39915, 318, "unreconstructible"),
        _label(40330, 320, "unreconstructible"),
    ],
    "kinetic": [
        {"shot": 39915, "time_ms": 316, "chi2": 87.7},
        {"shot": 40330, "time_ms": 320, "chi2": 174.4},
        {"shot": 39915, "time_ms": 319, "chi2": 50.0},  # no magnetics-only label at 319
    ],
}


def test_only_good_and_admissible_slices_are_selected(select):
    rows, _ = select.select(ANALYSIS)
    magnetic = [(r["shot"], r["time_ms"], r["efit_label"]) for r in rows if r["efit_lineage"] == select.MAGNETICS_ONLY]
    assert magnetic == [(39915, 316, "admissible"), (39915, 317, "good")]
    assert all(r["efit_label"] in ("good", "admissible") for r in rows)


def test_a_kinetic_slice_inherits_the_label_at_exactly_its_own_time(select):
    rows, dropped = select.select(ANALYSIS)
    kinetic = [r for r in rows if r["efit_lineage"] == select.ELECTRON_KINETIC]
    assert [(r["shot"], r["time_ms"], r["efit_label"], r["kinetic_chi2"]) for r in kinetic] == [(39915, 316, "admissible", 87.7)]
    reasons = {(d["shot"], d["time_ms"]): d["reason"] for d in dropped}
    assert reasons[(40330, 320)] == "label unreconstructible"
    # 319 ms has no label; the 318 ms neighbour must not be borrowed.
    assert reasons[(39915, 319)] == "no magnetics-only label at this time"


def test_efit_setting_is_the_preset_the_label_belongs_to(select):
    # A good slice is good under one preset but admissible under two: the
    # setting column names the preset its label comes from, on both lineages.
    analysis = {
        "labels": [
            {"shot": 39915, "time_ms": 317, "label": "good", "good": ["statistical_891"], "admissible": ["statistical_891", "routine_like"]},
            {"shot": 39915, "time_ms": 316, "label": "admissible", "good": [], "admissible": ["routine_like", "statistical_891"]},
        ],
        "kinetic": [{"shot": 39915, "time_ms": 317, "chi2": 1.0}],
    }
    rows, _ = select.select(analysis)
    settings = {(r["time_ms"], r["efit_lineage"]): r["efit_setting"] for r in rows}
    assert settings == {
        (316, select.MAGNETICS_ONLY): "routine_like;statistical_891",
        (317, select.MAGNETICS_ONLY): "statistical_891",
        (317, select.ELECTRON_KINETIC): "statistical_891",
    }


def test_duplicate_labels_are_refused(select):
    bad = {"labels": [_label(39915, 316, "good"), _label(39915, 316, "admissible")], "kinetic": []}
    with pytest.raises(ValueError, match="duplicate label"):
        select.select(bad)


def test_selection_round_trips_through_csv(select, tmp_path):
    rows, _ = select.select(ANALYSIS)
    back = select.read(select.write(rows, tmp_path / "slices.csv"))
    assert back == rows


def _surface(m, psi, dp):
    return {"m": m, "n": 1, "psi_n": psi, "q": float(m), "delta_prime_real": dp, "delta_prime_imag": 0.0}


def test_surfaces_pair_by_mode_number_and_position_not_by_index(build):
    primary = [_surface(3, 0.48, -10.0), _surface(4, 0.70, -20.0), _surface(5, 0.80, -30.0)]
    # The check run lost m=4 and lists surfaces in a different order.
    check = [_surface(5, 0.8004, -31.0), _surface(3, 0.48, 50.0)]
    paired = {s["m"]: s for s in build.pair_surfaces(primary, check)}
    assert paired[5]["delta_prime_check"] == -31.0 and paired[5]["resolved"]
    assert paired[4]["delta_prime_check"] is None and not paired[4]["resolved"]
    assert not paired[3]["resolved"]  # sign flip
    assert [paired[m]["position"] for m in (3, 4, 5)] == ["first", "interior", "last"]


def test_a_surface_that_moved_is_not_paired(build):
    paired = build.pair_surfaces([_surface(4, 0.70, -20.0)], [_surface(4, 0.71, -20.0)])
    assert paired[0]["delta_prime_check"] is None and not paired[0]["resolved"]


def test_resolution_tolerance(build):
    rtol = build.DELTA_PRIME_RTOL
    inside = build.pair_surfaces([_surface(4, 0.7, -100.0)], [_surface(4, 0.7, -100.0 * (1 - rtol / 2))])
    outside = build.pair_surfaces([_surface(4, 0.7, -100.0)], [_surface(4, 0.7, -100.0 * (1 - 2 * rtol))])
    assert inside[0]["resolved"] and not outside[0]["resolved"]


def test_summary_uses_resolved_interior_surfaces_only(build):
    primary = [_surface(3, 0.48, 6.5e5), _surface(4, 0.70, 40.0), _surface(5, 0.80, -30.0), _surface(6, 0.95, 900.0)]
    check = [_surface(3, 0.48, 6.4e5), _surface(4, 0.70, 41.0), _surface(5, 0.80, -31.0), _surface(6, 0.95, 890.0)]
    summary = build.matching_summary(build.pair_surfaces(primary, check), "stable", "rdcon")
    # m=3 (first) and m=6 (last) are resolved but excluded from the summary.
    assert summary["rdcon_n_resolved_surfaces"] == 4
    assert summary["rdcon_n_summary_surfaces"] == 2
    assert summary["rdcon_delta_prime_max"] == 40.0 and summary["rdcon_m_at_delta_prime_max"] == 4
    # Δ′ never gets a stable/unstable label (#939).
    assert summary["rdcon_status"] == "RESOLVED"


def test_summary_status_without_usable_surfaces(build):
    assert build.matching_summary([], "failed", "stride")["stride_status"] == "SOLVER_FAILURE"
    assert build.matching_summary([], None, "stride")["stride_status"] == "NOT_APPLICABLE"
    # Completed, but no rational surface in range: not a solver failure.
    assert build.matching_summary([], "stable", "stride")["stride_status"] == "NOT_APPLICABLE"
    three = [_surface(3, 0.5, -10.0), _surface(4, 0.7, -20.0), _surface(5, 0.8, -30.0)]
    disagree = build.pair_surfaces(three, [_surface(m["m"], m["psi_n"], -2 * m["delta_prime_real"]) for m in three])
    assert build.matching_summary(disagree, "stable", "stride")["stride_status"] == "NUMERICALLY_UNRESOLVED"
    # No 512 check at all is "not checked", never "unresolved".
    unchecked = build.pair_surfaces(three, [])
    assert build.matching_summary(unchecked, "stable", "stride", checked=False)["stride_status"] == "NOT_CHECKED"


@pytest.mark.parametrize(
    ("w256", "w512", "raw256", "raw512", "expected"),
    [
        (2.7, 2.71, "stable", "stable", "VALID_STABLE"),
        (-1.0, -0.9, "completed", "completed", "VALID_UNSTABLE"),
        (0.04, -0.2, "stable", "completed", "NUMERICALLY_UNRESOLVED"),
        (2.7, None, "stable", "failed", "NOT_CHECKED"),
        (2.7, None, "stable", None, "NOT_CHECKED"),
        (None, 2.7, "failed", "stable", "SOLVER_FAILURE"),
        (None, None, None, None, "NOT_APPLICABLE"),
    ],
)
def test_dcon_status(build, w256, w512, raw256, raw512, expected):
    assert build.dcon_status(w256, w512, raw256, raw512) == expected


def test_a_slice_without_its_equilibrium_is_invalid_not_unrun(build, select, tmp_path):
    row = select.select(ANALYSIS)[0][0]
    rows, surfaces = build.slice_rows(tmp_path, row, (1, 2), {"atlas_version": 1})
    assert [r["n_tor"] for r in rows] == [1, 2] and surfaces == []
    for record in rows:
        assert record["equilibrium_status"] == "missing_source"
        assert {record[k] for k in ("dcon_full_status", "dcon_trunc_status", "rdcon_status", "stride_status")} == {
            "INVALID_EQUILIBRIUM"
        }
        assert record["ideal_stable_full_edge"] is None


def test_zero_crossings_are_read_from_dcon_out(build, tmp_path):
    (tmp_path / "dcon.out").write_text(
        " psi = 1.000E-02\n Zero crossing at psi = 7.593E-01, q = 4.000E+00\n"
        " Zero crossing at psi = 9.100E-01, q = 7.250E+00\n"
    )
    assert build.zero_crossings(tmp_path) == [(0.7593, 4.0), (0.91, 7.25)]
    assert build.zero_crossings(tmp_path / "missing") == []


RDCON_RUN = Path(__file__).resolve().parent / "data" / "gpec" / "rdcon_39915_319_n1"


def _rdcon_run_with_unevaluated_band(tmp_path, lo, hi):
    """A copy of the RDCON fixture whose ``ca1`` is the raw zero on lo < psi_n < hi.

    That is what ``bal.f`` leaves on a Mercier-unstable (D_I > 0) band: the
    ballooning integral is never evaluated there and the profile keeps its
    initial 0 (cold review 0.8.0 delta-absorb-6 F1).
    """
    import xarray as xr

    with xr.open_dataset(RDCON_RUN / "rdcon_output_n1.nc") as ds:
        ds = ds.load()
    ca1 = ds["ca1"].values.copy()
    ca1[(ds["psi_n"].values > lo) & (ds["psi_n"].values < hi)] = 0.0
    ds["ca1"].values[:] = ca1
    ds.to_netcdf(tmp_path / "rdcon_output_n1.nc")
    return tmp_path


def test_ggj_at_a_surface_follows_the_library_reader(build, tmp_path):
    from vaft.code.gpec import read_pest3_matching_output

    run_dir = _rdcon_run_with_unevaluated_band(tmp_path, 0.6, 0.75)  # contains the m=4 surface at 0.698
    rows = build.matching_surfaces({"status": "stable", "run_dir": str(run_dir), "n": 1}, "rdcon")
    by_m = {row["m"]: row for row in rows}
    assert len(rows) == 9 and {3, 4, 5} <= set(by_m)
    # Inside the band the ballooning criterion was never evaluated: no value, not 0.
    assert by_m[4]["C_A"] is None
    assert by_m[4]["D_I"] == pytest.approx(-0.35825, abs=1e-4) and by_m[4]["D_R"] is not None and by_m[4]["H"] is not None
    # At the band edge no blend with the placeholder zero leaks out either.
    library = {r["m"]: r for r in read_pest3_matching_output(run_dir, solver="rdcon", mode=1).rational_surface_stability()}
    for m, row in by_m.items():
        assert row["C_A"] == library[m]["ca1"]
        assert row["D_I"] == library[m]["di"] and row["D_R"] == library[m]["dr"] and row["H"] == library[m]["h"]
    assert by_m[3]["C_A"] == pytest.approx(49.6923, abs=1e-3) and by_m[5]["C_A"] == pytest.approx(184.314, abs=1e-2)
    assert build.matching_surfaces({"status": "failed", "run_dir": str(run_dir), "n": 1}, "rdcon") == []


REFERENCE = Path(__file__).resolve().parent / "data" / "gpec" / "dcon_edge_792"


def test_dcon_columns_on_real_gpec_output(build):
    # The #792 reference: real DCON (GPEC e68d7ac2) for 39915@319 ms, n=1.
    full = build.dcon_columns({"status": "stable", "run_dir": str(REFERENCE / "full_edge"), "n": 1}, "dcon_full_256")
    assert full["dcon_full_256_W_t"] == pytest.approx(2.6923316420322827, rel=1e-9)
    assert full["dcon_full_256_edge_treatment"] == "full_edge"
    assert full["dcon_full_256_n_zero_crossings"] == 0
    # Local criteria are read from the reference run only, with their sign rules.
    assert full["mercier_evaluated"] is True and full["ballooning_evaluated"] is True
    assert full["ideal_interchange_unstable"] is (full["max_D_I"] > 0)
    assert full["ballooning_unstable"] is (full["min_C_A"] < 0)
    assert 0.0 <= full["psi_n_at_max_D_I"] <= 1.0
    trunc = build.dcon_columns({"status": "stable", "run_dir": str(REFERENCE / "peak_dw_truncated"), "n": 1}, "dcon_trunc_256")
    assert trunc["dcon_trunc_256_edge_treatment"] == "peak_dw_truncated"
    assert "max_D_I" not in trunc


def test_dcon_columns_of_a_failed_run_carry_only_the_status(build):
    assert build.dcon_columns({"status": "failed", "run_dir": "/nonexistent", "n": 1}, "dcon_full_512") == {
        "dcon_full_512_status_raw": "failed"
    }
    assert build.dcon_columns(None, "dcon_full_512") == {"dcon_full_512_status_raw": None}


@pytest.fixture(scope="module")
def batch(build):
    yield _load("run_batch")
    sys.modules.pop("run_batch", None)


def test_one_failing_slice_does_not_stop_preparation(batch):
    from concurrent.futures import Future

    future = Future()
    future.set_exception(KeyError("equilibrium.time_slice"))
    row = {"shot": 39915, "time_ms": 316, "efit_lineage": "electron-kinetic"}
    records = batch.prepared(future, row)
    assert records == [
        {
            "stage": "source",
            "slice": "39915.00316",
            "lineage": "electron-kinetic",
            "status": "error",
            "reason": "KeyError('equilibrium.time_slice')",
        }
    ]


def test_config_hash_is_matched_by_time_in_seconds(batch, tmp_path):
    import json

    manifest = tmp_path / "omas" / "efit" / "magnetic" / "39915" / "metadata" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    statuses = [
        {"time": 0.318, "provenance": {"configuration": {"scientific_sha256": "a318"}}},
        {"time": 0.319, "provenance": {"configuration": {"scientific_sha256": "a319"}}},
    ]
    manifest.write_text(json.dumps({"slice_statuses": statuses}))
    assert batch.magnetic_config_sha(tmp_path, {"shot": 39915, "time_ms": 319}) == "a319"
    assert batch.magnetic_config_sha(tmp_path, {"shot": 39915, "time_ms": 320}) is None


def test_duplicate_kinetic_entries_are_refused(select):
    bad = {"labels": ANALYSIS["labels"], "kinetic": [ANALYSIS["kinetic"][0], ANALYSIS["kinetic"][0]]}
    with pytest.raises(ValueError, match="duplicate electron-kinetic"):
        select.select(bad)


def test_rdcon_n2_gets_the_measured_memory_budget(batch):
    reserve, limit = batch.memory_policy("rdcon", 2)
    assert reserve >= 24_600 and limit > reserve  # measured peak 24.6 GB at mpsi 512 (#1460)
    assert batch.memory_policy("dcon", 2) == batch.MEMORY_MB_DEFAULT
    assert batch.memory_policy("rdcon", 3) == batch.MEMORY_MB_RDCON_HIGH_N
