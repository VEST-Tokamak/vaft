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
    assert summary["rdcon_status"] == "VALID_UNSTABLE"


def test_summary_status_without_usable_surfaces(build):
    assert build.matching_summary([], "failed", "stride")["stride_status"] == "SOLVER_FAILURE"
    assert build.matching_summary([], None, "stride")["stride_status"] == "NOT_APPLICABLE"
    lonely = build.pair_surfaces([_surface(4, 0.7, -20.0)], [])
    assert build.matching_summary(lonely, "stable", "stride")["stride_status"] == "NUMERICALLY_UNRESOLVED"


@pytest.mark.parametrize(
    ("w256", "w512", "raw", "expected"),
    [
        (2.7, 2.71, "stable", "VALID_STABLE"),
        (-1.0, -0.9, "completed", "VALID_UNSTABLE"),
        (0.04, -0.2, "stable", "NUMERICALLY_UNRESOLVED"),
        (2.7, None, "stable", "NUMERICALLY_UNRESOLVED"),
        (None, None, "failed", "SOLVER_FAILURE"),
        (None, None, None, "NOT_APPLICABLE"),
    ],
)
def test_dcon_status(build, w256, w512, raw, expected):
    assert build.dcon_status(w256, w512, raw) == expected


def test_zero_crossings_are_read_from_dcon_out(build, tmp_path):
    (tmp_path / "dcon.out").write_text(
        " psi = 1.000E-02\n Zero crossing at psi = 7.593E-01, q = 4.000E+00\n"
        " Zero crossing at psi = 9.100E-01, q = 7.250E+00\n"
    )
    assert build.zero_crossings(tmp_path) == [(0.7593, 4.0), (0.91, 7.25)]
    assert build.zero_crossings(tmp_path / "missing") == []


def test_ggj_is_interpolated_at_the_surface(build):
    grid = np.linspace(0.0, 1.0, 11)
    ggj = {"psi_n": grid, "di": -grid, "dr": -grid + 0.25, "h": 0.5 + 0 * grid, "ca1": 2 * grid}
    at = build.ggj_at(ggj, 0.45)
    assert at == pytest.approx({"D_I": -0.45, "D_R": -0.2, "H": 0.5, "C_A": 0.9})
    assert build.ggj_at(None, 0.45) == {"D_I": None, "D_R": None, "H": None, "C_A": None}


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
