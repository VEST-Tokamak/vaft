"""DCON post-processing from the ODS payload (#940, sections 8 and 10).

Real DCON output (the #792 fixtures: shot 39915 at 319 ms, n=1, full edge and
peak-dW truncated) is mapped through ``mhd_linear`` and read back with
``extract_dcon_stability``, so everything here runs from the final ODS.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.machine_mapping.mhd_linear import extract_dcon_stability, mhd_linear
from vaft.process.mhd_stability import (
    criterion_intervals,
    dcon_edge_comparison,
    dcon_edge_scan,
    dcon_local_stability,
)

REFERENCE = Path(__file__).resolve().parent / "data" / "gpec" / "dcon_edge_792"


def _row(tmp_path: Path, treatment: str) -> dict:
    source = tmp_path / treatment
    shutil.copytree(REFERENCE / treatment, source)
    ods = ODS(consistency_check=False)
    mhd_linear(ods, str(source), {"module": "dcon", "modes": [1]})
    [row] = extract_dcon_stability(ods)
    return row


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dcon940")
    return {t: _row(tmp, t) for t in ("full_edge", "peak_dw_truncated")}


def test_intervals_and_crossings_on_a_known_profile():
    psi = np.linspace(0.0, 1.0, 11)
    values = np.array([-1, -1, 1, 1, -1, -1, -1, 2, 2, 2, -1], dtype=float)
    out = criterion_intervals(psi, values, unstable_side="positive")
    last = 0.9 + 0.1 * 2 / 3
    np.testing.assert_allclose(out["zero_crossings"], [0.15, 0.35, 0.6 + 0.1 / 3, last])
    np.testing.assert_allclose(out["unstable_intervals"], [[0.15, 0.35], [0.6 + 0.1 / 3, last]])
    assert out["unstable_fraction"] == pytest.approx(5 / 11)
    assert out["extremum"] == 2 and out["psi_n_at_extremum"] == pytest.approx(0.7)
    flipped = criterion_intervals(psi, -values, unstable_side="negative")
    np.testing.assert_allclose(flipped["unstable_intervals"], out["unstable_intervals"])


def test_unevaluated_samples_break_intervals_and_are_never_marginal():
    psi = np.linspace(0.0, 1.0, 5)
    values = np.array([1.0, 1.0, 0.0, 1.0, -1.0])
    mask = np.array([True, True, False, True, True])
    out = criterion_intervals(psi, values, unstable_side="positive", evaluated=mask)
    # The unevaluated zero at 0.5 is neither a crossing nor a gap filler.
    assert out["zero_crossings"] == [pytest.approx(0.875)]
    assert out["unstable_intervals"] == [[0.0, 0.25], [0.75, pytest.approx(0.875)]]
    assert out["n_evaluated"] == 4
    empty = criterion_intervals(psi, values, unstable_side="positive", evaluated=np.zeros(5, bool))
    assert empty["unstable_intervals"] == [] and empty["extremum"] is None


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="unstable_side"):
        criterion_intervals([0, 1], [1, 2], unstable_side="up")
    with pytest.raises(ValueError, match="length"):
        criterion_intervals([0, 1], [1, 2, 3], unstable_side="positive")
    with pytest.raises(ValueError, match="increase"):
        criterion_intervals([1, 0], [1, 2], unstable_side="positive")


def test_local_stability_on_real_dcon(rows):
    row = rows["full_edge"]
    out = dcon_local_stability(row)
    # This equilibrium is Mercier and ballooning stable everywhere, but D_R > 0 on 20 surfaces.
    assert out["mercier"]["unstable_intervals"] == [] and out["mercier"]["unstable_fraction"] == 0
    assert out["ballooning"]["unstable_intervals"] == []
    resistive = out["resistive_interchange"]
    assert resistive["unstable_fraction"] == pytest.approx(20 / 129)
    assert resistive["extremum"] == pytest.approx(row["max_D_R"])
    assert resistive["psi_n_at_extremum"] == pytest.approx(row["psi_n_at_max_D_R"])
    assert out["ballooning"]["extremum"] == pytest.approx(row["min_C_A"])
    for lo, hi in resistive["unstable_intervals"]:
        assert lo <= hi


def test_an_unevaluated_criterion_is_none(rows):
    row = dict(rows["full_edge"], ballooning_evaluated=False, mercier_evaluated=None)
    assert dcon_local_stability(row) == {"mercier": None, "resistive_interchange": None, "ballooning": None}


def test_edge_scan_finds_dcons_own_truncation(rows):
    out = dcon_edge_scan(rows["peak_dw_truncated"])
    assert out["truncated_at_peak"] is True
    assert out["psi_n_at_peak"] == pytest.approx(rows["peak_dw_truncated"]["psilim"])
    # q(0.95) = 8.807 on the nominal grid 8 + i/20: 17 pre-edge entries (sing.f:226-238).
    assert out["peak_search_start"] == 17
    assert out["negative_intervals"] and len(out["zero_crossings_psi_n"]) >= 2
    assert dcon_edge_scan(rows["full_edge"]) is None


def test_pre_edge_entries_are_not_searched(rows):
    row = rows["peak_dw_truncated"]
    dw = np.asarray(row["edge_scan"]["dW"]).copy()
    dw[0] = 10 * abs(dw.real).max()
    raised = dict(row, edge_scan=dict(row["edge_scan"], dW=dw))
    assert dcon_edge_scan(raised)["truncated_at_peak"] is True


def test_edge_comparison(rows):
    full, truncated = rows["full_edge"], rows["peak_dw_truncated"]
    out = dcon_edge_comparison(full, truncated)
    assert out["delta_W_edge"] == pytest.approx(full["W_t"].real - truncated["W_t"].real)
    assert out["sign_agreement"] is True and out["n_tor"] == 1
    with pytest.raises(ValueError, match="full_edge"):
        dcon_edge_comparison(truncated, full)
    with pytest.raises(ValueError, match="different n"):
        dcon_edge_comparison(full, dict(truncated, n_tor=2))
