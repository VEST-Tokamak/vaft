"""RDCON/STRIDE results recoverable from the ODS alone (#939).

The fixture is the real RDCON output of ``test_gpec_rdcon_criteria.py``: shot
39915 at 319 ms, GPEC e68d7ac2, n=1, mpsi=256, CHEASE-refined.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.code.gpec import read_pest3_matching_output
from vaft.formula.stability import ggj_pressure_curvature_offset, ggj_resistive_parameter
from vaft.machine_mapping.mhd_linear import extract_rdcon_stability, mhd_linear

FIXTURE = Path(__file__).resolve().parent / "data" / "gpec" / "rdcon_39915_319_n1"


def _source(tmp_path: Path, solver: str) -> Path:
    target = tmp_path / solver
    target.mkdir(parents=True)
    shutil.copy(FIXTURE / "rdcon_output_n1.nc", target / f"{solver}_output_n1.nc")
    return target


@pytest.fixture(scope="module")
def native():
    return read_pest3_matching_output(FIXTURE, solver="rdcon", mode=1)


@pytest.fixture()
def mapped(tmp_path):
    ods = ODS(consistency_check=False)
    mhd_linear(ods, str(_source(tmp_path, "rdcon")), {"module": "rdcon", "modes": [1]})
    return ods


def test_the_profiles_and_matrix_round_trip_from_the_ods(mapped, native):
    [row] = extract_rdcon_stability(mapped)
    assert (row["solver"], row["n_tor"], row["time_slice"], row["msing"]) == ("rdcon", 1, 0, 9)
    np.testing.assert_array_equal(row["psi_n"], native.psi_n)
    for public, name in (("q", "q"), ("D_I", "di"), ("D_R", "dr"), ("H", "h")):
        np.testing.assert_array_equal(row[public], getattr(native, name))
    np.testing.assert_array_equal(np.isnan(row["C_A"]), np.isnan(native.ca1))
    np.testing.assert_array_equal(row["delta_prime_matrix"], native.Delta_prime)


def test_surfaces_join_ntms_with_the_fragment(mapped, native):
    [row] = extract_rdcon_stability(mapped)
    expected = native.rational_surface_stability()
    assert [s["m"] for s in row["surfaces"]] == [s["m"] for s in expected]
    for got, want in zip(row["surfaces"], expected):
        assert got["delta_prime"] == pytest.approx(complex(want["delta_prime_real"], want["delta_prime_imag"]))
        assert got["psi_n"] == pytest.approx(want["psi_n"]) and got["dr"] == pytest.approx(want["dr"])
    # The surface Delta-prime is the matrix diagonal, both from the ODS.
    np.testing.assert_allclose([s["delta_prime"] for s in row["surfaces"]], np.diag(row["delta_prime_matrix"]))


def test_ggj_identity_holds_on_the_mapped_profiles(mapped):
    [row] = extract_rdcon_stability(mapped)
    np.testing.assert_allclose(ggj_resistive_parameter(row["D_I"], row["H"]), row["D_R"], rtol=0, atol=1e-12)
    assert np.all(ggj_pressure_curvature_offset(row["H"]) >= 0)


def test_ggj_helpers():
    assert ggj_pressure_curvature_offset(0.5) == 0.0
    assert ggj_resistive_parameter(-0.1, 0.8) == pytest.approx(-0.1 + 0.09)
    # D_R >= D_I always: a surface can be Mercier stable yet resistively unstable.
    h = np.linspace(-2, 2, 9)
    assert np.all(ggj_resistive_parameter(np.zeros_like(h), h) >= 0)


def test_rdcon_and_stride_stay_apart(tmp_path):
    ods = ODS(consistency_check=False)
    mhd_linear(ods, str(_source(tmp_path, "rdcon")), {"module": "rdcon", "modes": [1]})
    mhd_linear(ods, str(_source(tmp_path, "stride")), {"module": "stride", "modes": [1]})
    rows = {row["solver"]: row for row in extract_rdcon_stability(ods)}
    assert set(rows) == {"rdcon", "stride"}
    assert all(len(r["surfaces"]) == 9 for r in rows.values())


def test_remapping_keeps_the_last_fragment(mapped, tmp_path):
    mhd_linear(mapped, str(_source(tmp_path / "again", "rdcon")), {"module": "rdcon", "modes": [1]})
    assert len(extract_rdcon_stability(mapped)) == 1


def test_reading_does_not_create_paths_and_v1_fragments_are_skipped():
    ods = ODS(consistency_check=False)
    assert extract_rdcon_stability(ods) == []
    assert "mhd_linear" not in ods.keys()
    ods["mhd_linear.code.parameters"] = '<parameters><solver name="rdcon" n_tor="1"><msing>9</msing></solver></parameters>'
    assert extract_rdcon_stability(ods) == []


def test_a_malformed_matrix_warns_instead_of_raising():
    ods = ODS(consistency_check=False)
    ods["mhd_linear.code.parameters"] = (
        '<parameters><solver name="rdcon" n_tor="1" version="2" time_slice="0" position="0">'
        '<msing>2</msing><delta_prime_matrix size="2" real="1 2 3" imag="0 0 0"/></solver></parameters>'
    )
    with pytest.warns(RuntimeWarning, match="malformed"):
        assert extract_rdcon_stability(ods) == []
