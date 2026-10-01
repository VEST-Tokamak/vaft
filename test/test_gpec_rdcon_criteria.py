"""RDCON local stability criteria (#939) and per-solver attribution of ntms surfaces (#143).

``test/data/gpec/rdcon_39915_319_n1/rdcon_output_n1.nc`` is a real RDCON output.
Shot 39915 at 319 ms is a good slice of the #1331 ``statistical_891``
campaign, CHEASE-refined; the run is GPEC e68d7ac2, n=1, mpsi=256. The file was
trimmed with ``nccopy -V`` to the matching matrices, the rational-surface
coordinates and the 1-D profiles.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.code.gpec import Pest3MatchingOutput, read_pest3_matching_output
from vaft.code.gpec._matching_output import SURFACE_COLUMNS
from vaft.machine_mapping.mhd_linear import mhd_linear, ntms_solver_surfaces

FIXTURE = Path(__file__).resolve().parent / "data" / "gpec" / "rdcon_39915_319_n1"


@pytest.fixture(scope="module")
def rdcon() -> Pest3MatchingOutput:
    return read_pest3_matching_output(FIXTURE, solver="rdcon", mode=1)


def test_rdcon_writes_dr_as_di_plus_h_minus_half_squared(rdcon):
    # rdcon/mercier.f:157 builds D_R from D_I and H; the identity is exact.
    assert rdcon.h is not None
    np.testing.assert_allclose(rdcon.dr, rdcon.di + (rdcon.h - 0.5) ** 2, rtol=0, atol=1e-12)


def test_profiles_are_on_the_solver_grid(rdcon):
    assert rdcon.psi_n.shape == rdcon.di.shape == rdcon.q.shape
    assert rdcon.psi_n[0] == pytest.approx(0.01) and rdcon.psi_n[-1] == pytest.approx(0.994)


def test_rational_surface_table_on_real_output(rdcon):
    rows = rdcon.rational_surface_stability()
    assert len(rows) == rdcon.msing == 9
    assert all(tuple(row) == SURFACE_COLUMNS for row in rows)
    first = rows[0]
    assert (first["m"], first["n"]) == (3, 1)
    assert first["q"] == pytest.approx(3.0)
    assert first["delta_prime_real"] == pytest.approx(-9.230442362470004, rel=1e-12)
    # The criteria are the profiles evaluated at the surface, by psi_n.
    assert first["di"] == pytest.approx(float(np.interp(first["psi_n"], rdcon.psi_n, rdcon.di)))
    assert first["dr"] == pytest.approx(first["di"] + (first["h"] - 0.5) ** 2, abs=1e-6)
    assert all(row["di"] < 0 for row in rows)  # Mercier-stable at every surface here


def test_a_surface_outside_the_profile_grid_gets_no_criteria(rdcon):
    assert rdcon._profile_at(rdcon.di, 0.999) is None
    assert rdcon._profile_at(None, 0.5) is None


def test_unevaluated_ballooning_is_nan_and_round_trips_as_null(rdcon):
    ca1 = rdcon.ca1.copy()
    ca1[0] = np.nan
    edited = Pest3MatchingOutput.from_dict({**rdcon.to_dict(), "ca1": [None] + list(ca1[1:])})
    payload = edited.to_dict()
    assert payload["schema_version"] == 2 and payload["ca1"][0] is None
    back = Pest3MatchingOutput.from_dict(payload)
    assert np.isnan(back.ca1[0]) and np.allclose(back.ca1[1:], ca1[1:])


def test_a_version_1_payload_still_reads(rdcon):
    payload = rdcon.to_dict()
    for name in ("psi_n", "q", "di", "dr", "h", "ca1"):
        payload.pop(name)
    payload["schema_version"] = 1
    old = Pest3MatchingOutput.from_dict(payload)
    assert old.di is None
    assert old.rational_surface_stability()[0]["di"] is None


def _source(tmp_path: Path, solver: str) -> Path:
    target = tmp_path / solver
    target.mkdir()
    shutil.copy(FIXTURE / "rdcon_output_n1.nc", target / f"{solver}_output_n1.nc")
    return target


def test_ntms_surfaces_are_attributed_to_the_solver_that_wrote_them(tmp_path):
    ods = ODS(consistency_check=False)
    mhd_linear(ods, str(_source(tmp_path, "rdcon")), {"module": "rdcon", "modes": [1]})
    mhd_linear(ods, str(_source(tmp_path, "stride")), {"module": "stride", "modes": [1]})
    rows = ntms_solver_surfaces(ods)
    by_solver = {name: [r for r in rows if r["solver"] == name] for name in ("rdcon", "stride")}
    assert [r["mode"] for r in by_solver["rdcon"]] == list(range(9))
    assert [r["mode"] for r in by_solver["stride"]] == list(range(9, 18))
    for row in rows:
        entry = ods["ntms"]["time_slice"][0]["mode"][row["mode"]]
        assert entry["m_pol"] == row["m"] and entry["n_tor"] == row["n_tor"] == 1
    # Local criteria travel with the surface at full float precision. The
    # identity holds exactly on the grid; between grid points it holds only to
    # interpolation error, because (h - 1/2)**2 is not linear.
    first = by_solver["rdcon"][0]
    surface = read_pest3_matching_output(FIXTURE, solver="rdcon", mode=1).rational_surface_stability()[0]
    assert (first["di"], first["dr"], first["h"], first["ca1"]) == (surface["di"], surface["dr"], surface["h"], surface["ca1"])
    assert first["dr"] == pytest.approx(first["di"] + (first["h"] - 0.5) ** 2, abs=1e-6)


def test_reading_attribution_does_not_create_the_ntms_ids():
    ods = ODS(consistency_check=False)
    assert ntms_solver_surfaces(ods) == []
    assert "ntms" not in ods.keys()


def test_version_1_fragments_are_not_attributed():
    ods = ODS(consistency_check=False)
    ods["ntms.code.parameters"] = '<parameters><solver name="rdcon" n_tor="1"><msing>9</msing></solver></parameters>'
    assert ntms_solver_surfaces(ods) == []
