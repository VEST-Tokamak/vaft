"""DCON/RDCON text readers against the netCDF of the same run.

The fixture is one real RDCON run (42893 at 337 ms, n = 1, mpsi 256, from the
Lane N batch). It wrote both ``rdcon_output_n1.nc`` and the text files.
``reference.json`` holds what :func:`read_pest3_matching_output` read from
that netCDF. ``dcon.out`` is cut after the energy table, with its profile
tables dropped; the reader does not use either.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vaft.code.gpec import (
    pest3_matrices_from_delta,
    read_dcon_text,
    read_delta_gw,
    read_pest3_matching_text,
)

RUN = Path(__file__).parent / "data" / "gpec_text" / "rdcon_42893_00337_n1"
REFERENCE = json.loads((RUN / "reference.json").read_text(encoding="utf-8"))


def test_text_run_reproduces_the_netcdf_delta_prime_diagonal():
    result = read_pest3_matching_text(RUN, mode=1)

    assert [int(m) for m in result.m] == REFERENCE["m"]
    np.testing.assert_allclose(result.psi_n_rational, REFERENCE["psi_n_rational"], atol=1e-3)
    np.testing.assert_allclose(result.q_rational, REFERENCE["q_rational"], atol=1e-3)
    diagonal = [row["delta_prime_real"] for row in result.delta_prime_diagonal()]
    # Four printed digits per delta entry; Delta_prime is a sum of four of them.
    np.testing.assert_allclose(diagonal, REFERENCE["delta_prime_diagonal_real"],
                               atol=4e-3 * REFERENCE["delta_abs_max"])
    assert np.all(np.sign(diagonal) == np.sign(REFERENCE["delta_prime_diagonal_real"]))
    assert result.total1 == pytest.approx(complex(*REFERENCE["total1"]), rel=1e-3)
    assert (result.mlow, result.mhigh, result.mpert, result.mband) == (
        REFERENCE["mlow"], REFERENCE["mhigh"], REFERENCE["mpert"], REFERENCE["mband"])


def test_dcon_text_reads_least_stable_energy_first():
    text = read_dcon_text(RUN / "dcon.out")

    totals = [row["total"].real for row in text["energies"]]
    assert text["energies"][0]["isol"] == 1
    assert totals[0] == pytest.approx(REFERENCE["total1"][0], rel=1e-3)
    assert totals == sorted(totals)
    assert text["truncation"]["mpsi"] == 256
    assert len(text["singular_surfaces"]) == len(REFERENCE["m"])


def test_pest3_matrices_follow_the_left_right_sums():
    """A delta that couples only 1l-1l gives every PEST3 matrix the same diagonal entry."""
    delta = np.zeros((4, 4), dtype=complex)
    delta[0, 0] = 2.0  # 1l, 1l
    delta[1, 1] = 5.0  # 1r, 1r
    delta[1, 0] = 1.0  # 1r, 1l
    matrices = pest3_matrices_from_delta(delta)

    # rdcon_netcdf.f: Delta' = rr - rl - lr + ll, A' = rr + rl + lr + ll.
    assert matrices["Delta_prime"][0, 0] == pytest.approx(5.0 - 1.0 + 2.0)
    assert matrices["A_prime"][0, 0] == pytest.approx(5.0 + 1.0 + 2.0)
    assert matrices["Delta_prime"][1, 1] == 0


def test_pest3_matrices_are_transposed_like_the_netcdf():
    """Off-diagonal coupling lands where the xarray read of the netCDF puts it."""
    delta = np.zeros((4, 4), dtype=complex)
    delta[1, 3] = 7.0  # Fortran delta(2, 4): surface 1 right, surface 2 right
    assert pest3_matrices_from_delta(delta)["Delta_prime"][1, 0] == pytest.approx(7.0)


def test_delta_gw_reads_one_square_block():
    delta = read_delta_gw(RUN / "delta_gw.out")
    assert delta.shape == (2 * len(REFERENCE["m"]),) * 2


def test_delta_gw_rejects_a_ragged_block(tmp_path):
    (tmp_path / "delta_gw.out").write_text(" Delta matrix:\n\n  1l  1.0E+00  0.0E+00\n  1r  1.0E+00\n",
                                           encoding="utf-8")
    with pytest.raises(ValueError, match="square"):
        read_delta_gw(tmp_path / "delta_gw.out")


def test_surface_count_mismatch_is_refused(tmp_path):
    for name in ("dcon.out", "delta_gw.out"):
        (tmp_path / name).write_text((RUN / name).read_text(encoding="utf-8"), encoding="utf-8")
    text = (tmp_path / "dcon.out").read_text(encoding="utf-8").splitlines()
    first = next(i for i, line in enumerate(text) if line.strip().startswith("Singular Surfaces"))
    row = next(i for i in range(first + 1, len(text)) if text[i].split()[:1] == ["1"])
    del text[row]
    (tmp_path / "dcon.out").write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="singular surfaces"):
        read_pest3_matching_text(tmp_path, mode=1)


def test_requested_mode_must_match_the_run():
    with pytest.raises(ValueError, match="n = 1 run"):
        read_pest3_matching_text(RUN, mode=2)


def test_off_diagonal_orientation_matches_the_netcdf():
    """The coupling terms are asymmetric, so a transposition error would swap them."""
    result = read_pest3_matching_text(RUN, mode=1)
    # These two entries sum small delta terms; their text rounding is ~0.01.
    scale = 0.05
    assert result.Delta_prime[0, 1].real == pytest.approx(REFERENCE["delta_prime_01"][0], abs=scale)
    assert result.Delta_prime[1, 0].real == pytest.approx(REFERENCE["delta_prime_10"][0], abs=scale)
    assert abs(REFERENCE["delta_prime_01"][0] - REFERENCE["delta_prime_10"][0]) > 2 * scale
