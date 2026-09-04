"""Unit tests for the GPEC-suite solver-native output containers.

These live in `vaft.code.gpec` (`DconOutput` for DCON; `Pest3MatchingOutput`
shared by RDCON/STRIDE) and are what `vaft.machine_mapping.mhd_linear` reads
from (see issue #170) -- exercised directly here, independent of the
IDS-populating layer, so a JSON round-trip or netCDF-reading regression is
caught at the source.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from vaft.code.gpec import (
    DconEigenfunction,
    DconOutput,
    Pest3MatchingOutput,
    read_dcon_output,
    read_pest3_matching_output,
    read_solutions_bin,
)


def test_dcon_output_json_round_trip():
    eigenfunction = DconEigenfunction(
        m=np.array([-3, -2, -1]),
        psi=np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]),
        rho=np.array([[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]),
        q=np.array([[1.0, 1.1], [1.0, 1.1], [1.0, 1.1]]),
        xi_psi_real=np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]),
        xi_psi_imag=np.array([[-1.0, -1.1], [-2.0, -2.1], [-3.0, -3.1]]),
        v4_real=np.zeros((3, 2)),
        v4_imag=np.zeros((3, 2)),
    )
    result = DconOutput(
        n_tor=1,
        mlow=-3,
        mhigh=-1,
        mpert=3,
        mband=0,
        psi_n=np.linspace(0, 1, 5),
        m=np.array([-3, -2, -1]),
        W_p_eigenvalue=np.array([-0.1 + 0.01j]),
        W_v_eigenvalue=np.array([0.05 + 0j]),
        W_t_eigenvalue=np.array([-0.05 + 0.01j]),
        di=np.array([0.1, 0.2]),
        dr=np.array([0.01, 0.02]),
        ca1=np.array([1.0, 2.0]),
        eigenfunction=eigenfunction,
        metadata={"run_dir": "/tmp/x"},
    )

    payload = result.to_dict()
    restored = DconOutput.from_dict(payload)

    assert restored.n_tor == 1
    assert restored.mlow == -3 and restored.mhigh == -1
    np.testing.assert_allclose(restored.W_t_eigenvalue, result.W_t_eigenvalue)
    np.testing.assert_array_equal(restored.eigenfunction.m, eigenfunction.m)
    np.testing.assert_allclose(restored.eigenfunction.xi_psi_real, eigenfunction.xi_psi_real)
    assert restored.total1 == pytest.approx(-0.05 + 0.01j)
    assert restored.stable_free_boundary is False  # Re(total1) < 0


def test_dcon_output_write_read_json(tmp_path):
    result = DconOutput(n_tor=2, mlow=-1, mhigh=1, mpert=3, mband=0)
    path = result.write_json(tmp_path / "dcon_native_n2.json")
    restored = DconOutput.read_json(path)
    assert restored.n_tor == 2
    assert restored.total1 is None
    assert restored.stable_free_boundary is None


def _write_dcon_netcdf(path, *, n, mlow, mhigh, w_t):
    import xarray as xr

    mpert = mhigh - mlow + 1
    ds = xr.Dataset(
        {"W_t_eigenvalue": (("mode", "i"), [[w_t, 0.0]])},
        coords={"i": [0, 1], "mode": [1]},
        attrs={"mlow": mlow, "mhigh": mhigh, "mpert": mpert, "mband": 0, "n": n},
    )
    ds.to_netcdf(path / f"dcon_output_n{n}.nc")


def _write_solutions_bin(path, *, blocks):
    with open(path / "solutions.bin", "wb") as f:
        for steps in blocks:
            for vec7 in steps:
                payload = struct.pack("<7f", *vec7)
                f.write(struct.pack("<i", len(payload)))
                f.write(payload)
                f.write(struct.pack("<i", len(payload)))
            f.write(struct.pack("<i", 0))


def test_read_dcon_output_labels_solutions_bin_blocks_by_true_m(tmp_path):
    _write_dcon_netcdf(tmp_path, n=1, mlow=-2, mhigh=0, w_t=-0.3)
    _write_solutions_bin(
        tmp_path,
        blocks=[
            [[0.1, 0.3, 1.0, 10.0, -10.0, 0.0, 0.0]],  # ipert=0 -> m=-2
            [[0.1, 0.3, 1.0, 20.0, -20.0, 0.0, 0.0]],  # ipert=1 -> m=-1
            [[0.1, 0.3, 1.0, 30.0, -30.0, 0.0, 0.0]],  # ipert=2 -> m=0
        ],
    )

    result = read_dcon_output(tmp_path, mode=1)

    assert result.mlow == -2 and result.mhigh == 0 and result.mpert == 3
    assert result.total1 == pytest.approx(-0.3 + 0j)
    assert result.eigenfunction.m.tolist() == [-2, -1, 0]
    assert result.eigenfunction.xi_psi_real[:, 0].tolist() == [10.0, 20.0, 30.0]
    assert "eigenfunction_mpert_mismatch" not in result.metadata


def test_read_dcon_output_without_solutions_bin_has_no_eigenfunction(tmp_path):
    _write_dcon_netcdf(tmp_path, n=1, mlow=-2, mhigh=0, w_t=0.1)
    result = read_dcon_output(tmp_path, mode=1)
    assert result.eigenfunction is None
    assert result.stable_free_boundary is True

# ---------------------------------------------------------------------------
# Quantities derived from the eigenfunction.  DCON's eigenvector normalization
# is arbitrary, so what is derivable from it at face value is exactly what that
# normalization cannot change.
# ---------------------------------------------------------------------------


def _eigenfunction(xi_real, *, m, q):
    xi_real = np.asarray(xi_real, dtype=float)
    n_m, n_psi = xi_real.shape
    return DconEigenfunction(
        m=np.asarray(m, dtype=int),
        psi=np.tile(np.linspace(0.1, 0.9, n_psi), (n_m, 1)),
        rho=np.zeros((n_m, n_psi)),
        q=np.tile(np.asarray(q, dtype=float), (n_m, 1)),
        xi_psi_real=xi_real,
        xi_psi_imag=np.zeros((n_m, n_psi)),
        v4_real=np.zeros((n_m, n_psi)),
        v4_imag=np.zeros((n_m, n_psi)),
    )


def _output(eigenfunction, *, n_tor=1):
    return DconOutput(
        n_tor=n_tor,
        mlow=int(eigenfunction.m[0]),
        mhigh=int(eigenfunction.m[-1]),
        mpert=int(eigenfunction.m.size),
        mband=0,
        eigenfunction=eigenfunction,
    )


def test_m_pol_dominant_is_the_true_poloidal_number_not_the_block_index():
    eigenfunction = _eigenfunction(
        [[1.0, 1.0], [0.0, 9.0], [2.0, 2.0]], m=[-3, -2, -1], q=[1.0, 2.0]
    )

    # Block index 1 is the largest; its physical label is m = -2.
    assert _output(eigenfunction).m_pol_dominant == -2


def test_m_pol_dominant_is_invariant_under_the_eigenvector_normalization():
    """This invariance is the whole reason the field can be written at all.

    `match/ideal.f:318-325` fixes the eigenfunction's amplitude by an arbitrary
    normalization, further rescaled by DCON's `ucrit` during integration, so no
    absolute amplitude is reportable. That factor is global -- one scalar
    multiplying every harmonic -- so which harmonic is largest survives it, and
    that is what `m_pol_dominant` reports.
    """
    xi = np.array([[1.0, 1.0], [0.0, 9.0], [2.0, 2.0]])
    baseline = _output(_eigenfunction(xi, m=[-3, -2, -1], q=[1.0, 2.0]))
    rescaled = _output(_eigenfunction(xi * 1e6, m=[-3, -2, -1], q=[1.0, 2.0]))

    assert baseline.m_pol_dominant == rescaled.m_pol_dominant == -2


def test_m_pol_dominant_is_none_without_an_eigenfunction():
    assert DconOutput(n_tor=1, mlow=-1, mhigh=1, mpert=3, mband=0).m_pol_dominant is None


def test_b_normal_reproduces_matchs_own_singular_factor():
    """`b = i (m - n q) xi` -- match/ideal.f:372, which computes it and never writes it.

    Recomputing it here rather than parsing it is what makes the perturbed
    normal field available at all; everything the formula needs is already in
    solutions.bin (q is its third column, m comes from the run's mlow).
    """
    eigenfunction = _eigenfunction([[1.5, 2.0], [3.0, 4.0]], m=[-2, -1], q=[1.0, 2.0])

    b = eigenfunction.b_normal(n_tor=1)

    expected = 1j * np.array([[(-2 - 1 * 1.0) * 1.5, (-2 - 1 * 2.0) * 2.0],
                              [(-1 - 1 * 1.0) * 3.0, (-1 - 1 * 2.0) * 4.0]])
    np.testing.assert_allclose(b, expected)
    # It vanishes on the resonant surface, where m = n q.
    resonant = _eigenfunction([[5.0]], m=[2], q=[2.0]).b_normal(n_tor=1)
    assert resonant[0, 0] == pytest.approx(0.0)




def _write_resistive_netcdf(path, *, module, n, mlow, mhigh, m_values, diag):
    import xarray as xr

    mpert = mhigh - mlow + 1
    msing = len(m_values)
    delta_prime = np.zeros((msing, msing, 2), dtype=float)
    a_prime = np.zeros((msing, msing, 2), dtype=float)
    for i, value in enumerate(diag):
        delta_prime[i, i, 0] = value.real
        delta_prime[i, i, 1] = value.imag
    delta_prime[0, 1, 0] = 0.777  # off-diagonal coupling term, no IDS home
    ds = xr.Dataset(
        {
            "Delta_prime": (("r", "r_prime", "i"), delta_prime),
            "A_prime": (("r", "r_prime", "i"), a_prime),
            "r": (("r",), m_values),
            "psi_n_rational": (("r",), [0.1 * (i + 1) for i in range(msing)]),
            "q_rational": (("r",), [float(m) / n for m in m_values]),
        },
        coords={"i": [0, 1]},
        attrs={"mlow": mlow, "mhigh": mhigh, "mpert": mpert, "mband": 0, "n": n},
    )
    ds.to_netcdf(path / f"{module}_output_n{n}.nc")


@pytest.mark.parametrize("solver", ["rdcon", "stride"])
def test_read_pest3_matching_output_shares_schema_across_rdcon_and_stride(tmp_path, solver):
    """Confirms the shared container is justified in practice, not just by
    comparing the Fortran source: the same reader/schema genuinely works for
    both solvers' netCDF."""
    _write_resistive_netcdf(
        tmp_path, module=solver, n=1, mlow=-8, mhigh=16, m_values=[3, 4], diag=[1.0 + 0.5j, -2.0 + 0j]
    )

    result = read_pest3_matching_output(tmp_path, solver=solver, mode=1)

    assert result.solver == solver
    assert result.msing == 2
    diag = result.delta_prime_diagonal()
    assert diag[0] == {"m": 3, "n": 1, "psi_n": pytest.approx(0.1), "q": pytest.approx(3.0), "delta_prime_real": pytest.approx(1.0), "delta_prime_imag": pytest.approx(0.5)}
    # The off-diagonal coupling term has no IDS home and only survives here.
    assert result.Delta_prime[0, 1].real == pytest.approx(0.777)


def test_pest3_matching_output_json_round_trip():
    result = Pest3MatchingOutput(
        solver="rdcon",
        n_tor=1,
        mlow=-8,
        mhigh=16,
        mpert=25,
        mband=0,
        m=np.array([3, 4]),
        psi_n_rational=np.array([0.3, 0.4]),
        q_rational=np.array([3.0, 4.0]),
        Delta_prime=np.array([[1.0 + 0.5j, 0.1j], [0.2, -2.0]]),
        total1=-1.0 - 0.5j,
    )
    restored = Pest3MatchingOutput.from_dict(result.to_dict())
    np.testing.assert_allclose(restored.Delta_prime, result.Delta_prime)
    assert restored.total1 == pytest.approx(-1.0 - 0.5j)
    assert restored.stable is False


def test_read_pest3_matching_output_rejects_unknown_solver(tmp_path):
    with pytest.raises(ValueError, match="dcon"):
        read_pest3_matching_output(tmp_path, solver="dcon", mode=1)


# --- regressions found in review of the #170 implementation --------------------

def test_eigenvalues_are_selected_by_mode_label_not_array_position():
    """The netCDF's `mode` coordinate labels the eigenvalues; DCON's
    least-stable one is label 1. Selecting positionally would silently report
    a different eigenvalue for any file whose `mode` coordinate is not in
    ascending order."""
    result = DconOutput(
        n_tor=1,
        mlow=-1,
        mhigh=1,
        mpert=3,
        mband=0,
        mode=np.array([2, 1]),
        W_t_eigenvalue=np.array([9.9 + 0j, -0.5 + 0j]),
        W_p_eigenvalue=np.array([8.8 + 0j, -0.4 + 0j]),
        W_v_eigenvalue=np.array([7.7 + 0j, -0.3 + 0j]),
    )
    assert result.total1 == pytest.approx(-0.5 + 0j)
    assert result.plasma1 == pytest.approx(-0.4 + 0j)
    assert result.vacuum1 == pytest.approx(-0.3 + 0j)
    assert result.stable_free_boundary is False


def test_eigenvalues_fall_back_to_first_entry_without_a_mode_coordinate():
    result = DconOutput(
        n_tor=1, mlow=-1, mhigh=1, mpert=3, mband=0, W_t_eigenvalue=np.array([-0.5 + 0j, 9.9 + 0j])
    )
    assert result.total1 == pytest.approx(-0.5 + 0j)


def test_mode_coordinate_survives_the_json_round_trip():
    result = DconOutput(
        n_tor=1, mlow=-1, mhigh=1, mpert=3, mband=0,
        mode=np.array([2, 1]), W_t_eigenvalue=np.array([9.9 + 0j, -0.5 + 0j]),
    )
    restored = DconOutput.from_dict(result.to_dict())
    assert restored.mode.tolist() == [2, 1]
    assert restored.total1 == pytest.approx(-0.5 + 0j)


@pytest.mark.parametrize("bad_length", [-8, 7])
def test_a_malformed_fortran_record_length_raises_instead_of_desyncing(tmp_path, bad_length):
    """A negative or non-multiple-of-4 marker means this is not a solutions.bin
    written by `match`; reading on from it would silently produce garbage
    harmonics rather than an error."""
    with open(tmp_path / "solutions.bin", "wb") as handle:
        handle.write(struct.pack("<i", bad_length))

    with pytest.raises(ValueError, match="malformed Fortran record-length marker"):
        read_solutions_bin(tmp_path / "solutions.bin", mlow=0)


def test_an_mpert_mismatch_warns_and_is_recorded_rather_than_passing_silently(tmp_path):
    """solutions.bin's block count should equal the netCDF's mpert; when it
    does not, the m labels derived from mlow are not trustworthy, so this must
    be visible rather than persisted as if it were a clean parse."""
    _write_dcon_netcdf(tmp_path, n=1, mlow=-2, mhigh=5, w_t=-0.3)  # mpert=8
    _write_solutions_bin(tmp_path, blocks=[[[0.1, 0.3, 1.0, 10.0, -10.0, 0.0, 0.0]]])  # 1 block

    with pytest.warns(RuntimeWarning, match="may be mislabeled"):
        result = read_dcon_output(tmp_path, mode=1)

    assert result.metadata["eigenfunction_mpert_mismatch"] == {
        "solutions_bin_n_ipert": 1,
        "netcdf_mpert": 8,
    }
