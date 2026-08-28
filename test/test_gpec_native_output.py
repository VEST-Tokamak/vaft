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
