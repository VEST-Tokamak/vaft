"""Unit tests for the GPEC-suite solver-native output containers.

These live in `vaft.code.gpec` (`DconOutput` for DCON; `Pest3MatchingOutput`
shared by RDCON/STRIDE) and are what `vaft.machine_mapping.mhd_linear` reads
from (see issue #170) -- exercised directly here, independent of the
IDS-populating layer, so a JSON round-trip or netCDF-reading regression is
caught at the source.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from gpec_nc_fixtures import DEFAULT_DCON_EQUILIBRIUM, write_dcon_output_nc

from vaft.code.gpec import (
    SCAN_COLUMNS,
    DconEigenfunction,
    DconOutput,
    Pest3MatchingOutput,
    dcon_scan_row,
    read_dcon_output,
    read_dcon_scan,
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


# ---------------------------------------------------------------------------
# The equilibrium summary, coordinate system, profiles and edge scan DCON writes
# alongside the eigenvalues (dcon/dcon_netcdf.f:90-215).
# ---------------------------------------------------------------------------


def test_dcon_output_carries_the_equilibrium_summary_as_plain_python_scalars(tmp_path):
    """The netCDF globals parse into `DconEquilibrium` as JSON-serializable scalars.

    The type assertion is the point: xarray hands these back as `np.float64` /
    `np.int32`, which `json.dumps` cannot serialize, so an uncoerced value would
    pass every value check here and only fail later, when a real run reaches
    `write_json` and the sidecar -- the eigenfunction's only lossless home --
    silently fails to be written.
    """
    write_dcon_output_nc(tmp_path, equilibrium=True, coordinates=True)
    result = read_dcon_output(tmp_path, mode=1)

    assert result.equilibrium is not None
    assert result.equilibrium.q0 == pytest.approx(DEFAULT_DCON_EQUILIBRIUM["q0"])
    assert result.equilibrium.li3 == pytest.approx(DEFAULT_DCON_EQUILIBRIUM["li3"])
    assert result.equilibrium.betan == pytest.approx(DEFAULT_DCON_EQUILIBRIUM["betan"])
    assert result.coordinates is not None
    assert result.coordinates.jacobian == "hamada"
    assert result.coordinates.mpsi == 128

    assert type(result.equilibrium.q0) is float
    assert type(result.coordinates.mpsi) is int
    assert type(result.coordinates.jacobian) is str


def test_a_file_without_the_summary_attributes_parses_with_empty_blocks(tmp_path):
    """A run from a GPEC build that writes fewer globals must parse, not raise."""
    write_dcon_output_nc(tmp_path, equilibrium=False, coordinates=False)
    result = read_dcon_output(tmp_path, mode=1)

    assert result.equilibrium is None
    assert result.coordinates is None
    assert result.edge_scan is None
    assert result.qlim is None
    assert result.mlow == -2  # the provenance it does carry is still read


def test_qlim_is_a_solver_truncation_and_never_an_equilibrium_scalar(tmp_path):
    """`qlim` belongs to the run, not the equilibrium, and the split is enforced here.

    Under `sas_flag` DCON rounds the edge limit onto the mode's own rational
    spacing (`qlim=(INT(nn*qlim)+dmlim)/nn`, dcon/sing.f:186) after capping it
    at the `qhigh` VAFT itself templates into dcon.in -- so it is n-dependent
    and configuration-dependent, and a reader who takes it for the equilibrium's
    edge safety factor `qa` gets a different number for the same plasma.
    """
    write_dcon_output_nc(tmp_path, equilibrium=True)
    result = read_dcon_output(tmp_path, mode=1)

    assert result.qlim == pytest.approx(8.0)
    assert result.psilim == pytest.approx(0.994)
    assert not hasattr(result.equilibrium, "qlim")
    assert not hasattr(result.equilibrium, "psilim")
    # qa is the equilibrium's own edge q and is a different value entirely.
    assert result.equilibrium.qa == pytest.approx(DEFAULT_DCON_EQUILIBRIUM["qa"])


def test_the_one_dimensional_profiles_are_read_on_the_psi_n_grid(tmp_path):
    written = write_dcon_output_nc(tmp_path, profiles=True)
    result = read_dcon_output(tmp_path, mode=1)

    for name in ("f", "mu0p", "dvdpsi", "q"):
        assert getattr(result, name) is not None, name
        assert getattr(result, name).shape == written["psi_n"].shape, name


def test_the_edge_scan_is_read_when_present_and_absent_by_default(tmp_path):
    """`dW_edge` vs `q_edge` is DCON's own stability-limit curve, and is optional.

    DCON writes it only when `psiedge < psilim` (dcon/sing.f:224), and VAFT's
    packaged dcon.in ships `psiedge=1` against `psihigh=0.994` -- so the absent
    case is the *default* one, not an error.
    """
    scanned = tmp_path / "scanned"
    scanned.mkdir()
    write_dcon_output_nc(scanned, edge_scan=True)
    result = read_dcon_output(scanned, mode=1)

    assert result.edge_scan is not None
    assert result.edge_scan.q.shape == result.edge_scan.dW.shape == result.edge_scan.psi_n.shape
    assert np.iscomplexobj(result.edge_scan.dW)
    # The curve crosses zero: that crossing is what makes it a limit curve.
    assert result.edge_scan.dW.real[0] > 0 > result.edge_scan.dW.real[-1]

    plain = tmp_path / "plain"
    plain.mkdir()
    write_dcon_output_nc(plain)
    assert read_dcon_output(plain, mode=1).edge_scan is None


def test_the_v2_sidecar_round_trips_every_new_block(tmp_path):
    write_dcon_output_nc(tmp_path, equilibrium=True, coordinates=True, profiles=True, edge_scan=True)
    result = read_dcon_output(tmp_path, mode=1)

    restored = DconOutput.read_json(result.write_json(tmp_path / "dcon_native_n1.json"))

    assert restored.to_dict()["schema_version"] == 2
    assert restored.equilibrium == result.equilibrium
    assert restored.coordinates == result.coordinates
    assert restored.qlim == result.qlim
    np.testing.assert_allclose(restored.q, result.q)
    np.testing.assert_allclose(restored.edge_scan.dW, result.edge_scan.dW)


def test_a_v1_sidecar_still_loads_under_v2_code(tmp_path):
    """Sidecars already on disk predate every field above and must keep loading."""
    payload = {
        "schema": "vaft.code.gpec.DconOutput",
        "schema_version": 1,
        "n_tor": 2, "mlow": -1, "mhigh": 1, "mpert": 3, "mband": 0,
        "W_t_eigenvalue": {"real": [-0.5, 0.2, 0.9], "imag": [0.0, 0.0, 0.0]},
        "mode": [1, 2, 3],
        "metadata": {"run_dir": "/somewhere/old"},
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = DconOutput.read_json(path)

    assert restored.n_tor == 2
    assert restored.total1 == pytest.approx(-0.5)
    assert restored.equilibrium is None
    assert restored.edge_scan is None


def test_a_sidecar_from_a_newer_vaft_refuses_rather_than_dropping_fields(tmp_path):
    """Reading a newer payload would silently discard whatever it added.

    A round trip through this code would then write those fields away, so the
    version skew must surface as an error at read time rather than as quiet
    data loss at the next write.
    """
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({
            "schema": "vaft.code.gpec.DconOutput", "schema_version": 3,
            "n_tor": 1, "mlow": 0, "mhigh": 0, "mpert": 1, "mband": 0,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="newer than this VAFT understands"):
        DconOutput.read_json(path)


# ---------------------------------------------------------------------------
# The scan harvest: one flat row per run, across a case directory.
# ---------------------------------------------------------------------------


def test_a_scan_row_has_every_column_even_when_the_run_carries_none_of_them():
    """A scan is a table; a table whose columns vary per row cannot be compared.

    Missing values are `None` rather than absent, so a consumer indexes columns
    instead of guarding every access -- and `None` rather than NaN, so "DCON did
    not report this" stays distinguishable from a computed non-number.
    """
    row = dcon_scan_row(DconOutput(n_tor=2, mlow=-1, mhigh=1, mpert=3, mband=0))

    assert sorted(row) == sorted(SCAN_COLUMNS)
    assert row["n_tor"] == 2
    assert row["q0"] is None and row["betan"] is None
    assert not any(isinstance(value, float) and np.isnan(value) for value in row.values())


def test_a_scan_row_carries_the_equilibrium_summary_and_the_stability_verdict(tmp_path):
    write_dcon_output_nc(tmp_path, equilibrium=True, w_t=-0.42)
    result = read_dcon_output(tmp_path, mode=1)

    row = dcon_scan_row(result, shot=39915, time_ms=325.0)

    assert row["shot"] == 39915
    assert row["time_ms"] == pytest.approx(325.0)
    assert row["q95"] == pytest.approx(DEFAULT_DCON_EQUILIBRIUM["q95"])
    assert row["li3"] == pytest.approx(DEFAULT_DCON_EQUILIBRIUM["li3"])
    assert row["total1_real"] == pytest.approx(-0.42)
    assert row["stable_free_boundary"] is False
    # qlim is the run's truncation, kept apart from the equilibrium's own qa.
    assert row["qlim"] == pytest.approx(8.0)
    assert row["qa"] == pytest.approx(DEFAULT_DCON_EQUILIBRIUM["qa"])


def test_read_dcon_scan_walks_the_layout_the_runner_writes(tmp_path):
    """`<workdir>/<time>/<module>/nn=<mode>/` -- the same layout `module_dir` builds."""
    for label, w_t in (("00320", -0.1), ("00325", -0.9), ("00330", 0.3)):
        run_dir = tmp_path / label / "dcon" / "nn=1"
        run_dir.mkdir(parents=True)
        write_dcon_output_nc(run_dir, equilibrium=True, w_t=w_t)

    rows = read_dcon_scan(tmp_path, modes=(1,), shot=39915)

    assert [row["time_ms"] for row in rows] == [320.0, 325.0, 330.0]
    assert [row["stable_free_boundary"] for row in rows] == [False, False, True]
    assert {row["shot"] for row in rows} == {39915}


def test_read_dcon_scan_warns_about_a_bad_cell_and_keeps_the_rest(tmp_path):
    """One unreadable run costs that cell, not the scan -- and never silently.

    A quietly shorter table is indistinguishable from a shorter run list, which
    is exactly the confusion that makes a scan untrustworthy.
    """
    for label in ("00320", "00325"):
        run_dir = tmp_path / label / "dcon" / "nn=1"
        run_dir.mkdir(parents=True)
        write_dcon_output_nc(run_dir, equilibrium=True)
    (tmp_path / "00325" / "dcon" / "nn=1" / "dcon_output_n1.nc").write_bytes(b"not a netcdf")

    with pytest.warns(RuntimeWarning, match="skipping unreadable DCON output"):
        rows = read_dcon_scan(tmp_path, modes=(1,))

    assert [row["time_ms"] for row in rows] == [320.0]


def test_the_scan_takes_identity_from_the_directory_not_the_file(tmp_path):
    """DCON writes shot/time as INT(...) of what it was told -- 0 for a VEST g-file.

    Trusting those would label every row of a scan `shot=0, time=0`, so they are
    kept only as metadata to cross-check against.
    """
    run_dir = tmp_path / "00325" / "dcon" / "nn=1"
    run_dir.mkdir(parents=True)
    write_dcon_output_nc(run_dir, equilibrium={"shot": 0, "time": 0})

    (row,) = read_dcon_scan(tmp_path, modes=(1,), shot=39915)

    assert row["shot"] == 39915
    assert row["time_ms"] == pytest.approx(325.0)
    assert read_dcon_output(run_dir, mode=1).metadata["nc_shot"] == 0
