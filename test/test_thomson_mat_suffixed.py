"""Tests for the shot-suffixed NeTe_{shot}.mat Thomson schema and sigma sanitizing."""

import numpy as np
import pytest
import scipy.io as sio

pytest.importorskip("omas")
from omas import ODS

from vaft.machine_mapping.thomson_scattering import thomson_scattering

SHOT = 90001


def _write_suffixed_mat(path, shot=SHOT):
    """7-channel, 3-time suffixed MAT mimicking NeTe_48224.mat (incl. dtypes)."""
    n_t, n_ch = 3, 7
    rng = np.random.default_rng(0)
    te = 50.0 + 100.0 * rng.random((n_t, n_ch))
    ne = 1e18 * (1.0 + rng.random((n_t, n_ch)))
    sigma_te = np.full((n_t, n_ch), 8.0, dtype=complex)
    sigma_ne = np.full((n_t, n_ch), 2e17)
    # poisoned entries: purely-imaginary sigmaTe, exactly-zero sigmaNe
    sigma_te[1, 2] = 0.0 + 5.0j
    sigma_ne[2, 4] = 0.0
    sio.savemat(
        str(path),
        {
            f"tsTime_{shot}": np.array([[299, 300, 301]], dtype=np.uint16),
            f"Rposition_{shot}": np.array([[475, 425, 370, 310, 255, 475, 650]], dtype=np.uint16),
            f"Te_{shot}": te,
            f"Ne_{shot}": ne,
            f"sigmaTe_{shot}": sigma_te,
            f"sigmaNe_{shot}": sigma_ne,
        },
    )
    return te, ne


def test_suffixed_schema_loads_seven_channels(tmp_path):
    mat = tmp_path / f"NeTe_{SHOT}.mat"
    te, ne = _write_suffixed_mat(mat)

    ods = ODS()
    thomson_scattering(ods, SHOT, mat_file=str(mat))

    assert len(ods["thomson_scattering.channel"]) == 7
    # positions come from the in-file Rposition (mm -> m), incl. the edge channel
    assert ods["thomson_scattering.channel.5.position.r"] == pytest.approx(0.475)
    assert ods["thomson_scattering.channel.6.position.r"] == pytest.approx(0.650)
    # time converted ms -> s
    np.testing.assert_allclose(
        ods["thomson_scattering.time"], np.array([0.299, 0.300, 0.301])
    )
    # data round-trips (real part)
    np.testing.assert_allclose(
        ods["thomson_scattering.channel.6.n_e.data"], ne[:, 6]
    )
    np.testing.assert_allclose(
        ods["thomson_scattering.channel.0.t_e.data"], te[:, 0]
    )


def test_suffixed_schema_masks_invalid_sigma(tmp_path):
    mat = tmp_path / f"NeTe_{SHOT}.mat"
    _write_suffixed_mat(mat)

    ods = ODS()
    thomson_scattering(ods, SHOT, mat_file=str(mat))

    te_err = np.asarray(ods["thomson_scattering.channel.2.t_e.data_error_upper"], float)
    ne_err = np.asarray(ods["thomson_scattering.channel.4.n_e.data_error_upper"], float)
    # purely-imaginary sigmaTe at (t=1, ch=2) and zero sigmaNe at (t=2, ch=4) -> NaN
    assert np.isnan(te_err[1])
    assert np.isnan(ne_err[2])
    # untouched entries stay finite and positive
    assert np.isfinite(te_err[0]) and te_err[0] > 0
    assert np.isfinite(ne_err[0]) and ne_err[0] > 0


def test_suffixed_schema_masks_dead_channel_sentinels(tmp_path):
    """Te=0.1 eV / Ne=1e17 sentinel entries (dead channels) become NaN."""
    mat = tmp_path / f"NeTe_{SHOT}.mat"
    n_t, n_ch = 3, 7
    te = np.full((n_t, n_ch), 80.0)
    ne = np.full((n_t, n_ch), 5e18)
    te[1, 5] = 0.1   # dead Te sentinel
    ne[1, 5] = 1e17  # dead Ne sentinel
    sio.savemat(
        str(mat),
        {
            f"tsTime_{SHOT}": np.array([[299, 300, 301]], dtype=np.uint16),
            f"Rposition_{SHOT}": np.array([[475, 425, 370, 310, 255, 475, 650]], dtype=np.uint16),
            f"Te_{SHOT}": te,
            f"Ne_{SHOT}": ne,
            f"sigmaTe_{SHOT}": np.full((n_t, n_ch), 0.004),
            f"sigmaNe_{SHOT}": np.full((n_t, n_ch), 4e15),
        },
    )
    ods = ODS()
    thomson_scattering(ods, SHOT, mat_file=str(mat))
    te_ch5 = np.asarray(ods["thomson_scattering.channel.5.t_e.data"], float)
    ne_ch5 = np.asarray(ods["thomson_scattering.channel.5.n_e.data"], float)
    assert np.isnan(te_ch5[1]) and np.isnan(ne_ch5[1])
    assert np.isfinite(te_ch5[0]) and np.isfinite(ne_ch5[0])


def test_suffixed_filename_found_without_mat_file(tmp_path):
    mat = tmp_path / f"NeTe_{SHOT}.mat"
    _write_suffixed_mat(mat)

    ods = ODS()
    thomson_scattering(ods, SHOT, data_root=str(tmp_path))
    assert len(ods["thomson_scattering.channel"]) == 7
