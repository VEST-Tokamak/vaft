"""`update_core_profiles_global_quantities_volume_average` produces ion
volume averages for an ODS, each slice with its own plasma mask
(cold review 0.8.0 workflows-and-pipelines P1 / F1).

The ion branch only ran when the ion container was a plain dict or list --
never for the ODS struct array a real core_profiles carries -- and, when
forced, stored the result as ``gq['ion'] = []``, which OMAS's coordinate
check refuses. Inside that dead branch an ion-only slice reused the previous
slice's plasma mask (75 % high in the verifier's repro) or hit an unbound
name.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("omas")

from omas import ODS

from vaft.omas.update import update_core_profiles_global_quantities_volume_average
from vaft.process.equilibrium import plasma_cell_weights, volume_average

R = np.linspace(0.1, 0.9, 41)
Z = np.linspace(-0.6, 0.6, 61)
R0 = 0.5
PSI_N_1D = np.linspace(0.0, 1.0, 50)
THETA = np.linspace(0.0, 2.0 * np.pi, 73)


def _equilibrium_slice(ts, t, a):
    ts["time"] = t
    Rm, Zm = np.meshgrid(R, Z, indexing="ij")
    ts["profiles_2d.0.grid.dim1"] = R
    ts["profiles_2d.0.grid.dim2"] = Z
    ts["profiles_2d.0.psi"] = ((Rm - R0) ** 2 + Zm**2) / a**2  # psi_N = r^2 / a^2
    ts["global_quantities.psi_axis"] = 0.0
    ts["global_quantities.psi_boundary"] = 1.0
    ts["profiles_1d.psi_norm"] = PSI_N_1D
    ts["profiles_1d.rho_tor_norm"] = np.sqrt(PSI_N_1D)
    ts["boundary.outline.r"] = R0 + a * np.cos(THETA)
    ts["boundary.outline.z"] = a * np.sin(THETA)


def _core_slice(ts, t, electrons):
    ts["time"] = t
    rho = np.linspace(0.0, 1.0, 30)
    ts["grid.rho_tor_norm"] = rho
    prof = 1.0e19 * (1.0 - rho**2)
    if electrons:
        ts["electrons.density"] = prof
        ts["electrons.temperature"] = prof
    ts["ion.0.density"] = prof
    ts["ion.0.temperature"] = prof


def _expected(a):
    Rm, Zm = np.meshgrid(R, Z, indexing="ij")
    psi_n = ((Rm - R0) ** 2 + Zm**2) / a**2
    prof = 1.0e19 * (1.0 - psi_n)
    weights = plasma_cell_weights(R, Z, psi_n, R0 + a * np.cos(THETA), a * np.sin(THETA))
    return volume_average(prof, psi_n, R, Z, weights=weights)[0], weights, psi_n, prof


@pytest.mark.parametrize("consistency_check", [True, False])
def test_an_ion_only_slice_gets_its_own_mask_on_an_ods(consistency_check):
    ods = ODS(consistency_check=consistency_check)
    _equilibrium_slice(ods["equilibrium.time_slice.0"], 0.30, 0.10)
    _core_slice(ods["core_profiles.profiles_1d.0"], 0.30, electrons=True)
    # Twice the minor radius, and no electron profiles at all.
    _equilibrium_slice(ods["equilibrium.time_slice.1"], 0.31, 0.20)
    _core_slice(ods["core_profiles.profiles_1d.1"], 0.31, electrons=False)

    with contextlib.redirect_stdout(io.StringIO()):
        update_core_profiles_global_quantities_volume_average(ods)

    gq = ods["core_profiles.global_quantities"]
    n_e = np.asarray(gq["n_e_volume_average"], float)
    assert n_e.shape == (2,) and np.isfinite(n_e[0]) and np.isnan(n_e[1])

    n_i = np.asarray(gq["ion.0.n_i_volume_average"], float)
    t_i = np.asarray(gq["ion.0.t_i_volume_average"], float)
    assert n_i.shape == (2,) and t_i.shape == (2,)

    # Slice 0: the same profile as the electrons on the same mask, so the
    # averages are identical (a = 0.10 m spans five cells, too coarse to pin
    # against the analytic value; slice 1 at a = 0.20 m is).
    assert n_i[0] == pytest.approx(n_e[0], rel=1e-12)
    expected_1, _, psi_n_1, prof_1 = _expected(0.20)
    stale = volume_average(prof_1, psi_n_1, R, Z, weights=_expected(0.10)[1])[0]
    assert n_i[1] == pytest.approx(expected_1, rel=2e-3)
    assert n_i[1] != pytest.approx(stale, rel=0.1), "slice 1 must not use slice 0's mask"
    np.testing.assert_allclose(t_i, n_i)


def test_a_single_ion_only_slice_is_averaged_not_nan():
    ods = ODS(consistency_check=False)
    _equilibrium_slice(ods["equilibrium.time_slice.0"], 0.30, 0.15)
    _core_slice(ods["core_profiles.profiles_1d.0"], 0.30, electrons=False)
    with contextlib.redirect_stdout(io.StringIO()):
        update_core_profiles_global_quantities_volume_average(ods)
    n_i = np.asarray(ods["core_profiles.global_quantities.ion.0.n_i_volume_average"], float)
    assert n_i.shape == (1,)
    assert n_i[0] == pytest.approx(_expected(0.15)[0], rel=2e-3)
