"""The volume beta and its normalized form (#1691).

``beta_B = 2 mu0 <p>_V / <B^2>_V`` divides by the total field energy inside the
plasma (Troyon 1984; Menard et al. 2004), not by the vacuum field at one radius
as the toroidal beta does. These tests pin the closed forms, the distinctness
from ``beta_tor``/``beta_normal``, and the summary columns on the packaged sample.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.constants import mu_0 as MU0

from vaft.formula.equilibrium import (
    beta_normal_from_beta_tor,
    beta_normal_from_beta_volume,
    beta_toroidal_from_p_B0,
    beta_volume_from_p_B2,
)


def test_the_volume_beta_is_the_ratio_of_the_averages():
    pressure, b2 = 2.0e3, 0.09
    assert beta_volume_from_p_B2(pressure, b2) == pytest.approx(2 * MU0 * pressure / b2)


def test_with_a_uniform_field_the_volume_beta_is_the_toroidal_beta():
    # <B^2> = B0^2: the two definitions coincide only here
    pressure, b0 = 5.0e3, 0.3
    assert beta_volume_from_p_B2(pressure, b0**2) == pytest.approx(beta_toroidal_from_p_B0(pressure, b0))


def test_the_normalized_volume_beta_uses_the_conventional_normalization():
    beta, a, b0, ip = 0.02, 0.3, 0.15, 1.0e5
    assert beta_normal_from_beta_volume(beta, a, b0, ip) == pytest.approx(beta_normal_from_beta_tor(beta, a, b0, ip))
    assert beta_normal_from_beta_volume(beta, a, -b0, -ip) == pytest.approx(100 * beta * a * b0 / 0.1)


@pytest.mark.parametrize("b2", [0.0, -1.0, float("nan")])
def test_a_non_positive_field_energy_is_refused(b2):
    with pytest.raises(ValueError):
        beta_volume_from_p_B2(1.0e3, b2)


def _sample():
    pytest.importorskip("omas")
    from vaft.omas.sample import sample_ods

    try:
        return sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")


def test_the_equilibrium_toroidal_field_reduces_to_the_vacuum_one_for_a_flat_f():
    from vaft.omas.process_wrapper import compute_magnetic_energy

    ods = copy.deepcopy(_sample())
    ts = ods["equilibrium.time_slice"][0]
    # the reference the function's vacuum field uses: global b0 / major_radius, else the vacuum b0 / r0
    # (checked with `in` first: reading a missing OMAS path creates it)
    b0 = (float(ts["global_quantities.b0"]) if "global_quantities.b0" in ts
          else float(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).flat[0]))
    r0 = (float(ts["global_quantities.major_radius"]) if "global_quantities.major_radius" in ts
          else float(np.asarray(ods["equilibrium.vacuum_toroidal_field.r0"], float).flat[0]))
    f_vacuum = b0 * r0
    ts["profiles_1d.f"] = np.full(len(ts["profiles_1d.psi"]), f_vacuum)
    vacuum = compute_magnetic_energy(ods, time_slice=0)
    equilibrium = compute_magnetic_energy(ods, time_slice=0, toroidal_field="equilibrium")
    assert equilibrium == pytest.approx(vacuum, rel=1e-12)
    with pytest.raises(ValueError):
        compute_magnetic_energy(ods, time_slice=0, toroidal_field="paramagnetic")


def test_the_equilibrium_toroidal_energy_matches_the_independent_f_psi_field():
    """F(psi) interpolation, psi units and ordering, checked against the 2-D field
    writer, which interpolates F(psi) on its own path (bicubic psi spline)."""
    from vaft.omas.process_wrapper import compute_magnetic_energy
    from vaft.omas.update import update_equilibrium_profiles_2d_b_field
    from vaft.process.equilibrium import fractional_cell_weights_from_boundary

    ods = copy.deepcopy(_sample())
    ts = ods["equilibrium.time_slice"][0]
    update_equilibrium_profiles_2d_b_field(ods, time_slice=0)
    r = np.asarray(ts["profiles_2d.0.grid.dim1"], float)
    z = np.asarray(ts["profiles_2d.0.grid.dim2"], float)
    b_tor = np.asarray(ts["profiles_2d.0.b_field_tor"], float)
    if b_tor.shape != (r.size, z.size):
        b_tor = b_tor.T
    rm, zm = np.meshgrid(r, z, indexing="ij")
    weights = fractional_cell_weights_from_boundary(
        rm, zm, np.asarray(ts["boundary.outline.r"], float), np.asarray(ts["boundary.outline.z"], float),
        samples_per_axis=5)
    d_volume = 2 * np.pi * rm * np.outer(np.gradient(r), np.gradient(z)) * weights
    independent = float(np.nansum(b_tor**2 / (2 * MU0) * d_volume))
    ours = compute_magnetic_energy(ods, time_slice=0, components="toroidal", toroidal_field="equilibrium")
    assert ours == pytest.approx(independent, rel=0.01)
    # and the real F(psi) is not the vacuum field: VEST's ohmic plasma is paramagnetic
    f = np.abs(np.asarray(ts["profiles_1d.f"], float))
    vacuum = compute_magnetic_energy(ods, time_slice=0, components="toroidal")
    assert (ours > vacuum) == (f[0] > f[-1])


def test_the_summary_reports_the_volume_beta_as_a_separate_definition():
    from vaft.compat import trapz_compat
    from vaft.database import _summary as summary_module
    from vaft.omas.process_wrapper import compute_magnetic_energy

    ods = _sample()
    before = set(ods["equilibrium.time_slice"][0]["global_quantities"].keys())
    rows = summary_module.extract_equilibrium_global(ods, 39915)
    ts = ods["equilibrium.time_slice"][0]
    # the new columns read no path that is not there: no empty nodes left behind
    assert not ({"b0", "major_radius"} - before) & set(ts["global_quantities"].keys())
    row = rows[0]
    for name in ("beta_volume_B2", "beta_normal_B2", "normalized_plasma_current"):
        assert name in summary_module.EQUILIBRIUM_GLOBAL_COLUMNS and np.isfinite(row[name])
    # beta_B is the ratio of the integrals: int p dV over the F(psi) field energy
    pressure_integral = trapz_compat(np.asarray(ts["profiles_1d.pressure"], float),
                                     x=np.asarray(ts["profiles_1d.volume"], float))
    energy = compute_magnetic_energy(ods, time_slice=0, toroidal_field="equilibrium")
    assert row["beta_volume_B2"] == pytest.approx(pressure_integral / energy, rel=1e-9)
    # one normalization for both, and I_N the abscissa of the Troyon line: exact identities
    assert row["beta_normal"] / row["beta_normal_B2"] == pytest.approx(
        row["beta_tor"] / row["beta_volume_B2"], rel=1e-9)
    assert row["beta_normal"] == pytest.approx(100 * row["beta_tor"] / row["normalized_plasma_current"], rel=1e-9)
