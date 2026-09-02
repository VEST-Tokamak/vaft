"""``profiles_2d.j_tor`` is the local current density, not a splatted average (#316).

The updater used to map ``profiles_1d.j_tor`` onto (R,Z) the way pressure is
mapped. ``profiles_1d.j_tor`` is not a flux function -- the DD defines it as the
flux-surface average ``<j_tor/R> / <1/R>`` -- so that produced a field constant on
each surface, where the real one varies across a surface as ``R`` and ``1/R``.

It was inert until #290 supplied the 1-D profile it needed, which is what armed
the defect.

A Solov'ev equilibrium is the anchor: ``p'`` and ``ff'`` are constants there, the
Grad-Shafranov source is exact, and ``evaluate_solovev`` returns the local
``j_phi`` in closed form -- an independent path to the same number.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("omas")

from matplotlib.path import Path as MplPath
from omas import ODS

import vaft.omas.update as update
from vaft.data.equilibrium import SolovevConstraint
from vaft.process.equilibrium import evaluate_solovev, solve_solovev_constraints

R0, MINOR, ELONGATION = 1.0, 0.30, 1.6


@pytest.fixture(scope="module")
def solovev():
    """A Solov'ev equilibrium as an ODS, plus its closed-form local current.

    The boundary is a prescribed ellipse on which psi is exactly zero, and psi is
    *positive* inside -- so the magnetic axis is the maximum, not the minimum.
    Getting that backwards leaves psi_norm meaningless while the pointwise test
    still passes, because constant ``p'``/``ff'`` interpolate to themselves
    whatever the coordinate says. The integral below is what catches it.
    """
    theta = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    model = solve_solovev_constraints(
        [
            SolovevConstraint(R0 + MINOR * np.cos(t), ELONGATION * MINOR * np.sin(t), "psi", 0.0)
            for t in theta
        ],
        pprime=-1.0e4, ffprime=0.05, rref=R0, psi_boundary=0.0,
    )
    assert model.rank == 5 and model.residual_norm < 1e-12

    r = np.linspace(R0 - 1.25 * MINOR, R0 + 1.25 * MINOR, 181)
    z = np.linspace(-1.25 * ELONGATION * MINOR, 1.25 * ELONGATION * MINOR, 181)
    grid_r, grid_z = np.meshgrid(r, z, indexing="ij")
    fields = evaluate_solovev(model, grid_r, grid_z)
    psi = np.asarray(fields["psi"], float)

    edge = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    boundary_r = R0 + MINOR * np.cos(edge)
    boundary_z = ELONGATION * MINOR * np.sin(edge)
    inside = MplPath(np.column_stack([boundary_r, boundary_z])).contains_points(
        np.column_stack([grid_r.ravel(), grid_z.ravel()])
    ).reshape(grid_r.shape)

    cell = (r[1] - r[0]) * (z[1] - z[0])
    plasma_current = float(np.sum(np.asarray(fields["j_phi"], float)[inside]) * cell)
    axis = np.unravel_index(np.nanargmax(np.where(inside, psi, -np.inf)), psi.shape)

    size = 65
    ods = ODS(consistency_check=False)
    slice_ = "equilibrium.time_slice.0."
    ods[slice_ + "profiles_2d.0.grid.dim1"] = r
    ods[slice_ + "profiles_2d.0.grid.dim2"] = z
    ods[slice_ + "profiles_2d.0.psi"] = psi
    ods[slice_ + "profiles_1d.psi"] = np.linspace(float(psi[axis]), 0.0, size)
    ods[slice_ + "profiles_1d.dpressure_dpsi"] = np.full(size, model.pprime)
    ods[slice_ + "profiles_1d.f_df_dpsi"] = np.full(size, model.ffprime)
    ods[slice_ + "global_quantities.psi_axis"] = float(psi[axis])
    ods[slice_ + "global_quantities.psi_boundary"] = 0.0
    ods[slice_ + "global_quantities.magnetic_axis.r"] = float(r[axis[0]])
    ods[slice_ + "global_quantities.magnetic_axis.z"] = float(z[axis[1]])
    ods[slice_ + "global_quantities.ip"] = plasma_current
    ods[slice_ + "boundary.outline.r"] = boundary_r
    ods[slice_ + "boundary.outline.z"] = boundary_z

    update.update_equilibrium_profiles_2d_j_tor(ods)
    return {
        "ods": ods,
        "derived": np.asarray(ods[slice_ + "profiles_2d.0.j_tor"], float),
        "analytic": np.asarray(fields["j_phi"], float),
        "inside": inside,
        "cell": cell,
        "ip": plasma_current,
        "psi": psi,
        "axis_rz": (float(r[axis[0]]), float(r[axis[0]])),
    }


def test_the_field_equals_the_closed_form_pointwise(solovev):
    """The whole content of #316: the local value, not the surface average."""
    covered = solovev["inside"] & np.isfinite(solovev["derived"])
    assert covered.sum() > 0.99 * solovev["inside"].sum()

    analytic = solovev["analytic"][covered]
    derived = solovev["derived"][covered]
    scale = np.max(np.abs(analytic))
    # VAFT carries the COCOS orientation (-sigma_Bp) where the Solov'ev closed
    # form is the textbook j = -Delta*psi/(mu0 R); the two differ by that sign,
    # which #290's flipped-orientation test is what pins.
    assert np.max(np.abs(np.abs(derived) - np.abs(analytic))) / scale < 1e-8
    assert np.all(np.sign(derived) == np.sign(derived[0]))


def test_the_field_is_not_constant_on_a_flux_surface(solovev):
    """What the old mapping produced. `j_phi` goes as `R p' + ff'/(mu0 R)`, so on
    one surface it varies by the aspect ratio across it -- a flux function would
    be flat, and that is exactly how the bug would look if it came back."""
    psi = solovev["psi"]
    axis_value = float(psi[solovev["inside"]].max())
    psi_norm = (psi - axis_value) / (0.0 - axis_value)
    band = solovev["inside"] & (psi_norm > 0.45) & (psi_norm < 0.55)
    assert band.sum() > 50

    values = np.abs(solovev["derived"][band])
    spread = values.max() / values.min() - 1.0
    assert spread > 0.2, f"only {spread:.1%} variation across a surface"


def test_the_integral_recovers_the_plasma_current(solovev):
    """Catches a broken psi_norm, which the pointwise test cannot: with constant
    `p'` and `ff'` the interpolation returns them whatever the coordinate says,
    so only the boundary mask and the integral exercise the normalization."""
    covered = solovev["inside"] & np.isfinite(solovev["derived"])
    current = float(np.nansum(solovev["derived"][covered]) * solovev["cell"])
    assert abs(abs(current) - abs(solovev["ip"])) / abs(solovev["ip"]) < 1e-3


def test_no_current_is_drawn_outside_the_boundary(solovev):
    """`p'` and `ff'` are only defined out to the LCFS; clipping them into the
    scrape-off layer would put current where there is none."""
    derived = solovev["derived"]
    assert np.isnan(derived).any(), "nothing was masked at all"
    outside = ~solovev["inside"]
    assert np.isnan(derived[outside]).mean() > 0.95


def test_the_packaged_sample_integrates_to_its_stored_plasma_current():
    """The same check against a real equilibrium rather than a synthetic one."""
    from vaft.omas.sample import sample_ods

    try:
        ods = sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")
    update.update_equilibrium_derived_profiles(ods)
    update.update_equilibrium_profiles_2d_j_tor(ods)

    ts = ods["equilibrium.time_slice.0"]
    derived = np.asarray(ts["profiles_2d.0.j_tor"], float)
    r = np.asarray(ts["profiles_2d.0.grid.dim1"], float)
    z = np.asarray(ts["profiles_2d.0.grid.dim2"], float)
    grid_r, grid_z = np.meshgrid(r, z, indexing="ij")
    boundary = MplPath(
        np.column_stack([
            np.asarray(ts["boundary.outline.r"], float),
            np.asarray(ts["boundary.outline.z"], float),
        ])
    )
    inside = boundary.contains_points(
        np.column_stack([grid_r.ravel(), grid_z.ravel()])
    ).reshape(grid_r.shape)

    current = float(np.nansum(np.where(inside, derived, 0.0)) * (r[1] - r[0]) * (z[1] - z[0]))
    assert current == pytest.approx(float(ts["global_quantities.ip"]), rel=0.01)
