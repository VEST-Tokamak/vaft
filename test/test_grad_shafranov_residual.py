"""Whether an equilibrium satisfies its own Grad-Shafranov equation.

This is a different question from the one EFIT's a-file answers. What VAFT
elsewhere calls the "Grad-Shafranov error" is EFIT's iteration-to-iteration
delta -- a statement about convergence. This asks whether the flux map that
came out is consistent with the p' and FF' that went in, which a comfortably
converged reconstruction can still fail.
"""

import numpy as np
import pytest

import vaft.data
import vaft.omas
from vaft.data.eqdsk import read_geqdsk
from vaft.formula.constants import MU0
from vaft.process._equilibrium_parametric import SolovevEquilibrium
from vaft.process.equilibrium import (
    evaluate_solovev,
    grad_shafranov_operator,
    grad_shafranov_residual,
)

KINETIC = "kineticEfit/g048224.00300"


@pytest.fixture(scope="module")
def solovev():
    """An analytic equilibrium, where the residual is known to be zero."""
    model = SolovevEquilibrium(np.array([0.03, -0.02, 0.015, 0.004, -0.001]), -1200.0, 0.08, 1.0)
    r = np.linspace(0.7, 1.3, 101)
    z = np.linspace(-0.4, 0.4, 91)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    return model, r, z, evaluate_solovev(model, rm, zm)


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------

def test_the_operator_reproduces_the_analytic_source(solovev):
    """Delta* psi must equal -mu0 R^2 p' - FF' for a Solov'ev field, exactly."""
    _, r, z, values = solovev
    computed = grad_shafranov_operator(values["psi"], r, z)
    core = (slice(3, -3), slice(3, -3))
    expected = values["grad_shafranov_source"]
    error = np.abs(computed - expected)[core].max() / np.abs(expected[core]).max()
    assert error < 1e-3, error


def test_the_operator_refuses_a_grid_it_cannot_orient():
    psi = np.zeros((7, 5))
    with pytest.raises(ValueError, match=r"indexed \(R, Z\)"):
        grad_shafranov_operator(psi, np.linspace(0.5, 1.0, 5), np.linspace(-1, 1, 7))
    with pytest.raises(ValueError, match="2-D"):
        grad_shafranov_operator(np.zeros(5), np.linspace(0.5, 1.0, 5), np.linspace(-1, 1, 5))


# ---------------------------------------------------------------------------
# The residual
# ---------------------------------------------------------------------------

def test_an_exact_equilibrium_has_no_residual(solovev):
    model, r, z, values = solovev
    psi_1d = np.linspace(values["psi"].min(), values["psi"].max(), 65)
    result = grad_shafranov_residual(
        values["psi"], r, z,
        psi_1d=psi_1d,
        pprime=np.full_like(psi_1d, model.pprime),
        ffprime=np.full_like(psi_1d, model.ffprime),
        psi_axis=float(values["psi"].min()),
        psi_boundary=float(values["psi"].max()),
    )
    assert np.nanmedian(result.relative) < 1e-3
    assert result.scale > 0.0
    assert np.isfinite(result.residual).sum() >= result.residual.size - 1


def test_a_wrong_source_shows_up_as_a_residual(solovev):
    """Doubling p' breaks force balance, and the residual has to say so."""
    model, r, z, values = solovev
    psi_1d = np.linspace(values["psi"].min(), values["psi"].max(), 65)
    kwargs = dict(
        psi_1d=psi_1d,
        ffprime=np.full_like(psi_1d, model.ffprime),
        psi_axis=float(values["psi"].min()),
        psi_boundary=float(values["psi"].max()),
    )
    honest = grad_shafranov_residual(
        values["psi"], r, z, pprime=np.full_like(psi_1d, model.pprime), **kwargs
    )
    wrong = grad_shafranov_residual(
        values["psi"], r, z, pprime=np.full_like(psi_1d, 2 * model.pprime), **kwargs
    )
    assert np.nanmedian(wrong.relative) > 100 * np.nanmedian(honest.relative)


def test_mismatched_profile_lengths_are_refused():
    r = np.linspace(0.5, 1.0, 9)
    z = np.linspace(-0.3, 0.3, 9)
    psi = np.outer(r, np.ones_like(z))
    with pytest.raises(ValueError, match="same length"):
        grad_shafranov_residual(
            psi, r, z, psi_1d=np.linspace(0, 1, 5),
            pprime=np.zeros(4), ffprime=np.zeros(5),
        )


def test_a_degenerate_flux_range_is_refused():
    r = np.linspace(0.5, 1.0, 9)
    z = np.linspace(-0.3, 0.3, 9)
    psi = np.zeros((9, 9))
    with pytest.raises(ValueError, match="finite and distinct"):
        grad_shafranov_residual(
            psi, r, z, psi_1d=np.zeros(5), pprime=np.zeros(5), ffprime=np.zeros(5),
        )


# ---------------------------------------------------------------------------
# Against real reconstructions
# ---------------------------------------------------------------------------

def test_chease_refinement_closes_the_equation_the_efit_left_open():
    """The quantitative statement of what refinement bought.

    Both are the same discharge at the same time; the CHEASE output is a
    converged fixed-boundary solve and satisfies its own equation to a few
    percent, where the EFIT reconstruction it started from does not.
    """
    efit = vaft.omas.compute_grad_shafranov_residual(
        read_geqdsk(str(vaft.data.data_path(KINETIC))).to_omas(), time_slice=0
    )
    chease = vaft.omas.compute_grad_shafranov_residual(
        read_geqdsk(str(vaft.data.data_path(KINETIC + ".chease"))).to_omas(), time_slice=0
    )
    assert np.nanmedian(chease.relative) < 0.1
    assert np.nanmedian(chease.relative) < np.nanmedian(efit.relative) / 10


def test_the_flux_convention_is_resolved_rather_than_assumed():
    """Reading a weber file as per-radian moves the two sides by (2*pi)**2.

    The wrapper converts, so it must land far closer than the raw reading.
    """
    ods = read_geqdsk(str(vaft.data.data_path(KINETIC + ".chease"))).to_omas()
    resolved = vaft.omas.compute_grad_shafranov_residual(ods, time_slice=0)

    slice_node = ods["equilibrium.time_slice"][0]
    grid = slice_node["profiles_2d.0.grid"]
    profiles = slice_node["profiles_1d"]
    raw = grad_shafranov_residual(
        np.asarray(slice_node["profiles_2d.0.psi"], float),
        np.asarray(grid["dim1"], float),
        np.asarray(grid["dim2"], float),
        psi_1d=np.asarray(profiles["psi"], float),
        pprime=np.asarray(profiles["dpressure_dpsi"], float),
        ffprime=np.asarray(profiles["f_df_dpsi"], float),
    )
    assert np.nanmedian(resolved.relative) < np.nanmedian(raw.relative) / 100


def test_a_slice_without_the_profiles_says_which_are_missing():
    ods = vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))
    del ods["equilibrium.time_slice.0.profiles_1d.f_df_dpsi"]
    with pytest.raises(ValueError, match="f_df_dpsi"):
        vaft.omas.compute_grad_shafranov_residual(ods, time_slice=0)


def test_the_packaged_reconstruction_is_consistent_while_the_current_is_flat():
    """39915 closes to sub-percent early and degrades as the plasma decays.

    Slices 0-3 sit on the current plateau; by 0.325 s Ip has fallen by a third
    and the magnetics-only reconstruction no longer satisfies the equation.
    """
    ods = vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))
    early = vaft.omas.compute_grad_shafranov_residual(ods, time_slice=0)
    late = vaft.omas.compute_grad_shafranov_residual(ods, time_slice=6)
    assert np.nanmedian(early.relative) < 0.05
    assert np.nanmedian(late.relative) > 10 * np.nanmedian(early.relative)
