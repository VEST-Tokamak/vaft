"""Toroidal driven current -> IMAS parallel current density.

This conversion is easy to get wrong by a factor that still looks plausible on
a plot, so the tests pin it from several independent directions: an analytic
case, the large-aspect-ratio limit it must reduce to, the definitional
relationship between the two IMAS fields, and -- where a real run is available
-- a solver's own <J.B>.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.equilibrium import parallel_current_from_toroidal

N = 6


def _uniform(value, n=N):
    return np.full(n, float(value))


# --------------------------------------------------------------------------
# Analytic: a prescribed field-aligned current must come back exactly
# --------------------------------------------------------------------------


def test_a_prescribed_field_aligned_current_is_recovered():
    """Construct dI from a known lambda, and require lambda back.

    With J = lambda B the shell toroidal current is lambda * int B_phi dA, and
    int B_phi dA = F <R^-2> dV / 2pi. Feeding that dI in must return the
    lambda it was built from, and j_par = lambda <B^2> / B0.
    """
    f, gm1, gm5, dV, b0 = 1.7, 3.1, 0.45, 0.22, 0.6
    lam = np.array([2.0, -1.0, 0.5, 4.0, 0.0, -3.5])

    dI = lam * (f * gm1 * dV / (2.0 * np.pi))
    got = parallel_current_from_toroidal(
        dI, f=_uniform(f), gm1=_uniform(gm1), gm5=_uniform(gm5),
        shell_volume=_uniform(dV), b0=b0,
    )

    np.testing.assert_allclose(got.lambda_, lam, rtol=1e-12)
    np.testing.assert_allclose(got.j_parallel, lam * gm5 / b0, rtol=1e-12)


def test_the_sign_of_the_driven_current_survives():
    """A counter-driven shell must stay negative; the sign is not imposed."""
    dI = np.array([1.0, -1.0, 2.0, -2.0, 0.0, 3.0])
    got = parallel_current_from_toroidal(
        dI, f=_uniform(1.0), gm1=_uniform(1.0), gm5=_uniform(1.0),
        shell_volume=_uniform(1.0), b0=1.0,
    )
    np.testing.assert_array_equal(np.sign(got.j_parallel), np.sign(dI))


def test_a_negative_b0_flips_the_result_and_nothing_else():
    kwargs = dict(f=_uniform(1.0), gm1=_uniform(1.0), gm5=_uniform(1.0),
                  shell_volume=_uniform(1.0))
    dI = np.linspace(1.0, 6.0, N)
    positive = parallel_current_from_toroidal(dI, b0=0.8, **kwargs)
    negative = parallel_current_from_toroidal(dI, b0=-0.8, **kwargs)
    np.testing.assert_allclose(negative.j_parallel, -positive.j_parallel)
    np.testing.assert_allclose(negative.lambda_, positive.lambda_)


# --------------------------------------------------------------------------
# The large-aspect-ratio limit it must reduce to
# --------------------------------------------------------------------------


@pytest.mark.parametrize("aspect", [10.0, 100.0, 1000.0])
def test_large_aspect_ratio_reduces_to_the_toroidal_current_density(aspect):
    """As R0/a grows, j_parallel -> dI/dA -- the intuition the old code used.

    A limiting check, not the implementation: at VEST's aspect ratio the two
    differ by about half.
    """
    minor, b0 = 1.0, 2.0
    r0 = aspect * minor
    dA = _uniform(0.05)
    dI = np.linspace(1.0, 6.0, N)

    got = parallel_current_from_toroidal(
        dI,
        f=_uniform(r0 * b0),        # F -> R0 B0
        gm1=_uniform(1.0 / r0**2),  # <R^-2> -> R0^-2
        gm5=_uniform(b0**2),        # <B^2> -> B0^2
        shell_volume=2.0 * np.pi * r0 * dA,
        b0=b0,
        shell_area=dA,
    )
    np.testing.assert_allclose(got.j_parallel, dI / dA, rtol=1e-12)


def test_a_spherical_tokamak_departs_from_that_limit():
    """The whole point: at low aspect ratio the proxy is materially wrong."""
    b0, r0 = 0.15, 0.4
    dA, dV = _uniform(0.004), _uniform(0.0095)
    dI = _uniform(2.0)
    got = parallel_current_from_toroidal(
        dI,
        f=_uniform(0.096),          # paramagnetic, well above R0*b0 = 0.06
        gm1=_uniform(5.4),          # <R^-2> at VEST scale
        gm5=_uniform(0.049),        # <B^2> >> b0^2 = 0.0225
        shell_volume=dV, b0=b0, shell_area=dA,
    )
    ratio = got.j_parallel / (dI / dA)
    assert np.all(ratio > 1.2), ratio


# --------------------------------------------------------------------------
# Consistency between the two IMAS fields
# --------------------------------------------------------------------------


def test_current_parallel_inside_is_the_surface_integral_of_j_parallel():
    dA = np.linspace(0.01, 0.06, N)
    got = parallel_current_from_toroidal(
        np.linspace(1.0, 6.0, N), f=_uniform(1.2), gm1=_uniform(2.5),
        gm5=_uniform(0.3), shell_volume=_uniform(0.2), b0=0.7, shell_area=dA,
    )
    np.testing.assert_allclose(
        got.current_parallel_inside, np.cumsum(got.j_parallel * dA)
    )


def test_without_an_area_the_cumulative_field_is_not_invented():
    got = parallel_current_from_toroidal(
        np.linspace(1.0, 6.0, N), f=_uniform(1.0), gm1=_uniform(1.0),
        gm5=_uniform(1.0), shell_volume=_uniform(1.0), b0=1.0,
    )
    assert got.current_parallel_inside is None


# --------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------


def test_a_zero_width_shell_yields_zero_rather_than_dividing_by_zero():
    dV = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2])
    got = parallel_current_from_toroidal(
        _uniform(1.0), f=_uniform(1.0), gm1=_uniform(1.0),
        shell_volume=dV, gm5=_uniform(1.0), b0=1.0,
    )
    assert np.isfinite(got.j_parallel).all()
    assert got.j_parallel[0] == 0.0


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="share a"):
        parallel_current_from_toroidal(
            np.ones(3), f=np.ones(4), gm1=np.ones(3), gm5=np.ones(3),
            shell_volume=np.ones(3), b0=1.0,
        )


@pytest.mark.parametrize("bad", [0.0, np.nan])
def test_an_unusable_b0_is_refused(bad):
    with pytest.raises(ValueError, match="b0"):
        parallel_current_from_toroidal(
            np.ones(N), f=_uniform(1.0), gm1=_uniform(1.0), gm5=_uniform(1.0),
            shell_volume=_uniform(1.0), b0=bad,
        )


def test_the_input_arrays_are_not_modified():
    dI = np.linspace(1.0, 6.0, N)
    original = dI.copy()
    parallel_current_from_toroidal(
        dI, f=_uniform(1.0), gm1=_uniform(1.0), gm5=_uniform(1.0),
        shell_volume=_uniform(1.0), b0=1.0,
    )
    np.testing.assert_array_equal(dI, original)


# --------------------------------------------------------------------------
# Against a solver's own <J.B>
# --------------------------------------------------------------------------

import os  # noqa: E402
from pathlib import Path  # noqa: E402

_RUN_DIR = os.environ.get("VAFT_NUBEAM_RUN_DIR")
_needs_run = pytest.mark.skipif(
    not _RUN_DIR or not Path(_RUN_DIR).is_dir(),
    reason="set VAFT_NUBEAM_RUN_DIR to a completed NUBEAM run",
)


def _plasma_state(run_dir):
    """The Plasma State's own equilibrium geometry, currents and <J.B>."""
    import warnings

    import xarray as xr

    directory = Path(run_dir)
    candidates = sorted(directory.glob("*.cdf"))
    for path in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with xr.open_dataset(path, decode_times=False) as ds:
                if {"jdotb", "curt", "g_eq", "gb2", "gr2i"} <= set(ds.variables):
                    return {k: np.asarray(ds[k].values) for k in
                            ("jdotb", "curt", "g_eq", "gb2", "gr2i", "vol", "area")}
    pytest.skip("no Plasma State carrying <J.B> in the run directory")


@_needs_run
def test_matches_the_solvers_own_jdotb_on_the_total_current():
    """The strongest available check on the machinery.

    Applied to the *total* enclosed toroidal current, the conversion must
    reproduce the equilibrium code's independently computed <J.B>/B0. The
    total current is not purely field-aligned -- it carries a diamagnetic
    part -- so this bounds the assumption rather than proving it, and the
    residual is expected to grow outward.
    """
    state = _plasma_state(_RUN_DIR)
    centre = lambda a: 0.5 * (a[:-1] + a[1:])
    b0 = 0.15  # VEST vacuum_toroidal_field.b0 for the validated case

    got = parallel_current_from_toroidal(
        np.diff(state["curt"]),
        f=centre(state["g_eq"]),
        gm1=centre(state["gr2i"]),
        gm5=centre(state["gb2"]),
        shell_volume=np.diff(state["vol"]),
        b0=b0,
        shell_area=np.diff(state["area"]),
    )
    reference = centre(state["jdotb"]) / b0
    ratio = got.j_parallel / reference

    # Core agreement is what matters; the separatrix shell is degenerate.
    assert ratio[0] == pytest.approx(1.0, abs=0.01)
    assert np.median(ratio[:-1]) == pytest.approx(1.0, abs=0.05)


@_needs_run
def test_the_spherical_tokamak_departure_is_recorded():
    """Pin the size of the error the previous mapping made, so it cannot
    silently return."""
    state = _plasma_state(_RUN_DIR)
    import warnings

    import xarray as xr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with xr.open_dataset(Path(_RUN_DIR) / "state_changes.cdf",
                             decode_times=False) as ds:
            driven = np.asarray(ds["curbeam"].values).ravel()

    centre = lambda a: 0.5 * (a[:-1] + a[1:])
    area = np.diff(state["area"])
    got = parallel_current_from_toroidal(
        driven, f=centre(state["g_eq"]), gm1=centre(state["gr2i"]),
        gm5=centre(state["gb2"]), shell_volume=np.diff(state["vol"]),
        b0=0.15, shell_area=area,
    )
    ratio = got.j_parallel / (driven / area)
    assert 1.4 < ratio.min() and ratio.max() < 1.7, (ratio.min(), ratio.max())

    # And the cumulative field is nowhere near the toroidal current.
    assert got.current_parallel_inside[-1] / driven.sum() == pytest.approx(
        1.535, abs=0.05
    )


# --------------------------------------------------------------------------
# The ODS adapter, the generic path for other codes
# --------------------------------------------------------------------------


def _equilibrium_ods(*, f=1.7, gm1=3.1, gm5=0.45, b0=0.6, shells=N):
    from omas import ODS

    ods = ODS()
    base = "equilibrium.time_slice.0.profiles_1d"
    edges = np.linspace(0.0, 1.0, shells + 1)
    ods[f"{base}.rho_tor_norm"] = edges
    ods[f"{base}.f"] = np.full(edges.size, f)
    ods[f"{base}.gm1"] = np.full(edges.size, gm1)
    ods[f"{base}.gm5"] = np.full(edges.size, gm5)
    # dV = 0.22 and dA = 0.05 per shell
    ods[f"{base}.volume"] = np.arange(edges.size) * 0.22
    ods[f"{base}.area"] = np.arange(edges.size) * 0.05
    ods["equilibrium.vacuum_toroidal_field.b0"] = [b0]
    return ods, edges


def test_the_ods_adapter_agrees_with_the_array_function():
    from vaft.omas.process_wrapper import compute_parallel_current_from_toroidal

    ods, edges = _equilibrium_ods()
    dI = np.linspace(1.0, 6.0, N)

    through_ods = compute_parallel_current_from_toroidal(ods, dI, edges)
    direct = parallel_current_from_toroidal(
        dI, f=_uniform(1.7), gm1=_uniform(3.1), gm5=_uniform(0.45),
        shell_volume=_uniform(0.22), b0=0.6, shell_area=_uniform(0.05),
    )
    np.testing.assert_allclose(through_ods.j_parallel, direct.j_parallel, rtol=1e-10)
    np.testing.assert_allclose(
        through_ods.current_parallel_inside, direct.current_parallel_inside, rtol=1e-10
    )


def test_an_equilibrium_without_flux_surface_averages_is_refused():
    """gm1 and gm5 are derived leaves; a raw g-file read lacks them. Refusing
    is the point -- the alternative is a geometry-free approximation."""
    from vaft.omas.process_wrapper import compute_parallel_current_from_toroidal

    ods, edges = _equilibrium_ods()
    del ods["equilibrium.time_slice.0.profiles_1d.gm5"]
    with pytest.raises(ValueError, match="gm5"):
        compute_parallel_current_from_toroidal(ods, np.ones(N), edges)


def test_a_mismatched_boundary_count_is_refused():
    from vaft.omas.process_wrapper import compute_parallel_current_from_toroidal

    ods, edges = _equilibrium_ods()
    with pytest.raises(ValueError, match="boundaries"):
        compute_parallel_current_from_toroidal(ods, np.ones(N + 2), edges)


def test_the_adapter_does_not_mutate_the_ods():
    from vaft.omas.process_wrapper import compute_parallel_current_from_toroidal

    ods, edges = _equilibrium_ods()
    before = set(ods.flat().keys())
    compute_parallel_current_from_toroidal(ods, np.ones(N), edges)
    assert set(ods.flat().keys()) == before
