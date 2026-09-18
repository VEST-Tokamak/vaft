"""The plasma is the inside of the LCFS outline, not every cell with 0 <= psiN <= 1.

Outside the plasma psi turns over near the coils, so a flux threshold admits
exterior cells.  On the packaged VEST samples it made the plasma volume 1.7 to
18 times too large and the volume-averaged pressure 30-50 % too small, and it
inflated the reconstructed diamagnetic flux three- to eight-fold.  Each check
here is against a quantity the grid mask cannot influence: the contour-traced
profiles_1d volume and pressure integral, and closed-form polygon integrals of
the outline itself (Green's theorem).
"""

import copy
import logging
import warnings

import numpy as np
import pytest

pytest.importorskip("omas")
pytest.importorskip("skimage")

from vaft.omas.process_wrapper import compute_volume_averaged_pressure
from vaft.omas.sample import sample_ods
from vaft.omas.update import update_equilibrium_profiles_1d_geometry
from vaft.process.equilibrium import (
    calculate_reconstructed_diamagnetic_flux,
    plasma_cell_weights,
    volume_average,
)

# (shot, slice): early and late closed-flux slices on both storage conventions.
SLICES = [(39915, 0), (39915, 7), (41524, 0), (41524, 4), (41672, 1), (41672, 14)]


@pytest.fixture(scope="module")
def samples():
    logging.disable(logging.WARNING)
    try:
        traced = {}
        for shot in {s for s, _ in SLICES}:
            ods = sample_ods(shot)
            reference = copy.deepcopy(ods)
            update_equilibrium_profiles_1d_geometry(reference)
            traced[shot] = (ods, reference)
        yield traced
    finally:
        logging.disable(logging.NOTSET)


def _grid(ts):
    r = np.asarray(ts["profiles_2d.0.grid.dim1"], float)
    z = np.asarray(ts["profiles_2d.0.grid.dim2"], float)
    psi = np.asarray(ts["profiles_2d.0.psi"], float)
    if psi.shape != (r.size, z.size):
        psi = psi.T
    axis, boundary = float(ts["global_quantities.psi_axis"]), float(ts["global_quantities.psi_boundary"])
    psi_n = (psi - axis) / (boundary - axis)
    outline = (np.asarray(ts["boundary.outline.r"], float), np.asarray(ts["boundary.outline.z"], float))
    return r, z, psi, axis, boundary, psi_n, outline


@pytest.mark.parametrize("shot, idx", SLICES)
def test_the_weighted_volume_is_the_traced_plasma_volume(samples, shot, idx):
    ods, reference = samples[shot]
    r, z, _, _, _, psi_n, outline = _grid(ods["equilibrium.time_slice"][idx])
    weights = plasma_cell_weights(r, z, psi_n, *outline)
    _, volume = volume_average(np.ones_like(psi_n), psi_n, r, z, weights=weights)
    traced = float(np.asarray(reference[f"equilibrium.time_slice.{idx}.profiles_1d.volume"])[-1])
    assert volume == pytest.approx(traced, rel=5e-3)
    # The flux threshold alone is what this replaces; it is not close.
    _, threshold_volume = volume_average(np.ones_like(psi_n), psi_n, r, z)
    assert threshold_volume > 1.5 * traced


@pytest.mark.parametrize("shot, idx", SLICES)
def test_the_volume_averaged_pressure_is_the_profile_integral(samples, shot, idx):
    ods, reference = samples[shot]
    ts = reference["equilibrium.time_slice"][idx]
    volume = np.asarray(ts["profiles_1d.volume"], float)
    pressure = np.asarray(ts["profiles_1d.pressure"], float)
    expected = np.trapezoid(pressure, volume) / volume[-1]
    averaged = compute_volume_averaged_pressure(copy.deepcopy(ods), option="equilibrium")[idx]
    assert averaged == pytest.approx(expected, rel=0.02)


def _polygon_integral_of_inverse_r(outline_r, outline_z):
    """Area integral of 1/R over the polygon, as the contour integral of ln R dZ."""
    r = np.r_[outline_r, outline_r[:1]]
    z = np.r_[outline_z, outline_z[:1]]
    # Exact for straight edges: int ln R dZ along R linear in Z.
    r0, r1, dz = r[:-1], r[1:], np.diff(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_log = np.where(
            np.isclose(r0, r1), np.log(r0), (r1 * np.log(r1) - r0 * np.log(r0)) / (r1 - r0) - 1.0
        )
    return abs(float(np.sum(mean_log * dz)))


@pytest.mark.parametrize("shot, idx", SLICES)
def test_a_uniform_f_excess_integrates_over_the_outline_alone(samples, shot, idx):
    # With F = F_vac + c everywhere inside, the reconstructed diamagnetic flux
    # is c * (area integral of 1/R), which the outline fixes in closed form.
    ods, _ = samples[shot]
    r, z, psi, axis, boundary, psi_n, outline = _grid(ods["equilibrium.time_slice"][idx])
    f_vac, excess = 0.08, 1.0e-3
    psi_n_1d = np.linspace(0.0, 1.0, 65)
    f_1d = np.full_like(psi_n_1d, f_vac + excess)
    weights = plasma_cell_weights(r, z, psi_n, *outline)
    flux = calculate_reconstructed_diamagnetic_flux(
        r, z, psi, axis, boundary, psi_n_1d, f_1d, f_vac, weights=weights
    )
    assert flux == pytest.approx(excess * _polygon_integral_of_inverse_r(*outline), rel=0.01)


def test_without_an_outline_the_threshold_is_the_fallback_and_says_so():
    r = np.linspace(0.1, 1.0, 20)
    z = np.linspace(-0.5, 0.5, 21)
    psi_n = np.random.default_rng(0).uniform(-0.5, 1.5, (r.size, z.size))
    with pytest.warns(RuntimeWarning, match="outline"):
        weights = plasma_cell_weights(r, z, psi_n)
    np.testing.assert_array_equal(weights, ((psi_n >= 0) & (psi_n <= 1)).astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, with_weights = volume_average(np.ones_like(psi_n), psi_n, r, z, weights=weights)
        _, without = volume_average(np.ones_like(psi_n), psi_n, r, z)
    assert with_weights == pytest.approx(without, rel=1e-12)


def test_weights_of_the_wrong_shape_are_refused():
    r = np.linspace(0.1, 1.0, 20)
    z = np.linspace(-0.5, 0.5, 21)
    with pytest.raises(ValueError, match="shape"):
        volume_average(np.ones((20, 21)), np.zeros((20, 21)), r, z, weights=np.ones((21, 20)))


@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_the_reconstructed_diamagnetic_flux_has_the_measured_size(samples, shot):
    # Against the diamagnetic loop -- a measurement, not a grid quantity.  With
    # the flux threshold the reconstruction grew to 3.8 times the measurement
    # on 41524 as the plasma shrank; inside the outline it stays within
    # 0.66-1.19 on the first five slices of all three shots.  Magnitude only:
    # the two carry opposite signs on every slice because the packaged
    # reconstructions are paramagnetic against a diamagnetic measurement --
    # a reconstruction disagreement (#385, #386), not a masking one.
    from vaft.omas.process_wrapper import compute_diamagnetic_flux_measured_vs_computed

    ods, _ = samples[shot]
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = compute_diamagnetic_flux_measured_vs_computed(copy.deepcopy(ods))
    finally:
        logging.disable(logging.NOTSET)
    for idx in sorted(rows)[:5]:
        ratio = abs(rows[idx]["computed"] / rows[idx]["measured"])
        assert 0.6 < ratio < 1.3, (idx, ratio)


def test_the_reconstructed_diamagnetic_flux_is_efits_own_cdflux(samples):
    # EFIT integrates the same quantity over its own plasma and writes it to
    # the a-file.  At 0.319 s (sample slice 3) that is 1.506e-3 Wb; the
    # threshold mask gave 2.91e-3, inside the outline this gives 1.54e-3.
    from pathlib import Path

    import vaft
    from vaft.data.aeqdsk import read_aeqdsk
    from vaft.omas.process_wrapper import compute_reconstructed_diamagnetic_flux

    ods, _ = samples[39915]
    assert float(ods["equilibrium.time"][3]) == pytest.approx(0.319)
    afile = read_aeqdsk(Path(vaft.__file__).parent / "data" / "efit" / "a039915.00319").scalars
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        computed = compute_reconstructed_diamagnetic_flux(copy.deepcopy(ods), 3)
    assert computed == pytest.approx(afile["cdflux"], rel=0.05)


@pytest.mark.parametrize("shot, idx", SLICES)
def test_a_flat_profile_averages_to_itself_inside_the_outline(samples, shot, idx):
    # An outline-weighted edge cell can sit just past psiN = 1.  Mapped with
    # psi_to_rz's default zero there, a flat profile averages below itself;
    # continued at its edge value it averages to exactly itself.
    from vaft.process.equilibrium import psi_to_rz

    ods, _ = samples[shot]
    r, z, psi, axis, boundary, psi_n, outline = _grid(ods["equilibrium.time_slice"][idx])
    weights = plasma_cell_weights(r, z, psi_n, *outline)
    grid_psi_n = np.linspace(0.0, 1.0, 33)
    flat, _ = psi_to_rz(grid_psi_n, np.ones_like(grid_psi_n), psi, axis, boundary, fill_outside="edge")
    assert volume_average(flat, psi_n, r, z, weights=weights)[0] == pytest.approx(1.0, rel=1e-12)
    zeroed, _ = psi_to_rz(grid_psi_n, np.ones_like(grid_psi_n), psi, axis, boundary)
    assert volume_average(zeroed, psi_n, r, z, weights=weights)[0] < 1.0


def test_psi_to_rz_refuses_an_unknown_fill():
    from vaft.process.equilibrium import psi_to_rz

    with pytest.raises(ValueError, match="fill_outside"):
        psi_to_rz(np.linspace(0, 1, 5), np.ones(5), np.zeros((3, 3)), 0.0, 1.0, fill_outside="nan")
