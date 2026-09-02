"""How well the flux-surface trace converges, and what that costs beta_pol (#317, #318).

#318 asks whether `int p dV`, `V` and `L_pol` converge with flux-surface
resolution, because the two `beta_pol` conventions differ by
`R0 L_pol^2 / (2V)` and a definition is only as good as the geometry under it.
#317 asks how far the packaged reference can be trusted near the axis.

Both are answered against a Solov'ev equilibrium with a *prescribed elliptical
boundary*, so `V`, the cross-section area and the perimeter are known in closed
form -- no reference file, no solver of ours in the loop. Resolution is the only
variable.

The headline: `V` and `area` converge at second order, and `L_pol` does not. It
reaches ~2e-5 and stops, because a traced polygon's perimeter is limited by where
marching squares places its vertices rather than by the cell size. The
circumference convention divides by `L_pol` squared, so that floor is the one
that matters for it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("omas")
pytest.importorskip("skimage")

from matplotlib.path import Path as MplPath
from scipy.integrate import quad

from vaft.data.equilibrium import SolovevConstraint
from vaft.process.equilibrium import (
    evaluate_solovev,
    flux_surface_quantities,
    solve_solovev_constraints,
)

R0, MINOR, ELONGATION = 1.0, 0.30, 1.6

#: Closed-form geometry of the prescribed boundary
#: ``R = R0 + a cos t``, ``Z = kappa a sin t``. Volume is Pappus' theorem with the
#: ellipse's centroid at ``R0``; the perimeter has no elementary form and is
#: quadratured to far below the tolerances here.
EXACT_AREA = np.pi * MINOR * (ELONGATION * MINOR)
EXACT_VOLUME = 2.0 * np.pi * R0 * EXACT_AREA
EXACT_LENGTH_POL = quad(
    lambda t: np.hypot(MINOR * np.sin(t), ELONGATION * MINOR * np.cos(t)),
    0.0, 2.0 * np.pi, limit=400,
)[0]

#: ``grid points -> (volume, area, length_pol)`` relative error, measured.
#: Volume and area fall by ~4x per doubling, which is second order in the cell
#: size. ``length_pol`` does not: it reaches ~2e-5 and stays there.
CONVERGENCE = {
    65: (5.19e-04, 5.18e-04, 1.68e-04),
    129: (1.23e-04, 1.22e-04, 1.88e-05),
    257: (3.10e-05, 3.08e-05, 1.54e-05),
    513: (7.27e-06, 7.14e-06, 2.44e-05),
}


@pytest.fixture(scope="module")
def model():
    theta = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    solved = solve_solovev_constraints(
        [
            SolovevConstraint(
                R0 + MINOR * np.cos(t), ELONGATION * MINOR * np.sin(t), "psi", 0.0
            )
            for t in theta
        ],
        pprime=-1.0e4, ffprime=0.05, rref=R0, psi_boundary=0.0,
    )
    assert solved.rank == 5
    return solved


def _traced(model, size, levels=65):
    r = np.linspace(R0 - 1.25 * MINOR, R0 + 1.25 * MINOR, size)
    z = np.linspace(-1.25 * ELONGATION * MINOR, 1.25 * ELONGATION * MINOR, size)
    grid_r, grid_z = np.meshgrid(r, z, indexing="ij")
    psi = np.asarray(evaluate_solovev(model, grid_r, grid_z)["psi"], float)
    edge = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    inside = MplPath(
        np.column_stack(
            [R0 + MINOR * np.cos(edge), ELONGATION * MINOR * np.sin(edge)]
        )
    ).contains_points(np.column_stack([grid_r.ravel(), grid_z.ravel()])).reshape(psi.shape)
    # psi is positive inside this equilibrium, so the axis is its maximum.
    return flux_surface_quantities(
        psi, r, z, float(psi[inside].max()), 0.0,
        np.linspace(0.0, 1.0, levels), axis_rz=(R0, 0.0), boundary=None,
    )


@pytest.mark.parametrize("size", sorted(CONVERGENCE))
def test_the_traced_geometry_matches_the_closed_form(model, size):
    surfaces = _traced(model, size)
    volume_error, area_error, length_error = CONVERGENCE[size]
    for name, derived, exact, expected in (
        ("volume", surfaces["volume"][-1], EXACT_VOLUME, volume_error),
        ("area", surfaces["area"][-1], EXACT_AREA, area_error),
        ("length_pol", surfaces["length_pol"][-1], EXACT_LENGTH_POL, length_error),
    ):
        measured = abs(float(derived) - exact) / exact
        # Twice the measured value: tight enough to catch a regression, loose
        # enough not to fail on a different BLAS or scikit-image release.
        assert measured < 2 * expected, f"{name} at N={size}: {measured:.2e}"


def test_volume_and_area_converge_at_second_order(model):
    """Four-fold error reduction per doubling. This is what says the trace is
    limited by the cell size and nothing else."""
    sizes = sorted(CONVERGENCE)
    errors = {
        size: abs(float(_traced(model, size)["volume"][-1]) - EXACT_VOLUME) / EXACT_VOLUME
        for size in sizes
    }
    for coarse, fine in zip(sizes[:-1], sizes[1:]):
        ratio = errors[coarse] / errors[fine]
        assert 3.0 < ratio < 5.5, f"N={coarse}->{fine} improved {ratio:.1f}x, expected ~4"


def test_the_perimeter_does_not_converge_with_the_grid(model):
    """The finding #318 wanted: `L_pol` is not limited by the cell size.

    It reaches ~2e-5 by N=129 and then stops improving -- a traced polygon's
    perimeter is set by where marching squares puts its vertices. Anything
    dividing by `L_pol` inherits that floor, and the circumference `beta_pol`
    divides by its square, so it carries ~4e-5 of irreducible geometry error
    however fine the map is. That is small, but it is a floor and not a
    tolerance that refinement will lower.
    """
    coarse = abs(float(_traced(model, 129)["length_pol"][-1]) - EXACT_LENGTH_POL) / EXACT_LENGTH_POL
    fine = abs(float(_traced(model, 513)["length_pol"][-1]) - EXACT_LENGTH_POL) / EXACT_LENGTH_POL
    assert coarse < 1e-4 and fine < 1e-4, "both are already accurate"
    # Four times the resolution buys nothing; second order would be 16x.
    assert fine > coarse / 4.0, "length_pol converged after all -- update the finding"


def test_the_pressure_integral_inherits_the_volume_convergence(model):
    """`int p dV` is what both `beta_pol` definitions integrate, so it is the
    third quantity #318 asks about. It adds no error of its own: `p` is a flux
    function evaluated on the level grid, and the only geometry in the integral
    is `V(psi_n)`. Refining the map therefore moves it by what it moves the
    volume, and no more.
    """
    from vaft.compat import trapz_compat

    levels = np.linspace(0.0, 1.0, 129)
    # A constant p' makes the pressure linear in psi; the scale cancels in the
    # relative comparison, so unit slope is enough.
    pressure = 1.0 - levels

    integrals = [
        float(trapz_compat(pressure, x=np.asarray(_traced(model, size, levels=levels.size)["volume"], float)))
        for size in (129, 513)
    ]
    shift = abs(integrals[0] - integrals[1]) / abs(integrals[1])
    assert shift < 2 * CONVERGENCE[129][0], (
        f"int p dV moved {shift:.2e} between N=129 and N=513, more than the "
        f"volume's own {CONVERGENCE[129][0]:.2e} error at the coarse grid"
    )


# --- how far apart the two beta_pol conventions actually sit (#318) -----------

#: ``shot -> (V, L_pol, R0, 2V/(R0 L_pol^2))`` measured against the live HSDS
#: database with this branch's code, first equilibrium slice of each shot.
#: Hard-coded on purpose -- tests do not reach the database, and the point is the
#: *spread*, which no packaged sample can show. Captured 2026-09-02.
GEOMETRY_FACTOR_SURVEY = {
    39513: (0.570857, 1.918036, 0.349612, 0.887683),
    39915: (0.946903, 2.364861, 0.400000, 0.846572),
    39917: (0.846029, 2.259428, 0.400000, 0.828625),
    40325: (0.983508, 2.418847, 0.400000, 0.840487),
    40328: (0.043911, 0.673759, 0.199145, 0.971460),
    44403: (0.024602, 0.522520, 0.400000, 0.450532),
    45531: (0.921311, 2.334620, 0.397342, 0.850823),
    45538: (0.027215, 0.551251, 0.183137, 0.978043),
}

#: Three of the eight resolve to an R_0 that is not 0.4 and is *kept*, because
#: their stored ``b0*r0`` agrees with ``tf``'s ``B*R``: the pair is internally
#: consistent, just quoted at a different radius. The resolver added in #238
#: compares the product rather than ``r0`` itself precisely so it can tell those
#: apart from the corruption in #325 -- but it means ``beta_pol`` is normalized
#: at a shot-dependent radius even when nothing is wrong, which is its own
#: comparability problem.
SHOTS_WITH_A_CONSISTENT_NON_STANDARD_R0 = (39513, 40328, 45531, 45538)


def test_the_survey_is_self_consistent():
    """The recorded numbers satisfy the identity they were gathered to measure, so
    a transcription slip cannot pass unnoticed."""
    for shot, (volume, length_pol, r0, factor) in GEOMETRY_FACTOR_SURVEY.items():
        assert 2.0 * volume / (r0 * length_pol**2) == pytest.approx(factor, rel=2e-4), shot


def test_the_two_beta_pol_conventions_are_far_apart_and_shot_dependent():
    """#318's cross-shot question. The DD and EFIT `beta_pol` differ by exactly
    `2V/(R0 L_pol^2)`, and that factor is **not** a constant to correct for:
    across eight VEST shots it runs 0.45 to 0.98, so the two definitions disagree
    by anywhere from 2% to 122% depending on the discharge.

    That is the argument against ever treating one as an estimate of the other,
    and against a single conversion factor between the columns.
    """
    factors = [entry[3] for entry in GEOMETRY_FACTOR_SURVEY.values()]
    assert min(factors) < 0.5 and max(factors) > 0.95
    assert max(factors) / min(factors) > 2.0
    # Never above 1: the DD form is the smaller of the two on every shot seen.
    assert all(factor < 1.0 for factor in factors)


def test_small_plasmas_are_where_the_conventions_diverge_most():
    """The spread is not noise, it is physics: the factor is a shape ratio, and
    the smallest plasmas sit at both ends of it. Anyone comparing `beta_pol`
    across a shot range needs the geometry, not a scale factor.
    """
    by_volume = sorted(GEOMETRY_FACTOR_SURVEY.values())
    large = [entry[3] for entry in by_volume if entry[0] > 0.5]
    assert len(large) >= 4
    # The well-formed discharges cluster; it is the marginal ones that scatter.
    assert max(large) - min(large) < 0.07


def test_a_consistent_r0_is_kept_even_when_it_is_not_the_machine_radius():
    """Half the survey stores an `r0` other than 0.4 that the resolver keeps,
    because `b0*r0` agrees with `tf` -- the pair is right, just quoted at another
    radius. Distinguishing that from #325's corruption is the whole reason the
    resolver compares the product and not `r0`.

    The consequence is worth stating: `beta_pol` is then normalized at a
    shot-dependent radius even when no data is wrong, so the column is not
    directly comparable across a shot range.
    """
    kept = {
        shot
        for shot, (_v, _l, r0, _f) in GEOMETRY_FACTOR_SURVEY.items()
        if abs(r0 - 0.4) > 1e-6
    }
    assert kept == set(SHOTS_WITH_A_CONSISTENT_NON_STANDARD_R0)
    assert all(0.15 < GEOMETRY_FACTOR_SURVEY[shot][2] < 0.45 for shot in kept)


# --- what the packaged reference is worth, and where it is not (#317) ---------

#: The engine's agreement with a **second, independent** OMFIT dataset: shot
#: 39915 as the HSDS database holds it, a 37-leaf `profiles_1d` on a 513x513 map.
#: Captured 2026-09-02; tests do not reach the database.
#:
#: These sit 10-40x below the same comparison against the packaged 129x129
#: kineticEfit reference (`volume`/`area` 4.7e-3 there), which is what says the
#: discrepancy documented in `test_equilibrium_derived_profiles` is grid
#: resolution rather than method -- the question #317 was opened to settle.
HSDS_513_AGREEMENT = {
    "volume": 1.0e-4,
    "area": 1.3e-4,
    "surface": 3.3e-5,
    "gm1": 2.3e-4,
    "gm9": 1.2e-4,
    "gm8": 5.6e-5,
    "elongation": 8.4e-5,
}


def test_the_two_references_place_the_resolution_story_consistently():
    """A 513-point map agrees 10-40x better than a 129-point one, and the
    Solov'ev convergence above says the trace is second order in the cell size.
    Those two have to be telling the same story, or one of them is wrong."""
    from test_equilibrium_derived_profiles import INTERIOR_TOLERANCE

    for leaf in ("volume", "area"):
        coarse = INTERIOR_TOLERANCE[leaf]      # tolerance against the 129-pt reference
        fine = HSDS_513_AGREEMENT[leaf]        # measured against the 513-pt one
        assert fine < coarse / 10.0, leaf

    # Second order over a 4x refinement predicts ~16x; the references differ in
    # more than resolution, so an order of magnitude is the honest claim.
    predicted = CONVERGENCE[129][0] / CONVERGENCE[513][0]
    assert predicted > 10.0


def test_an_outlier_q_on_axis_is_reported_and_not_repaired():
    """#317's open question. `Phi = integral(q dpsi)` carries `q[0]` into
    `rho_tor_norm` and every kinetic profile mapped onto it, and EFIT's axis value
    is often unreliable -- 8.07 against a neighbourhood of 1.89 on the packaged
    kineticEfit reference.

    The decision recorded here is to warn, not to fix. Extrapolating `q[0]` from
    its neighbours brings the coordinate 5.8x closer to what that reference
    stores, which is good evidence the outlier is what they disagree about, and
    exactly why repairing it silently would be wrong: it swaps a measurement for
    a guess and hides a solver problem behind a plausible-looking coordinate.
    """
    from omas import load_omas_json

    from vaft.data._derived import rho_tor_profile
    from vaft.data.resources import data_path

    path = data_path("kineticEfit/ods_48224_300ms.json")
    if not path.exists():  # pragma: no cover - repository-only sample
        pytest.skip("kineticEfit reference ODS is not packaged in this build")
    profiles = load_omas_json(str(path), consistency_check=False)[
        "equilibrium.time_slice.0.profiles_1d"
    ]
    q = np.asarray(profiles["q"], float)
    psi = np.asarray(profiles["psi"], float)
    assert abs(q[0]) / np.median(np.abs(q[1:5])) > 4.0, "fixture no longer has the outlier"

    with pytest.warns(RuntimeWarning, match="q on axis"):
        as_given = rho_tor_profile(q, psi)

    # Used as given: the coordinate still reflects the outlier.
    repaired_q = q.copy()
    repaired_q[0] = 2 * q[1] - q[2]
    repaired = rho_tor_profile(repaired_q, psi)
    assert np.max(np.abs(as_given.rho_tor_norm - repaired.rho_tor_norm)) > 0.04

    stored = np.asarray(profiles["rho_tor_norm"], float)
    assert np.max(np.abs(as_given.rho_tor_norm - stored)) > 0.03
    assert np.max(np.abs(repaired.rho_tor_norm - stored)) < 0.01


def test_a_smooth_q_profile_is_not_flagged():
    """The check has to stay quiet on ordinary data or it is worthless."""
    from vaft.data._derived import rho_tor_profile
    from vaft.omas.sample import sample_ods

    try:
        profiles = sample_ods()["equilibrium.time_slice.0.profiles_1d"]
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        rho_tor_profile(np.asarray(profiles["q"], float), np.asarray(profiles["psi"], float))
    assert not [w for w in caught if "q on axis" in str(w.message)]
