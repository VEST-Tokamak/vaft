"""The ODS-level neoclassical provider and the Sauter/Redl-vs-NEO study (#550 phase 6).

Measured on the packaged VEST 48224 state, with the seven-radius NEO run committed by
phase 5. The numbers these tests pin, so they are readable without running anything:

    Sauter peak  11.61 kA/m^2 at rho 0.552,  I_bs 2.58 kA
    Redl   peak  12.16 kA/m^2 at rho 0.557,  I_bs 2.71 kA
    NEO    peak                 at rho 0.52,  I_bs 2.19 kA  (1.5 % of Ip)

    Redl vs Sauter : integrated 4.91 %, RMS/peak 4.11 %
    vs NEO         : Sauter integrated -8.97 %, RMS/peak 7.56 %
                     Redl   integrated -5.42 %, RMS/peak 4.47 %

The headline is the last pair: Redl sits closer to the drift-kinetic answer than Sauter
on a spherical tokamak, by both measures. Everything here runs offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.omas.neoclassical import MODELS, compute_bootstrap_current
from vaft.validation.neoclassical import (
    METRICS,
    bootstrap_models,
    model_agreement,
    model_comparison,
)

PROFILE_RUN = Path(__file__).parent / "data" / "gacode" / "neo_vest_48224_profile"

SAMPLE = None
try:  # pragma: no cover - depends on the repository-only sample
    from vaft.data.resources import data_path

    _candidate = Path(data_path("kineticEfit/ods_48224_300ms.json"))
    SAMPLE = _candidate if _candidate.exists() else None
except Exception:
    SAMPLE = None

pytestmark = pytest.mark.skipif(
    SAMPLE is None, reason="the packaged 48224 kinetic sample is a repository-only asset"
)

#: Excludes the near-axis region the packaged reconstruction cannot support (#317).
BAND = (0.05, 1.0)


@pytest.fixture(scope="module")
def sample():
    from omas import load_omas_json

    # consistency_check=False: the committed sample carries leaves the installed DD no
    # longer recognises, which is a property of the sample.
    return load_omas_json(str(SAMPLE), consistency_check=False)


@pytest.fixture
def ods(sample):
    import copy

    return copy.deepcopy(sample)


@pytest.fixture
def mapped(ods):
    """The sample with NEO's bootstrap current written into it, as phase 5 does."""
    from vaft.code.gacode.neo.outputs import collect_neo_outputs
    from vaft.machine_mapping.neoclassical import core_profiles_from_neo

    core_profiles_from_neo(ods, collect_neo_outputs(PROFILE_RUN), time=0.3, time_index=0)
    return ods


# --------------------------------------------------------------------------
# The provider
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_the_provider_returns_a_profile_on_the_equilibrium_grid(ods, model):
    result = compute_bootstrap_current(ods, model=model, z_eff=2.0, rho_range=BAND)
    grid = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"])
    np.testing.assert_allclose(result.rho_tor_norm, grid)
    assert result.model == model
    assert np.isfinite(result.j_bootstrap).sum() > 100


def test_the_provider_agrees_with_the_kernels_it_wraps(ods):
    """The peak and the integral, against the values measured by hand."""
    result = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    area = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.area"])
    assert np.nanmax(result.j_bootstrap) == pytest.approx(11.61e3, rel=0.02)
    assert result.rho_tor_norm[np.nanargmax(result.j_bootstrap)] == pytest.approx(0.552, abs=0.02)
    assert result.integrated_current(area) == pytest.approx(2.58e3, rel=0.05)


def test_the_result_is_normalised_by_b0_so_it_matches_the_ids_definition(ods):
    """IMAS j_bootstrap is <J.B>/B0; the kernel returns <j_par B>."""
    result = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    finite = np.isfinite(result.parallel_current)
    np.testing.assert_allclose(
        result.j_bootstrap[finite], result.parallel_current[finite] / result.b0, rtol=1e-12
    )
    assert result.b0 == pytest.approx(
        float(np.ravel(ods["equilibrium.vacuum_toroidal_field.b0"])[0])
    )


def test_the_equilibriums_own_trapped_fraction_is_preferred(ods):
    """It integrated the field-strength distribution; the circular form approximates it."""
    result = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    assert result.provenance["trapped_fraction"] == "equilibrium.profiles_1d.trapped_fraction"
    np.testing.assert_allclose(
        result.trapped_fraction,
        np.asarray(ods["equilibrium.time_slice.0.profiles_1d.trapped_fraction"]),
    )


def test_the_circular_approximation_is_the_documented_fallback(ods):
    """And it differs from the shaped value, which is why it is only a fallback."""
    from vaft.formula.neoclassical import trapped_particle_fraction

    stored = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.trapped_fraction"]).copy()
    del ods["equilibrium.time_slice.0.profiles_1d.trapped_fraction"]
    result = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    assert "circular" in result.provenance["trapped_fraction"]
    assert not np.allclose(result.trapped_fraction, stored, atol=1e-3)
    assert trapped_particle_fraction(0.3) > 0.0  # the fallback is the documented one


def test_a_missing_equilibrium_leaf_is_named_not_substituted(ods):
    del ods["equilibrium.time_slice.0.profiles_1d.q"]
    with pytest.raises(ValueError, match="missing q"):
        compute_bootstrap_current(ods, model="sauter", z_eff=2.0)


def test_a_missing_kinetic_profile_is_named_not_substituted(ods):
    del ods["core_profiles.profiles_1d.0.electrons.temperature"]
    with pytest.raises(ValueError, match="electron temperature"):
        compute_bootstrap_current(ods, model="sauter", z_eff=2.0)


def test_without_a_charge_the_provider_refuses_rather_than_assuming_one(ods):
    """A machine default belongs in vest.yaml with a status, not in a generic routine."""
    with pytest.raises(ValueError, match="no zeff"):
        compute_bootstrap_current(ods, model="sauter", rho_range=BAND)


def test_an_unknown_model_is_refused(ods):
    with pytest.raises(ValueError, match="model must be one of"):
        compute_bootstrap_current(ods, model="hinton", z_eff=2.0)


def test_the_flux_convention_is_applied_and_recorded(ods):
    """The kernels differentiate against psi per radian; the DD stores full weber."""
    result = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    assert result.provenance["wb_per_radian_factor"] == pytest.approx(1.0 / (2 * np.pi))


def test_the_provider_does_not_write_into_the_ods(ods):
    before = set(ods.flat().keys())
    compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    assert set(ods.flat().keys()) == before


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------


def test_the_study_evaluates_both_analytic_models(ods):
    models = bootstrap_models(ods, z_eff=2.0, rho_range=BAND)
    assert set(models["series"]) == {"sauter", "redl"}
    assert models["rho_tor_norm"].size > 100


def test_a_stored_result_joins_the_comparison_under_the_name_that_wrote_it(mapped):
    """Phase 5 writes core_profiles.code.name = NEO, which is where the label comes from."""
    models = bootstrap_models(mapped, z_eff=2.0, rho_range=BAND)
    assert "neo" in models["series"]
    assert models["series"]["neo"]["source"] == "stored"
    assert models["provenance"]["stored_series"] == "neo"


def test_the_comparison_returns_numbers_and_no_status(mapped):
    """#550: a model difference is a result, not a verdict."""
    comparison = model_comparison(bootstrap_models(mapped, z_eff=2.0, rho_range=BAND))
    flat = repr(comparison).lower()
    for word in ("pass", "fail", "warn", "status"):
        assert word not in flat, f"{word!r} leaked into a metrics-only result"


def test_redl_sits_closer_to_neo_than_sauter_does(mapped):
    """The headline of the whole umbrella, measured on VEST's own kinetic state."""
    comparison = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="neo"
    )
    sauter = comparison["models"]["sauter"]
    redl = comparison["models"]["redl"]
    assert abs(redl["integrated_relative_difference"]) < abs(
        sauter["integrated_relative_difference"]
    )
    assert redl["rms_over_peak"] < sauter["rms_over_peak"]
    # Both are within a factor of two of the solver: different models, not a broken unit.
    for entry in (sauter, redl):
        assert abs(entry["integrated_relative_difference"]) < 0.5


def test_the_measured_metrics_stay_where_they_were(mapped):
    comparison = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="neo"
    )
    assert comparison["models"]["sauter"]["rms_over_peak"] == pytest.approx(0.076, abs=0.02)
    assert comparison["models"]["redl"]["rms_over_peak"] == pytest.approx(0.045, abs=0.02)
    sauter_vs_redl = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="sauter"
    )["models"]["redl"]
    assert sauter_vs_redl["integrated_relative_difference"] == pytest.approx(0.049, abs=0.02)


def test_the_disagreement_has_no_single_direction_against_trapping(ods):
    """The claim Redl was implemented for, and which this state cannot support.

    It was asserted as True here until #808. The gap is U-shaped in f_trap: large and
    badly scattered in the innermost bin, a minimum near 0.75-0.8, then a clean rise to
    the edge. The old lowest-third/highest-third verdict read that as "grows" and a
    straight-line fit reads it as "shrinks"; neither is a property of the models.
    """
    comparison = model_comparison(
        bootstrap_models(ods, z_eff=2.0, rho_range=BAND), reference="sauter"
    )
    trend = comparison["models"]["redl"]["trend"]
    assert trend["f_trap_high"] > trend["f_trap_low"]
    assert trend["grows_with_trapping"] is None
    assert trend["monotonic"] is None
    assert "falls and rises" in trend["reason"]

    # The shape that makes it so, visible to a reader rather than only to this test.
    differences = [entry["difference"] for entry in trend["bins"]]
    assert differences[0] > differences[1], "the innermost bin is the contaminated one"
    assert differences[1:] == sorted(differences[1:]), "and outside it the gap rises"


@pytest.mark.parametrize(
    "z_eff, impurity",
    [(2.0, None), (2.0, "C"), (1.0, None)],
    ids=["published", "carbon", "hydrogen"],
)
def test_the_trend_verdict_does_not_change_with_the_treatment(ods, z_eff, impurity):
    """#808's acceptance criterion: whatever it reports, it must not silently reverse.

    These three treatments differ by under a percentage point in the integrated
    comparison and by the effective charge, and the old two-point verdict gave True,
    False and False for them. All three must now give the same answer, and it must be
    the one that says the question is not answerable here.
    """
    comparison = model_comparison(
        bootstrap_models(ods, z_eff=z_eff, impurity=impurity, rho_range=BAND),
        reference="sauter",
    )
    trend = comparison["models"]["redl"]["trend"]
    assert trend["grows_with_trapping"] is None
    assert trend["monotonic"] is None
    assert trend["points"] > 6


def test_a_monotonic_sequence_still_gets_an_answer():
    """The refusal must be about the data, not a metric that can no longer decide."""
    from vaft.validation.neoclassical import _monotonic_direction

    rising = [
        {"difference": 0.01, "stderr": 0.0005},
        {"difference": 0.02, "stderr": 0.0005},
        {"difference": 0.04, "stderr": 0.0005},
    ]
    assert _monotonic_direction(rising) == (1, None)
    falling = [dict(entry) for entry in reversed(rising)]
    assert _monotonic_direction(falling) == (-1, None)


def test_a_step_inside_its_own_scatter_is_flat_rather_than_a_reversal():
    """Otherwise noise on one bin would veto an otherwise clean direction."""
    from vaft.validation.neoclassical import _monotonic_direction

    noisy = [
        {"difference": 0.010, "stderr": 0.002},
        {"difference": 0.009, "stderr": 0.002},  # down, but well inside the scatter
        {"difference": 0.030, "stderr": 0.002},
    ]
    direction, reason = _monotonic_direction(noisy)
    assert direction == 1 and reason is None


def test_a_sequence_with_no_resolved_step_reports_why():
    from vaft.validation.neoclassical import _monotonic_direction

    flat = [{"difference": 0.01, "stderr": 0.01} for _ in range(4)]
    direction, reason = _monotonic_direction(flat)
    assert direction is None
    assert "within its own scatter" in reason


def test_the_trend_is_measured_only_where_the_current_is_significant(ods):
    """A relative metric across a zero crossing is dominated by the crossing."""
    comparison = model_comparison(
        bootstrap_models(ods, z_eff=2.0, rho_range=BAND), reference="sauter"
    )
    trend = comparison["models"]["redl"]["trend"]
    assert trend["evaluated_above"] == 0.1
    # Nothing like the 487 % a pointwise ratio reaches across the whole profile.
    assert trend["difference_high"] < 1.0


def test_the_effect_size_says_how_much_bootstrap_there_is_to_compare(mapped):
    """A 5 % disagreement about a 2 % current is a different claim from one about 40 %."""
    comparison = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="neo"
    )
    effect = comparison["effect_size"]
    ip = float(mapped["equilibrium.time_slice.0.global_quantities.ip"])
    assert effect["integrated_current"] / ip == pytest.approx(0.015, abs=0.01)
    assert 0.3 < effect["peak_rho"] < 0.8


def test_neos_unsolved_radii_are_excluded_not_assumed(mapped):
    """Phase 5 leaves j_bootstrap NaN outside the surfaces NEO solved."""
    models = bootstrap_models(mapped, z_eff=2.0, rho_range=BAND)
    comparison = model_comparison(models, reference="neo")
    total = models["rho_tor_norm"].size
    assert comparison["overlap"] < total
    assert comparison["models"]["sauter"]["points"] == comparison["overlap"]


def test_a_reference_with_no_overlap_reports_a_reason_not_a_number(ods):
    models = bootstrap_models(ods, z_eff=2.0, rho_range=BAND)
    models["series"]["empty"] = {
        "j_bootstrap": np.full(models["rho_tor_norm"].size, np.nan),
        "source": "stored",
    }
    comparison = model_comparison(models, reference="sauter")
    assert comparison["models"]["empty"]["reason"]
    assert all(comparison["models"]["empty"][metric] is None for metric in METRICS)


def test_an_unknown_reference_is_refused(ods):
    with pytest.raises(ValueError, match="is not among the evaluated series"):
        model_comparison(bootstrap_models(ods, z_eff=2.0, rho_range=BAND), reference="nclass")


# --------------------------------------------------------------------------
# Tolerances belong to the caller
# --------------------------------------------------------------------------


def test_the_tolerances_are_the_callers_and_are_echoed_back(mapped):
    comparison = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="neo"
    )
    agreement = model_agreement(comparison, {"rms_over_peak": 0.20})
    assert agreement["tolerances"] == {"rms_over_peak": 0.20}
    assert agreement["models"]["redl"]["joint"] is True


def test_a_tighter_tolerance_separates_the_two_models(mapped):
    """Which is the point of letting the caller choose it."""
    comparison = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="neo"
    )
    agreement = model_agreement(comparison, {"rms_over_peak": 0.06})
    assert agreement["models"]["redl"]["joint"] is True
    assert agreement["models"]["sauter"]["joint"] is False


def test_an_unknown_metric_is_refused_rather_than_ignored(mapped):
    comparison = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="neo"
    )
    with pytest.raises(ValueError, match="unknown metric"):
        model_agreement(comparison, {"rms_over_pea": 0.1})


def test_an_empty_tolerance_mapping_decides_nothing_and_says_so(mapped):
    comparison = model_comparison(
        bootstrap_models(mapped, z_eff=2.0, rho_range=BAND), reference="neo"
    )
    with pytest.raises(ValueError, match="nothing to decide"):
        model_agreement(comparison, {})


# --------------------------------------------------------------------------
# Review findings
# --------------------------------------------------------------------------


def test_gradients_ignore_the_points_the_mask_rejects(ods):
    """np.gradient is centred, so a zero-density boundary corrupts its neighbour.

    The packaged profiles reach exactly zero at the edge. Taking the derivative over the
    whole array made the last *usable* radius 23 % wrong, at a point the result reports.
    """
    result = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    finite = np.flatnonzero(np.isfinite(result.j_bootstrap))
    last = int(finite[-1])

    # Reproduce both readings of the pressure gradient at that radius.
    eq = "equilibrium.time_slice.0.profiles_1d"
    cp = "core_profiles.profiles_1d.0"
    psi_rad = np.asarray(ods[f"{eq}.psi"]) / (2 * np.pi)
    ne = np.asarray(ods[f"{cp}.electrons.density_thermal"])
    te = np.asarray(ods[f"{cp}.electrons.temperature"])
    ti = np.asarray(ods[f"{cp}.ion.0.temperature"])
    pressure = ne * te * 1.602176634e-19 + ne * ti * 1.602176634e-19
    physical = (ne > 0) & (te > 0) & (ti > 0)

    contaminated = np.gradient(pressure, psi_rad)[last]
    correct = np.full(pressure.size, np.nan)
    correct[physical] = np.gradient(pressure[physical], psi_rad[physical])
    # The two readings really do differ there -- otherwise this test proves nothing.
    assert abs(contaminated / correct[last] - 1.0) > 0.1

    # And the provider used the correct one: scaling the reported current back through
    # its own coefficients would land on `correct`, not on `contaminated`.
    assert np.isfinite(result.j_bootstrap[last])
    with_contamination = result.j_bootstrap[last] * contaminated / correct[last]
    assert not np.isclose(result.j_bootstrap[last], with_contamination)


def test_a_profile_too_short_for_a_centred_gradient_is_refused(ods):
    cp = "core_profiles.profiles_1d.0"
    density = np.asarray(ods[f"{cp}.electrons.density_thermal"]).copy()
    density[2:] = 0.0
    ods[f"{cp}.electrons.density_thermal"] = density
    with pytest.raises(ValueError, match="centred gradient"):
        compute_bootstrap_current(ods, model="sauter", z_eff=2.0)


def test_a_stored_series_on_another_grid_is_rejected_with_a_reason(mapped):
    """Equal length is not equal radii, and the sample's two grids happen to coincide."""
    cp = "core_profiles.profiles_1d.0"
    shifted = np.asarray(mapped[f"{cp}.grid.rho_tor_norm"]).copy()
    shifted = shifted * 0.9 + 0.05  # same length, different radii
    mapped[f"{cp}.grid.rho_tor_norm"] = shifted

    models = bootstrap_models(mapped, z_eff=2.0, rho_range=BAND)
    assert "neo" not in models["series"]
    rejected = models["provenance"]["stored_series_rejected"]
    assert rejected["label"] == "neo"
    assert "different radii" in rejected["reason"]


def test_a_stored_series_on_the_same_grid_is_admitted(mapped):
    """The control for the test above: the sample's grids do coincide."""
    models = bootstrap_models(mapped, z_eff=2.0, rho_range=BAND)
    assert "neo" in models["series"]
    assert "stored_series_rejected" not in models["provenance"]


def test_the_radial_extent_is_derived_when_the_equilibrium_lacks_it(ods):
    """r_inboard/r_outboard are derived leaves; many reconstructions lack them.

    They are rebuilt on an isolated copy, so the caller's ODS is left as it was.
    """
    eq = "equilibrium.time_slice.0.profiles_1d"
    stored = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)
    del ods[f"{eq}.r_inboard"]
    del ods[f"{eq}.r_outboard"]
    derived = compute_bootstrap_current(ods, model="sauter", z_eff=2.0, rho_range=BAND)

    assert f"{eq}.r_inboard" not in ods, "the caller's ODS must not gain the derived leaf"

    # The derived extent comes from interpolating the 2-D flux map and the stored one
    # from whatever wrote the sample, so they are close but not identical. What has to
    # hold is that the physics is the same: the profile's scale and where it peaks.
    area = np.asarray(ods[f"{eq}.area"])
    assert np.nanmax(derived.j_bootstrap) == pytest.approx(
        np.nanmax(stored.j_bootstrap), rel=0.01
    )
    assert derived.integrated_current(area) == pytest.approx(
        stored.integrated_current(area), rel=0.01
    )
    finite = np.isfinite(stored.j_bootstrap) & np.isfinite(derived.j_bootstrap)
    difference = derived.j_bootstrap[finite] - stored.j_bootstrap[finite]
    rms = np.sqrt(np.mean(difference**2)) / np.nanmax(np.abs(stored.j_bootstrap))
    assert rms < 0.01
