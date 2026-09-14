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


def test_the_disagreement_grows_with_the_trapped_fraction(ods):
    """The claim the phase exists to support, and why Redl was implemented at all."""
    comparison = model_comparison(
        bootstrap_models(ods, z_eff=2.0, rho_range=BAND), reference="sauter"
    )
    trend = comparison["models"]["redl"]["trend"]
    assert trend["f_trap_high"] > trend["f_trap_low"]
    assert trend["grows_with_trapping"] is True
    assert trend["difference_high"] > trend["difference_low"]


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
