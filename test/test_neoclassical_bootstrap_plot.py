"""The bootstrap-current plot says why a model is missing (cold review plot F3).

Every ``ValueError`` of the provider was swallowed whenever the ODS carried a
stored ``j_bootstrap``: a misspelt ``models=`` and a missing ion temperature
both produced the stored curve alone, with no warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from _sample_fixtures import sample_ods
from _synthetic_inputs import make_neoclassical
from vaft.plot.backend.recipes import RECIPES

BUILD = RECIPES["neoclassical_profile_bootstrap_current"].builder


@pytest.fixture()
def stored():
    ods = make_neoclassical(sample_ods())
    rho = np.asarray(ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"], dtype=float)
    ods["core_profiles.profiles_1d.0.j_bootstrap"] = 1.0e4 * (1.0 - rho**2)
    ods["core_profiles.code.name"] = "NEO"
    return ods


def test_a_complete_input_draws_the_models_without_a_warning(stored):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = BUILD(stored)
    assert {"sauter", "redl"} <= {series.label for series in model.series}


def test_a_provider_refusal_is_reported_beside_the_stored_curve(stored):
    del stored["core_profiles.profiles_1d.0.ion.0.temperature"]
    with pytest.warns(UserWarning, match="analytic bootstrap models not drawn.*temperature"):
        model = BUILD(stored)
    assert [series.label for series in model.series] == ["neo"]


def test_an_unknown_model_is_refused_not_replaced_by_the_stored_curve(stored):
    with pytest.raises(ValueError, match="model must be one of"):
        BUILD(stored, models=("no_such_model",))
