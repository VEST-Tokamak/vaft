"""`compute_tau_E_engineering_parameters` reports a missing equilibrium leaf
by name and does not create it (cold review 0.8.0
process-ml-and-omas-wrappers N1).

On the consistency_check=False ODSs VAFT hands out, the bare
``float(eq_ts['boundary.elongation'])`` raised ``TypeError: float() ...
'ODS'`` and attached an empty ``boundary.elongation`` node; the summary
swallowed the exception per slice, so the symptom was an empty row.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("omas")

from vaft.omas import formula_wrapper as fw
from vaft.omas.sample import sample_ods


def test_a_missing_elongation_is_named_and_not_materialised():
    from _synthetic_inputs import make_power_balance

    ods = make_power_balance(sample_ods(39915))
    i = len(ods["equilibrium.time_slice"]) // 2
    ts = ods["equilibrium.time_slice"][i]
    assert ods.consistency_check is False
    assert "boundary.elongation" not in ts
    # update_equilibrium_boundary cannot derive it either on this sample.
    assert "profiles_1d.elongation" not in ts

    logging.disable(logging.WARNING)
    try:
        with pytest.raises(KeyError, match="boundary.elongation"):
            fw.compute_tau_E_engineering_parameters(ods, i)
    finally:
        logging.disable(logging.NOTSET)

    boundary = ods["equilibrium.time_slice"][i]["boundary"]
    assert "elongation" not in list(boundary.keys())
    assert "boundary.elongation" not in ods["equilibrium.time_slice"][i]
