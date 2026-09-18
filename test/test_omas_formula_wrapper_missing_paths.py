"""`compute_magnetic_shear` names the node it lacks instead of creating it.

An OMAS read of an absent path hands back an empty ODS *and creates the path*,
so the wrapper used to fail inside ``numpy.gradient`` with a shape error that
named neither the ODS nor the node, and left an empty ``profiles_1d.r`` behind
(cold review docs D2: ``docs/_guide/Formula.md`` ran it on the packaged sample,
which has no such node).
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.omas.formula_wrapper import compute_magnetic_shear


def _ods(with_r: bool) -> ODS:
    ods = ODS(consistency_check=False)
    base = "equilibrium.time_slice.0.profiles_1d"
    ods[f"{base}.q"] = np.linspace(1.0, 4.0, 11)
    if with_r:
        ods[f"{base}.r"] = np.linspace(0.0, 0.3, 11)
    return ods


def test_a_missing_radius_is_named_and_not_created():
    ods = _ods(with_r=False)
    with pytest.raises(KeyError, match=r"equilibrium\.time_slice\.0\.profiles_1d\.r"):
        compute_magnetic_shear(ods, 0)
    assert "equilibrium.time_slice.0.profiles_1d.r" not in ods


def test_the_shear_is_unchanged_when_the_nodes_exist():
    ods = _ods(with_r=True)
    r, q = np.linspace(0.0, 0.3, 11), np.linspace(1.0, 4.0, 11)
    np.testing.assert_allclose(compute_magnetic_shear(ods, 0), (r / q) * np.gradient(q, r))
