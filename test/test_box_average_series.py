"""One owner for the box average at every sample (#2162 B3.10).

Pipeline 1's ``generate_constraints_ods._box_averaged`` evaluated
:func:`vaft.code.efit.legacy.box_average` at every sample with a cumulative-sum
difference.  It moves next to ``box_average`` as ``box_average_series``.  The
frozen copy below is the workflow helper verbatim from develop fbede03a6: the
moved function is held to it bit for bit (same arithmetic), and to the
per-sample ``box_average`` -- the definition -- to 1e-12 relative (summation
order differs: a running sum against ``np.mean``).
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.code.efit.legacy import box_average


def _frozen_box_averaged(clock, data, half_width):
    lo = np.searchsorted(clock, clock - half_width, side="left")
    hi = np.searchsorted(clock, clock + half_width, side="right")
    total = np.concatenate(([0.0], np.cumsum(data)))
    return (total[hi] - total[lo]) / np.maximum(hi - lo, 1)


_RNG = np.random.default_rng(2162)
_CLOCK = np.arange(0.26, 0.36, 4e-5)
_CASES = {
    "ip_like": 5e4 * np.exp(-(((_CLOCK - 0.31) / 0.01) ** 2)) + 300.0 * _RNG.standard_normal(_CLOCK.size),
    "ringing": np.where(np.arange(_CLOCK.size) % 97 == 0, 4.1e4, 2.0e3),
    "constant": np.full(_CLOCK.size, -1.25e4),
}


def _series():
    from vaft.code.efit.legacy import box_average_series

    return box_average_series


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("half_width", [0.0, 4e-5, 5e-4, 2e-3, 1.0])
def test_moved_function_is_the_frozen_workflow_helper(case, half_width):
    data = _CASES[case]
    np.testing.assert_array_equal(
        _series()(_CLOCK, data, half_width), _frozen_box_averaged(_CLOCK, data, half_width)
    )


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("half_width", [0.0, 4e-5, 5e-4, 2e-3])
def test_it_is_box_average_at_every_sample(case, half_width):
    data = _CASES[case]
    expected = np.array([box_average(_CLOCK, data, float(t), half_width) for t in _CLOCK])
    scale = np.max(np.abs(data))
    np.testing.assert_allclose(_series()(_CLOCK, data, half_width), expected, rtol=1e-12, atol=1e-12 * scale)


def test_irregular_clock():
    clock = np.sort(_RNG.uniform(0.0, 1.0, 3_000))
    data = np.sin(40 * clock)
    expected = np.array([box_average(clock, data, float(t), 0.01) for t in clock])
    np.testing.assert_allclose(_series()(clock, data, 0.01), expected, rtol=1e-12, atol=1e-12)


def test_the_workflow_uses_it():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / (
        "workflow/automatic_pipeline_1_routine_data_processing/generate_constraints_ods.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_box_averaged" not in names
    assert "box_average_series" in path.read_text(encoding="utf-8")
