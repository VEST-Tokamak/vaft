"""The canonical EFIT channel conventions, made explicit (issue #186).

`vaft.omas.efit_quality`, `vaft.validation` and the EFIT plot recipes used to
derive their own channel masks and their own chi-square channel set, so the same
family could report two different residual RMS values depending on which entry
point produced it.  These tests pin the single convention they now share:

* residual-unit statistics run over the **fitted** channels -- ``enabled`` with a
  finite residual, independent of whether the normalization was recoverable;
* sigma-unit statistics run over the **normalizable** channels, a strict subset;
* a family's chi-square aggregate sums the **enabled** channels only.
"""

from __future__ import annotations

import math

import numpy as np
from omas import ODS
import pytest

from vaft.omas.efit_quality import (
    FAMILIES,
    constraint_table,
    family_chi_squared_sum,
    fit_quality_metrics,
    fitted_mask,
    normalizable_mask,
    normalized_residuals,
    sigma_unit_factor,
    slice_times,
)


def _table(channels, *, family="bpol_probe"):
    """One constraint family whose channels are described literally.

    Each entry is ``(measured, reconstructed, weight, chi_squared)``.
    """
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = [0.30]
    root = "equilibrium.time_slice.0"
    ods[f"{root}.time"] = 0.30
    for index, (measured, reconstructed, weight, chi) in enumerate(channels):
        base = f"{root}.constraints.{family}.{index}"
        ods[f"{base}.measured"] = measured
        ods[f"{base}.reconstructed"] = reconstructed
        ods[f"{base}.weight"] = weight
        ods[f"{base}.chi_squared"] = chi
        ods[f"{base}.source"] = f"{family}_ch{index}"
    return constraint_table(ods, time_slice=0, family=family, is_array=True)


# ---------------------------------------------------------------------------
# The three definitions
# ---------------------------------------------------------------------------

def test_fitted_mask_keeps_enabled_channels_with_a_finite_residual():
    table = _table(
        [
            (1.5, 1.0, 0.01, 0.25e-4),      # enabled, finite residual  -> in
            (0.0, 0.0, 0.0, 0.0),           # missing (zero measured)   -> out
            (2.0, 0.0, 0.0, 0.0),           # disabled (zero weight)    -> out
            (float("nan"), 1.0, 0.01, 1.0),  # enabled, nan residual    -> out
        ]
    )

    np.testing.assert_array_equal(fitted_mask(table), [True, False, False, False])


def test_fitted_mask_survives_an_unrecoverable_normalization():
    """A prescribed family has no chi-square, so ``k`` -- and every ``z`` -- is nan.

    The residual is still perfectly well defined, and reporting it is the whole
    point of keeping the residual-unit mask independent of the normalization.
    """
    table = _table([(1.0, 1.0, 0.01, 0.0), (2.0, 2.0, 0.01, 0.0)])

    k, _spread = sigma_unit_factor(table)
    z = normalized_residuals(table, k)

    assert math.isnan(k)
    assert np.all(np.isnan(z))
    assert fitted_mask(table).all()
    assert not normalizable_mask(table, z).any()


def test_normalizable_mask_is_a_strict_subset_of_the_fitted_mask():
    table = _table(
        [
            (1.5, 1.0, 0.01, 0.25e-4),
            (1.2, 1.0, float("nan"), 1.0),  # enabled, but no usable weight
            (0.0, 0.0, 0.0, 0.0),
        ]
    )
    z = normalized_residuals(table, sigma_unit_factor(table)[0])

    fitted = fitted_mask(table)
    normalizable = normalizable_mask(table, z)

    assert np.all(normalizable <= fitted)
    assert normalizable.sum() < fitted.sum()


def test_family_chi_squared_sum_ignores_channels_efit_never_fitted():
    """A zero-weight channel did not enter the minimization, whatever it stores."""
    table = _table(
        [
            (1.5, 1.0, 0.01, 4.0),   # enabled
            (2.0, 0.0, 0.0, 99.0),   # disabled, but carries a stored chi-square
            (0.0, 0.0, 0.0, 7.0),    # missing, likewise
        ]
    )

    assert family_chi_squared_sum(table) == pytest.approx(4.0)
    # The naive sum over every channel is what the old validation layer used.
    assert float(np.nansum(table.chi_squared)) == pytest.approx(110.0)


def test_family_chi_squared_sum_skips_non_finite_enabled_entries():
    table = _table(
        [(1.5, 1.0, 0.01, 4.0), (1.2, 1.0, 0.01, float("nan"))]
    )

    assert family_chi_squared_sum(table) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# What the unification changes, end to end
# ---------------------------------------------------------------------------

def test_prescribed_family_reports_a_zero_residual_rms_not_nan():
    """The behaviour change: an exactly honoured input now says so.

    ``pf_current`` is handed back unchanged, so its chi-square is identically
    zero and ``k`` is unrecoverable.  Keyed to the normalized residual, the
    reported RMS was ``nan`` -- indistinguishable from "not measured".  Keyed to
    the residual, it is ``0.0``: the input was honoured exactly.
    """
    from test_efit_fit_quality import _complete_efit_quality_ods

    metrics = fit_quality_metrics(_complete_efit_quality_ods(), time_slice=0)
    entry = metrics["families"]["pf_current"]

    assert entry["fit_role"] == "prescribed"
    assert entry["residual_rms_display"] == 0.0
    # Prescribed families still contribute no goodness-of-fit statistics.
    assert "z_rms" not in entry


def test_validation_and_fit_quality_agree_family_by_family():
    """The point of the unification: one number per family, not two."""
    from test_efit_fit_quality import _complete_efit_quality_ods
    from vaft.validation import _efit_metrics

    ods = _complete_efit_quality_ods()
    report = _efit_metrics(ods)

    for index in range(slice_times(ods).size):
        quality = fit_quality_metrics(ods, time_slice=index)["families"]
        reported = report["slices"][index]["families"]
        for family, _title, _unit, scale, is_array in FAMILIES:
            if family not in quality:
                continue
            table = constraint_table(
                ods, time_slice=index, family=family, is_array=is_array
            )
            # fit_quality_metrics reports the RMS in display units; the
            # validation layer reports it in SI.  Same channels, same formula.
            np.testing.assert_allclose(
                quality[family]["residual_rms_display"],
                reported[family]["residual_rms"] * scale,
                rtol=1e-12,
                atol=0.0,
            )
            assert quality[family]["chi_squared_sum"] == pytest.approx(
                reported[family]["chi_squared_sum"]
            )
            assert reported[family]["chi_squared_sum"] == pytest.approx(
                family_chi_squared_sum(table)
            )
