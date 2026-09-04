"""The #308 residual is gated on the diagnostics-stage assessment (#343/#189).

A product carries no marker saying whether its magnetics were ever assessed,
so the gate re-runs the assessment, projects it into a copy, and the residual
function refuses channels the gate excluded. On 39915 that is one probe --
H3-08, 40x its family median -- and it moves the normalized residual by 11%.
"""
from __future__ import annotations

import gzip
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from omas import ODS, load_omas_json

from vaft.omas.vacuum_magnetics import (
    QualityGate,
    VacuumMagneticsError,
    evaluation_mask,
    plasma_free_residual,
    quality_gate,
    synthetic_vacuum_magnetics,
)
from vaft.validation.vacuum_benchmark import benchmark_wall_currents, plasma_free_interval


def _real_shot(shot: int) -> ODS:
    import vaft.machine_mapping.em_coupling as em
    from vaft.data import resources

    try:
        source = resources.data_path(f"samples/{shot}/source/pipeline-until-efit.json.gz")
    except Exception:  # pragma: no cover
        pytest.skip("packaged pipeline sample unavailable")
    if not Path(source).is_file():
        pytest.skip("packaged pipeline sample is repository-only")
    with gzip.open(source, "rt") as handle, tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as plain:
        shutil.copyfileobj(handle, plain)
        plain_path = plain.name
    try:
        ods = load_omas_json(plain_path, consistency_check=False)
    finally:
        Path(plain_path).unlink(missing_ok=True)
    del ods["em_coupling"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        em.em_coupling(ods, shot=shot)
    return ods


@pytest.fixture(scope="module")
def shot_39915() -> ODS:
    return _real_shot(39915)


def _rms(channels, window) -> float:
    blocks = []
    for c in channels:
        m = evaluation_mask(c, window)
        if not m.any():
            continue
        scale = float(np.sqrt(np.mean(c.measured[m] ** 2)))
        blocks.append((c.measured[m] - c.coil_eddy[m]) / (scale if scale > 0 else 1.0))
    stacked = np.concatenate(blocks)
    return float(np.sqrt(np.mean(stacked**2)))


def test_the_gate_excludes_h3_08_on_39915_for_the_reasons_343_detects(shot_39915):
    interval = plasma_free_interval(shot_39915)
    window = (interval.start, interval.end)
    gated, gate = quality_gate(shot_39915, window=window)
    # The eight IMPA channels were already invalid in the product; the gate
    # reports every unusable channel in the window, whichever stage decided.
    impa = tuple(name for name in gate.excluded if name.startswith("IMPA"))
    assert len(impa) == 8
    assert set(gate.excluded) == set(impa) | {"MagneticFieldProbe_H3-08_Bz"}
    assert all(gate.reasons[name] for name in impa)
    reasons = set(gate.reasons["MagneticFieldProbe_H3-08_Bz"])
    assert {"implausible_magnitude", "population_outlier"} <= reasons
    assert gate.validity_source == "re-assessed here"
    assert gate.assessed == 87
    # the source was not written
    assert "validity" not in shot_39915["magnetics.b_field_pol_probe.25.field"]
    assert "validity" in gated["magnetics.b_field_pol_probe.25.field"]


def test_gating_removes_the_probe_and_materially_changes_the_residual(shot_39915):
    """The user asked that this be reported either way: it does change it."""
    interval = plasma_free_interval(shot_39915)
    window = (interval.start, interval.end)
    kw = dict(per_family=None, window=window, validity_window=window)
    ungated = synthetic_vacuum_magnetics(benchmark_wall_currents(shot_39915), **kw)
    gated_ods, gate = quality_gate(shot_39915, window=window)
    gated = synthetic_vacuum_magnetics(benchmark_wall_currents(gated_ods), **kw)
    assert len(ungated) - len(gated) == 1
    assert "MagneticFieldProbe_H3-08_Bz" not in {c.name for c in gated}
    before, after = _rms(ungated, window), _rms(gated, window)
    assert after < before
    # measured 0.256 -> 0.228 over the window to the PF-pickup crossing; over
    # the window to the light's onset (#409) 0.356 -> 0.339, 4.8 %
    assert (before - after) / before > 0.04
    stacked = plasma_free_residual(gated, window, gate=gate, normalize=True)
    assert np.isclose(float(np.sqrt(np.mean(stacked**2))), after)


def test_the_residual_refuses_channels_the_gate_excluded(shot_39915):
    interval = plasma_free_interval(shot_39915)
    window = (interval.start, interval.end)
    _, gate = quality_gate(shot_39915, window=window)
    ungated = synthetic_vacuum_magnetics(
        benchmark_wall_currents(shot_39915), per_family=None, window=window, validity_window=window
    )
    with pytest.raises(VacuumMagneticsError, match="H3-08"):
        plasma_free_residual(ungated, window, gate=gate)


def test_a_supplied_report_is_used_instead_of_re_assessing(shot_39915):
    from vaft.validation.magnetics import validate_magnetics_signals

    report = validate_magnetics_signals(shot_39915)
    _, gate = quality_gate(shot_39915, report=report)
    assert gate.validity_source == "report supplied by caller"
    assert "MagneticFieldProbe_H3-08_Bz" in gate.excluded


def test_the_record_is_serialisable_and_carries_thresholds(shot_39915):
    interval = plasma_free_interval(shot_39915)
    _, gate = quality_gate(shot_39915, window=(interval.start, interval.end))
    record = gate.record()
    by_name = {row["channel"]: row["reasons"] for row in record["excluded"]}
    assert "implausible_magnitude" in by_name["MagneticFieldProbe_H3-08_Bz"]
    assert record["thresholds"]["population_peak_factor"] == 4.0
    assert record["thresholds"]["max_plausible_amplitude"] == {"b_field_pol_probe": 1.0}
    import json

    json.dumps(record)


def test_a_gate_with_nothing_excluded_is_a_pure_pass_through():
    gate = QualityGate(
        assessed=0, window=None, excluded=(), partially_masked=(), reasons={}, config={},
        validity_source="test",
    )
    gate.check([])
    assert plasma_free_residual([], None, gate=gate).shape == (0,)


def test_a_window_with_no_magnetics_samples_is_refused_not_reported_clean(shot_39915):
    """The magnetics record ends at 0.36 s; a gate over 0.50-0.60 s used to
    come back 'assessed 87, nothing excluded'."""
    with pytest.raises(VacuumMagneticsError, match="no samples"):
        quality_gate(shot_39915, window=(0.50, 0.60))


def test_a_gate_for_another_window_is_refused(shot_39915):
    interval = plasma_free_interval(shot_39915)
    window = (interval.start, interval.end)
    gated, gate = quality_gate(shot_39915, window=window)
    channels = synthetic_vacuum_magnetics(
        benchmark_wall_currents(gated), per_family=None, window=window, validity_window=window
    )
    other = (window[0], window[0] + 0.5 * (window[1] - window[0]))
    with pytest.raises(VacuumMagneticsError, match="built for window"):
        plasma_free_residual(channels, other, gate=gate)
    # the window it was built for is, of course, accepted
    assert plasma_free_residual(channels, window, gate=gate).size > 0
