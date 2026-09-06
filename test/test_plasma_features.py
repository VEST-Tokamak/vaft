"""The representative peaks of issue #409 (PR-vi).

The packaged products pin the measured peaks inside the plasma window the
timing found -- and 39915's slow H-alpha shows why a representative peak is
not the loudest sample: its raw maximum is a one-sample spike sitting on the
window's last sample.  The synthetic shots exercise every outcome the composer
must report rather than guess: nothing computed without a timing, a
configured line the product lacks, a railed line, a spike on the window edge.
"""
from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import vaft.omas
import vaft.omas.plasma_features as module
from vaft.machine_mapping.utils import resolve_plasma_features_policy
from vaft.omas.plasma_features import (
    ABSENT,
    DIAMAGNETIC_BASE,
    NOT_COMPUTED,
    Feature,
    PlasmaFeatures,
    line_by_label,
    plasma_features,
)
from vaft.omas.plasma_timing import HALPHA_LABEL, PlasmaTimingError, plasma_timing

from _plasma_timing_fixtures import DT, RNG, current, grid, light, pickup_only, pipeline_ods, synthetic_ods


# ---------------------------------------------------------------------------
# Packaged shots
# ---------------------------------------------------------------------------

PACKAGED = {
    # Ip peak (A), H-alpha peak, diamagnetic peak (Wb), CIII peak
    39915: dict(ip=83852.0, h_alpha=0.2986, diamagnetic=-1.553e-3, ciii=0.7005),
    41524: dict(ip=222026.0, h_alpha=0.8498, diamagnetic=-8.545e-3, ciii=3.596),
    41672: dict(ip=127105.0, h_alpha=0.5586, diamagnetic=-3.900e-3, ciii=0.9297),
}


@pytest.mark.parametrize("shot", sorted(PACKAGED))
def test_the_packaged_products_pin_the_peaks_inside_the_window(shot):
    ods = pipeline_ods(shot)
    expected = PACKAGED[shot]

    features = plasma_features(ods)

    assert isinstance(features, PlasmaFeatures) and features.computed and features.reason is None
    onset, offset = features.window
    for feature in features.features().values():
        assert feature.found, feature.name
        assert onset <= feature.time <= offset, feature.name
    assert features.ip.value == pytest.approx(expected["ip"], rel=1e-3)
    raw_in_window = np.asarray(ods["magnetics.ip.0.data"])[(ods["magnetics.time"] >= onset) & (ods["magnetics.time"] <= offset)]
    assert features.ip.value == pytest.approx(raw_in_window.max(), rel=1e-2)
    assert features.ip.time in np.asarray(ods["magnetics.time"])
    assert features.ip.notes["timing_source"] == "h_alpha_primary"
    assert features.h_alpha.label == HALPHA_LABEL and features.h_alpha.notes["role"] == "h_alpha_primary"
    assert features.h_alpha.value == pytest.approx(expected["h_alpha"], rel=1e-3)
    assert features.h_alpha.notes["railed"] is False
    assert features.lines["OI_7770"].base.startswith("spectrometer_uv.channel.1.")
    assert features.lines["CIII_1909"].base.startswith("spectrometer_uv.channel.2.")
    assert features.lines["CIII_1909"].value == pytest.approx(expected["ciii"], rel=1e-3)
    assert features.diamagnetic.value == pytest.approx(expected["diamagnetic"], rel=1e-3)
    assert features.diamagnetic.value < 0 and "reference_flat" in features.diamagnetic.flags
    assert features.diamagnetic.notes == {"method_name": None, "saturated": False}
    assert features.flags == ()
    json.dumps(features.record())
    json.dumps(features.summary())


def test_39915s_loudest_h_alpha_sample_is_a_spike_on_the_window_edge():
    ods = pipeline_ods(39915)
    features = plasma_features(ods)
    h_alpha = features.h_alpha

    raw = np.asarray(ods["spectrometer_uv.channel.0.processed_line.0.intensity.data"])
    assert h_alpha.peak.evidence["raw_max"] == pytest.approx(raw.max(), rel=1e-6)
    assert h_alpha.peak.evidence["raw_max"] > 0.9
    assert features.window[1] - 2 * DT <= h_alpha.peak.evidence["raw_max_time"] <= features.window[1]
    assert h_alpha.value < 0.5
    assert "raw_max_outside_run" in h_alpha.flags
    assert "peak_at_window_edge" not in h_alpha.flags


def test_a_railed_line_is_noted():
    policy = resolve_plasma_features_policy()
    with_gamma = replace(policy, lines={"H-gamma_4340": dict(policy.h_alpha)})

    features = plasma_features(pipeline_ods(41524), policy=with_gamma)

    gamma = features.lines["H-gamma_4340"]
    assert gamma.found and gamma.notes["railed"] is True and "railed" in gamma.flags
    assert gamma.peak.evidence["raw_max"] == pytest.approx(5.0, abs=1e-6)   # the digitizer's rail
    assert gamma.value < 5.0                                                 # the rail was brief: a spike, refused
    assert features.h_alpha.notes["railed"] is False


def test_the_compact_sample_measures_the_same_peaks():
    features = plasma_features(vaft.omas.sample_ods())
    assert features.ip.value == pytest.approx(PACKAGED[39915]["ip"], rel=1e-3)
    assert features.diamagnetic.value == pytest.approx(PACKAGED[39915]["diamagnetic"], rel=1e-3)


def test_a_timing_handed_in_is_used_as_is():
    ods = pipeline_ods(41672)
    timing = plasma_timing(ods)
    assert plasma_features(ods, timing=timing).summary() == plasma_features(ods).summary()


# ---------------------------------------------------------------------------
# Synthetic shots
# ---------------------------------------------------------------------------


def with_extras(ods, t, *, diamagnetic=True, line=None):
    """``synthetic_ods`` plus a diamagnetic node and, optionally, a labelled line on channel 2."""
    if diamagnetic:
        flux = np.zeros_like(t)
        inside = (t >= 0.306) & (t <= 0.331)
        flux[inside] = -1e-3 * np.sin(np.pi * (t[inside] - 0.306) / 0.025)
        ods[f"{DIAMAGNETIC_BASE}.data"] = flux
    if line is not None:
        label, data = line
        ods["spectrometer_uv.channel.2.processed_line.0.label"] = "H-beta_4861"   # OMAS arrays cannot skip an index
        ods["spectrometer_uv.channel.2.processed_line.0.intensity.data"] = np.zeros_like(t)
        ods["spectrometer_uv.channel.2.processed_line.1.label"] = label
        ods["spectrometer_uv.channel.2.processed_line.1.intensity.data"] = np.asarray(data, dtype=float)
    return ods


def test_a_synthetic_shot_peaks_inside_its_window():
    t = grid()
    ods = with_extras(synthetic_ods(slow=light(t), ip=current(t), t=t), t,
                      line=("CIII_1909", 0.5 * light(t, onset=0.308)))

    features = plasma_features(ods)

    assert features.computed
    onset, offset = features.window
    assert features.ip.found and onset <= features.ip.time <= offset
    assert features.ip.value == pytest.approx(60e3, rel=0.03)
    assert features.h_alpha.found and features.h_alpha.value == pytest.approx(1.0, abs=0.05)
    assert features.lines["CIII_1909"].found and features.lines["CIII_1909"].value == pytest.approx(0.5, abs=0.05)
    assert features.lines["OI_7770"].flags == (ABSENT,) and "line_absent:OI_7770" in features.flags
    assert features.lines["OI_7770"].reason.startswith("no processed_line carries label")
    assert features.diamagnetic.found and features.diamagnetic.value == pytest.approx(-1e-3, rel=0.05)
    assert features.diamagnetic.time == pytest.approx(0.3185, abs=5e-4)


def test_a_spike_on_the_window_edge_does_not_move_the_peak():
    t = grid()
    slow = light(t)
    clean = plasma_features(synthetic_ods(slow=slow, ip=current(t), t=t))
    spiked = slow.copy()
    edge = int(np.argmin(np.abs(t - clean.window[1])))
    spiked[edge] += 5.0
    features = plasma_features(synthetic_ods(slow=spiked, ip=current(t), t=t))

    assert features.h_alpha.value == pytest.approx(clean.h_alpha.value, rel=1e-3)
    assert features.h_alpha.time == clean.h_alpha.time
    assert features.h_alpha.peak.evidence["raw_max"] > 5.0
    assert features.h_alpha.peak.evidence["raw_max_time"] == pytest.approx(t[edge])


def test_light_without_a_current_pulse_measures_no_ip_peak():
    t = grid()
    features = plasma_features(synthetic_ods(slow=light(t), ip=pickup_only(t), t=t))

    assert features.computed and features.timing.source == "h_alpha_primary"
    assert "ip_no_pulse" in features.ip.flags
    assert not features.ip.found and "all_candidates_impulsive" in features.ip.flags
    assert features.h_alpha.found


def test_nothing_is_computed_without_a_timing():
    t = grid()
    dark = 0.002 * RNG.standard_normal(t.size)
    features = plasma_features(synthetic_ods(slow=dark, ip=pickup_only(t), t=t))

    assert not features.computed and features.window is None
    assert features.reason == features.timing.fallback_reason and "ip_principal" in features.reason
    assert features.flags == ("no_plasma_timing",)
    for name, feature in features.features().items():
        assert feature.flags == (NOT_COMPUTED,), name
        assert feature.reason == features.reason
        assert not feature.found and feature.value is None
    assert set(features.lines) == set(resolve_plasma_features_policy().lines)
    json.dumps(features.record())


def test_a_product_without_plasma_current_raises_as_the_timing_does():
    t = grid()
    with pytest.raises(PlasmaTimingError):
        plasma_features(synthetic_ods(slow=light(t), t=t))


def test_a_missing_diamagnetic_node_is_absent():
    t = grid()
    features = plasma_features(synthetic_ods(slow=light(t), ip=current(t), t=t))
    assert features.diamagnetic.flags == (ABSENT,) and f"{features.diamagnetic.name}_{ABSENT}" in features.flags


def test_lines_are_found_by_their_stored_label():
    t = grid()
    ods = with_extras(synthetic_ods(slow=light(t), ip=current(t), t=t), t, line=("OII_3726", light(t)))
    assert line_by_label(ods, "OII_3726") == ("spectrometer_uv.channel.2.processed_line.1.intensity", 2, 1)
    assert line_by_label(ods, "CIII_1909") is None
    assert line_by_label(ods, HALPHA_LABEL) == ("spectrometer_uv.channel.0.processed_line.0.intensity", 0, 0)


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["vaft.omas.plasma_features", "vaft.omas.shot_class"])
def test_the_composers_read_only_and_stay_off_the_plot_layer(name):
    import importlib

    target = importlib.import_module(name)
    text = Path(target.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    assert set(m for m in imported if m.startswith("vaft.validation")) <= {"vaft.validation.imas"}
    assert not [m for m in imported if m.startswith(("vaft.plot", "vaft.database"))]
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target_ in node.targets
        if isinstance(target_, ast.Subscript) and isinstance(target_.value, ast.Name) and target_.value.id == "ods"
    ]
    assert writes == []
    assert "set_path" not in text


def test_reading_a_product_materialises_nothing():
    t = grid()
    ods = synthetic_ods(slow=light(t), ip=current(t), t=t)
    before = sorted(map(str, ods.flat().keys()))

    plasma_features(ods)

    assert sorted(map(str, ods.flat().keys())) == before
