"""The discharge-timing events of issue #409 (PR-v).

The packaged products pin the measured events -- the ohmic onset, the
solenoid-driven loop-voltage excursion on the inboard-midplane loop and its
zero crossing (or the flagged approach that does not cross) -- and the
synthetic shots exercise every outcome the composer must report rather than
guess: a stored voltage preferred to ``-dflux/dt``, no inboard loop, no coil
by the configured name, an idle coil, and the actuators that are not mapped.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

import vaft.omas.discharge_timing as module
from vaft.machine_mapping.utils import (
    DischargeTimingPolicy,
    VestConfigurationError,
    resolve_discharge_timing_policy,
)
from vaft.omas.discharge_timing import (
    NOT_PRESENT,
    VOLTAGE_DERIVED,
    VOLTAGE_MEASURED,
    CoilOnset,
    DischargeTiming,
    LoopVoltageEvent,
    coil_onsets,
    discharge_timing,
    inboard_midplane_loop,
    loop_voltage,
    loop_voltage_event,
)

from _plasma_timing_fixtures import DT, RNG, grid, pipeline_ods


# ---------------------------------------------------------------------------
# Packaged shots: the measured events
# ---------------------------------------------------------------------------

PACKAGED = {
    # OH onset, excursion time/value, zero crossing, approach flag
    39915: dict(oh=0.2925, excursion=(0.2931, -5.44), crossing=0.3022, approach=False),
    41524: dict(oh=0.2936, excursion=(0.2942, -6.49), crossing=(0.310, 0.320), approach=True),
    41672: dict(oh=0.2936, excursion=(0.2941, -6.47), crossing=(0.310, 0.320), approach=True),
}
FIRED = {"PF1", "PF5", "PF6", "PF9", "PF10"}
IDLE = {"PF2", "PF3", "PF4", "PF7", "PF8"}


@pytest.mark.parametrize("shot", sorted(PACKAGED))
def test_the_packaged_products_pin_the_loop_voltage_event(shot):
    ods = pipeline_ods(shot)
    expected = PACKAGED[shot]

    timing = discharge_timing(ods)

    assert timing.oh is not None and timing.oh.name == "PF1" == timing.oh_coil
    assert timing.oh_onset == pytest.approx(expected["oh"], abs=5e-4)
    assert timing.oh.polarity == "positive"
    vloop = timing.vloop
    assert vloop.loop_index == 5 and vloop.loop_name == "Flux Loop - #10"
    assert vloop.position == pytest.approx((0.091, 0.04))
    assert vloop.voltage_source == VOLTAGE_DERIVED and "voltage_derived" in vloop.flags
    assert vloop.anchor_time == timing.oh_onset
    assert vloop.excursion_time == pytest.approx(expected["excursion"][0], abs=5e-4)
    assert vloop.excursion_value == pytest.approx(expected["excursion"][1], abs=0.3)
    assert vloop.found and timing.vloop_time == vloop.zero_crossing
    if isinstance(expected["crossing"], tuple):
        lo, hi = expected["crossing"]
        assert lo <= vloop.zero_crossing <= hi
    else:
        assert vloop.zero_crossing == pytest.approx(expected["crossing"], abs=5e-4)
    assert ("approached_without_crossing" in vloop.flags) is expected["approach"]
    if expected["approach"]:
        assert vloop.approach_time == pytest.approx(0.3070, abs=5e-4)
        assert abs(vloop.approach_min) < 0.3
    else:
        assert vloop.approach_min is None and vloop.approach_time is None
    assert "vloop_not_found" not in timing.flags and "oh_not_fired" not in timing.flags


@pytest.mark.parametrize("shot", sorted(PACKAGED))
def test_idle_coils_have_no_onset_and_fired_ones_do(shot):
    timing = discharge_timing(pipeline_ods(shot))

    by_name = {coil.name: coil for coil in timing.pf_onsets}
    assert set(by_name) == FIRED | IDLE
    assert [coil.index for coil in timing.pf_onsets] == list(range(10))
    for name in FIRED:
        assert by_name[name].found, name
        assert by_name[name].polarity in ("positive", "negative")
        assert timing.span.tstart - timing.span.baseline_lead_s <= by_name[name].time < timing.span.tend
    for name in IDLE:
        assert not by_name[name].found, name
        assert "reference_flat" in by_name[name].flags
        assert by_name[name].polarity is None
    assert by_name["PF6"].polarity == "negative"
    assert timing.summary()["pf_onsets"]["PF2"] is None
    json.dumps(timing.record())
    json.dumps(timing.summary())


def test_the_unmapped_actuators_are_reported_not_guessed():
    timing = discharge_timing(pipeline_ods(39915))

    assert timing.ec is None and timing.gas is None
    assert set(timing.not_present) == {"ec", "gas"} == set(NOT_PRESENT)
    assert "not_implemented" in timing.not_present["ec"]
    assert "response" in timing.not_present["gas"]


# ---------------------------------------------------------------------------
# Synthetic shots
# ---------------------------------------------------------------------------

OH_ONSET = 0.2925
COILS = ("PF1", "PF2", "PF3")


def coil_current(t, *, onset=OH_ONSET, peak=4000.0, rise=3e-3, sign=1.0, noise=2.0):
    y = np.zeros_like(t)
    on = t >= onset
    y[on] = sign * peak * np.clip((t[on] - onset) / rise, 0.0, 1.0)
    return y + noise * RNG.standard_normal(t.size)


def loop_volts(t, *, onset=OH_ONSET, depth=-5.0, decay_s=0.004, cross=True, noise=0.02):
    """The synthetic solenoid swing of ``test_process_onset``: a dip that decays and
    crosses on a slow positive ramp, or stalls just short of zero."""
    y = np.zeros_like(t)
    on = t >= onset
    tau = t[on] - onset
    swing = depth * np.exp(-tau / decay_s)
    swing += 1.0 * np.clip((tau - 0.018) / 0.010, 0.0, 1.0) if cross else -0.4
    y[on] = swing
    return y + noise * RNG.standard_normal(t.size)


def discharge_ods(
    *,
    t=None,
    coils=None,
    loops=None,
) -> ODS:
    """An ODS with ``pf_active`` coils ``{name: current}`` and flux loops
    ``[(r, z, {"flux": ..., "voltage": ...}), ...]`` on the analysis grid."""
    t = grid() if t is None else t
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 40000
    if coils is None:
        coils = {"PF1": coil_current(t), "PF2": np.zeros_like(t), "PF3": coil_current(t, onset=0.31, sign=-1.0)}
    ods["pf_active.ids_properties.homogeneous_time"] = 1
    ods["pf_active.time"] = t
    for index, (name, current) in enumerate(coils.items()):
        ods[f"pf_active.coil.{index}.name"] = name
        ods[f"pf_active.coil.{index}.identifier"] = name
        ods[f"pf_active.coil.{index}.current.data"] = np.asarray(current, dtype=float)
    if loops is None:
        v = loop_volts(t)
        loops = [
            (0.13, 0.65, {"flux": -np.cumsum(v) * DT}),        # inboard family, far from the midplane
            (0.091, 0.04, {"flux": -np.cumsum(v) * DT}),       # the inboard-midplane loop
            (0.55, 0.0, {"flux": np.zeros_like(t)}),           # outboard
        ]
    ods["magnetics.ids_properties.homogeneous_time"] = 1
    ods["magnetics.time"] = t
    for index, (r, z, records) in enumerate(loops):
        base = f"magnetics.flux_loop.{index}"
        ods[f"{base}.name"] = f"loop {index}"
        ods[f"{base}.position.0.r"] = r
        ods[f"{base}.position.0.z"] = z
        for node, data in records.items():
            ods[f"{base}.{node}.data"] = np.asarray(data, dtype=float)
    return ods


def test_a_synthetic_solenoid_swing_is_anchored_and_crossed():
    t = grid()
    timing = discharge_timing(discharge_ods(t=t))

    assert timing.oh_onset == pytest.approx(OH_ONSET, abs=5 * DT)   # 2 % threshold lag on a 3 ms ramp
    assert [c.found for c in timing.pf_onsets] == [True, False, True]
    assert timing.pf_onsets[2].polarity == "negative" and timing.pf_onsets[0].polarity == "positive"
    vloop = timing.vloop
    assert vloop.loop_index == 1 and vloop.voltage_source == VOLTAGE_DERIVED
    assert vloop.excursion_value == pytest.approx(-5.0, abs=0.5)
    assert vloop.excursion_time == pytest.approx(OH_ONSET, abs=1e-3)
    assert OH_ONSET + 0.018 < vloop.zero_crossing < OH_ONSET + 0.030
    assert vloop.zero_crossing in t
    assert timing.flags == ("voltage_derived",)


def test_a_stored_voltage_is_preferred_to_the_flux_derivative():
    t = grid()
    v = loop_volts(t)
    ods = discharge_ods(t=t, loops=[(0.091, 0.0, {"flux": np.zeros_like(t), "voltage": v})])

    read = loop_voltage(ods, 0)
    assert read is not None and read[2] == VOLTAGE_MEASURED
    np.testing.assert_array_equal(read[1], v)
    timing = discharge_timing(ods)
    assert timing.vloop.voltage_source == VOLTAGE_MEASURED
    assert "voltage_derived" not in timing.vloop.flags
    assert timing.vloop.found

    policy = resolve_discharge_timing_policy()
    derived = DischargeTimingPolicy(
        window=policy.window, baseline_lead_s=policy.baseline_lead_s, ohmic_coil=policy.ohmic_coil,
        loop_voltage={"selection": "inboard_midplane", "prefer_measured_voltage": False},
        coil=policy.coil, vloop=policy.vloop,
    )
    assert discharge_timing(ods, policy=derived).vloop.voltage_source == VOLTAGE_DERIVED


def test_the_inboard_loop_nearest_the_midplane_is_chosen():
    t = grid()
    ods = discharge_ods(t=t)
    assert inboard_midplane_loop(ods) == 1

    # a loop without any record is skipped even when it is nearer
    ods = discharge_ods(t=t, loops=[(0.09, 0.0, {}), (0.091, 0.04, {"flux": -np.cumsum(loop_volts(t)) * DT})])
    assert inboard_midplane_loop(ods) == 1
    # only outboard loops: none
    ods = discharge_ods(t=t, loops=[(0.55, 0.0, {"flux": np.zeros_like(t)})])
    assert inboard_midplane_loop(ods) is None
    timing = discharge_timing(ods)
    assert not timing.vloop.found and timing.vloop.loop_index == -1
    assert "loop_not_found" in timing.vloop.flags and "vloop_not_found" in timing.flags


def test_a_missing_ohmic_coil_is_a_flag_not_an_error():
    t = grid()
    ods = discharge_ods(t=t, coils={"PF9": coil_current(t), "PF10": np.zeros_like(t)})

    timing = discharge_timing(ods)

    assert timing.oh is None and timing.oh_onset is None
    assert "oh_coil_not_found" in timing.flags
    assert "no_oh_anchor" in timing.vloop.flags and not timing.vloop.found
    assert timing.vloop.loop_index == 1   # the loop was still identified
    json.dumps(timing.record())


def test_an_idle_ohmic_coil_is_oh_not_fired():
    t = grid()
    ods = discharge_ods(t=t, coils={"PF1": np.zeros_like(t)})

    timing = discharge_timing(ods)

    assert timing.oh is not None and not timing.oh.found
    assert "oh_not_fired" in timing.flags and "no_oh_anchor" in timing.vloop.flags


def test_a_swing_that_never_crosses_is_reported():
    t = grid()
    v = loop_volts(t, cross=False)
    ods = discharge_ods(t=t, loops=[(0.091, 0.0, {"voltage": v})])

    vloop = discharge_timing(ods).vloop

    assert not vloop.found
    assert "no_zero_crossing" in vloop.flags
    assert vloop.excursion_value == pytest.approx(-5.0, abs=0.5)


def test_a_flat_loop_has_no_excursion_at_the_anchor():
    t = grid()
    ods = discharge_ods(t=t, loops=[(0.091, 0.0, {"voltage": 0.02 * RNG.standard_normal(t.size)})])

    vloop = discharge_timing(ods).vloop

    assert not vloop.found and "no_oh_excursion" in vloop.flags


def test_a_vacuum_product_without_pf_active_is_still_a_record():
    t = grid()
    ods = ODS(consistency_check=False)
    ods["magnetics.ids_properties.homogeneous_time"] = 1
    ods["magnetics.time"] = t
    ods["magnetics.flux_loop.0.position.0.r"] = 0.091
    ods["magnetics.flux_loop.0.position.0.z"] = 0.0
    ods["magnetics.flux_loop.0.flux.data"] = np.zeros_like(t)

    timing = discharge_timing(ods)

    assert isinstance(timing, DischargeTiming)
    assert timing.pf_onsets == ()
    assert "no_pf_active" in timing.flags and "oh_coil_not_found" in timing.flags
    assert isinstance(timing.vloop, LoopVoltageEvent) and not timing.vloop.found


def test_coil_onsets_and_the_event_are_reusable_pieces():
    t = grid()
    ods = discharge_ods(t=t)
    onsets = coil_onsets(ods)
    assert all(isinstance(c, CoilOnset) for c in onsets)
    event = loop_voltage_event(ods, anchor=onsets[0])
    assert event.found and event.anchor_time == onsets[0].time
    assert loop_voltage_event(ods, anchor=None).flags[-1] == "no_oh_anchor"


def test_a_misconfigured_policy_is_refused_at_load(tmp_path):
    import yaml

    from vaft.machine_mapping.utils import _resolve_info_file_path, load_yaml

    document = load_yaml(_resolve_info_file_path(None))
    bad = dict(document)
    bad["discharge_timing"] = {**document["discharge_timing"], "ohmic_coil": None}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(VestConfigurationError, match="ohmic_coil"):
        resolve_discharge_timing_policy(info_file=str(path))


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def test_the_composer_reads_only_and_stays_off_the_plot_layer():
    text = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    assert [m for m in imported if m.startswith("vaft.validation")] == ["vaft.validation.imas"]
    assert not [m for m in imported if m.startswith(("vaft.plot", "vaft.database"))]
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "ods"
    ]
    assert writes == []
    assert "set_path" not in text


def test_reading_a_product_materialises_nothing():
    t = grid()
    ods = discharge_ods(t=t, coils={"PF1": coil_current(t)}, loops=[(0.091, 0.0, {"flux": np.zeros_like(t)})])
    before = sorted(map(str, ods.flat().keys()))

    discharge_timing(ods)

    assert sorted(map(str, ods.flat().keys())) == before
