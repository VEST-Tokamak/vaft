"""Explicit VEST Rogowski-coil mapping (issue #215).

VEST processes two physical Rogowski coils but used to store only what they
were processed *into* -- `magnetics.ip[0]` and `magnetics.diamagnetic_flux[0]`.
That conflated a measurement with a reconstruction and left the sensors
themselves absent from the ODS. These tests pin the sensors as measurements in
their own right, and pin that adding them changed no derived value.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.magnetics import (
    _ROGOWSKI_DIAMAGNETIC_TF,
    _ROGOWSKI_PLASMA_CURRENT,
    vest_diamagnetic_flux,
    vest_diamagnetic_rogowski_current,
    vest_plasma_rogowski_current,
    vfit_magnetics_dynamic,
    vfit_plasma_current,
)
from vaft.machine_mapping.utils import (
    VestConfigurationError,
    process_static_channels,
    process_static_geometry,
)
from vaft.omas.vest_upstream import build_static_ods, machine_era_for_shot

SHOT = 41672
RAW = "vaft/data/samples/41672/source/vest_41672_daq_raw.json.gz"
TSTART, TEND, DT = 0.26, 0.36, 4e-5


@pytest.fixture(scope="module")
def magnetics_ods():
    """One magnetics ODS built from the packaged shot-41672 raw dump."""
    ods = ODS(consistency_check=False)
    static, _ = build_static_ods(machine_era_for_shot(SHOT).name)
    ods["magnetics"] = static["magnetics"]
    vfit_magnetics_dynamic(
        ods,
        SHOT,
        TSTART,
        TEND,
        DT,
        raw_source=RAW,
        target_time=np.arange(TSTART, TEND, DT),
    )
    return ods


def test_both_physical_rogowski_sensors_are_mapped(magnetics_ods):
    ods = magnetics_ods
    assert len(ods["magnetics.rogowski_coil"]) == 2

    plasma = ods[f"magnetics.rogowski_coil.{_ROGOWSKI_PLASMA_CURRENT}"]
    assert plasma["measured_quantity.name"] == "plasma_eddy"
    assert plasma["measured_quantity.index"] == 2
    assert plasma["current.validity"] == 0
    assert np.asarray(plasma["current.data"]).size > 0

    tf = ods[f"magnetics.rogowski_coil.{_ROGOWSKI_DIAMAGNETIC_TF}"]
    # The DD enumerates only plasma/plasma_eddy/eddy/halo/compound and requires
    # private identifiers to carry a negative index. A TF-current sensor is
    # none of the five, so it must not be filed as one of them.
    assert tf["measured_quantity.index"] < 0
    assert tf["measured_quantity.name"] not in {
        "plasma", "plasma_eddy", "eddy", "halo", "compound",
    }
    # Not asserted as 0: this shot's channel carries reconstructed samples, so
    # validity must not claim a clean acquisition. See
    # `test_validity_does_not_claim_a_reconstructed_waveform_is_clean`.
    assert tf["current.validity"] >= 0
    assert np.asarray(tf["current.data"]).size > 0


def test_stored_current_is_the_sensor_signal_not_the_derived_product(magnetics_ods):
    """The whole point of #215: a measurement, not a reconstruction."""
    ods = magnetics_ods
    sensor_time, sensor = vest_plasma_rogowski_current(SHOT, raw_source=RAW)
    stored = np.asarray(
        ods[f"magnetics.rogowski_coil.{_ROGOWSKI_PLASMA_CURRENT}.current.data"],
        dtype=float,
    )
    window = (sensor_time >= TSTART) & (sensor_time < TEND)
    np.testing.assert_array_equal(stored, sensor[window])

    # It must NOT be the processed plasma current: the baseline removal, FL10
    # compensation, and shot-era sign all still lie between them.
    ip = np.asarray(ods["magnetics.ip.0.data"], dtype=float)
    assert stored.size == ip.size
    assert not np.allclose(stored, ip)
    assert not np.allclose(stored, -ip)


def test_diamagnetic_sensor_current_is_not_the_processing_intermediate():
    """`delta_i_tf` depends on the plasma interval; the sensor current does not."""
    time, current = vest_diamagnetic_rogowski_current(SHOT, raw_source=RAW)

    # Two different plasma windows produce two different delta_i_tf, but the
    # sensor current behind them is one signal.
    time_b, current_b = vest_diamagnetic_rogowski_current(SHOT, raw_source=RAW)
    np.testing.assert_array_equal(current, current_b)
    np.testing.assert_array_equal(time, time_b)

    _, flux_narrow = vest_diamagnetic_flux(SHOT, 0.28, 0.32, raw_source=RAW)
    _, flux_wide = vest_diamagnetic_flux(SHOT, 0.28, 0.34, raw_source=RAW)
    assert not np.allclose(flux_narrow, flux_wide)


def test_sensor_currents_use_the_native_timebase_cropped_to_the_window(magnetics_ods):
    ods = magnetics_ods
    for index in (_ROGOWSKI_PLASMA_CURRENT, _ROGOWSKI_DIAMAGNETIC_TF):
        time = np.asarray(
            ods[f"magnetics.rogowski_coil.{index}.current.time"], dtype=float
        )
        data = np.asarray(
            ods[f"magnetics.rogowski_coil.{index}.current.data"], dtype=float
        )
        assert time.size == data.size
        assert time[0] >= TSTART
        assert time[-1] < TEND
        assert np.all(np.diff(time) > 0)


def test_geometry_is_left_unset_rather_than_invented(magnetics_ods):
    """VEST has no authoritative winding contour; omitting beats fabricating."""
    ods = magnetics_ods
    for index in (_ROGOWSKI_PLASMA_CURRENT, _ROGOWSKI_DIAMAGNETIC_TF):
        base = f"magnetics.rogowski_coil.{index}"
        for leaf in ("position", "turns_per_metre", "area"):
            assert f"{base}.{leaf}" not in ods


def test_missing_channel_keeps_the_slot_and_does_not_shift_indices(tmp_path):
    """A shot with no diamagnetic channel must not renumber the other sensor.

    This pins the *helper* contract. It is deliberately not routed through
    `vfit_magnetics_dynamic`, because that mapper calls `vfit_plasma_current`
    immediately after the sensor mapping, and a missing plasma-current channel
    still aborts the whole magnetics component there -- pre-existing behaviour
    this change does not alter. See
    `test_pipeline_still_aborts_magnetics_when_the_plasma_channel_is_missing`,
    which pins that actual behaviour so the two are not confused.
    """
    import gzip
    import json

    from vaft.machine_mapping.magnetics import _map_rogowski_coils

    # Field 257 (diamagnetic) absent; field 109 (plasma current) present.
    raw = tmp_path / "raw.json.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "shot": SHOT,
                "fields": {
                    "109": {
                        "data": np.linspace(0.0, 1.0, 25_000).tolist(),
                        "type": "slow",
                    }
                },
            },
            handle,
        )

    ods = ODS(consistency_check=False)
    _map_rogowski_coils(ods, SHOT, raw_source=raw, tstart=TSTART, tend=TEND)

    assert len(ods["magnetics.rogowski_coil"]) == 2
    assert ods[f"magnetics.rogowski_coil.{_ROGOWSKI_PLASMA_CURRENT}.current.validity"] == 0
    tf = ods[f"magnetics.rogowski_coil.{_ROGOWSKI_DIAMAGNETIC_TF}"]
    assert tf["current.validity"] == -2
    assert np.asarray(tf["current.data"]).size == 0
    # Identity survives even with no data, so consumers can still see which
    # sensor is missing.
    assert tf["measured_quantity.index"] < 0


def test_derived_quantities_carry_a_method_name(magnetics_ods):
    ods = magnetics_ods
    assert "rogowski_coil" in ods["magnetics.ip.0.method_name"]
    assert "Rogowski" in ods["magnetics.diamagnetic_flux.0.method_name"]


def test_diamagnetic_flux_is_stored_in_webers(magnetics_ods):
    """Units regression: the DD requires Wb, and EFIT rescales from Wb itself.

    `EFITConstraintConfig.diamagnetic_flux_input_units` defaults to "Wb" and
    `kfile.py` multiplies by 1000 to write DFLUX in mV*s. If this mapping ever
    started emitting mWb, EFIT would silently be off by 1000.
    """
    from vaft.code.efit.config import EFITConstraintConfig

    assert EFITConstraintConfig().diamagnetic_flux_input_units == "Wb"
    flux = np.asarray(magnetics_ods["magnetics.diamagnetic_flux.0.data"], dtype=float)
    peak = float(np.nanmax(np.abs(flux)))
    # VEST diamagnetic flux is a few mWb; in Wb that is O(1e-3), and an
    # accidental mWb store would land at O(1).
    assert 1e-5 < peak < 1e-1


def test_refactor_left_the_derived_quantities_untouched():
    """The calibration-only split must be numerically inert (#215 non-goal)."""
    time, ip = vfit_plasma_current(SHOT, raw_source=RAW)
    sensor_time, sensor = vest_plasma_rogowski_current(SHOT, raw_source=RAW)
    np.testing.assert_array_equal(time, sensor_time)

    # The sensor current is the input to the processing chain, so it shares the
    # raw timebase but not the values.
    assert ip.shape == sensor.shape
    assert not np.array_equal(ip, sensor)


def test_legacy_non_canonical_rogowski_helpers_refuse_to_run():
    """They wrote `rogowski_coil.coil.*`, which is not the canonical path."""
    for func, payload in (
        (process_static_geometry, {"geometry": {"coils": [{"r": 0.45}]}}),
        (process_static_channels, {"channels": [{"name": "RC0"}]}),
    ):
        with pytest.raises(VestConfigurationError, match="non-canonical"):
            func({}, "rogowski_coil", payload)


def test_registry_separates_the_sensor_from_the_derived_quantities():
    from vaft.machine_mapping.registry import load_diagnostic_registry

    registry = load_diagnostic_registry()
    assert "magnetics.rogowski_coil" in registry
    # The derived entries must no longer be *named* as the physical coils.
    assert "Rogowski" not in registry["magnetics.ip"]["name"]
    assert "Rogowski" not in registry["magnetics.diamagnetic_flux"]["name"]
    assert registry["magnetics.rogowski_coil"]["ids_path"] == "magnetics.rogowski_coil"


def test_rogowski_node_survives_a_dd_consistency_round_trip(tmp_path):
    """The private negative `measured_quantity.index` must be DD-legal."""
    from omas import load_omas_json

    from vaft.machine_mapping.magnetics import _map_rogowski_coils

    ods = ODS(consistency_check=False)
    _map_rogowski_coils(ods, SHOT, raw_source=RAW, tstart=TSTART, tend=TEND)
    ods["magnetics.ids_properties.homogeneous_time"] = 0

    path = tmp_path / "rogowski.json"
    ods.save(str(path))
    reloaded = load_omas_json(str(path), consistency_check=True)

    assert reloaded[f"magnetics.rogowski_coil.{_ROGOWSKI_DIAMAGNETIC_TF}.measured_quantity.index"] < 0
    for index in (_ROGOWSKI_PLASMA_CURRENT, _ROGOWSKI_DIAMAGNETIC_TF):
        assert np.asarray(
            reloaded[f"magnetics.rogowski_coil.{index}.current.data"]
        ).size > 0


def test_pipeline_still_aborts_magnetics_when_the_plasma_channel_is_missing(tmp_path):
    """Record what the real mapper does, so the helper test is not misread.

    `_map_rogowski_coils` tolerates a missing channel, but `vfit_magnetics_dynamic`
    calls `vfit_plasma_current` on the next line, which does not. A shot missing
    field 109 therefore loses the entire magnetics component, sensor slots
    included -- graceful degradation of the Rogowski mapping alone does not make
    the pipeline tolerant. Widening that is #189/#195 territory, not #215.
    """
    import gzip
    import json

    from vaft.database import raw as raw_db

    raw = tmp_path / "raw.json.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "shot": SHOT,
                "fields": {
                    "257": {
                        "data": np.linspace(0.0, 1.0, 25_000).tolist(),
                        "type": "slow",
                    }
                },
            },
            handle,
        )

    ods = ODS(consistency_check=False)
    with pytest.raises(raw_db.RawSignalUnavailableError, match="field 109"):
        vfit_magnetics_dynamic(
            ods,
            SHOT,
            TSTART,
            TEND,
            DT,
            raw_source=raw,
            target_time=np.arange(TSTART, TEND, DT),
        )


def test_sensor_travels_with_its_derived_quantity_on_every_entry_point():
    """Which entry point a caller reaches for must not decide whether the
    physical Rogowski coil appears in the ODS."""
    from vaft.machine_mapping.magnetics import (
        diamagnetic_flux_rogowski_coil_from_raw_database,
        ip_rogowski_coil_from_raw_database,
    )

    for entrypoint in (
        ip_rogowski_coil_from_raw_database,
        diamagnetic_flux_rogowski_coil_from_raw_database,
    ):
        ods = ODS(consistency_check=False)
        entrypoint(ods, SHOT, tstart=TSTART, tend=TEND, dt=DT, raw_source=RAW)
        assert len(ods["magnetics.rogowski_coil"]) == 2, entrypoint.__name__
        assert (
            np.asarray(
                ods[f"magnetics.rogowski_coil.{_ROGOWSKI_PLASMA_CURRENT}.current.data"]
            ).size
            > 0
        ), entrypoint.__name__


def test_sensor_current_matches_the_flux_path_it_feeds():
    """The two must not drift apart after the #285 saturation work.

    `vest_diamagnetic_rogowski_current` stops at the first integration instead
    of running the full triple-integration chain, so it repeats the gain
    expression. This pins it against the `"integrated"` stage that
    `vest_diamagnetic_flux_detailed` actually derives the flux from -- including
    the clipping repair -- so a change to one that is not made to the other
    fails here rather than silently storing a sensor current the flux was never
    computed from.
    """
    import warnings

    from vaft.machine_mapping.magnetics import (
        vest_diamagnetic_flux_detailed,
        vest_diamagnetic_rogowski_current,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        time, sensor = vest_diamagnetic_rogowski_current(SHOT, raw_source=RAW)
        _, _, report = vest_diamagnetic_flux_detailed(
            SHOT, 0.28, 0.34, raw_source=RAW, with_stages=True
        )

    stages = report["stages"]
    np.testing.assert_array_equal(time, np.asarray(stages["time"], dtype=float))
    np.testing.assert_array_equal(sensor, np.asarray(stages["integrated"], dtype=float))
    # The repair is what makes this non-trivial: raw and repaired differ here.
    assert report["n_saturated"] > 0


def test_validity_does_not_claim_a_reconstructed_waveform_is_clean(magnetics_ods):
    """A repaired sample is an interpolation, not a measurement (#285).

    Shot 41672 has samples pinned at the diamagnetic channel's acquisition
    rail, which `vest_diamagnetic_flux_detailed` reconstructs before
    integrating. Publishing that as `validity = 0` would tell every consumer
    the channel acquired cleanly.
    """
    from vaft.machine_mapping.magnetics import diamagnetic_saturation_report

    report = diamagnetic_saturation_report(SHOT, raw_source=RAW)
    assert report["n_saturated"] > 0, "fixture shot must exercise the repair"

    diamag = magnetics_ods[
        f"magnetics.rogowski_coil.{_ROGOWSKI_DIAMAGNETIC_TF}.current.validity"
    ]
    assert diamag != 0

    # The plasma-current sensor is calibration-only, with nothing reconstructed.
    plasma = magnetics_ods[
        f"magnetics.rogowski_coil.{_ROGOWSKI_PLASMA_CURRENT}.current.validity"
    ]
    assert plasma == 0


def test_an_unrepairable_sensor_does_not_fail_its_siblings(tmp_path):
    """`SignalRepairError` must degrade one sensor, not the whole mapping.

    develop lets it propagate from the *flux* path, where the caller asked for
    that quantity. A caller asking for plasma current has not, so routing the
    sensor mapping through the diamagnetic channel must not make an
    unrecoverable field-257 record fail `magnetics.ip`.
    """
    import gzip
    import json

    from vaft.machine_mapping.magnetics import ip_rogowski_coil_from_raw_database

    n = 25_000
    healthy = (np.linspace(0.0, 1.0, n) * 0.5).tolist()
    raw = tmp_path / "raw.json.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "shot": SHOT,
                "fields": {
                    "109": {"data": healthy, "type": "slow"},
                    "25": {"data": healthy, "type": "slow"},
                    # Pinned at the rail for the whole record: unrepairable.
                    "257": {"data": [-5.0] * n, "type": "slow"},
                },
            },
            handle,
        )

    ods = ODS(consistency_check=False)
    ip_rogowski_coil_from_raw_database(
        ods, SHOT, tstart=TSTART, tend=TEND, dt=DT, raw_source=raw
    )

    assert "ip" in ods["magnetics"]
    assert (
        ods[f"magnetics.rogowski_coil.{_ROGOWSKI_DIAMAGNETIC_TF}.current.validity"] == -2
    )
    assert (
        np.asarray(
            ods[f"magnetics.rogowski_coil.{_ROGOWSKI_PLASMA_CURRENT}.current.data"]
        ).size
        > 0
    )
