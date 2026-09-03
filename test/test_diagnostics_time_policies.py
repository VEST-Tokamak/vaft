"""Per-component diagnostics time windows (issue #244).

The processed diagnostics product used to force every mapped signal onto one
grid, ``0.26 <= t < 0.36 s``. That window is right for equilibrium magnetics
and wrong for signals whose physics spans the discharge: VEST's slow DAQ
records TF and barometry over the full 0-1 s sequence, so the single window
threw away the TF ramp-up/ramp-down and the entire pre-breakdown prefill
history. These tests pin the replacement -- policy per component, resolved
from configuration -- and that both coverages coexist in one ODS.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import (
    LIMITER_SHUNT_CHANNELS,
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    vest_equilibrium_magnetics_channel_definitions,
)
from vaft.machine_mapping.tf import vfit_tf_dynamic
from vaft.machine_mapping.utils import (
    DiagnosticsTimePolicy,
    DiagnosticsTimePolicyTable,
    VestConfigurationError,
    build_window_time_axis,
    resolve_diagnostics_time_policies,
)
from vaft.omas.vest_upstream import (
    _policy_time,
    build_diagnostics_ods,
    build_static_ods,
    machine_era_for_shot,
    write_stage_product,
)


SHOT = 43017
SLOW_DT = 4e-5
FULL_RECORD_SAMPLES = 25_000


def test_configured_policy_splits_analysis_from_full_discharge():
    policies = resolve_diagnostics_time_policies()

    for component in ("magnetics", "pf_active", "spectrometer_uv", "impa"):
        assert policies[component].name == "analysis"
        assert (policies[component].tstart, policies[component].tend) == (0.26, 0.36)
    # ec_power has no mapper yet (#165); registering it now means the EC
    # mapping inherits the full-discharge window rather than the short one.
    for component in ("tf", "barometry", "ec_power"):
        assert policies[component].name == "full_discharge"
        assert (policies[component].tstart, policies[component].tend) == (0.0, 1.0)


def test_stage_arguments_retune_the_analysis_window_only():
    """`tstart`/`tend`/`dt` have always meant the analysis window; still do."""
    policies = resolve_diagnostics_time_policies(
        analysis_override={"tstart": 0.24, "tend": 0.34, "dt": 1e-5}
    )

    assert (policies["magnetics"].tstart, policies["magnetics"].tend) == (0.24, 0.34)
    assert policies["magnetics"].dt == 1e-5
    assert (policies["tf"].tstart, policies["tf"].tend) == (0.0, 1.0)
    assert policies["tf"].dt == SLOW_DT


def test_overrides_repoint_a_component_without_touching_the_rest():
    policies = resolve_diagnostics_time_policies(
        overrides={"components": {"spectrometer_uv": "full_discharge"}}
    )

    assert policies["spectrometer_uv"].name == "full_discharge"
    assert policies["magnetics"].name == "analysis"


def test_unknown_window_and_unregistered_component_are_configuration_errors():
    with pytest.raises(VestConfigurationError, match="unknown window"):
        resolve_diagnostics_time_policies(
            overrides={"components": {"tf": "no_such_window"}}
        )

    policies = resolve_diagnostics_time_policies()
    with pytest.raises(VestConfigurationError, match="No diagnostics time policy"):
        policies["not_a_component"]


def test_full_discharge_grid_is_half_open_at_25_khz():
    time = _policy_time(resolve_diagnostics_time_policies()["tf"])

    assert time.size == 25_000
    assert time[0] == pytest.approx(0.0)
    assert time[-1] < 1.0
    np.testing.assert_allclose(np.diff(time), SLOW_DT)


def test_window_axis_clips_to_source_instead_of_extrapolating():
    source = np.arange(0.0, 0.5, SLOW_DT)

    axis = build_window_time_axis(source, 0.0, 1.0, SLOW_DT)

    assert axis[0] == pytest.approx(0.0)
    assert axis[-1] <= source[-1]

    with pytest.raises(VestConfigurationError, match="does not overlap"):
        build_window_time_axis(np.arange(0.0, 0.1, SLOW_DT), 0.5, 1.0, SLOW_DT)


def _static_ods(tmp_path):
    static, manifest = build_static_ods(machine_era_for_shot(SHOT).name)
    path = tmp_path / "static.json.gz"
    write_stage_product(
        static, manifest, output=path, metadata=tmp_path / "static-manifest.json"
    )
    return path


@pytest.fixture(scope="module")
def full_record_diagnostics(tmp_path_factory):
    """One 25 000-sample build, shared by the tests that assert on it.

    Two tests below need exactly this product and differ only in what they
    inspect -- one reads `ods`, the other reads `manifest`. Two builds cost
    127 s where one costs 76 s, so sharing takes ~50 s off the suite.

    Module-scoped, and therefore handed out without copying: consumers must
    treat both values as read-only. That holds today -- one consumer passes
    `ods` to `write_stage_product`, which serializes it and shallow-copies the
    manifest before adding its own key, mutating neither. A consumer that does
    mutate would leak into whichever test runs next, so give it its own build
    rather than adding to this fixture.
    """
    tmp_path = tmp_path_factory.mktemp("full-record")
    raw = tmp_path / "raw.json.gz"
    # A full slow-DAQ record: 25 000 samples at 4e-5 s, spanning 0-1 s.
    _write_raw_dump(raw, FULL_RECORD_SAMPLES)
    return build_diagnostics_ods(
        shot=SHOT, raw_source=raw, static_ods=_static_ods(tmp_path)
    )


def _magnetics_field_codes() -> list[int]:
    codes = {
        int(entry["field_code"])
        for entry in vest_equilibrium_magnetics_channel_definitions()
    }
    codes |= {int(entry["field_code"]) for entry in TOROIDAL_MIRNOV_REFERENCE_CHANNELS}
    codes |= {int(entry["field_code"]) for entry in LIMITER_SHUNT_CHANNELS}
    return sorted(codes)


def _write_raw_dump(path, samples, *, include_magnetics=True):
    """Write a slow-DAQ archive of `samples` samples at 4e-5 s from t=0.

    The loader recomputes time as ``arange(n) * SLOW_DT`` from the sample
    count, so `samples` alone fixes the record's coverage.
    """
    codes = [1, 12]  # TF coil and barometry main gauge
    if include_magnetics:
        codes += _magnetics_field_codes() + [5, 25, 59, 62, 65, 109, 257]
    time = np.arange(samples) * SLOW_DT
    fields = {
        str(code): {
            "data": (np.sin(time * 10.0) * 0.01 + 0.02).tolist(),
            "type": "slow",
        }
        for code in sorted(set(codes))
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"shot": SHOT, "fields": fields}, handle)


def test_full_and_short_windows_coexist_in_one_diagnostics_ods(
    full_record_diagnostics, tmp_path
):
    """The acceptance criterion: no truncation, no unintended resampling."""
    samples = FULL_RECORD_SAMPLES
    ods, manifest = full_record_diagnostics

    tf_time = np.asarray(ods["tf.time"], dtype=float)
    pressure_time = np.asarray(
        ods["barometry.gauge.0.pressure.time"], dtype=float
    )
    magnetics_time = np.asarray(ods["magnetics.time"], dtype=float)

    for full in (tf_time, pressure_time):
        assert full.size == samples - 1
        assert full[0] == pytest.approx(0.0)
        assert full[-1] == pytest.approx(1.0 - 2 * SLOW_DT)
        np.testing.assert_allclose(np.diff(full), SLOW_DT)

    assert magnetics_time.size == 2_500
    assert magnetics_time[0] == pytest.approx(0.26)
    assert magnetics_time[-1] < 0.36

    # Each IDS keeps its own coordinate structure; a longer TF axis does not
    # make the product heterogeneous.
    assert ods["tf.ids_properties.homogeneous_time"] == 1
    assert ods["barometry.ids_properties.homogeneous_time"] == 0

    assert np.asarray(ods["tf.b_field_tor_vacuum_r.data"]).size == tf_time.size
    assert np.asarray(ods["tf.coil.0.current.data"]).size == tf_time.size
    assert np.asarray(ods["barometry.gauge.0.pressure.data"]).size == pressure_time.size

    components = manifest["time_grid"]["components"]
    assert components["tf"]["policy"] == "full_discharge"
    assert components["barometry"]["policy"] == "full_discharge"
    assert components["magnetics"]["policy"] == "analysis"
    assert manifest["time_grid"]["policies"]["full_discharge"] == {
        "tstart": 0.0,
        "tend": 1.0,
        "dt": SLOW_DT,
    }
    # A record one sample short of the nominal end is the DAQ's own half-open
    # convention, not lost coverage.
    assert manifest["time_grid"]["source_clipping"] is False

    from omas import load_omas_json

    output = tmp_path / "diagnostics.json"
    write_stage_product(
        ods, manifest, output=output, metadata=tmp_path / "diagnostics-manifest.json"
    )
    reloaded = load_omas_json(str(output), consistency_check=True)
    assert np.asarray(reloaded["tf.time"]).size == tf_time.size
    assert np.asarray(reloaded["magnetics.time"]).size == magnetics_time.size


def test_short_source_realizes_a_narrower_span_and_says_so(tmp_path):
    """A source shorter than its window is reported, never extrapolated."""
    raw = tmp_path / "raw.json.gz"
    samples = 5_000  # 0.2 s of slow DAQ, not the full 1 s
    _write_raw_dump(raw, samples, include_magnetics=False)

    ods, manifest = build_diagnostics_ods(
        shot=SHOT, raw_source=raw, static_ods=_static_ods(tmp_path)
    )

    tf_time = np.asarray(ods["tf.time"], dtype=float)
    assert tf_time.size == samples - 1
    assert tf_time[-1] < 0.2  # no extrapolation past the record

    entry = manifest["time_grid"]["components"]["tf"]
    assert entry["source_clipping"] is True
    assert entry["requested_end"] == 1.0
    assert entry["realized_end_exclusive"] == pytest.approx(0.2, abs=2 * SLOW_DT)
    assert entry["missing_samples"] == 25_000 - (samples - 1)
    assert manifest["time_grid"]["source_clipping"] is True


def test_widening_the_window_does_not_change_the_analysis_interval():
    """Extending TF coverage must add samples, not alter the existing ones.

    Both mappers filter and baseline the *whole* native record before
    resampling, so the window only selects which part of an already-processed
    waveform is stored. Values inside 0.26-0.36 s therefore have to survive the
    widening untouched -- otherwise every equilibrium reconstruction built on
    the old product would silently shift.
    """
    raw = "vaft/data/samples/41672/source/vest_41672_daq_raw.json.gz"
    shot = 41672

    short, full = {}, {}
    vfit_tf_dynamic(short, shot, 0.26, 0.36, SLOW_DT, raw_source=raw)
    vfit_tf_dynamic(full, shot, 0.0, 1.0, SLOW_DT, raw_source=raw)

    short_time = np.asarray(short["tf"]["time"], dtype=float)
    full_time = np.asarray(full["tf"]["time"], dtype=float)
    overlap = (full_time >= short_time[0] - 1e-12) & (
        full_time <= short_time[-1] + 1e-12
    )
    assert overlap.sum() == short_time.size
    assert full_time.size > short_time.size

    # The two grids are the same instants to within `arange` round-off
    # (~4e-14 s), which is what makes the interpolated values comparable.
    np.testing.assert_allclose(full_time[overlap], short_time, rtol=0.0, atol=1e-12)
    for path in ("b_field_tor_vacuum_r", "coil"):
        if path == "coil":
            short_data = np.asarray(short["tf"]["coil"][0]["current"]["data"])
            full_data = np.asarray(full["tf"]["coil"][0]["current"]["data"])
        else:
            short_data = np.asarray(short["tf"][path]["data"])
            full_data = np.asarray(full["tf"][path]["data"])
        np.testing.assert_allclose(
            full_data[overlap], short_data, rtol=1e-9, atol=0.0
        )


def test_realized_axis_is_checked_against_its_policy_not_itself():
    """Review finding: full-window axes were validated against themselves.

    TF and barometry build their axis inside the mapper, and the stage reads it
    back out of the product. Comparing that axis to the node it came from is a
    tautology, so the realized coordinate is checked against the policy that
    asked for it instead -- otherwise a mapper emitting the wrong cadence, or
    running past the window, would validate cleanly and ship.
    """
    from omas import ODS

    from vaft.omas.vest_upstream import _validate_diagnostics_time_coordinates

    full = DiagnosticsTimePolicy("full_discharge", 0.0, 1.0, SLOW_DT)
    policies = DiagnosticsTimePolicyTable(
        {"tf": full}, windows={"full_discharge": full}, default=full
    )

    def validate(axis):
        ods = ODS(consistency_check=False)
        ods["tf.time"] = axis
        return _validate_diagnostics_time_coordinates(
            ods, {"tf": axis}, policies=policies
        )

    # linspace over the wrong span: dt=6e-5 against a 4e-5 policy, running
    # 50% past the window's exclusive end.
    with pytest.raises(ValueError, match="not uniform at the full_discharge cadence"):
        validate(np.linspace(0.0, 1.5, 25_000))

    with pytest.raises(ValueError, match="at or past the .* exclusive end"):
        validate(np.arange(0.0, 1.2, SLOW_DT))

    # The correct grid, and a legitimately clipped one, must still pass.
    validate(np.arange(0.0, 1.0, SLOW_DT))
    validate(np.arange(0.0, 0.2, SLOW_DT))


def test_a_default_naming_an_unconfigured_window_is_rejected():
    """Review finding: the override used to invent the missing window.

    `analysis_override` carries the stage's explicit tstart/tend/dt -- which
    the Snakemake rule always passes -- so applying it before validating
    `default` let a typo materialize a window out of those values instead of
    failing.
    """
    for override in (
        {"tstart": 0.26, "tend": 0.36, "dt": SLOW_DT},
        {"tstart": None, "tend": None, "dt": None},
    ):
        with pytest.raises(VestConfigurationError, match="is not configured"):
            resolve_diagnostics_time_policies(
                analysis_override=override, overrides={"default": "analisys"}
            )


def test_every_mapped_component_reports_its_coverage(full_record_diagnostics):
    """Review finding: langmuir_probes was mapped but left out of the record."""
    _, manifest = full_record_diagnostics

    mapped = {
        name
        for name, status in manifest["channel_status"].items()
        if status["status"] != "unavailable"
    }
    recorded = set(manifest["time_grid"]["components"])
    # impa writes into the magnetics IDS rather than one of its own, so it has
    # no coordinate to report; every other mapped component must have one.
    assert mapped - {"impa"} <= recorded
