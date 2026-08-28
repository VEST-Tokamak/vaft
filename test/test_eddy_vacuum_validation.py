"""Eddy validation by synthetic vacuum magnetics (issue #139).

The four physics tests the issue specifies are built on synthetic ODSs whose
measured signals are *constructed* from the forward model, so they assert the
physics rather than a golden image: if the response projection, the units, or
the coil/eddy split were wrong, the constructed residual would not vanish.
"""

from __future__ import annotations

import copy
import json
import math
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.impa import IMPA_POLOIDAL_ANGLE
from vaft.machine_mapping.magnetics import (
    INBOARD_FLUX_LOOP_MAX_R,
    INBOARD_PROBE_MAX_R,
    OUTBOARD_FLUX_LOOP_MIN_R,
    OUTBOARD_PROBE_MIN_R,
    POLOIDAL_ANGLE,
    SIDE_PROBE_MIN_ABS_Z,
)
from vaft.omas.process_wrapper import (
    compute_point_response_ods,
    compute_point_vacuum_fields_ods,
)
from vaft.omas.vacuum_magnetics import (
    B_FIELD_POL_PROBE,
    FLUX_LOOP,
    VacuumMagneticsError,
    eddy_improvement,
    plasma_onset_time,
    probe_family,
    residual_onset,
    residual_rms,
    select_vacuum_channels,
    synthetic_vacuum_magnetics,
    vacuum_magnetics_metrics,
)

PLASMA_ONSET = 0.30
N_TIME = 240
# One inboard and one outboard channel of each kind: the coverage the validation
# contract requires, at the smallest size that still exercises both families.
PROBES = (("inboard_probe", 0.05, 0.1), ("outboard_probe", 0.85, 0.1))
LOOPS = (("inboard_loop", 0.10, 0.3), ("outboard_loop", 0.70, 0.3))


def _synthetic_ods(*, eddy_scale: float = 1.0, plasma_amplitude: float = 0.0):
    """An eddy-stage ODS whose measurements are exactly coil + eddy (+ plasma).

    Two rectangular PF coils and two passive loops drive real Green's-function
    responses at two B probes and two flux loops; every measured signal is then
    synthesized from those same responses, so the pre-plasma residual is zero by
    construction unless the forward model reads the ODS differently than this
    builder wrote it.
    """
    ods = ODS(consistency_check=False)
    time = np.linspace(0.27, 0.34, N_TIME)

    coil_currents = np.array(
        [1.0e4 * np.sin(np.linspace(0.0, 2.0, N_TIME)), -6.0e3 * np.linspace(0.0, 1.0, N_TIME)]
    )
    for index, (r, z) in enumerate(((0.15, 0.6), (0.9, -0.4))):
        base = f"pf_active.coil.{index}"
        ods[f"{base}.name"] = f"PF{index}"
        ods[f"{base}.element.0.turns_with_sign"] = 10.0
        ods[f"{base}.element.0.geometry.rectangle.r"] = r
        ods[f"{base}.element.0.geometry.rectangle.z"] = z
        ods[f"{base}.current.data"] = coil_currents[index]
        ods[f"{base}.current.time"] = time
    ods["pf_active.time"] = time

    loop_currents = eddy_scale * np.array(
        [-2.0e3 * np.gradient(coil_currents[0]) / np.gradient(time).mean() * 1e-4,
         1.5e3 * np.cos(np.linspace(0.0, 3.0, N_TIME))]
    )
    for index, (r, z) in enumerate(((0.2, 0.9), (0.95, 0.0))):
        base = f"pf_passive.loop.{index}"
        ods[f"{base}.name"] = f"passive{index}"
        ods[f"{base}.element.0.geometry.geometry_type"] = 2
        ods[f"{base}.element.0.geometry.rectangle.r"] = r
        ods[f"{base}.element.0.geometry.rectangle.z"] = z
        ods[f"{base}.current"] = loop_currents[index]
    ods["pf_passive.time"] = time

    positions = [(r, z) for _, r, z in PROBES] + [(r, z) for _, r, z in LOOPS]
    psi, b_z, b_r = compute_point_response_ods(ods, [[r, z] for r, z in positions])
    direction_r, direction_z = math.cos(POLOIDAL_ANGLE), math.sin(POLOIDAL_ANGLE)

    # A plasma-like contribution switched on at PLASMA_ONSET, so the residual has
    # something physical to find.
    plasma_shape = plasma_amplitude * np.where(
        time < PLASMA_ONSET, 0.0, 1.0 - np.exp(-(time - PLASMA_ONSET) / 0.004)
    )
    ip = 8.0e4 * plasma_shape if plasma_amplitude else np.zeros_like(time)
    ods["magnetics.ip.0.data"] = ip + 1.0e2 * np.sin(np.linspace(0, 40, N_TIME))
    ods["magnetics.ip.0.time"] = time
    ods["magnetics.time"] = time

    expected: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for index, (name, r, z) in enumerate(PROBES):
        response = b_r[index] * direction_r + b_z[index] * direction_z
        coil = response[:2] @ coil_currents
        eddy = response[2:4] @ loop_currents
        base = f"magnetics.{B_FIELD_POL_PROBE}.{index}"
        ods[f"{base}.name"] = name
        ods[f"{base}.position.r"] = r
        ods[f"{base}.position.z"] = z
        ods[f"{base}.poloidal_angle"] = POLOIDAL_ANGLE
        ods[f"{base}.field.data"] = coil + eddy + 1.0e-3 * plasma_shape
        ods[f"{base}.field.time"] = time
        expected[(B_FIELD_POL_PROBE, index)] = {"coil": coil, "coil_eddy": coil + eddy}
    for offset, (name, r, z) in enumerate(LOOPS):
        index = len(PROBES) + offset
        coil = psi[index][:2] @ coil_currents
        eddy = psi[index][2:4] @ loop_currents
        base = f"magnetics.{FLUX_LOOP}.{offset}"
        ods[f"{base}.name"] = name
        ods[f"{base}.position.0.r"] = r
        ods[f"{base}.position.0.z"] = z
        ods[f"{base}.flux.data"] = coil + eddy + 5.0e-3 * plasma_shape
        ods[f"{base}.flux.time"] = time
        expected[(FLUX_LOOP, offset)] = {"coil": coil, "coil_eddy": coil + eddy}
    return ods, time, expected


@pytest.fixture(scope="module")
def vacuum_ods():
    return _synthetic_ods()


@pytest.fixture(scope="module")
def plasma_ods():
    return _synthetic_ods(plasma_amplitude=1.0)


def _pre_plasma(time):
    return np.asarray(time) < PLASMA_ONSET


# --- the four physics checks issue #139 specifies --------------------------

def test_coil_plus_eddy_measurement_gives_a_near_zero_pre_plasma_residual(vacuum_ods):
    ods, time, _ = vacuum_ods
    channels = synthetic_vacuum_magnetics(ods)
    window = _pre_plasma(channels[0].time)
    assert len(channels) == 4
    for channel in channels:
        scale = residual_rms(channel.measured, window)
        assert residual_rms(channel.residual, window) < 1e-9 * max(scale, 1.0), channel.name


def test_a_synthetic_plasma_puts_the_residual_onset_at_the_injected_onset(plasma_ods):
    ods, _time, _ = plasma_ods
    channels = synthetic_vacuum_magnetics(ods)
    for channel in channels:
        window = channel.time < PLASMA_ONSET
        onset = residual_onset(channel.time, channel.residual, window)
        assert np.isfinite(onset), channel.name
        # Within a couple of samples of where the plasma term was switched on.
        assert abs(onset - PLASMA_ONSET) < 3.0 * float(np.diff(channel.time).mean())


def test_coil_plus_eddy_beats_coil_only_when_the_eddy_term_is_nonzero(plasma_ods):
    ods, _time, _ = plasma_ods
    channels = synthetic_vacuum_magnetics(ods)
    for channel in channels:
        window = channel.time < PLASMA_ONSET
        assert residual_rms(channel.residual, window) < residual_rms(
            channel.coil_residual, window
        ), channel.name
        assert eddy_improvement(channel.coil_residual, channel.residual, window) > 0.99


def test_each_forward_model_preserves_its_own_physical_quantity_and_unit(vacuum_ods):
    ods, _time, expected = vacuum_ods
    channels = {(c.kind, c.index): c for c in synthetic_vacuum_magnetics(ods)}

    for key, reference in expected.items():
        channel = channels[key]
        assert np.allclose(channel.coil, reference["coil"], rtol=1e-10, atol=0.0)
        assert np.allclose(channel.coil_eddy, reference["coil_eddy"], rtol=1e-10, atol=0.0)

    # Flux loops carry Wb and B probes carry T; a stray 2*pi on the flux side or
    # a Br/Bz mix-up on the probe side would break the exact agreement above.
    assert {c.unit for k, c in channels.items() if k[0] == FLUX_LOOP} == {"Wb"}
    assert {c.unit for k, c in channels.items() if k[0] == B_FIELD_POL_PROBE} == {"T"}
    flux = channels[(FLUX_LOOP, 0)]
    assert not np.allclose(flux.coil_eddy, expected[(FLUX_LOOP, 0)]["coil_eddy"] / (2 * math.pi))


# --- conventions -----------------------------------------------------------

def test_stored_poloidal_angle_declares_the_plus_bz_the_probes_measure():
    # Issue #169: the stored angle now *is* the measured direction, so the
    # IMAS (cos, sin) projection of it is +Bz and consumers may read it.
    assert POLOIDAL_ANGLE == pytest.approx(math.pi / 2)
    assert math.cos(POLOIDAL_ANGLE) == pytest.approx(0.0, abs=1e-12)
    assert math.sin(POLOIDAL_ANGLE) == pytest.approx(1.0)


def test_packaged_reference_odss_carry_the_corrected_angle():
    # The packaged references were relabelled with the constant; had they not
    # been, freshly generated ODSs would disagree with them.
    path = files("vaft.data.omas") / "39915.json"
    if not path.is_file():
        pytest.skip("packaged reference ODS is not installed")
    with path.open("r", encoding="utf-8") as handle:
        probes = json.load(handle)["magnetics"]["b_field_pol_probe"]
    assert probes
    assert {probe["poloidal_angle"] for probe in probes} == {POLOIDAL_ANGLE}


def test_impa_bz_sensors_share_the_plus_bz_orientation():
    # The IMPA Bz sensors measure the same quantity, so their nominal angle --
    # the base a measured crosstalk misalignment is offset from -- matches.
    assert IMPA_POLOIDAL_ANGLE == pytest.approx(POLOIDAL_ANGLE)


def test_probe_projection_follows_the_stored_angle_per_channel(vacuum_ods):
    # A channel mounted differently from the rest must project differently:
    # the forward model reads each probe's own angle, not one global constant.
    ods, _time, _ = vacuum_ods
    rotated = copy.deepcopy(ods)
    rotated[f"magnetics.{B_FIELD_POL_PROBE}.0.poloidal_angle"] = 0.0

    base = {c.index: c for c in synthetic_vacuum_magnetics(ods) if c.kind == B_FIELD_POL_PROBE}
    turned = {c.index: c for c in synthetic_vacuum_magnetics(rotated) if c.kind == B_FIELD_POL_PROBE}

    assert not np.allclose(base[0].coil_eddy, turned[0].coil_eddy)
    assert np.allclose(base[1].coil_eddy, turned[1].coil_eddy)


def test_vacuum_wrapper_returns_br_and_bz_the_way_it_names_them(vacuum_ods):
    # Regression: compute_point_response_ods returns (Psi, Bz, Br), and this
    # wrapper used to unpack it as (psi, br, bz).
    ods, _time, _ = vacuum_ods
    point = [(0.6, 0.05)]
    _psi_resp, bz_resp, br_resp = compute_point_response_ods(ods, point)
    _time_out, _psi, br, bz = compute_point_vacuum_fields_ods(ods, point, mode="pf_active")

    coil = np.array([ods[f"pf_active.coil.{i}.current.data"] for i in range(2)])
    assert np.allclose(br[:, 0], br_resp[0][:2] @ coil)
    assert np.allclose(bz[:, 0], bz_resp[0][:2] @ coil)
    assert not np.allclose(br[:, 0], bz[:, 0])


def test_channel_families_follow_the_efit_submitted_boundaries():
    assert probe_family(B_FIELD_POL_PROBE, INBOARD_PROBE_MAX_R - 0.01, 0.0) == "inboard"
    assert probe_family(B_FIELD_POL_PROBE, OUTBOARD_PROBE_MIN_R + 0.01, 0.0) == "outboard"
    assert probe_family(B_FIELD_POL_PROBE, 0.4, SIDE_PROBE_MIN_ABS_Z + 0.01) == "side"
    assert probe_family(FLUX_LOOP, INBOARD_FLUX_LOOP_MAX_R - 0.01, 0.0) == "inboard_flux_loop"
    assert probe_family(FLUX_LOOP, OUTBOARD_FLUX_LOOP_MIN_R + 0.01, 0.0) == "outboard_flux_loop"


# --- selection and error reporting ------------------------------------------

def test_flatlined_and_short_channels_are_excluded(vacuum_ods):
    ods, time, _ = vacuum_ods
    ods = ods.copy()
    ods[f"magnetics.{B_FIELD_POL_PROBE}.0.field.data"] = np.zeros_like(time)
    selected = select_vacuum_channels(ods)
    assert (B_FIELD_POL_PROBE, 0) not in {(row["kind"], row["index"]) for row in selected}
    assert selected, "the remaining live channels must still be selected"


def test_explicit_channel_selection_reports_what_is_missing(vacuum_ods):
    ods, _time, _ = vacuum_ods
    assert len(select_vacuum_channels(ods, channels=[(FLUX_LOOP, 1)])) == 1
    with pytest.raises(VacuumMagneticsError, match=r"b_field_pol_probe\[9\]"):
        select_vacuum_channels(ods, channels=[(B_FIELD_POL_PROBE, 9)])


def test_an_ods_without_the_eddy_solve_is_reported(vacuum_ods):
    ods, _time, _ = vacuum_ods
    without_passive = ODS(consistency_check=False)
    without_passive["pf_active"] = ods["pf_active"]
    with pytest.raises(VacuumMagneticsError, match="no pf_passive loops"):
        synthetic_vacuum_magnetics(without_passive)


# --- metrics ---------------------------------------------------------------

def test_metrics_record_every_quantity_the_stage_owes(plasma_ods):
    ods, _time, _ = plasma_ods
    channels = synthetic_vacuum_magnetics(ods)
    metrics = vacuum_magnetics_metrics(
        channels,
        plasma_onset=PLASMA_ONSET,
        plasma_current=(ods["magnetics.ip.0.time"], ods["magnetics.ip.0.data"]),
    )
    assert set(metrics["channels"][0]) >= {
        "residual_rms_coil",
        "residual_rms_coil_eddy",
        "improvement",
        "residual_onset",
        "onset_delta",
        "family",
        "unit",
    }
    assert set(metrics["summary"]) >= {
        "median_improvement",
        "min_improvement",
        "onset_coherence",
        "median_onset_delta",
        "channels_without_onset",
    }
    assert metrics["summary"]["channels_without_onset"] == 0
    # A single injected onset means the channels must agree on when it happened.
    assert metrics["summary"]["onset_coherence"] < 3.0 * float(
        np.diff(channels[0].time).mean()
    )
    assert set(metrics["families"]) == {c.family for c in channels}


def test_metrics_need_a_pre_plasma_window(plasma_ods):
    ods, time, _ = plasma_ods
    channels = synthetic_vacuum_magnetics(ods)
    with pytest.raises(VacuumMagneticsError, match="no pre-plasma samples"):
        vacuum_magnetics_metrics(channels, plasma_onset=float(time[0]))


def test_plasma_onset_is_reported_when_there_is_no_plasma_current():
    empty = ODS(consistency_check=False)
    with pytest.raises(VacuumMagneticsError, match="plasma-current onset"):
        plasma_onset_time(empty)


# --- the stage as the workflow runs it ---------------------------------------

def test_eddy_stage_writes_both_figures_and_a_metrics_block(tmp_path, plasma_ods):
    from vaft.validation import render_stage_plots, stage_plot_filenames

    ods, _time, _ = plasma_ods
    directory = tmp_path / "plot"
    manifest = render_stage_plots("eddy", ods, directory)

    assert {path.name for path in directory.iterdir()} == set(
        stage_plot_filenames("eddy", required_only=True)
    )
    assert all(row["status"] == "generated" for row in manifest["plots"])
    assert manifest["metrics"]["summary"]["channel_count"] == 4


# --- the real shot -----------------------------------------------------------

def test_packaged_shot_reproduces_its_measured_vacuum_magnetics():
    """The headline physics, so a forward-model regression fails loudly.

    Shot 39915: adding the eddy response must remove most of the pre-plasma
    residual on every selected channel, and the residual must emerge close to
    the measured plasma-current onset.
    """
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    channels = synthetic_vacuum_magnetics(ods)
    metrics = vacuum_magnetics_metrics(
        channels,
        plasma_onset=plasma_onset_time(ods),
        plasma_current=(ods["magnetics.ip.0.time"], ods["magnetics.ip.0.data"]),
    )
    families = {channel.family for channel in channels}
    assert {"inboard", "outboard"} <= families
    assert any(family.endswith("flux_loop") for family in families)

    assert metrics["summary"]["min_improvement"] > 0.4
    assert metrics["summary"]["median_improvement"] > 0.7
    assert abs(metrics["summary"]["median_onset_delta"]) < 3.0e-3
    assert metrics["summary"]["onset_coherence"] < 1.0e-2


# --- review regression ------------------------------------------------------

def test_the_residual_band_and_the_onset_markers_use_the_same_sigma(plasma_ods):
    """`sigma` reaches the metrics, not only the drawn band.

    The figure draws its noise band at `sigma`, and annotates each panel with a
    Delta-t computed from the metrics. If `sigma` stopped at the band the two
    would describe different thresholds on the same axes.
    """
    import vaft.omas as vomas
    from vaft.omas._plot_recipes import _vacuum_channels

    ods, _time, _ = plasma_ods
    loose = _vacuum_channels(ods, {"sigma": 2.0})[1]
    tight = _vacuum_channels(ods, {"sigma": 40.0})[1]

    loose_onsets = [row["residual_onset"] for row in loose["channels"]]
    tight_onsets = [row["residual_onset"] for row in tight["channels"]]
    # A far stricter threshold must move, or lose, the detected onsets.
    assert loose_onsets != tight_onsets

    figure, axes = vomas.plot_magnetics_overview_plasma_residual(ods, sigma=2.0)
    titles = [ax.get_title() for ax in axes.ravel() if ax.get_visible()]
    assert any("Δt" in title for title in titles)
    labels = {
        text.get_text()
        for ax in axes.ravel()
        if ax.get_legend() is not None
        for text in ax.get_legend().get_texts()
    }
    assert any("±2σ" in label for label in labels)
