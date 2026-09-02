"""Eddy validation by synthetic vacuum magnetics (issue #139).

The four physics tests the issue specifies are built on synthetic ODSs whose
measured signals are *constructed* from the forward model, so they assert the
physics rather than a golden image: if the response projection, the units, or
the coil/eddy split were wrong, the constructed residual would not vanish.
"""

from __future__ import annotations

import copy
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

from vaft.formula.magnetics import probe_axis
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
    evaluation_mask,
    plasma_free_residual,
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
# Deterministic measurement noise on the plasma fixture, as a fraction of each
# channel's own pre-plasma eddy signal.
#
# ``residual_onset`` takes its threshold from the pre-plasma residual's own
# noise and reports nan when that noise is exactly zero.  A measured signal
# synthesized to cancel the forward model to the last bit leaves it nothing to
# measure: whether any rounding survives the interpolation onto the pf_active
# grid is a platform detail, not physics, so the onset tests would rest on
# float error.  This floor is small enough that the eddy term still explains
# more than 99% of the pre-plasma residual, and large enough that a stricter
# sigma has a band it can actually move the detected onset out of.
MEASUREMENT_NOISE = 0.01
NOISE_SEED = 20260829


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

    rng = np.random.default_rng(NOISE_SEED)

    positions = [(r, z) for _, r, z in PROBES] + [(r, z) for _, r, z in LOOPS]
    psi, b_z, b_r = compute_point_response_ods(ods, [[r, z] for r, z in positions])
    # DD: poloidal_angle is clockwise from +R, so the sensitive axis is
    # (cos, -sin).  The synthetic "measured" signal must be built with the same
    # projection the forward model uses, or the fixture tests the wrong sign.
    direction_r, direction_z = probe_axis(POLOIDAL_ANGLE)

    # A plasma-like contribution switched on at PLASMA_ONSET, so the residual has
    # something physical to find.
    plasma_shape = plasma_amplitude * np.where(
        time < PLASMA_ONSET, 0.0, 1.0 - np.exp(-(time - PLASMA_ONSET) / 0.004)
    )
    ip = 8.0e4 * plasma_shape if plasma_amplitude else np.zeros_like(time)
    ods["magnetics.ip.0.data"] = ip + 1.0e2 * np.sin(np.linspace(0, 40, N_TIME))
    ods["magnetics.ip.0.time"] = time
    ods["magnetics.time"] = time
    pre_plasma = time < PLASMA_ONSET

    def noise(eddy):
        # Scaled by the eddy term the channel carries, so every channel keeps
        # the same signal-to-noise and the vacuum fixture stays exact.
        if not plasma_amplitude:
            return 0.0
        return MEASUREMENT_NOISE * float(np.std(eddy[pre_plasma])) * rng.standard_normal(N_TIME)

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
        ods[f"{base}.field.data"] = coil + eddy + 1.0e-3 * plasma_shape + noise(eddy)
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
        ods[f"{base}.flux.data"] = coil + eddy + 5.0e-3 * plasma_shape + noise(eddy)
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
    # Issue #288.  Issue #169 set this to pi/2 on the reading that the IMAS
    # projection is (cos, sin).  The DD says poloidal_angle is a *clockwise*
    # angle from +R, so the projection is (cos, -sin) and +Bz is 3*pi/2.  The
    # pi/2 value declared -Bz to any DD-conformant reader; it only looked right
    # because the consumer projected with the opposite handedness too.
    assert POLOIDAL_ANGLE == pytest.approx(3 * math.pi / 2)
    assert math.cos(POLOIDAL_ANGLE) == pytest.approx(0.0, abs=1e-12)
    assert -math.sin(POLOIDAL_ANGLE) == pytest.approx(1.0)


def test_packaged_reference_odss_carry_the_corrected_angle():
    # Issue #166 moved the references into the representation-neutral sample
    # registry.  Read through the public loader so this also covers compressed
    # OMAS and native IMAS storage.
    import vaft

    ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    probes = ods["magnetics.b_field_pol_probe"]
    assert len(probes) == 76
    assert {
        float(ods[f"magnetics.b_field_pol_probe.{index}.poloidal_angle"])
        for index in range(len(probes))
    } == {
        POLOIDAL_ANGLE
    }


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
    # "usable", not merely "present": the window is now intersected with the
    # validity the diagnostics stage established (issue #189).
    with pytest.raises(VacuumMagneticsError, match="no usable pre-plasma samples"):
        vacuum_magnetics_metrics(channels, plasma_onset=float(time[0]))


def test_plasma_onset_is_reported_when_there_is_no_plasma_current():
    empty = ODS(consistency_check=False)
    with pytest.raises(VacuumMagneticsError, match="plasma-current onset"):
        plasma_onset_time(empty)


# --- the stage as the workflow runs it ---------------------------------------

def test_eddy_stage_writes_both_figures_and_a_metrics_block(tmp_path, plasma_ods):
    from vaft.database.production_qa import render_stage_plots, stage_plot_filenames

    ods, _time, _ = plasma_ods
    directory = tmp_path / "plot"
    manifest = render_stage_plots("eddy", ods, directory)

    assert {path.name for path in directory.iterdir()} == set(
        stage_plot_filenames("eddy", required_only=True)
    )
    assert all(row["status"] == "generated" for row in manifest["plots"])
    assert manifest["metrics"]["summary"]["channel_count"] == 4


# --- the real shot -----------------------------------------------------------

def test_canonical_pipeline_shot_reproduces_its_measured_vacuum_magnetics():
    """The headline physics, so a forward-model regression fails loudly.

    Shot 39915: adding the eddy response must remove most of the pre-plasma
    residual on every selected channel, and the residual must emerge close to
    the measured plasma-current onset.
    """
    import vaft

    manifest = vaft.data.sample_manifest(39915)
    sample_root = vaft.data.sample(39915, "omas").parent
    ods = vaft.omas.load(sample_root / manifest["generation"]["canonical_source"])
    # Validate all submitted geometric families without letting unrelated
    # channels with different calibration/status contracts redefine this
    # coil-plus-eddy benchmark.
    channels = synthetic_vacuum_magnetics(
        ods,
        channels=[
            (B_FIELD_POL_PROBE, 1),
            (B_FIELD_POL_PROBE, 27),
            (B_FIELD_POL_PROBE, 48),
            (FLUX_LOOP, 5),
            (FLUX_LOOP, 0),
        ],
    )
    metrics = vacuum_magnetics_metrics(
        channels,
        plasma_onset=plasma_onset_time(ods),
        plasma_current=(ods["magnetics.ip.0.time"], ods["magnetics.ip.0.data"]),
    )
    families = {channel.family for channel in channels}
    assert {"inboard", "outboard", "side"} <= families
    assert {"inboard_flux_loop", "outboard_flux_loop"} <= families

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


def test_plasma_free_residual_stacks_channels_for_a_fitter(vacuum_ods):
    """#308 minimises this vector, so its length and content must be predictable."""
    ods, _, _ = vacuum_ods
    channels = synthetic_vacuum_magnetics(ods)
    window = (float(channels[0].time[0]), PLASMA_ONSET)

    residual = plasma_free_residual(channels, window)
    expected = sum(int(np.count_nonzero(evaluation_mask(c, window))) for c in channels)
    assert residual.shape == (expected,)

    # This fixture's measurement is exactly coil + eddy before onset.
    assert np.max(np.abs(residual)) < 1e-9

    # Normalisation puts tesla and weber channels on a comparable footing
    # without changing how many samples are compared.
    assert plasma_free_residual(channels, window, normalize=True).shape == (expected,)


def test_pf_only_benchmark_runs_on_the_real_machine_geometry():
    """End-to-end on the packaged shot #190 cites as its working regression.

    This is the check the synthetic fixtures cannot make: 950 real passive
    loops, the real coupling matrices, and a real PF programme. #308 will call
    `benchmark_wall_currents` inside a fit loop, so this is also its cost.

    Before plasma onset the PF-only and routine drives coincide, so the
    residual must agree between them; that it does is what makes the PF-only
    solve a drop-in objective for the calibration.
    """
    import gzip
    import shutil
    import tempfile
    from pathlib import Path

    from omas import load_omas_json

    from vaft.data import resources
    from vaft.validation.vacuum_benchmark import benchmark_wall_currents

    try:
        source = resources.data_path("samples/39915/source/pipeline-until-efit.json.gz")
    except Exception:  # pragma: no cover - packaging-dependent
        pytest.skip("packaged 39915 pipeline sample is unavailable")
    if not Path(source).is_file():
        pytest.skip("packaged 39915 pipeline sample is repository-only")

    with gzip.open(source, "rt") as handle, tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as plain:
        shutil.copyfileobj(handle, plain)
        plain_path = plain.name
    try:
        ods = load_omas_json(plain_path, consistency_check=False)
    finally:
        Path(plain_path).unlink(missing_ok=True)

    n_loop = len(ods["pf_passive.loop"])
    assert n_loop == 950

    solved = benchmark_wall_currents(ods)
    pf_only = np.array(
        [np.asarray(solved[f"pf_passive.loop.{i}.current"], dtype=float) for i in range(n_loop)]
    )
    assert pf_only.shape[0] == n_loop
    assert np.all(np.isfinite(pf_only))

    onset = plasma_onset_time(ods)
    routine_channels = synthetic_vacuum_magnetics(ods)
    window = (float(routine_channels[0].time[0]), onset)
    pf_channels = synthetic_vacuum_magnetics(solved)

    routine_rms = float(
        np.sqrt(np.mean(plasma_free_residual(routine_channels, window, normalize=True) ** 2))
    )
    pf_rms = float(
        np.sqrt(np.mean(plasma_free_residual(pf_channels, window, normalize=True) ** 2))
    )
    # Pre-onset the two drives are the same, so the residuals must be too.
    assert pf_rms == pytest.approx(routine_rms, rel=0.05)
    # And the model does not yet explain the wall response -- which is the
    # gap #308 exists to close. If this ever falls near zero without a
    # calibration landing, the benchmark has stopped being informative.
    assert 0.01 < pf_rms < 1.0


# ---------------------------------------------------------------------------
# Wall authority: how much of a reading the wall term can explain at all
# ---------------------------------------------------------------------------

def _authority_channel(name, eddy_scale, *, kind="flux_loop", family="outboard_flux_loop", index=0):
    from vaft.omas.vacuum_magnetics import VacuumChannel

    time = np.linspace(0.0, 1.0, 400)
    rng = np.random.default_rng(index + 7)
    coil = 0.10 * np.sin(2.0 * np.pi * time)
    eddy = eddy_scale * np.cos(2.0 * np.pi * time)
    measured = coil + eddy + 1.0e-4 * rng.standard_normal(time.size)
    measured[time >= 0.5] += 0.05  # a "plasma" after onset
    return VacuumChannel(
        name=name, kind=kind, family=family, index=index, r=0.6, z=0.0, unit="Wb",
        time=time, measured=measured, coil=coil, coil_eddy=coil + eddy,
    )


def test_wall_authority_is_the_eddy_term_as_a_fraction_of_the_reading():
    from vaft.formula.statistics import rms

    channel = _authority_channel("loud", 0.05)
    mask = channel.time < 0.5
    expected = rms(channel.eddy_term[mask]) / rms(channel.measured[mask])
    assert channel.wall_authority(mask) == pytest.approx(expected)
    assert _authority_channel("deaf", 0.0).wall_authority(mask) == pytest.approx(0.0, abs=1e-12)


def test_scored_improvement_summaries_skip_channels_the_wall_cannot_reach():
    """A channel whose wall term is a few percent of its reading has an
    improvement whose sign is noise; ask the summary to leave it out and it
    must, while the unfloored summary still counts everyone."""
    from vaft.omas.vacuum_magnetics import vacuum_magnetics_metrics

    channels = (_authority_channel("a", 0.05, index=0), _authority_channel("b", 0.04, index=1), _authority_channel("deaf", 0.0005, index=2))
    every = vacuum_magnetics_metrics(channels, plasma_onset=0.5)
    scored = vacuum_magnetics_metrics(channels, plasma_onset=0.5, min_wall_authority=0.1)

    assert every["summary"]["scored"]["count"] == 3
    assert every["summary"]["scored"]["improvement"]["median"] == every["summary"]["median_improvement"]
    assert scored["summary"]["scored"]["count"] == 2
    assert scored["summary"]["scored"]["improvement"]["min"] > 0.9
    assert scored["summary"]["min_improvement"] == every["summary"]["min_improvement"]
    assert all("wall_authority" in row for row in scored["channels"])
    assert scored["channels"][2]["wall_authority"] < 0.1
    assert scored["summary"]["wall_authority"]["max"] > 0.1


def test_an_excluded_channel_still_reports_its_wall_authority():
    from vaft.omas.vacuum_magnetics import channel_residual_metrics

    channel = _authority_channel("short", 0.05)
    row = channel_residual_metrics(channel, window=(0.0, 0.5), min_samples=10_000)
    assert row["status"] == "excluded"
    assert "wall_authority" in row


def test_the_packaged_shots_inboard_flux_loops_have_little_wall_authority():
    """The physics behind the scoring floor: vessel currents flow at larger R
    and their flux nearly cancels through a small inboard loop, so on the
    packaged shot the inboard loops sit an order of magnitude below the
    outboard ones."""
    import vaft
    import vaft.omas
    from vaft.omas.vacuum_magnetics import (
        plasma_onset_time,
        synthetic_vacuum_magnetics,
        vacuum_magnetics_metrics,
    )

    # The full IMAS artifact carries solved pf_passive currents; the compact
    # wheel sample does not, and this is a statement about the machine's
    # geometry, not about any one eddy solve.
    try:
        path = vaft.data.sample(41672, "imas")
    except Exception:  # repository-only artifact
        pytest.skip("sample 41672 is not available in this checkout")
    ods = vaft.omas.load(path)
    onset = plasma_onset_time(ods)
    metrics = vacuum_magnetics_metrics(
        synthetic_vacuum_magnetics(ods, per_family=2), plasma_onset=onset, min_wall_authority=0.1
    )
    by_family = {}
    for row in metrics["channels"]:
        by_family.setdefault(row["family"], []).append(row["wall_authority"])
    # An order of magnitude apart, whichever representatives the selection picks.
    assert max(by_family["inboard_flux_loop"]) * 5.0 < min(by_family["outboard_flux_loop"])
    scored = metrics["summary"]["scored"]
    assert scored["count"] == sum(1 for row in metrics["channels"] if row["wall_authority"] >= 0.1)
    assert scored["count"] < metrics["summary"]["channel_count"]
    assert scored["improvement"]["min"] > 0.5
# --- issue #190: the wall must be driveable by the PF coils alone -----------
