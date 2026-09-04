"""Display sign policy (issue #307): one multiplier per figure, inferred in the
processing layer from the active region, never rectification, never a guess."""

from __future__ import annotations

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft
import vaft.omas
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import ORIENTATIONS, RECIPES, build_model
from vaft.process.signal_processing import SignalOrientation, infer_signal_orientation


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def _pulse(sign: float, noise: float = 0.0, seed: int = 0):
    time = np.linspace(0.0, 1.0, 2001)
    envelope = np.exp(-((time - 0.5) / 0.08) ** 2)
    rng = np.random.default_rng(seed)
    return sign * envelope + noise * rng.standard_normal(time.size), time


def test_the_dominant_sign_is_read_from_the_active_region():
    assert infer_signal_orientation(_pulse(+1)[0]).multiplier == 1
    negative = infer_signal_orientation(_pulse(-1)[0])
    assert negative == SignalOrientation(-1, negative.statistic, negative.count, True, "")
    assert negative.statistic < 0 and negative.count < 2001  # the zero stretches did not vote


def test_noise_and_excursions_do_not_rectify_anything():
    noisy, _ = _pulse(-1, noise=0.15)
    verdict = infer_signal_orientation(noisy)
    assert verdict.multiplier == -1
    oriented = noisy * verdict.multiplier
    assert np.array_equal(np.sign(oriented), -np.sign(noisy))  # zero crossings kept, samples not flipped alone
    positive_with_dip, _ = _pulse(+1)
    positive_with_dip[900:950] = -0.5
    assert infer_signal_orientation(positive_with_dip).multiplier == 1


def test_an_explicit_mask_overrides_the_automatic_region():
    signal = np.concatenate([np.full(700, -1.0), np.full(300, 0.3)])
    assert infer_signal_orientation(signal).multiplier == -1
    assert infer_signal_orientation(signal, mask=np.arange(1000) >= 700).multiplier == 1


def test_nans_and_ambiguity_are_handled_not_guessed():
    signal, _ = _pulse(-1)
    signal[::7] = np.nan
    assert infer_signal_orientation(signal).multiplier == -1
    assert not infer_signal_orientation(np.zeros(100)).resolved
    assert not infer_signal_orientation(np.array([1.0, -1.0, 2.0])).resolved
    balanced = np.concatenate([np.full(500, 1.0), np.full(500, -1.0)])
    assert not infer_signal_orientation(balanced).resolved
    assert infer_signal_orientation(balanced).multiplier == 1  # canonical fallback


def _magnetics(sign: float) -> omas.ODS:
    ods = omas.ODS(consistency_check=False)
    values, time = _pulse(sign)
    ods["magnetics.time"] = time
    ods["magnetics.ip.0.data"] = values * 1e5
    ods["magnetics.ip.0.time"] = time
    ods["magnetics.diamagnetic_flux.0.data"] = -np.abs(values) * 1e-3
    ods["magnetics.diamagnetic_flux.0.time"] = time
    ods["dataset_description.data_entry.pulse"] = 1
    return ods


def test_intuitive_is_the_default_for_signed_conventions_and_visible_when_it_flips():
    assert ORIENTATIONS == ("canonical", "intuitive")
    for name in ("plasma_current_time", "diamagnetic_flux_time", "equilibrium_time_plasma_current", "tf_coil_time_b_t"):
        assert RECIPES[name].orientation == "intuitive", name
    assert RECIPES["barometry_time_pressure"].orientation == "canonical"
    negative = _magnetics(-1)
    flat = len(negative.flat())
    model = build_model("plasma_current_time", normalize_entries(negative))
    assert model.series[0].y.max() > 0 and model.title.endswith("— intuitive orientation (sign flipped)")
    canonical = build_model("plasma_current_time", normalize_entries(negative), orientation="canonical")
    assert canonical.series[0].y.min() < 0 and "intuitive" not in canonical.title
    assert np.allclose(model.series[0].y, -canonical.series[0].y)
    assert len(negative.flat()) == flat and float(np.min(negative["magnetics.ip.0.data"])) < 0
    positive = build_model("plasma_current_time", normalize_entries(_magnetics(+1)))
    assert positive.series[0].y.max() > 0 and "intuitive" not in positive.title
    with pytest.raises(ValueError, match="orientation must be one of"):
        build_model("plasma_current_time", normalize_entries(negative), orientation="up")


def test_the_packaged_shot_shows_a_positive_diamagnetic_envelope_and_an_unflipped_current(sample):
    figure, axes = vaft.omas.plot_diamagnetic_flux_time(sample)
    assert axes.lines[0].get_ydata().max() > 1.0 and "(sign flipped)" in axes.get_title()
    plt.close(figure)
    figure, axes = vaft.omas.plot_diamagnetic_flux_time(sample, orientation="canonical")
    assert axes.lines[0].get_ydata().min() < -1.0 and "intuitive" not in axes.get_title()
    plt.close(figure)
    figure, axes = vaft.omas.plot_plasma_current_time(sample)
    assert "intuitive" not in axes.get_title()  # VEST samples store ip > 0 today (#307 note)
    plt.close(figure)


def test_synthetic_markers_flip_with_their_waveform():
    ods = _magnetics(-1)
    for index, time in enumerate((0.45, 0.5, 0.55)):
        ods[f"equilibrium.time_slice.{index}.time"] = time
        ods[f"equilibrium.time_slice.{index}.constraints.ip.reconstructed"] = -0.9e5
    model = build_model("plasma_current_time", normalize_entries(ods), synthetic="equilibrium")
    roles = {trace.role: trace for trace in model.series}
    assert roles["reconstruction"].y.min() > 0 and roles[""].y.max() > 0


def test_no_plotting_path_rectifies_with_abs():
    import pathlib

    # The canonical pipeline only: the legacy vaft.plot.<view> modules are
    # deprecated wholesale and their abs() uses were audited in #307.
    root = pathlib.Path(vaft.__file__).parent / "plot"
    offenders = []
    files = list((root / "backend").rglob("*.py")) + list((root / "renderers").rglob("*.py")) + [
        root / "style.py", root / "display.py", pathlib.Path(vaft.__file__).parent / "omas" / "interactive.py",
    ]
    for path in files:
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if "np.abs(" in line and "sign" in line and "orientation" not in line:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, offenders


def test_each_shot_is_oriented_by_its_own_convention():
    entries = normalize_entries(_magnetics(-1), label="key") + normalize_entries(_magnetics(+1), label="key")
    entries = [("neg", entries[0][1]), ("pos", entries[1][1])]
    model = build_model("plasma_current_time", entries)
    by_entry = {trace.entry: trace for trace in model.series}
    canonical = build_model("plasma_current_time", entries, orientation="canonical")
    stored = {trace.entry: trace for trace in canonical.series}
    assert np.allclose(by_entry["neg"].y, -stored["neg"].y)  # flipped by its own convention
    assert np.allclose(by_entry["pos"].y, stored["pos"].y)   # left alone by its own
    assert model.title.endswith("(sign flipped)")


def test_asymmetric_error_bars_change_places_with_the_sign():
    ods = _magnetics(-1)
    n = ods["magnetics.ip.0.data"].size
    ods["magnetics.ip.0.data_error_lower"] = np.full(n, 1e3)
    ods["magnetics.ip.0.data_error_upper"] = np.full(n, 5e3)
    canonical = build_model("plasma_current_time", normalize_entries(ods), orientation="canonical").series[0]
    flipped = build_model("plasma_current_time", normalize_entries(ods)).series[0]
    assert np.allclose(flipped.yerr[0], canonical.yerr[1]) and np.allclose(flipped.yerr[1], canonical.yerr[0])


def test_the_detector_wants_a_one_dimensional_signal():
    with pytest.raises(ValueError, match="one-dimensional"):
        infer_signal_orientation(np.ones((2, 3)))


def test_discovery_advertises_the_sign_policy(sample):
    record = vaft.omas.available_plots(sample, query="diamagnetic").find("diamagnetic_flux_time")
    assert record.orientation == {"default": "intuitive", "options": ["canonical", "intuitive"]}
    assert "orientation: intuitive by default" in str(vaft.omas.available_plots(sample, query="diamagnetic"))
    assert vaft.omas.available_plots(sample, query="barometry").find("barometry_time_pressure").orientation["default"] == "canonical"
