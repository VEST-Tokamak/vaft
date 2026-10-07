"""Presentation contract for the Atlas Thomson-pressure comparison."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import vaft.plot as plots  # noqa: E402


def test_pressure_comparison_maps_the_channel_interval_to_half_x_through_x():
    points = pd.DataFrame({
        "p_e_pa": [100.0, 200.0],
        "p_efit_pa": [150.0, 400.0],
        "efit_lineage": ["magnetics", "electron_kinetic"],
    })
    fig, ax = plots.kinetic_state_pressure_comparison(points, format="slide")
    assert fig.get_size_inches()[0] == pytest.approx(plots.FORMATS["slide"].width_in)
    assert ax.title.get_fontsize() >= 18
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert any("p_{EFIT}=p_e" in line.get_label() for line in ax.lines)
    assert any("p_{EFIT}=2p_e" in line.get_label() for line in ax.lines)
    offsets = [tuple(point) for group in ax.collections if hasattr(group, "get_offsets")
               for point in group.get_offsets()]
    assert (200.0, 150.0) in offsets
    assert (400.0, 400.0) in offsets
    plt.close(fig)


def test_pressure_comparison_rejects_nonpositive_or_missing_data():
    with pytest.raises(ValueError, match="missing columns"):
        plots.kinetic_state_pressure_comparison(pd.DataFrame({"p_e_pa": [1.0]}))
    with pytest.raises(ValueError, match="positive finite"):
        plots.kinetic_state_pressure_comparison(pd.DataFrame({
            "p_e_pa": [0.0], "p_efit_pa": [1.0], "efit_lineage": ["magnetics"],
        }))


def test_virial_pair13_comparison_uses_matched_beta_estimates():
    states = pd.DataFrame({
        "beta_p_volume": [0.20, 0.30, 0.40],
        "beta_p_pair_13": [0.201, 0.302, float("nan")],
        "efit_lineage": ["magnetics", "electron_kinetic", "magnetics"],
    })
    fig, ax = plots.kinetic_state_virial_pair13_comparison(states, format="slide")
    assert fig.get_size_inches()[0] == pytest.approx(plots.FORMATS["slide"].width_in)
    assert ax.title.get_fontsize() >= 18
    assert ax.get_xlim() == ax.get_ylim()
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == r"$\beta_{p,13}=\beta_{p,vol}$"
    offsets = [tuple(point) for group in ax.collections for point in group.get_offsets()]
    assert offsets == [(0.2, 0.201), (0.3, 0.302)]
    assert any("Median relative\ndifference" in label.get_text() for label in ax.texts)
    plt.close(fig)


def test_virial_pair13_comparison_requires_finite_pairs():
    with pytest.raises(ValueError, match="missing columns"):
        plots.kinetic_state_virial_pair13_comparison(pd.DataFrame({"beta_p_volume": [0.2]}))
    with pytest.raises(ValueError, match="finite beta_p pairs"):
        plots.kinetic_state_virial_pair13_comparison(pd.DataFrame({
            "beta_p_volume": [0.2], "beta_p_pair_13": [float("nan")],
            "efit_lineage": ["magnetics"],
        }))


def test_virial_li_comparison_uses_internal_inductance_pairs():
    states = pd.DataFrame({
        "li_volume": [0.60, 0.70, 0.80],
        "li_pair_13": [0.602, 0.704, float("nan")],
        "efit_lineage": ["magnetics", "electron_kinetic", "magnetics"],
    })
    fig, ax = plots.kinetic_state_virial_li_comparison(states, format="slide")
    assert fig.get_size_inches()[0] == pytest.approx(plots.FORMATS["slide"].width_in)
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.lines[0].get_label() == r"$l_{i,13}=l_{i,vol}$"
    offsets = [tuple(point) for group in ax.collections for point in group.get_offsets()]
    assert offsets == [(0.6, 0.602), (0.7, 0.704)]
    assert "internal inductance" in ax.get_title().lower()
    plt.close(fig)
