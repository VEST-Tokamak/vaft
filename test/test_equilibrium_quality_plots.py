"""Population renderers for the #1644 cohort tables: tables in, figures out, no ODS."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vaft.plot import equilibrium_quality as plots  # noqa: E402


def _points():
    rng = np.random.default_rng(0)
    rows = []
    for label, scale in (("good", 0.02), ("admissible", 0.2), ("unreconstructible", 0.5)):
        for i in range(20):
            m = rng.uniform(-5, 5)
            rows.append({"family": "bpol_probe", "unit": "mT", "measured": m, "reconstructed": m * (1 + scale),
                         "z": rng.normal(0, 1 + 5 * scale), "fitted": True, "quality_label": label})
    return pd.DataFrame(rows)


def test_measured_vs_reconstructed_keeps_identity_and_reports_absolute_error():
    fig, ax = plots.equilibrium_quality_measured_vs_reconstructed(_points(), family="bpol_probe")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "y = x" in labels and any(t.startswith("High quality (n=20, MAE") for t in labels)
    assert any(t.startswith("Admissible (n=20, MAE") for t in labels)
    assert any(t.startswith("Failure (n=20, MAE") for t in labels)
    assert ax.get_xlabel() == "measured [mT]"
    matplotlib.pyplot.close(fig)


def test_measured_vs_reconstructed_uses_slide_presentation():
    fig, ax = plots.equilibrium_quality_measured_vs_reconstructed(
        _points(), family="bpol_probe", format="slide",
    )
    assert fig.get_size_inches()[0] == 11.0
    assert ax.title.get_fontsize() >= 18
    assert ax.get_legend().get_texts()[0].get_fontsize() >= 14
    matplotlib.pyplot.close(fig)


def test_diagnostic_slides_use_one_large_figure_per_present_family():
    slides = plots.equilibrium_quality_diagnostic_slides(_points())
    assert list(slides) == ["bpol_probe"]
    fig, ax = slides["bpol_probe"]
    assert fig.get_size_inches()[0] == 11.0
    assert ax.get_title() == "Poloidal probes"
    matplotlib.pyplot.close(fig)


def test_diagnostic_grid_has_four_panels_and_one_shared_legend():
    families = ("bpol_probe", "flux_loop", "ip", "diamagnetic_flux")
    points = pd.concat([_points().assign(family=family) for family in families], ignore_index=True)
    fig, axes = plots.equilibrium_quality_diagnostic_grid(points)
    assert axes.shape == (2, 2)
    assert fig.get_size_inches()[0] == 9.8
    assert axes[0, 1].get_position().x0 - axes[0, 0].get_position().x1 < 0.15
    assert all(ax.get_legend() is None for ax in axes.ravel())
    assert [ax.get_title() for ax in axes.ravel()] == [
        "Poloidal probes", "Flux loops", "Plasma current", "Diamagnetic flux",
    ]
    assert axes[1, 1].get_xlim() == (0.0, 8.0)
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "High quality", "Admissible", "Failure", "y = x",
    ]
    matplotlib.pyplot.close(fig)


def test_diamagnetic_flux_view_focuses_both_axes_without_dropping_outliers():
    points = pd.DataFrame({
        "family": ["diamagnetic_flux", "diamagnetic_flux"],
        "unit": ["mWb", "mWb"],
        "measured": [5.0, 12.0],
        "reconstructed": [6.0, 180.0],
        "z": [0.1, 3.0],
        "fitted": [True, True],
        "quality_label": ["good", "good"],
    })
    fig, ax = plots.equilibrium_quality_measured_vs_reconstructed(
        points, family="diamagnetic_flux", format="slide",
    )
    assert ax.get_xlim() == (0.0, 8.0)
    assert ax.get_ylim() == (0.0, 8.0)
    assert ax.get_title() == "Diamagnetic flux"
    assert any(tuple(line.get_xdata()) == (0.0, 8.0) for line in ax.lines)
    assert "n=2" in ax.get_legend().get_texts()[0].get_text()
    assert "MAE 84.50 mWb" in ax.get_legend().get_texts()[0].get_text()
    matplotlib.pyplot.close(fig)


def test_residual_ecdf_draws_one_step_per_cohort():
    fig, ax = plots.equilibrium_quality_residual_distribution(_points(), family="bpol_probe")
    assert len([line for line in ax.get_lines() if line.get_drawstyle().startswith("steps")]) == 3
    matplotlib.pyplot.close(fig)


def test_reduced_chi2_ecdf_shades_the_study_band():
    table = pd.DataFrame({"quality_label": ["good", "good", "admissible"], "probe_reduced_chi2": [1.0, 1.2, 7.0]})
    fig, ax = plots.equilibrium_quality_reduced_chi2(table)
    assert any("study band" in t.get_text() for t in ax.get_legend().get_texts())
    matplotlib.pyplot.close(fig)


def test_validation_matrix_marks_thomson_as_its_own_row():
    census = {"matrix": {
        "rule_convergence": {"good": {"n": 2, "pass": 1.0, "fail": 0.0}, "admissible": {"n": 1, "pass": 1.0, "fail": 0.0},
                             "unreconstructible": {"n": 3, "pass": 0.0, "fail": 1.0}},
        "thomson_status": {"good": {"n": 2, "pass": 0.5, "fail": 0.0}, "admissible": {"n": 1, "pass": 0.0, "fail": 1.0},
                           "unreconstructible": {"n": 0}},
    }}
    fig, ax = plots.equilibrium_quality_validation_matrix(census)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["convergence", "Thomson (separate)"]
    assert any(line.get_ydata()[0] == 0.5 for line in ax.get_lines())  # the rule line above Thomson
    matplotlib.pyplot.close(fig)


def test_selection_funnel_draws_one_bar_per_cohort():
    funnel = {"rules": ["finite", "dwdt_fraction"], "cohorts": {
        "good": {"candidates": 3, "selected": 1, "rejected": 2,
                 "exclusions": [{"rule": "finite", "removed_in_sequence": 1}, {"rule": "dwdt_fraction", "removed_in_sequence": 1}]},
        "admissible": {"candidates": 2, "selected": 1, "rejected": 1,
                       "exclusions": [{"rule": "finite", "removed_in_sequence": 0}, {"rule": "dwdt_fraction", "removed_in_sequence": 1}]},
    }}
    fig, ax = plots.equilibrium_quality_selection_funnel(funnel)
    assert [t.get_text().split("\n")[0] for t in ax.get_yticklabels()] == ["good", "admissible-only"]
    assert any("selected" in t.get_text() for t in ax.texts)
    matplotlib.pyplot.close(fig)
