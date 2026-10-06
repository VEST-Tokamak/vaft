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


def test_measured_vs_reconstructed_keeps_the_identity_line_and_reports_z_stats():
    fig, ax = plots.equilibrium_quality_measured_vs_reconstructed(_points(), family="bpol_probe")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "y = x" in labels and any(t.startswith("good (n=20, z bias") for t in labels)
    assert ax.get_xlabel() == "measured [mT]"
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
