"""The transport-atlas renderers (#1427): labels, renderer contract, colour scale, mode branch.

Synthetic atlas tables only; no solver, ODS or file is read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.mathtext as mathtext  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from vaft.plot import transport_atlas as ta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _table(n=6):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "efit_lineage": ["magnetics", "magnetics", "electron_kinetic"] * (n // 3),
        "efit_quality": ["good", "admissible"] * (n // 2),
        "a_over_lne": rng.uniform(0, 3, n), "a_over_lte": rng.uniform(0, 3, n),
        "f_e": np.linspace(0.1, 0.9, n), "q_tot_gb": np.logspace(-1, 2, n),
        "omega_at_gamma_max_ion_scale": [0.1, -0.2, 0.3, -0.4, 0.0, np.nan][:n],
        "gamma_max_ion_scale": [0.05, 0.02, 0.04, 0.01, 0.03, 0.02][:n],
        "qe_gb": rng.normal(size=n), "qi_gb": rng.normal(size=n),
    })


def test_every_label_renders_and_names_a_schema_column():
    spec = importlib.util.spec_from_file_location(
        "transport_atlas_build", ROOT / "workflow" / "transport_atlas" / "build_atlas.py")
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build
    spec.loader.exec_module(build)
    parser = mathtext.MathTextParser("path")
    for column in ta.LABELS:
        assert column in build.SCHEMA, column
        parser.parse(ta.axis_label(column))
    assert ta.axis_label("a_over_lne") == "$a/L_{n_e}$"
    assert ta.axis_label("gamma_max") == r"$\gamma_{\max}$ [$c_s/a$]"
    assert ta.axis_label("unlisted_column") == "unlisted_column"
    sys.modules.pop(spec.name, None)


def test_scatter_follows_the_renderer_contract():
    table = _table()
    fig, ax = plt.subplots()
    out_fig, out_ax = ta.transport_atlas_scatter(table, "a_over_lne", "a_over_lte", "f_e", ax=ax)
    assert out_ax is ax and out_fig is fig
    assert ax.get_xlabel() == "$a/L_{n_e}$" and ax.get_ylabel() == "$a/L_{T_e}$"
    assert ax.vaft_drawn == 6
    plt.close(fig)


def test_the_colour_scale_spans_only_the_rows_drawn():
    table = _table()
    fig, ax = ta.transport_atlas_scatter(table, "a_over_lne", "a_over_lte", "f_e",
                                         where=lambda t: t["f_e"] < 0.5)
    colorbar = fig.axes[-1]
    assert colorbar.get_ylim()[1] < 0.5  # rows with f_e >= 0.5 do not stretch it
    assert ax.vaft_drawn == int((table["f_e"] < 0.5).sum())
    plt.close(fig)


def test_mode_branch_counts_directions_and_skips_zero_or_missing():
    fig, ax = ta.transport_atlas_mode_branch(_table())
    assert ax.vaft_counts == {"electron": 2, "ion": 2}
    assert ax.get_xlabel() == "$a/L_{n_e}$"
    plt.close(fig)


def test_a_missing_column_is_named():
    with pytest.raises(KeyError, match="no column 'gamma_max'"):
        ta.transport_atlas_scatter(_table(), "gamma_max", "a_over_lte")


def test_the_renderer_imports_no_data_layer():
    source = (ROOT / "vaft" / "plot" / "transport_atlas.py").read_text()
    for forbidden in ("import omas", "from omas", "vaft.database", "vaft.omas", "vaft.code"):
        assert forbidden not in source


def test_open_markers_are_visible_without_a_colour_column():
    fig, ax = ta.transport_atlas_scatter(_table(), "a_over_lne", "a_over_lte")
    for collection in ax.collections:
        assert len(collection.get_edgecolors()) > 0  # an open marker with no edge is invisible
    plt.close(fig)


def test_mode_branch_skips_stable_surfaces_and_marks_quality():
    table = _table()
    table.loc[0, "gamma_max_ion_scale"] = -0.01  # stable: omega 0.1 must not count
    fig, ax = ta.transport_atlas_mode_branch(table)
    assert ax.vaft_counts == {"electron": 1, "ion": 2}
    filled = [c for c in ax.collections if len(c.get_facecolors()) and c.get_facecolors()[0][3] > 0]
    assert filled  # good rows are filled
    assert any("admissible" in t.get_text() for t in ax.get_legend().get_texts())
    plt.close(fig)


def test_rows_of_unknown_lineage_do_not_stretch_the_colour_scale():
    table = _table()
    table.loc[0, "efit_lineage"] = "unknown"
    table.loc[0, "f_e"] = 99.0
    fig, ax = ta.transport_atlas_scatter(table, "a_over_lne", "a_over_lte", "f_e")
    assert fig.axes[-1].get_ylim()[1] < 1.0
    plt.close(fig)


def test_schema_units_and_label_units_agree():
    spec = importlib.util.spec_from_file_location(
        "transport_atlas_build_units", ROOT / "workflow" / "transport_atlas" / "build_atlas.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    expected = {"-": "", "Q_GB": "", "Gamma_GB": "", "c_s/a": r"$c_s/a$", "W/m^2": r"W m$^{-2}$",
                "m^-2 s^-1": r"m$^{-2}$ s$^{-1}$", "m^2/s": r"m$^2$ s$^{-1}$", "m": "m", "T": "T"}
    for column, (_, unit) in ta.LABELS.items():
        assert unit == expected[build.SCHEMA[column][0]], column


def test_independent_ti_is_judged_by_the_ratio_not_the_spelling():
    spec = importlib.util.spec_from_file_location(
        "transport_atlas_plot_cli", ROOT / "workflow" / "transport_atlas" / "plot_atlas.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    table = pd.DataFrame({
        "ti_lineage": ["ti_eq_te_assumed", "ti_te_1.2_measured", "pressure_partition_inferred",
                       "measured", None, "unresolved"],
        "ti_te_ratio": [1.0, 1.2, np.nan, np.nan, np.nan, np.nan],
    })
    assert cli.independent_ti(table).tolist() == [False, False, True, True, False, False]
