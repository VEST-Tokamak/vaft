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
