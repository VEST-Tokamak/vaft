"""Run only the small cross-shot example cells, never the full legacy notebook."""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

pytest.importorskip("omas")


def test_unified_notebook_cells_render_and_discover(monkeypatch):
    path = Path(__file__).parents[1] / "notebooks/vest_experimental_data_list.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    names = (
        "unified-fixture-load", "unified-geometry-plot",
        "unified-kinetic-plot", "unified-discovery",
    )
    assert all(cells[name]["outputs"] == [] for name in names)
    monkeypatch.setattr(plt, "show", lambda: None)
    # Count the figures these cells draw, not whatever an earlier test in the
    # session left open: on the 0.8.0 release gate three stray single-axes
    # figures from a preceding file made this read 5 instead of 2.
    plt.close("all")
    scope = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            exec("".join(cells[name]["source"]), scope)
    assert "kinetic_overview_profiles" in stream.getvalue()
    assert "machine_geometry_poloidal" in stream.getvalue()
    assert scope["manifest"]["physical_discharge"] is False
    figures = [plt.figure(number) for number in plt.get_fignums()]
    try:
        assert len(figures) == 2
        assert sorted(len(figure.axes) for figure in figures) == [1, 4]
        for figure in figures:
            titles = [axis.get_title() for axis in figure.axes]
            if figure._suptitle is not None:
                titles.append(figure._suptitle.get_text())
            assert any("Cross-shot composite" in title for title in titles)
    finally:
        plt.close("all")


def test_four_view_atlas_cell_is_small_and_marked(monkeypatch):
    path = Path(__file__).parents[1] / "notebooks/vest_experimental_data_list.ipynb"
    cells = {cell["id"]: cell for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]}
    assert cells["unified-atlas-views"]["outputs"] == []
    monkeypatch.setattr(plt, "show", lambda: None)
    scope = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("unified-fixture-load", "unified-atlas-views"):
            exec("".join(cells[name]["source"]), scope)
    try:
        figure = scope["fig"]
        assert len(figure.axes) == 4
        assert "Cross-shot composite" in figure._suptitle.get_text()
        assert "Geometry reference: shot 39915" in figure._suptitle.get_text()
        assert [axis.get_title() for axis in figure.axes] == [
            "R–Z", "Top (X–Y)", "3-D machine space", "Calibrated FAST camera pixels"
        ]
    finally:
        plt.close("all")
