"""The DCON operating-space figure, and the harvest it draws.

The figure lives in `workflow/` rather than in `vaft.plot` because it needs a
continuous per-point colour channel that no view model carries, and because a
scan is cross-shot while the stage plots are per-shot. Living outside the
package does not have to mean living outside the tests: the script is loaded the
same way `test_pipeline1_paths.py` loads pipeline 1's helper.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gpec_nc_fixtures import write_dcon_output_nc

from vaft.code.gpec import read_dcon_scan


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_3_data_summary"
    / "plot_dcon_scan.py"
)
SPEC = importlib.util.spec_from_file_location("plot_dcon_scan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _scan_tree(tmp_path, cells):
    """A canonical `<time>/<module>/nn=<mode>/` tree, one DCON run per cell."""
    for label, overrides, w_t in cells:
        run_dir = tmp_path / label / "dcon" / "nn=1"
        run_dir.mkdir(parents=True)
        write_dcon_output_nc(run_dir, equilibrium=overrides, w_t=w_t)
    return read_dcon_scan(tmp_path, modes=(1,), shot=39915)


def test_the_scatter_colours_by_time_on_a_continuous_scale(tmp_path):
    """This is the capability the typed view layer cannot express today.

    A `Series` carries one style for a whole trace and no per-point colour, so a
    time-coloured scan would have to be binned into discrete traces -- which
    replaces a colourbar with a legend and loses information.
    """
    rows = _scan_tree(
        tmp_path,
        [
            ("00320", {"q95": 5.0, "li3": 0.80}, -0.5),
            ("00325", {"q95": 6.0, "li3": 0.85}, -0.1),
            ("00330", {"q95": 7.0, "li3": 0.90}, 0.3),
        ],
    )

    figure, axes = MODULE.plot_operating_space(rows)

    (points,) = axes.collections
    np.testing.assert_allclose(points.get_array(), [320.0, 325.0, 330.0])
    np.testing.assert_allclose(points.get_offsets(), [[5.0, 0.80], [6.0, 0.85], [7.0, 0.90]])
    # A colourbar, not a legend: the second axes is the scale itself.
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "time [ms]"
    assert axes.get_legend() is None
    assert "shot 39915" in axes.get_title()
    assert axes.get_xlabel() == "q95" and axes.get_ylabel() == "li3"


def test_an_uncomputed_stability_verdict_is_its_own_class_not_unstable(tmp_path):
    """`stable_free_boundary` is tri-state, and the third state is not a verdict.

    DCON leaves it `None` when it computed no total-energy eigenvalue at all
    (`vac_flag=false`), which must not be drawn as if the run had been found
    unstable.
    """
    rows = _scan_tree(
        tmp_path,
        [
            ("00320", {"q95": 5.0, "li3": 0.80}, -0.5),
            ("00325", {"q95": 6.0, "li3": 0.85}, 0.3),
        ],
    )
    rows.append({**rows[0], "time_ms": 335.0, "stable_free_boundary": None, "q95": 7.0})

    figure, axes = MODULE.plot_operating_space(rows, colour="stability")

    assert sorted(text.get_text() for text in axes.get_legend().get_texts()) == [
        "not computed",
        "stable",
        "unstable",
    ]


def test_rows_missing_the_plotted_columns_are_dropped_not_drawn_as_zero(tmp_path):
    rows = _scan_tree(tmp_path, [("00320", {"q95": 5.0, "li3": 0.80}, -0.5)])
    rows.append({**rows[0], "time_ms": 325.0, "li3": None})

    figure, axes = MODULE.plot_operating_space(rows)

    (points,) = axes.collections
    assert len(points.get_offsets()) == 1
    assert "1 runs" in axes.get_title()


def test_a_scan_with_nothing_plottable_fails_loudly(tmp_path):
    rows = _scan_tree(tmp_path, [("00320", False, -0.5)])  # no equilibrium block at all

    with pytest.raises(SystemExit, match="no run in this scan"):
        MODULE.plot_operating_space(rows)


def test_the_rejection_names_the_colour_column_when_that_is_what_failed(tmp_path):
    """A scan with no numeric time label fails on `time_ms`, not on the axes.

    `read_dcon_scan` leaves `time_ms` None whenever a run directory is not named
    by a number, which `rt.time_label` produces whenever the time is unknown --
    so this is the realistic way a whole scan becomes unplottable, and naming
    the axes instead would send the reader to the two columns that were fine.
    """
    rows = _scan_tree(tmp_path, [("00320", {"q95": 5.0, "li3": 0.80}, -0.5)])
    rows[0]["time_ms"] = None

    with pytest.raises(SystemExit, match="time_ms"):
        MODULE.plot_operating_space(rows)


def test_the_figure_is_written_where_it_was_asked_for(tmp_path):
    rows = _scan_tree(tmp_path, [("00320", {"q95": 5.0, "li3": 0.80}, -0.5)])
    target = tmp_path / "operating_space.png"

    MODULE.plot_operating_space(rows, output=target)

    assert target.exists() and target.stat().st_size > 0
