"""The plasma-free vacuum benchmark as plots (issue #190, plotting roadmap #1242 C3).

``magnetics_table_vacuum_benchmark`` and ``magnetics_overview_vacuum_benchmark``
show one :func:`~vaft.validation.vacuum_benchmark.run_benchmark_case`;
``magnetics_table_vacuum_benchmark_aggregate`` shows
:func:`~vaft.validation.vacuum_benchmark.aggregate_benchmark` over several
entries.  The benchmark grades nothing, so the views must not either: a row's
status is a fact the case records (evaluated, flagged as contradicting its own
array, excluded for too few samples), never a pass or a fail, and nothing the
case reports -- an excluded channel, an entry that cannot supply a case -- is
silently dropped.
"""

from __future__ import annotations

import contextlib
import copy
import io
import warnings

import numpy as np
import pytest
from matplotlib.figure import Figure

import vaft
import vaft.omas
import vaft.plot
from vaft.plot.backend import options as option_schema
from vaft.plot.backend.recipes import build_model
from vaft.plot.models import LineSeries, Panels, Table
from vaft.validation.vacuum_benchmark import BenchmarkError, run_benchmark_case

from test_vacuum_benchmark import N_TIME, TIME, plasma_shot, vacuum_shot  # noqa: F401 -- fixtures

TABLE = "magnetics_table_vacuum_benchmark"
OVERVIEW = "magnetics_overview_vacuum_benchmark"
AGGREGATE = "magnetics_table_vacuum_benchmark_aggregate"


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _column(model: Table, name: str) -> list:
    return [cell.value for cell in model.column(name)]


def _note(cell) -> str:
    """A Note cell's text: inline when short, else a footnote (as the C2 table does)."""
    return cell.value or cell.note


def _rows_by_channel(model: Table) -> dict[str, tuple]:
    names = _column(model, "Channel")
    return dict(zip(names, model.rows))


@pytest.fixture(scope="module")
def sample():
    return _quiet(vaft.omas.load, vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def sample_case(sample):
    return _quiet(run_benchmark_case, sample, shot=39915)


@pytest.fixture(scope="module")
def sample_table(sample):
    return _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark, sample)


# ---------------------------------------------------------------------------
# One case as a table
# ---------------------------------------------------------------------------


def test_the_table_has_one_row_per_scored_channel_in_benchmark_order(vacuum_shot):
    case = run_benchmark_case(vacuum_shot)
    model = _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark, vacuum_shot)

    assert isinstance(model, Table)
    names = [row["name"] for row in case["metrics"]["channels"]]
    assert len(model.rows) == len(names) == 4
    assert _column(model, "Channel") == names
    assert _column(model, "#") == [1, 2, 3, 4]
    assert _column(model, "Status") == ["evaluated"] * 4
    for row, model_row in zip(case["metrics"]["channels"], model.rows):
        assert model_row[5].value == pytest.approx(row["improvement"])
        assert model_row[6].value == pytest.approx(row["normalized_residual"])
        assert model_row[7].value == pytest.approx(row["correlation"])
        assert model_row[8].value == pytest.approx(row["wall_authority"])
    assert "vacuum case" in model.caption
    assert "sufficient: yes" in model.caption
    # Statuses are words, not the pass/warn/fail classification a renderer may colour.
    assert all(cell.status == "" for row in model.rows for cell in row)
    assert any("No thresholds and no verdict" in note for note in model.notes)


def test_the_packaged_shot_flags_its_contradicting_probe(sample_table, sample_case):
    rows = _rows_by_channel(sample_table)
    assert len(sample_table.rows) == len(sample_case["metrics"]["channels"]) == 73
    flagged = {entry["channel"] for entry in sample_case["channels"]["flagged"]}
    assert flagged == {"MagneticFieldProbe_C4-04"}
    row = rows["C4-04"]
    assert row[4].value == "flagged"
    assert "array contradiction" in _note(row[9]) and "scored medians" in _note(row[9])
    assert _column(sample_table, "Status").count("flagged") == 1
    assert _column(sample_table, "Status").count("evaluated") == 72
    scored = sample_case["metrics"]["summary"]["scored"]
    assert f"Scored medians over {scored['count']} channels" in sample_table.caption
    assert f"improvement {scored['improvement']['median']:.3g}" in sample_table.caption
    assert "shot 39915" in sample_table.title


def test_a_flag_from_the_array_review_is_a_row_status(vacuum_shot, monkeypatch):
    from vaft.validation import vacuum_benchmark

    monkeypatch.setattr(vacuum_benchmark, "array_contradictions", lambda channels, window: [
        {"channel": "outboard_probe", "kind": "b_field_pol_probe", "reason": "array_contradiction",
         "fraction": 0.5, "from": window[0], "to": window[1]},
    ])
    model = _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark, vacuum_shot)
    status = dict(zip(_column(model, "Channel"), _column(model, "Status")))
    assert status == {
        "inboard_probe": "evaluated", "outboard_probe": "flagged",
        "inboard_loop": "evaluated", "outboard_loop": "evaluated",
    }
    assert "Scored medians over 3 channels" in model.caption


def test_an_excluded_channel_is_a_row_with_its_reason(vacuum_shot):
    window = run_benchmark_case(vacuum_shot)["validation_window"]
    inside = np.flatnonzero(TIME >= window[0])
    validity = np.zeros(N_TIME, dtype=int)
    validity[inside[1:]] = -1  # one usable sample left in the window
    vacuum_shot["magnetics.flux_loop.1.flux.validity_timed"] = validity

    model = _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark, vacuum_shot)
    row = _rows_by_channel(model)["outboard_loop"]
    assert row[4].value == "excluded"
    assert "usable sample" in _note(row[9])
    assert row[5].missing
    assert "1 excluded" in model.title


def test_options_reach_the_benchmark_and_bad_values_are_refused(vacuum_shot):
    model = _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark, vacuum_shot, resistance_scale=2.0)
    assert "resistance scale 2" in model.caption
    with pytest.raises(BenchmarkError, match="solver history"):
        _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark, vacuum_shot, n_tau=500.0)
    for bad in ({"per_family": True}, {"per_family": 0}, {"resistance_scale": -1.0},
                {"n_tau": "three"}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            build_model(TABLE, [("x", vacuum_shot)], **bad)


def test_the_benchmark_options_are_declared_only_by_the_benchmark_views():
    for name in (TABLE, OVERVIEW, AGGREGATE):
        option_schema.validate_options(name, {"resistance_scale": 1.5, "n_tau": 3.0, "per_family": 2})
    with pytest.raises(ValueError, match="n_tau"):
        option_schema.validate_options("magnetics_overview_vacuum", {"n_tau": 3.0})


def test_the_rendered_table_prints_as_text(sample_table):
    rendered = vaft.plot.magnetics_table_vacuum_benchmark(sample_table)
    text = str(rendered)
    assert "C4-04" in text and "flagged" in text
    assert "<table" in rendered.html()


# ---------------------------------------------------------------------------
# One case drawn
# ---------------------------------------------------------------------------


def test_the_overview_draws_the_three_scores_with_a_zero_reference(vacuum_shot):
    figure, _axes = _quiet(vaft.omas.plot_magnetics_overview_vacuum_benchmark, vacuum_shot)
    assert isinstance(figure, Figure)

    model = _quiet(vaft.omas.extract_magnetics_overview_vacuum_benchmark, vacuum_shot)
    assert isinstance(model, Panels) and len(model.models) == 3
    assert all(isinstance(panel, LineSeries) for panel in model.models)
    zero = [s for s in model.models[0].series if s.label.startswith("0:")]
    assert len(zero) == 1 and np.all(zero[0].y == 0.0)
    assert not any(s.label.startswith("0:") for panel in model.models[1:] for s in panel.series)
    # Every evaluated channel is one point per panel, at its table position.
    points = sorted(x for s in model.models[1].series for x in s.x)
    assert points == [1.0, 2.0, 3.0, 4.0]
    # Family colours are intent tokens, never literal colours.
    assert all(s.style["color"].startswith("palette:") for s in model.models[1].series)


def test_flagged_probes_are_hollow_and_named_and_excluded_ones_are_not_drawn(sample):
    model = _quiet(vaft.omas.extract_magnetics_overview_vacuum_benchmark, sample)
    for panel in model.models:
        flagged = [s for s in panel.series if s.label == "flagged (array contradiction)"]
        assert len(flagged) == 1
        assert flagged[0].style["markerfacecolor"] == "none"
        assert list(flagged[0].x) == [44.0]  # C4-04's row in the table
        families = [s for s in panel.series if s.style.get("color", "").startswith("palette:")]
        assert 44.0 not in {x for s in families for x in s.x}
        assert sum(s.x.size for s in families) == 72
    assert "1 flagged" in model.suptitle and "excluded: none" in model.suptitle


# ---------------------------------------------------------------------------
# Across entries
# ---------------------------------------------------------------------------


def test_the_aggregate_lists_every_axis(vacuum_shot, plasma_shot):
    model = _quiet(
        vaft.omas.extract_magnetics_table_vacuum_benchmark_aggregate,
        [vacuum_shot, plasma_shot], label=["1", "2"],
    )
    groups = _column(model, "Group")
    values = list(zip(groups, _column(model, "Value")))
    assert [g for g in dict.fromkeys(groups)] == ["case", "family", "excitation", "machine era"]
    assert ("case", "1") in values and ("case", "2") in values
    assert ("excitation", "PF0,PF1") in values
    assert ("machine era", "unknown") in values  # no pulse, so no VEST era
    assert {v for g, v in values if g == "family"} == {
        "inboard", "outboard", "inboard_flux_loop", "outboard_flux_loop",
    }
    assert "2 cases" in model.caption and "undriven cases: none" in model.caption


def test_an_entry_without_a_case_is_a_row_with_its_reason(vacuum_shot, plasma_shot):
    plasma_shot["magnetics.ip.0.validity"] = -2
    model = _quiet(
        vaft.omas.extract_magnetics_table_vacuum_benchmark_aggregate,
        [vacuum_shot, plasma_shot], label=["good", "bad"],
    )
    rows = {(row[0].value, row[1].value): row for row in model.rows}
    bad = rows[("case", "bad")]
    assert "no benchmark case" in _note(bad[8]) and "unusable" in _note(bad[8])
    assert bad[2].missing and bad[4].missing
    assert ("case", "good") in rows
    assert "1 without one" in model.title


def test_an_entry_without_the_circuits_is_a_row_not_a_crash(vacuum_shot):
    stripped = copy.deepcopy(vacuum_shot)
    del stripped["pf_passive"]
    model = _quiet(
        vaft.omas.extract_magnetics_table_vacuum_benchmark_aggregate,
        [vacuum_shot, stripped], label=["whole", "no wall"],
    )
    rows = {(row[0].value, row[1].value): row for row in model.rows}
    assert "carries no pf_passive.loop.{i}.resistance" in _note(rows[("case", "no wall")][8])
    assert rows[("case", "whole")][2].value == 1


def test_an_aggregate_of_nothing_says_so(plasma_shot):
    plasma_shot["magnetics.ip.0.validity"] = -2
    model = _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark_aggregate, plasma_shot)
    assert model.caption.startswith("aggregate empty: no case produced an evaluated channel")
    assert len(model.rows) == 1 and "unusable" in _note(model.rows[0][8])


def test_the_packaged_aggregate_names_the_flagged_probe_and_its_era(sample):
    model = _quiet(vaft.omas.extract_magnetics_table_vacuum_benchmark_aggregate, sample)
    values = list(zip(_column(model, "Group"), _column(model, "Value")))
    assert ("case", "39915") in values
    assert ("machine era", "vest-pre-43017-pf1906") in values
    assert "flagged channels: 39915: C4-04" in model.caption
    rendered = vaft.plot.magnetics_table_vacuum_benchmark_aggregate(model)
    assert "C4-04" in str(rendered)


def test_two_entries_with_the_same_label_stay_two_cases(vacuum_shot):
    model = _quiet(
        build_model, AGGREGATE, [("1", vacuum_shot), ("1", copy.deepcopy(vacuum_shot))],
    )
    assert [v for g, v in zip(_column(model, "Group"), _column(model, "Value")) if g == "case"] == [
        "1", "1 (2)",
    ]
