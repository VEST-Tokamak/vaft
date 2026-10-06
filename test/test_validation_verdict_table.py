"""The validation verdict table: every registered check, one row each (roadmap #1242 C2).

``equilibrium_table_validation`` tabulates what :mod:`vaft.validation` concluded
about one equilibrium slice.  It adds no physics and no tolerance: the statuses
must be the ones the validation functions return when called directly, a check
that cannot be evaluated stays a ``not_available`` row with its reason, and the
caption's overall status is :func:`~vaft.validation.equilibrium.aggregate_status`
of the rows.
"""

from __future__ import annotations

import contextlib
import copy
import io
import re
import warnings

import numpy as np
import pytest

import vaft
import vaft.omas
import vaft.plot
from vaft.plot import registry
from vaft.plot.backend.recipes import resolve_time_slice
from vaft.plot.models import Table
from vaft.plot.renderers.tables import TextView
from vaft.validation import equilibrium as V
from vaft.validation.model import ValidationStatus
from vaft.validation.registry import CHECKS

NAME = "equilibrium_table_validation"
ORDER = ("pass", "warn", "fail", "indeterminate", "not_available")


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def sample():
    return _quiet(vaft.omas.load, str(vaft.data.data_path("samples/39915/omas.json.gz")))


@pytest.fixture(scope="module")
def table(sample):
    return _quiet(vaft.omas.extract_equilibrium_table_validation, sample)


def _rows(model: Table) -> dict[str, tuple]:
    """``{registry key: row}`` read back from the Category and Check columns."""
    return {f"{row[0].value}.{row[1].value}": row for row in model.rows}


def _status(row) -> str:
    return row[model_column("Status")].status


def model_column(name: str) -> int:
    return ("Category", "Check", "Measure", "Value", "Criterion", "Status", "Note").index(name)


def _direct(ods, index: int) -> dict[str, str]:
    """Every check's status from the validation functions themselves, one by one."""
    statuses = {
        "verification.structure": V.verify_structure(ods, time_slice=index)["status"],
        "verification.continuity": V.verify_continuity(ods)["status"],
        "verification.convention": V.verify_convention(ods, time_slice=index)["status"],
        "verification.convergence": V.verify_convergence(ods, time_slice=index)["status"],
    }
    for category, results in (
        ("diagnostic_fit", V.validate_magnetic_fit(ods, time_slice=index)),
        ("physical_validity", V.validate_physical(ods, time_slice=index)),
        ("independent_validation", V.validate_independent(ods, time_slice=index)),
    ):
        statuses.update({f"{category}.{check}": result["status"] for check, result in results.items()})
    return {key: str(ValidationStatus(value)) for key, value in statuses.items()}


# ---------------------------------------------------------------------------
# registration and discovery
# ---------------------------------------------------------------------------


def test_it_is_a_canonical_equilibrium_table():
    spec = registry.get_spec(NAME)
    assert (spec.subject, spec.view, spec.quantity) == ("equilibrium", "table", "validation")
    assert spec.model is Table
    assert NAME in vaft.plot.__all__ and getattr(vaft.plot, NAME) is spec.renderer
    assert NAME in {record.name for record in vaft.plot.available_plots(view="table")}


def test_discovery_offers_it_on_the_sample(sample):
    found = {record.name: record for record in vaft.omas.available_plots(sample, view="table")}
    assert NAME in found and found[NAME].backends == ()


# ---------------------------------------------------------------------------
# the rows are the validation layer's verdicts
# ---------------------------------------------------------------------------


def test_every_registered_check_appears_exactly_once_in_registry_order(table):
    keys = [f"{row[0].value}.{row[1].value}" for row in table.rows]
    assert keys == list(CHECKS)
    assert len(set(keys)) == len(CHECKS) == 22


def test_every_registered_check_names_its_measure_or_none():
    from vaft.plot.backend.tables import _VERDICT_MEASURE

    assert set(_VERDICT_MEASURE) == set(CHECKS)


def test_the_statuses_are_the_validation_functions_own(sample, table):
    index = resolve_time_slice(sample)[0]
    direct = _quiet(_direct, sample, index)
    assert set(direct) == set(CHECKS)
    shown = {key: _status(row) for key, row in _rows(table).items()}
    assert shown == direct


def test_the_rows_agree_with_the_unified_report(sample, table):
    index = resolve_time_slice(sample)[0]
    report = _quiet(V.validate_equilibrium, sample, time_slice=index)
    for key, row in _rows(table).items():
        category, check = key.split(".", 1)
        if key == "verification.continuity":
            # One slice has no continuity; the table judges the whole IDS.
            assert report[category][check]["status"] == "not_available"
            assert "whole IDS, 9 stored slices" in row[model_column("Note")].value
            continue
        assert _status(row) == report[category][check]["status"], key


def test_the_graded_value_is_the_one_the_check_reported(sample, table):
    index = resolve_time_slice(sample)[0]
    physical = _quiet(V.validate_physical, sample, time_slice=index)
    rows = _rows(table)
    value = model_column("Value")
    assert rows["physical_validity.virial_identity"][value].value == pytest.approx(physical["virial_identity"]["rms"])
    assert rows["physical_validity.pressure_consistency"][value].value == pytest.approx(
        physical["pressure_consistency"]["log_ratio"]
    )
    fit = _quiet(V.validate_magnetic_fit, sample, time_slice=index)
    assert rows["diagnostic_fit.bpol_probe"][value].value == pytest.approx(fit["bpol_probe"]["z_rms"])
    # A rule over several facts has no single number, and none is invented.
    assert rows["verification.structure"][value].missing
    assert rows["physical_validity.virial_parameter_plausibility"][value].missing


def test_not_available_rows_carry_their_reason(table):
    unavailable = [row for row in table.rows if _status(row) == "not_available"]
    # 39915: no solver convergence record, the #891 unknown uncertainty model
    # on all six fit checks, no core_profiles and no Thomson scattering.
    assert len(unavailable) == 9
    for row in unavailable:
        note = row[model_column("Note")]
        assert (note.value or note.note), f"{row[0].value}.{row[1].value} gives no reason"


def test_the_caption_is_the_aggregate_of_the_rows(table):
    statuses = [_status(row) for row in table.rows]
    overall = str(V.aggregate_status(statuses))
    counts = ", ".join(f"{statuses.count(status)} {status.upper()}" for status in ORDER)
    assert table.caption == f"overall {overall.upper()} (aggregate of 22 checks): {counts}"
    assert sum(statuses.count(status) for status in ORDER) == 22


def test_a_failing_slice_reads_as_fail(sample):
    broken = copy.deepcopy(sample)
    index = resolve_time_slice(sample)[0]
    pressure = np.asarray(broken[f"equilibrium.time_slice.{index}.profiles_1d.pressure"], dtype=float).copy()
    pressure[3] = np.nan
    broken[f"equilibrium.time_slice.{index}.profiles_1d.pressure"] = pressure
    model = _quiet(vaft.omas.extract_equilibrium_table_validation, broken, time_slice=index)
    rows = _rows(model)
    row = rows["physical_validity.pressure_profile"]
    assert _status(row) == "fail"
    assert row[model_column("Note")].value == "profiles_1d.pressure has non-finite values"
    assert {key: _status(r) for key, r in rows.items()} == _quiet(_direct, broken, index)
    assert model.caption.startswith("overall FAIL")


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------


def test_text_markdown_and_html_are_deterministic(sample, table):
    again = _quiet(vaft.omas.plot_equilibrium_table_validation, sample)
    first = vaft.plot.equilibrium_table_validation(table)
    assert isinstance(again, TextView) and again.model == table
    for form in ("text", "markdown", "html"):
        assert getattr(again, form)() == getattr(first, form)()
    text = first.text()
    assert "NOT_AVAILABLE" in text and "INDETERMINATE" in text and table.caption in text
    html = first.html()
    shown = re.findall(r'data-status="([^"]*)"', html)
    assert shown == [_status(row) for row in table.rows]
    # Status is a token for a stylesheet, never a colour chosen here.
    assert "color" not in html and "background" not in html


@pytest.mark.parametrize(
    "keywords, refused",
    [
        ({"format": "screen"}, "format="),
        ({"theme": "minimal"}, "theme="),
        ({"backend": "plotly"}, "backend='plotly'"),
        ({"ax": object()}, "ax="),
    ],
)
def test_figure_keywords_are_refused(sample, keywords, refused):
    with pytest.raises(TypeError, match="returns text, not a figure") as raised:
        _quiet(vaft.omas.plot_equilibrium_table_validation, sample, **keywords)
    assert refused in str(raised.value)
    with pytest.raises(TypeError, match="draws nothing"):
        vaft.omas.extract_equilibrium_table_validation(sample, format="screen")


def test_time_snaps_to_a_stored_slice_as_the_summary_does(sample):
    model = _quiet(vaft.omas.extract_equilibrium_table_validation, sample, time=0.3195)
    assert model.title.startswith("Equilibrium validation #39915 — t = 319.00 ms (slice 4 of 9")
    assert "nearest stored slice to t = 319.50 ms" in model.title
    by_slice = _quiet(vaft.omas.extract_equilibrium_table_validation, sample, time_slice=3)
    assert by_slice.rows == model.rows
    assert {key: _status(row) for key, row in _rows(by_slice).items()} == _quiet(_direct, sample, 3)
    with pytest.raises(ValueError, match="either time= or time_slice="):
        _quiet(vaft.omas.extract_equilibrium_table_validation, sample, time=0.3, time_slice=1)


def test_no_equilibrium_is_refused_before_anything_is_built():
    import omas

    empty = omas.ODS(consistency_check=False)
    empty["magnetics.ip.0.time"] = np.array([0.0, 1.0])
    empty["magnetics.ip.0.data"] = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="no usable equilibrium slice|no time slices"):
        vaft.omas.plot_equilibrium_table_validation(empty)
    assert NAME not in {row.name for row in vaft.omas.available_plots(empty)}


def test_the_imas_adapter_reaches_the_same_verdicts(table):
    imas = pytest.importorskip("imas")
    import vaft.imas

    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    try:
        native = _quiet(vaft.imas.extract_equilibrium_table_validation, entry)
    finally:
        entry.close()
    assert [_status(row) for row in native.rows] == [_status(row) for row in table.rows]
