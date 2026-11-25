"""Every summary preset declares what a row is and what each column measures.

The contract (grain, identity, time axis, one ColumnSpec per column) changes no
value and no column name. It travels with the table in ``attrs`` so that a
consumer -- an operating-space plot, a join -- reads the meaning instead of
guessing it from a suffix.
"""

import numpy as np
import pandas as pd
import pytest

import vaft
from vaft.database._summary import PRESETS
from vaft.database._summary_contract import (
    ColumnSpec,
    PresetContract,
    TimeAxis,
    check_preset_contract,
    known_quantities,
    unit_problem,
)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_declares_a_complete_contract(name):
    preset = PRESETS[name]
    assert preset.contract is not None, name
    assert preset.contract.name == name
    assert check_preset_contract(preset.contract, preset.columns, preset.key_columns) == []


@pytest.mark.parametrize("unit", ["", "-", "s", "kA", "m^3", "1e19 m^-3", "W m^-2", "% m T/MA", "MA m^-1 T^-1"])
def test_valid_units(unit):
    assert unit_problem(unit) is None


@pytest.mark.parametrize("unit", ["kA ", "1e19", "furlong", "m^0.5", "m**3"])
def test_invalid_units(unit):
    assert unit_problem(unit) is not None


def test_the_check_reports_each_broken_rule():
    contract = PresetContract(
        name="bad", version=1, grain="slab", identity=("shot", "nope"), time=None,
        columns=(ColumnSpec("shot", "", "shot"), ColumnSpec("shot", "", "again"),
                 ColumnSpec("beta", "%", "", quantity="normalized_beta"),
                 ColumnSpec("ghost", "-", "not produced")),
    )
    problems = "\n".join(check_preset_contract(contract, ("shot", "beta", "time_s"), ("shot", "eq_index")))
    for expected in ("more than once", "without a spec: ['time_s']", "does not produce: ['ghost']",
                     "grain 'slab'", "identity column 'nope'", "needs a time axis", "key column 'eq_index'",
                     "empty definition", "'normalized_beta' is in ['% m T/MA']"):
        assert expected in problems, expected


def test_a_declared_quantity_is_the_registered_quantity_in_its_unit():
    quantities = known_quantities()
    for preset in PRESETS.values():
        for spec in preset.contract.columns:
            if spec.quantity is not None:
                assert spec.unit in quantities[spec.quantity], (preset.contract.name, spec.name)


def test_a_quantity_registered_in_two_units_accepts_either():
    assert {"1e19 m^-3", "1e20 m^-3"} <= known_quantities()["line_average_density"]


def test_names_that_differ_from_the_dd_form_say_so():
    """beta_pol/li_3 are the DD forms; the B_pa-normalised forms are other quantities."""
    eq = PRESETS["equilibrium_global"].contract
    assert eq.column("li_3").quantity == "internal_inductance_li3"
    assert eq.column("virial_li_volume").quantity == "internal_inductance_li1"
    assert eq.column("beta_tor").quantity is None  # a fraction; the registered toroidal_beta is in percent
    assert "L_pol^2 R_0 / (2V)" in eq.column("beta_pol_circumference").definition


def test_overview_lists_each_preset_once():
    overview = vaft.database.summary_contract()
    assert "efit_reliability" not in set(overview["preset"])  # the deprecated alias
    assert set(overview["preset"]) == set(PRESETS) - {"efit_reliability"}
    shot = overview.set_index("preset").loc["shot_overview"]
    assert shot["grain"] == "shot" and shot["time"] == ""
    columns = vaft.database.summary_contract("equilibrium_global")
    assert list(columns["column"]) == list(PRESETS["equilibrium_global"].columns)


@pytest.fixture(scope="module")
def sample_table():
    return vaft.database.summary_from_ods({42962: vaft.omas.sample_ods(42962)}, preset="equilibrium_global")


def test_the_table_carries_its_contract_and_keeps_its_columns(sample_table):
    preset = PRESETS["equilibrium_global"]
    assert tuple(sample_table.columns) == preset.columns
    attrs = sample_table.attrs
    assert attrs["preset"] == "equilibrium_global" and attrs["grain"] == "slice"
    assert attrs["time_axis"]["column"] == "time_s" and attrs["skipped"] == []
    assert attrs["units"]["ip_kA"] == "kA" and attrs["units"]["beta_normal"] == "% m T/MA"
    assert attrs["quantities"]["li_3"] == "internal_inductance_li3"
    # definitions stay in the contract: attrs is deep-copied into every derived frame
    assert "definitions" not in attrs
    assert vaft.database.summary_contract(attrs["preset"])["definition"].str.len().gt(0).all()


def test_declared_quantities_hold_on_the_data(sample_table):
    """Recompute the normalised quantities from the table's own columns."""
    t = sample_table
    ip_ma = t["ip_kA"].abs() / 1e3
    b0 = t["vacuum_b0_T"].abs() * t["vacuum_r0_m"] / 0.4  # back to the reference radius (0.4 m here)
    a = t["minor_radius_m"]
    np.testing.assert_allclose(t["normalized_plasma_current"], ip_ma / (a * b0), rtol=1e-9)
    np.testing.assert_allclose(t["beta_normal"], 100 * t["beta_tor"] * a * b0 / ip_ma, rtol=1e-9)
    np.testing.assert_allclose(t["beta_normal_B2"], 100 * t["beta_volume_B2"] * a * b0 / ip_ma, rtol=1e-9)
    np.testing.assert_allclose(t["aspect_ratio"], t["major_radius_m"] / a, rtol=1e-12)
    # l_i(1) = l_i(3) L_pol^2 R_0 / (2V), and beta_pol_circumference / beta_pol is exactly that factor
    li1 = t["li_3"] * t["beta_pol_circumference"] / t["beta_pol"]
    close = np.isclose(t["virial_li_volume"], li1, rtol=0.02)
    assert close.sum() >= len(t) - 1, (t["virial_li_volume"] / li1).round(3).tolist()


def test_an_empty_table_carries_its_contract_too():
    table = vaft.database.summary_from_ods({}, preset="shot_overview")
    assert table.empty and table.attrs["grain"] == "shot" and table.attrs["skipped"] == []


def test_a_boundary_is_drawn_from_a_summary_table_without_renaming(sample_table):
    """beta_normal against li_3 is the beta_N-l_i(3) plane; no rename and no units= needed."""
    import matplotlib

    matplotlib.use("Agg")
    from vaft.plot.operational_space import operational_space_population, population_overlay

    plan = population_overlay(sample_table, "beta_n_li")
    assert "troyon" in [curve.key for curve in plan.curves], plan.omitted
    figure, ax = operational_space_population(sample_table, "beta_n_li")
    assert np.isfinite(ax.get_xlim()).all()
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_a_column_that_is_not_the_quantity_still_draws_nothing(sample_table):
    """beta_tor is a fraction, not the Troyon plane's percent toroidal beta: no boundary."""
    from vaft.plot.operational_space import population_overlay

    plan = population_overlay(sample_table, "troyon", y="beta_tor")
    assert not plan.curves
    reasons = " ".join(reason for _, reason in plan.omitted)
    assert "'beta_tor' is not the projection's 'toroidal_beta'" in reasons, reasons
    assert "normalized_current" not in reasons  # the x axis resolved from attrs


def test_column_lookup_by_declared_quantity():
    from vaft.plot.operational_space import _axis_mismatch, _column_for
    from vaft.formula import boundaries as fb

    table = pd.DataFrame({"a": [1.0], "b": [2.0], "elongation": [1.5]})
    assert _column_for(table, "normalized_beta") == "normalized_beta"  # no attrs: unchanged behaviour
    table.attrs["quantities"] = {"a": "normalized_beta", "b": "normalized_beta"}
    assert _column_for(table, "normalized_beta") == "normalized_beta"  # ambiguous: not guessed
    table.attrs["quantities"] = {"a": "normalized_beta", "elongation": "area_elongation"}
    assert _column_for(table, "normalized_beta") == "a"
    quantity = fb.BoundaryQuantity("elongation", "kappa", "-")
    # a column named like a quantity but declared as another one is refused
    assert _axis_mismatch("elongation", quantity, {"elongation": "-"}, table.attrs["quantities"])
    assert _axis_mismatch("elongation", quantity, {"elongation": "-"}, {}) is None


def test_the_contract_changes_no_value(sample_table):
    """The frame is the one the extractor builds: same values with the contract stripped."""
    from dataclasses import replace

    from vaft.database import _summary

    original = _summary.PRESETS["equilibrium_global"]
    _summary.PRESETS["equilibrium_global"] = replace(original, contract=None)
    try:
        bare = vaft.database.summary_from_ods({42962: vaft.omas.sample_ods(42962)}, preset="equilibrium_global")
    finally:
        _summary.PRESETS["equilibrium_global"] = original
    assert set(bare.attrs) == {"skipped"}
    pd.testing.assert_frame_equal(bare, sample_table, check_like=False)


def test_time_axis_defaults():
    assert TimeAxis().column == "time_s" and TimeAxis().tolerance_s is None


def test_the_database_guide_table_matches_the_declared_contracts():
    from pathlib import Path

    guide = (Path(__file__).resolve().parents[1] / "docs/_guide/Database.md").read_text(encoding="utf-8")
    block = guide.split("<!-- preset-contract-table", 1)[1].split("<!-- /preset-contract-table -->", 1)[0]
    rows = [line for line in block.splitlines() if line.startswith("| `")]
    documented = {line.split("|")[1].strip().strip("`"): line for line in rows}
    overview = vaft.database.summary_contract().set_index("preset")
    assert set(documented) == set(overview.index)
    for preset, line in documented.items():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        entry = overview.loc[preset]
        assert cells[1] == entry["grain"], preset
        assert cells[2] == ", ".join(f"`{name}`" for name in entry["identity"].split(", ")), preset
        assert cells[4] == str(entry["columns"]) and cells[5] == str(entry["with_quantity"]), preset
        expected_time = f"`{entry['time']}` (`{entry['time_reference']}`)" if entry["time"] else "—"
        assert cells[3] == expected_time, preset
