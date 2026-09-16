"""The table/namelist A/B (#695): its reading of the two files, and its answer.

The study runs EFIT; what is pinned here is the field comparison the design
rests on and the result it produced. The design argument was that everything
EFIT reads from `mhdin.dat` is the same in both files, so the namelist leg of
the 2x2 should be null -- and the null leg is what makes the table leg
attributable. Both halves are asserted, against the committed record.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TABLE = Path(__file__).resolve().parent / "data" / "efit_table_geometry_ab.json"
SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_tables" / "table_geometry_ab.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("table_geometry_ab", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except Exception:
        del sys.modules[spec.name]
        raise
    yield loaded
    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE.read_text(encoding="utf-8"))


def test_the_two_by_two_crosses_the_two_files(module):
    cases = module.CASES
    assert {name: (case["tables"], case["namelist"]) for name, case in cases.items()} == {
        "PP": ("packaged", "packaged"),
        "PF": ("packaged", "regenerated"),
        "FP": ("regenerated", "packaged"),
        "FF": ("regenerated", "regenerated"),
    }
    # PP is the #171 baseline's configuration and everything is measured
    # against it, so it must be the reference.
    assert module.REFERENCE_CASE == "PP"


def test_an_angle_written_a_full_turn_apart_is_not_a_difference(module, tmp_path):
    """The reading the whole design rests on.

    Every one of the 64 probe angles reads -270 in the packaged file and +90
    in the regenerated one. Calling that a difference would have made the
    namelist leg look substantive and the table result unattributable; calling
    a genuine change "formatting" would do the reverse. Both directions are
    tested.
    """
    import f90nml

    def write(path, **in3):
        f90nml.Namelist({"machinein": {"nfcoil": 16}, "in3": in3}).write(str(path), force=True)
        return path

    left = write(tmp_path / "left.dat", amp2=[-270.0, -270.0], rsi=[0.592, 0.792], hf=[0.1, 0.1])
    right = write(tmp_path / "right.dat", amp2=[90.0, 90.0], rsi=[0.592, 0.792], hf=[0.1, 0.2])

    report = module.namelist_difference(left, right)
    assert report["fields"]["amp2"]["kind"] == "whole_turns"
    assert "rsi" not in report["fields"], "an identical field is not a difference"
    assert report["fields"]["hf"]["kind"] == "value"
    assert report["substantive"] == ["hf"]

    # A field that grew is reported as a length change, which is what the coil
    # description does: 16 lumped conductors become 302 filaments.
    longer = write(tmp_path / "longer.dat", amp2=[90.0, 90.0], rsi=[0.592, 0.792], hf=[0.1, 0.1, 0.1])
    grown = module.namelist_difference(left, longer)
    assert grown["fields"]["hf"]["kind"] == "length"
    assert grown["fields"]["hf"]["packaged_size"] == 2
    assert grown["fields"]["hf"]["regenerated_size"] == 3


def test_a_float_that_differs_in_its_last_bit_is_formatting(module, tmp_path):
    import f90nml

    def write(path, values):
        f90nml.Namelist({"in3": {"zvs": values}}).write(str(path), force=True)
        return path

    left = write(tmp_path / "a.dat", [0.1, 0.2])
    right = write(tmp_path / "b.dat", [0.1 + 1.1e-16, 0.2])
    report = module.namelist_difference(left, right)
    assert report["fields"]["zvs"]["kind"] == "formatting"
    assert report["substantive"] == []


def test_what_actually_differs_between_the_two_files_is_the_coil_description(table):
    """The recorded reading, so the argument cannot drift from the files."""
    difference = table["namelist_difference"]
    assert difference["fields"]["amp2"]["kind"] == "whole_turns"
    for name in ("turnfc", "rsi", "zsi", "xmp2", "ymp2", "smp2"):
        assert name not in difference["fields"], f"{name} is identical and must stay so"
    # Everything substantive is the F-coil description or a label, and all of
    # it is an EFUND input that reaches EFIT only through the tables.
    assert set(difference["substantive"]) <= {
        "af", "af2", "fcid", "fcname", "fcturn", "hf", "lpname", "nfcoil", "rf", "rsisvs", "wf", "zf"
    }
    assert difference["fields"]["nfcoil"]["kind"] in {"value", "length"}


def test_swapping_the_namelist_alone_changes_nothing(table):
    """The null leg, which is what makes the table leg attributable.

    If this ever stops being null, the table result has to be re-derived: the
    two files would no longer differ only in what EFUND consumes.
    """
    for shot, block in table["shots"].items():
        packaged = block["cases"]["PP"]["summary"]
        swapped = block["cases"]["PF"]["summary"]
        assert swapped == packaged, shot

        both = block["cases"]["FF"]["summary"]
        tables_only = block["cases"]["FP"]["summary"]
        assert both == tables_only, shot

    changed = table["shots"]["39915"]["cases"]["PF"]["vs_reference"]
    assert changed["recovered"] == [] and changed["lost"] == []


def test_the_packaged_table_is_what_loses_the_separatrix(table):
    """#695's answer, and the population it accounts for.

    The #171 baseline attributed 17 slices to `findax` -- the separatrix point
    landing off grid. Regenerating the Green table takes that to 2 and leaves
    `bound` untouched, so the table explains that whole population and none of
    the collapse block.
    """
    counts = module_totals(table)
    assert counts["PP"]["produced"] == 31 and counts["PP"]["accepted"] == 18
    assert counts["FF"]["produced"] == 46 and counts["FF"]["accepted"] == 30
    assert counts["PF"] == counts["PP"] and counts["FP"] == counts["FF"]

    findax = {
        name: sum(block["cases"][name]["summary"]["findax_failures"] for block in table["shots"].values())
        for name in table["cases"]
    }
    bound = {
        name: sum(block["cases"][name]["summary"]["bound_failures"] for block in table["shots"].values())
        for name in table["cases"]
    }
    assert findax["PP"] == 17 and findax["FF"] == 2
    assert len(set(bound.values())) == 1, "the collapse block must not answer to the table"


def test_the_extra_yield_is_not_a_better_fit(table):
    """The limit on what this study can conclude.

    On every slice both tables reconstruct, EFIT reports the same chi-square;
    the boundary is what moves. So the regenerated table is not shown to be
    the more correct one here, and a later reading that treats the yield as
    proof of correctness fails this test.
    """
    for shot, block in table["shots"].items():
        metrics = block["cases"]["FP"]["vs_reference"]["metrics"]
        assert metrics["chisq"]["median_abs"] == 0.0, shot
        assert metrics["chisq"]["max_abs"] <= 1e-8, shot
        # The boundary does move, or there would be nothing to attribute.
        assert metrics["q95"]["median_relative"] > 0.01, shot


def module_totals(table):
    counts = {name: {"produced": 0, "accepted": 0, "slices": 0} for name in table["cases"]}
    for block in table["shots"].values():
        for name, record in block["cases"].items():
            summary = record["summary"]
            counts[name]["produced"] += summary["produced_an_equilibrium"]
            counts[name]["accepted"] += summary["accepted"]
            counts[name]["slices"] += summary["slices"]
    return counts


def test_the_totals_helper_agrees_with_the_script(module, table):
    assert module.totals(table) == module_totals(table)


def test_the_report_renders_from_the_committed_table(module, table):
    text = module.markdown(table)
    assert text.startswith("# The packaged Green table against a regenerated one")
    for name in table["cases"]:
        assert f"| {name} |" in text
    assert "What this says" in text
