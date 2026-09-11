"""The domain and grid study (#459): its guards, and the result it recorded.

The study runs EFIT; what is pinned here is the bookkeeping it rests on and
the findings, so a later change that quietly reverses either fails loudly.
The guard that matters most is the table check: EFIT reads the computational
box out of the Green table and the file name records only the grid, so a case
run against a table built for a different box is undetectable from EFIT's own
output. That check is tested against a table it must refuse.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TABLE = Path(__file__).resolve().parent / "data" / "efit_domain_grid.json"
SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_numerics" / "domain_grid.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("domain_grid", SCRIPT)
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


def _table_directory(tmp_path, *, nw, nh, domain):
    directory = tmp_path / f"table_{nw}x{nh}"
    directory.mkdir()
    (directory / f"ec{nw}{nh}.ddd").write_bytes(b"")
    rleft, rright, zbotto, ztop = domain
    (directory / "efund_table_manifest.json").write_text(
        json.dumps(
            {
                "efund": {
                    "config": {
                        "grid": {
                            "nw": nw, "nh": nh,
                            "rleft": rleft, "rright": rright, "zbotto": zbotto, "ztop": ztop,
                        }
                    },
                    "config_sha256": "0" * 64,
                },
                "table": {"identity": "1" * 64},
                "generated_at": "2026-09-07T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_a_table_built_for_another_box_is_refused(module, tmp_path):
    """The check EFIT cannot make for itself.

    Two tables for the same grid and different domains are named identically,
    and EFIT takes `rgrid`/`zgrid` from whichever it opens. Running case B
    against case A's table would silently reconstruct a different machine and
    report nothing unusual, so the manifest is read here instead.
    """
    routine = _table_directory(tmp_path, nw=129, nh=129, domain=module.ROUTINE_DOMAIN)

    assert module.verify_table(routine, module.CASES["A"])["table"]["identity"]

    with pytest.raises(ValueError, match="box"):
        module.verify_table(routine, module.CASES["B"])
    with pytest.raises(ValueError, match="129x129"):
        module.verify_table(routine, module.CASES["C"])


def test_a_directory_without_a_manifest_is_not_a_table(module, tmp_path):
    """File names carry the grid and nothing else, so they are not evidence."""
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "ec129129.ddd").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="efund_table_manifest.json"):
        module.verify_table(bare, module.CASES["A"])


def test_an_incomplete_table_is_refused(module, tmp_path):
    directory = _table_directory(tmp_path, nw=129, nh=129, domain=module.ROUTINE_DOMAIN)
    (directory / "ec129129.ddd").unlink()
    with pytest.raises(FileNotFoundError, match="ec129129.ddd"):
        module.verify_table(directory, module.CASES["A"])


def test_the_cases_are_a_two_by_two_and_the_cells_follow_from_them(module):
    grids = {name: tuple(case["grid"]) for name, case in module.CASES.items()}
    domains = {name: tuple(case["domain"]) for name, case in module.CASES.items()}
    assert grids["A"] == grids["B"] and grids["C"] == grids["D"]
    assert domains["A"] == domains["C"] and domains["B"] == domains["D"]
    assert grids["A"] != grids["C"] and domains["A"] != domains["B"]

    # The routine box is 9.0 x 23.4 mm: 2.6 times taller than it is wide, which
    # is the anisotropy #459 asks about.
    width, height = module.cell_size(module.CASES["A"])
    assert width == pytest.approx((1.2 - 0.05) / 128 * 1e3)
    assert height / width == pytest.approx(2.61, abs=0.01)
    # Doubling nh roughly squares that away.
    assert module.cell_size(module.CASES["C"])[1] / width == pytest.approx(1.30, abs=0.01)

    # `rleft` is 0.05 m in every case. #459's text quotes the current box as
    # starting at R = 0, and it never has: the Green functions are singular on
    # the machine axis.
    assert all(domain[0] == 0.05 for domain in domains.values())


def test_a_ratio_is_withheld_when_there_is_nothing_to_be_relative_to(module):
    """`zm` sits at zero on an up-down symmetric machine.

    Dividing a 1 cm axis shift by a median of ~0 reports a change of millions
    of percent, which is arithmetic rather than physics. The absolute numbers
    are always reported; the ratio is only reported when it means something.
    """
    def slices(values, key):
        return [
            {
                "time_ms": 300 + index,
                "collapsed": False,
                "solver_errors": [],
                "afile": {"jflag": 1, key: value},
            }
            for index, value in enumerate(values)
        ]

    # A quantity that is exactly zero has no scale at all.
    flat = module.compare(slices([0.0, 0.0, 0.0], "zm"), slices([1.0, 1.0, 1.0], "zm"))
    assert flat["metrics"]["zm"]["median_relative"] is None
    assert flat["metrics"]["zm"]["median_abs"] == pytest.approx(1.0)

    # And the shape the real runs have: a median in the tens of microns beside
    # an excursion of a centimetre, which is a quantity centred on zero rather
    # than a quantity of size 3e-5.
    centred = module.compare(
        slices([3.7e-5, -2.1e-5, 1.2], "zm"), slices([1.2, 1.2, 1.2], "zm")
    )
    assert centred["metrics"]["zm"]["median_relative"] is None

    # Where there is a scale, the ratio is reported.
    offset = module.compare(slices([20.0, 21.0, 22.0], "rm"), slices([20.1, 21.1, 22.1], "rm"))
    assert offset["metrics"]["rm"]["median_relative"] == pytest.approx(0.1 / 21.0, rel=1e-6)


def test_the_phases_come_from_the_current_and_not_from_the_clock(module):
    from omas import ODS

    ods = ODS(consistency_check=False)
    current = [10.0, 60.0, 100.0, 95.0, 40.0]
    for index, value in enumerate(current):
        ods[f"equilibrium.time_slice.{index}.constraints.ip.measured"] = value * 1e3
    labels = module.phases(ods, np.asarray([0.300, 0.301, 0.302, 0.303, 0.304]))
    assert labels == {300: "ramp_up", 301: "ramp_up", 302: "flat_top", 303: "flat_top", 304: "ramp_down"}


def test_every_case_was_run_against_a_table_that_matches_it(table):
    """Without this the study is comparing four unknown configurations."""
    for name, case in table["cases"].items():
        assert case["table"]["table_identity"], name
        assert case["cell_mm"][0] > 0 and case["cell_mm"][1] > 0
    identities = {case["table"]["table_identity"] for case in table["cases"].values()}
    assert len(identities) == len(table["cases"]), "two cases share a table"
    # One acceptance bar across the 2x2, or the cases are judged differently.
    assert table["acceptance_envelope"]["aminor_min"] > 0


def test_neither_the_domain_nor_the_grid_recovers_a_slice(table):
    """#459's answer for EFIT, and the reason the losses go back to the solver.

    If a future change makes a smaller box or a finer grid recover the block,
    this fails and the conclusion has to be revisited -- which is what it is
    for.
    """
    totals = {name: {"produced": 0, "accepted": 0, "slices": 0} for name in table["cases"]}
    for block in table["shots"].values():
        for name, record in block["cases"].items():
            summary = record["summary"]
            totals[name]["produced"] += summary["produced_an_equilibrium"]
            totals[name]["accepted"] += summary["accepted"]
            totals[name]["slices"] += summary["slices"]

    assert totals["A"]["slices"] == 82
    for name in ("B", "C", "D"):
        assert totals[name]["slices"] == totals["A"]["slices"], name
        assert totals[name]["produced"] <= totals["A"]["produced"], name

    # The reduced domain is not neutral, it is worse, and it is worse in the
    # specific way the box predicts: a nearer grid edge, so `findax` starts
    # rejecting separatrix points for being off grid.
    assert totals["B"]["produced"] < totals["A"]["produced"] - 5
    edge_failures = {
        name: sum(block["cases"][name]["summary"]["findax_failures"] for block in table["shots"].values())
        for name in table["cases"]
    }
    assert edge_failures["B"] > edge_failures["A"]
    assert edge_failures["D"] > edge_failures["C"]


def test_halving_the_vertical_cell_does_not_move_the_reconstruction(table):
    """The resolution answer: under a percent on every global quantity.

    `terror` is excluded because it is the Grad-Shafranov residual rather than
    a physical quantity, and `zm` because it sits at zero -- its ratio is
    withheld by design and its absolute change is asserted instead.
    """
    physical = ("aminor", "area", "betap", "elong", "li", "q95", "qstar", "rm", "volume")
    for shot, block in table["shots"].items():
        metrics = (block["cases"]["C"].get("vs_reference") or {}).get("metrics", {})
        assert metrics, shot
        for name in physical:
            relative = metrics[name]["median_relative"]
            assert relative is not None and relative < 0.01, (shot, name, relative)
        # The fit itself does not move at all: EFIT reports the same chi-square.
        assert metrics["chisq"]["max_abs"] == 0.0, shot

    # Where the grid does show: the axis height is resolved to the Z cell. On
    # 39915 every compared slice moves by 1.172 cm, which is half of the
    # coarse cell (23.44 mm) to the digit, so this is quantisation and not a
    # different equilibrium.
    axis = table["shots"]["39915"]["cases"]["C"]["vs_reference"]["metrics"]["zm"]
    coarse_cell_mm = (1.5 - -1.5) / 128 * 1e3
    assert axis["median_abs"] == pytest.approx(coarse_cell_mm / 2 / 10.0, rel=1e-3)
    assert axis["median_abs"] == pytest.approx(axis["max_abs"], rel=1e-4)


def test_the_report_renders_from_the_committed_table(module, table):
    text = module.markdown(table)
    assert text.startswith("# The computational domain and the grid")
    for shot in table["shots"]:
        assert f"## {shot}" in text
    assert "What this says" in text
