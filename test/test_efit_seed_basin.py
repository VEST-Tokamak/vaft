"""The seed-basin study (#588): its bookkeeping, and the result it recorded.

The study runs EFIT; what is pinned here is the classification and comparison
it rests on, plus the finding, so a later change that quietly reverses either
fails loudly. The one trap this study actually hit — handing EFIT only the
profile orders and leaving the seed at its default, so every row of the sweep
came out identical — is pinned as its own test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TABLE = Path(__file__).resolve().parent / "data" / "efit_seed_basin.json"
SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_numerics" / "seed_basin.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("seed_basin", SCRIPT)
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


def _slice(*, collapsed=False, afile=None, solver=()):
    return {
        "time_ms": 320,
        "collapsed": collapsed,
        "afile": afile,
        "solver_errors": [{"routine": name, "detail": ""} for name in solver],
    }


def test_a_slice_is_classified_by_what_became_of_it(module):
    assert module.outcome(_slice(collapsed=True)) == "collapsed"
    assert module.outcome(_slice()) == "no_output"
    assert module.outcome(_slice(afile={"jflag": 0})) == "flagged"
    assert module.outcome(_slice(afile={"jflag": 1})) == "accepted"
    # A collapsed slice stays collapsed even if something wrote an a-file for
    # it: the collapse is the finding, not the file.
    assert module.outcome(_slice(collapsed=True, afile={"jflag": 1})) == "collapsed"


def test_the_comparison_names_what_moved_in_each_direction(module):
    routine = {307: "collapsed", 320: "no_output", 325: "accepted", 326: "flagged"}
    slices = [
        dict(_slice(collapsed=True), time_ms=307),
        dict(_slice(afile={"jflag": 1}), time_ms=320),
        dict(_slice(), time_ms=325),
        dict(_slice(afile={"jflag": 1}), time_ms=326),
    ]
    summary = module.summarize(slices, routine)
    assert summary["recovered"] == [320]
    assert summary["lost"] == [325]
    assert summary["changed_vs_routine"][326] == ["flagged", "accepted"]
    assert 307 not in summary["changed_vs_routine"]


def test_the_summary_counts_what_the_study_is_about(module):
    slices = [
        dict(_slice(collapsed=True, solver=("bound",)), time_ms=307),
        dict(_slice(solver=("findax",)), time_ms=320),
        dict(_slice(afile={"jflag": 1}), time_ms=325),
    ]
    summary = module.summarize(slices)
    assert summary["slices"] == 3
    assert summary["collapsed"] == 1
    assert summary["produced_an_equilibrium"] == 1
    assert summary["accepted"] == 1
    assert summary["bound_failures"] == 1 and summary["findax_failures"] == 1


def test_the_whole_configuration_reaches_efit_not_only_the_profile_orders(module):
    """The trap this study hit: a sweep that silently varies nothing.

    `EFITConfig` accepts `npprime`/`nffprime` as legacy shorthand. Passing
    only those leaves `initialization` at its default, so every row of a seed
    sweep emits the same `AELIP` and the study measures its own harness.
    """
    from vaft.code.efit.config import EFITScientificConfig, efit_parameter_grid

    scientific = efit_parameter_grid(
        EFITScientificConfig(), {"initialization.minor_radius": [0.1]}
    )[0]
    assert scientific.initialization.minor_radius == 0.1

    source = SCRIPT.read_text(encoding="utf-8")
    # The config handed to EFIT must carry the initialization, or the sweep is
    # a no-op. Assert on the call the study actually makes.
    assert "initialization=scientific.initialization" in source
    assert "npprime=scientific.profile.kppcur" not in source


def test_the_routine_row_reproduces_the_termination_baseline(table):
    """Without this the study is measuring its own harness, not the seed."""
    for shot, block in table["shots"].items():
        routine = block["routine"]
        assert routine["slices"] > 0
        assert routine["produced_an_equilibrium"] <= routine["slices"]
        assert routine["collapsed"] > 0, shot
    # 39915's baseline, as merged: 22 plasma slices, 7 producing an
    # equilibrium, 9 collapsed.
    anchor = table["shots"]["39915"]["routine"]
    assert anchor["slices"] == 22
    assert anchor["produced_an_equilibrium"] == 7
    assert anchor["collapsed"] == 9


def test_the_sweep_actually_varied_the_seed(table):
    """Every row differing from the routine row in *some* shot, or it is a no-op."""
    changed = sum(
        1
        for block in table["shots"].values()
        for row in block["sweep"]
        if row["summary"].get("recovered") or row["summary"].get("lost")
    )
    assert changed > 0, "no seed changed any outcome; the knob is not connected"
    assert table["routine_seed"] == {
        "ellipse_rzero": 0.4,
        "zzero": 0.0,
        "minor_radius": 0.3,
        "elongation": 1.6,
    }
    # The radial axis must be the seed centre alone. `rzero` also drives
    # RZERO, RCENTR and through it BTOR, so sweeping it would move the seed,
    # the normalisation and the vacuum toroidal field together.
    assert "initialization.ellipse_rzero" in table["axes"]
    assert "initialization.rzero" not in table["axes"]


def test_the_collapse_block_does_not_answer_to_the_seed(table):
    """#588's answer: the leading collapse is not an initialization problem.

    If a future change makes the seed recover the collapse block, this test
    fails and the conclusion handed to #459 must be revisited — which is
    exactly what it is here for.
    """
    total = recoverable = 0
    for shot, block in table["shots"].items():
        collapsed = sorted(
            int(item["time_ms"]) for item in block["routine_slices"] if item["collapsed"]
        )
        assert collapsed, shot
        recovered = sorted(
            {
                time
                for row in block["sweep"]
                for time in row["summary"].get("recovered", [])
                if time in collapsed
            }
        )
        total += len(collapsed)
        recoverable += len(recovered)
        # The structure, not a count: only slices at the trailing edge of the
        # block -- the ones adjacent to where the fit starts working -- ever
        # recover. The seed can move that boundary by a slice or two and
        # cannot touch the block itself.
        for time in recovered:
            position_from_end = len(collapsed) - 1 - collapsed.index(time)
            assert position_from_end <= 1, (shot, time, collapsed)
    assert recoverable / total < 0.2, (recoverable, total)


def test_the_seed_centre_moves_without_moving_the_toroidal_field():
    """The defect this study found in our own configuration.

    `rzero` is three namelist quantities at once. `ellipse_rzero` separates
    the seed from the rest, and defaults to following `rzero` so the routine
    k-file is unchanged.
    """
    from vaft.code.efit.config import EFITInitializationConfig

    routine = EFITInitializationConfig()
    assert routine.ellipse_rzero is None
    assert routine.seed_rzero == routine.rzero == 0.4

    moved = EFITInitializationConfig(ellipse_rzero=0.30)
    assert moved.seed_rzero == 0.30
    assert moved.rzero == 0.4, "the reference radius and BTOR must not move with the seed"

    with pytest.raises(ValueError):
        EFITInitializationConfig(ellipse_rzero=0.0)


def test_the_basin_is_asymmetric_about_the_routine_seed(table):
    """#588's answer: today's seed sits near the outboard edge.

    Inboard of 0.4 the sweep gains slices on every shot; outboard it loses
    them, badly. If that reverses, the recommendation has to be revisited.
    """
    inboard_lost = outboard_lost = 0
    for block in table["shots"].values():
        for row in block["sweep"]:
            if row["axis"] != "initialization.ellipse_rzero":
                continue
            lost = len(row["summary"].get("lost", []))
            if row["value"] < 0.4:
                inboard_lost += lost
            elif row["value"] > 0.4:
                outboard_lost += lost
    assert outboard_lost > inboard_lost, (inboard_lost, outboard_lost)


def test_the_report_renders_from_the_committed_table(module, table):
    text = module.markdown(table)
    assert text.startswith("# The first-slice seed")
    for shot in table["shots"]:
        assert f"## {shot}" in text
    assert "What this says" in text
