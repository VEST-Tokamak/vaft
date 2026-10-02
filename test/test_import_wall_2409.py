"""The wall 2409 additions asset and its importer (issue #956).

The fifteen conductors wall 2409 adds are only geometry from VFIT; every
coupling entry is VAFT's own arithmetic. These tests hold the importer to
that: its method must reproduce the shipped 950-loop asset before it is
trusted, and the packaged additions must recompute from their own loops.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "workflow/em_coupling/import_wall_2409.py"
SPEC = importlib.util.spec_from_file_location("import_wall_2409", SCRIPT)
TOOL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOL)


def test_the_method_reproduces_the_shipped_950_loop_coupling():
    worst = TOOL.check_convention()
    assert set(worst) == {"passive_passive", "self_term", "passive_active_1906", "passive_active_2507"}
    assert max(worst.values()) < 1e-11


def test_the_packaged_additions_verify_without_the_vfit_checkout():
    assert TOOL.verify_asset() == []


def test_the_additions_are_fifteen_sus_loops_at_the_nominal_resistance():
    with np.load(TOOL.ASSET) as data:
        loops = json.loads(str(data["loops"]))
        provenance = json.loads(str(data["provenance"]))
    assert len(loops) == 15
    assert {loop["name"] for loop in loops} == {"W12"}
    assert {loop["resistivity"] for loop in loops} == {7.8e-7}
    r = sorted(float(np.mean(loop["element"][0]["geometry"]["outline"]["r"])) for loop in loops)
    z = {round(float(np.mean(loop["element"][0]["geometry"]["outline"]["z"])), 9) for loop in loops}
    # 20 mm pitch from 0.26 to 0.52 m, the innermost element trimmed to 0.24075 m.
    np.testing.assert_allclose(r, [0.24075, *np.linspace(0.26, 0.52, 14)], atol=1e-12)
    assert z == {-1.164}
    assert provenance["first_shot"] == 43017
    assert provenance["issue"].endswith("/956")
    # VFIT's own matrices were compared, not copied, and agreed within 1 %.
    assert max(provenance["vfit_matrix_agreement"].values()) < TOOL.DONOR_AGREEMENT


def test_a_corrupted_asset_is_reported(tmp_path):
    with np.load(TOOL.ASSET) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["mutual_passive_active_2507"] = arrays["mutual_passive_active_2507"] * 1.01
    corrupted = tmp_path / "corrupted.npz"
    np.savez(corrupted, **arrays)

    assert TOOL.verify_asset(corrupted) == [
        "mutual_passive_active_2507 does not recompute from the asset's own loops"
    ]
