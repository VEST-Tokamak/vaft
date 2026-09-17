"""The coupling-asset repair tool (issue #373).

It repairs exactly one defect -- a material factor applied from one side of a
two-material cross block -- and refuses everything else, so an asset it has
written can be trusted to be that repair and nothing more.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "workflow/em_coupling/regenerate_passive_coupling.py"
SPEC = importlib.util.spec_from_file_location("regenerate_passive_coupling", SCRIPT)
TOOL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOL)

A = np.array([0, 1])
B = np.array([2, 3])


def _symmetric_4x4() -> np.ndarray:
    rng = np.random.default_rng(373)
    base = rng.uniform(1.0, 2.0, size=(4, 4))
    return (base + base.T) / 2.0


def _defective(factor: float = 1.04) -> tuple[np.ndarray, np.ndarray]:
    """A symmetric matrix and its one-sided-factor sibling: the B-side of the
    cross block divided by ``factor``, as the donor produced it."""
    clean = _symmetric_4x4()
    defective = clean.copy()
    defective[np.ix_(B, A)] = clean[np.ix_(A, B)].T / factor
    return clean, defective


def test_repair_reproduces_the_first_group_side_exactly():
    clean, defective = _defective()
    repaired, report = TOOL.repair_passive_passive(defective, A, B)
    np.testing.assert_array_equal(repaired, clean)
    assert report["material_factor"] == pytest.approx(1.04, abs=1e-12)
    assert report["output_asymmetry"] == 0.0
    assert report["input_asymmetry"] > 0.0
    assert report["cross_entries_changed"] == 2 * A.size * B.size


def test_repair_refuses_a_cross_ratio_that_is_not_one_constant():
    _clean, defective = _defective()
    defective[2, 0] *= 1.0 + 1e-9
    with pytest.raises(TOOL.RepairRefused, match="single constant"):
        TOOL.repair_passive_passive(defective, A, B)


def test_repair_refuses_within_block_asymmetry():
    _clean, defective = _defective()
    defective[0, 1] *= 1.01
    with pytest.raises(TOOL.RepairRefused, match="within-material"):
        TOOL.repair_passive_passive(defective, A, B)


def test_repair_refuses_groups_that_do_not_partition_the_loops():
    _clean, defective = _defective()
    with pytest.raises(TOOL.RepairRefused, match="partition"):
        TOOL.repair_passive_passive(defective, A, np.array([2]))
    with pytest.raises(TOOL.RepairRefused, match="non-finite"):
        bad = defective.copy()
        bad[1, 2] = np.nan
        TOOL.repair_passive_passive(bad, A, B)


def test_an_already_symmetric_matrix_is_not_repaired():
    clean = _symmetric_4x4()
    with pytest.raises(TOOL.AlreadySymmetric):
        TOOL.repair_passive_passive(clean, A, B)


def test_material_groups_reads_two_contiguous_materials_from_the_static_geometry():
    groups = TOOL.material_groups()
    assert len(groups) == 2
    sus, tungsten = (groups[key] for key in sorted(groups, reverse=True))
    assert sus.size == 720 and tungsten.size == 230
    assert sus.tolist() == list(range(0, 720)) and tungsten.tolist() == list(range(720, 950))


def test_verify_asset_accepts_the_packaged_asset():
    provenance = TOOL.verify_asset()
    assert provenance["passive_material_factor"] == 1.04
    assert provenance["convention"] == "sus_side"
    assert provenance["output_asymmetry"] == 0.0


def _synthetic_asset(path: Path, *, defective: bool) -> Path:
    """A five-key asset with the packaged shapes, tiny values, no provenance."""
    rng = np.random.default_rng(347)
    n = 950
    base = rng.uniform(1e-7, 1e-6, size=(n, n))
    pp = (base + base.T) / 2.0
    if defective:
        pp[720:, :720] /= 1.04
    aa = rng.uniform(1e-5, 1e-4, size=(10, 10))
    aa = (aa + aa.T) / 2.0
    arrays = {
        "mutual_active_active_1906": aa,
        "mutual_passive_active_1906": rng.uniform(1e-7, 1e-6, size=(n, 10)),
        "mutual_active_active_2507": aa * 1.1,
        "mutual_passive_active_2507": rng.uniform(1e-7, 1e-6, size=(n, 10)),
        "mutual_passive_passive": pp,
    }
    np.savez_compressed(path, **arrays)
    return path


def test_the_cli_repairs_writes_provenance_without_pickle_and_is_idempotent(tmp_path, capsys):
    asset = _synthetic_asset(tmp_path / "asset.npz", defective=True)
    assert TOOL.main(["--asset", str(asset), "--verify"]) == 2  # no provenance, asymmetric
    assert TOOL.main(["--asset", str(asset), "--repair", "--dry-run"]) == 0
    assert TOOL.read_provenance(asset) is None  # dry run wrote nothing

    assert TOOL.main(["--asset", str(asset), "--repair"]) == 0
    with np.load(asset, allow_pickle=False) as data:
        assert TOOL.PROVENANCE_KEY in data.files
        provenance = json.loads(str(data[TOOL.PROVENANCE_KEY][()]))
        pp = np.asarray(data["mutual_passive_passive"])
    assert np.array_equal(pp, pp.T)
    assert provenance["passive_material_factor"] == pytest.approx(1.04, abs=1e-12)
    assert provenance["groups"]["sus"]["n"] == 720 and provenance["groups"]["tungsten"]["n"] == 230
    assert provenance["source_sha256"] and provenance["static_geometry_sha256"]
    assert TOOL.main(["--asset", str(asset), "--verify"]) == 0

    before = asset.read_bytes()
    assert TOOL.main(["--asset", str(asset), "--repair"]) == 0
    assert "already exactly reciprocal" in capsys.readouterr().out
    assert asset.read_bytes() == before


def test_the_cli_refuses_an_asset_it_cannot_explain(tmp_path, capsys):
    asset = _synthetic_asset(tmp_path / "asset.npz", defective=True)
    with np.load(asset) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    arrays["mutual_passive_passive"][5, 7] *= 1.5  # within-block damage
    np.savez_compressed(asset, **arrays)
    assert TOOL.main(["--asset", str(asset), "--repair"]) == 2
    assert "refused" in capsys.readouterr().err
    assert TOOL.read_provenance(asset) is None
