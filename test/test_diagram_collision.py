"""collision_processes: three distinct branches and their leaves, checked by role (#1215)."""

import numpy as np
import pytest

import vaft.diagram
from vaft.formula.constants import E_ALPHA, QE

LEAVES = {
    "collision:atomic": ("process:ionization", "process:recombination", "process:excitation",
                         "process:charge_exchange"),
    "collision:nuclear": ("reaction:dd", "reaction:dt"),
    "collision:coulomb": ("collision:coulomb:species", "collision:coulomb:effects"),
}


def _box(diagram, role):
    outline = [it for it in diagram.scene.role(role) if getattr(it, "closed", False)]
    assert len(outline) == 1, role
    xs, ys = np.asarray(outline[0].points).T
    return xs.min(), xs.max(), ys.min(), ys.max()


def _text(diagram, role):
    return " ".join(it.text for it in diagram.scene.role(role) if hasattr(it, "text"))


def test_the_three_branches_hang_from_the_root_and_are_distinct():
    d = vaft.diagram.collision_processes()
    root = _box(d, "collision")
    boxes = [_box(d, branch) for branch in LEAVES]
    for x0, x1, y0, y1 in boxes:
        assert y1 < root[2]  # below the root
    spans = sorted((b[0], b[1]) for b in boxes)
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))  # side by side, not overlapping
    for branch in LEAVES:
        arrows = [it for it in d.scene.role(branch) if hasattr(it, "end")]
        assert any(a.start[1] <= root[2] + 1e-9 for a in arrows), branch  # an arrow leaves the root


@pytest.mark.parametrize("branch", LEAVES)
def test_every_leaf_sits_under_its_branch(branch):
    d = vaft.diagram.collision_processes()
    bx0, bx1, by0, _ = _box(d, branch)
    for leaf in LEAVES[branch]:
        x0, x1, y0, y1 = _box(d, leaf)
        assert y1 < by0, leaf
        assert bx0 - 0.1 < 0.5 * (x0 + x1) < bx1 + 1.0, leaf  # in the branch's column


def test_the_content_the_issue_asks_for():
    d = vaft.diagram.collision_processes()
    species = _text(d, "collision:coulomb:species")
    for pair in ("$e$--$e$", "$e$--$i$", "$i$--$i$", "impurity--main ion"):
        assert pair in species
    effects = _text(d, "collision:coulomb:effects")
    for effect in ("pitch-angle scattering", "momentum exchange", "energy equilibration"):
        assert effect in effects
    for role, word in (("process:ionization", "ionization"), ("process:recombination", "recombination"),
                       ("process:excitation", "excitation"), ("process:charge_exchange", "charge exchange"),
                       ("reaction:dd", "D--D"), ("reaction:dt", "D--T")):
        assert word in _text(d, role)


def test_the_alpha_energy_is_the_constant():
    d = vaft.diagram.collision_processes()
    assert f"{E_ALPHA / QE / 1e6:.1f}" in _text(d, "reaction:dt")
    assert d.model["alpha_energy_MeV"] == pytest.approx(3.5, abs=0.05)


def test_collision_processes_is_deterministic_exported_and_optional_note():
    assert vaft.diagram.collision_processes().tikz == vaft.diagram.collision_processes().tikz
    assert "collision_processes" in vaft.diagram.__all__
    assert vaft.diagram.collision_processes().scene.role("note")
    assert not vaft.diagram.collision_processes(labels=False).scene.role("note")
    with pytest.raises(ValueError, match="labels"):
        vaft.diagram.collision_processes(labels="no")
