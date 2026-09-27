"""Collision processes in fusion plasmas: a classification diagram (#1215).

Three branches, kept apart because they are different physics: Coulomb
collisions between charged particles (small-angle, cumulative relaxation),
atomic processes between charged particles and neutrals or bound electrons,
and nuclear fusion reactions. No collision frequency, Coulomb logarithm or
cross-section is drawn; those belong to formula-backed diagrams such as the
collisionality regimes of #1111.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from vaft.formula.constants import E_ALPHA, QE

from ._concept import Box, box, connector
from ._render import Diagram
from ._scene import Arrow, Label, Polyline, Scene

_BRANCH_Y = -2.4
_BRANCH_W, _BRANCH_H = 5.2, 1.5
_ITEM_W, _ITEM_H = 5.0, 0.95
_ITEM_STEP = 1.3
_BRANCH_X = {"coulomb": -6.4, "atomic": 0.0, "nuclear": 6.6}


def _alpha_energy_MeV() -> float:
    return E_ALPHA / QE / 1e6


#: the leaves of each branch: (role, LaTeX text)
def _leaves() -> dict:
    return {
        "atomic": [
            ("process:ionization", "ionization\\\\ $e + A \\to A^{+} + 2e$"),
            ("process:recombination", "recombination\\\\ $A^{+} + e \\to A + h\\nu$"),
            ("process:excitation", "excitation\\\\ $e + A \\to A^{*} + e,\\ A^{*} \\to A + h\\nu$"),
            ("process:charge_exchange", "charge exchange\\\\ $D^{+} + D^{0} \\to D^{0} + D^{+}$"),
        ],
        "nuclear": [
            ("reaction:dd", "D--D\\\\ $D + D \\to {}^{3}\\mathrm{He} + n$ or $T + p$"),
            ("reaction:dt", f"D--T\\\\ $D + T \\to {{}}^{{4}}\\mathrm{{He}}\\,({_alpha_energy_MeV():.1f}"
                            "\\,\\mathrm{MeV}) + n$"),
        ],
    }


def _spine(parent: Box, children: Sequence[Box], role: str) -> List:
    """A tree spine down the parent's left side with a short arrow into each child."""
    x = parent.x - 0.5 * parent.width + 0.3
    top = parent.y - 0.5 * parent.height
    bottom = children[-1].y
    items: List = [Polyline.of([(x, top), (x, bottom)], "connector line", role=role)]
    for child in children:
        items.append(Arrow((x, child.y), (child.x - 0.5 * child.width - 0.08, child.y), "connector", role=role))
    return items


def _check_labels(labels) -> bool:
    if not isinstance(labels, bool):
        raise ValueError(f"labels must be True or False, not {labels!r}")
    return labels


def collision_processes(*, labels: bool = True) -> Diagram:
    r"""Collision processes in fusion plasmas: Coulomb, atomic and nuclear.

    Coulomb collisions (e--e, e--i, i--i, impurity--main ion) act through
    many small-angle deflections and relax the distribution: pitch-angle
    scattering, momentum exchange and energy equilibration. Atomic processes
    (ionization, recombination, excitation, charge exchange) change charge
    states or bound electrons. Nuclear reactions (D--D, D--T) change nuclei;
    the D--T alpha energy is ``vaft.formula.constants.E_ALPHA``. A
    classification, not a quantitative comparison.
    """
    labels = _check_labels(labels)
    items: List = []
    root = box(0.0, 0.0, 9.0, 1.1, "\\textbf{Collision processes in fusion plasmas}", role="collision",
               latex=True)
    items += list(root.items)
    branch_text = {
        "coulomb": "\\textbf{Coulomb}\\\\ charged--charged",
        "atomic": "\\textbf{Atomic}\\\\ bound electrons and neutrals",
        "nuclear": "\\textbf{Nuclear}\\\\ fusion reactions",
    }
    branches = {}
    for key, x in _BRANCH_X.items():
        b = box(x, _BRANCH_Y, _BRANCH_W, _BRANCH_H, branch_text[key], role=f"collision:{key}", latex=True)
        branches[key] = b
        items += list(b.items)
        items.append(connector(root, b, role=f"collision:{key}"))
    # Coulomb: who collides, then what the collisions do
    species = box(_BRANCH_X["coulomb"], _BRANCH_Y - 2.3, _BRANCH_W, 1.5,
                  "$e$--$e$, \\ $e$--$i$, \\ $i$--$i$\\\\ impurity--main ion",
                  role="collision:coulomb:species", latex=True)
    effects = box(_BRANCH_X["coulomb"], _BRANCH_Y - 4.9, _BRANCH_W, 2.0,
                  "pitch-angle scattering\\\\ momentum exchange\\\\ energy equilibration",
                  role="collision:coulomb:effects", latex=True)
    items += list(species.items) + list(effects.items)
    items += [connector(branches["coulomb"], species, role="collision:coulomb:species"),
              connector(species, effects, role="collision:coulomb:effects")]
    # atomic and nuclear: a spine of leaves under each branch
    for key, leaves in _leaves().items():
        parent = branches[key]
        x = parent.x + 0.75
        children = []
        for i, (role, text) in enumerate(leaves):
            y = _BRANCH_Y - 0.5 * _BRANCH_H - 0.9 - i * _ITEM_STEP
            child = box(x, y, _ITEM_W, _ITEM_H, text, style="concept leaf", role=role, latex=True)
            children.append(child)
            items += list(child.items)
        items += _spine(parent, children, role=f"collision:{key}")
    if labels:
        items.append(Label((0.0, _BRANCH_Y - 6.6),
                           "Classification only: no collision frequencies, cross-sections or rates",
                           "note", anchor="north", role="note"))
    model = {"branches": tuple(_BRANCH_X), "leaves": {k: tuple(r for r, _ in v) for k, v in _leaves().items()},
             "alpha_energy_MeV": _alpha_energy_MeV()}
    return Diagram("collision_processes", Scene(tuple(items)), model=model)
