"""The cross-geometry MHD mode map (#1574): typed relations, no false one-to-one correspondences."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._equations import formula_equation
from vaft.diagram._mode_geometry import FAMILIES, RELATIONS
from vaft.formula.geometry import cylindrical_parallel_wavenumber, slab_parallel_wavenumber
from vaft.formula.stability import helical_phase


@pytest.fixture(scope="module")
def diagram():
    return vaft.diagram.mhd_mode_geometry_map()


def _boxes(d):
    """node -> (x0, x1, y0, y1) of its outline."""
    out = {}
    for name in d.model["nodes"]:
        (outline,) = [it for it in d.scene.role(f"node:{name}") if getattr(it, "closed", False)]
        xy = np.asarray(outline.points)
        out[name] = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    return out


def test_every_mode_sits_in_its_family_band_and_geometry_column(diagram):
    nodes, families, columns = diagram.model["nodes"], diagram.model["families"], diagram.model["columns"]
    for name, family in families.items():
        top, bottom, _ = FAMILIES[family]
        assert bottom < nodes[name][1] < top, (name, family)
    geometry = {"rayleigh_taylor": "slab", "slab_layer": "slab", "interchange": "cylinder", "sausage": "cylinder",
                "kink": "cylinder", "cylindrical_tearing": "cylinder", "rigid_shift": "cylinder"}
    for name, (x, _) in nodes.items():
        if name in ("slab", "cylinder", "torus"):
            continue
        if name in geometry:
            assert x == pytest.approx(columns[geometry[name]]), name
        else:  # everything else is toroidal
            lo, hi = columns["torus"]
            assert lo - 1e-9 <= x <= hi + 1e-9, name
    # the families named by the issue are all on the map
    for required in ("interchange", "mercier", "ballooning", "infernal", "peeling", "peeling_ballooning",
                     "sausage", "kink", "internal_kink", "external_kink", "rwm", "slab_layer",
                     "cylindrical_tearing", "toroidal_tearing", "ntm", "vde"):
        assert required in families, required


def test_relations_are_typed_and_drawn_in_their_own_style(diagram):
    used = {rel for _, _, rel in diagram.model["edges"]}
    assert used == set(RELATIONS)
    styles = {style for style, _ in RELATIONS.values()}
    assert len(styles) == len(RELATIONS)  # four relations, four arrows
    for start, end, rel in diagram.model["edges"]:
        (arrow,) = [it for it in diagram.scene.role(f"edge:{rel}:{start}->{end}") if hasattr(it, "start")]
        assert arrow.style == RELATIONS[rel][0], (start, end)


def test_no_false_one_to_one_correspondences(diagram):
    edges = {(a, b): rel for a, b, rel in diagram.model["edges"]}
    # coordinate headers are the only exact relabellings
    assert {k for k, rel in edges.items() if rel == "exact"} == {("cylinder", "slab"), ("torus", "cylinder")}
    # peeling, RWM and NTM add physics: branches, never limits of the linear family
    assert edges[("external_kink", "peeling")] == "branch"
    assert edges[("external_kink", "rwm")] == "branch"
    assert edges[("toroidal_tearing", "ntm")] == "branch"
    # Rayleigh-Taylor and the rigid shift are analogues only
    assert edges[("rayleigh_taylor", "interchange")] == "analogue"
    assert edges[("rigid_shift", "vde")] == "analogue"
    # the VDE is not in the kink genealogy, and the m = 0 sausage has no tokamak descendant
    assert [a for a, b in edges if b == "vde"] == ["rigid_shift"]
    assert not [b for a, b in edges if a == "sausage"]
    # interchange is a mechanism, not a mode number: no edge from the m = 0 / m = 1 morphology boxes into it
    assert not [a for a, b in edges if b == "interchange" and a in ("sausage", "kink")]


def test_header_relations_come_from_the_formula_catalog(diagram):
    for name, function in (("slab", slab_parallel_wavenumber), ("cylinder", cylindrical_parallel_wavenumber),
                           ("torus", helical_phase)):
        (label,) = diagram.scene.role(f"equation:{name}")
        assert formula_equation(function) in label.text


def _crosses(p0, p1, rect, samples=200):
    x0, x1, y0, y1 = rect
    t = np.linspace(0.0, 1.0, samples)[:, None]
    pts = np.asarray(p0) + t * (np.asarray(p1) - np.asarray(p0))
    inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    return bool(inside.any())


def test_no_arrow_runs_through_a_box_it_does_not_connect(diagram):
    boxes = _boxes(diagram)
    for start, end, rel in diagram.model["edges"]:
        (arrow,) = [it for it in diagram.scene.role(f"edge:{rel}:{start}->{end}") if hasattr(it, "start")]
        for name, rect in boxes.items():
            if name in (start, end):
                continue
            assert not _crosses(arrow.start, arrow.end, rect), (start, end, name)


def test_boxes_do_not_overlap(diagram):
    boxes = list(_boxes(diagram).items())
    for i, (a, ra) in enumerate(boxes):
        for b, rb in boxes[i + 1:]:
            apart = ra[1] <= rb[0] or rb[1] <= ra[0] or ra[3] <= rb[2] or rb[3] <= ra[2]
            assert apart, (a, b)


def test_labels_switch_off_the_legend_and_note():
    bare = vaft.diagram.mhd_mode_geometry_map(labels=False)
    assert not bare.scene.role("note")
    assert not bare.scene.role("legend:branch")
    with pytest.raises(ValueError):
        vaft.diagram.mhd_mode_geometry_map(labels="yes")
