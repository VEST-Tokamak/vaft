"""The ballooning-formulation hierarchy (#1637): reduction chain, implementations, shared normalisation."""

import pytest

import vaft.diagram
from vaft.diagram._ballooning_formulations import IMPLEMENTATIONS, REDUCTION_STEPS
from vaft.diagram._equations import formula_equation
from vaft.formula.equilibrium import ballooning_alpha_from_volume, shear_from_volume
from vaft.formula.stability import s_alpha_ballooning_stable


@pytest.fixture(scope="module")
def diagram():
    return vaft.diagram.ballooning_formulation_hierarchy()


def test_the_reduction_is_a_chain_from_the_general_equation_to_the_cht_model(diagram):
    edges = set(diagram.model["edges"])
    chain = ["general", *[k for k, _ in REDUCTION_STEPS], "vaft"]
    for a, b in zip(chain, chain[1:]):
        assert (a, b) in edges, (a, b)
    # the full-geometry codes come straight from the general equation, not through any reduction step
    for code in ("dcon", "gpec_jl"):
        assert ("general", code) in edges
        assert not any(b == code and a != "general" for a, b in edges)
    # every formulation reaches the comparison
    for code in IMPLEMENTATIONS:
        assert (code, "compare") in edges


def _outlines(d):
    import numpy as np
    out = {}
    for name in d.model["nodes"]:
        (outline,) = [it for it in d.scene.role(f"node:{name}") if getattr(it, "closed", False)]
        xy = np.asarray(outline.points)
        out[name] = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    return out


def test_boxes_do_not_overlap_and_no_arrow_crosses_a_box_it_does_not_join(diagram):
    import numpy as np
    boxes = _outlines(diagram)
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = boxes[a], boxes[b]
            assert ra[1] <= rb[0] or rb[1] <= ra[0] or ra[3] <= rb[2] or rb[3] <= ra[2], (a, b)
    t = np.linspace(0.0, 1.0, 200)[:, None]
    for start, end in diagram.model["edges"]:
        (arrow,) = [it for it in diagram.scene.role(f"edge:{start}->{end}") if hasattr(it, "start")]
        pts = np.asarray(arrow.start) + t * (np.asarray(arrow.end) - np.asarray(arrow.start))
        for name, (x0, x1, y0, y1) in boxes.items():
            if name in (start, end):
                continue
            inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
            assert not inside.any(), (start, end, name)


def test_equations_come_from_the_catalog(diagram):
    for role, fn in (("equation:cht", s_alpha_ballooning_stable), ("equation:shear", shear_from_volume),
                     ("equation:alpha", ballooning_alpha_from_volume)):
        texts = [it.text for it in diagram.scene.role(role)]
        assert any(formula_equation(fn) in t for t in texts), role


def test_deterministic_exported_and_labels_switch_off_the_note():
    fn = vaft.diagram.ballooning_formulation_hierarchy
    assert fn().tikz == fn().tikz
    assert "ballooning_formulation_hierarchy" in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(labels="yes")
