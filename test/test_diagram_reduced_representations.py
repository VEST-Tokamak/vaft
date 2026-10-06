"""Reduced-representation diagrams (#1626): drawn from the catalog's Reduction metadata."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._reduced_representations import FAMILIES, relation_kind
from vaft.formula import catalog
from vaft.formula._taxonomy import REDUCTION_FAMILIES, Relation


def test_hierarchy_counts_come_from_the_catalog():
    counts = vaft.diagram.reduced_representation_hierarchy().model["counts"]
    tagged = [s for s in catalog.list_formulas() if s.reduction is not None]
    by_output = sum(v for (rep, _), v in counts.items() if rep in ("field_2d", "profile_1d", "scalar_0d"))
    assert by_output == sum(1 for s in tagged if s.reduction.output in ("field_2d", "profile_1d", "scalar_0d"))
    # a dimensionless profile exists: dimensionless is an axis of its own
    assert counts[("profile_1d", True)] > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_family_graph_edges_carry_the_catalogued_kind(family):
    d = vaft.diagram.reduction_graph(family)
    for source, target, kind, formula in d.model["edges"]:
        (branch,) = [it for it in d.scene.role(f"edge:{source}->{target}") if hasattr(it, "points")]
        if formula is not None:
            assert kind == catalog.describe(formula).reduction.kind
            assert branch.style == "connector"
        else:
            assert branch.style == "connector feedback"
    # layers increase along every edge
    layers = d.model["layers"]
    assert all(layers[t] > layers[s] for s, t, *_ in d.model["edges"])


@pytest.mark.parametrize("family", FAMILIES)
def test_family_graph_boxes_do_not_overlap_and_arrows_avoid_other_boxes(family):
    d = vaft.diagram.reduction_graph(family)
    boxes = {}
    for name in d.model["nodes"]:
        (outline,) = [it for it in d.scene.role(f"node:{name}") if getattr(it, "closed", False)]
        xy = np.asarray(outline.points)
        boxes[name] = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = boxes[a], boxes[b]
            assert ra[1] <= rb[0] or rb[1] <= ra[0] or ra[3] <= rb[2] or rb[3] <= ra[2], (a, b)
    t = np.linspace(0.0, 1.0, 200)[:, None]
    lines = [(f"edge:{s}->{tgt}", (s, tgt)) for s, tgt, *_ in d.model["edges"]]
    lines += [(f"bus:{s}", (s,)) for s in {e[0] for e in d.model["edges"]}]
    for role, joined in lines:
        for item in [it for it in d.scene.role(role) if hasattr(it, "points")]:
            xy = np.asarray(item.points)
            for p0, p1 in zip(xy[:-1], xy[1:]):
                pts = p0 + t * (p1 - p0)
                for name, (x0, x1, y0, y1) in boxes.items():
                    if name in joined:
                        continue
                    inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
                    assert not inside.any(), (family, role, name)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_arrow_of_a_source_starts_at_one_point_and_enters_its_target_centre(family):
    d = vaft.diagram.reduction_graph(family)
    nodes = d.model["nodes"]
    for source in {e[0] for e in d.model["edges"]}:
        starts = set()
        for item in [it for it in d.scene.role(f"bus:{source}") if hasattr(it, "points")][:1]:
            starts.add(tuple(np.round(item.points[0], 9)))
        if not starts:  # a single level target: the edge itself starts at the source
            (edge,) = [e for e in d.model["edges"] if e[0] == source]
            (branch,) = [it for it in d.scene.role(f"edge:{source}->{edge[1]}") if hasattr(it, "points")]
            starts.add(tuple(np.round(branch.points[0], 9)))
        (start,) = starts
        assert start[1] == pytest.approx(nodes[source][1])  # the right-side midpoint
    for source, target, *_ in d.model["edges"]:
        (branch,) = [it for it in d.scene.role(f"edge:{source}->{target}") if hasattr(it, "points")]
        end = branch.points[-1]
        assert end[1] == pytest.approx(nodes[target][1])  # enters at the left-side midpoint
        assert branch.points[0][1] == pytest.approx(end[1])  # the branch itself is horizontal


def test_a_relation_without_a_formula_or_kind_is_rejected():
    with pytest.raises(ValueError):
        relation_kind(Relation(("q",), "s_hat"))
    with pytest.raises(ValueError):
        relation_kind(Relation(("q",), "s_hat", "stability.helical_phase"))  # no Reduction section


@pytest.mark.parametrize("name, kwargs", [("reduced_representation_hierarchy", {}),
                                          ("reduction_graph", {"family": "current_q"})])
def test_deterministic_exported_and_labels(name, kwargs):
    fn = getattr(vaft.diagram, name)
    assert fn(**kwargs).tikz == fn(**kwargs).tikz
    assert name in vaft.diagram.__all__
    assert fn(**kwargs).scene.role("note") and not fn(**kwargs, labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(**kwargs, labels="yes")


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        vaft.diagram.reduction_graph("transport")
    assert set(FAMILIES) == set(REDUCTION_FAMILIES)


def test_layout_accepts_a_relation_with_sources_at_different_depths(monkeypatch):
    import vaft.diagram._reduced_representations as module
    families = dict(REDUCTION_FAMILIES)
    families["mixed"] = (Relation(("j_phi",), "q", kind="integral"),
                         Relation(("j_phi", "q"), "s_hat", kind="differential"))
    monkeypatch.setattr(module, "REDUCTION_FAMILIES", families)
    d = module.reduction_graph("mixed")
    assert d.model["layers"]["s_hat"] == 2
