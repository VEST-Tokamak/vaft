"""vaft.diagram.side_by_side: several rendered diagrams in one SVG."""

import re

import pytest

import vaft.diagram


class _Fake:
    def __init__(self, name, width):
        self.name = name
        self.svg = (
            "<?xml version='1.0' encoding='UTF-8'?>\n"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}pt' height='50pt' viewBox='0 0 {width} 50'>"
            "<defs><path id='g0' d='M0 0L1 1'/><clipPath id='c1'><rect width='1' height='1'/></clipPath></defs>"
            "<use xlink:href='#g0' clip-path='url(#c1)'/></svg>"
        )


def test_panels_sit_side_by_side_with_their_ids_kept_apart():
    row = vaft.diagram.side_by_side(_Fake("a", 100), _Fake("b", 40), gap_pt=10)
    svg = row.svg
    assert svg.startswith("<svg") and "width='150.000pt'" in svg
    ids = re.findall(r"id='([^']+)'", svg)
    assert sorted(ids) == ["p0-c1", "p0-g0", "p1-c1", "p1-g0"]
    assert "href='#p0-g0'" in svg and "url(#p1-c1)" in svg and "#g0'" not in svg
    assert "x='110.000'" in svg   # second panel starts after the first and the gap
    assert row._repr_svg_() == svg


def test_side_by_side_needs_a_diagram_and_is_not_a_catalogued_builder():
    with pytest.raises(ValueError):
        vaft.diagram.side_by_side()
    assert "side_by_side" not in vaft.diagram.__all__
