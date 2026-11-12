"""The simplified input-output view of stability and transport (``stability_transport_io``).

It restates nothing: its codes, output IDS and assumptions come from the
cross-domain architecture figure, and these tests hold the two together.
"""
import pytest

import vaft.diagram
from vaft.diagram import _cross_domain_concept as C
from vaft.diagram import _stability_transport_io as S
from vaft.diagram import build
from vaft.diagram._scene import Arrow, Label


def _text(d, role):
    return " ".join(i.text for i in d.scene.role(role) if isinstance(i, Label))


def test_outputs_are_the_ids_the_architecture_figure_writes():
    d = vaft.diagram.stability_transport_io()
    assert d.model["outputs"] == {"mhd_stability": ("mhd_linear", "ntms"),
                                  "transport": ("core_transport", "core_profiles", "gyrokinetics_local")}
    for key, section in (("mhd_stability", "mhd"), ("transport", "transport")):
        assert d.model["outputs"][key] == tuple(ids for ids, _, _ in C.IMAS[section])
        for ids in d.model["outputs"][key]:
            assert ids.replace("_", "\\_") in _text(d, f"output:{key}")


def test_codes_are_the_architecture_figure_operations():
    d = vaft.diagram.stability_transport_io()
    assert d.model["codes"]["mhd_stability"] == ("DCON", "RDCON", "GPEC")
    assert d.model["codes"]["transport"] == ("Braginskii", "Sauter, Redl", "NEO", "TGLF", "CGYRO")
    assert len(d.model["codes"]["mhd_stability"]) == len(C.OPERATIONS["mhd"])
    assert len(d.model["codes"]["transport"]) == len(C.OPERATIONS["transport"])
    for key, codes in d.model["codes"].items():
        for code in codes:
            assert code in _text(d, f"node:{key}")


def test_shared_inputs_and_what_each_calculation_reads():
    d = vaft.diagram.stability_transport_io()
    assert d.model["inputs"] == ("equilibrium", "core_profiles", "assumptions")
    assert set(d.model["reads"]["transport"]) == set(d.model["inputs"])
    # measured profiles feed transport, not stability: MHD stability takes its pressure from the equilibrium
    assert "core_profiles" not in d.model["reads"]["mhd_stability"]
    assert "equilibrium" in d.model["reads"]["mhd_stability"]
    for line in S.ASSUMPTIONS:
        assert line in _text(d, "input:assumptions")
    # one arrow from the input group to each calculation, and one from each to its outputs
    arrows = {i.role for i in d.scene.items if isinstance(i, Arrow)}
    assert {"edge:inputs->mhd_stability", "edge:inputs->transport",
            "edge:mhd_stability->output:mhd_stability", "edge:transport->output:transport"} <= arrows


def test_labels_flag_and_registration():
    bare = vaft.diagram.stability_transport_io(labels=False)
    assert not any(isinstance(i, Label) for i in bare.scene.items if i.role.startswith("group"))
    # no annotations beside the arrows: what each calculation reads lives in the model
    full = vaft.diagram.stability_transport_io()
    assert not any(i.role.startswith(("reads:", "writes:")) for i in full.scene.items)
    assert "IMAS Stability IDS" in _text(full, "output:mhd_stability")
    assert "IMAS Transport IDS" in _text(full, "output:transport")
    assert "MHD Stability Model" in _text(full, "node:mhd_stability")
    assert "Transport Model" in _text(full, "node:transport")
    assert any("Shared Plasma State" in i.text for i in full.scene.role("group:inputs") if isinstance(i, Label))
    with pytest.raises(ValueError):
        vaft.diagram.stability_transport_io(labels="yes")
    assert build.CANONICAL["stability_transport_io.svg"] == ("stability_transport_io", {})
    assert "stability_transport_io" in vaft.diagram.__all__


def test_the_combined_figure_pairs_the_equilibrium_with_the_io_view():
    d = vaft.diagram.equilibrium_stability_transport()
    a = vaft.diagram.integrated_scientific_framework("equilibrium", frame=False, analysis=False, references=False,
                                                     ids_key=True, boundary_models=True)
    b = vaft.diagram.stability_transport_io()
    # the left half is drawn as is, the hub naming its IDS key; the right is the same drawing moved right, at the same scale (one font size)
    for role in ("node:hub", "node:experiment"):
        assert d.scene.role(role) == a.scene.role(role)
    assert not d.scene.role("references")
    # code names look like the equilibrium routes: the same size, upright and black
    assert d.model["equilibrium"]["activities"] == ("experiment", "theory", "fixed_boundary", "free_boundary")
    assert "Reconstruction" in d.scene.role("node:experiment")[1].text
    assert "{\\small CHEASE}" in d.scene.role("node:fixed_boundary")[1].text
    assert "{\\small TokaMaker, TES}" in d.scene.role("node:free_boundary")[1].text
    assert not d.scene.role("node:data_driven") and not d.scene.role("node:modelling")
    assert "{\\small DCON, RDCON, GPEC}" in d.scene.role("node:mhd_stability")[1].text
    assert "color" not in d.scene.role("node:transport")[1].text
    moved = d.scene.role("node:transport")[0].points
    original = b.scene.role("node:transport")[0].points
    dx = {round(m[0] - o[0], 9) for m, o in zip(moved, original)}
    assert len(dx) == 1 and dx.pop() > 0 and all(m[1] == o[1] for m, o in zip(moved, original))
    # nothing of b overlaps a
    a_right = max(p[0] for i in a.scene.items if hasattr(i, "points") for p in i.points)
    b_left = min(p[0] for i in d.scene.items if hasattr(i, "points") and i.role in
                 {r.role for r in b.scene.items} for p in i.points)
    assert a_right < b_left
    # the equilibrium is handed from a to b, and the panels are lettered
    assert d.scene.role("edge:hub->equilibrium") and ("hub", "equilibrium", "forward") in d.model["edges"]
    assert "\\ttfamily equilibrium" in d.scene.role("node:hub")[1].text
    assert not any(i.role.startswith("panel:") for i in d.scene.items)  # one flow, no panel letters
    # no analysis node: the downstream use of the equilibrium is panel b
    assert not d.scene.role("node:analysis") and not d.scene.role("node:simulation_input")
    assert build.CANONICAL["equilibrium_stability_transport.svg"] == ("equilibrium_stability_transport", {})


def test_the_measurement_variant_feeds_the_reconstruction_and_the_profiles():
    d = vaft.diagram.equilibrium_stability_transport(measurements=True)
    edges = set(d.model["edges"])
    # the reconstruction only writes the equilibrium; the other routes still exchange with it
    assert ("experiment", "hub", "forward") in edges and ("theory", "hub", "both") in edges
    assert {("measurement", "experiment", "forward"), ("measurement", "core_profiles", "forward"),
            ("equilibrium", "core_profiles", "forward"), ("assumptions", "core_profiles", "forward")} <= edges
    assert d.model["measurements"] == ("wall", "pf_active", "pf_passive", "em_coupling",
                                       "magnetics", "thomson_scattering", "charge_exchange")
    card = next(i for i in d.scene.role("node:measurement") if isinstance(i, Label)).text
    assert "\\textbf{Experiment}" in card and "thomson\\_scattering" in card
    for role in ("edge:measurement->experiment", "edge:measurement->core_profiles",
                 "edge:equilibrium->core_profiles", "edge:assumptions->core_profiles"):
        assert d.scene.role(role)
    # the default figure is unchanged: no measurement card, the reconstruction exchanges both ways
    plain = vaft.diagram.equilibrium_stability_transport()
    assert not plain.scene.role("node:measurement") and plain.model["measurements"] == ()
    assert ("experiment", "hub", "both") in plain.model["edges"]
    with pytest.raises(ValueError):
        vaft.diagram.equilibrium_stability_transport(measurements="yes")
    with pytest.raises(ValueError):
        vaft.diagram.stability_transport_io(derived_profiles=1)
    with pytest.raises(ValueError, match="one_way"):
        vaft.diagram.integrated_scientific_framework("equilibrium", frame=False, analysis=False, references=False,
                                                     boundary_models=True, one_way=("modelling",))
