"""Plasma-model guide figures: read from the formalism records, not hand-listed."""

import pytest

import vaft.diagram
from vaft.diagram._plasma_models import EDGE_KINDS, EXTERNAL_CODES
from vaft.validation.model_contracts import ADAPTER_MODES

BUILDERS = ("plasma_model_reduction_map", "formalism_axes_map", "solver_placement_map", "kinetic_mhd_coupling",
            "reduced_model_ancestry")


@pytest.mark.parametrize("name", BUILDERS)
def test_deterministic_exported_and_labels(name):
    fn = getattr(vaft.diagram, name)
    assert name in vaft.diagram.__all__
    assert fn().tikz == fn().tikz
    assert fn().scene.role("note") or fn().scene.role("roles")
    with pytest.raises(ValueError):
        fn(labels="yes")


def _records():
    return {m.key: m.formalism for m in ADAPTER_MODES}


def test_reduction_map_nodes_are_computed_from_the_records():
    nodes = vaft.diagram.plasma_model_reduction_map().model["nodes"]
    records = _records()
    assert set(nodes["hybrid"]) == {k for k, f in records.items() if f is not None and f.bulk_description == "hybrid"}
    assert set(nodes["equilibrium"]) == {k for k, f in records.items()
                                         if f is not None and f.scientific_operation == "equilibrium"}
    assert "gacode:cgyro_nonlinear" in nodes["gyrokinetic"] and "gacode:neo" in nodes["drift_kinetic"]
    assert set(nodes["reduced_gyrokinetic"]) == {"gacode:tglf_linear", "gacode:tglf"}
    assert nodes["reduced_surrogate"] == ["gacode:tglf_nn"]
    # every classified mode lands in a node
    placed = {k for ids in nodes.values() for k in ids}
    assert {k for k, f in records.items() if f is not None} <= placed
    assert "gpec:rdcon_rmatch" in nodes["resistive_mhd"]


def test_axes_map_draws_each_run_through_its_record_values():
    from vaft.code.formalism import AXES

    d = vaft.diagram.formalism_axes_map()
    assert d.model["columns"] == tuple(AXES)
    records = _records()
    for key, values in d.model["runs"].items():
        assert values == tuple(getattr(records[key], f) for f in AXES)
    # nothing runs into the numerical-representation box
    (outline,) = [i for i in d.scene.role("extensions") if hasattr(i, "points")]
    xs = [p[0] for p in outline.points]
    for key in d.model["runs"]:
        for line in (i for i in d.scene.role(f"run:{key}") if hasattr(i, "points")):
            assert all(p[0] < min(xs) for p in line.points)


def test_placement_covers_every_mode_and_every_external_code():
    d = vaft.diagram.solver_placement_map()
    assert set(d.model["rows"]) == {m.key for m in ADAPTER_MODES}
    assert all(d.model["rows"][m.key] is None for m in ADAPTER_MODES if m.formalism is None)
    assert d.model["external"] == tuple(c.code for c in EXTERNAL_CODES)
    # external codes are not adapters
    names = {part.strip().lower() for c in EXTERNAL_CODES for part in c.code.split("/")}
    assert not names & {m.adapter for m in ADAPTER_MODES}
    # drawn hollow
    assert all("dashed" in i.style for c in EXTERNAL_CODES for i in d.scene.role(f"external:{c.code}")
               if hasattr(i, "points"))
    assert "EXTERNAL_CODES" in vaft.diagram.__all__ and vaft.diagram.EXTERNAL_CODES is EXTERNAL_CODES


def test_coupling_places_modes_by_their_kinetic_coupling():
    m = vaft.diagram.kinetic_mhd_coupling().model
    assert set(m["loop"]) == {"gpec:dcon_kinetic", "gpec:gpec_kinetic"}
    assert m["one_way"] == ("pentrc:ntv_closure",)
    assert set(m["bulk"]) == {"gpec:dcon_ideal", "gpec:gpec_ideal", "gpec:stride"}   # every ideal-only mode
    assert m["outside"] == ("gpec:rdcon_rmatch",)


def test_ancestry_rows_are_the_reduced_modes_and_the_surrogate_has_no_reduction_edge():
    d = vaft.diagram.reduced_model_ancestry()
    reduced = {m.key for m in ADAPTER_MODES if m.formalism is not None and m.formalism.bulk_description == "reduced"}
    assert set(d.model["rows"]) == reduced
    assert d.model["parents"]["gacode:neo_analytic"] == "drift_kinetic"
    solid = [a for a in d.scene.role("edge:asymptotic_reduction")]
    assert len(solid) == len(reduced) - 1               # TGLF-NN descends from TGLF, not by reduction
    assert len(d.scene.role("edge:surrogate_ancestry")) == 1


def test_edge_kinds_are_the_four_of_the_guide():
    assert EDGE_KINDS == ("asymptotic_reduction", "model_coupling", "software_dependency", "surrogate_ancestry")


@pytest.mark.parametrize("name", BUILDERS)
def test_built_and_using_only_the_guides_edge_kinds(name):
    from vaft.diagram.build import CANONICAL

    assert f"{name}.svg" in CANONICAL
    extra = {"edge:fluid_model", "edge:realization", "edge:closure", "edge:coefficient", "edge_label"}
    roles = {i.role for i in getattr(vaft.diagram, name)().scene.items if i.role.startswith("edge")}
    assert roles <= {f"edge:{k}" for k in EDGE_KINDS} | extra


def test_reduction_map_root_is_hollow():
    d = vaft.diagram.plasma_model_reduction_map()
    assert all("dashed" in i.style for i in d.scene.role("node:vlasov_maxwell") if hasattr(i, "points"))


def test_an_unplaced_coupling_or_mode_is_refused(monkeypatch):
    import dataclasses

    import vaft.validation.model_contracts as mc

    modes = list(mc.ADAPTER_MODES)
    dcon = next(m for m in modes if m.key == "gpec:dcon_kinetic")
    odd = dataclasses.replace(dcon, mode="pressure_coupled",
                              formalism=dataclasses.replace(dcon.formalism, kinetic_coupling="pressure"))
    monkeypatch.setattr(mc, "ADAPTER_MODES", modes + [odd])
    with pytest.raises(RuntimeError, match="kinetic_coupling"):
        vaft.diagram.kinetic_mhd_coupling()
