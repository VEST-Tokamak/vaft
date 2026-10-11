"""Alternative TGLF configurations of one state coexist (#1900, item 2).

Each SAT/field configuration is persisted as its own *transitional* HSDS source,
``{family root}/tglf-sat<n>-<field>``, until the #1128/#1129 occurrence model
replaces it. These tests pin the catalog, the destination rule, the migration
mapping to the future ``ids_properties.name``, and -- the acceptance criterion --
that two configurations built for one state and saved side by side read back
unchanged rather than the second overwriting the first.

No server is touched: products go to a temporary FileDB only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from vaft.database import sources
from vaft.database.filedb import FileDB, FileDBPathError, OMASStage, stage_lineage

DATA = Path(__file__).parent / "data" / "gacode"
SHOT = 39915

#: A gyro-Bohm normalisation that is internally consistent:
#: Gamma_GB = n_e c_s (rho_s/a)^2 with n_e = 1e19, and Q_GB = Gamma_GB T_e at 20 eV.
_A, _RHO_S, _C_S, _NE, _TE = 0.3, 0.005, 1.0e5, 1.0e19, 20 * 1.602176634e-19
_GAMMA_GB = _NE * _C_S * (_RHO_S / _A) ** 2
_Q_GB = _GAMMA_GB * _TE
_NORMALISATION = {"a_m": _A, "b_unit_T": 0.1, "rho_s_m": _RHO_S, "c_s_m_s": _C_S,
                  "q_gb_W_m2": _Q_GB, "gamma_gb_m2_s": _GAMMA_GB}
_RHO_TOR_NORM = 0.65


def _parameters(configuration: str) -> dict:
    sat, field = sources.parse_tglf_configuration(configuration)
    return {"sat_rule": sat, "use_bper": field != "es", "use_bpar": field == "em-bper-bpar"}


def _tree(root: Path, fixture: str, *, label: str | None = None, lineage: str = "magnetics",
          quality: str = "good", time_ms: int = 316, parameters: dict | None = None,
          extra_unsolved: bool = False, state_identity: str = "state-39915-316",
          ti_lineage: str = "ti_eq_te_assumed", revision: str = "b493397",
          projection_scale: float = 1.0, project_fluxes: bool = True) -> Path:
    """A routine-TGLF state tree built around a committed native TGLF run."""
    from vaft.code.gacode.tglf.runner import read_tglf_case

    configuration = f"tglf-{fixture}"
    label = label or configuration
    state_dir = root / label / str(SHOT) / lineage / f"{time_ms:05d}"
    workdir = state_dir / "r0.70"
    shutil.copytree(DATA / f"tglf_vest_39915_r0.70_{fixture}", workdir)
    native = read_tglf_case(workdir)
    native.write_json(workdir / "outputs.json")
    surfaces = [{"r_over_a": 0.7, "status": "solved", "run_identity": f"run-{fixture}",
                 "local": {"species": ["e", "H", "C"], "vexb_shear": None,
                           "normalisation": _NORMALISATION}}]
    if extra_unsolved:
        surfaces.append({"r_over_a": 0.5, "status": "failed", "runtime_status": "timeout"})
    energy = projection_scale * np.asarray(native.gbflux["energy"]) * _Q_GB
    particle = projection_scale * np.asarray(native.gbflux["particle"]) * _GAMMA_GB
    row = {"r_over_a": 0.7, "rho_tor_norm": _RHO_TOR_NORM}
    if project_fluxes:
        row.update({
            "electron_energy_flux_W_m2": float(energy[0]),
            "electron_particle_flux_m2_s": float(particle[0]),
            "ion_energy_flux_W_m2": {"H": float(energy[1]), "C": float(energy[2])},
            "ion_particle_flux_m2_s": {"H": float(particle[1]), "C": float(particle[2])},
        })
    state = {
        "shot": SHOT, "time_efit_s": time_ms / 1000.0, "efit_lineage": lineage,
        "efit_quality": quality, "ti_lineage": ti_lineage,
        "state_identity": state_identity, "status": "partial" if extra_unsolved else "solved",
        "tglf_config": label, "tglf_parameters": parameters or _parameters(configuration),
        "gacode_revision": revision, "surfaces": surfaces,
        "core_transport": {"surfaces": [row]},
    }
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return state_dir


def _build(state_dir):
    from vaft.omas.vest_upstream import build_turbulent_transport_ods

    return build_turbulent_transport_ods(state_directories=state_dir)


# --------------------------------------------------------------------------- catalog


def test_the_catalog_holds_36_transitional_sparse_writable_sources():
    transitional = [s for s in sources.known_sources(environment={}) if s.transitional_until]
    assert len(transitional) == 3 * 4 * 3 == 36
    assert {s.name for s in transitional} == {
        f"{root}/{config}"
        for root in sources.FAMILY_SOURCES.values()
        for config in sources.tglf_configurations()
    }
    for entry in transitional:
        assert entry.sparse and entry.writable, entry.name
        assert entry.transitional_until == sources.TRANSITIONAL_UNTIL == "#1128/#1129"
        assert entry.purpose.startswith("TRANSITIONAL")
        assert entry.parent in sources.CATALOG and entry.parent == entry.derived_parent
        assert sources.resolve(entry.name, writable=True) == entry.name
        # The purpose names the configuration and the occurrence it migrates to.
        family = next(f for f, root in sources.FAMILY_SOURCES.items() if root == entry.parent)
        config = entry.name.split("/")[-1]
        assert sources.turbulent_transport_occurrence_name(family, config) in entry.purpose
    # Nothing that predates #1900 is marked transitional.
    assert not sources.describe("main").transitional_until


def test_the_configuration_grammar_is_the_driver_s():
    assert len(sources.tglf_configurations()) == 12
    assert sources.tglf_configuration(3, "em-bper") == "tglf-sat3-em-bper"
    assert sources.parse_tglf_configuration("tglf-sat2-em-bper-bpar") == (2, "em-bper-bpar")
    for bad in ("tglf-sat4-es", "tglf-sat3-em", "sat3-es", "tglf-sat3-es/x", None, 3):
        with pytest.raises(sources.HSDSSourceError):
            sources.parse_tglf_configuration(bad)
    with pytest.raises(sources.HSDSSourceError):
        sources.tglf_configuration(True, "es")


# --------------------------------------------------------------------------- destination


def test_two_configurations_of_one_state_go_to_distinct_sources():
    one = sources.replication_for_stage(
        "turbulent_transport", family="magnetic", product="tglf-sat3-em-bper")
    two = sources.replication_for_stage(
        "turbulent_transport", family="magnetic", product="tglf-sat2-es")
    assert one.source == "main/tglf-sat3-em-bper"
    assert two.source == "main/tglf-sat2-es"
    assert one.ids == two.ids == ("core_transport",)
    assert one.optional and one.produced_by == "corrective"
    # Never re-owns the state it ran on.
    assert not {"core_profiles", "equilibrium"} & set(one.ids)
    # A per-configuration stage has no single destination to advertise.
    assert "turbulent_transport" not in sources.STAGE_SOURCE


def test_a_missing_or_invalid_configuration_is_refused():
    with pytest.raises(sources.HSDSSourceError, match="needs product=.*tglf-sat3-em-bper"):
        sources.replication_for_stage("turbulent_transport", family="magnetic")
    for bad in ("dcon-peeling", "tglf-sat9-es", "tglf-sat3-em"):
        with pytest.raises(sources.HSDSSourceError, match="not a TGLF configuration"):
            sources.replication_for_stage("turbulent_transport", product=bad)


def test_electron_and_magnetic_lineages_publish_beneath_different_roots():
    assert sources.family_for_atlas_lineage("magnetics") == "magnetic"
    assert sources.family_for_atlas_lineage("electron_kinetic") == "electron"
    with pytest.raises(sources.HSDSSourceError):
        sources.family_for_atlas_lineage("kinetic")
    magnetic = sources.source_for_stage("turbulent_transport", family="magnetic",
                                        product="tglf-sat2-es")
    electron = sources.source_for_stage("turbulent_transport", family="electron",
                                        product="tglf-sat2-es")
    kinetic = sources.source_for_stage("turbulent_transport", family="kinetic",
                                       product="tglf-sat2-es")
    assert (magnetic, electron, kinetic) == (
        "main/tglf-sat2-es", "electron-efit/tglf-sat2-es", "kinetic-efit/tglf-sat2-es")


def test_the_filedb_path_carries_family_and_configuration():
    db = FileDB("/nonexistent/filedb")
    lineage = stage_lineage("turbulent_transport", family="electron", refinement="chease",
                            product="tglf-sat3-em-bper")
    assert lineage == {"family": "electron", "product": "tglf-sat3-em-bper"}
    path = db.omas_product(OMASStage.TURBULENT_TRANSPORT, shot=SHOT, **lineage)
    assert path.relative_to(db.root).as_posix() == (
        "omas/turbulent_transport/electron/tglf-sat3-em-bper/39915/output/"
        "turbulent_transport.json.gz")
    with pytest.raises(FileDBPathError):
        stage_lineage("turbulent_transport", family="magnetic")
    with pytest.raises(FileDBPathError, match="not a TGLF configuration"):
        db.omas("turbulent_transport", shot=SHOT, family="magnetic", product="rdcon")
    with pytest.raises(FileDBPathError, match="refinement"):
        db.omas("turbulent_transport", shot=SHOT, family="magnetic", refinement="chease",
                product="tglf-sat2-es")


# --------------------------------------------------------------------------- migration


def test_the_migration_mapping_names_the_future_occurrence_without_an_integer():
    assert (sources.turbulent_transport_occurrence_name("magnetic", "tglf-sat3-em-bper")
            == "tglf-sat3-em-bper-from-magnetic-efit")
    assert (sources.turbulent_transport_occurrence_name("electron", "tglf-sat0-es")
            == "tglf-sat0-es-from-electron-efit")
    rows = sources.turbulent_transport_migration()
    assert len(rows) == 36
    assert {row.source for row in rows} == {
        s.name for s in sources.known_sources(environment={}) if s.transitional_until}
    assert len({row.name for row in rows}) == 36
    for row in rows:
        assert row.ids == "core_transport"
        assert row.occurrence is None and "#1129" in row.allocated_by


def test_the_migration_table_is_documented():
    doc = (Path(__file__).parents[1] / "docs" / "_guide" / "Database.md")
    if not doc.exists():
        pytest.skip("docs are not part of the distribution")
    text = doc.read_text(encoding="utf-8")
    for root in sources.FAMILY_SOURCES.values():
        assert f"`{root}/tglf-sat<n>-<field>`" in text
    for lineage in sources.FAMILY_LINEAGE_NAMES.values():
        assert f"`tglf-sat<n>-<field>-from-{lineage}`" in text
    assert "#1129" in text


# --------------------------------------------------------------------------- product


def test_the_product_carries_its_future_name_and_upstream_identity(tmp_path):
    ods, manifest, native = _build(_tree(tmp_path, "sat2-em-bper", extra_unsolved=True))
    name = sources.turbulent_transport_occurrence_name("magnetic", "tglf-sat2-em-bper")
    assert ods["core_transport.ids_properties.name"] == name == manifest["occurrence_name"]
    assert list(ods["core_transport.ids_properties.provenance.node.0.sources"]) == [
        "equilibrium/magnetic-efit"]
    assert manifest["transitional_source"] == "main/tglf-sat2-em-bper"
    assert manifest["status"] == "partial"           # r/a = 0.5 failed and is not mapped
    assert "core_profiles" not in ods and "equilibrium" not in ods
    parameters = json.loads(ods["core_transport.model.0.code.parameters"])
    assert parameters["occurrence_name"] == name
    (state,) = parameters["states"]
    assert state["state_identity"] == "state-39915-316"
    assert state["gacode_revision"] == "b493397"
    assert [s["run_identity"] for s in state["surfaces"]] == ["run-sat2-em-bper"]
    assert parameters["tglf_parameters"] == {"sat_rule": 2, "use_bper": True, "use_bpar": False}
    # Only the solved surface is mapped, at the driver's rho_tor_norm.
    base = "core_transport.model.0.profiles_1d.0"
    assert np.allclose(ods[f"{base}.grid_flux.rho_tor_norm"], [_RHO_TOR_NORM])
    assert np.allclose(ods[f"{base}.electrons.energy.flux"], [40.456 * _Q_GB])
    assert ods[f"{base}.ion.1.element.0.z_n"] == 6.0
    assert native["states"][0]["surfaces"][0]["outputs"]["gbflux"]


def test_two_configurations_of_one_state_coexist_in_one_filedb(tmp_path):
    from vaft.database._local import load_ods
    from vaft.omas.vest_upstream import write_turbulent_transport_product

    db = FileDB(tmp_path / "filedb")
    expected = {"tglf-sat0-es": 6.8656 * _Q_GB, "tglf-sat2-em-bper": 40.456 * _Q_GB}
    products = {}
    for fixture in ("sat0-es", "sat2-em-bper"):
        ods, manifest, native = _build(_tree(tmp_path / "runs", fixture))
        configuration = manifest["configuration"]
        lineage = stage_lineage("turbulent_transport", family=manifest["family"],
                                product=configuration)
        output = db.omas_product("turbulent_transport", shot=SHOT, **lineage)
        metadata = db.omas_manifest("turbulent_transport", shot=SHOT, **lineage)
        write_turbulent_transport_product(ods, manifest, native, output=output, metadata=metadata)
        products[configuration] = (output, output.read_bytes())

    (first, first_bytes), (second, _) = products.values()
    assert first != second
    # The second save left the first byte-identical, and both read back as built.
    assert first.read_bytes() == first_bytes
    for configuration, (output, _) in products.items():
        ods, _ = load_ods(output)
        assert ods["core_transport.ids_properties.name"] == (
            f"{configuration}-from-magnetic-efit")
        flux = ods["core_transport.model.0.profiles_1d.0.electrons.energy.flux"]
        assert np.allclose(flux, [expected[configuration]])
        assert json.loads(ods["core_transport.model.0.code.parameters"])["tglf_config"] == (
            configuration)


def test_the_electron_lineage_is_filed_and_named_apart(tmp_path):
    ods, manifest, _ = _build(_tree(tmp_path, "sat0-es", lineage="electron_kinetic"))
    assert manifest["family"] == "electron"
    assert manifest["transitional_source"] == "electron-efit/tglf-sat0-es"
    assert ods["core_transport.ids_properties.name"] == "tglf-sat0-es-from-electron-efit"


def test_several_times_of_one_shot_share_one_product(tmp_path):
    from vaft.omas.vest_upstream import build_turbulent_transport_ods

    late = _tree(tmp_path / "a", "sat0-es", time_ms=330)
    early = _tree(tmp_path / "b", "sat0-es", time_ms=316)
    ods, _, _ = build_turbulent_transport_ods(state_directories=[late, early])
    assert np.allclose(ods["core_transport.time"], [0.316, 0.330])
    with pytest.raises(ValueError, match="two states at one time"):
        build_turbulent_transport_ods(state_directories=[early, early])
    other = _tree(tmp_path / "c", "sat2-em-bper", time_ms=340)
    with pytest.raises(ValueError, match="one shot, family and configuration"):
        build_turbulent_transport_ods(state_directories=[early, other])


def test_states_that_do_not_qualify_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not one of"):
        _build(_tree(tmp_path / "q", "sat0-es", quality="unreconstructible"))
    # A label that does not name its settings would file one configuration's
    # result under another's source.
    with pytest.raises(ValueError, match="does not name its settings"):
        _build(_tree(tmp_path / "l", "sat0-es", label="tglf-sat3-es",
                     parameters=_parameters("tglf-sat0-es")))
    # The label and parameters agree, but the native run used other settings.
    with pytest.raises(ValueError, match="input.tglf ran SAT_RULE=0"):
        _build(_tree(tmp_path / "n", "sat0-es", label="tglf-sat1-es",
                     parameters=_parameters("tglf-sat1-es")))
    with pytest.raises(sources.HSDSSourceError):
        _build(_tree(tmp_path / "k", "sat0-es", lineage="kinetic"))


def _write(db: FileDB, state_dir: Path) -> Path:
    from vaft.omas.vest_upstream import write_turbulent_transport_product

    ods, manifest, native = _build(state_dir)
    lineage = stage_lineage("turbulent_transport", family=manifest["family"],
                            product=manifest["configuration"])
    output = db.omas_product("turbulent_transport", shot=SHOT, **lineage)
    write_turbulent_transport_product(
        ods, manifest, native, output=output,
        metadata=db.omas_manifest("turbulent_transport", shot=SHOT, **lineage))
    return output


def test_another_ti_assumption_of_one_state_is_refused_not_overwritten(tmp_path):
    """The path names no Ti assumption, so a second state there would replace the
    first. Until #1128/#1129 it is refused; rewriting the same state is allowed."""
    db = FileDB(tmp_path / "filedb")
    output = _write(db, _tree(tmp_path / "a", "sat0-es"))
    stored = output.read_bytes()
    assert _write(db, _tree(tmp_path / "b", "sat0-es")) == output   # same state: idempotent
    other = _tree(tmp_path / "c", "sat0-es", state_identity="state-39915-316-inferred-ti",
                  ti_lineage="inferred")
    with pytest.raises(ValueError, match="already holds the states"):
        _write(db, other)
    assert output.read_bytes() == stored


def test_a_multi_state_product_refuses_states_it_would_misdescribe(tmp_path):
    from vaft.omas.vest_upstream import build_turbulent_transport_ods

    early = _tree(tmp_path / "a", "sat0-es", time_ms=316)
    variants = {
        "ti_lineage": dict(ti_lineage="inferred"),
        "gacode_revision": dict(revision="0000000"),
        "tglf_parameters": dict(parameters={**_parameters("tglf-sat0-es"), "kygrid_model": 4}),
    }
    for key, change in variants.items():
        late = _tree(tmp_path / key, "sat0-es", time_ms=330, **change)
        with pytest.raises(ValueError, match=f"{key} .* differs"):
            build_turbulent_transport_ods(state_directories=[early, late])


def test_a_rebuild_that_disagrees_with_the_projection_or_checks_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="beyond rtol"):
        _build(_tree(tmp_path / "a", "sat0-es", projection_scale=1.01))
    with pytest.raises(ValueError, match="no rebuilt flux could be compared"):
        _build(_tree(tmp_path / "b", "sat0-es", project_fluxes=False))


def test_pipeline_paths_resolve_a_turbulent_transport_replication_record(tmp_path):
    import sys

    workflow = (Path(__file__).parents[1] / "workflow"
                / "automatic_pipeline_1_routine_data_processing")
    if not workflow.exists():
        pytest.skip("workflow scripts are not part of the distribution")
    sys.path.insert(0, str(workflow))
    try:
        from paths import PipelinePaths
    finally:
        sys.path.remove(str(workflow))

    paths = PipelinePaths(str(tmp_path), "filedb")
    record = paths.replication_record(SHOT, "turbulent_transport", "tglf-sat3-em-bper")
    assert Path(record).relative_to(tmp_path).as_posix() == (
        "omas/turbulent_transport/magnetic/tglf-sat3-em-bper/39915/metadata/replication.json")
    with pytest.raises(FileDBPathError, match="not a TGLF configuration"):
        paths.replication_record(SHOT, "turbulent_transport", "rdcon")
    with pytest.raises(FileDBPathError, match="TGLF configuration"):
        paths.replication_record(SHOT, "turbulent_transport")
