"""The generated plot and diagram catalogs, and the check that no public name escapes a catalog.

Python only: the generators and ``docs/scripts/catalog_coverage.py`` are what
``docs/build.py`` runs before Jekyll, so everything here works without Ruby.
The rendered-page half of the check lives in ``validate_docs.rb`` and is
exercised by ``test_docs_site.py``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coverage():
    return _load_script("vaft_docs_catalog_coverage", DOCS / "scripts" / "catalog_coverage.py")


@pytest.fixture(scope="module")
def build():
    return _load_script("vaft_docs_build_for_catalogs", DOCS / "build.py")


@pytest.fixture(scope="module")
def snapshots():
    """Every catalog this branch declares, generated once from this checkout."""
    import vaft.diagram.docs_catalog as diagram_catalog
    import vaft.formula.catalog as formula_catalog
    import vaft.plot.docs_catalog as plot_catalog
    import vaft.process.catalog as process_catalog

    return {
        "formula": formula_catalog.documentation_snapshot(),
        "process": process_catalog.documentation_snapshot(),
        "plot": plot_catalog.documentation_snapshot(),
        "diagram": diagram_catalog.documentation_snapshot(),
    }


# --------------------------------------------------------------------------
# the generators
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["vaft.plot.docs_catalog", "vaft.diagram.docs_catalog"])
def test_generation_is_deterministic(module, tmp_path):
    outputs = []
    for attempt in range(2):
        target = tmp_path / f"{attempt}.yml"
        subprocess.run(
            [sys.executable, "-m", module, "--output", str(target),
             "--provenance-commit", "0" * 40, "--provenance-ref", "origin/develop"],
            cwd=str(ROOT), check=True, capture_output=True,
        )
        outputs.append(target.read_bytes())
    assert outputs[0] == outputs[1]
    snapshot = yaml.safe_load(outputs[0])
    assert snapshot["provenance"] == {"commit": "0" * 40, "ref": "origin/develop"}


@pytest.mark.parametrize("kind", ["plot", "diagram"])
def test_provenance_is_omitted_unless_given(snapshots, kind):
    assert "provenance" not in snapshots[kind]


@pytest.mark.parametrize("kind", ["plot", "diagram"])
def test_every_recorded_source_checksum_matches_the_tree(snapshots, kind):
    sources = snapshots[kind]["source"]
    assert sources
    for entry in sources:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_the_sources_are_in_the_form_build_py_verifies(build, tmp_path):
    import vaft.plot.docs_catalog as plot_catalog

    target = plot_catalog.export_documentation_snapshot(tmp_path / "plot.yml")
    assert build._declared_sources(target) == [
        (entry["path"], entry["sha256"]) for entry in yaml.safe_load(target.read_text())["source"]
    ]


def test_the_plot_catalog_holds_every_registered_plot_whatever_its_status(snapshots):
    from vaft.plot import registry

    specs = registry.specs(status=None)
    rows = snapshots["plot"]["plots"]
    assert [row["name"] for row in rows] and {row["name"] for row in rows} == {spec.name for spec in specs}
    by_name = {spec.name: spec for spec in specs}
    for row in rows:
        spec = by_name[row["name"]]
        assert (row["subject"], row["view"], row["quantity"], row["status"]) == (
            spec.subject, spec.view, spec.quantity, spec.status)
        assert row["adapter"] == f"vaft.omas.plot_{spec.stem}"
        assert row["required_paths"] == list(spec.required_paths)
        lines, start = inspect.getsourcelines(inspect.unwrap(spec.renderer))
        assert row["source"]["line"] == start
        assert (ROOT / row["source"]["path"]).is_file()


def test_the_plot_catalog_is_ordered_subject_then_view(snapshots):
    from vaft.plot import registry, taxonomy

    subjects = list(taxonomy.SUBJECTS)
    keys = [(subjects.index(row["subject"]), registry.VIEWS.index(row["view"]))
            for row in snapshots["plot"]["plots"]]
    assert keys == sorted(keys)


def test_the_diagram_catalog_holds_every_canonical_asset(snapshots):
    from vaft.diagram import build as diagram_build

    snapshot = snapshots["diagram"]
    assert [row["asset"] for row in snapshot["assets"]] == list(diagram_build.CANONICAL)
    for row in snapshot["assets"]:
        record, _ = diagram_build.read_record(DOCS / "assets" / "diagrams" / row["asset"])
        assert row["svg_sha256"] == record["svg_sha256"]
        assert (DOCS / row["svg"]).is_file()
        assert row["asset"] in next(b for b in snapshot["builders"] if b["name"] == row["builder"])["assets"]


def test_the_diagram_catalog_lists_the_formats_diagram_save_writes(snapshots, tmp_path):
    from vaft.diagram import _render

    import vaft
    import vaft.diagram

    formats = snapshots["diagram"]["formats"]
    assert [f["suffix"] for f in formats] == list(_render.SAVE_FORMATS)
    assert formats[0]["suffix"] == ".svg"  # the canonical artifact leads
    assert {f["suffix"]: f["requires"] for f in formats}[".tex"] == []  # no TeX needed for the source
    # the recipe the page prints is `d = <call>`: every call must evaluate to that very diagram
    for row in snapshots["diagram"]["assets"]:
        assert row["call"].startswith("vaft.diagram.")
    row = next(r for r in snapshots["diagram"]["assets"] if r["builder"] == "rational_surface")
    diagram = eval(row["call"], {"vaft": vaft})
    assert isinstance(diagram, vaft.diagram.Diagram)
    assert diagram.source_sha256 == row["source_sha256"]
    # each suffix the page lists is one Diagram.save accepts; .tex needs no toolchain
    saved = diagram.save(tmp_path / "x.tex")
    assert saved.read_text(encoding="utf-8") == diagram.tikz
    with pytest.raises(ValueError, match=re.escape(".svg, .tex or .pdf")):
        diagram.save(tmp_path / "x.png")


def test_the_diagram_catalog_records_the_formula_functions_a_builder_calls(snapshots):
    island = next(b for b in snapshots["diagram"]["builders"] if b["name"] == "magnetic_island")
    ids = {f["id"] for f in island["formula"]}
    assert {"vaft.formula.stability.helical_phase", "vaft.formula.equilibrium.miller_surface"} <= ids
    import vaft.formula

    for builder in snapshots["diagram"]["builders"]:
        for f in builder["formula"]:
            module = importlib.import_module(f["id"].rsplit(".", 1)[0])
            assert callable(getattr(module, f["name"])), f


# --------------------------------------------------------------------------
# the coverage check: it passes on the real tree ...
# --------------------------------------------------------------------------


def test_every_layer_is_fully_catalogued(coverage, snapshots):
    assert coverage.check_formula(snapshots["formula"]) == []
    assert coverage.check_process(snapshots["process"]) == []
    assert coverage.check_plot(snapshots["plot"]) == []
    assert coverage.check_diagram(snapshots["diagram"], ROOT) == []


def _docs_tree(tmp_path, snapshots):
    docs = tmp_path / "docs"
    (docs / "_data").mkdir(parents=True)
    names = {"formula": "formula_catalog.yml", "process": "process_catalog.yml",
             "plot": "plot_catalog.yml", "diagram": "diagram_catalog.yml"}
    # Only the generators whose snapshots this helper writes; the API catalog
    # has its own tests (test_docs_api.py).
    declared = yaml.safe_load((DOCS / "generators.yml").read_text(encoding="utf-8"))
    declared["generators"] = [g for g in declared["generators"] if Path(g["output"]).name in names.values()
                              or g["module"] == "vaft.machine_mapping.registry"]
    (docs / "generators.yml").write_text(yaml.safe_dump(declared), encoding="utf-8")
    for kind, name in names.items():
        (docs / "_data" / name).write_text(yaml.safe_dump(snapshots[kind]), encoding="utf-8")
    return docs


def test_the_script_checks_what_generators_yml_declares(coverage, snapshots, tmp_path):
    docs = _docs_tree(tmp_path, snapshots)
    assert coverage.check(docs, root=ROOT) == []
    (docs / "_data" / "plot_catalog.yml").unlink()
    assert coverage.check(docs, root=ROOT) == ["_data/plot_catalog.yml is declared but was not generated"]


def test_the_script_refuses_to_judge_another_copy_of_vaft(coverage, snapshots, tmp_path):
    docs = _docs_tree(tmp_path, snapshots)
    problems = coverage.check(docs)            # root defaults to tmp_path, which is not where vaft lives
    assert len(problems) == 1 and "not from the tree being documented" in problems[0]


# --------------------------------------------------------------------------
# ... and it fails when a public name is missing or a catalog entry vanished
# --------------------------------------------------------------------------


def _without(snapshot, key, name):
    mutated = copy.deepcopy(snapshot)
    mutated[key] = [row for row in mutated[key] if row["name"] != name]
    assert len(mutated[key]) == len(snapshot[key]) - 1
    return mutated


def test_a_formula_missing_from_its_catalog_is_caught(coverage, snapshots):
    problems = coverage.check_formula(_without(snapshots["formula"], "formulas", "greenwald_density"))
    assert any("greenwald_density" in p and "no catalog entry describes it" in p for p in problems), problems


def test_a_process_missing_from_its_catalog_is_caught(coverage, snapshots):
    victim = snapshots["process"]["functions"][0]["name"]
    problems = coverage.check_process(_without(snapshots["process"], "functions", victim))
    assert any(victim in p and "no catalog entry describes it" in p for p in problems), problems


def test_a_plot_missing_from_its_catalog_is_caught(coverage, snapshots):
    problems = coverage.check_plot(_without(snapshots["plot"], "plots", "plasma_current_time"))
    assert problems == [
        # from the namespace users type, and from the registry
        "plot: vaft.plot.plasma_current_time renders registered plot plasma_current_time, which is not in plot_catalog.yml",
        "plot: registered plot plasma_current_time is not in plot_catalog.yml",
    ]


def test_a_diagram_builder_missing_from_its_catalog_is_caught(coverage, snapshots):
    problems = coverage.check_diagram(_without(snapshots["diagram"], "builders", "hugill"), ROOT)
    assert "diagram: public builder vaft.diagram.hugill is not in diagram_catalog.yml" in problems


def test_a_diagram_asset_missing_from_its_catalog_is_caught(coverage, snapshots):
    mutated = copy.deepcopy(snapshots["diagram"])
    mutated["assets"] = [row for row in mutated["assets"] if row["asset"] != "hugill.svg"]
    problems = coverage.check_diagram(mutated, ROOT)
    assert "diagram: canonical asset hugill.svg is not in diagram_catalog.yml" in problems
    assert "diagram: docs/assets/diagrams/hugill.svg is committed but in no catalog entry (orphaned)" in problems


def test_an_orphaned_svg_is_caught(coverage, snapshots, tmp_path):
    root = tmp_path / "tree"
    shutil.copytree(DOCS / "assets" / "diagrams", root / "docs" / "assets" / "diagrams")
    (root / "docs" / "assets" / "diagrams" / "left_behind.svg").write_text("<svg/>", encoding="utf-8")
    (root / "docs" / "assets" / "diagrams" / "troyon.svg").unlink()
    problems = coverage.check_diagram(snapshots["diagram"], root)
    assert "diagram: docs/assets/diagrams/left_behind.svg is committed but in no catalog entry (orphaned)" in problems
    assert "diagram: catalog asset troyon.svg has no committed SVG in docs/assets/diagrams" in problems


@pytest.mark.parametrize("kind, key, check", [
    ("formula", "formulas", "check_formula"),
    ("process", "functions", "check_process"),
])
def test_a_catalog_entry_whose_function_vanished_is_caught(coverage, snapshots, kind, key, check):
    mutated = copy.deepcopy(snapshots[kind])
    ghost = copy.deepcopy(mutated[key][0])
    ghost.update(name="no_such_function", id="ghost.no_such_function", aliases=[])
    mutated[key].append(ghost)
    problems = getattr(coverage, check)(mutated)
    assert any("ghost.no_such_function vanished" in p for p in problems), problems


def test_a_plot_or_diagram_entry_that_vanished_is_caught(coverage, snapshots):
    plot = copy.deepcopy(snapshots["plot"])
    plot["plots"].append({**plot["plots"][0], "id": "gone_time", "name": "gone_time"})
    assert "plot: catalog entry gone_time vanished: the registry no longer holds it" in coverage.check_plot(plot)

    diagram = copy.deepcopy(snapshots["diagram"])
    diagram["builders"].append({**diagram["builders"][0], "id": "gone", "name": "gone"})
    diagram["assets"].append({**diagram["assets"][0], "id": "gone", "asset": "gone.svg", "builder": "gone"})
    problems = coverage.check_diagram(diagram, ROOT)
    assert "diagram: catalog builder gone vanished: vaft.diagram no longer exports it" in problems
    assert "diagram: catalog asset gone.svg vanished: vaft.diagram.build.CANONICAL no longer declares it" in problems


def test_a_new_public_function_nobody_catalogued_is_caught(coverage, snapshots, monkeypatch):
    """The real condition: a name added to the package namespace, not to the catalog's own walk."""
    import vaft.formula

    def undocumented_formula(x):
        return x

    undocumented_formula.__module__ = "vaft.formula"
    undocumented_formula.__qualname__ = "undocumented_formula"
    monkeypatch.setattr(vaft.formula, "undocumented_formula", undocumented_formula, raising=False)
    monkeypatch.setattr(vaft.formula, "__all__", [*vaft.formula.__all__, "undocumented_formula"])
    problems = coverage.check_formula(snapshots["formula"])
    assert problems == ["formula: vaft.formula.undocumented_formula is public as vaft.formula.undocumented_formula "
                        "but no catalog entry describes it"]


def test_an_entry_describing_a_different_function_of_the_same_name_is_caught(coverage, snapshots, monkeypatch):
    import vaft.formula

    def impostor(x):
        return x

    monkeypatch.setattr(vaft.formula, "greenwald_density", impostor)
    problems = coverage.check_formula(snapshots["formula"])
    assert any("public as vaft.formula.greenwald_density but no catalog entry describes it" in p for p in problems)


# --------------------------------------------------------------------------
# the build runs it
# --------------------------------------------------------------------------


def _fake_track(build, tmp_path, script):
    root = tmp_path / "track"
    (root / "docs" / "scripts").mkdir(parents=True)
    if script is not None:
        (root / "docs" / "scripts" / "catalog_coverage.py").write_text(script, encoding="utf-8")
    spec = build.TRACKS["development"]
    return build.Track(spec=spec, ref="HEAD", commit="0" * 40, commit_date="", root=root)


def test_the_build_fails_when_coverage_fails(build, tmp_path):
    track = _fake_track(build, tmp_path, "import sys; print('plot: x is missing', file=sys.stderr); sys.exit(1)\n")
    with pytest.raises(build.BuildError, match="plot: x is missing"):
        build.check_catalog_coverage(track, {}, quiet=True)


def test_the_build_passes_when_coverage_passes_or_the_track_predates_it(build, tmp_path):
    build.check_catalog_coverage(_fake_track(build, tmp_path / "a", "print('ok')\n"), {}, quiet=True)
    build.check_catalog_coverage(_fake_track(build, tmp_path / "b", None), {}, quiet=True)


# --------------------------------------------------------------------------
# enumerations the generators do not perform (review of #1275)
# --------------------------------------------------------------------------


def test_the_plot_catalog_lists_the_plotting_functions_outside_the_registry(snapshots):
    import vaft.plot as plot
    from vaft.plot import _migration

    entry_points = {row["name"]: row["status"] for row in snapshots["plot"]["entry_points"]}
    assert {name for name, status in entry_points.items() if status == "legacy"} == set(_migration.LEGACY)
    assert "plot_parameter_history" in entry_points and entry_points["plot_parameter_history"] == "support"
    assert not set(entry_points) & {row["name"] for row in snapshots["plot"]["plots"]}
    for name in entry_points:
        assert inspect.isfunction(getattr(plot, name))


def test_a_plot_function_missing_from_its_catalog_is_caught(coverage, snapshots):
    for victim in ("plot_parameter_history", "plot_scaling_fit"):
        problems = coverage.check_plot(_without(snapshots["plot"], "entry_points", victim))
        assert problems == [f"plot: vaft.plot.{victim} is a public plotting function but is not in plot_catalog.yml"]


def test_a_new_plot_function_in_the_namespace_is_caught(coverage, snapshots, monkeypatch):
    import vaft.plot

    def plot_brand_new():
        """Draws something."""

    monkeypatch.setattr(vaft.plot, "plot_brand_new", plot_brand_new, raising=False)
    monkeypatch.setattr(vaft.plot, "__all__", sorted([*vaft.plot.__all__, "plot_brand_new"]))
    assert coverage.check_plot(snapshots["plot"]) == [
        "plot: vaft.plot.plot_brand_new is a public plotting function but is not in plot_catalog.yml"]


@pytest.mark.parametrize("package, check", [("vaft.formula", "check_formula"), ("vaft.process", "check_process")])
def test_a_submodule_outside_the_import_order_is_caught(coverage, snapshots, monkeypatch, tmp_path, package, check):
    """A new submodule on disk is public whether or not the package lists it."""
    module = importlib.import_module(package)
    (tmp_path / "newthing.py").write_text(
        '"""New."""\n__all__ = ["brand_new"]\n\n\ndef brand_new(x):\n    """New."""\n    return x\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "__path__", [*module.__path__, str(tmp_path)])
    kind = package.split(".")[1]
    try:
        problems = getattr(coverage, check)(snapshots[kind])
    finally:
        sys.modules.pop(f"{package}.newthing", None)
        if hasattr(module, "newthing"):
            delattr(module, "newthing")
    assert problems == [f"{kind}: {package}.newthing.brand_new is public as {package}.newthing.brand_new "
                        f"but no catalog entry describes it"]


def test_a_diagram_builder_nobody_exported_is_caught(coverage, snapshots, monkeypatch):
    from vaft.diagram import _ripple

    def hidden_diagram() -> "Diagram":  # noqa: F821 -- annotation text is what is read
        """New."""

    hidden_diagram.__module__ = _ripple.__name__
    hidden_diagram.__annotations__["return"] = "Diagram"
    monkeypatch.setattr(_ripple, "hidden_diagram", hidden_diagram, raising=False)
    problems = coverage.check_diagram(snapshots["diagram"], ROOT)
    assert problems == [
        "diagram: vaft.diagram._ripple.hidden_diagram builds a Diagram but is not in diagram_catalog.yml",
        "diagram: vaft.diagram._ripple.hidden_diagram builds a Diagram but is not exported as vaft.diagram.hidden_diagram",
    ]


def test_a_vaft_module_served_from_another_tree_is_caught(coverage, snapshots, tmp_path, monkeypatch):
    import types

    stray = types.ModuleType("vaft.somewhere_else")
    stray.__file__ = str(tmp_path / "elsewhere" / "vaft" / "somewhere_else.py")
    monkeypatch.setitem(sys.modules, "vaft.somewhere_else", stray)
    assert coverage.foreign_modules(ROOT) == [f"vaft.somewhere_else ({stray.__file__})"]
    docs = _docs_tree(tmp_path, snapshots)
    problems = coverage.check(docs, root=ROOT)
    assert problems == [f"vaft.somewhere_else ({stray.__file__}) was imported from outside the tree being documented ({ROOT})"]


def test_the_plot_catalog_hashes_every_file_of_the_package(snapshots):
    recorded = {entry["path"] for entry in snapshots["plot"]["source"]}
    on_disk = {path.relative_to(ROOT).as_posix() for path in (ROOT / "vaft" / "plot").rglob("*.py")
               if "__pycache__" not in path.parts}
    assert on_disk <= recorded
    # Beyond the package: the thumbnail manifest, the samples its pictures were drawn from,
    # and vaft/_docstring.py, which reads every source span.
    assert all(path in {"docs/assets/plots/manifest.json", "vaft/_docstring.py"} or path.startswith("vaft/data/samples/")
               for path in recorded - on_disk), recorded - on_disk


def test_the_diagram_catalog_hashes_the_formula_modules_it_resolved(snapshots):
    recorded = {entry["path"] for entry in snapshots["diagram"]["source"]}
    for builder in snapshots["diagram"]["builders"]:
        for f in builder["formula"]:
            assert f"vaft/formula/{f['category']}.py" in recorded


def test_plot_rows_carry_the_committed_thumbnail(snapshots):
    import json

    manifest = ROOT / "docs" / "assets" / "plots" / "manifest.json"
    if not manifest.is_file():
        pytest.skip("this branch commits no plot thumbnails")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["plots"]
    for row in snapshots["plot"]["plots"]:
        thumbnail = row["thumbnail"]
        assert thumbnail["status"] == recorded[row["name"]]["status"]
        if thumbnail["status"] == "rendered":
            assert thumbnail["png"] == f"assets/plots/{row['name']}.png"
            assert thumbnail["png_sha256"] == recorded[row["name"]]["png_sha256"]
            # Staleness only warns (#1322): a renderer change must not fail this test.
            assert isinstance(thumbnail["stale"], str)
    recorded_sources = {entry["path"] for entry in snapshots["plot"]["source"]}
    assert "docs/assets/plots/manifest.json" in recorded_sources


def test_a_hand_edited_thumbnail_fails_the_coverage_check(coverage, snapshots, tmp_path):
    root = tmp_path / "tree"
    shutil.copytree(ROOT / "docs" / "assets" / "plots", root / "docs" / "assets" / "plots")
    rendered = next(row["name"] for row in snapshots["plot"]["plots"] if row["thumbnail"]["status"] == "rendered")
    png = root / "docs" / "assets" / "plots" / f"{rendered}.png"
    png.write_bytes(png.read_bytes() + b"\0")
    problems = coverage.check_plot_thumbnails(snapshots["plot"], root)
    assert problems == [f"plot thumbnail {rendered}.png: does not match manifest.json (edited by hand?)"]
