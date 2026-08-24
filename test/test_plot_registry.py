"""Discovery parity between the registry, ``__all__`` and ``dir(vaft.plot)``."""

import ast
import inspect
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

import vaft.plot
from vaft.plot import registry
from vaft.plot.models import ViewModel


def test_available_plots_matches_the_registry_and_all():
    names = {row["name"] for row in vaft.plot.available_plots()}
    assert names == set(registry.canonical_names())
    assert names <= set(vaft.plot.__all__)


def test_every_canonical_name_is_bound_on_the_package():
    for name in registry.canonical_names():
        assert hasattr(vaft.plot, name), name
        assert vaft.plot.__all__.count(name) == 1


def test_canonical_names_are_re_exported_explicitly_not_via_globals():
    # Static tooling must see vaft.plot.<name>, so __init__ imports each
    # renderer by name rather than assigning into globals() at import time.
    source = Path(vaft.plot.__file__).read_text(encoding="utf-8")
    for name in registry.canonical_names():
        assert f"    {name},\n" in source, name

    # No module-level code may synthesize names into globals(); the lazy
    # deprecation cache inside __getattr__ is not module-level and is fine.
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Call):
                function = inner.value.func
                assert getattr(function, "id", None) != "globals", ast.dump(node)[:120]


def test_every_canonical_renderer_is_a_real_module_level_def():
    # Regression on the import-time globals() generation in the old onedim
    # module: generated functions were invisible to docs and static tooling.
    for spec in registry.specs():
        function = spec.renderer
        assert isinstance(function, types.FunctionType), spec.name
        module = inspect.getmodule(function)
        assert getattr(module, spec.name, None) is function, spec.name
        assert function.__doc__, spec.name
        assert inspect.getsourcefile(function).endswith(".py")


def test_no_modules_or_omas_symbols_leak_into_the_namespace():
    # The old package wildcard-imported every submodule plus ``from omas import
    # *``, exposing 42 modules and ~76 unrelated OMAS functions.
    for name in vaft.plot.__all__:
        value = getattr(vaft.plot, name)
        assert not isinstance(value, types.ModuleType), name
        module_name = getattr(value, "__module__", "vaft.plot")
        assert module_name.startswith("vaft.plot"), (name, module_name)

    for leaked in ("np", "plt", "sns", "omas_h5", "omas_imas", "unumpy", "patches"):
        assert not hasattr(vaft.plot, leaked), leaked


def test_dir_exposes_the_canonical_set():
    listed = set(dir(vaft.plot))
    assert set(vaft.plot.__all__) <= listed
    assert not any(isinstance(getattr(vaft.plot, n, None), types.ModuleType) for n in listed)


def test_specs_declare_their_model_and_data_requirements():
    for spec in registry.specs():
        assert issubclass(spec.model, ViewModel), spec.name
        assert spec.description.strip(), spec.name
        assert spec.view in registry.VIEWS, spec.name
        assert spec.name.startswith(spec.domain + "_"), spec.name
        assert not spec.name.startswith("plot_"), spec.name
        assert spec.ids, spec.name
        for path in spec.required_paths + spec.optional_paths:
            assert path.split(".")[0] in spec.ids, (spec.name, path)


def test_canonical_names_follow_the_domain_view_quantity_grammar():
    for spec in registry.specs():
        expected = "_".join(
            part for part in (spec.domain, spec.view, spec.quantity) if part
        )
        # ``quantity`` may be dropped when domain/view is already unambiguous.
        short = f"{spec.domain}_{spec.view}"
        assert spec.name in {expected, short}, spec.name


def test_registry_refuses_to_replace_an_entry():
    spec = registry.specs()[0]
    clash = registry.PlotSpec(
        name=spec.name, model=spec.model, renderer=lambda model, **kw: None,
        domain=spec.domain, view=spec.view, description="clash",
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register(clash)


def test_get_spec_names_the_discovery_helper_when_missing():
    with pytest.raises(KeyError, match="available_plots"):
        registry.get_spec("no_such_plot")
