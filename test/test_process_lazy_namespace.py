"""`vaft.process` resolves its submodules lazily, over a surface that is now its own.

The package used to star-import all thirteen aggregated submodules at load
time.  Because importing any submodule first executes this package, asking for
one pure kernel cost the whole subtree: `from vaft.process.signal_processing
import smooth` needed scipy and got omas, Matplotlib, ipywidgets,
uncertainties, pandas, sklearn and statsmodels as well -- 3181 modules where
1015 would do (issue #249).

The second half of that issue is what the package *exposed*.  Six submodules
declared no `__all__`, so a star import took everything they imported along
with everything they defined, and `vaft/process/profile.py` opened with
`from omas import *` -- for nothing, as it turned out; not one OMAS name was
used in the file.  So `vaft.process.load_omas_json`, `.machine_mappings`,
`.mdstree`, `.IntSlider`, `.raw_db` and 134 others were public API by accident.

`test/data/process_export_inventory.json` records all 336 names as they were
before the change.  `REMOVED` below accounts for every one this narrowing
drops, and the tests check both that the name is really gone and that the
module named beside it really provides it, so the inventory cannot rot into a
list of assertions nobody can act on.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys

import pytest

import vaft.process


INVENTORY = pathlib.Path(__file__).parent / "data" / "process_export_inventory.json"

#: The submodules this package star-imported, in the order it did.
_IMPORT_ORDER = (
    "profile",
    "equilibrium",
    "camera_geometry",
    "signal_processing",
    "fluctuation",
    "soft_x_rays",
    "electromagnetics",
    "numerical",
    "magnetics",
    "statistical_analysis",
    "atomic",
    "langmuir",
    "impa",
)

#: Heavy dependencies no narrow process import has any business loading.
#: `vaft.database` is here because `vaft/database/raw.py` imports Matplotlib at
#: module scope purely as an availability probe -- the chain issue #268
#: describes, which used to run through `vaft/process/magnetics.py`.
_HEAVY = (
    "omas",
    "matplotlib",
    "ipywidgets",
    "pandas",
    "statsmodels",
    "sklearn",
    "uncertainties",
    "vaft.database",
)

#: Every name the narrowing drops, and the module that actually provides it.
#: The second element says whether the name *is* a module (`vaft.process.np`
#: was numpy itself) or an attribute of one.  Nothing in this repository
#: referenced any of them through `vaft.process`; the map exists so anyone who
#: did is told where to go.
REMOVED: dict[str, tuple[str, str]] = {
    'Any': ('typing', 'attr'),
    'Callable': ('typing', 'attr'),
    'CodeParameters': ('omas', 'attr'),
    'Dict': ('typing', 'attr'),
    'IntSlider': ('ipywidgets', 'attr'),
    'List': ('typing', 'attr'),
    'MU0': ('vaft.formula', 'attr'),
    'NUMBA_AVAILABLE': ('vaft.process.electromagnetics', 'attr'),
    'ODC': ('omas', 'attr'),
    'ODS': ('omas', 'attr'),
    'ODX': ('omas', 'attr'),
    'OmasDynamicException': ('omas', 'attr'),
    'Optional': ('typing', 'attr'),
    'RectBivariateSpline': ('scipy.interpolate', 'attr'),
    'RegularGridInterpolator': ('scipy.interpolate', 'attr'),
    'Sequence': ('typing', 'attr'),
    'Tuple': ('typing', 'attr'),
    'Union': ('typing', 'attr'),
    'browse_imas': ('omas', 'attr'),
    'calculate_distance': ('vaft.formula', 'attr'),
    'cocos_transform': ('omas', 'attr'),
    'codeparams_xml_load': ('omas', 'attr'),
    'codeparams_xml_save': ('omas', 'attr'),
    'coherence': ('scipy.signal', 'attr'),
    'csd': ('scipy.signal', 'attr'),
    'cumtrapz_compat': ('vaft.compat', 'attr'),
    'dataclass': ('dataclasses', 'attr'),
    'define_cocos': ('omas', 'attr'),
    'del_omas_s3': ('omas', 'attr'),
    'different_ods': ('omas', 'attr'),
    'find_peaks': ('scipy.signal', 'attr'),
    'fit_profile': ('vaft.formula', 'attr'),
    'get_actor_io_ids': ('omas', 'attr'),
    'get_plot_scale_and_unit': ('omas', 'attr'),
    'green_br_bz': ('vaft.formula', 'attr'),
    'green_br_bz_exact': ('vaft.formula', 'attr'),
    'green_psi_exact': ('vaft.formula', 'attr'),
    'green_r': ('vaft.formula', 'attr'),
    'identify_cocos': ('omas', 'attr'),
    'imas_versions': ('omas', 'attr'),
    'interact': ('ipywidgets', 'attr'),
    'interp1d': ('scipy.interpolate', 'attr'),
    'latest_imas_version': ('omas', 'attr'),
    'latexit': ('omas', 'attr'),
    'list_omas_s3': ('omas', 'attr'),
    'load_omas_ascii': ('omas', 'attr'),
    'load_omas_ds': ('omas', 'attr'),
    'load_omas_dx': ('omas', 'attr'),
    'load_omas_h5': ('omas', 'attr'),
    'load_omas_hdc': ('omas', 'attr'),
    'load_omas_imas': ('omas', 'attr'),
    'load_omas_iter_scenario': ('omas', 'attr'),
    'load_omas_json': ('omas', 'attr'),
    'load_omas_machine': ('omas', 'attr'),
    'load_omas_mongo': ('omas', 'attr'),
    'load_omas_nc': ('omas', 'attr'),
    'load_omas_pkl': ('omas', 'attr'),
    'load_omas_s3': ('omas', 'attr'),
    'loadmat': ('scipy.io', 'attr'),
    'logger': ('vaft.process.statistical_analysis', 'attr'),
    'logging': ('logging', 'module'),
    'machine_expression_types': ('omas', 'attr'),
    'machine_mapping_function': ('omas', 'attr'),
    'machine_mappings': ('omas', 'attr'),
    'machines': ('omas', 'attr'),
    'math': ('math', 'module'),
    'mdstree': ('omas', 'attr'),
    'mdsvalue': ('omas', 'attr'),
    'ndarray': ('numpy', 'attr'),
    'np': ('numpy', 'module'),
    'numba': ('numba', 'module'),
    'ods_2_odx': ('omas', 'attr'),
    'ods_sample': ('omas', 'attr'),
    'odx_2_ods': ('omas', 'attr'),
    'omas_ascii': ('omas.omas_ascii', 'module'),
    'omas_core': ('omas.omas_core', 'module'),
    'omas_cython': ('omas.omas_cython', 'module'),
    'omas_dir': ('omas', 'attr'),
    'omas_ds': ('omas.omas_ds', 'module'),
    'omas_environment': ('omas', 'attr'),
    'omas_h5': ('omas.omas_h5', 'module'),
    'omas_hdc': ('omas.omas_hdc', 'module'),
    'omas_imas': ('omas.omas_imas', 'module'),
    'omas_info': ('omas', 'attr'),
    'omas_info_node': ('omas', 'attr'),
    'omas_json': ('omas.omas_json', 'module'),
    'omas_machine': ('omas.omas_machine', 'module'),
    'omas_mongo': ('omas.omas_mongo', 'module'),
    'omas_nc': ('omas.omas_nc', 'module'),
    'omas_physics': ('omas.omas_physics', 'module'),
    'omas_plot': ('omas.omas_plot', 'module'),
    'omas_rcparams': ('omas', 'attr'),
    'omas_s3': ('omas.omas_s3', 'module'),
    'omas_sample': ('omas.omas_sample', 'module'),
    'omas_setup': ('omas.omas_setup', 'module'),
    'omas_structure': ('omas.omas_structure', 'module'),
    'omas_symbols': ('omas.omas_symbols', 'module'),
    'omas_testdir': ('omas', 'attr'),
    'omas_uda': ('omas.omas_uda', 'module'),
    'omas_utils': ('omas.omas_utils', 'module'),
    'os': ('os', 'module'),
    'pd': ('pandas', 'module'),
    'pearsonr': ('scipy.stats', 'attr'),
    'probe_endpoints': ('omas', 'attr'),
    'raw_db': ('vaft.database.raw', 'module'),
    'rcparams_environment': ('omas', 'attr'),
    'rms': ('vaft.formula', 'attr'),
    'save_omas_ascii': ('omas', 'attr'),
    'save_omas_ds': ('omas', 'attr'),
    'save_omas_dx': ('omas', 'attr'),
    'save_omas_h5': ('omas', 'attr'),
    'save_omas_hdc': ('omas', 'attr'),
    'save_omas_imas': ('omas', 'attr'),
    'save_omas_json': ('omas', 'attr'),
    'save_omas_mongo': ('omas', 'attr'),
    'save_omas_nc': ('omas', 'attr'),
    'save_omas_pkl': ('omas', 'attr'),
    'save_omas_s3': ('omas', 'attr'),
    'savgol_filter': ('scipy.signal', 'attr'),
    'search_in_array_structure': ('omas', 'attr'),
    'search_ion': ('omas', 'attr'),
    'signal': ('scipy.signal', 'module'),
    'sm': ('statsmodels.api', 'module'),
    'test_machine_mapping_functions': ('omas', 'attr'),
    'through_omas_ascii': ('omas', 'attr'),
    'through_omas_ds': ('omas', 'attr'),
    'through_omas_dx': ('omas', 'attr'),
    'through_omas_h5': ('omas', 'attr'),
    'through_omas_hdc': ('omas', 'attr'),
    'through_omas_imas': ('omas', 'attr'),
    'through_omas_json': ('omas', 'attr'),
    'through_omas_mongo': ('omas', 'attr'),
    'through_omas_nc': ('omas', 'attr'),
    'through_omas_pkl': ('omas', 'attr'),
    'through_omas_s3': ('omas', 'attr'),
    'transform_current': ('omas', 'attr'),
    'unumpy': ('uncertainties.unumpy', 'module'),
    'utilities': ('omas.utilities', 'module'),
    'vaft': ('vaft', 'module'),
}


def _inventory() -> dict[str, dict]:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))["names"]


def _import_in_subprocess(statement: str) -> tuple[set[str], set[str]]:
    """The `vaft.process.*` submodules and heavy packages `statement` loads."""
    code = (
        "import sys\n"
        f"{statement}\n"
        "print(' '.join(sorted(m for m in sys.modules "
        "if m.startswith('vaft.process.'))))\n"
        "print(' '.join(sorted(m for m in sys.modules if m in "
        f"{_HEAVY!r})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    submodules, heavy = result.stdout.splitlines()[:2]
    return set(submodules.split()), set(heavy.split())


# ---------------------------------------------------------------------------
# Laziness
# ---------------------------------------------------------------------------

def test_importing_the_package_alone_imports_no_submodule():
    submodules, heavy = _import_in_subprocess("import vaft.process")

    assert submodules == set()
    assert heavy == set()


def test_importing_one_submodule_does_not_drag_in_its_siblings():
    submodules, _ = _import_in_subprocess("import vaft.process.signal_processing")

    assert submodules == {"vaft.process.signal_processing"}


def test_a_scipy_only_kernel_costs_nothing_heavier_than_scipy():
    """The point of the exercise, stated as the import that motivated it."""
    submodules, heavy = _import_in_subprocess(
        "from vaft.process.signal_processing import smooth"
    )

    assert submodules == {"vaft.process.signal_processing"}
    assert heavy == set()


@pytest.mark.parametrize(
    "submodule",
    ["camera_geometry", "cocos", "fluctuation", "impa", "langmuir",
     "magnetics", "numerical", "signal_processing"],
)
def test_submodules_that_need_no_heavy_dependency_do_not_load_one(submodule):
    _, heavy = _import_in_subprocess(f"import vaft.process.{submodule}")

    assert heavy == set()


def test_magnetics_no_longer_pulls_in_widgets_or_the_database():
    """The three module-scope imports issue #249 named, as a regression test.

    `vaft/process/magnetics.py` imported ipywidgets and `vaft.database` at
    module scope, and imported `define_baseline`/`subtract_baseline` back out
    of `vaft.process` -- a self-referential package import that only worked
    because the star-import chain had already bound them.
    """
    _, heavy = _import_in_subprocess("import vaft.process.magnetics")

    assert "ipywidgets" not in heavy
    assert "vaft.database" not in heavy
    assert "matplotlib" not in heavy


def test_no_submodule_imports_the_package_it_lives_in():
    """Re-entrancy: a submodule importing `vaft.process` during package init.

    Under the old eager chain this happened to work.  Under a lazy
    `__getattr__` it is a re-entrant resolution, so it must not come back.
    """
    package = pathlib.Path(vaft.process.__file__).parent
    offenders = [
        path.name
        for path in sorted(package.glob("*.py"))
        if path.name != "__init__.py"
        and "from vaft.process import " in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


# ---------------------------------------------------------------------------
# The narrowed surface
# ---------------------------------------------------------------------------

def test_the_inventory_covers_every_name_the_package_used_to_expose():
    inventory = _inventory()

    assert len(inventory) == 336
    assert sum(1 for entry in inventory.values() if entry["owned"]) == 172


def test_every_previously_exposed_name_is_kept_or_accounted_for():
    """No name may vanish silently: it is still here, or it is in REMOVED."""
    surviving = set(vaft.process.__all__)
    unaccounted = sorted(set(_inventory()) - surviving - set(REMOVED))

    assert unaccounted == []


def test_nothing_is_listed_as_removed_while_still_being_exposed():
    still_here = sorted(set(REMOVED) & set(vaft.process.__all__))

    assert still_here == []


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_each_dropped_name_is_reachable_where_the_map_says_it_is(name):
    where, kind = REMOVED[name]
    try:
        module = importlib.import_module(where)
    except ModuleNotFoundError as error:
        # Some of these live in an optional compiled extension of a dependency:
        # omas ships omas_cython.pyx as source and builds it only where a
        # compiler was available, so it is absent from a stock Windows install.
        # This test exists to prove the map points somewhere real; whether the
        # dependency happened to be built with Cython is not VAFT's contract.
        if error.name == where:
            pytest.skip(f"{where} is not built in this environment")
        raise

    if kind == "module":
        assert module.__name__ == where
    else:
        assert hasattr(module, name), f"{where} does not provide {name!r}"


#: The one name a submodule really does bind that is still dropped on purpose.
#: `statistical_analysis` assigns a module-level `logging.getLogger(...)`, which
#: an AST scan cannot tell from an intended export.  It is still reachable as
#: `vaft.process.statistical_analysis.logger`.
_OWNED_BUT_DROPPED = frozenset({"logger"})


def test_the_dropped_names_are_the_ones_no_submodule_defines():
    """Nothing owned was dropped, apart from the one name listed above."""
    inventory = _inventory()
    owned = {name for name, entry in inventory.items() if entry["owned"]}

    assert (owned & set(REMOVED)) == _OWNED_BUT_DROPPED


def test_every_submodule_now_declares_what_it_exports():
    """The actual fix: an explicit `__all__` everywhere, so nothing leaks."""
    missing = [
        name
        for name in _IMPORT_ORDER
        if not hasattr(importlib.import_module(f"vaft.process.{name}"), "__all__")
    ]

    assert missing == []


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_import_order_is_not_load_bearing():
    """Unlike `vaft.formula`, no two submodules disagree about a name.

    Eight names were reachable from two submodules before this change and every
    one of them was the same object twice.  If that ever stops being true the
    package must say so rather than resolve by import order, which is what
    `_resolve` asserts and what this test pins.
    """
    seen: dict[str, tuple[str, object]] = {}
    conflicts = []
    for key in _IMPORT_ORDER:
        module = importlib.import_module(f"vaft.process.{key}")
        for name in module.__all__:
            value = getattr(module, name)
            if name in seen and seen[name][1] is not value:
                conflicts.append((name, seen[name][0], key))
            seen.setdefault(name, (key, value))

    assert conflicts == []


def test_a_genuine_collision_is_refused_rather_than_resolved_by_order(monkeypatch):
    import vaft.process.numerical as numerical
    import vaft.process.signal_processing as signal_processing

    monkeypatch.setattr(numerical, "collide", object(), raising=False)
    monkeypatch.setattr(signal_processing, "collide", object(), raising=False)
    monkeypatch.setattr(
        numerical, "__all__", [*numerical.__all__, "collide"], raising=False
    )
    monkeypatch.setattr(
        signal_processing,
        "__all__",
        [*signal_processing.__all__, "collide"],
        raising=False,
    )

    with pytest.raises(AttributeError, match="different objects"):
        vaft.process._resolve("collide")


def test_the_same_object_seen_twice_is_not_a_collision():
    """`define_baseline` reached the namespace through two submodules."""
    from vaft.process.signal_processing import define_baseline

    assert vaft.process.define_baseline is define_baseline


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

def test_submodules_are_reachable_as_attributes():
    for name in (*_IMPORT_ORDER, "cocos"):
        assert importlib.import_module(f"vaft.process.{name}") is getattr(
            vaft.process, name
        )


def test_the_parametric_api_still_resolves_through_equilibrium():
    """`equilibrium` keeps `_equilibrium_parametric` as its public location."""
    from vaft.process._equilibrium_parametric import validate_equilibrium

    assert vaft.process.validate_equilibrium is validate_equilibrium
    assert vaft.process.equilibrium.validate_equilibrium is validate_equilibrium


def test_star_import_binds_the_declared_surface():
    namespace: dict[str, object] = {}
    exec("from vaft.process import *", namespace)  # noqa: S102

    bound = {name for name in namespace if not name.startswith("__")}
    assert bound == set(vaft.process.__all__)


def test_dir_offers_the_submodules_before_anything_is_imported():
    listed = set(dir(vaft.process))

    assert set(_IMPORT_ORDER) | {"cocos"} <= listed


def test_an_unknown_name_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        vaft.process.definitely_not_a_process_function
