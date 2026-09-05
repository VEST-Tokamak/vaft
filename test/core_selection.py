"""The `develop` gate's test selection, declared in one place.

VAFT runs two different gates (#515). `main` proves release confidence: the
whole suite, on Linux and on Windows, `slow` tests included. `develop` proves
development confidence -- is this change safe to integrate? -- and that
question does not need thirty-plus minutes of cross-platform scientific
regression to answer.

This module is what `develop` is gated on. `test/conftest.py` marks every item
collected from a module named here with ``pytest.mark.core``, so the selection
is one reviewable list rather than a `pytestmark` line scattered across forty
files: a reviewer can see the entire develop gate in a single diff, and a
module cannot drift into or out of the gate without that diff.

Membership was chosen from measured per-module wall clock, not from filenames.
The bar is a contract that fails loudly and cheaply -- import and namespace
shape, public API surface, layer boundaries, registries and taxonomies,
serialization round-trips, packaging and documentation policy. What is
deliberately *not* here is the scientific and regression coverage: equilibrium
solves, COCOS conformance, replication, benchmarks, anything that shells out to
an external code or a notebook. Those are release qualification, and they still
run in full -- on the `main` gate, and on the push to `develop` after a PR
lands.

CI runs this as ``pytest -m "core and not perf"``. The `perf` half is there for
the same reason the Windows leg drops it: a `perf` test asserts a wall-clock
ratio, and a job whose whole purpose is to finish in minutes is the last place
such a budget should be believed. It is deselected rather than excluded by
module, because a module like test_formula_catalog.py is twenty API-contract
tests and one timing budget, and the twenty are exactly what develop wants.
`slow` is different -- it is applied module-wide, so those modules are simply
not listed here.

Adding an entry costs every future PR its runtime, so it needs a reason that
fits on one line. `test/test_core_selection.py` enforces the rest: every entry
must exist, no `slow` module may be listed, the gate expression must still be
what CI runs, and the marker must be applied early enough that ``-m core``
actually selects something.
"""

from __future__ import annotations

from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent

#: Paths relative to ``test/``, grouped by the contract each group protects.
#: Sorted within each group; ``test_core_selection.py`` enforces that.
CORE_MODULES: tuple[str, ...] = (
    # Import and namespace shape. If these break, nothing downstream is
    # trustworthy -- and they are the cheapest tests in the repository.
    "test_compat_runtime.py",
    "test_data_code_namespace.py",
    "test_database_namespace.py",
    "test_formula_lazy_namespace.py",
    "test_import.py",
    "test_local_importers.py",
    "test_process_lazy_namespace.py",
    # Public API surface: catalogs, registries and the CLI must keep agreeing
    # with what the packages actually export.
    "test_cli.py",
    "test_formula_catalog.py",
    "test_plot_discovery.py",
    "test_plot_registry.py",
    "test_plot_submodule.py",
    "test_process_catalog.py",
    # Layer boundaries. Source-level architecture checks -- no solves, no I/O.
    "contracts/test_machine_mapping_boundaries.py",
    "test_api_layer_boundaries.py",
    "test_no_bare_downsample.py",
    "test_no_pyplot_outside_plot.py",
    "test_plot_backend_boundaries.py",
    "test_validation_architecture.py",
    # Registry, taxonomy and display policy: the vocabulary the rest of the
    # package indexes itself by.
    "test_diagnostic_registry.py",
    "test_display_policy.py",
    "test_layout_contract.py",
    "test_plot_contract.py",
    "test_plot_taxonomy.py",
    # Serialization and schema smoke. The ODS/IMAS shapes everything reads and
    # writes, plus the canonical-IDS contract fixtures.
    "contracts/test_contract_legacy_rejections.py",
    "contracts/test_contract_samples.py",
    "contracts/test_contract_synthetic.py",
    "contracts/test_models_magnetics.py",
    "contracts/test_models_plasma.py",
    "contracts/test_models_uncertainty.py",
    "test_dataset_description.py",
    "test_eqdsk_omas_roundtrip.py",
    "test_path_exists.py",
    # Packaging and documentation policy. Metadata reads; they catch the
    # breakage `package` cannot see until it is already building a wheel.
    "contracts/test_dependency_policy_matrix.py",
    "test_data_resources.py",
    "test_docstring_engine.py",
    "test_formula_docstrings.py",
    "test_packaging_issue45.py",
    "test_process_docstrings.py",
    # The gate's own contract.
    "test_core_selection.py",
)


# Measured and deliberately left out, recorded so they are not re-added by
# someone reading only the group headings:
#
#   test_plot_backend_access.py    loads a sample shot and sweeps every
#                                  registered plot spec against it. A real
#                                  contract, but it was 29% of this gate on its
#                                  own, and it is a conformance sweep rather
#                                  than a cheap boundary check. The import-time
#                                  half of the same contract is
#                                  test_plot_backend_boundaries.py, which is
#                                  here.
#   test_vest_yaml_boundaries.py   despite the name, revision-overlap checks
#                                  across shot ranges -- shot-data validation,
#                                  not an architectural boundary, and 18% of
#                                  the gate. It belongs to the full suite.
#
# Both still run on the main gate and on the push to develop.


def core_paths() -> tuple[Path, ...]:
    """Absolute paths of the declared core modules."""
    return tuple(TEST_ROOT / relative for relative in CORE_MODULES)
