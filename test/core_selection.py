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
    "test_coil_geometry_3d_shim.py",
    "test_compat_runtime.py",
    "test_data_code_namespace.py",
    "test_database_export.py",
    "test_database_namespace.py",
    "test_formula_lazy_namespace.py",
    "test_import.py",
    "test_local_importers.py",
    "test_process_lazy_namespace.py",
    # Public API surface: catalogs, registries and the CLI must keep agreeing
    # with what the packages actually export.
    "test_cli.py",
    "test_formula_catalog.py",
    "test_help.py",
    "test_hsds_configure.py",
    "test_mcp_server.py",
    "test_mcp_tools.py",
    "test_plot_discovery.py",
    "test_plot_registry.py",
    "test_plot_submodule.py",
    "test_process_catalog.py",
    "test_setup.py",
    # Layer boundaries. Source-level architecture checks -- no solves, no I/O.
    "contracts/test_machine_mapping_boundaries.py",
    "test_api_layer_boundaries.py",
    "test_code_parameters_writers.py",
    "test_no_bare_downsample.py",
    "test_no_pyplot_outside_plot.py",
    "test_plot_backend_boundaries.py",
    "test_validation_architecture.py",
    # Flux-coordinate conventions, and where a renderer puts a feature because
    # of them. Cheap: a Solov'ev equilibrium and a synthetic GPEC run, no
    # solver. Added after V5P found an island figure drawing its O-points a
    # quarter period and a whole angle convention away from where a Poincare
    # trace puts them, with every shipped test passing -- the kind of defect
    # only an absolute-placement check sees, and which the develop gate has to
    # be able to see.
    "test_gpec_island_geometry.py",
    "test_magnetic_island.py",
    # Machine geometry: source vertices, unknown phi, camera units and mask.
    # Registry, taxonomy and display policy: the vocabulary the rest of the
    # package indexes itself by.
    "test_diagnostic_registry.py",
    "test_diagnostics_interactive.py",
    "test_display_policy.py",
    "test_equilibrium_field_2d.py",
    "test_layout_contract.py",
    "test_line_abscissa.py",
    "test_machine_geometry_registry.py",
    "test_magnetics_spatial.py",
    "test_mirnov_spatial_phase.py",
    "test_parameter_history.py",
    "test_plot_3d_contract.py",
    "test_plot_contract.py",
    "test_plot_intent.py",
    "test_plot_presentation.py",
    "test_plot_taxonomy.py",
    "test_process_magnetics_geometry.py",
    "test_profile_coordinates.py",
    "test_selection_validity.py",
    "test_spectrogram_methods.py",
    # The launch contract every external-code adapter goes through. Stub
    # programs only (`external_code_stubs`); no physics code is ever run.
    # The in-process memory guard beside it: fake cgroup trees and env only.
    # The process-tree stop behind LocalBackend runs small Python/sh trees.
    # Its memory admission and RSS limit (#1460): a 300 MiB Python child, a
    # limit far below it, and a ledger with a fake MemAvailable.
    "test_code_execution.py",
    "test_code_resources.py",
    "test_memory_gate.py",
    "test_process_tree.py",
    "test_slurm_backend.py",
    # Serialization and schema smoke. The ODS/IMAS shapes everything reads and
    # writes, plus the canonical-IDS contract fixtures and the canonical
    # public-database tables (synthetic rows, mocked network).
    "contracts/test_contract_legacy_rejections.py",
    "contracts/test_contract_samples.py",
    "contracts/test_contract_synthetic.py",
    "contracts/test_models_magnetics.py",
    "contracts/test_models_plasma.py",
    "contracts/test_models_uncertainty.py",
    "test_code_parameters_contract.py",
    "test_code_parameters_entry_payload.py",
    "test_database_summary.py",
    "test_dataset_description.py",
    "test_eqdsk_omas_roundtrip.py",
    "test_path_exists.py",
    "test_public_confinement.py",
    "test_public_profile.py",
    "test_public_transition.py",
    "test_shotlog.py",
    "test_turbulent_transport_summary.py",
    # Packaging and documentation policy. Metadata reads; they catch the
    # breakage `package` cannot see until it is already building a wheel.
    "contracts/test_dependency_policy_matrix.py",
    "test_data_resources.py",
    "test_docstring_engine.py",
    "test_formula_docstrings.py",
    "test_notebook_outputs.py",
    "test_packaging_issue45.py",
    "test_process_docstrings.py",
    # ML backbone (#669). The framework-free lifecycle -- group split, fingerprint,
    # hash-pinned artifact, registry resolution -- on the NumPy backend, and the
    # scikit-learn (ONNX) and PyTorch backends, which skip where not installed;
    # and the checker of the vaft-nn registry they resolve published models from.
    "test_check_vaft_nn.py",
    "test_process_ml_contract.py",
    "test_process_ml_sklearn.py",
    "test_process_ml_torch.py",
    # Documentation drift. File reads and getattr only: what the READMEs claim
    # VAFT is, the site's navigation contract, and whether a documented snippet
    # names an API that exists -- a library rename breaks the last without its
    # author ever opening docs/, which is exactly what develop should catch.
    # The committed diagram SVGs are checked against their TikZ source too.
    "test_diagram_render.py",
    "test_docs_api.py",
    "test_docs_catalogs.py",
    "test_docs_content.py",
    "test_docs_snippets.py",
    "test_docs_sources.py",
    "test_docs_thumbnails.py",
    "test_readme_consistency.py",
    # Operational boundaries (#1067): every published limit is called and
    # checked against its source's numbers and its permitted side. Pure NumPy.
    "test_formula_boundaries.py",
    # Operational-space projections (#1425): a boundary is drawn only on its
    # own quantities; the population renderer reads tables, never ODS.
    "test_li_qa.py",
    "test_operational_space.py",
    # Diagram physics: every drawn O-point, drift and field is the formula's.
    # The s-alpha charts are not here: their boundary solves cost ~2.5 min.
    "test_diagram_ballooning.py",
    "test_diagram_blob.py",
    "test_diagram_cold_plasma_waves.py",
    "test_diagram_collision.py",
    "test_diagram_cylindrical_modes.py",
    "test_diagram_disruption.py",
    "test_diagram_divertor_footprint.py",
    "test_diagram_equilibrium_phenomena.py",
    "test_diagram_field_aligned.py",
    "test_diagram_field_configurations.py",
    "test_diagram_geometry.py",
    "test_diagram_gs_equilibrium.py",
    "test_diagram_guiding_center.py",
    "test_diagram_harmonic.py",
    "test_diagram_infrastructure.py",
    "test_diagram_integrated_modeling.py",
    "test_diagram_iteration_dynamics.py",
    "test_diagram_magnetic_island.py",
    "test_diagram_marfe.py",
    "test_diagram_mhd_waves.py",
    "test_diagram_nbi.py",
    "test_diagram_particle_motion.py",
    "test_diagram_platform.py",
    "test_diagram_pwi.py",
    "test_diagram_ripple.py",
    "test_diagram_sfl_coordinates.py",
    "test_diagram_sfl_coordinates_part2.py",
    "test_diagram_slab_parity.py",
    "test_diagram_spatial.py",
    "test_diagram_spectroscopy.py",
    "test_diagram_tearing.py",
    "test_diagram_tokamak_geometry.py",
    "test_diagram_transport_regimes.py",
    "test_diagram_vaft_concepts.py",
    "test_diagram_vde.py",
    "test_diagram_wall_conditioning.py",
    # The new-shot worker (#58): SQLite state, fake SQL and a fake runner only.
    # The per-shot master lock (#913): an in-memory HSDS, ~4 s of threads.
    "test_hsds_master_lock.py",
    "test_pipeline_worker.py",
    # Stability atlas (lane N): real DCON/RDCON output (trimmed netCDF files)
    # read back through the readers, the edge classifier and the ntms mapping,
    # and the #141 scan driver's template patching. No solver runs.
    "test_gpec_dcon_edge_reference.py",
    "test_gpec_rdcon_criteria.py",
    "test_mhd_linear_dcon_payload.py",
    "test_stability_atlas_build.py",
    "test_stability_atlas_controls.py",
    "test_stability_rdcon_stride_benchmark.py",
    # Kinetic state (lane K, #1430/#1454): Thomson against EFIT pressure on
    # synthetic multi-slice equilibria stored out of time order. Pure NumPy.
    "test_kinetic_state.py",
    # Transport atlas (lane T): the shared transport-state resolver on the packaged
    # 48224 ODS made multi-slice with offset times, the TGLF spectrum parser on the
    # reg05 fixture, and the routine driver with a fake runner. No solver runs.
    # The atlas renderers draw synthetic tables only.
    "test_plot_transport_atlas.py",
    "test_transport_state.py",
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
