# VAFT Notebook Collection

## Purpose

These notebooks show what research VAFT supports, worked end to end on real
VEST data. Where [`tutorial/`](../tutorial/README.md) teaches the workflow one
session at a time, this collection is the reference: each notebook takes a
scientific question and follows it from measurement to result.

The collection mixes mature workflows with first-draft shells for the
Snakemake-based pipeline that is still being built. The shells define the
intended structure of future reproducible workflows; the executable notebooks
provide practical context for database access, OMAS/IMAS conversion, plotting,
monitoring, profile fitting, confinement scaling, and publication figures.

> **Index in transition.** The groups below are organized by pipeline stage.
> Issue [#330](https://github.com/VEST-Tokamak/vaft/issues/330) reorganizes them
> by research question — discharge formation, diagnostic interpretation,
> equilibrium reconstruction, confinement, MHD and transient events, kinetic
> profiles, database-scale analysis. Until that lands, treat the grouping as
> structural rather than scientific.

## Relation to the Snakemake-Based VAFT Pipeline

The long-term goal is to use Snakemake for reproducible execution, dependency tracking, batch processing, and file organization. Notebooks should support that pipeline by documenting scientific context, inspecting representative data products, and clarifying interfaces between workflow stages.

The intended division of responsibility is:

- Snakemake rules manage execution, input/output dependencies, configuration, and batch processing.
- VAFT source modules provide reusable database, signal-processing, equilibrium, stability, and plotting functions.
- Notebooks explain workflow intent, inspect representative cases, validate intermediate products, and record open implementation tasks.

Pipeline notebooks are expanded as the reusable VAFT functions behind them become stable; the ones that remain documentation shells are the five that need an external Fortran code, and each says so in its own **Requirements to run this page** section.

## Notebook Groups

### Database, Data Structure, and Conversion

- `initialize_external_fusion_codes.ipynb`: External-code root initialization, executable layout, and validation.
- `database_initialization_and_load.ipynb`: Existing guide for VAFT library setup and VEST database loading.
- `vest_raw_signal_sql_database.ipynb`: The VEST MySQL raw-signal database — table and field naming, shot organization, and 1D signal loading, worked against the packaged raw archive so the structure can be read without a database connection.
- `vest_experimental_data_list.ipynb`: Existing VEST OMAS initial guide and experimental data overview.
- `read_and_convert_data_structure.ipynb`: Existing notebook for reading and converting structured equilibrium or diagnostic data. 
- `imas_omas_data_conversion.ipynb`: Existing notebook for IMAS/OMAS data conversion.

### Core Diagnostic and Startup Pipeline

- `magnetic_diagnostics_processing.ipynb`: Raw magnetic diagnostics from acquisition to processed signal — calibration, filtering, and the stage-by-stage waveforms — including a worked diamagnetic-Rogowski acquisition-saturation section (issue #285) showing raw and integrated signals, original vs corrected, on the packaged reference shots.
- `fluctuation_diagnostics_analysis.ipynb`: Fluctuation spectral analysis — Welch PSD, power-law spectral index, spectral breaks, band powers and spectrograms — with the theory behind each routine, demonstrated on VEST magnetic probes and soft X-rays.
- `soft_x_ray_signal_analysis.ipynb`: VEST SXR workflow — LOS geometry, traces, spectrogram, chord-time patterns, plus band-decomposed chord maps, optional vacuum-shot PF-noise subtraction, Be/Al two-filter electron temperature, and a two-point toroidal mode-number estimate ported from the validated VEST SXR Viewer.
- `eddy_current_calculation_and_startup_analysis.ipynb`: PF passive eddy-current solve on the packaged shot — circuit assembly from the machine description, the induced currents written back into the ODS, and the vacuum field they produce, checked against the flux loops in the plasma-free window and read as loop voltage, decay index, the midplane null and a 2D null map.
- `fast_camera_video_analysis.ipynb`: VEST FAST-camera frames from the packaged sample — loading, time synchronization, the calibration geometry projected onto a real frame, and the equilibrium and field-line overlays that read plasma behaviour off it.

### Electromagnetic Response, Equilibrium, and Stability

- `electromagnetic_response_modeling_with_efund.ipynb`: Blocked on EFUND, which ships inside an EFIT build; the notebook documents what it needs and what `vaft.process.electromagnetics` already derives without it.
- `magnetic_equilibrium_reconstruction_with_efit.ipynb`: Blocked on an EFIT installation; the notebook documents how to obtain and configure one, and which parts of the workflow — constraints, k-files, the parameter grid — `vaft.code.efit` runs without it.
- `forward_equilibrium_using_TokaMaker.ipynb`: Forward free-boundary equilibrium with TokaMaker (Open FUSION Toolkit) driven by measured PF currents.
- `time_dependent_equilibrium_using_TokaMaker.ipynb`: VEST vessel eddy currents, wall eigenmodes, quasi-static shot evolution, and vertical-stability growth rates with TokaMaker.
- `free_boundary_pf_coil_scan.ipynb`: Free-boundary PF-coil-current scans with TokaMaker — commanded/materialized currents, per-case topology classification (limited/near-null/SN/DN), continuation with manifests and resume.
- `mhd_equilibrium_analysis.ipynb`: One equilibrium end to end — convention validation, global descriptors with provenance, derived profiles, flux-surface averages, the three radial coordinates, rational surfaces and a traced field line, closing with a GEQDSK handoff whose descriptors round-trip.
- `parametric_equilibrium_descriptors.ipynb`: Convention-aware global descriptors and GEQDSK/ODS parity.
- `local_miller_equilibrium_fitting.ipynb`: Local Miller fitting, reconstruction errors, and separatrix limits.
- `analytic_solovev_equilibrium.ipynb`: Constant-source analytic Solov'ev construction and gridded-field verification.
- `edge_and_boundary_representation.ipynb`: Limiter/diverted topology, X-points, gaps, and separatrix balance.
- `linear_ideal_stability_analysis_with_dcon.ipynb`: Blocked on a GPEC-suite build; the notebook documents the `GPECHOME` layout and the case preparation and output validation `vaft.code.gpec` does without the solver.
- `linear_resistive_stability_analysis_with_rdcon.ipynb`: Blocked on the same GPEC-suite build, for RDCON and its matching data.
- `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`: Blocked on GPEC itself; the notebook notes that its output readers work on results produced elsewhere, so a run from another machine can still be analysed here.
- `vest_nbi_analysis_with_nubeam.ipynb`: Neutral-beam deposition, heating, current drive and loss accounting for VEST with NUBEAM. Needs a completed NUBEAM run, named by `VAFT_NUBEAM_RUN_DIR`; without one each section reports what it would show and skips.

### Analysis, Visualization, Reporting, and Comparison

- `plotting_sample_using_vaft_plot_module.ipynb`: Existing examples for plotting sample data with the VAFT plot module.
- `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`: Existing profile-fitting and kinetic-diagnostic example notebook.
- `confinement_time_scaling.ipynb`: Existing confinement time scaling analysis notebook.
- `tokamak_power_balance.ipynb`: Radiation loss channels on one power-density basis — a temperature scan at assumed flat density, then the same channels integrated over the measured Thomson profiles of the packaged kinetic-EFIT sample (shot 48224 at 300 ms), which the flat estimate underestimates by a factor of two.
- `shot_characteristics_classification.ipynb`: Per-shot feature records from the packaged shots — timing with its detector and agreement, equilibrium descriptors, a class label with its threshold sensitivity shown, a review state beside the automatic proposal, and the summary table an aggregation rule would write.
- `vest_daily_monitoring.ipynb`: Existing daily monitoring notebook for VEST data review.
- `multiple_tokamak_comparison.ipynb`: Cross-device comparison against public upstream data — VEST, DIII-D, MAST-U, JET, TCV and SPARC equilibria fetched from their own repositories as IMAS netCDF, ODS JSON and GEQDSK, loaded through one `vaft.omas.load` path, then compared as physical and normalized geometry, global descriptors, COCOS conventions and profiles.
- `publication_figures.ipynb`: Publication figures built from packaged data — three reconstructions of shot 48224 at 300 ms (EFIT, its CHEASE refinement, kinetic EFIT) overlaid through the canonical renderers, and a Mirnov spectrogram of shot 45531 from the packaged raw archive. Sections needing the external stability history report that and skip.

## Recommended Reading Order

Use the following order as the main technical path through the notebooks. Existing notebooks are included where they provide useful setup, reference material, or downstream analysis context.

1. `initialize_external_fusion_codes.ipynb`
2. `database_initialization_and_load.ipynb`
3. `vest_raw_signal_sql_database.ipynb`
4. `vest_experimental_data_list.ipynb`
5. `read_and_convert_data_structure.ipynb`
6. `imas_omas_data_conversion.ipynb`
7. `magnetic_diagnostics_processing.ipynb`
8. `fluctuation_diagnostics_analysis.ipynb`
9. `eddy_current_calculation_and_startup_analysis.ipynb`
10. `electromagnetic_response_modeling_with_efund.ipynb`
11. `magnetic_equilibrium_reconstruction_with_efit.ipynb`
12. `mhd_equilibrium_analysis.ipynb`
13. `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`
14. `linear_ideal_stability_analysis_with_dcon.ipynb`
15. `linear_resistive_stability_analysis_with_rdcon.ipynb`
16. `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`
17. `vest_nbi_analysis_with_nubeam.ipynb`
18. `plotting_sample_using_vaft_plot_module.ipynb`
19. `shot_characteristics_classification.ipynb`
20. `vest_daily_monitoring.ipynb`
21. `fast_camera_video_analysis.ipynb`
22. `confinement_time_scaling.ipynb`
23. `multiple_tokamak_comparison.ipynb`
24. `publication_figures.ipynb`

For a shorter review focused only on the notebooks still waiting on an external
Fortran code, read their **Requirements to run this page** sections:

1. `magnetic_equilibrium_reconstruction_with_efit.ipynb`
2. `electromagnetic_response_modeling_with_efund.ipynb`
3. `linear_ideal_stability_analysis_with_dcon.ipynb`
4. `linear_resistive_stability_analysis_with_rdcon.ipynb`
5. `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`

## Current Development Status

The notebook collection is mixed in maturity:

- Existing notebooks contain exploratory examples, setup notes, plotting demonstrations, conversion tests, monitoring views, and analysis prototypes.
- Five notebooks remain shells, every one of them blocked on an external Fortran code rather than on a missing VAFT API: EFIT, EFUND, DCON, RDCON and GPEC. Each carries a **Requirements to run this page** section naming the code, the environment variable and executable layout it needs, how to check readiness, and which parts of its workflow VAFT already performs without the binary. Building those codes is tracked in [#226](https://github.com/VEST-Tokamak/vaft/issues/226).
- The Snakemake-based VAFT pipeline is still incomplete, so notebook sections should be treated as design documentation until the corresponding rules and source modules are implemented.

## Open Tasks

- Map every notebook to the corresponding Snakemake rule, source module, or analysis responsibility.
- Confirm authoritative input and output schemas for database signals, processed diagnostics, equilibrium files, stability outputs, camera data, and summary spreadsheets.
- Standardize terminology, file naming, units, coordinate conventions, time-base conventions, and sign conventions across old and new notebooks.
- Decide which existing exploratory code should be moved into reusable VAFT source modules.
- Add representative shot examples only after data-access, privacy, and reproducibility requirements are confirmed.
- Add validation checks, provenance metadata, and quality-control summaries for each pipeline stage.
- Decide how notebooks should be executed, rendered, or archived by the Snakemake pipeline.
- ##### Keep existing notebooks available while gradually aligning them with the pipeline documentation structure.
