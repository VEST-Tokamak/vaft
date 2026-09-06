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

The newly added pipeline notebooks are documentation shells only. They should be expanded gradually as Snakemake rules and reusable VAFT functions become stable.

## Notebook Groups

### Database, Data Structure, and Conversion

- `initialize_external_fusion_codes.ipynb`: External-code root initialization, executable layout, and validation.
- `database_initialization_and_load.ipynb`: Existing guide for VAFT library setup and VEST database loading.
- `vest_raw_signal_sql_database.ipynb`: Planned documentation for the VEST MySQL raw-signal database structure and 1D signal loading.
- `vest_experimental_data_list.ipynb`: Existing VEST OMAS initial guide and experimental data overview.
- `read_and_convert_data_structure.ipynb`: Existing notebook for reading and converting structured equilibrium or diagnostic data. 
- `imas_omas_data_conversion.ipynb`: Existing notebook for IMAS/OMAS data conversion.

### Core Diagnostic and Startup Pipeline

- `magnetic_diagnostics_processing.ipynb`: Planned raw magnetic diagnostics processing, calibration, filtering, and processed signal format. Includes a worked diamagnetic-Rogowski acquisition-saturation section (issue #285) showing raw and integrated signals, original vs corrected, on the packaged reference shots.
- `fluctuation_diagnostics_analysis.ipynb`: Fluctuation spectral analysis — Welch PSD, power-law spectral index, spectral breaks, band powers and spectrograms — with the theory behind each routine, demonstrated on VEST magnetic probes and soft X-rays.
- `soft_x_ray_signal_analysis.ipynb`: VEST SXR workflow — LOS geometry, traces, spectrogram, chord-time patterns, plus band-decomposed chord maps, optional vacuum-shot PF-noise subtraction, Be/Al two-filter electron temperature, and a two-point toroidal mode-number estimate ported from the validated VEST SXR Viewer.
- `eddy_current_calculation_and_startup_analysis.ipynb`: Planned PF passive eddy-current calculation and startup analysis.
- `fast_camera_video_analysis.ipynb`: Planned VEST camera image/video loading, synchronization, and visual plasma behavior analysis.

### Electromagnetic Response, Equilibrium, and Stability

- `electromagnetic_response_modeling_with_efund.ipynb`: Planned EFUND-aligned electromagnetic response modeling and coupling matrix construction.
- `magnetic_equilibrium_reconstruction_with_efit.ipynb`: Planned EFIT-based magnetic equilibrium reconstruction.
- `forward_equilibrium_using_TokaMaker.ipynb`: Forward free-boundary equilibrium with TokaMaker (Open FUSION Toolkit) driven by measured PF currents.
- `time_dependent_equilibrium_using_TokaMaker.ipynb`: VEST vessel eddy currents, wall eigenmodes, quasi-static shot evolution, and vertical-stability growth rates with TokaMaker.
- `free_boundary_pf_coil_scan.ipynb`: Free-boundary PF-coil-current scans with TokaMaker — commanded/materialized currents, per-case topology classification (limited/near-null/SN/DN), continuation with manifests and resume.
- `mhd_equilibrium_analysis.ipynb`: Planned equilibrium loading, coordinate transformation, flux-surface analysis, and MHD interpretation.
- `parametric_equilibrium_descriptors.ipynb`: Convention-aware global descriptors and GEQDSK/ODS parity.
- `local_miller_equilibrium_fitting.ipynb`: Local Miller fitting, reconstruction errors, and separatrix limits.
- `analytic_solovev_equilibrium.ipynb`: Constant-source analytic Solov'ev construction and gridded-field verification.
- `edge_and_boundary_representation.ipynb`: Limiter/diverted topology, X-points, gaps, and separatrix balance.
- `linear_ideal_stability_analysis_with_dcon.ipynb`: Planned linear ideal MHD stability analysis using DCON from the GPEC package.
- `linear_resistive_stability_analysis_with_rdcon.ipynb`: Planned linear resistive MHD stability analysis using RDCON from the GPEC package.
- `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`: Planned perturbed equilibrium and 3D response analysis using GPEC.
- `vest_nbi_analysis_with_nubeam.ipynb`: Neutral-beam deposition, heating, current drive and loss accounting for VEST with NUBEAM. Needs a completed NUBEAM run, named by `VAFT_NUBEAM_RUN_DIR`; without one each section reports what it would show and skips.

### Analysis, Visualization, Reporting, and Comparison

- `plotting_sample_using_vaft_plot_module.ipynb`: Existing examples for plotting sample data with the VAFT plot module.
- `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`: Existing profile-fitting and kinetic-diagnostic example notebook.
- `confinement_time_scaling.ipynb`: Existing confinement time scaling analysis notebook.
- `shot_characteristics_classification.ipynb`: Planned representative signal extraction, shot classification, and summary spreadsheet generation.
- `vest_daily_monitoring.ipynb`: Existing daily monitoring notebook for VEST data review.
- `multiple_tokamak_comparison.ipynb`: Cross-device comparison against public upstream data — VEST, DIII-D, MAST-U, JET, TCV and SPARC equilibria fetched from their own repositories as IMAS netCDF, ODS JSON and GEQDSK, loaded through one `vaft.omas.load` path, then compared as physical and normalized geometry, global descriptors, COCOS conventions and profiles.
- `publication_figures.ipynb`: Existing or planned notebook for publication figure preparation.

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

For a shorter review focused only on the planned Snakemake pipeline shells, read:

1. `vest_raw_signal_sql_database.ipynb`
2. `magnetic_diagnostics_processing.ipynb`
3. `eddy_current_calculation_and_startup_analysis.ipynb`
4. `electromagnetic_response_modeling_with_efund.ipynb`
5. `magnetic_equilibrium_reconstruction_with_efit.ipynb`
6. `mhd_equilibrium_analysis.ipynb`
7. `linear_ideal_stability_analysis_with_dcon.ipynb`
8. `linear_resistive_stability_analysis_with_rdcon.ipynb`
9. `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`
10. `shot_characteristics_classification.ipynb`
11. `fast_camera_video_analysis.ipynb`

## Current Development Status

The notebook collection is mixed in maturity:

- Existing notebooks contain exploratory examples, setup notes, plotting demonstrations, conversion tests, monitoring views, and analysis prototypes.
- New pipeline notebooks are first-draft Markdown shells and do not yet implement full workflows.
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
