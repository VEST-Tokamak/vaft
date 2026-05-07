# VAFT Pipeline Notebook Collection

## Purpose

This directory collects first-draft English notebook shells for documenting and organizing the incomplete Snakemake-based VAFT pipeline. The notebooks are intended to describe data sources, intermediate products, physics-analysis stages, and open implementation work before the workflows are fully automated.

Existing notebooks in this directory are preserved. The new notebooks added for the pipeline documentation are intentionally non-executable shells at this stage.

## Relation to the Snakemake-Based VAFT Pipeline

The notebooks are organized around planned Snakemake pipeline stages. Each notebook should eventually explain the expected inputs, expected outputs, validation checks, and downstream dependencies for one pipeline area. The notebooks should not replace Snakemake rules; instead, they should make the rule structure, data contracts, and physics context easier to review.

The expected long-term relationship is:

- Snakemake rules manage reproducible execution, file dependencies, configuration, and batch processing.
- Notebooks document the scientific context, inspect representative outputs, and provide review-friendly examples.
- Shared VAFT source code should hold reusable functions once workflow details are mature enough to implement.

## Recommended Reading Order

1. `vest_raw_signal_sql_database.ipynb`
2. `magnetic_diagnostics_processing.ipynb`
3. `fluctuation_diagnostics_analysis.ipynb`
4. `eddy_current_calculation_and_startup_analysis.ipynb`
5. `electromagnetic_response_modeling_with_efund.ipynb`
6. `magnetic_equilibrium_reconstruction_with_efit.ipynb`
7. `mhd_equilibrium_analysis.ipynb`
8. `linear_ideal_stability_analysis_with_dcon.ipynb`
9. `linear_resistive_stability_analysis_with_rdcon.ipynb`
10. `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`
11. `shot_characteristics_classification.ipynb`
12. `fast_camera_video_analysis.ipynb`
13. `multiple_tokamak_comparison.ipynb`

This order starts with source data and core diagnostics, then moves through electromagnetic response, equilibrium reconstruction, stability analysis, aggregate shot characterization, camera review, and cross-device comparison.

## Current Development Status

These notebooks are initial documentation shells only. They contain Markdown structure for titles, overviews, objectives, expected inputs, expected outputs, pipeline context, main section headings, summaries, open implementation tasks, and related notebooks.

No full workflows, database queries, Snakemake rules, physics calculations, or reusable VAFT source-code changes are implemented in this notebook pass.

## Open Tasks

- Confirm the authoritative Snakemake rule graph and map each notebook to one or more planned rules.
- Define stable input and output schemas for each pipeline stage.
- Add representative shot examples after data-access and privacy constraints are confirmed.
- Move mature reusable logic into VAFT source modules instead of leaving it embedded in notebooks.
- Add validation checks, provenance metadata, and quality-control summaries for notebook examples.
- Decide how notebook outputs should be rendered or archived by the Snakemake pipeline.
- Keep terminology, file naming, units, and coordinate conventions consistent across notebooks.
