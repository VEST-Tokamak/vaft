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
- `analytic_island_model_and_synthetic_response_model.ipynb`: Forward model of an analytic magnetic island (#886). Resonant surface from `q = m/n`, the PEST straight-field-line angle, the helical flux and separatrix, island-only and flattening emissivity, exact path-length line integrals through the VEST SXR chords, and a rigidly rotating island. The same island spec is placed on an analytic Solov'ev equilibrium and on its CHEASE refinement, which is a cached fixture, so the notebook runs offline.
- `eddy_current_calculation_and_startup_analysis.ipynb`: PF passive eddy-current solve on the packaged shot — circuit assembly from the machine description, the induced currents written back into the ODS, and the vacuum field they produce, checked against the flux loops in the plasma-free window and read as loop voltage, decay index, the midplane null and a 2D null map.
- `fast_camera_video_analysis.ipynb`: VEST FAST-camera frames from the packaged sample — loading, time synchronization, the calibration geometry projected onto a real frame, and the equilibrium and field-line overlays that read plasma behaviour off it.

### Electromagnetic Response, Equilibrium, and Stability

- `electromagnetic_response_modeling_with_efund.ipynb`: Blocked on EFUND, which ships inside an EFIT build; the notebook documents what it needs and what `vaft.process.electromagnetics` already derives without it.
- `magnetic_equilibrium_reconstruction_with_efit.ipynb`: Blocked on an EFIT installation; the notebook documents how to obtain and configure one, and which parts of the workflow — constraints, k-files, the parameter grid — `vaft.code.efit` runs without it.
- `convergence_study_of_efit.ipynb`: EFIT's Picard trajectory per slice (issue #1038) — chi-square, flux increment and magnetic-axis height at every iteration, what the cumulative iteration counter means, and the same slice under `NXITER=1` and `NXITER=3` — read from saved iteration histories, so it runs without EFIT.
- `forward_equilibrium_using_TokaMaker.ipynb`: Forward free-boundary equilibrium with TokaMaker (Open FUSION Toolkit) driven by measured PF currents.
- `time_dependent_equilibrium_using_TokaMaker.ipynb`: VEST vessel eddy currents, wall eigenmodes, quasi-static shot evolution, and vertical-stability growth rates with TokaMaker.
- `free_boundary_pf_coil_scan.ipynb`: Free-boundary PF-coil-current scans with TokaMaker — commanded/materialized currents, per-case topology classification (limited/near-null/SN/DN), continuation with manifests and resume.
- `conference_operational_space_atlas.ipynb`: Conference operational-space atlas (#1456). The #1331 Tier A good and admissible EFIT states on β_N–l_i, Troyon, q95–l_i and the Wesson 1989 l_i–q_ψ plane, with registered boundaries drawn only on their own quantities, coloured by Lane K's R_W and by the Lane N/T atlas layers when they are present. Reads `$VAFT_ATLAS_DIR`; not offline.
- `mhd_equilibrium_analysis.ipynb`: One equilibrium end to end — convention validation, global descriptors with provenance, derived profiles, flux-surface averages, the three radial coordinates, rational surfaces and a traced field line, closing with a GEQDSK handoff whose descriptors round-trip. The mode-resolved Grad-Shafranov residual on nested surfaces against an exact Solov'ev floor (#948).
- `parametric_equilibrium_descriptors.ipynb`: Convention-aware global descriptors and GEQDSK/ODS parity. Toroidal current-density moments: total current, centroid, covariance and odd moments of the sample and of the analytic topologies (#943).
- `local_miller_equilibrium_fitting.ipynb`: Local Miller fitting, reconstruction errors, and separatrix limits. Bean-shaped surfaces: the inboard indentation coefficient, its analytic onset and opt-in fitting (#941).
- `analytic_solovev_equilibrium.ipynb`: Constant-source analytic Solov'ev construction and gridded-field verification; prescribed X-point topology (smooth, double null, single null) through the Cerfon–Freidberg constraint path, saddle checks, gridded X-point recovery and the near-separatrix limit of Miller fits (#938); analytic L-/H-mode profile kernels (generalized parabolic, Groebner mtanh) and how a 1-D profile differs from the exact Solov'ev sources (#552); a VEST-like analytic baseline initialized from the 39915 EFIT boundary and solved through physical constraints, carried through EquilibriumData → GEQDSK → ODS with canonical plots, Miller characterization, equilibrium and MHD-relevant descriptors, and geometry (kappa, delta, a/R0) and source (Cerfon–Freidberg A, Ip) scans (#883); a pressure scan (direct p' at fixed FF', and at fixed boundary, F_boundary and Ip) tracking beta_p, beta_t, li, q and the LCFS-bounded toroidal-flux perturbation to its paramagnetic–diamagnetic zero crossing, beta_p,zero found by root finding and compared with the large-aspect-ratio beta_p ~ 1 (#1198).
- `analytic_guazzotto_freidberg_equilibrium.ipynb`: Guazzotto–Freidberg (2021, Part 1) analytic equilibria with quadratic p and F² -- vanishing edge pressure and current -- as an eigenvalue problem: the paper's Table 4 reproduced with its three discrepancies noted, a VEST-like shape in all three topologies, the midplane contrast with Solov'ev, and the nu ≈ beta_p scan (#1148).
- `analytic_fixed_to_free_boundary.ipynb`: Public-API Solov’ev and Guazzotto–Freidberg PF-current fitting through independent direct Green and TokaMaker `get_vfixed()` routes, with free-boundary closure, optional shape refinement, and final current-frozen verification (#1608). Native solves run only when enabled on an OFT host.
- `analytic_plasma_state_presets.ipynb`: Analytic L-mode, H-mode (edge pedestal), ITB and H-mode + ITB kinetic states on one fixed Solov'ev geometry -- `n_e`, `T_e`, `T_i` from the #552 kernels, pressure and its gradient derived from them, barrier position and width semantics, per-channel ITBs, the `f(psi_N) -> f(R, Z)` projection on two geometries, and why none of these states is Grad-Shafranov self-consistent (#1045).
- `synthetic_kinetic_profiles_from_equilibrium.ipynb`: Synthetic `n_e`, `T_e`, `T_i` and ion densities from the 39915 g-file and an analytic Solov'ev equilibrium through the #122 fidelity ladder (Level 0 legacy sqrt split, Level 1 analytic shapes, Level 2 scalar normalization, Level 3 prescribed a/L) -- equilibrium-constrained, thermal-energy and kinetic pressure closures with residual tables, quasi-neutrality and Z_eff with one impurity, refused and failed assumptions, the `core_profiles` output, and what is assumed vs derived; nothing is transport-predicted.
- `self_consistent_equilibrium_kinetic_iteration.ipynb`: An assumption-driven self-consistent equilibrium and kinetic-profile state on the 39915 g-file (#123) -- the `equilibrium_pressure` mode (p_eq decomposed by #122, no CHEASE) and the `kinetic_pressure` mode (CHEASE re-solved for the kinetic pressure, profiles regenerated from the same spec, repeat), with the iteration history of pressure, profile, q, coordinate-map and scalar residuals, initial vs final p, q, n_e and rho_tor, the held/recomputed table, three current policies (initial FF' shape or an analytic g, I_p or q95 held), a refused hollow pressure and an iteration cap, and the equilibrium + core_profiles ODS. Needs a CHEASE executable for the solves (coarse demo mesh); without one those cells skip.
- `compact_equilibrium_representation.ipynb`: Compact representations of an existing equilibrium with explicit fidelity -- the best-fit Solov'ev model and the MXH-Chebyshev representation (Xie & Li 2026), with its error against the number of parameters (#1166).
- `equilibrium_representation_reference.ipynb`: One traceable equilibrium (the packaged 39915 EFIT slice at 319 ms, COCOS 11, psi in Wb) carried through the #1201 chain -- psi_N, rho_pol, rho_tor and r/a kept distinct, the R-Z flux map, a PEST grid validated by recomputing q, Miller and Fourier fits with residuals, synthetic kinetic profiles and a/L_T in four gradient conventions, rational surfaces, a prescribed 3/1 island checked against a traced field line, a 3-D embedding and the calibrated camera projection -- with a same-state check at every step and "is not the same as" callouts. The GPEC response is a stated gap (needs GPEC, no packaged output).
- `equilibrium_refinement_using_chease.ipynb`: CHEASE fixed-boundary refinement of an EFIT g-file, qualified: the boundary and profiles CHEASE is given, what the solve holds and changes, and solver-mesh convergence. Needs a CHEASE executable for the solves; input preparation runs without one.
- `fixed_boundary_parametric_scan_using_chease.ipynb`: Local sensitivity of that equilibrium to pressure, current peaking and shape, as requested, materialized and achieved, with failed cases kept. Needs a CHEASE executable for the scans.
- `generate_0d_synthetic_equilibrium_using_chease.ipynb`: A fixed-boundary CHEASE equilibrium synthesized from 0D descriptors (R0, a, kappa, delta, Bt, Ip) and explicit p'/FF' source shapes; current or q95 normalization, requested vs achieved descriptors, non-uniqueness under different source assumptions, H-mode and ITB barrier pressure profiles (and an edge-current FF') as source shapes with requested vs achieved p, p' and q (#1166 scope B), the refused targets (#120), and an outer solve that meets Ip with q95 and/or a beta target at once by moving the source knobs, with its iteration history and a not_reachable case (#120 multi-target).
- `edge_and_boundary_representation.ipynb`: Limiter/diverted topology, X-points, gaps, and separatrix balance. Internal flux surfaces as arc-length Fourier series: symmetric and asymmetric harmonic profiles, truncation error, and Miller vs Fourier on a single null (#945). Seven benchmark shapes separating "not Miller-like" from "not representable", with model-independent concavity and asymmetry observables, and Contour-based CHEASE scan boundaries (#942).
- `linear_ideal_stability_analysis_with_dcon.ipynb`: Ideal MHD stability (delta-W by toroidal mode) with DCON, run on the packaged VEST equilibrium into a temporary directory, mapped into `mhd_linear` and drawn through the plot catalog. Needs `$GPECHOME`; without it each section reports what it would show and skips.
- `linear_resistive_stability_analysis_with_rdcon.ipynb`: Resistive stability with RDCON -- the classical tearing index Delta-prime per rational surface, mapped into `ntms.deltaw`, with DCON alongside for the ideal context. Needs the same `$GPECHOME`.
- `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`: Blocked on GPEC itself; the notebook notes that its output readers work on results produced elsewhere, so a run from another machine can still be analysed here.
- `vest_nbi_analysis_with_nubeam.ipynb`: Neutral-beam deposition, heating, current drive and loss accounting for VEST with NUBEAM. Runs the case stored in `vaft/data/nubeam/vest_case` into a temporary directory; needs `$NUBEAMHOME`, and without it each section reports what it would show and skips.
- `neoclassical_transport_with_neo.ipynb`: Neoclassical transport and bootstrap current for VEST with NEO (GACODE). Converts the packaged 48224 kinetic state to `input.gacode`, runs NEO into a temporary directory, maps the result into IMAS, and compares it with the Sauter and Redl analytic models; needs `$GACODEHOME` and `$GACODE_PLATFORM`, and without them each section reports what it would show and the analytic comparison still runs.
- `turbulent_transport_with_tglf.ipynb`: Turbulent transport for VEST with TGLF (GACODE) and the TGLF-NN surrogate. Builds the local TGLF input at five surfaces from the packaged 48224 kinetic state, runs TGLF, and audits every public TGLF-NN family against the same input -- none is in domain, because VEST's ion-to-electron temperature ratio is an order of magnitude below every training set. Needs `$GACODEHOME` and `$GACODE_PLATFORM` for the native run and `$TURBULENTTRANSPORTHOME` for the models; the audit needs neither, and without any of them the temperature-ratio finding still runs.

### Analysis, Visualization, Reporting, and Comparison

- `plotting_sample_using_vaft_plot_module.ipynb`: Existing examples for plotting sample data with the VAFT plot module.
- `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`: Existing profile-fitting and kinetic-diagnostic example notebook.
- `confinement_time_scaling.ipynb`: Existing confinement time scaling analysis notebook.
- `tokamak_power_balance.ipynb`: Radiation loss channels on one power-density basis — a temperature scan at assumed flat density, then the same channels integrated over the measured Thomson profiles of the packaged kinetic-EFIT sample (shot 48224 at 300 ms), which the flat estimate underestimates by a factor of two.
- `shot_characteristics_classification.ipynb`: Per-shot feature records from the packaged shots — timing with its detector and agreement, equilibrium descriptors, a class label with its threshold sensitivity shown, a review state beside the automatic proposal, and the summary table an aggregation rule would write.
- `vest_daily_monitoring.ipynb`: Existing daily monitoring notebook for VEST data review.
- `multiple_tokamak_comparison.ipynb`: Cross-device comparison against public upstream data — VEST, DIII-D, MAST-U, JET, TCV and SPARC equilibria fetched from their own repositories as IMAS netCDF, ODS JSON and GEQDSK, loaded through one `vaft.omas.load` path, then compared as physical and normalized geometry, global descriptors, COCOS conventions and profiles.
- `multi_machine_database_comparison.ipynb`: Published multi-machine databases mapped into common VAFT semantics (#1205) and compared with VEST through the same APIs — ITPA DB5.2.3 H-mode confinement (IPB98(y,2) reproduces the database's own H-factor), the public TCV and ITPA TC-26 L-H transition sets as one event table, and ITPA PR08 profiles mapped to `core_profiles`, `equilibrium`, `core_sources` and `core_transport` without interpolation. Downloads are checksum-pinned; TC-26 is read from a local copy set in `VAFT_TC26_CSV`.
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
12. `convergence_study_of_efit.ipynb`
13. `mhd_equilibrium_analysis.ipynb`
14. `profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`
15. `linear_ideal_stability_analysis_with_dcon.ipynb`
16. `linear_resistive_stability_analysis_with_rdcon.ipynb`
17. `perturbed_equilibrium_and_3d_response_with_gpec.ipynb`
18. `vest_nbi_analysis_with_nubeam.ipynb`
19. `neoclassical_transport_with_neo.ipynb`
20. `turbulent_transport_with_tglf.ipynb`
21. `plotting_sample_using_vaft_plot_module.ipynb`
22. `shot_characteristics_classification.ipynb`
23. `vest_daily_monitoring.ipynb`
24. `fast_camera_video_analysis.ipynb`
25. `confinement_time_scaling.ipynb`
26. `multiple_tokamak_comparison.ipynb`
27. `multi_machine_database_comparison.ipynb`
28. `publication_figures.ipynb`

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
