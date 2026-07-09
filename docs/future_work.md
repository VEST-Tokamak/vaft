# VAFT Future Work Roadmap

This roadmap consolidates the future work stated in the platform paper
(Yun *et al* 2025 *Plasma Phys. Control. Fusion* **67** 115021) with the
current state of this repository (branch `new_pipeline_with_notebooks`).
Items are grouped by the four pillars the platform aims to unify:
experimental data integration, simulation-routine stabilization, code
interoperability, and education/collaborative archiving.

Legend: **[P §x]** = commitment stated in the paper section x, **[R]** =
gap observed in the repository.

## 1. Experimental data integration and machine mapping

- **Map remaining diagnostics into the IMAS database** [P §4.1]:
  Ion Doppler Spectroscopy / CES (a `charge_exchange` mapping exists but
  still reads ad-hoc `.mat` exports), helicity injection, mm-wave
  interferometer, filterscopes beyond H/C/O, fast camera video. Target:
  every routinely acquired signal reachable through
  `vaft.machine_mapping` with pin-to-pin channel provenance.
- **Contribute missing IDS definitions upstream** [P §4.1]: specialized
  VEST systems (e.g. helicity injection) have no home in IMAS DD 3.41.0;
  propose PDM entries or document interim `code.parameters` conventions.
- **Formalize the manual-upload pathway** [P §4.1]: predefined formats,
  automatic schema validation on upload, and a documented validation
  status for each dataset.
- **Data maturity labels** [P §4.1]: attach `preliminary` / `calibrated`
  / `validated` flags per IDS entry so downstream users can filter by
  reliability; surface these in `vaft.database` load functions.
- **Data governance** [P §4.1]: sharing, authentication, and licensing
  procedures for the HSDS server (per-user ACLs, external collaborator
  onboarding, dataset DOIs for publications).
- **Systematic anomaly screening at intake** [P §4.1]: promote the
  current statistical outlier detection / filtering into a documented,
  configurable QC stage of the routine pipeline, with per-shot QC
  reports archived alongside the data.
- **Backfill and channel-history tracking** [R]: encode instrument
  setting changes (calibration factors, DAQ specs, channel remapping
  across maintenance periods) as versioned configuration rather than
  shot-number `if` branches in `workflow/*/config.yaml` and
  `vaft/machine_mapping/magnetics.py`.

## 2. Equilibrium and simulation routine stabilization

- **Periodic EM-model recalibration** [P §4.2]: automate refitting of
  passive-structure resistance parameters from dedicated vacuum shots as
  a scheduled Snakemake rule; alert when model-vs-measurement residuals
  drift (hardware changes, plasma-wall deformation).
- **EFIT convergence for high-ramp discharges** [P §5, §4.3]: shots
  above ~213 kA were excluded from the paper's operational-space
  analysis because reconstructions failed to converge. Work items:
  refined eddy-current profile inputs, basis-function/weighting scans,
  hollow-J_phi-aware initial guesses, and systematic convergence
  diagnostics (extend `workflow/automatic_pipeline_3_data_summary/`
  reliability tooling into pass/fail flags stored in the equilibrium
  IDS).
- **Kinetic equilibrium reconstruction** [P §4.3]: incorporate T_i,
  v_phi, Z_eff measurements; uncertainty-weighted profile fitting with
  physically consistent basis functions; multi-diagnostic integration
  for the same parameter; eventual internal constraints (e.g. MSE-class
  measurements) to constrain q-profiles.
- **Integrated data analysis (IDA) with uncertainty propagation**
  [P §6]: end-to-end statistical uncertainty quantification with
  informed priors propagated through diagnostics → equilibrium →
  stability, building an operational/physics parameter database with
  error bars.
- **Surrogate models for equilibrium solvers** [P §6]: deep-learning
  surrogates for EFIT/VFIT to mitigate convergence failures and enable
  fast between-shot reconstruction; predictive models for IRE
  identification.
- **Validation against fluctuation diagnostics** [P §4.4]: current
  equilibria cannot yet reproduce Mirnov spectrogram trends; establish a
  routine comparison between DCON/RDCON predictions and measured mode
  activity (the new `vaft.plot.mirnov` / `soft_x_rays` modules are the
  starting point).
- **Density-limit analysis** [P §5.2]: Hugill-diagram study pending
  accumulation of interferometer and Thomson density statistics.

## 3. Code coupling and data interoperability

- **Uniform adapter contract** [R]: `vaft.code.base.CodeRunner` now
  covers EFIT, CHEASE, GPEC (DCON/RDCON), and TES; document the
  prepare/run/collect contract and port remaining ad-hoc scripts in
  `workflow/` onto it. Candidate next adapters: transport and
  forward-modeling codes used with the platform (OMFIT modules,
  TRIASSIC, FUSE).
- **IMAS DD version strategy** [R]: samples and database entries pin DD
  3.40.x/3.41.0; define a migration path to DD 4.x and keep
  `vaft.imas` (OMAS ↔ imas-python AL5) round-trip tested.
- **Coordinate and grid interpolation utilities** [R]: consolidate
  psi_N / rho_tor mapping, straight-field-line (PEST) conversion
  (currently a prototype in `test/test_sfl_conversion.py`), and
  time-base resampling into `vaft.process` with unit tests, so data can
  be exchanged between codes on any grid.
- **Convention enforcement** [R]: single source of truth for COCOS,
  sign, units, and time-base conventions across old and new modules
  (also flagged in `notebooks/README.md` open tasks).
- **Packaged reference data** [R]: `vaft/data` is now organized by
  domain (efit / gpec / imas / omas / geometry / legacy) with
  `vaft.data.resources` accessors; keep large binaries in Git LFS and
  publish bulk datasets externally (e.g. Zenodo DOI) instead of the
  package.

## 4. Education and collaborative archiving

- **Complete the pipeline notebook shells** [R]: 13 of the notebooks in
  `notebooks/` are markdown-only design shells; implement them against
  packaged sample data so each Snakemake stage has an executable
  tutorial.
- **Executed-documentation CI** [R]: run tutorial notebooks in CI
  against packaged samples (offline mode), render to GitHub Pages /
  jupyter-book; keep `nbstripout` enforcement so outputs never bloat the
  repository.
- **Reproducible pipeline configuration** [R]: `workflow/*/config.yaml`
  contains machine-specific absolute paths (`/srv/...`, `/Users/yun/...`);
  parametrize via environment variables or a site-config layer so
  collaborators can run the pipeline outside the SNU server.
- **Onboarding path** [P §3.3]: keep the wiki/tutorial flow (database
  registration → load → plot → analysis) synchronized with releases;
  versioned PyPI releases with changelogs.
- **Archival policy** [P §3.2, §4.1]: define which pipeline products are
  authoritative (HSDS) versus derived summaries (the xlsx histories in
  `workflow/automatic_pipeline_3_data_summary/`), with provenance
  metadata (code versions, config hashes) recorded in each IDS
  `code` section.

## 5. Platform and infrastructure

- **HSDS operations** [P §3.2]: authentication hardening, per-user
  folders/ACLs, monitoring, and the documented real-time backup chain
  (Google Drive + NAS); publish a restore procedure.
- **Continuous integration** [R]: keep `pytest test/` green (import
  smoke tests, contract tests, adapter tests), add lint and packaging
  checks, and run the offline pipeline on a sample shot as an
  integration test.
- **Dataset growth** [P §5]: continue routine accumulation (~7000 shots
  over three years so far) and periodically regenerate the
  operational-space and stability databases as reconstruction quality
  improves.
