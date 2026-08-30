# VAFT Introductory Tutorial Course

This directory is the technical foundation for the six-session introductory
VAFT course proposed in [issue #185](https://github.com/VEST-Tokamak/vaft/issues/185).
It defines the course contract and contains structural notebook and slide
scaffolds. Detailed teaching material will be completed one session at a time.

## Audience and prerequisites

The course is for students and researchers who are new to VAFT and may also be
new to fusion-plasma data analysis. Learners should be comfortable reading
basic Python and running Jupyter notebooks. The course itself introduces the
VAFT, OMAS/IMAS, VEST-database, and tokamak-physics concepts needed by each
exercise; it does not assume prior experience with VAFT or external fusion
codes.

Use a source checkout with the development dependencies installed:

```bash
python -m pip install -e ".[dev]"
```

The optional lab path also requires configured VEST HSDS credentials and the
relevant external-code roots, such as `CHEASEHOME`, `EFITHOME`, and `GPECHOME`.

## Course progression

| Session | Topic | Workflow role | Status |
| --- | --- | --- | --- |
| 01 | Getting Started with VAFT | diagnostic data and public plotting APIs | complete |
| 02 | Operation Scenario and Vacuum Fields | discharge operation and vacuum-field interpretation | scaffold |
| 03 | Equilibrium and Kinetic Profiles | reconstruction, profiles, and forward equilibrium | scaffold |
| 04 | Fluctuations and Transient Events | spectral analysis and event interpretation | scaffold |
| 05 | MHD Linear Stability and 3D Perturbed Equilibrium | equilibrium-to-stability/response modelling | scaffold |
| 06 | Operational Space and Statistics | cross-shot filtering, limits, and statistical analysis | scaffold |

The intended learning path is:

```text
diagnostic data -> discharge operation -> equilibrium/profiles
-> fluctuations/transients -> MHD stability/3D response
-> database-scale operational-space analysis
```

## Execution contract

Tutorial development uses two explicit modes:

- **Offline mode** is the default. It uses existing VAFT package/repository
  samples, geometry, equilibrium files, or inputs generated in the notebook.
  It must not require credentials, network access, or external executables.
- **Lab mode** enables read-only public HSDS access and installed external
  fusion codes. A notebook must preflight each optional capability and show a
  concise learner-facing skip message when it is unavailable.

Future executable notebooks will use `VAFT_TUTORIAL_MODE=offline` (the default)
or `VAFT_TUTORIAL_MODE=lab`. Generated files belong under the directory named
by `VAFT_TUTORIAL_OUTPUT_DIR`; when unset, use `tutorial/outputs/<session>/`.
That directory is ignored by Git. Lab mode must remain read-only with respect
to shared databases.

An offline run must execute every cell without failing. For sessions whose full
scientific result requires CHEASE, DCON, RDCON, GPEC, or live campaign data,
offline mode covers data/input preparation and the interpretation contract;
the unavailable solver or database step is skipped explicitly. Tutorial code
must call VAFT public APIs and must not reproduce internal implementation logic.

## Notebook authoring contract

Each numbered notebook uses the following top-level sections in this order:

1. Session Overview
2. Physical Context
3. Load / Prepare Data
4. Guided Analysis
5. Interpretation Checkpoints
6. Integrated Analysis
7. Independent Exercise
8. Takeaways and Next Steps

Each notebook stores `metadata.vaft_tutorial` with its session number,
`status`, and supported modes. Change `status` from `scaffold` to `complete`
only after the session has its detailed outline, teaching text, executable
guided analysis, independent exercise, and verification evidence.

Committed notebooks are source artifacts, not result archives. Every code cell
must have `execution_count: null` and `outputs: []`. Run exploratory or teaching
copies in `tutorial/outputs/`, then clear the committed source notebook before
review. Do not add precomputed tables, model results, or tutorial-only datasets.

Existing notebooks under `notebooks/` remain specialized references. Reuse
their public VAFT workflows where mature, but keep detailed theory and narrow
research procedures there rather than copying them into this introductory
course.

## Slide and figure contract

Each session has one standalone 16:9 Beamer source and one PDF with the same
stem. The preamble is deliberately small and duplicated so any deck can be
compiled independently. Every deck searches `figures/common/` and its own
numbered figure directory.

Use vector PDF figures when practical; PNG and JPEG are acceptable for raster
data such as camera images. Add only source/teaching figures, include attribution
and license information in `figures/README.md`, and never use the figure tree to
store generated notebook results.

Build all decks from the repository root with:

```bash
make -C tutorial slides
```

LaTeX intermediates are written to `tutorial/.build/`; only the six requested
PDFs are copied into the tutorial directory and committed.

Whenever you change a deck source or a figure it pulls in, rebuild that deck and
commit the regenerated PDF in the same change. CI enforces this: it checks that
every changed deck input ships a rebuilt PDF, then compiles all six decks from
scratch and confirms the rebuild reproduces the committed page structure.

The build pins `SOURCE_DATE_EPOCH` and `FORCE_SOURCE_DATE`, so rebuilding on one
machine reproduces byte-identical PDFs. That does not hold across TeX Live
releases, because pdfTeX records its own version in every file it writes. CI
therefore compares page structure rather than bytes, and the paired-rebuild
check is what keeps a committed PDF from drifting away from its source.

## Validation and contribution sequence

Run the repository contract before committing tutorial changes:

```bash
python test/verify_tutorial.py
make -B -C tutorial slides
python test/verify_tutorial.py
```

To reproduce the CI freshness checks locally, compare your branch against its
base and against a scratch copy of the committed decks:

```bash
python test/verify_tutorial_freshness.py pairing --base origin/develop --head HEAD
```

Develop sessions in numerical order. For each session:

1. Write and review its detailed learning objectives and outline.
2. Add the minimum physical context required by the guided work.
3. Implement the offline path, then gated lab extensions.
4. Add interpretation checkpoints and the independent exercise.
5. Execute from a clean environment and inspect all generated results.
6. clear the committed notebook, rebuild its PDF, run validation, and update
   the status table above.

Session 01 is the first completed content milestone. It uses packaged shot 39915
to inspect IDS roots and make magnetic, PF-coil, and UV-spectrometer plots
through VAFT public APIs. Its lab branch demonstrates a read-only HSDS magnetic
data selection when credentials are available. The existing
[plotting sample notebook](../notebooks/plotting_sample_using_vaft_plot_module.ipynb)
remains a specialized reference with broader research-oriented examples.

Session 02 is the next content milestone. The presence of all six scaffold
artifacts does not mean the remaining course content is complete.
