# VAFT Introductory Tutorial Course

A six-session introductory course that teaches VAFT through the scientific
lifecycle it exists to support:

```text
VEST machine and experiment -> validated, analysis-ready data
  -> visualization and interpretation -> physics analysis and modeling
  -> reproducible scientific result
```

The sessions are executable research examples, not an API catalogue. Each one
answers a question a VEST researcher actually has, and the infrastructure is
introduced as it becomes needed rather than up front.

This directory holds the course contract, the session notebooks, and their slide
decks. Detailed teaching material is completed one session at a time; the
structure was proposed in
[issue #185](https://github.com/VEST-Tokamak/vaft/issues/185).

## Audience and prerequisites

The course is for students and researchers who are new to VAFT and may also be
new to fusion-plasma data analysis. Learners should be comfortable reading
basic Python and running Jupyter notebooks. The course itself introduces the
VAFT, OMAS/IMAS, VEST-database, and tokamak-physics concepts needed by each
exercise; it does not assume prior experience with VAFT or external fusion
codes.

Use a source checkout with the development dependencies installed. If you are
setting up a machine for the first time, follow [`install/README.md`](../install/README.md)
and verify the result before the session starts:

```bash
bash install/linux.sh          # or macos.sh / windows_wsl.sh / windows_native.ps1
python install/check_vaft_environment.py
python -m pip install -e ".[dev]"
```

Environment setup is not a learning objective of this course. Budget 15-20
minutes for it beforehand.

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

## Session 01 is different on purpose

Session 01 is a beginner's walkthrough of the everyday VAFT workflow:

```text
load an ODS -> inspect its IDS roots -> look at the geometry
            -> plot diagnostics -> look at the equilibrium
```

It therefore does **not** use the eight-heading analysis structure below, which
is designed for the analysis sessions. Its headings are pinned separately in
`test/verify_tutorial.py`, and its own contract is different in three ways:

- **Two imports only**, `vaft` and `matplotlib.pyplot`, explained in prose
  rather than presented as boilerplate.
- **One pattern, repeated.** Every diagnostic plot is
  `vaft.omas.plot_<subject>_<view>(ods)` followed by `plt.show()`. No `savefig`,
  no dynamic `getattr` dispatch, and no notebook-local helper functions --
  tutorial code demonstrates VAFT rather than reimplementing it.
- **External links live in an appendix.** Session 01 ends with an *Additional
  Resources* section collecting IMAS/OMAS tutorials, per-language API docs and
  Data Dictionary references. They are deliberately kept out of the
  introduction, where they would weigh down a beginner's first pass, and a test
  enforces that separation. The appendix also records which Data Dictionary
  version VAFT reads through OMAS, checked against the package rather than
  restated in prose.
- **The plotting concepts are taught, not assumed.** Session 01 explains the
  `{subject}_{view}[_{quantity}]` naming grammar (issue #251) and the scientific
  display policy (issue #256) -- why an axis reads `kA` when the ODS stores
  amperes, and what `yunit=` does. One cell exists purely to demonstrate those
  concepts and is exempt from the one-pattern rule.
- **No tutorial machinery.** No exercise framework, no mode switching, no output
  directories, no repository discovery. Exercises are ordinary comment blocks a
  student edits, and nothing the notebook does depends on hidden validation.

`test/test_tutorial_session_01.py` enforces all of this, including that no cell
echoes a data object: one `repr(ods)` is about half a megabyte of array text.

If Session 01 needs a view that VAFT does not provide, add the view to VAFT as a
public `vaft.omas.plot_*` API and call it from the notebook. Do not compensate
with plotting code in the notebook.

## The solution repository

Completed and executed copies of the notebooks live in the private
`vaft-tutorial-solution` repository. They are a review and teaching oracle, never
a runtime dependency: the public test suite passes without any access to them,
and nothing in public CI may reference them.

## Presentation sources: two kinds, during the pilot

Sessions 02-06 are hand-written Beamer `.tex` decks with committed PDFs, as
described below. **Session 01 is different**: its slides are a Quarto `.qmd`
source in [`presentations/`](presentations/README.md), rendered to Reveal.js
HTML and Beamer PDF, with nothing committed.

That is the pilot slice of [issue #322](https://github.com/VEST-Tokamak/vaft/issues/322),
which proposes QMD as the canonical presentation source repository-wide. It is
deliberately one deck: the pilot exists so the two forms can be compared before
the convention spreads. Do not migrate the remaining decks until that review has
happened, and do not author a deck in both formats -- one source per deck is the
whole point.

The rest of this section governs the five Beamer decks.

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

LaTeX intermediates are written to `tutorial/.build/`; only the five requested
PDFs are copied into the tutorial directory and committed.

Whenever you change a deck source or a figure it pulls in, rebuild that deck and
commit the regenerated PDF in the same change. CI enforces this: it checks that
every changed deck input ships a rebuilt PDF, then compiles all five Beamer
decks from scratch and confirms the rebuild reproduces the committed page structure.

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

Session 01 is the first completed content milestone. It stays on the packaged
shot 39915 throughout: IDS roots, the composed machine geometry, the magnetic,
PF-coil, TF, spectrometer and barometry diagnostics, and the equilibrium flux
map and profiles -- all through public `vaft.omas.plot_*` APIs. It also teaches
how those plots are named and how their axes are scaled, so a reader can find a
plot they have not been shown. It needs no credentials and no repository-only
data.

When the plotting API changes, this session's explanations are part of the
change: the canonical reference for plotting policy is
[the plotting sample notebook](../notebooks/plotting_sample_using_vaft_plot_module.ipynb),
and Session 01 must not contradict it. The existing
[plotting sample notebook](../notebooks/plotting_sample_using_vaft_plot_module.ipynb)
remains a specialized reference with broader research-oriented examples.

Session 02 is the next content milestone. The presence of all six scaffold
artifacts does not mean the remaining course content is complete.
