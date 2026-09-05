# Contributing to VAFT

## Branches

Feature and fix work targets `develop`. `main` carries releases.

## What CI asks on each branch

The two branches answer different questions, and they are gated accordingly.

`develop` asks:

> Is this change safe to integrate with ongoing development?

`main` asks:

> Is this integrated state scientifically validated and supported across
> VAFT's declared platforms?

| | `develop` | `main` |
| --- | --- | --- |
| Required checks | `package`, `core-test` | `package`, `test`, `tutorial` |
| Test selection | `pytest -m "core and not perf"` on Linux | the whole suite on Linux **and** Windows, `slow` tests included |
| Answers | development confidence | release confidence and supported-platform qualification |
| Measured wall clock | ~4 min | ~35 min |

Full cross-platform validation is essential for VAFT — Windows is a first-class
supported platform, and the Windows leg has caught defects that were invisible
elsewhere. What changed in #515 is *when* the full platform claim is proven, not
whether it is. It is release qualification, not a toll on every intermediate
feature PR, which is what a ~34-minute gate on every merge into `develop` had
become. Measured on the run that introduced the split: `test (Linux)` 27.8 min
and `test (Windows)` 34.1 min, against `core-test` at 3m46s and `package` at
44s.

The full matrix still runs on the **push** to `develop`, after a PR lands. That
costs no merge latency and is what keeps the trade honest: a platform regression
surfaces within one merge, on the branch that caused it, rather than waiting for
`develop -> main` qualification. It is deliberately not a required check there.

`slow` tests — notebook execution and the tutorial session — run only on the
`main` gate. Nothing is skipped anywhere; those tests move gates, they do not
stop running.

## Branch protection is configured as code

Both branches' protection lives in
[`.github/rulesets/develop.json`](.github/rulesets/develop.json) and
[`.github/rulesets/main.json`](.github/rulesets/main.json), not only in the
repository's web settings. Edit the file, get it reviewed like any other change,
then re-apply it — that way the configuration is reviewable and recoverable
rather than being whatever someone last clicked.

What `develop.json` enforces:

| Rule | Effect |
| --- | --- |
| `required_status_checks` | The `Package CI` workflow's `core-test` and `package` jobs must both pass before anything merges into `develop`. |
| `deletion` | `develop` cannot be deleted. |
| `non_fast_forward` | `develop` cannot be force-pushed. |
| `bypass_actors` | The Admin repository role may bypass the above, for repository recovery only — see below. |

`main.json` is the same shape with no bypass actors and a third required check,
`tutorial`: a release must not ship decks that no longer build.

Three decisions worth knowing about, recorded here rather than buried in the API
payload:

- **`package` is required alongside the test gate on both branches.** It
  builds the distributions, checks the data policy and wheel size, and
  smoke-imports the installed wheel.
  It is cheap and already green, and it catches packaging breakage that the
  test suite does not.
- **Repository admins can bypass, for repository recovery only.**
  `bypass_actors` carries the Admin repository role (`RepositoryRole` id 5) in
  `always` mode, so a `develop` that CI itself cannot unblock — a broken
  workflow file, a wedged runner, a bad merge that makes the suite unrunnable —
  can still be repaired directly. **That is the entire intended use.** Normal
  work goes through a pull request with the required checks green, the same as
  for anyone else; the bypass existing does not make it an ordinary route
  around a red build. GitHub cannot enforce "emergencies only" — it is a
  capability, and this paragraph is the policy. Bypass use is visible in the
  repository's rule-insights log, so it is auditable after the fact.
- **"Require branches to be up to date before merging" is off**
  (`strict_required_status_checks_policy: false`). Turning it on forces a
  rebase and a full re-run on nearly every merge; at this repository's merge
  rate the cost outweighs the narrow class of semantic conflict it catches.

`tutorial` is deliberately **not** required on `develop`. It installs TeX Live
and rebuilds every slide deck, so making it blocking there would gate ordinary
feature work on a slow, toolchain-sensitive job. It is required on `main`,
where a deck that no longer builds is a release defect.

### Applying it

Needs repository admin rights:

```bash
gh api -X POST repos/VEST-Tokamak/vaft/rulesets --input .github/rulesets/develop.json
```

To update an existing ruleset in place, find its id and `PUT` instead:

```bash
gh api repos/VEST-Tokamak/vaft/rulesets --jq '.[] | "\(.id) \(.name) \(.enforcement)"'
```

```bash
gh api -X PUT repos/VEST-Tokamak/vaft/rulesets/<id> --input .github/rulesets/develop.json
```

### Verifying it

```bash
gh api repos/VEST-Tokamak/vaft/rulesets --jq '.[] | "\(.id) \(.name) \(.target) \(.enforcement)"'
```

`develop` and `main` should both appear as `branch active`. Nothing verifies
that the applied rulesets still match these files — if they drift, re-apply the
file. To see the rules a branch actually resolves to:

```bash
gh api repos/VEST-Tokamak/vaft/rules/branches/develop
```

### The `main` ruleset

`main` is protected by ruleset id `2009677`, checked in as
[`.github/rulesets/main.json`](.github/rulesets/main.json). It is active,
targets `refs/heads/main`, and requires `package`, `test` and `tutorial`. It
carries **no** bypass actors: unlike `develop`, an admin cannot push a release
past a red build.

It predates the develop ruleset and was for a long time described here as
disabled and targeting `~ALL` refs. That has not been true since it was
retargeted; the file is now the record, and it is applied and verified exactly
like `develop.json` above, substituting its own id.

## Formula docstrings

Every public function in `vaft/formula` documents itself in one layout, and
`test/test_formula_docstrings.py` enforces it: a one-sentence summary, the
definition (math in `$..$`), numpydoc `Parameters` / `Returns` whose description
paragraph closes with a unit tag such as `[Wb/rad]` or `[-]`, and any of the
sections `Convention`, `Physical interpretation`, `Assumptions`, `Validity`,
`Limitations`, `Numerical notes`, `References` (`.. [1] Author, Journal vol
(year) page, Eq. (n).`).  Empirical fits open `Validity` with the sentence
`Empirical fit.`; convention-sensitive functions carry a `Convention` section;
a known defect goes under `Limitations` as `Tracked in #NNN` rather than into
the kernel.  The policy lists that decide which functions are definitional,
convention-sensitive or empirical live in the test file.

The docstring is the only source of truth: `vaft.formula.describe(name)`,
`vaft.formula.search(text)` and `vaft.formula.list_formulas(category=...)` parse
it on demand (the catalog is never imported by `import vaft.formula` or by a
physics submodule -- `test/test_formula_catalog.py` pins that), and the site's
reference pages under `/reference/formula/` are generated from the same text by
`python -m vaft.formula.catalog`, which `docs/build.py` runs for you.

## Processing docstrings

`vaft/process` uses the same parser (`vaft/_docstring.py`) and the same unit-tag
rule, and a different vocabulary, because a processing routine answers a
different question: how is this input turned into this output?  Its sections
are `Processing steps`, `Input semantics`, `Output semantics`, `Defaults`,
`Convention`, `Assumptions`, `Applicability`, `Limitations`, `Provenance`.
`Applicability` opens with `Machine-independent.` or `VEST-specific.`;
`Defaults` classifies each default that matters (legacy compatibility value,
validated-workflow default, numerical convenience, ...); `Provenance` takes
the same `.. [1] text` form as `References` but cites a MATLAB file, a
`vest.yaml` key or an issue as readily as a paper.
`test/test_process_docstrings.py` enforces it, and its `PENDING` list names the
submodules not yet converted (#418-#421): a module leaves that list only when
every function in it conforms, and the catalog's own `conforming` flag is what
the site reads to decide which categories get a reference page.

`vaft.process.describe(name)`, `vaft.process.search(text)` and
`vaft.process.list_processes(category=...)` read the docstrings on demand and
are never imported by `import vaft.process`; `/reference/process/` is generated
by `python -m vaft.process.catalog`, which `docs/build.py` runs for you.

## Documentation

The site published at <https://vest-tokamak.github.io/vaft/> is built from
`docs/` on the code branches: the root from `main`, `/develop/` from `develop`.
Both are composed and published as one commit on `gh-pages`, which holds
generated output only and must never be edited by hand.

```bash
cd docs && bundle install
python build.py                 # dry run: both tracks, composed and validated
npm run docs:serve              # local preview of this branch's pages
npm run test:docs               # build and validate the stable track
```

A push to `main` or `develop` rebuilds both tracks and republishes the whole
site through `.github/workflows/docs.yml`. A page belongs to the branch whose
library it documents, so a fix wanted on both tracks travels from `develop` to
`main` like any other change. `docs/README.md` has the details.

## Documentation surfaces

The section above covers how the site is built.
This one covers what belongs where. Each surface has one job, and adding
material to the wrong one is how the README became a manual, a history, and an
architecture document at once. Adding material to the wrong one is how
the README became a manual, a history, and an architecture document at once.

| Surface | Job | Not this |
| --- | --- | --- |
| `README.md` / `README.ko.md` | Landing page: what VAFT is, what you can do with it, how to start, where to go next | Reference material, extended history, API detail |
| `tutorial/` | Taught course, offline-first, one session at a time | Research procedures, narrow techniques |
| `notebooks/` + `notebooks/README.md` | Research workflows indexed by scientific question | Teaching scaffolding |
| `docs/` -> [the site](https://vest-tokamak.github.io/vaft/) | Long-form workflows, API reference, machine and research archive | Anything that must ship with the source |
| `THIRD_PARTY_NOTICES.md` | Licence reproduction — a distribution obligation, so it ships with the source | Prose about dependencies |

Three rules:

1. **Keep the two READMEs semantically synchronized.** `README.ko.md` is not
   optional and not a lagging translation. `test/test_readme_consistency.py`
   pins the identity narrative, the four infrastructure concepts and their
   order, and every relative link.
2. **Do not present planned capability as shipped.** Semantic knowledge graphs,
   machine-actionable provenance, digital-twin integration and autonomous
   research agents are long-term direction. Label them, or the test fails.
3. **Use the shared vocabulary** (issue #330): *scientific infrastructure* rather
   than only "Python library"; *VEST reference implementation* without implying
   VAFT is VEST-only; *validated / analysis-ready data*; *traceable and
   reproducible*; *machine-agnostic tokamak research*. VAFT **interoperates
   with** community physics codes — it does not own or embed them.

## Notebooks

Notebook outputs are normalized by the repository's pre-commit hook. Install it
with `pre-commit install`; the hook retains only static text and image results.
To normalize notebooks manually, run:

```bash
python notebooks/_clean_outputs.py notebooks/*.ipynb
```

## Running the tests

```bash
python -m pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

Tests that need external resources — a CHEASE build, the VEST HSDS server, a
mounted diagnostic share — skip themselves with a reason. Everything else must
pass in an environment built from `pyproject.toml` alone.

### Markers

Four markers are registered in `pyproject.toml`, and they are how CI slices the
suite.

| Marker | Meaning | Where it runs |
| --- | --- | --- |
| `core` | the development-confidence selection | the required `develop` gate |
| `slow` | release-confidence checks — notebook and tutorial execution | the `main` gate only |
| `perf` | performance budgets, calibrated against the Linux runner class | the full Linux suite only — deselected on Windows and in the `develop` gate |
| `integration` | opt-in read-only HSDS or IMAS access | wherever the resource is reachable; self-skips otherwise |

To run what gates a PR into `develop`, before pushing:

```bash
python -m pytest -q -m "core and not perf"
```

`perf` is deselected there for the same reason the Windows leg drops it: a
wall-clock ratio is the last thing to trust in a job built to be quick.

`core` is not written into test modules by hand. It is one declared list in
[`test/core_selection.py`](test/core_selection.py), applied during collection by
`test/conftest.py`, so the whole develop gate is reviewable in a single diff.
`test/test_core_selection.py` holds the list to what it declares — every entry
must exist, no `slow` module may be in it, the workflow must still run the
expression above, and the marker must be applied early enough that `-m core`
selects anything at all.

Adding a test to the core selection means adding a line to that list, with a
reason. The bar is a contract that fails loudly and cheaply: import and
namespace shape, public API surface, layer boundaries, registries and
taxonomies, serialization round-trips, packaging policy. Scientific regression
and validation coverage belongs in the full suite, which the `main` gate and the
post-merge `develop` run both execute in full.
