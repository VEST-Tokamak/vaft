# Contributing to VAFT

## Branches

Feature and fix work targets `develop`. `main` carries releases.

## Branch protection is configured as code

Both protected branches carry their configuration in the repository, not only in
the web settings: [`.github/rulesets/develop.json`](.github/rulesets/develop.json)
and [`.github/rulesets/main.json`](.github/rulesets/main.json). Edit the file,
get it reviewed like any other change, then re-apply it — that way the
configuration is reviewable and recoverable rather than being whatever someone
last clicked.

What `develop` enforces:

| Rule | Effect |
| --- | --- |
| `required_status_checks` | The `Package CI` workflow's `test` and `package` jobs must both pass before anything merges into `develop`. |
| `deletion` | `develop` cannot be deleted. |
| `non_fast_forward` | `develop` cannot be force-pushed. |
| `bypass_actors` | The Admin repository role may bypass the above, for repository recovery only — see below. |

What `main` enforces:

| Rule | Effect |
| --- | --- |
| `pull_request` | `main` cannot be pushed to directly. Every change arrives as a pull request. Zero approving reviews are required — see below. |
| `required_status_checks` | The `Package CI` workflow's `test`, `package` **and `tutorial`** jobs must all pass before anything merges into `main`. |
| `deletion` | `main` cannot be deleted. |
| `non_fast_forward` | `main` cannot be force-pushed. |
| `bypass_actors` | The Admin repository role may bypass the above, for repository recovery only — see below. |

Five decisions worth knowing about, recorded here rather than buried in the API
payload:

- **`package` is required alongside `test`.** It builds the distributions,
  checks the data policy and wheel size, and smoke-imports the installed wheel.
  It is cheap and already green, and it catches packaging breakage that the
  test suite does not.
- **`main` requires a pull request but zero approving reviews.** The rule exists
  to close direct pushes: required status checks do not apply to a push, so
  without it the checks below gate only the route nobody was forced to take.
  Requiring an *approval* on top would be a different thing — on a repository
  where releases are routinely solo-authored, GitHub will not let the author
  approve their own pull request, so every release would have to go through the
  admin bypass, and a bypass used on every merge is not a recovery valve. The
  gate that binds here is a green build, not a rubber stamp. (Issue #269 asked
  for one approving review; this is a deliberate divergence, for that reason.)
- **`tutorial` is required on `main` and deliberately not on `develop`.** It
  installs TeX Live and rebuilds every slide deck — slow and toolchain-sensitive,
  too expensive to sit in front of everyday merges. `main` carries releases and
  ships the built decks, so a deck that no longer rebuilds from its source is a
  release defect there, and the cost is paid once per release rather than once
  per merge.
- **Repository admins can bypass, for repository recovery only.**
  `bypass_actors` carries the Admin repository role (`RepositoryRole` id 5) in
  `always` mode on both branches, so a branch that CI itself cannot unblock — a
  broken workflow file, a wedged runner, a bad merge that makes the suite
  unrunnable — can still be repaired directly. **That is the entire intended
  use.** Normal work goes through a pull request with the required checks green,
  the same as for anyone else; the bypass existing does not make it an ordinary
  route around a red build. GitHub cannot enforce "emergencies only" — it is a
  capability, and this paragraph is the policy. Bypass use is visible in the
  repository's rule-insights log, so it is auditable after the fact.
- **"Require branches to be up to date before merging" is off**
  (`strict_required_status_checks_policy: false`) on both branches. Turning it
  on forces a rebase and a full re-run on nearly every merge; at this
  repository's merge rate the cost outweighs the narrow class of semantic
  conflict it catches.

`Contract tests`, from `Bootstrap CI`, is **not** required on either branch,
though the workflow is built so that it could be: it runs on every pull request
and reports "not applicable" as a success when the change touches nothing the
bootstrap depends on. A required check has to report on every pull request or
the pull request can never merge — see the comment at the top of
[`.github/workflows/bootstrap-ci.yml`](.github/workflows/bootstrap-ci.yml).

### Applying it

Needs repository admin rights. To create a ruleset that does not exist yet:

```bash
gh api -X POST repos/VEST-Tokamak/vaft/rulesets --input .github/rulesets/develop.json
```

Both rulesets already exist, so the everyday operation is an in-place `PUT`.
Find the ids:

```bash
gh api repos/VEST-Tokamak/vaft/rulesets --jq '.[] | "\(.id) \(.name) \(.enforcement)"'
```

Then re-apply the file that describes each one:

```bash
gh api -X PUT repos/VEST-Tokamak/vaft/rulesets/21861910 --input .github/rulesets/develop.json
```

```bash
gh api -X PUT repos/VEST-Tokamak/vaft/rulesets/2009677 --input .github/rulesets/main.json
```

### Verifying it

```bash
gh api repos/VEST-Tokamak/vaft/rulesets --jq '.[] | "\(.id) \(.name) \(.target) \(.enforcement)"'
```

`develop` and `main` should each appear as `branch active`, and no other ruleset
should exist. Nothing verifies that an applied ruleset still matches its file —
if it drifts, re-apply the file. To see the rules a branch actually resolves to:

```bash
gh api repos/VEST-Tokamak/vaft/rules/branches/develop
```

```bash
gh api repos/VEST-Tokamak/vaft/rules/branches/main
```

The two resolve independently. `develop` should return exactly `deletion`,
`non_fast_forward` and `required_status_checks`; `main` should return those
three plus `pull_request`. A ruleset whose `conditions.ref_name.include` reads
`~ALL` rather than a single branch would apply to both — and to every working
branch besides — so a change that appears on the wrong branch is the first thing
to check for there.

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
