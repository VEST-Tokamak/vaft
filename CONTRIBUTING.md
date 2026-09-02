# Contributing to VAFT

## Branches

Feature and fix work targets `develop`. `main` carries releases.

## Branch protection is configured as code

`develop`'s protection lives in [`.github/rulesets/develop.json`](.github/rulesets/develop.json),
not only in the repository's web settings. Edit the file, get it reviewed like
any other change, then re-apply it — that way the configuration is reviewable
and recoverable rather than being whatever someone last clicked.

What it enforces:

| Rule | Effect |
| --- | --- |
| `required_status_checks` | The `Package CI` workflow's `test` and `package` jobs must both pass before anything merges into `develop`. |
| `deletion` | `develop` cannot be deleted. |
| `non_fast_forward` | `develop` cannot be force-pushed. |
| `bypass_actors` | The Admin repository role may bypass the above, for repository recovery only — see below. |

Three decisions worth knowing about, recorded here rather than buried in the API
payload:

- **`package` is required alongside `test`.** It builds the distributions,
  checks the data policy and wheel size, and smoke-imports the installed wheel.
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

`tutorial`, the third `Package CI` job, is deliberately **not** required. It
installs TeX Live and rebuilds every slide deck, so making it blocking would
gate merges on a slow, toolchain-sensitive job.

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

`develop` should appear as `branch active`. Nothing verifies that the applied
ruleset still matches this file — if it drifts, re-apply the file. To see the
rules it resolves to for the branch itself:

```bash
gh api repos/VEST-Tokamak/vaft/rules/branches/develop
```

### The pre-existing `main` ruleset

The repository already carries a ruleset named `main` (id `2009677`). It is
`enforcement: "disabled"`, and despite its name it targets `~ALL` refs and
would require two approving reviews everywhere if switched on. It is not
managed by this file. Decide explicitly whether to enable, retarget, or delete
it — leaving a disabled ruleset next to an active one is how a repository ends
up with protection nobody can account for.

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
reference pages are generated from the same text with
`python -m vaft.formula.catalog --output <gh-pages checkout>/_data/formula_catalog.yml`.

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
