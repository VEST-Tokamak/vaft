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
| `required_status_checks` | `test` and `package` (both jobs of the `Package CI` workflow) must pass before anything merges into `develop`. |
| `deletion` | `develop` cannot be deleted. |
| `non_fast_forward` | `develop` cannot be force-pushed. |

Two decisions worth knowing about, recorded here rather than buried in the API
payload:

- **`package` is required alongside `test`.** It builds the distributions,
  checks the data policy and wheel size, and smoke-imports the installed wheel.
  It is cheap and already green, and it catches packaging breakage that the
  test suite does not.
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

`develop` should appear as `branch active`. To see the rules it resolves to for
the branch itself:

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
