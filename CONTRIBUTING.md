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

Three decisions worth knowing about, recorded here rather than buried in the API
payload:

- **`package` is required alongside `test`.** It builds the distributions,
  checks the data policy and wheel size, and smoke-imports the installed wheel.
  It is cheap and already green, and it catches packaging breakage that the
  test suite does not.
- **No bypass actors.** `bypass_actors` is empty, so the required checks bind
  administrators too, and there is no direct-push hotfix path — including when
  CI itself is what is broken. Getting a fix in then means either fixing CI
  first or temporarily relaxing the ruleset in Settings → Rules. If that trade
  is wrong for this team, add a repository-admin bypass actor to
  `.github/rulesets/develop.json` and re-apply it.
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
