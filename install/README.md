# VAFT installation and environment verification

This directory is the single entry point for setting up VAFT on a new machine,
verifying that the setup works, and keeping an existing checkout current during
a course.

It covers the **VAFT / Python / HSDS / JupyterLab environment only**. Building
the external Fortran codes (CHEASE, DCON, GPEC) is a separate topic tracked in
[issue #226](https://github.com/VEST-Tokamak/vaft/issues/226).

Budget about 15–20 minutes from a nearly clean machine.

## Which script do I run?

| Platform | Command (from the repository root) |
| --- | --- |
| Linux (Ubuntu is the validated reference) | `bash install/linux.sh` |
| macOS (Apple silicon or Intel) | `bash install/macos.sh` |
| Windows, native | `powershell -ExecutionPolicy Bypass -File install\windows_native.ps1` |
| Windows, inside WSL2 | `bash install/windows_wsl.sh` |

**Native Windows is a first-class VAFT path. WSL2 is never required.** Choose
the WSL2 path only if you prefer a Linux shell or intend to build the external
Fortran codes later.

Every script accepts `--check-only` (`-CheckOnly` in PowerShell), which runs the
environment checker and changes nothing.

## Prerequisites

A script inside VAFT cannot bootstrap a machine that has no way to obtain VAFT
in the first place, so two things must be installed by hand first:

1. **Git** — <https://git-scm.com/downloads>
   (on macOS, `xcode-select --install` also provides it).
2. **Miniconda** — <https://www.anaconda.com/docs/getting-started/miniconda/install>

On Windows, either use the "Anaconda PowerShell Prompt" or run
`conda init powershell` once and reopen PowerShell, so that `conda` is on `PATH`.

For the WSL2 path only, install WSL2 and a distribution first
(`wsl --install -d Ubuntu` from an administrator PowerShell), then install Git
and **Linux** Miniconda *inside* the distribution.

The bootstrap scripts deliberately do **not** install Git, Conda, or WSL2 for
you. Installing system-wide tooling is your decision, not the repository's.

## First-time setup

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
bash install/linux.sh          # or macos.sh / windows_wsl.sh / windows_native.ps1
hsconfigure                    # only if you need the remote VEST database
conda activate vaft && jupyter lab
```

The bootstrap runs the environment check for you at the end. Run it again at any
time with the command in [Environment check](#environment-check).

```text
install Git + Miniconda
        ↓
git clone VAFT
        ↓
run one platform setup command
        ↓
configure HSDS credentials interactively
        ↓
run environment check
        ↓
launch JupyterLab
```

## What the bootstrap changes on your machine

The scripts are deliberately transparent. They change exactly three things:

1. **The `vaft` Conda environment** — created from [`environment.yml`](../environment.yml)
   if absent, otherwise updated in place with `conda env update --prune`.
   No other Conda environment is touched.
2. **An editable VAFT installation inside that environment** —
   `python -m pip install -e .` from your checkout, so `import vaft` uses the
   source tree you cloned. The script then verifies that `vaft.__file__` really
   does resolve inside your checkout, rather than to an unrelated installed copy.
3. **A user-level Jupyter kernelspec** named `vaft`, displayed as
   **Python (vaft)**. It is registered with a fixed `--name vaft`, which
   overwrites any previous spec of the same name, so rerunning the script can
   never accumulate duplicate kernels.

Each script then finishes by running the environment checker below and adopting
its exit status, so a bootstrap that reports success has actually been verified.
Verification lives in that one script rather than being reimplemented per
platform.

They do **not** modify your repository checkout, your shell profile, your Conda
configuration, or any system package manager.

### Why editable mode?

The course uses a VAFT checkout that keeps changing over the semester. An
editable installation means `git pull` is normally enough to pick up new code —
you do not reinstall after every change. **This does not imply that you are
expected to develop VAFT itself.**

### Where dependencies are declared

[`pyproject.toml`](../pyproject.toml) is the single source of truth for every
Python dependency. [`environment.yml`](../environment.yml) declares only
conda-level concerns — the interpreter version, `pip`, `git`, and JupyterLab —
and everything else arrives through `python -m pip install -e .`.

A package must never appear in both files. `test/test_install_bootstrap.py`
enforces that rule, and also checks that the Python pinned in `environment.yml`
satisfies `requires-python` in `pyproject.toml`, so the two cannot drift apart
silently.

## HSDS configuration

The remote VEST database is reached through HSDS. Configure it with the
interactive tool that ships with `h5pyd`:

```bash
conda activate vaft
hsconfigure
```

| Field | Value |
| --- | --- |
| Server endpoint | `http://147.46.36.244:5101` |
| Username | contact [peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr) |
| Password | contact [peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr) |

`hsconfigure` writes `~/.hscfg` in your home directory. That file holds your
credentials and belongs **only** there.

The VAFT bootstrap scripts never ask for, read, store, print, or transmit your
credentials. They only report whether `~/.hscfg` exists and which keys it sets —
never a value. There is intentionally no
`setup_vaft --username ... --password ...`: credentials passed as command-line
flags are retained in shell history and visible in process listings.

Never commit a real `.hscfg`. The repository ignores it, and
[`vaft/.hscfg.example`](../vaft/.hscfg.example) is the placeholder template.

Tutorial 01 and the rest of the offline course material run without any
credentials at all, from data packaged inside the repository.

## Environment check

```bash
conda run -n vaft python install/check_vaft_environment.py
```

```text
VAFT environment check
----------------------
[PASS] supported Python
[PASS] expected Conda environment
[PASS] VAFT import
[PASS] VAFT resolves to this checkout
[PASS] h5pyd / HSDS client
[PASS] HSDS command-line tools
[PASS] JupyterLab
[PASS] ipykernel
[PASS] Python (vaft) kernel
[PASS] Git
[WARN] HSDS configuration
[SKIP] HSDS connection
```

The default run is entirely offline: it needs no credentials, contacts no
server, and reads no credential value. Add `--include-network` to additionally
probe the HSDS endpoint using the credentials already in `~/.hscfg`. Add
`--json` for machine-readable output. The exit status is `0` only when nothing
failed.

`[WARN]` marks an optional capability. Missing HSDS credentials are a warning
rather than a failure, because the whole offline course -- Tutorial 01
included -- runs from data packaged in the repository. They become a failure
only under `--include-network`, where they are genuinely required.

Every failure names the corrective action:

```text
[WARN] HSDS configuration
       /home/student/.hscfg does not exist; needed only for remote database access
       -> Run `hsconfigure`, then rerun this check.
```

```text
[FAIL] expected Conda environment
       expected the `vaft` environment, but the active interpreter is
       /opt/miniconda3 (CONDA_DEFAULT_ENV=base)
       -> Run `conda activate vaft`, then rerun this check.
```

## Updating VAFT

VAFT changes during the semester. Keeping your checkout current is routine
maintenance, not a Git lesson.

```bash
git status
git pull --ff-only
conda run -n vaft python install/check_vaft_environment.py
```

`git pull --ff-only` refuses to create a merge commit; it either fast-forwards
cleanly or stops and tells you so. That is deliberate.

**Rerun the editable installation when dependency metadata changed** — that is,
whenever the pull touched `pyproject.toml` or `environment.yml`, or whenever the
checker reports a failure that mentions a missing or outdated package:

```bash
conda run -n vaft python -m pip install -e .
conda run -n vaft python install/check_vaft_environment.py
```

If the Python dependency set changed substantially, rerunning the whole
bootstrap script is always safe — it is idempotent.

## When an update is blocked by your own changes

If you edited notebooks or other files, `git pull --ff-only` may refuse to
proceed. Set your work aside, update, then bring it back:

```bash
git status
git stash push -m "before VAFT update"
git pull --ff-only
git stash pop
```

> **`git stash pop` can itself produce conflicts.** If you and the upstream
> repository changed the same part of the same file, Git stops and leaves
> conflict markers in your working tree. This is normal and your work is not
> lost — it is still in the stash until the pop succeeds.

**If you hit conflicts, stop.** Do not run `git reset --hard`, `git checkout --`,
or `git clean -fd`: those commands permanently destroy the work you were trying
to protect. Nothing in this repository will ever run them for you — the
bootstrap and checker never stash, reset, clean, overwrite, or discard anything
in your checkout.

Ask your instructor, or hand the situation to an AI coding agent with a prompt
like this one:

```text
I updated the VAFT repository and now have Git conflicts.
Please inspect `git status` and the conflicted files, preserve my local tutorial/notebook work,
integrate the upstream changes safely, and do not discard or overwrite my work.
```

## Troubleshooting

**`conda: command not found`** — Conda is installed but not on `PATH`. Reopen
your shell. On Windows, use the "Anaconda PowerShell Prompt" or run
`conda init powershell` once and reopen PowerShell.

**`running scripts is disabled on this system` (Windows)** — invoke the script
as `powershell -ExecutionPolicy Bypass -File install\windows_native.ps1` rather
than changing your machine-wide execution policy.

**`[FAIL] VAFT resolves to this checkout`** — an unrelated `vaft` (typically an
old `pip install vaft` from PyPI) is shadowing your clone. Fix it with:

```bash
conda run -n vaft python -m pip uninstall -y vaft
conda run -n vaft python -m pip install -e .
```

**`[FAIL] Python (vaft) kernel`, "expected exactly one kernelspec"** — remove
the duplicates and let the bootstrap re-register a single kernel:

```bash
jupyter kernelspec uninstall vaft
bash install/linux.sh
```

**JupyterLab shows no "Python (vaft)" kernel** — you started Jupyter from a
different environment. Run `conda activate vaft` first, or rerun the bootstrap.

**`hsconfigure: command not found`** — the `h5pyd` command-line tools live
inside the environment. Run `conda activate vaft` first.

**A network check fails but everything else passes** — that is expected off
campus or without credentials. The whole offline course works anyway; the HSDS
connection is only needed for live database access.

**Something else** — rerun the bootstrap script (it is safe to repeat), then
paste the full output of `conda run -n vaft python install/check_vaft_environment.py`
into your question.

## Verification status

| Path | How it is verified |
| --- | --- |
| Linux | Automated in CI (`.github/workflows/bootstrap-ci.yml`), run twice for idempotency |
| macOS | Automated in CI, run twice for idempotency |
| Windows native | Automated in CI, run twice for idempotency |
| Windows WSL2 | Syntax and static checks in CI; the full run is verified **manually**, because GitHub-hosted runners cannot start WSL2 |
