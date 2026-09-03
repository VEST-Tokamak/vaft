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
| macOS (Apple silicon) | `bash install/macos.sh` |
| Windows, native | `powershell -ExecutionPolicy Bypass -File install\windows_native.ps1` |
| Windows, inside WSL2 | `bash install/windows_wsl.sh` |

**Native Windows is a first-class VAFT path. WSL2 is never required.** Choose
the WSL2 path only if you prefer a Linux shell or intend to build the external
Fortran codes later.

Every script accepts `--check-only` (`-CheckOnly` in PowerShell), which runs the
environment checker and changes nothing.

On native Windows, if an existing `vaft` environment uses a different Python
minor version than `environment.yml`, the bootstrap stops before Conda enters a
long in-place solve. Rebuild only that environment with:

```powershell
powershell -ExecutionPolicy Bypass -File install\windows_native.ps1 -Recreate
```

`-Recreate` removes and recreates the `vaft` Conda environment only. It does not
touch other environments or checkout files; record any extra packages you added
to `vaft` before using it. `-Recreate` cannot be combined with `-CheckOnly`.

To undo an installation, see [Uninstalling](#uninstalling).

### Apple silicon only on macOS

VAFT depends on `imas_core`, which publishes wheels for Apple silicon macOS,
Linux x86_64 and Windows — there is no Intel macOS wheel and no source
distribution. The installation cannot succeed on an Intel Mac. Use Apple
silicon, a Linux machine, or WSL2. The bootstrap detects this case and says so
rather than leaving you with a pip resolver error.

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
bash install/linux.sh          # or install/macos.sh, install/windows_wsl.sh
                               # on native Windows, see the table above
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
   if absent, otherwise updated in place. The update deliberately does not
   pass `--prune`, so anything you installed into the environment yourself is
   left alone. No other Conda environment is touched.
2. **An editable VAFT installation inside that environment** —
   `python -m pip install -e .` from your checkout, so `import vaft` uses the
   source tree you cloned. The environment check then confirms that `vaft`
   really does resolve inside your checkout rather than to an unrelated
   installed copy.
3. **A user-level Jupyter kernelspec** named `vaft`, displayed as
   **Python (vaft)**. It is registered with a fixed `--name vaft`. Jupyter
   keys kernelspecs by name, so a rerun replaces the spec rather than adding a
   second one.

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

The VAFT bootstrap scripts never ask for, store, print, or transmit your
credentials. The checker reads `~/.hscfg` only to report whether it exists and
which keys it sets — a value never enters the report. There is intentionally no
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

## Uninstalling

Removing VAFT is the exact inverse of the bootstrap, and it exists so the
installation can be tested honestly. On a machine that already has VAFT, a
rerun of the bootstrap only ever takes the *update* path; the only way to
exercise a real first installation again is to remove what the last one left
behind.

| Platform | Command (from the repository root) |
| --- | --- |
| Linux, macOS, Windows inside WSL2 | `bash install/uninstall.sh` |
| Windows, native | `powershell -ExecutionPolicy Bypass -File install\uninstall_windows_native.ps1` |

One POSIX entry point covers three platforms because removal is identical on
all of them; there is no per-platform difference to write down.

```bash
bash install/uninstall.sh --dry-run   # print the plan, change nothing
bash install/uninstall.sh             # list what will go, then confirm
bash install/uninstall.sh --yes       # skip the confirmation, for scripting
```

The PowerShell script takes `-DryRun`, `-Yes` and `-KeepBuildArtifacts`.

### What it removes

Exactly the artifacts listed in
[What the bootstrap changes](#what-the-bootstrap-changes-on-your-machine), in
the one order that works:

1. **The user-level `vaft` kernelspec** — first, because it is removed through
   `conda run -n vaft`, and once the environment is gone there is no
   interpreter left to run that command. If the environment is already absent
   or its Jupyter is broken, the script falls back to deleting the kernelspec
   directory so the spec can never be orphaned.
2. **The `vaft` Conda environment** — `conda env remove --name vaft`. The
   editable installation lives inside it and goes with it.
3. **The `vaft.egg-info/` directory** the editable install leaves in your
   checkout — the one leftover a reinstall would otherwise inherit. Pass
   `--keep-build-artifacts` (`-KeepBuildArtifacts`) to leave it.

`build/` and `dist/` are deliberately *not* touched. The bootstrap never
creates them; `python -m build` does, when a maintainer cuts a release. Deleting
someone's release artifacts is not this script's job.

If the `vaft` environment is active in your shell, the script stops before
removing anything and asks you to `conda deactivate` first. Conda refuses to
delete an environment you are standing in, and since the kernelspec has to go
first, a refusal part-way would leave you with a working environment and no
kernel.

### What it never removes

- **`~/.hscfg`.** The bootstrap never wrote it — `hsconfigure` did, when you
  ran it — and it holds your HSDS credentials. Uninstalling VAFT should not
  make you type them again.
- **Any Conda environment whose name is not exactly `vaft`.** The removal is
  pinned to `--name vaft`, with no prefix or pattern match, so an environment
  you named `vaft-experiment` is never in scope. This is enforced by a test,
  not left to convention.
- **Your repository checkout**, beyond that one directory. Like the bootstrap,
  the uninstaller runs no destructive Git command: the artifact is deleted by
  explicit path, never with `git clean`.
- **Conda or Git themselves.** The bootstrap refuses to install them, so the
  uninstaller has no business removing them.

### Testing that an installation is repeatable

Running the uninstaller twice is a no-op: with nothing installed every step
reports `SKIP` and the script exits 0. That makes the following loop the
supported way to re-verify a machine end to end:

```bash
bash install/macos.sh
bash install/uninstall.sh --yes
bash install/macos.sh
bash install/uninstall.sh --yes
```

The second installation is the one that matters. If it does not behave like the
first — same environment created from scratch, same kernel registered, same
checker verdict — then the uninstaller left state behind. CI runs exactly this
cycle on Linux, macOS and native Windows.

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

**`[FAIL] Python (vaft) kernel`, "no `vaft` kernel is registered"** — register
it by rerunning the bootstrap, which is safe to repeat:

```bash
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
| Linux | Automated in CI (`.github/workflows/bootstrap-ci.yml`): install → uninstall → install → uninstall |
| macOS | Automated in CI, same install/uninstall cycle |
| Windows native | Automated in CI, same install/uninstall cycle |
| Windows WSL2 | Syntax and static checks in CI; the full run is verified **manually**, because GitHub-hosted runners cannot start WSL2 |
