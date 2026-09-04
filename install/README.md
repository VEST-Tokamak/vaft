# VAFT installation and environment verification

This directory is the single entry point for setting up VAFT on a new machine,
verifying that the setup works, and keeping an existing checkout current during
a course.

The platform scripts below cover the **VAFT / Python / HSDS / JupyterLab
environment**. Building the external Fortran codes is optional and independent,
and has its own entry points: see
[External fusion codes](#external-fusion-codes-chease-and-dcongpec).

Budget about 15–20 minutes from a nearly clean machine.

## Which script do I run?

| Platform | Command (from the repository root) |
| --- | --- |
| Linux (Ubuntu is the validated reference) | `bash install/linux.sh` |
| macOS (Apple silicon) | `bash install/macos.sh` |
| Windows, native | `powershell -ExecutionPolicy Bypass -File install\windows_native.ps1` |
| Windows, inside WSL2 | `bash install/windows_wsl.sh` |

**Native Windows is a first-class VAFT path. WSL2 is never required.** CHEASE
and the DCON/GPEC suite build natively too; choose the WSL2 path only if you
prefer a Linux shell.

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

## External fusion codes (CHEASE and DCON/GPEC)

The bootstrap above gives you VAFT, Python and JupyterLab. The external Fortran
codes are optional and independent: you only need this section if you want VAFT
to *run* CHEASE or the DCON/GPEC suite rather than only prepare their inputs.

```text
Need VAFT only?            -> the platform script above; you are done
Need CHEASE?               -> obtain CHEASE, then install\install_chease_windows.ps1
Need DCON/GPEC?            -> obtain GPEC,   then install\install_gpec_windows.ps1
DCON/GPEC on Windows?      -> read the known limitation below first; use WSL2
```

**You obtain the source yourself.** The installers take the path to a checkout
you already have and never clone, fetch, pull, change a revision, or initialise
a submodule. Which revision was built is a fact you state and the installer
records, not one it infers — with more than one checkout on a machine, that is
the difference between reproducible provenance and a guess.

```powershell
git clone https://github.com/PrincetonUniversity/GPEC.git C:\git\GPEC
git -c core.symlinks=true clone https://gitlab.epfl.ch/spc/chease.git C:\git\CHEASE
```

CHEASE is cloned with `core.symlinks=true` on purpose. It commits several
sources as symbolic links, one of which is compiled into the plain `chease`
target, and without symlink support Git for Windows writes a short text file
naming the target instead. The build then fails with a Fortran syntax error
that says nothing about the cause. Creating symbolic links needs Developer Mode
or an elevated prompt; if you already have a checkout without them, the
installer's `-MaterializeSymlinks` copies each target over its placeholder
instead, which rewrites those tracked files in your CHEASE tree.

### Installing

```powershell
powershell -ExecutionPolicy Bypass -File install\install_chease_windows.ps1 C:\git\CHEASE
powershell -ExecutionPolicy Bypass -File install\install_gpec_windows.ps1 C:\git\GPEC
```

Both build with the MinGW-w64 gfortran toolchain from MSYS2, which produces
ordinary Windows executables — no WSL2, no emulation layer at run time.

Add `-InstallToolchain` to let the installer set MSYS2 up for you with `winget`
and `pacman`. Without it, a machine with no toolchain stops with the exact
commands to run yourself. That follows the same rule as the rest of this
directory: Git, Conda and now a Fortran compiler are things you decide to
install, not things a script installs behind you.

| Switch | Effect |
| --- | --- |
| `-InstallToolchain` | Install MSYS2 and the compiler packages. The only switch that changes anything outside the prefix. |
| `-Prefix <path>` | Install somewhere other than `%LOCALAPPDATA%\vaft\external\<code>`. |
| `-MaterializeSymlinks` | CHEASE only. Replace symbolic-link placeholders with copies of their targets. |
| `-NoEnvironmentWiring` | Do not set `CHEASEHOME` / `GPECHOME`; print the command instead. |
| `-CheckOnly` | Run the checker and change nothing. |
| `-Uninstall` | Remove the prefix and the environment variable. Your source tree is untouched. |

### What gets installed, and where

Everything lands in one self-contained prefix outside every checkout:

```text
%LOCALAPPDATA%\vaft\external\gpec\
    bin\   dcon.exe rdcon.exe stride.exe gpec.exe match.exe rmatch.exe
           + the runtime libraries those need
    logs\  the full build output
    vaft-external-install.json
```

The runtime libraries are **copied next to the executables** rather than reached
through `PATH`. Windows searches an executable's own directory first, so the
prefix works from a plain terminal, from `conda run`, and from a Jupyter kernel
started by a server in a different environment, with no `PATH` change anywhere.
Putting MSYS2's `bin` on `PATH` instead would place a second `libcrypto`,
`libssl` and `zlib` ahead of Anaconda's for every process in that session, which
breaks unrelated things in ways that are hard to trace back here.

`vaft-external-install.json` records the source path and revision, the toolchain,
the exact make command and every installed file. It is what `-Uninstall` reads,
and what the checker compares your checkout against.

### How VAFT finds them

The installer sets `CHEASEHOME` or `GPECHOME` as a **user** environment
variable. The registered "Python (vaft)" kernel starts the environment's
`python.exe` directly rather than through Conda activation, so an `activate.d`
script would never reach a notebook; a user variable reaches every newly started
process. Nothing machine-wide is changed.

Open a new terminal, or restart JupyterLab, before expecting it to take effect.

To set it yourself instead:

```powershell
[Environment]::SetEnvironmentVariable('GPECHOME', "$env:LOCALAPPDATA\vaft\external\gpec", 'User')
$env:GPECHOME = "$env:LOCALAPPDATA\vaft\external\gpec"
```

VAFT resolves `$GPECHOME/bin/dcon` and `$CHEASEHOME/bin/chease` — the documented
POSIX names — and finds the native `.exe` beside them automatically.

### Verifying

```powershell
conda run -n vaft python install/check_chease.py --source C:\git\CHEASE
conda run -n vaft python install/check_gpec.py   --source C:\git\GPEC
```

Each checker reports one line per layer, so a failure names what broke rather
than handing you compiler output: toolchain, source checkout, revision, build
record, executables, whether they load with a bare `PATH`, whether VAFT's own
resolution finds them, a real run, the products, and whether the numbers are
sound. The GPEC checker runs upstream's own Solov'ev regression case and
exercises the real DCON → GPEC handoff rather than starting two binaries
separately.

### CHEASE and the `nideal` default

`CHEASEConfig.nideal` defaults to `11`, which reproduces the VEST `jsk95`
workflow against the CHEASE build that group uses. Upstream CHEASE accepts 1
through 10 and rejects 11 outright:

```
WRONG VALUE FOR NIDEAL IT HAS TO BE 1,2,3,4,5,6,7,8,9 OR 10
 after cotrol, output_flag =         -798
```

So a CHEASE built from the public repository refuses the default configuration,
on every platform -- this is a code-version difference, not a Windows one. Pass
`CHEASEConfig(nideal=6)`, which is upstream's own default and the one documented
as writing the EQDSK that VAFT reads back, or use the CHEASE revision the VEST
workflow was written against.

`check_chease.py` detects exactly this: it runs with the VAFT default first, and
if CHEASE rejects it, retries with 6 and reports a WARN naming the difference
rather than a failure.

### Two suite tests start running once CHEASE is installed

`test/test_chease_adapter.py` gates three tests on `$CHEASEHOME` so they skip on
a machine without CHEASE. Installing it un-skips them, and two then fail against
a CHEASE built from the public repository:

- `test_run_chease_integration_when_available` — the same `nideal` mismatch as
  above: CHEASE refuses the default configuration and writes no EQDSK.
- `test_run_chease_gfile_and_equivalent_ods_input_agree` — `ZMAXIS` differs from
  the reference by about 1.5e-5 relative, against an `rtol` of 1e-7.

Both compare against the CHEASE build the VEST workflow was written for, so they
are measuring the difference between two CHEASE revisions rather than anything
about VAFT or about Windows. CI never sets `$CHEASEHOME`, so they stay skipped
there; you will only see them locally, and only after installing CHEASE.

If you need a green local suite before that is resolved, unset `CHEASEHOME` for
the run:

```powershell
$env:CHEASEHOME = $null; python -m pytest -q
```

### What a native Windows build does differently

Two differences from a Linux build are real, and both are reported rather than
papered over.

**The DCON/GPEC suite is serial.** LSODE and ZVODE mark a COMMON block
`!$OMP THREADPRIVATE`, and gfortran expresses that with an assembler directive
the PE object format has no equivalent for, so an OpenMP build cannot assemble
at all. The installer builds without OpenMP. Results are unaffected; long runs
take longer than the same case on Linux.

**The DCON/GPEC suite does not yet finish a run on Windows.** This is the one
reason that path is not called supported, and it is not VAFT's defect or GPEC's.

MSYS2's netCDF package *and* its HDF5 both link the AWS C++ S3 SDK. That SDK
registers an `atexit` handler which waits on a condition variable that is never
signalled, so a program linked against either writes every output correctly,
prints its normal-termination message, and then never exits -- and the process
cannot be killed, because the thread is blocked inside the kernel. Six lines of
Fortran reproduce it with no GPEC or VAFT code involved:

```fortran
program probe
  use netcdf
  integer :: ncid, ierr
  ierr = nf90_create("probe.nc", NF90_CLOBBER, ncid)
  ierr = nf90_close(ncid)
end program probe
```

Building netCDF without S3 is **not** sufficient, because HDF5 pulls the same
SDK in on its own. A complete fix needs an S3-free HDF5 as well as an S3-free
netCDF, or -- better -- for MSYS2 to stop enabling S3 in those packages.

Until then:

- **CHEASE is fully supported natively.** It links neither library.
- **For DCON/GPEC, use WSL2** (`install/windows_wsl.sh` plus the Linux build
  recipe in `workflow/automatic_pipeline_1_routine_data_processing/DEPLOYMENT.md`).
- If you build an S3-free HDF5 and netCDF yourself, point `-NetcdfHome` at them.

Everything else about the native GPEC build is verified and works: it compiles,
`check_gpec.py` finds all six executables through `$GPECHOME`, each starts with
`PATH` stripped to `System32`, and DCON solves upstream's Solov'ev regression
case to the correct energies. `check_gpec.py` reports the exit defect as its own
named layer rather than leaving you with a run that never returns.

### What the build leaves in your source tree

The upstream Makefiles write their objects and binaries inside the checkout you
pointed at. GPEC ignores its own `*.o`, `*.mod` and `bin/`, but not the
`<module>/<name>.exe` files a Windows build produces, so `git status` in your
GPEC tree will show a few untracked executables. CHEASE ignores `chease` but not
`chease.exe`, for the same reason. Neither installer changes a tracked file
unless you pass `-MaterializeSymlinks`.

`-Uninstall` removes the prefix and the environment variable. It never touches
your source tree, MSYS2, or anything `pacman` installed.

### Linux and macOS

Not yet automated — tracked in
[issue #226](https://github.com/VEST-Tokamak/vaft/issues/226). Build by hand
with the recipe in
`workflow/automatic_pipeline_1_routine_data_processing/DEPLOYMENT.md`, then set
`CHEASEHOME` / `GPECHOME` the same way. `install/check_chease.py` and
`install/check_gpec.py` run on every platform, so the verification half is
available today.

### Tested toolchain

| Component | Version used for the Windows verification |
| --- | --- |
| MSYS2 | 20260611 (UCRT64 environment) |
| gcc / gfortran | 16.2.0 |
| GNU make | 4.4.1 |
| OpenBLAS | 0.3.34 |
| netCDF-C / netCDF-Fortran | 4.9.3 / 4.6.1, built without S3 |
| CHEASE | `fb46366` |
| GPEC | `e68d7ac2` (v1.5.7-611) |

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
| CHEASE, Windows native | Verified **manually** on a clean Windows 11 machine: build, VAFT discovery, a refinement of a packaged equilibrium, and its comparison metrics. Not automated -- hosted runners have no Fortran toolchain, and a full build takes tens of minutes. |
| DCON/GPEC, Windows native | **Not supported yet.** Builds, resolves and computes correctly, but a run never terminates -- see the AWS S3 note above. WSL2 is the documented fallback. The script-level guarantees are pinned by `test/test_install_bootstrap.py`, which runs in CI on every platform. |
| CHEASE and DCON/GPEC, Linux and macOS | Installers not yet written -- tracked in [issue #226](https://github.com/VEST-Tokamak/vaft/issues/226). The checkers run on every platform today. |
