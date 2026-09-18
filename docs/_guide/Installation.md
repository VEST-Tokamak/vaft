---
title: Start here
author: Sun jae Lee
date: 2026-07-01 09:00
category: guide
layout: post
permalink: /workflows/start-here/
guide:
  architecture: Entry point from installation to an offline ODS and optional public data.
  prerequisites: Python 3.10–3.13, Git, and a fresh virtual environment.
  expected: A local plasma-current plot, followed optionally by read-only shot 39915 metadata.
  status: Verified offline and against public HSDS.
related:
  notebooks: [database-initialization, plotting-sample]
  api: [omas, database, plot]
  data_sources: [sample-ods, hsds-public]
  outputs: [first-result, hsds-39915]
---

This path gets a new user from a clean Python environment to a visible result without requiring
database credentials or an external fusion code. Public VEST data access is the optional second step.

## 1. Install VAFT from source

VAFT supports Python 3.10–3.13. Use a virtual environment and install the repository source:

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

The declared dependencies include OMAS, IMAS-Python, plotting tools, Snakemake and `h5pyd`. NumPy 2
is the default numerical stack.

`pip install vaft` installs the latest published release from PyPI instead. Install from source, as
above, when you need unreleased changes from `develop` or the tutorial notebooks that live in the
repository.

### Optional-dependency groups

The project defines five extras; none is needed for the first result on this page:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `sklearn` | scikit-learn | `fit_profile(fitting_function='gp_sklearn')`; the default `'gp'` mode runs on SciPy alone |
| `surrogate` | onnxruntime | running a TGLF neural-network surrogate (`vaft.code.gacode.tglf.surrogate`); resolving a model and auditing an input need no extra |
| `tokamaker` | openfusiontoolkit | `vaft.code.tokamaker`, the one external code VAFT drives in-process |
| `ml` | torch, onnx, onnxruntime, scikit-learn, skl2onnx | the `torch` and `sklearn` backends of `vaft.process.ml` and ONNX export; datasets, splits, the `numpy` backend and resolving a published model need no extra |
| `dev` | pytest, pytest-xdist, pre-commit and the two runtimes above | running the test suite and contributing |

```bash
python -m pip install -e ".[dev]"            # development tooling
python -m pip install -e ".[sklearn,surrogate]"
```

### Updating an existing installation

Because the installation is editable, updating the checkout updates VAFT. Run these from the checkout,
setting your own edits aside first. Running a notebook counts as editing it, because its outputs are
saved into the file:

```bash
git status
git stash push -m "before VAFT update"   # only if `git status` lists modified files
git pull --ff-only
git stash pop                             # only if you stashed
python -m pip install -e .
```

Restart any running Jupyter kernel afterwards; a kernel that was already running keeps the old VAFT in
memory. The [update procedure in install/README.md](https://github.com/VEST-Tokamak/vaft/blob/develop/install/README.md#updating-vaft)
covers each step, what to do when `git pull` or `git stash pop` stops, and which commands must never be
used to recover.

## 2. Produce the first offline result

The packaged sample follows the same OMAS/IMAS paths as a VEST shot:

```python
import matplotlib.pyplot as plt
import vaft

ods = vaft.omas.sample_ods()
print(sorted(ods.keys()))

vaft.omas.plot_plasma_current_time(ods)
plt.show()
```

Seeing the plasma-current trace completes the credential-free first workflow. Continue with
[experimental interpretation]({{ site.baseurl }}/workflows/experimental-interpretation/) or configure
the public database below.

## 3. Configure read-only public HSDS access

Run `hsconfigure` and enter the endpoint plus credentials supplied by the VEST team. Credentials stay
in the user configuration and must never be committed to a notebook or documentation asset.

```bash
>> hsconfigure
Enter new values or accept defaults in brackets with Enter.

Server endpoint []: http://147.46.36.244:5101
Username []: [assigned_username]
Password []: [assigned_password]
API Key [None]: 
Testing connection...
connection ok
Quit? (Y/N)Y
```

A successful read uses the public namespace and does not modify the database:

<!-- docs-snippet: skip needs-database (talks to a VEST database source) -->
```python
import vaft

with vaft.database.open(39915, source="public", paths="equilibrium") as ods:
    print(ods["equilibrium.time"])
```

Use `vaft.database.open()` for lazy exploratory reads and `vaft.database.load()` when a workflow needs
a staged eager object. Remote saving is restricted to authorized operators; this workflow never calls
`vaft.database.save()`.

## 4. Optional external fusion codes

VAFT can prepare and collect inputs for EFIT, CHEASE, GPEC/DCON/RDCON, TES, NUBEAM, the GACODE suite
(NEO and TGLF) and TokaMaker. Configure only the codes you have installed:

```bash
export EFITHOME=/path/to/efit
export CHEASEHOME=/path/to/chease
export GPECHOME=/path/to/gpec
export TESHOME=/path/to/tes
export NUBEAMHOME=/path/to/nubeam
export GACODEHOME=/path/to/gacode
```

Each executable belongs under its root’s `bin/` directory. The workflow guides degrade to deterministic
input preparation when a binary is absent.

**TokaMaker is the exception.** VAFT drives it in-process rather than as a subprocess, so there is no
`bin/` and no `$TOKAMAKERHOME`: "installed" means "importable in this interpreter". Its upstream, the
Open FUSION Toolkit, publishes wheels, so pip can express it — and it stays an optional extra, because
the wheel carries compiled libraries and every other workflow works without it:

```bash
pip install 'vaft[tokamaker]'     # or: pip install openfusiontoolkit
```

Nothing needs to be exported afterwards. `vaft.code.tokamaker` raises an actionable `ImportError` when
the toolkit is absent, and `OFT_ROOTPATH` is only for pointing at a source build instead of the wheel.

EFIT is licensed software that VAFT neither bundles nor fetches: obtain authorized access to the
source through the EFIT-AI channel and agree to its users agreement first, then build it from your
own tree with `install/install_efit.sh`, or on native Windows with
`install/install_efit_windows.ps1 <source> -AcceptEfitUsersAgreement` (see the EFIT section of
[install/README.md](https://github.com/VEST-Tokamak/vaft/blob/develop/install/README.md)). One
`EFITHOME` serves both the reconstruction code (`bin/efit`) and the Green-table generator
(`bin/efund`); there is no separate root for EFUND.

NUBEAM differs from the others in two ways. Its root must also hold the PREACT and ADAS reaction
databases at `share/preact` and `share/adas`, because `nubeam_comp_exec` aborts when either is
unset, and both must stay writable — the table code caches newly computed reaction tables into them.
And VAFT ships the build recipe rather than the source, since NTCC requires each user to accept its
licence first; see [`install/nubeam/`](https://github.com/VEST-Tokamak/vaft/tree/develop/install/nubeam).
That build runs on macOS/Apple Silicon and on native Windows. The adapter runs NUBEAM and parses
its native output; `vaft.machine_mapping.core_sources` and `vaft.machine_mapping.distributions` map
the profiles into IMAS, while the Monte Carlo marker records stay in the native container.

GACODE differs from every other code here in three ways, and each one breaks an assumption stated
above. It **builds in place**, so `$GACODEHOME` is the source checkout rather than a separate prefix.
Each suite member carries its own `bin`, so the executables are `neo/bin/neo` and `tglf/bin/tglf`, not
`bin/neo`. And it needs `$GACODE_PLATFORM`, the tag the tree was built with, which selects
`platform/exec/exec.$GACODE_PLATFORM` at run time — an unset or wrong value otherwise fails deep inside
a shell script without naming itself. VAFT derives GACODE's own `GACODE_ROOT` and `GACODE_PLATFORM`
from `$GACODEHOME` rather than redefining them, and accepts a pre-set `GACODE_ROOT` as a fallback, so a
tree built this way stays usable from a plain shell:

`$GACODE_PLATFORM` is a build tag rather than a directory, so it does not belong in the roots block
above:

```bash
export GACODE_PLATFORM=GFORTRAN_OSX_BREW
```

Build it with `install/gacode/linux.sh` or `install/gacode/macos.sh --gacode-root <source>`
— both build `neo,tglf` by default, the set VAFT drives — and
verify with `install/check_gacode.py`. VAFT drives NEO for neoclassical transport and the bootstrap
current, and TGLF for turbulent transport; see
[`install/gacode/`](https://github.com/VEST-Tokamak/vaft/tree/develop/install/gacode).

The TGLF-NN surrogate is a different kind of dependency and is worth separating from the rest of this
section: it needs **no GACODE build and no compiler**. What it needs is pretrained networks, which VAFT
neither ships nor downloads:

```bash
export TURBULENTTRANSPORTHOME=/path/to/TurbulentTransport.jl    # model files, no bin/
```

That root holds no executables at all — it is a
[TurbulentTransport.jl](https://github.com/ProjectTorreyPines/TurbulentTransport.jl) checkout whose
`models/` directory carries the pretrained ensembles. VAFT also finds one in an existing Julia depot
with no variable set. `onnxruntime` is optional
(`pip install 'vaft[surrogate]'`) and is needed only to *run* a network; deciding whether a model
applies to a given plasma needs neither the runtime nor a prediction, and is the question to ask first.

VAFT's own trained models are a third kind. `vaft.process.ml` trains them, and
[`vaft-nn`](https://github.com/VEST-Tokamak/vaft-nn) publishes them. This is a **private registry
repository**: each version's manifest and hashes are in git, and the weights are GitHub Release
assets. Point VAFT at a checkout of it the same way:

```bash
git clone git@github.com:VEST-Tokamak/vaft-nn.git ~/git/vaft-nn
export VAFT_NN_HOME=~/git/vaft-nn         # registry: models/<name>/releases.yaml + manifests
export VAFT_NN_CACHE=~/scratch/vaft-nn    # optional; default is the platform cache directory
gh auth login                             # once; fetching reuses the GitHub CLI's login
python install/check_vaft_nn.py           # registry, cache and gh access, layer by layer
```

`vaft.process.ml.fetch_model(name, version=...)` downloads a release into the cache.
Alternatively, `load_model(..., fetch=True)` fetches on a cache miss. Either way, a file is accepted
only if its SHA-256 matches the manifest the registry pins. VAFT never reads or stores a GitHub token
itself. Training and inference with the `torch` and `sklearn` backends need
`pip install 'vaft[ml]'`; resolving and verifying a model does not.

On Windows, set the same roots as user environment variables so that a new
terminal and a Jupyter kernel both inherit them:

```powershell
[Environment]::SetEnvironmentVariable('CHEASEHOME', "$env:LOCALAPPDATA\vaft\external\chease", 'User')
[Environment]::SetEnvironmentVariable('GPECHOME',   "$env:LOCALAPPDATA\vaft\external\gpec",   'User')
[Environment]::SetEnvironmentVariable('EFITHOME',   "$env:LOCALAPPDATA\vaft\external\efit",   'User')
[Environment]::SetEnvironmentVariable('GACODEHOME', "$env:LOCALAPPDATA\vaft\external\gacode", 'User')
```

The executable under `bin/` may be the native `chease.exe` or `dcon.exe`; VAFT
resolves the documented POSIX name to it.

CHEASE and the DCON/GPEC suite build on every platform from `install/`. On Linux
and macOS, against a checkout you obtained yourself:

```bash
bash install/install_chease.sh --source ~/git/CHEASE
bash install/install_gpec.sh   --source ~/git/GPEC
```

Each installs into `<source>/vaft-install` — the path to point `CHEASEHOME` or
`GPECHOME` at, which the script prints when it finishes — and then runs the
matching checker, which refines a packaged equilibrium for CHEASE and drives the
real DCON-to-GPEC handoff for GPEC. `install/README.md` carries the per-platform
detail, including the Debian package list and the three Linux build settings that
otherwise fail quietly.


## Expected outputs

- Offline: a plasma-current plot from the packaged sample ODS.
- HSDS: readable metadata or IDS paths for public shot 39915.
- External codes: an explicit readiness report before any solver is launched.
