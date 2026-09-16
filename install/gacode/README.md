# GACODE: build and verify

GACODE is the General Atomics code suite for kinetic and transport modelling.
This directory builds it; `vaft.code.gacode` then runs it. VAFT drives two suite
members today: **NEO**, the drift-kinetic neoclassical solver, and **TGLF**, the
quasilinear turbulent-transport model. CGYRO shares the same profile and runtime
layer and is still tracked in
[issue #553](https://github.com/VEST-Tokamak/vaft/issues/553).

**The source is not here, deliberately.** VAFT owns the build recipe and the
adapter contract; the source stays external, obtained from
[gafusion/gacode](https://github.com/gafusion/gacode). Every script takes
`--gacode-root` naming a tree you already hold and writes nothing into the VAFT
checkout.

**GACODE builds in place.** There is no separate installation prefix: the
executables land inside the source tree (`neo/src/neo`, launched through
`neo/bin/neo`). `$GACODEHOME` therefore points at the checkout itself, which is
why this code has no `<root>/local` the way NUBEAM does.

**macOS / Apple Silicon and Linux.** Windows is not covered here. None of this
runs in CI; the VAFT test suite passes with GACODE absent.

| File | Purpose |
| --- | --- |
| `macos.sh` | Installs the Homebrew dependencies, builds the shared and `f2py` libraries and the requested suite members, and optionally runs the NEO `reg18` regression case. |
| `linux.sh` | The same build, against distribution packages. Names what is missing rather than installing it. |

Both default to `--codes neo,tglf`, which is the set `install/check_gacode.py` requires and `vaft.code.gacode` resolves; a `neo`-only tree fails its own verification.

## Usage

```bash
# Linux
bash install/gacode/linux.sh --gacode-root ~/git/gacode --check
# macOS / Apple Silicon
bash install/gacode/macos.sh --gacode-root ~/git/gacode --check


export GACODEHOME=~/git/gacode
export GACODE_PLATFORM=TUMBLEWEED        # Linux; GFORTRAN_OSX_BREW on macOS
python install/check_gacode.py --source ~/git/gacode
```

On Linux, install the toolchain yourself first — `linux.sh` names what is
missing and stops, because a package install needs root and a compiler is your
decision:

```bash
apt install gfortran make openmpi-bin libopenmpi-dev \
            liblapack-dev libblas-dev libfftw3-dev libnetcdff-dev
```

### Why `TUMBLEWEED` on Linux

Upstream ships about ninety platform tags and every one is named for a site or
a distribution; there is no generic "Linux + gfortran" entry to select.
`TUMBLEWEED` is the default because its settings are the ones a stock Linux box
already satisfies: `mpifort`, `-fallow-argument-mismatch` (which gfortran 10 and
newer require), and system `lapack`/`blas`/`fftw` rather than a hand-built
OpenBLAS under somebody's home directory — which is what rules out the
otherwise-similar `MINT`. `--platform` overrides it if your site has its own.

Two consequences worth knowing. `TUMBLEWEED` compiles with `-march=native`, so
the binaries are tuned to the machine that built them; that is right for a local
build and wrong for one you mean to copy to a different CPU. And the tag selects
`platform/exec/exec.$GACODE_PLATFORM` as well as the build file, so `linux.sh`
refuses a tag that has only one of the two — a tag that builds but cannot launch
fails later, inside a shell script, without naming itself.

## The environment contract, and why VAFT does not replace it

GACODE's own build and run scripts read two variables:

| Variable | Meaning |
| --- | --- |
| `GACODE_ROOT` | the suite tree |
| `GACODE_PLATFORM` | selects `platform/build/make.inc.$GACODE_PLATFORM` for the build and `platform/exec/exec.$GACODE_PLATFORM` for the run |

VAFT adds `GACODEHOME` to match the `$XHOME` convention every other external
code in this repository uses (`GPECHOME`, `CHEASEHOME`, `EFITHOME`,
`NUBEAMHOME`), and **derives** `GACODE_ROOT` and `GACODE_PLATFORM` from it for
the subprocess rather than redefining them. A tree built here therefore stays
usable from a plain shell that sources `shared/bin/gacode_setup`, and
`vaft.code.gacode` accepts a pre-set `GACODE_ROOT` as a compatibility fallback
when `GACODEHOME` is unset.

## Three failure modes worth knowing before you hit them

**The launcher needs `pygacode` on `PYTHONPATH`.** `neo/bin/neo` shells out to
`neo_parse.py`, which imports `gacodeinput` from `f2py/pygacode`. When that
import fails the launcher does *not* stop -- it carries on, and NEO then aborts
with a Fortran runtime error about a missing `./input.neo.gen`, which points at
the wrong thing entirely. `vaft.code.gacode` always sets `PYTHONPATH` itself
for this reason.

**`GACODE_PLATFORM` must match the build.** `neo/bin/neo` executes
`platform/exec/exec.$GACODE_PLATFORM`; an unset or wrong value fails deep inside
a shell script without naming the variable. `vaft.code.gacode` resolves it
explicitly and lists the available platforms when it cannot.

**Two installed TurbulentTransport versions can answer to one model name.** A
Julia depot routinely carries several releases of a package side by side, and
upstream's own `models/SEMVER` says a minor bump means the TGLF settings behind
an unchanged family name changed. VAFT hashes every ensemble member *and* every
normalisation file and **refuses** when two roots disagree, rather than picking
the first -- naming both locations and their versions so you can choose with
`model_dir=` or an explicit path. Copies that agree resolve normally and the
extra location is recorded.

## TGLF-NN models are a separate external artifact

The surrogate backend (`vaft.code.gacode.tglf.surrogate`) needs no GACODE build and
no compiler. It needs pretrained networks, which VAFT does not ship: they are large,
they are upstream's, and vendoring them would put model weights in a physics
repository. Nothing here downloads them either -- `import vaft` stays offline.

Point VAFT at a checkout you already have:

```bash
export TURBULENTTRANSPORTHOME=/path/to/TurbulentTransport.jl
```

(`TURBULENTTRANSPORT_ROOT` is accepted too, the way `GACODE_ROOT` is.) A Julia depot
that already has the package is found without any variable set. Resolution order is
explicit path, then `model_dir=`, then the variable, then the depot; when two roots
hold the same family name with different bytes VAFT refuses rather than choosing,
because upstream versions the networks behind a stable name.

Only *running* a network needs `onnxruntime` (`pip install 'vaft[surrogate]'`).
Deciding whether a model applies to a given plasma does not -- that is a question
about the input and the training moments, and `audit_training_domain` answers it
with neither the runtime nor a prediction.

What that answer can and cannot be: the ONNX distribution ships the normalisation
moments (`xm`/`xsigma`) but **not** the per-input training bounds, which exist only
inside the upstream Julia `.bson`. So the measure is a standard-deviation distance,
not a containment test, and every audit reports `bounds_available = False` to say
so. Read `in_domain` as "nothing is far from what this model was trained on", never
as a guarantee.

Upstream publishes ONNX for only 14 of its ~100 families, all of them
spherical-tokamak; the rest are Julia `.bson` and cannot be read from Python.

## Verified

**macOS/arm64.** Built against `gafusion/gacode` `6357db30` (2026-07-22) with
Homebrew gfortran 15.2 and Open MPI. The NEO `reg18` regression case reproduces
its shipped `out.neo.prec` value `0.12268957E+02` exactly.

**Linux/x86_64.** Built against `gafusion/gacode` `b49339750` with
gfortran 11.4.0 and Open MPI 4.1.2 on Ubuntu 22.04.4, `GACODE_PLATFORM=TUMBLEWEED`,
`--codes neo,tglf`. `reg18` reproduces `0.12268957E+02` — **the same value, to
every digit, as the macOS build above** — and `install/check_gacode.py` reports
every layer green.

The two platforms agreeing bit-for-bit on `reg18` is the point of recording the
number here rather than only "it passed".
