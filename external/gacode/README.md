# GACODE: build and verify

GACODE is the General Atomics code suite for kinetic and transport modelling.
This directory builds it; `vaft.code.gacode` then runs it. NEO, the
drift-kinetic neoclassical solver, is the first backend VAFT drives; TGLF and
CGYRO share the same profile and runtime layer and are tracked in
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

**macOS / Apple Silicon.** Linux and Windows are not covered here. None of this
runs in CI; the VAFT test suite passes with GACODE absent.

| File | Purpose |
| --- | --- |
| `macos.sh` | Installs the Homebrew dependencies, builds the shared and `f2py` libraries and the requested suite members, and optionally runs the NEO `reg18` regression case. |

## Usage

```bash
bash external/gacode/macos.sh --gacode-root ~/git/gacode --check
export GACODEHOME=~/git/gacode
python install/check_gacode.py --source ~/git/gacode
```

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

## Two failure modes worth knowing before you hit them

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

## Verified

Built against `gafusion/gacode` `6357db30` (2026-07-22) with Homebrew
gfortran 15.2 and Open MPI on macOS/arm64. The NEO `reg18` regression case
reproduces its shipped `out.neo.prec` value `0.12268957E+02` exactly.
