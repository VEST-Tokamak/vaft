# NUBEAM: build, validate, run

NUBEAM is the NTCC neutral-beam Monte Carlo code. This directory builds it and
establishes that the build is trustworthy; `vaft.code.nubeam` then runs it.

**The source is not here, deliberately.** NTCC requires each user to accept its
licence before downloading NUBEAM, so VAFT owns the build recipe and the adapter
contract while the source stays external. Every script takes `--nubeam-root`
naming a tree you already hold, and writes nothing into the VAFT checkout.

**macOS / Apple Silicon only.** Linux and Windows are tracked in
[issue #226](https://github.com/VEST-Tokamak/vaft/issues/226). None of this runs
in CI; the VAFT test suite passes with NUBEAM absent.

| File | Purpose |
| --- | --- |
| `macos.sh` | Builds NUBEAM, the Plasma State generator, `update_state` and `preact_init`, and populates the PREACT/ADAS databases. |
| `run-local-validation.sh` | Runs a shipped reference case (D3D or TFTR) and compares it to the reference output that ships with it. |
| `run-local-vest.sh` | G-EQDSK → Plasma State → NUBEAM, the full VEST chain, locally. |
| `compare-plasma-state.py` | Profile-by-profile comparison of two Plasma State files. |
| `_case_edit.py` | The text edits the shell would otherwise need `sed`/`awk` for. |
| `VALIDATION.md` | What the reference cases actually showed, and how to read it. |

## Usage

```bash
bash external/nubeam/macos.sh --nubeam-root ~/git/nubeam --accept-ntcc-terms
export NUBEAMHOME=~/git/nubeam/local
bash external/nubeam/run-local-validation.sh --nubeam-root ~/git/nubeam --case d3d
```

## Two portability constraints worth knowing

**Paths are budgeted, not merely long.** `nubeam_comp_exec` composes every
filename in a `character*140` buffer (`subroutine echo`,
`nubeam_comp_exec.F90:2058`), so

```
len(workdir) + 1 + len(runid) + 32  <=  140
```

A longer path is truncated with no diagnostic and fails later as
`?plasma_state_get: file open failure`, which names the input state rather than
the path. `vaft.code.nubeam` checks this before running and says so;
`vaft.compat.short_temporary_directory` allocates a scratch directory that fits.

**No `sed` or `awk`.** The edits these scripts need are the two that diverge
between GNU and BSD: `sed -i` takes a backup suffix on BSD and none on GNU, and
the two spell a whole-line replacement differently. `_case_edit.py` does them in
Python instead, reusing the adapter's own implementations so the harness and the
library cannot drift apart. It therefore needs the `vaft` environment;
`macos.sh`, which only drives compilers and `make`, does not.
