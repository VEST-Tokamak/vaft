# NUBEAM: build, validate, run

NUBEAM is the NTCC neutral-beam Monte Carlo code. This directory builds it and
establishes that the build is trustworthy; `vaft.code.nubeam` then runs it.

**The source is not here, deliberately.** NTCC requires each user to accept its
licence before downloading NUBEAM, so VAFT owns the build recipe and the adapter
contract while the source stays external. Every script takes `--nubeam-root`
naming a tree you already hold, and writes nothing into the VAFT checkout.

**Linux, macOS / Apple Silicon and native Windows.** None of this runs in CI;
the VAFT test suite passes with NUBEAM absent.

| File | Purpose |
| --- | --- |
| `macos.sh` | Builds NUBEAM, the Plasma State generator, `update_state` and `preact_init`, and populates the PREACT/ADAS databases. |
| `windows.ps1` | The same, for native Windows: finds MSYS2, runs the recipe, colocates the runtime DLLs, and sets `NUBEAMHOME`. |
| `windows.sh` | The Windows build recipe itself, run inside MSYS2 by `windows.ps1`. |
| `linux.sh` | The Linux build. Same contract as `macos.sh`, against distribution packages; it names what is missing rather than installing it. |
| `uninstall.sh` | Removes what a POSIX run generated, driven by `.nubeam-install-manifest` rather than by a list kept here. Refuses any recorded path outside the source tree, and leaves a `share/Make.local` you have edited — it stops being the installer's once you change it. `--dry-run` lists without removing; a second run is a quiet no-op. |
| `run-local-validation.sh` | Runs a shipped reference case (D3D or TFTR) and compares it to the reference output that ships with it. |
| `run-local-vest.sh` | G-EQDSK → Plasma State → NUBEAM, the full VEST chain, locally. |
| `compare-plasma-state.py` | Profile-by-profile comparison of two Plasma State files. |
| `_case_edit.py` | The text edits the shell would otherwise need `sed`/`awk` for. |
| `VALIDATION.md` | What the reference cases actually showed, and how to read it. |

## Usage

Linux — install the toolchain yourself first; `linux.sh` names what is missing
and stops, because a package install needs root:

```bash
apt install gfortran gcc g++ make curl libnetcdff-dev liblapack-dev libblas-dev

bash install/nubeam/linux.sh --nubeam-root ~/git/nubeam --accept-ntcc-terms
export NUBEAMHOME=~/git/nubeam/local
bash install/nubeam/run-local-validation.sh --nubeam-root ~/git/nubeam --case d3d
```

macOS:

```bash
bash install/nubeam/macos.sh --nubeam-root ~/git/nubeam --accept-ntcc-terms
export NUBEAMHOME=~/git/nubeam/local
bash install/nubeam/run-local-validation.sh --nubeam-root ~/git/nubeam --case d3d
```

Windows, from an ordinary PowerShell prompt. **Experimental and unverified:**
the `windows.ps1` wrapper has not been run end to end in the form shipped here
(its PowerShell is verified by reading only), so confirm any success it reports
with `-CheckOnly`. `windows.sh` can also be run directly from a UCRT64 shell.

```powershell
powershell -ExecutionPolicy Bypass -File external\nubeam\windows.ps1 C:\git\NUBEAM -AcceptNtccTerms
powershell -ExecutionPolicy Bypass -File external\nubeam\windows.ps1 C:\git\NUBEAM -CheckOnly
```

`-AcceptNtccTerms` is what authorises the download of PSPLINE, PREACT and
XPLASMA; neither script ever accepts the agreement for you. `windows.ps1` sets
`NUBEAMHOME` as a user variable, which is what reaches a Jupyter kernel: the
registered kernelspec launches `python.exe` directly, so a conda `activate.d`
hook never runs for a notebook.

## What differs on Windows

MinGW-w64 is a Windows target wearing a POSIX-shaped userland, and NUBEAM was
written for the userland rather than for the target. Four gaps follow, and
`windows.sh` fills each in the generated build directory rather than by editing
the NTCC source:

| Gap | What `windows.sh` does |
| --- | --- |
| `portlib/c_execsystem.c` needs `fork`, `execvp`, `clearenv`, `O_NONBLOCK` and `<sys/wait.h>`, none of which MinGW has | Compiles a replacement over `_spawnvpe(_P_WAIT, ...)`, the same operation in one call, keeping portlib's documented status codes. The shell-server optimisation reports itself unavailable, which its Fortran caller already handles by spawning the command itself. |
| `portlib/trsocket.c` calls `close`, `read` and `write` on sockets, which on Windows are not file descriptors | Recompiles that one file with the three redirected to `closesocket`, `recv` and `send`. Scoped to the file, because `sglib/sgsys.c` includes the same header and calls all three on ordinary file descriptors. |
| `mkdir` takes no mode, and `SO_REUSEPORT`, `sys/un.h`, `termios.h` and `endian.h` are absent | A small set of force-included compatibility headers. |
| MSYS2's netCDF links the AWS C++ S3 SDK, whose `atexit` handler deadlocks after the program has finished | Requires a netCDF built without S3 and NCZarr. `install/install_gpec_windows.ps1 -BuildDependencies` produces one, and `windows.ps1` finds it by default. This is the same defect that hung DCON. |

`plasma_state_test` is built only when the 2021 server tree that carries its
source is present under `vendor/server-ntcc-2021`; the NTCC dependency archives
ship no main program for it. Cases that read an existing Plasma State do not
need it, and the build says plainly when it is skipped.

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

**No in-place `sed`.** The edits these scripts need are the two that diverge
between GNU and BSD: `sed -i` takes a backup suffix on BSD and none on GNU, and
the two spell a whole-line replacement differently. `_case_edit.py` does them in
Python instead, reusing the adapter's own implementations so the harness and the
library cannot drift apart. It therefore needs the `vaft` environment;
`macos.sh`, which only drives compilers and `make`, does not.
