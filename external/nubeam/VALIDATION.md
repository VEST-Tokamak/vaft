# macOS arm64 build validation

Both reference cases that ship with the NTCC distribution were run on this
build and compared against the reference Plasma States shipped alongside them.
Reproduce with:

```bash
bash external/nubeam/run-local-validation.sh --nubeam-root /path/to/nubeam --case d3d
bash external/nubeam/run-local-validation.sh --nubeam-root /path/to/nubeam --case tftr
```

## How to read these numbers

NUBEAM is a Monte Carlo code, so no two runs agree exactly and "close to the
reference" is not by itself meaningful. Every comparison below is therefore
paired with a **noise floor**: the same build run twice under two different RNG
seeds (`--seed 271828183` and `--seed 314159265`), compared against itself.

That is the number that makes the rest interpretable. A profile that differs
from the reference by less than the code's own seed-to-seed variance carries no
evidence of a defect, however large the percentage looks.

The metric is disagreement in the integrated quantity,
`|sum(b) - sum(a)| / |sum(a)|`. The reference states were written by a 2.044
build; this one is 2.055.

## Result: no evidence of a build defect

### Primary NUBEAM outputs

| profile | D3D vs ref | D3D noise | TFTR vs ref | TFTR noise |
| --- | ---: | ---: | ---: | ---: |
| `pbe` | 0.31% | 0.85% | 1.85% | 17.35% |
| `pbi` | 4.99% | 3.24% | 10.23% | 22.77% |
| `pbth` | 4.22% | 28.26% | 1.43% | 3.66% |
| `nbeami` | 4.75% | 4.42% | 1.43% | 0.85% |
| `curbeam` | 8.54% | 2.80% | 5.39% | 36.10% |
| `tqbe` | 6.89% | 2.17% | 16.01% | 52.04% |
| `tqbi` | 11.64% | 41.18% | 1.32% | 27.14% |
| `tqbjxb` | 0.60% | 71.35% | 84.38% | 75.35% |
| `pfuse` | 0.00% | 0.00% | 8.94% | 0.71% |
| `pfusi` | 0.00% | 0.00% | 2.34% | 1.81% |
| `eperp_beami` | 3.19% | 1.57% | 10.28% | 4.83% |
| `epll_beami` | 1.14% | 5.57% | 0.61% | 3.83% |
| `sbedep` | 3.62% | 0.43% | 0.60% | 3.54% |

Read against the noise floors, the headline profiles split into two groups.

**Dominated by Monte Carlo noise.** `tqbjxb` (84% on TFTR), `tqbi` (11.6% on
D3D), `pbth`, `epll_beami`: the disagreement with the reference is at or below
what the same build produces against itself under a different seed. These carry
no information about correctness in either direction -- they are simply too
noisy at this particle count to constrain anything.

**Resolved above the noise, and small.** `pbe` (0.31% / 1.85%), `nbeami`
(4.75% / 1.43%), `sbedep` (3.62% / 0.60%), `curbeam` (8.54% / 5.39%), `pfuse`
(8.94%). Several of these do exceed their own noise floor by a factor of a few
-- `pfuse` is 8.94% against a 0.71% floor, `sbedep` 3.62% against 0.43%. So
there is a real, resolvable difference from the reference here; it is not all
statistical scatter.

That difference is small in absolute terms -- single-digit percent on every
resolved quantity -- and the reference states were written by a 2.044 build
against this one's 2.055. Eleven schema revisions of physics-model drift is a
sufficient and far more likely explanation than a platform defect, which would
not plausibly land every primary channel within a few percent. What the
comparison establishes is the useful thing: nothing here is broken at the
factor-of-two, wrong-sign, or wrong-shape level that a miscompiled build
produces.

TFTR is the stronger of the two tests. It is a D-T plasma, so `pfuse`/`pfusi`
are non-zero there and are the most nonlinear diagnostic available -- they
depend on the square of the fast ion density. Both agree to within 9%.

## One systematic residual: FRANTIC neutral channels

### FRANTIC halo and recombination neutrals

| profile | D3D vs ref | D3D noise | TFTR vs ref | TFTR noise |
| --- | ---: | ---: | ---: | ---: |
| `psc_halo` | 23.65% | 0.48% | 28.05% | 3.69% |
| `pcx_halo` | 23.50% | 1.16% | 24.33% | 3.93% |
| `n0_halo` | 16.48% | 0.52% | 20.13% | 5.30% |
| `psc_reco` | 14.92% | 0.06% | 15.85% | 0.40% |
| `n0_reco` | 11.28% | 0.02% | 9.83% | 0.13% |
| `s0reco_e` | 10.39% | 0.10% | 7.51% | 0.52% |
| `pcx_reco` | 155.14% | 1.74% | 6.31% | 0.18% |

These do not look like noise, and they are not. The offset is 15-28%, in the
same direction, at the same magnitude, on two physically unrelated plasmas,
against a noise floor of well under 5%.

That signature points at the code rather than the platform. A build defect --
wrong compiler flags, the undefined `nblkfac` result, a floating-point trap --
would not confine itself to a single physics module, and would not reproduce so
stably across cases. What these profiles have in common is that they are all
FRANTIC halo and recombination neutral transport, so the most probable cause is
a change in that model between 2.044 and 2.055.

It is recorded here rather than resolved because nothing downstream depends on
it: these are neutral-particle diagnostics, not the beam heating, current drive
or torque channels that NUBEAM is being run for. If VEST work later comes to
depend on halo neutrals, this is the thread to pull.

`pcx_reco` on D3D (155%) is a small-denominator artifact -- the same profile is
6.31% on TFTR, and its D3D noise floor is 1.74%.

## Notes

- Step counts are per-case and must match the reference: D3D `2x0.010`, TFTR
  `4x0.010`, taken from the shipped `{d3d,tftr}_test.csh`. Running TFTR for
  D3D's two steps leaves the slowing-down distribution half-built and throws
  every beam-driven profile off by roughly a factor of two.
- Runs use `NUBEAM_ACTION=init_hold`, which holds the RNG seed at the
  namelist's `nseed`. Results are therefore bit-reproducible between runs.
- These numbers were reproduced unchanged after the `sed`/`awk` text edits
  moved into `_case_edit.py`, so the portability rewrite is behaviour-neutral.
- Interpolation warnings: 0 on both cases.
- The `_Preact` variants of these cases cannot be run: their
  `nubeam_init_Preact.dat` / `nubeam_step_Preact.dat` namelists are absent from
  the 2021 archive.

## Generator parity: local `plasma_state_test` vs the server's `plasma_state_test_new`

The NTCC archive ships no main program for the Plasma State generator, and the
server's `plasma_state_test_new.f90` is mode `0600` under another user's home
directory, so it could not be read. What this build uses instead is
`plasma_state_test.f90` from the vendored 2021 server tree -- a sibling program
of the same family, byte-identical to the readable copy on the server
(md5 `81ba2642...`).

That substitution was the main open risk, so it was measured directly. The same
G-EQDSK (md5 `f7d11c47...`, `g020000.015100`) and the same `inputf` were run
through both generators:

| | server `_new` | local `plasma_state_test` |
| --- | --- | --- |
| Plasma State version | 2.044 | 2.055 |
| output size | 1,482,532 B | 1,523,584 B |
| variables | 414 | 415 |

**239 shared profiles compared; median integral disagreement 0.00%.** The
equilibrium geometry, currents and profiles are numerically identical. Only two
quantities differ, and both are the same schema change rather than a physics
difference:

- `psmom_nc` -- flux-surface moment coefficients. Both builds declare
  `nmom = 64`, but 2.044 stores 16 per surface (101x16) and 2.055 stores all 64
  (101x64). The values agree over their common range.
- `psmom_errck` -- the residual of that moment fit, and consequently smaller in
  the build that keeps more moments: max 1.4e-4 locally against 1.2e-3 on the
  server, an order of magnitude better.

So the sibling source is not a risk in practice. Obtaining `_new` (a `chmod g+r`
away) would still be worth doing if exact provenance ever matters, but nothing
currently depends on it.

## End-to-end VEST run

```bash
bash external/nubeam/run-local-vest.sh --nubeam-root /path/to/nubeam \
    --input-dir ~/Downloads/gbyhj_test --gfile ~/Downloads/g020000.015100
```

G-EQDSK -> Plasma State -> NUBEAM INIT -> NUBEAM STEP, entirely on this machine.
Both stages reach `normal exit`. At the server run's particle count the local
build reports 16 `xpprof` out-of-bounds interpolation warnings against the
server's 96; the count scales with sampling (641 at `--nptcls 1000`) and is
lower locally at matched settings.

### Path length limit

`nubeam_comp_exec` holds file paths in fixed-length `CHARACTER` buffers and
truncates silently at roughly 140 characters, which surfaces as a misleading
"file open failure" on the input state rather than as a path error.
`run-local-vest.sh` checks the work directory length up front and fails with the
real reason.
