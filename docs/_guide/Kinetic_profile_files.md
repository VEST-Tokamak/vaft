---
title: Kinetic profile file formats
author: VEST team
date: 2026-09-14 09:00
category: guide
layout: post
permalink: /reference/kinetic-profile-files/
guide:
  architecture: One container behind three profile file formats, and the TRANSP conversion that fills it.
  prerequisites: A TRANSP run, or a .kin, pfile or MARS deck to read.
  expected: A KineticProfiles set whose units and radial coordinate are stated rather than assumed.
---

Three file formats carry kinetic profiles into the codes VAFT drives, and they disagree about
almost everything: what the radial coordinate is, what units the numbers are in, and what the
rotation column means. `vaft.data.kinetic_profiles.KineticProfiles` is the one container all three
read into, and this page is the contract each of them keeps with it.

## The container fixes the units

Densities in m⁻³, temperatures in eV, angular frequencies in rad/s, pressures in Pa, electric fields
in V/m — the set `vaft.data.KINETIC_UNITS` names, and the set GPEC's own reader consumes, so the
`.kin` path needs no conversion at all. Every reader converts into them and records what it
converted from. Nothing carries free-text units around.

Two fields exist because one of them kept being written into the other:

| field | what it is |
| --- | --- |
| `omega_tor` | toroidal angular velocity, ω_φ |
| `omega_exb` | E×B rotation frequency, ω_E |

They differ by the ion diamagnetic frequency. The `.kin` rotation column is ω_E, so `write_kin`
refuses a set that carries only ω_φ.

## The radial coordinate is never rescaled on read

`normalization.method` says how the coordinate came to be what it is, and the value is not always
`"as_read"`:

| format | coordinate in the file | `method` |
| --- | --- | --- |
| `.kin` | ψ_N | `as_read` |
| pfile | ψ_N (`psinorm`) | `as_read` |
| MARS `PROF*.IN` | `s` = √ψ_poloidal (header key 1) | `mars_s_squared` |
| TRANSP | `PLFLX / PLFLXA` on `XB` | `transp_plflx_over_plflxa` |

Making a coordinate span exactly [0, 1] is `normalize_psi()`, a separate named operation that has no
default method — passing `axis=`/`edge=` and silently getting a min–max stretch instead would be the
rescale this whole design exists to prevent, only with a provenance record vouching for it.

## Reading a TRANSP run

<!-- docs-snippet: skip needs-external-code (runs an external code or pipeline stage) -->
```python
from vaft.code.transp import read_transp_profiles

profiles = read_transp_profiles("45453X01.CDF", time_s=0.750)
profiles.psi_norm[-1]                 # exactly 1.0 — the last closed flux surface
profiles.provenance["omega_exb"]      # "-d(VRPOT)/d(PLFLX) ... [VOLTS per Wb/rad = rad/s]"
profiles.provenance["target_grid"]    # which grid, and why
```

This is the one place under `vaft.code` that converts anything. Every other module there keeps its
code's own names and units; this one converts because turning a run into a `.kin` needs the two
radial grids reconciled and the potential differentiated, and doing that at each call site is how
the grids came to be conflated in the first place.

**ψ_N is `PLFLX / PLFLXA`** — the flux enclosed by each zone boundary over the flux enclosed by the
plasma boundary. Not `(P − P[0]) / (P[−1] − P[0])`, which declares the innermost zone boundary to be
the magnetic axis and moves the whole grid inward by 0.0049 for the MAST reference run.

**ω_E is `−dΦ/dψ` from `VRPOT` and `PLFLX`**, and needs no unit conversion: `VRPOT` is in volts and
`PLFLX` in webers per radian, and a volt per weber is an inverse second.

**The result is on the zone-boundary grid `XB`.** TRANSP writes the kinetic profiles on the zone
centres `X` and the flux on the boundaries `XB`, and the two interleave — `X[i]` is half a cell
inside `XB[i]`, and in a real run they are the *same length*, which is why nothing anywhere resolves
a grid by counting. Choosing `XB` leaves ψ_N and ω_E untouched and interpolates only the smooth
kinetic profiles. Measured on the reference run at 750 ms, the choice is not symmetric:

| default `target_grid="XB"` | `target_grid="X"` |
| --- | --- |
| edge ψ_N stays exactly 1.0 | edge ψ_N falls to 0.9857, losing the last closed surface |
| `n_e` moves ≤ 4.2% of peak, resampling | ψ_N moves by up to 0.036 |
| the one clamped point is the outermost, and moves by nothing | the clamped point is the **innermost**, where a constant stands in badly for a quantity varying quadratically towards the axis |

The edge matters because GPEC's `read_kin` re-splines onto a uniform [0, 1] grid *with
extrapolation*: a set whose outermost point is the last closed surface is the one it can use.
Whichever grid is chosen, the provenance records which quantities were interpolated and how many
points fell outside their source range.

### What changed against the legacy converter

The scripts this replaces produced `sample/transp_example/input/g045453.00750.kin` by going through
a 201-point pfile. Reading the same run directly:

| quantity | agreement |
| --- | --- |
| `n_e`, `n_i` | ≤ 1.8% of peak |
| `T_e`, `T_i` | ≤ 1.2% of peak |
| `omega_tor` | ≤ 7.8% of peak, all points within 10% |
| `omega_exb` | 97.5% of points within 10% of peak (98.5% against the pfile's own `omgeb`) |

The profile deltas are resampling — the reference was interpolated twice, CDF → 201-point pfile →
`.kin`, against one step here.

ω_E is the one quantity with real outliers, and there are five rather than one: ψ_N = 0.120, 0.125,
0.990, 0.995 and 1.000, the largest being 26% at the edge. Three are at the boundary, where a
one-sided derivative on the file's own 20-point grid differs from the same formula applied to a
ten-times interpolant of it. The pair at ψ_N ≈ 0.12 is not explained by that and is a genuine
difference in where the derivative is evaluated. Against the pfile's own `omgeb`, which is derived
the same way, three points exceed 10% instead of five.

The migration's fix-forward decision records such a change rather than driving it to zero.
Differentiating the file's own data is the more defensible of the two, and `omega_tor` is now read
from `OMEG_VTR` — the measured charge-exchange rotation the legacy writer preferred — rather than
from `OMEGA`, which is a different quantity in the same file.

## The three file formats

`.kin` (GPEC), the Osborne pfile, and MARS `PROF*.IN` each have their own section in
[Equilibrium and kinetic profiles]({{ site.baseurl }}/workflows/equilibrium-kinetic-profiles/),
covering the header and footer rules, the unit ladders, the rotation mappings and what each reader
refuses. In short:

- **`.kin`** — six columns, `psi ni ne ti te wexb`. The header/footer rule is GPEC's own
  `readtable`: the first contiguous block of numeric lines, and nothing after it.
- **pfile** — a species block and 22 sections, each with its own unit and an inline derivative
  column that is preserved rather than recomputed, because recomputing it reproduces none of the
  reference files.
- **MARS `PROF*.IN`** — one quantity per file. `PROFROT.IN` is ω_φ and `PROFWE.IN` is ω_E; they are
  never merged and neither takes precedence.
