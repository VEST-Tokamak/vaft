# Issue #666 synthetic-equilibrium isolation

This study separates diagnostic consistency and VacTH initialization from the
subsequent NICE Grad--Shafranov reconstruction. It uses the same 41672/331 ms
VEST geometry, currents, 63 probes, five flux loops, uncertainties, COCOS 11
path, mesh, and pinned NICE executable as the reference run.

Two noiseless plasma sources are independent of the measured magnetic data:

1. A closed Solov'ev equilibrium (axis R=0.4 m, kappa=1.4), normalized to
   125798.1 A.
2. A local OpenFUSIONToolkit/TokaMaker free-boundary solution at exactly 331 ms
   (axis R=0.4770 m, Ip=125893.9 A).

For both, the toroidal current density is integrated inside the LCFS and the
external diagnostics are generated with VAFT's independent filament Green
function. The analytic Solov'ev polynomial is not extrapolated outside the
plasma. NICE-native active-coil responses are added exactly once; passive
response is absent because this is the already passive-subtracted native
representation. Flux-loop input uses the validated sign of -1.

## Results

| source | VacTH Ip | Ip error | max probe error | max loop error | full iterations | final linear residual | full reconstructed Ip | magnetic cost | numerical stage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Solov'ev | 125941 A | +0.114% | 0.026 sigma | 0.026 sigma | 29 | 6.48e-11 | 76983.6 A | 3428.25 | success |
| TokaMaker | 126140 A | +0.195% | 0.011 sigma | 0.017 sigma | 20 | 4.63e-11 | 84819.9 A | 3359.31 | success |

VacTH reproduces both sources accurately. Its standalone output is not counted
as final numerical success because its convergence table contains only NICE's
`-9e40` sentinel rather than a nonlinear reconstruction residual. This still
demonstrates that
the COCOS conversion, flux-loop sign, sensor geometry, active-coil mapping,
mesh, and VacTH observation operator can recover a self-consistent noiseless
data set at this current and geometry scale.

The full reconstruction is a **numerical stage success** for both sources: it
produces a finite equilibrium and reaches `1e-10`. Diagnostic-table sign
warnings, magnetic cost, Ip agreement, and equilibrium quality are retained but
do not gate this stage. It does not preserve the imposed plasma current, which
remains a deferred physics-quality issue. Increasing `iterMaxDirInitRecon` to
10 does not change that quality result.

Combined with the measured-data isolation, this identifies two distinct
problems:

- The real 41672/331 ms conditioned diagnostics are not representable by VacTH
  at their stated uncertainties, explaining the failed initialization.
- After nearly exact synthetic VacTH recovery, the nonlinear equilibrium stage
  converges algebraically and therefore passes the current stage gate, but it
  retains only 61--67% of the requested current. This is recorded for the later
  physics-quality assessment rather than treated as a present-stage failure.

Reproduce after preparing the pinned baseline case and executable at the paths
declared in the scripts:

```bash
PYTHONPATH=. python validation/nice_issue_666/synthetic_equilibria/run_solovev.py
PYTHONPATH=. python validation/nice_issue_666/synthetic_equilibria/run_tokamaker.py
```

Machine-readable results are written to
`/tmp/nice331-solovev-20260914/{summary.json,tokamaker_summary.json}`. The
TokaMaker equilibrium is `/tmp/tokamaker-41672-331-20260914/g041672.00331`.

## NICE export-path audit

The pinned NICE revision (`7ad1ea8f`) has no GEQDSK/g-file writer: repository
searches find none of the standard GEQDSK fields (`RMAXIS`, `SIMAG`, `SIBRY`,
`FPOL`, `PSIRZ`, `NBBBS`) in an export implementation. The standalone
`main_recon.cc` path calls `solver.OutputTxt()` and
`solver.OutputEqui2Txt(...)`. A separate IMAS-enabled path calls
`SaveEquiNiceToIds(...)`, which writes an equilibrium IDS directly rather than
a GEQDSK file and is not used by the validated standalone executable.

VAFT already implements GEQDSK -> equilibrium ODS in `vaft.data.eqdsk`, and
that path was exercised by the local TokaMaker g-file used in this study. It is
therefore unnecessary to introduce a lossy synthetic GEQDSK intermediate for
standalone NICE. The adapter consumes NICE's native equilibrium tables
directly; its unstructured nodal psi is interpolated once onto the separable
129 x 129 `profiles_2d.0` R-Z grid required by IMAS/OMAS overview plotting.
