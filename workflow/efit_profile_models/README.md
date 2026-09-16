# EFIT profile-model uncertainty (#579)

Numerical results are written to the requested output directory and are not
version-controlled with the workflow.

This workflow varies EFIT's fitted `P'` and `FF'` representation while holding
the numerical, spatial, temporal, diagnostic, and initialization choices fixed.
It is an uncertainty study, not a search for the lowest chi-square model.

The baseline is explicit:

- `ellipse_rzero = 0.32 m`, independently of `RZERO = RCENTR = 0.4 m`;
- `ICINIT = 2`, so every slice uses the same independent ellipse seed;
- `ERRMIN = 1e-2`, `SAICON = 80`, `ICONVR = 2`, `NXITER = 1`;
- 129 x 129 packaged response tables with canonical VEST geometry and the VEST
  acceptance envelope;
- 1 ms reconstruction cadence and a 0.5 ms constraint-averaging window;
- the same upstream channel-quality decisions and constraint values for every
  model.

Run the five-model pilot over the reference set:

```bash
cd <checkout> && EFITHOME=~/git/efit/vaft-install python - <<'EOF'
import os, runpy, sys
root = os.getcwd()
sys.path.insert(0, root)
import vaft; assert vaft.__file__.startswith(root), vaft.__file__
sys.argv = ["profile_model_study.py", "--output", "/scratch/efit-profile-models"]
runpy.run_path("workflow/efit_profile_models/profile_model_study.py", run_name="__main__")
EOF
```

Running the script by path instead would put its own directory at `sys.path[0]`,
so an editable install elsewhere — the main checkout, when this is a worktree —
would supply `vaft` and the study would silently measure the wrong tree.
`PYTHONPATH` does not fix that, which is why the launcher asserts on
`vaft.__file__`. Pass `--shots` to narrow; the default is every reference-set
discharge with a packaged product.

Add the `(2,3)`, `(3,2)`, and `(3,3)` order cases with `--full-matrix`. Every
requested time remains in the output, including collapsed and missing-output
slices. Quantitative profile and geometry differences use paired times where
both the candidate and `(2,2)` baseline produced an equilibrium.

The raw EFIT files remain beneath `shot_<shot>/<model>/`. The JSON report keeps
the resolved configuration and digest, outcome of every requested time, a-file
metrics, diagnostic-resolved m-file fit measures, LCFS, and 1-D profiles.
Re-running the command resumes completed model runs whose scientific digest and
analysis schema match.

The reported reference-R current profile is reconstructed from the g-file as
`j_phi(R_axis, psi) = R_axis p'(psi) + FF'(psi) / (mu_0 R_axis)`. It is a
consistent decomposition diagnostic, not a flux-surface average.

Interpretation limits are explicit:

- absolute pressure and `beta_p` are not qualified while #386/#659 remain open;
- q comparisons exclude approximately `psi_N < 0.05` because of #317;
- a model that produces fewer equilibria is not compared on its easier subset
  without also reporting the changed outcome population;
- the fit comparison uses EFIT's flux-loop plus magnetic-probe chi-square from
  the m-file; the reported total is dominated by the model-invariant plasma-
  current term and is retained only as a diagnostic;
- a smaller magnetic chi-square is not evidence of a better model when
  geometry, condition number, or temporal jitter deteriorates;
- `unchanged` in the signed comparison means the two models agreed to within
  `1e-6` relative, which is the files' own precision and not a physical
  tolerance. A half-percent change counts as a change, and how big a change is
  gets read off the signed median rather than off the counts;
- the model-induced spread is reported over the times **every** model produced
  an equilibrium. A sigma over whatever happened to reconstruct at each slice
  is a sigma over a different ensemble at every slice, which is the selection
  effect that made `NXITER = 3` look like an improvement in #171. The
  all-available figures are kept in the JSON and are not the headline, and the
  model that empties the common population is named rather than dropped;
- the chi-square is not reduced by the fitted coefficient count. EFIT does not
  report how many free parameters it used, and
  `magnetics_chisq_per_active_signal` divides by the number of active signals,
  not by degrees of freedom.
