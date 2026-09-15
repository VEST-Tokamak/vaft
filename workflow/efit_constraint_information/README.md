# EFIT constraint-information study (#663)

This workflow keeps the qualified issue-#579 `(2,2)` zero-edge equilibrium,
seed, numerical controls, grid, inputs, and cadence fixed. It changes only
which EFIT constraint families participate and their submitted objective
strength.

Run the complete matrix on the primary discharge:

```bash
env -u EFIT PYTHONPATH=$PWD EFITHOME=/path/to/efit-install \
  python \
  workflow/efit_constraint_information/constraint_information_study.py \
  --output /tmp/vaft-efit-constraint-primary --shots 41672 --matrix complete
```

Run the baseline and ablations on the confirmation discharges:

```bash
env -u EFIT PYTHONPATH=$PWD EFITHOME=/path/to/efit-install \
  python \
  workflow/efit_constraint_information/constraint_information_study.py \
  --output /tmp/vaft-efit-constraint-confirmation \
  --shots 39915,41524 --matrix confirmation
```

Each output directory contains `constraint_information.json`,
`constraint_information.md`, per-variant raw EFIT products, and six PNG
figures. Every requested slice remains in the JSON, including collapsed and
missing-output outcomes. Physics comparisons use common produced plasma
slices; outcome-population changes are reported separately.

These numerical products belong in the chosen output directory and are not
version-controlled with the workflow.

The complete strength scan extends diamagnetic flux through ×10 and ×100,
then linearly from ×1,000 through ×10,000 in ×1,000 increments. This locates
the onset of outcome loss and distinguishes a merely overwhelmed constraint
from one that remains effectively inactive over four orders of magnitude.
