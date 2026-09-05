# Plasma-free magnetic-response benchmark (issue #190)

Qualifies the VEST **active-coil / passive-wall / magnetic-sensor forward model**
against measured plasma-free response. It answers a machine-model question, not
a per-shot one:

> Can one physically consistent wall model reproduce the measured plasma-free
> magnetic response across representative shots, PF excitations, sensors and
> machine eras?

This does not replace the routine eddy-stage QA in
`workflow/automatic_pipeline_1_routine_data_processing`. That stage keeps its
compact channel subset and its per-shot verdict; once a model revision is
qualified here, the routine stage can use it.

## What makes it a qualification rather than a fit

- The passive wall is driven by **measured PF currents alone**. The routine
  stage lets plasma filaments drive the same solver, which is right for
  processing a plasma shot and disqualifying here — the plasma current would
  partly explain the response being validated.
- The **solver-input window and the validation window are different**. The
  solver starts from `I_wall = 0`, so the comparison opens only after the
  wall's own slowest L/R time has had time to forget that assumption. The
  margin is computed from the eigenvalues of the wall model in use, not guessed.
- **Nothing is re-fitted to reduce the residual.** The optional
  `--resistance-scale` varies one global factor for the #117 sensitivity study.
  Independently fitting hundreds of loop resistances against the same magnetic
  data used to validate them would make this an underconstrained fit.
- Signal validity comes from the diagnostics stage (#189) and is **read, never
  written**. A channel that disagrees with the forward model is evidence about
  the model.

## Running it

```bash
python workflow/vacuum_benchmark/run_vacuum_benchmark.py \
    --output benchmark.json --packaged-sample
```

The packaged sample needs no database access and is what the regression test
covers. A real campaign passes processed ODSs directly:

```bash
python workflow/vacuum_benchmark/run_vacuum_benchmark.py --output benchmark.json \
    --case 39915=/srv/vest.filedb/.../39915/eddy/omas.json.gz \
    --case 41524=/srv/vest.filedb/.../41524/diagnostics/omas.json.gz
```

Eligible cases are dedicated vacuum shots and plasma shots with a validated
plasma-free interval; `plasma_free_interval` decides which, from the plasma onset of the
shared timing policy (H-alpha by label, else the plasma-current principal pulse; issue #409),
snapped to the `pf_active` grid, and records the choice under `plasma_free_evidence`.
An **eddy-free flat-top is not wanted** — the transient chain from coil
excitation through wall current to magnetic response is the validation target.

The case list is an argument rather than a committed table: the source shots
live in the VEST database, not in this repository. Record the invocation
alongside the output; each case's manifest carries the static-model revision
its residuals were measured against.

## Reading the output

`aggregate` cross-tabulates by case, channel, family, PF excitation and machine
era. Which axis a poor result concentrates on is the diagnosis:

| pattern | suspicion |
| --- | --- |
| one channel, across many excitations | probe calibration / geometry / acquisition |
| many channels, whenever one PF response dominates | coil geometry / current calibration / coupling |
| a similar rise or decay mismatch everywhere | passive-wall resistance / passive-passive coupling |
| quality changes across geometry revisions | static-model provenance |
| one shot among consistent neighbours | acquisition / baseline / timing |

There are deliberately **no acceptance thresholds**. #190 is explicit that broad
scientific bounds must wait until the VEST benchmark distribution has been
inspected.

Each case also lists `channels.flagged`: B-probes that contradict their own array over the
plasma-free window (a lone probe reading a field its two nearest neighbours on the same
radius do not see, for more than 5 % of the window). They are evaluated and reported like
every other channel but kept out of `metrics.summary.scored`, whose `excluded_flagged` names
them: a sensor finding, not a wall-model one. On the packaged 39915 that is C4-04, which
follows the PF1 ramp its neighbours do not see.
