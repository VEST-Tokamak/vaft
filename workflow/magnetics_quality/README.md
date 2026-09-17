# Is this shot's magnetics fit to reconstruct from?

Every open EFIT study reconstructs from the same magnetics — cadence (#468),
termination (#171), initialization (#196), domain and grid (#459),
profile-model uncertainty (#579) — and none of them establishes that the input
was sound. The pieces to establish it exist and none answers alone: the
quality layer (#189) reports per-channel metrics without judging a shot, the
flux-loop back-test (#295) judges flux loops only, and the vacuum benchmark
(#190) asks a machine-model question. `scan_magnetics_quality.py` composes
them into one per-shot verdict over the window the routine pipeline would
actually reconstruct.

```bash
PYTHONPATH=$PWD python workflow/magnetics_quality/scan_magnetics_quality.py \
    --packaged-samples --table test/data/magnetics_quality.json --markdown /tmp/mq.md
PYTHONPATH=$PWD python workflow/magnetics_quality/scan_magnetics_quality.py \
    --shots 41000-41100 --source main --table /tmp/mq.json
```

The table it writes is what `test/test_magnetics_quality_corpus.py` pins.
Shots the source does not carry are recorded as `absent` and shots whose
assessment raises are recorded as `error`: which shots could not be judged is
part of the answer, not a gap in it.

## What decides the verdict, and what does not

`FitnessPolicy` holds every threshold in one frozen dataclass, echoed into the
table as provenance. They are **coverage floors, not signal thresholds** — the
detectors and their thresholds live in `MagneticsQualityConfig` and are
reported beside these, never restated here.

| verdict | means |
| --- | --- |
| `fit` | every family keeps its witness floor at every slice, no channel rejected or missing, and the measured record covers the window |
| `degraded` | a channel is rejected or missing, or the record ends inside the window, while coverage still holds |
| `unfit` | a family falls below its floor at some slice, the usable fraction falls below `min_usable_fraction`, or the record misses more than a tenth of the window |

The witness floor is a **minimum over slices**: a family that loses members
part-way through the window is as unusable for those slices as one that never
had them. Flux-loop families are exempt because VEST has too few loops to
spare one.

Vacuum-model disagreement is reported beside the verdict and never inside it.
Issue #190 is explicit that scientific acceptance bounds wait for the VEST
distribution, so a residual is evidence to read, not a threshold to fail.

## What it found on the three packaged shots

All three are **degraded**, none is unfit, and the reasons differ:

| shot | verdict | window [s] | slices | usable frac (min) | condemned probes | measured to [s] | model residual (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 39915 | degraded | 0.306–0.331 | 26 | 0.97 | H3-08 | 0.340 | 0.016 |
| 41524 | degraded | 0.315–0.336 | 22 | 0.95 | H3-08, C3-01, C3-05 | 0.340 | 0.037 |
| 41672 | degraded | 0.312–0.352 | 41 | 0.91 | H3-08, C1-01, C3-01, C3-02, C3-05, C4-02 | 0.360 | 0.026 |

Five findings worth carrying into the studies:

- **No EFIT-facing channel carries a stored validity in the packaged pre-EFIT
  products.** 0 of 75; the only 8 channels with a record are the IMPA probes,
  which have no waveform at all. "All valid" and "never looked at" are the
  same bytes in a product, so this sweep assesses and gates into a copy rather
  than trusting what it is handed. A constraint build that reads projected
  validity from these products would find nothing and fit everything — which
  is exactly why the constraint stage now refuses an unassessed product.
- **H3-08 (`b_field_pol_probe[25]`) is condemned on all three shots**, and not
  marginally: its amplitude is 40 times its family's median and exceeds the
  physical ceiling, with no usable sample anywhere in the record. The
  condemned set grows with shot number — one probe on 39915, three on 41524,
  six on 41672 — all of them inboard or C-array.
- **One EFIT-facing channel is missing on every shot**, `b_field_pol_probe[0]`
  (H1-01). It is a deterministic placeholder at weight zero, not a silent gap.
- **The routine window ends before the record does on all three shots.**
  39915's magnetics hold their last value from 0.340 s, but its window ends at
  0.331 s, so the hold does not touch the reconstructed slices. That is worth
  stating precisely, because the hold is real and a study that widened the
  window past 0.340 s would be fitting held values.
- **The flux loops agree with the vacuum model on all three shots**: 11 of 11
  usable, normalized residual 1.6–3.7 % median and 7.6 % at worst. Wall
  authority is low (0.05–0.06 median), so the loops are only weakly sensitive
  to wall currents in these windows — which is a caveat on the model layer,
  not on the loops.

The last two together are the input-side answer to #295: nothing in the
evidence rejects a flux loop on any packaged shot, and the automatic layer
instead rejects **probes** the manual list never named.

## A caveat this sweep had to work around

The packaged pre-EFIT products carry an `em_coupling.mutual_passive_passive`
that predates the repair of #347/#373 and is asymmetric by 1.27e-3, which the
wall-mode basis refuses as ill-posed — so the vacuum model cannot be consulted
on them as they stand. The sweep re-maps `em_coupling` from the canonical
repaired asset before consulting the model, which is what the pipeline would
carry today, and says so in every row. The magnetics are untouched by this:
it decides only whether the model layer can speak, never whether a channel is
usable.

## Where this sits

Read with `workflow/efit_channel_selection/README.md` (#295, the flux-loop
evidence and the retired manual list) and `workflow/vacuum_benchmark/`
(#190, the machine-model qualification). This directory asks the third
question those two leave open: taken together, is the shot's magnetics fit to
reconstruct from?
