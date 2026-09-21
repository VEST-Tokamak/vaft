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
| 39915 | degraded | 0.306–0.331 | 26 | 0.96 | H3-08, C4-04 | 0.340 | 0.016 |
| 41524 | degraded | 0.315–0.336 | 22 | 0.85 | H3-08, C3-01, C3-05; H3-01, C1-01, C3-02, C4-02, C4-04, C4-06, L-07 | 0.340 | 0.037 |
| 41672 | degraded | 0.312–0.352 | 41 | 0.76 | H3-08, C1-01, C3-01, C3-02, C3-05, C4-02; H3-01, H3-03, H3-06, H3-07, H3-09, C2-04, C3-03, C4-03, C4-04, C4-06, L-07 | 0.360 | 0.026 |

(Scanned 2026-09-18. Probes after the semicolon are condemned by the array
review (#977) -- or, for C4-04, by its recorded fault, which the review
reaches independently.)

Five findings worth carrying into the studies:

- **The packaged pre-EFIT products now carry the diagnostics stage's
  projected validity** for 74 of 75 EFIT-facing channels (all but the missing
  H1-01), since they were regenerated on the current pipeline (5dfcc64a); on
  the 2026-09-07 scan it was 0 of 75. The sweep still assesses and gates into
  a copy rather than trusting what it is handed: a stored projection is the
  verdict of the detectors that ran when the product was made, and the array
  review (#977) postdates these products.
- **H3-08 (`b_field_pol_probe[25]`) is condemned on all three shots**, and not
  marginally: its amplitude is 40 times its family's median and exceeds the
  physical ceiling, with no usable sample anywhere in the record. The
  condemned set grows with shot number — two probes on 39915, ten on 41524,
  seventeen on 41672 — all of them B-probes.
- **Most of that growth is the array review (#977)**, which condemns a probe
  that departs from its two array neighbours by more than the array's own
  amplitude for 30 % or more of its record. The family-amplitude rule cannot
  see these: three outboard probes on 41524 read five times the field their
  array reads yet sit at 3.5–3.9× the family median, and C4-04 is a sign
  fault inside the family. On 41672 the lower inboard H3 set zig-zags
  (+175, +82, +140, +32, −28, +45, −14 mT at one plasma instant) while the
  upper H1 set opposite it falls smoothly; on 39915 the same set is smooth.
  Every condemned probe sits at 0.35 of its record or more, every survivor
  at 0.24 or less (`test_the_array_margin_holds_on_the_reference_shots`).
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
