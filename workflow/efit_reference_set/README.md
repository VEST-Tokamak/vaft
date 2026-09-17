# The EFIT-quality reference set

Issue #171 asks for a curated reference shot set, and #196, #468, #459 and
#579 all inherit whatever it turns out to be. A set is only useful if every
shot is in it for a stated reason and the reason is checked rather than
remembered, so `build_reference_set.py` declares the set and then verifies
every claim about each shot against the packaged files. It runs EFIT on
nothing and changes no product.

```bash
PYTHONPATH=$PWD python workflow/efit_reference_set/build_reference_set.py \
    --table test/data/efit_reference_set.json --markdown /tmp/ref.md
```

Exit status is 1 if a declared file is missing, so a set that has drifted from
the repository fails rather than quietly reporting less.

## The two arms, and why one shot cannot serve both

- **Magnetics / input quality** is what the reconstruction is constrained by.
  Varying it across shots is how a study separates "the fit is poor" from
  "the input was poor". `workflow/magnetics_quality` supplies the verdict.
- **Independent kinetic information** — Thomson scattering, charge exchange,
  fitted profiles — is how a reconstruction's pressure and stored energy can
  be checked without using the magnetics that produced it.

VEST's packaged data does not offer many shots with both. What it does offer:

| shot | arms | magnetics verdict | window [s] | condemned | Thomson | in window | kinetic ODS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 39915 | both | degraded | 0.306–0.331 | 1 | 5 ch, 0.308–0.317 s | 10/10 | – |
| 41524 | magnetics | degraded | 0.315–0.336 | 3 | – | – | – |
| 41672 | magnetics | degraded | 0.312–0.352 | 6 | – | – | – |
| 48224 | kinetic | – | – | – | 7 ch, 0.298–0.307 s | – | yes |
| 46051 | kinetic | – | – | – | 5 ch, 0.300–0.309 s | – | – |

## What each shot is for

**39915 — the dual-arm anchor.** It already carried Thomson data nobody had
put beside its magnetics: `legacy/NeTe_Shot39915_v9_rev.mat`, five
polychromators at R = 0.255 to 0.475 m, ten times from 0.308 to 0.317 s. Its
routine EFIT window is 0.306–0.331 s, so **all ten Thomson samples fall inside
the reconstructed window**. This is the one shot where a reconstruction can be
checked against independent kinetic information at the times it was actually
computed for, and where the stored 2023 reconstruction (#119's reference)
exists to compare against. Its magnetics hold their last value from 0.340 s,
which is outside the window and therefore harmless here — but it bounds how
far a study may widen the window.

**41524 and 41672 — magnetics variation.** Three and six condemned probes
against 39915's one, with later, shorter and longer windows. They are what
makes the input-quality axis a variable rather than a constant. Neither has
packaged kinetic data.

**48224 — the kinetic cross-check.** The packaged kinetic-EFIT sample: seven
Thomson channels, forty charge-exchange channels, core profiles fitted on 129
points from seven measurements, and three stored equilibria (the magnetic
reconstruction, the kinetic-EFIT one and a CHEASE refinement) at 0.300 s.
Two limits are worth stating plainly. It carries **no magnetics**, so its
input quality cannot be assessed offline and its reconstruction cannot be
rebuilt from constraints here — it can only be read. And issue #317 records
that the packaged kinetic ODS is not trustworthy within `psi_N < 0.05`: an
unphysical near-axis `dvolume_dpsi` ramp traceable to a `q[0]` outlier. Any
study using it must exclude the near-axis region or say why it did not.

**46051 — the spare.** Thomson only, five channels, 0.300–0.309 s, with no
equilibrium or magnetics packaged. It is held in the set so that a second
kinetic case exists if 48224's near-axis defect blocks a study.

## What this set cannot do, from this checkout

The remote database serves the domain structure but refuses to read any
dataset for an ordinary account, and the raw SQL database is not reachable.
So the set is what is packaged, and it cannot be widened from here. When
either becomes readable, `scan_magnetics_quality.py --shots` extends the
magnetics arm without change, and a shot gains the kinetic arm as soon as a
Thomson file resolves for it.

Two gaps a future pass should close, in order of value:

1. **39915's Thomson is not in its packaged product.** It maps cleanly from
   the MAT file through `vaft.machine_mapping.thomson_scattering`, so the
   pre-EFIT product could carry `thomson_scattering` and the kinetic check
   would need no side-loading.
2. **48224 has no packaged magnetics.** With them it would become a second
   dual-arm shot, and the kinetic arm would stop being a read-only one.
