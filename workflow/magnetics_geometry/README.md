# Magnetics wiring history (issue #956)

`vaft/machine_mapping/vest.yaml` records, under
`diagnostics.equilibrium_magnetics.processing`, two per-shot facts about the
equilibrium probes that the packaged geometry files cannot:

- `wiring` -- which raw field and calibration sign feed each probe position.
  Through shot 39437 (2023-05-29) the outboard probes at Z = +0.06 and
  Z = -0.42 read fields 170 and 225 (negative); from 39438 (2023-05-30) the
  reverse, which is the 2409 layout `MD.yaml` carries.
- `known_faults` -- positions whose acquisition was broken. The Z = +0.06
  probe (field 170) carried a DC offset and a gain unrelated to its
  neighbours up to 36480, and sat at the DAQ rail for 36822-36905; the
  diagnostics stage marks it invalid there, so EFIT drops it.

`verify_wiring.py` re-derives both from the raw archive, read-only:

```
PYTHONPATH=. python workflow/magnetics_geometry/verify_wiring.py --range 39430 39450
```

For each shot it scores every assignment of fields 170/225 (both signs) to
the two positions against the linear prediction of their untouched
neighbours in the R = 0.796 m column, reports the winner and its margin over
the runner-up, and checks it against the layout vest.yaml resolves; it also
reports the Z = +0.06 channel's mean voltage, gain and residual against the
recorded faults. It exits 1 when a confident winner disagrees with vest.yaml.

What the table does **not** encode, and why:

- VFIT's `shot == 39204 -> ver_2310` override. 39204 fits the pre-39438
  layout 2.5x better than 2310, like every shot around it.
- The other rows in which VFIT's 2302/2305 files differ from 2409 (field 197
  sign; fields 179/180 calibration). Field 197 keeps the 2409 sign on
  pre-39438 shots (gain +1.08 against its neighbours on 37000); the
  solenoid's near field makes the inboard neighbour test too poor to decide
  179/180.
- Individual-shot glitches after the recorded windows (e.g. a +0.06 gain of
  0.7 or 1.3 on a few 2022-09 shots). Those belong to the per-shot
  validation layer, not to a machine-history table.
