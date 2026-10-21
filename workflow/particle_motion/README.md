# Particle motion reference experiments

These examples use VAFT's centered Boris full-orbit integration and an
independent static Hamiltonian guiding-center (GC) integration. They require
NumPy, SciPy, Matplotlib, Pillow and, for canonical SVG export, LaTeX/dvisvgm.
They require no external orbit solver or video encoder.

```sh
python -m workflow.particle_motion.run --output docs/assets/particle-motion
python -m workflow.particle_motion.run --check   # regenerate in scratch, compare, write nothing
python -m pytest -q test/test_process_particle.py test/test_particle_animation.py test/test_particle_motion_assets.py
```

Every file is rendered into a scratch directory first, and only files whose
bytes changed are copied into the output. An unchanged regeneration touches
nothing, and files the generator no longer produces are removed. `manifest.json`
records the SHA-256 and size of every other file under `artifacts`.

`--check` (#2117) reports four kinds of problem:
- missing or orphaned files;
- integrity: a file that no longer matches its manifest digest;
- stale data: a sidecar or the manifest's science differs from what the current
  source generates;
- render drift: an SVG, PNG or GIF that differs only after rendering.

Render drift depends on the TeX, Matplotlib and Pillow versions, so it is a
warning that names this machine's toolchain. `--strict` makes it an error. The
first three fail the check. On the reference machine, a regeneration is
byte-identical for all files and takes about 40 s.

The generator writes a deterministic physical-time GIF, quantitative panel image
and provenance sidecar per example, and a static SVG for all but the tokamak
and symmetry-breaking examples (their gyro-resolved SVGs are ~440 KB and no page embeds them;
`static` is null in the manifest). `manifest.json` associates these
artifacts for future gallery ingestion. It also writes synchronized mirror
panels as `mirror-diagnostics.gif`. GIF timing is playback timing; physical
sample values remain in the sidecar. Images are not numerical validation.

Validation is in `test/test_process_particle.py`: exact uniform motion,
charge-sensitive drifts, harmonic mirror period/action, independent tokamak
field-line quadrature, full-orbit/GC ordering scan, a nonadiabatic control,
canonical-momentum conservation and step refinement, divergence/curl audits,
COCOS adapter checks, and domain failure behavior.

Illustrative defaults have q=1 C and m=1 kg with fields and velocities in SI.
They are reference particles, not physical electron/ion mass ratios. Supply
physical charge, mass and a gyro-resolving dt for a species-specific experiment.
The analytic circular toroidal field is not a force-balanced VEST equilibrium.

No issue is automatically closed. #2007's slowly varying field experiment,
external solver comparisons, collisions, transport, and time-dependent fields
remain follow-ups. A varying canonical momentum in the 3D example does not
establish stochasticity, secular transport, orbit loss, or NTV torque.
