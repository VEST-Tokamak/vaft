# Particle motion reference experiments

These examples use VAFT's centered Boris full-orbit integration and an
independent static Hamiltonian guiding-center (GC) integration. They require
NumPy, SciPy, Matplotlib, Pillow and, for canonical SVG export, LaTeX/dvisvgm.
They require no external orbit solver or video encoder.

```sh
python -m workflow.particle_motion.run --output docs/assets/particle-motion
python -m pytest -q test/test_process_particle.py test/test_particle_animation.py
```

The generator writes a static SVG, deterministic physical-time GIF, quantitative
panel image and provenance sidecar per example. `manifest.json` associates these
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
