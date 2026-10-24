# Native nonlinear CGYRO reference

VEST #39915, 317 ms, magnetics, r/a=0.70, EM (phi + Apar physics).
Only the phi spatial history was written. Saved parallel coverage is theta=0 only.

21 contiguous raw records, t=100–104 a/c_s, extracted from the retained 3DST
run by `workflow.gyrokinetic.capture_reference`. No resampling or solver run.
`manifest.json` pins full-source hashes, fixture hashes, extraction and origin.
`oracle.json` records exact full-array equality with pygacode b493397 and a
5.42e-16 maximum spatial error against its actual FFT reconstruction at t=100.

This short fixture validates parser/plot execution, not nonlinear saturation,
stationarity, locality validity or numerical convergence. Original version/info
metadata are retained from the full run; they do not claim the fixture is the
entire evolution. Never reconstruct a global or full parallel field from it.

Offline: `python -m workflow.gyrokinetic.build_result_reference --out /tmp/gk`.
