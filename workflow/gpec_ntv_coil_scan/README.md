# NTV torque over coil-group phasors (lane N, #1887)

This directory holds the direct GPEC → PENTRC scan that tests whether the signed NTV torque on a fixed plasma is a Hermitian quadratic form of the coil-group phasors, `T(c) = cᴴQc`. The algebra (probes, `Q`, qualification, optima) is solver-free in `vaft.process.ntv_coil_response`. Results land in the #1178 operating-space result (`vaft.process.coil_operating_space`) through `ntv_operating_space`.

| script | issue | what it does |
|---|---|---|
| `scan_ntv_coil_torque.py` | #1887 | One cell per candidate: DCON, GPEC with that candidate's `coil.in`, then `vaft.code.gpec.run_pentrc` and the canonical reader. The design is the K² probes, a reference with 0.5×/2×/−1×/rotated copies, seeded held-out points, and optionally the model's optima (`--design confirm --from …`). Resumable; `euler.bin` is shared by hard link, GPEC's `*_fun_n*.out` field dumps are deleted, and a stalled PENTRC log is truncated after the run ends (it still grows until the timeout, #1929). A cell is reused only when it completed under the same case, coil frame, unit and tolerance; failed cells are rerun. Writes `scan_<tolerance>.json`; `--design confirm --from scan_<tolerance>.json` refuses a `Q` that does not qualify and writes `confirm_<tolerance>.json`. |

Coil currents follow the #1178 convention: `I_k = A cos(n φ_k + δ)` with `c = A e^{+iδ}`. Here `φ_k` is each coil's toroidal centroid in its `.dat` file's frame, and one phasor unit is `--unit` A per turn.

The study's converged result is committed as `test/data/gpec/ntv_diiid_like_1887/scan.json`. `test/test_ntv_coil_reference_1887.py` replays it without a solver.

Decisions the scan encodes:
- **Reference case.** The default case is upstream's `docs/examples/DIIID_ideal_example`. It is an *artificial* TokaMaker DIII-D-like H-mode, not DIII-D #147131 (which is not in GPEC e68d7ac2). Its kinetic file carries ω_E, not ω_tor.
- **PENTRC template.** VAFT's packaged `pentrc.in` is used, because the example's lacks keys the adapter writes. PENTRC settings: ions only, D and C, `harmonic` collisions, `nl = 4`, methods fgar/tgar/pgar.
- **Tolerances decide the answer.** With the example's defaults (`atol_psi = rtol_psi = 1e-4`), small and cancelling torques are off by several percent, and the quadratic model is wrongly rejected (#1887). `--tolerance converged` (`1e-6`, `rtol_xlmda = 1e-7`) is the setting that qualified it. Do not lower `atol_xlmda` below about 1e-10: PENTRC derives its energy tolerance from it, and the energy LSODE then stalls.
- **Scalar.** The scalar is the signed enclosed torque at the last dynamic-grid point. The full radial profile stays in each cell's `pentrc_output_n1.nc`.
