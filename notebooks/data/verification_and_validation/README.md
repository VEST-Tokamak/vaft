# Verification and validation Atlas snapshot

This directory contains the Tier A Atlas state, equilibrium-quality, and confinement tables
copied from `user1@147.46.36.244:/home/user1/runs/campaign/atlas` on 2026-10-08.
The state Atlas was generated on 2026-10-02. `snapshot_manifest.json` records
the source paths, producer revisions, row counts, and SHA-256 of every copied
table. The original Atlas manifests and schemas are retained in their original
relative locations. No legacy `public` namespace data or XLSX summary is used.

`v1/virial_identity.csv` was evaluated for all 150 state keys from the 49
selected FileDB EFIT products. The export checks each product SHA-256 against
`state.csv`, matches the requested equilibrium time within 0.6 ms, and calls
`vaft.omas.process_wrapper.compute_virial_equilibrium_quantities_ods`. It
stores the normalized E1/E2/E3 residuals and their RMS, along with the
volume-integral and pair-13 virial estimates of poloidal beta and internal
inductance from the same slice. The identity residuals are separate from the
pair-13 `virial_status` decision; neither plot introduces a new acceptance
rule. The FileDB products
themselves are not included.

To inspect newer results, set `VAFT_ATLAS_DIR` to the root containing `v1/`
and `equilibrium_quality/`. If that Atlas lacks `v1/virial_identity.csv`, set
`VAFT_CAMPAIGN_FILEDB` to its matching FileDB; the notebook computes the
residuals in memory. The same FileDB variable enables detailed representative
EFIT views. Do not mix the snapshot's residual CSV with a newer Atlas: the
notebook verifies state keys and product hashes before plotting.
