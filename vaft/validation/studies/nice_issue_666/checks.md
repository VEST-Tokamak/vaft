# Verification

- Pinned NICE upstream CTest `test_recon`: passed, 3.01 s, using
  `/tmp/nice-build-clang2` (2026-09-08).
- Adapter, EFIT missing-channel/broken-probe, namespace, import, and sample
  checks: 58 passed in 117.70 s (2026-09-08).
- Final NICE regression file after additional conversion/identity/response
  checks: 20 passed in 8.77 s (2026-09-09).
- Ruff, Black check, and `git diff --check`: passed (2026-09-09).
- Offline wheel build (`pip wheel . --no-deps --no-build-isolation`): passed.
  Wheel SHA256: `793d272ce4589b3cd98f7ad96d83c1fe8e1485914951754cb834696ab0a37de2`.
  Verified inclusion of the VEST parameter XML, compatibility header, and
  reproducible study module.
- CHEASE's separate existing ZMAXIS tolerance assertion still fails; see
  `chease_separate_failure.md`. No assertion tolerance was changed.

Scientific verification is separate from these software checks: 0/34 window
slices accepted, and no accepted 41672@331 ms NICE or exact-time EFIT result.
