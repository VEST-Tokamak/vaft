---
layout: post
title: VAFT data-asset audit
permalink: /architecture/data_assets/
---

# VAFT data-asset audit (#2307)

This folder holds the evidence and the policy that the data-distribution work under #2304 builds on.

| File | Contents | #2307 deliverable |
| --- | --- | --- |
| [`inventory.yaml`](inventory.yaml) | Asset groups with tree size, LFS and wheel membership, consumers and proposed placement. One row per registered reference sample, with every SHA-256 its manifest pins checked against the stored file. | A |
| [`storage_policy.md`](storage_policy.md) | Categories, placement and retention rules, the EFUND baseline contract, historical references, conflicts with earlier decisions. | B |
| [`migration_matrix.md`](migration_matrix.md) | Each asset group: where it is, where it goes, who depends on it, which issue moves it, and what must pass first. | C |
| [`measurements/2026-11-25.json`](measurements/2026-11-25.json) | Dated tree, history, LFS, sample-version, package and GitHub sizes. | D |

Everything here is produced by [`workflow/data_asset_audit/audit_assets.py`](https://github.com/VEST-Tokamak/vaft/blob/develop/workflow/data_asset_audit/audit_assets.py). The script only runs Git plumbing against a ref and never touches the working tree, so it also works in a sparse checkout. It does not import `vaft`.

## Rerunning

```bash
# Deterministic inventory of a tree (default HEAD); --check exits 1 on drift.
python workflow/data_asset_audit/audit_assets.py inventory
python workflow/data_asset_audit/audit_assets.py inventory --check

# Dated measurement. Fetch first so the history scope equals what a clone gets.
git fetch --prune --tags origin
python workflow/data_asset_audit/audit_assets.py measure \
    --ref origin/develop --ref origin/main \
    --output docs/architecture/data_assets/measurements/$(date +%F).json
```

- **Reads the committed ref.** The script reads the ref, not the working tree. After changing it or its groups, commit first, then regenerate and commit the inventory.
- **Ignores its own files.** The audit leaves its own files (this folder, the script, its test) out of every count, so a regenerated inventory does not depend on itself.
- **Not a CI gate.** `inventory --check` is a review aid. The consumer lists change whenever a test starts or stops naming a sample, so a CI gate would fail unrelated pull requests.

## Baseline measurement (2026-11-25)

Refs: `origin/develop` f878d31ae, `origin/main` 548d12d2e. History scope: the 30 origin branches and tags a full clone downloads.

These numbers measure different things. Do not compare them with each other:

| Metric | Value | Method |
| --- | --- | --- |
| Current tree, develop | **941 MB**, 3,497 files (629 MB packed locally) | `git ls-tree -r -l`, LFS files count as pointers; packed = `objectsize:disk` of the tree's blobs |
| Current tree, main | 907 MB, 3,314 files (603 MB packed) | same |
| History, clone scope | **3.01 GB packed** (14.58 GB raw, 75,801 blobs) | `rev-list --objects` + `cat-file --batch-check`, local packing |
| GitHub repository size | 3.18 GB (3,103,164 KiB = 2.96 GiB) | `gh api repos/VEST-Tokamak/vaft .size` |
| Git LFS objects, current tree | 165.1 MB (4 EFIT tables) | sizes recorded in the pointers |
| Git LFS objects, all history | 482.5 MB (11 objects, 3 table generations); 317.4 MB in the 7 not in today's trees | every pointer in history |
| PyPI 0.9.0 wheel / sdist | 18.1 MB / 28.0 MB | PyPI JSON API |

### Where the clone's bytes are

These are the history sizes by asset group, packed. "Raw" is the uncompressed size of every stored version.

| Group | Blobs | Packed | Raw | Share of packed |
| --- | ---: | ---: | ---: | ---: |
| `samples.shot.sources` (pre-EFIT source products, DAQ raw) | 43 | 977 MB | 1,003 MB | 32 % |
| `samples.shot.representations` (`omas.json.gz`, `imas.nc`) | 88 | 899 MB | 1,920 MB | 30 % |
| `notebooks.outputs` | 849 | 279 MB | 1,006 MB | 9 % |
| `history.retired_layouts` (no longer in the tree) | 46 | 261 MB | 939 MB | 9 % |
| `history.gh_pages_backups` (only on `backup/gh-pages-20261021` and tag `archive/gh-pages-develop`) | 53,336 | 245 MB | 8,137 MB | 8 % |
| `legacy.raw_fixtures` | 90 | 120 MB | 218 MB | 4 % |
| `samples.wheel_projection` | 28 | 62 MB | 189 MB | 2 % |
| everything else | 21,321 | 169 MB | 1,171 MB | 6 % |

Reference samples take **2.2 GB of the 3.0 GB** a clone downloads. That counts every sample group plus the retired layouts. Compressed sample files barely delta against their earlier versions: the source products pack at 97 % of their raw size. So every regeneration adds roughly the full file size to history again.

The current `origin/gh-pages` is already a single commit. Only 5 MB of it is not shared with a code branch. The 245 MB lives on a backup branch and an archive tag.

### Current tree (develop)

| Group | Files | Bytes |
| --- | ---: | ---: |
| `samples.shot.representations` | 11 | 298.7 MB |
| `legacy.raw_fixtures` | 83 | 193.5 MB |
| `samples.shot.sources` | 20 | 179.2 MB |
| `notebooks.outputs` | 61 | 53.5 MB |
| `docs.media` | 566 | 51.6 MB |
| `samples.wheel_projection` | 3 | 20.8 MB |
| `workflow.results` | 24 | 20.4 MB |
| `efit.gfile_fixtures` | 6 | 14.7 MB |
| `test.fixtures` | 308 | 13.5 MB |
| `samples.shot.stage_products` | 15 | 12.5 MB |
| other groups | 170 | 43.0 MB |
| source code and other small unclaimed files | 2,230 | 39.4 MB |

### History that has no retention rule today

- **Reference samples.** 24 sample `reference_id`s have been registered over time: 7 current, 17 superseded. This counts repository and wheel manifests, including the retired `packaging/wheel_samples`. Their manifests record 120 distinct representation byte versions, and all 120 are still somewhere in history. Some survive only under a retired path such as `packaging/wheel_samples/39915`. History is today the only copy of the superseded ones.
- **`reference_id` reused for changed bytes.** Ten `reference_id`s name more than one version of a representation, four of them current:
  - `vest-39915-until-efit-v1` to `v5`. From v1 to today, the compact wheel copy (`wheel_samples/39915`) has carried the same `reference_id` as the repository sample but different bytes. A wheel install and a repository checkout therefore deliver different data under one name (`inventory.yaml` → `reference_ids_shared_by_different_bytes`).
  - `vest-41524/41672-pipeline-full-imas-v2` and `-v6`, and `vest-48224-kinetic-300ms-v2`: bytes changed without a new id.

  A `reference_id` alone has therefore not identified bytes.
- **EFIT Green tables.** Three generations are in LFS history: the 2026-07 packaged table (be3be688b), the #695 regeneration (4d22d4484) and the current #708 all-circuits table (e2f73cea8). `ep129129.ddd` has not changed since #695, so it belongs to the current baseline. The other 7 objects, 317.4 MB, exist only in history. They are in this machine's LFS store; whether the LFS server still has them was not checked.
- **Retired layouts.** `history.retired_layouts` (261 MB) holds files for shots with no registered sample, such as `vaft/data/39916.json`, `39917.json` and `44740_imas_equilibrium.h5`.

The [policy](storage_policy.md#4-historical-references) treats all of these as historical references: each is archived before #207 may purge it, or retired by a recorded decision.

### What a migration could remove (estimate)

This is an estimate from the packed sizes above, not a measured rewrite. Repacking after a purge changes the deltas. Every step requires the archive step above first.

| Step | Removes (packed) | Clone after |
| --- | ---: | ---: |
| Now | | ~3.0 GB |
| #2328 + #207: sample representations, sources and retired layouts (archived to Releases first) | ~2.14 GB | ~0.87 GB |
| + #2117 retention decision: delete `backup/gh-pages-20261021` and `archive/gh-pages-develop` | ~0.24 GB | ~0.63 GB |
| + decision on `legacy.raw_fixtures` | up to ~0.12 GB | ~0.51 GB |

Notebook outputs (0.28 GB) stay in the estimate because no issue owns them yet (see the [policy](storage_policy.md#8-unresolved-decisions)).

### Available now, without a history rewrite

- **Partial clone.** `git clone --filter=blob:none` downloads blobs only for the checked-out commit. The develop tree's blobs take 629 MB packed locally (941 MB raw). An actual partial clone was not timed. The audit runs git with lazy fetching disabled, so in such a clone it counts absent blobs as missing rather than downloading them, and its sizes are then incomplete.
- **CI.** Three jobs clone the full history with `fetch-depth: 0`: `docs.yml` (two jobs) and `bootstrap-ci.yml`. They need commit history and refs, not old blobs. `actions/checkout`'s `filter: blob:none` would cut each of these from ~3 GB to the trees they actually read: `docs/build.py` archives both `origin/main` and `origin/develop`, so about two trees for the docs jobs. This belongs to #2331; check it against `docs/build.py`, which reads `origin/main` and `origin/develop` itself.
