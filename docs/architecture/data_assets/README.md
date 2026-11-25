# VAFT data-asset audit (#2307)

This folder holds the evidence and the policy that the data-distribution work under #2304 builds on.

| File | Contents | #2307 deliverable |
| --- | --- | --- |
| [`inventory.yaml`](inventory.yaml) | Asset groups with their tree size, LFS and wheel membership, consumers and proposed placement; one row per registered reference sample, checked against its manifest | A |
| [`storage_policy.md`](storage_policy.md) | Categories, placement and retention rules, the EFUND baseline contract, conflicts with earlier decisions | B |
| [`migration_matrix.md`](migration_matrix.md) | Each asset group: where it is, where it goes, who depends on it, which issue moves it, and what must pass first | C |
| [`measurements/2026-11-25.json`](measurements/2026-11-25.json) | Dated tree, history, LFS, package and GitHub sizes | D |

Everything here is produced by [`workflow/data_asset_audit/audit_assets.py`](../../../workflow/data_asset_audit/audit_assets.py). The script only runs Git plumbing against a ref: it does not touch the working tree, so it also works in a sparse checkout. It does not import `vaft`.

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

`inventory --check` is a review aid, not a CI gate. The consumer lists change whenever a test starts or stops naming a sample, so a CI gate would fail unrelated pull requests.

## Baseline measurement (2026-11-25)

Refs: `origin/develop` f878d31ae, `origin/main` 548d12d2e. History scope: the 29 origin branches and tags a full clone downloads.

These four numbers measure different things. Do not compare them with each other:

| Metric | Value | Method |
| --- | --- | --- |
| Current tree, develop | **941 MB**, 3,497 files | `git ls-tree -r -l` (LFS files count as pointers) |
| Current tree, main | 907 MB, 3,314 files | same |
| History, clone scope | **3.01 GB packed** (14.58 GB raw, 75,769 blobs) | `rev-list --objects` + `cat-file --batch-check`, local packing |
| GitHub repository size | 3.10 GB (3,102,296 KiB) | `gh api repos/VEST-Tokamak/vaft .size` |
| Git LFS objects (4 EFIT tables) | 165.1 MB | sizes recorded in the LFS pointers |
| PyPI 0.9.0 wheel / sdist | 18.1 MB / 28.0 MB | PyPI JSON API |

### Where the clone's bytes are

These are the history sizes by asset group, packed. "Raw" is the uncompressed size of every stored version.

| Group | Blobs | Packed | Raw | Share of packed |
| --- | ---: | ---: | ---: | ---: |
| `samples.shot.sources` (pre-EFIT source products, DAQ raw) | 43 | 977 MB | 1,003 MB | 32 % |
| `samples.shot.representations` (`omas.json.gz`, `imas.nc`) | 88 | 899 MB | 1,920 MB | 30 % |
| `notebooks.outputs` | 849 | 279 MB | 1,006 MB | 9 % |
| `history.retired_sample_layouts` (no longer in the tree) | 46 | 261 MB | 939 MB | 9 % |
| `history.gh_pages` (generated site) | 49,200 | 251 MB | 7,964 MB | 8 % |
| `legacy.raw_fixtures` | 90 | 120 MB | 218 MB | 4 % |
| `samples.wheel_projection` | 28 | 62 MB | 189 MB | 2 % |
| everything else | 25,425 | 162 MB | 1,343 MB | 5 % |

Reference samples, counting all four sample groups, take **2.2 GB of the 3.0 GB** a clone downloads. Compressed sample files barely delta against their earlier versions: the source products pack at 97 % of their raw size. So every regeneration adds roughly the full file size to history again.

### Current tree

| Group | Files | Bytes |
| --- | ---: | ---: |
| `samples.shot.representations` | 11 | 298.7 MB |
| `legacy.raw_fixtures` | 83 | 193.5 MB |
| `samples.shot.sources` | 20 | 179.2 MB |
| `notebooks.outputs` | 61 | 53.5 MB |
| `docs.media` | 566 | 51.6 MB |
| `samples.wheel_projection` | 3 | 20.8 MB |
| `workflow.results` | 26 | 20.4 MB |
| `efit.gfile_fixtures` | 6 | 14.7 MB |
| `test.fixtures` | 308 | 13.5 MB |
| `samples.shot.stage_products` | 23 | 13.2 MB |
| other groups | 160 | 42.2 MB |
| source code and other small unclaimed files | 2,230 | 39.4 MB |

### What a migration could remove (estimate)

This is an estimate from the packed sizes above, not a measured rewrite. Repacking after a purge changes the deltas.

| Step | Removes (packed) | Clone after |
| --- | ---: | ---: |
| Now | | ~3.0 GB |
| #2328 + #207: sample representations, sources and retired layouts | ~2.14 GB | ~0.9 GB |
| + #2117: single-commit gh-pages | ~0.25 GB | ~0.6 GB |
| + decision on `legacy.raw_fixtures` | up to ~0.12 GB | ~0.5 GB |

Notebook outputs (0.28 GB) remain in the estimate because no issue owns them yet (see the [policy](storage_policy.md#7-unresolved-decisions)).

### Available now, without a history rewrite

- **Partial clone.** `git clone --filter=blob:none` downloads blobs only for the checked-out commit. The develop tree's blobs take 631 MB packed locally (941 MB raw). An actual partial clone was not timed.
- **CI.** Three jobs clone the full history with `fetch-depth: 0`: `docs.yml` (two jobs) and `bootstrap-ci.yml`. They need commit history and refs, not old blobs. `actions/checkout`'s `filter: blob:none` would cut each of these from ~3 GB to roughly one tree. This belongs to #2331; check it against `docs/build.py`, which reads `origin/main` and `origin/develop` itself.
