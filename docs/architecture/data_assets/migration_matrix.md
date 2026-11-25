# Consumer and migration matrix

Deliverable C of #2307. One row per asset group of [`inventory.yaml`](inventory.yaml). It covers the groups that move, and the groups that stay but have a decision attached.

- **Consumers** are the files under `.github docs notebooks test tutorial vaft workflow` that match the group's search patterns. `inventory.yaml` lists every file. These are text matches: paths built at run time are missed, and broad patterns over-count. The g-file pattern, for instance, also matches FileDB g-file names.
- **Gate** is what must pass before the next step. Every move ends with #207, which may purge only groups whose consumers no longer read from the checkout.

## Groups that move out of Git

| Group | Now | Target | Consumers (files) | Owner | Compatibility risk | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| `samples.shot.representations` (11 files, 298.7 MB) | Git | Release asset + `references/samples` cache | 179: test 138, docs 14, vaft 10, notebooks 7, tutorial 7, workflow 3 | #2328 publish → #2329 resolve → #2331 consumers | `vaft.data.sample()` must keep returning a local `Path`. about 35 test files `pytest.skip` when a sample is missing, so a broken resolver would show up as skips, not failures. | Every manifest SHA-256 matches the published asset (already true in the tree, see `inventory.yaml` → `samples`). #2331's run with the large files removed passes. |
| `samples.shot.sources` (20 files, 179.2 MB) | Git | Release asset, next to its representations | 58: test 31, workflow 16, vaft 7, docs 3, notebooks 1 | #2328, #2331 | Workflows regenerate samples from `generation.canonical_source`, so they need the source resolvable by manifest path. | The generation command still runs with the source fetched from the Release. |
| `efit.green_tables_lfs` (4 LFS pointers, 165.1 MB objects) | Git LFS, CI restores them from an Actions cache | One baseline Release + `generated/efund` cache | 19: workflow 9, test 6, docs 2, vaft 1, .github 1 | #2330, then #2331 for CI | Three test files (`test_efit_packaged_table.py`, `test_efit_tables_workflow.py`, `test_efund_adapter.py`) skip through `test/lfs_assets.py` when a table is still a pointer. Once LFS is gone, that helper must check the resolver instead. `workflow/cluster_regeneration` hand-fetches LFS oids for hosts without git-lfs. | Baseline SHA-256 equals `table.files`. A regenerated-vs-baseline comparison is reported with a tolerance ([policy §4.3](storage_policy.md#43-substitution-contract)). CI has no LFS step. |
| `efit.generated_tables_plain` (`brzgfc.dat`, `mhdout.dat`, 7.0 MB) | Git | Same baseline Release as the `.ddd` set | 8: workflow 4, test 3, vaft 1 | #2330 | These files are part of the EFUND output set but stored outside LFS. Moving only the `.ddd` files would split the baseline. | Same as the row above. |
| `history.retired_sample_layouts` (history only, 261 MB packed) | History | Purged | none in the tree | #207 | – | Each retired path is confirmed superseded by a registered sample ([policy §7](storage_policy.md#7-unresolved-decisions)). |

## Groups with an open decision

These stay where they are until the decision in [policy §7](storage_policy.md#7-unresolved-decisions) is made.

| Group | Size | Consumers (files) | Owner | Note |
| --- | --- | --- | --- | --- |
| `legacy.raw_fixtures` | 83 files, 193.5 MB | 28: test 11, docs 7, notebooks 6, vaft 2, workflow 2 | #2328 | `41514.h5` (38 MB) and the 39915 camera PNGs have no textual consumer. |
| `samples.shot.stage_products` | 23 files, 13.2 MB | 15 | #2328 | Grew on 2026-11-24 (#2306). |
| `notebooks.outputs` | 61 files, 53.5 MB | not searched | unowned | 279 MB of history. |
| `workflow.results` | 26 files, 20.4 MB | not searched | #2307 follow-up | |

## Groups that stay in Git

| Group | Size | Consumers (files) | Why it stays |
| --- | --- | --- | --- |
| `samples.shot.manifests` | 7 files, 0.1 MB | via the sample API | Identity record the resolver verifies against. |
| `samples.wheel_projection` | 3 files, 20.8 MB | 3 | Packaged offline sample. `setup.py` swaps it in for `samples/39915`. |
| `samples.tables` | 21 files, 8.4 MB | 28 | Tier A / operational-history tables with byte-exact pinned sources. |
| `samples.kinetic_efit` | 6 files, 6.9 MB | 62 | The 48224 historical validation reference pins these g-files. |
| `efit.gfile_fixtures` | 6 files, 14.7 MB | 138 (over-counted) | Small fixtures; `g040330.*` are 4.3 MB each. Revisit if more are added. |
| `efit.inputs`, `data.geometry_config`, `data.small_fixtures`, `notebooks.data`, `test.fixtures` | 0.3 / 4.7 / 3.4 / 11.4 / 13.5 MB | not searched | Source, configuration and lightweight fixtures. |
| `docs.media`, `history.gh_pages` | 51.6 MB tree, 288 MB history | not searched | Owned by #2117. |

## Work assigned to each issue

| Issue | Takes from this audit |
| --- | --- |
| #2317 (I-2) | Cache namespaces and lifecycle ([policy §5](storage_policy.md#5-local-storage-lifecycle)); the MCP artifact root is an access boundary, not storage. |
| #2328 (I-3) | Publish the representations and sources of the 7 registered samples; extend `manifest.yaml` (39915 lacks `source.kind`); decide the asset format (xz vs gzip), the `legacy.raw_fixtures` split, the stage products, and how `source.filedb_root` appears in the public catalog. |
| #2329 (I-4) | Resolver behind the unchanged sample API; checkout hit first, so existing clones keep working during migration. |
| #2330 (I-5) | Baseline = all six files of `table.files`, identity `e28142c7…`; an identity function that hashes only what EFUND reads (the shipped `mhdin.dat` hash differs on purpose); remove both `.gitattributes` LFS rules (`vaft/data/efit_table/**` names a directory that does not exist). |
| #2331 (I-6) | The about 35 test files that skip on missing samples, `test/lfs_assets.py`, the LFS cache in `package-ci.yml`, and `filter: blob:none` for the three `fetch-depth: 0` checkouts (`docs.yml` ×2, `bootstrap-ci.yml`). |
| #207 | Purge `samples.shot.representations`, `samples.shot.sources`, `history.retired_sample_layouts` and the LFS pointers after #2331's gate; estimated ~2.1 GB of the ~3.0 GB clone; add the 5 MB size guard ([policy §2](storage_policy.md#size-rule-for-new-files)). |
