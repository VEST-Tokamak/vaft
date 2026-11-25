# Consumer and migration matrix

Deliverable C of #2307. One row per asset group of [`inventory.yaml`](inventory.yaml), plus the historical items of [policy §4](storage_policy.md#4-historical-references).

- **Consumers** are the files under `.github docs notebooks test tutorial vaft workflow` that match the group's search patterns. The audit's own files are excluded, and `inventory.yaml` lists every file. These are text matches: paths built at run time are missed, and broad patterns over-count. The g-file pattern, for instance, also matches FileDB g-file names. Per-shot consumer lists are in `inventory.yaml` → `samples[].consumers`.
- **Gate** is what must pass before the next step. Every purge ends with #207, which may remove only items whose consumers no longer read from the checkout and which have been archived or retired ([policy §4.2](storage_policy.md#42-rules)).

## Groups that move out of Git

| Group | Now | Target | Consumers (files) | Owner | Compatibility risk | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| `samples.shot.representations` (11 files, 298.7 MB) | Git | Release asset + `references/samples` cache | 210: test 162, docs 14, notebooks 14, vaft 10, tutorial 7, workflow 3 | #2328 publish → #2329 resolve → #2331 consumers | `vaft.data.sample()` must keep returning a local `Path`. About 35 test files `pytest.skip` when a sample is missing, so a broken resolver would show up as skips, not failures. | All manifest pins match (true today: 49 of 49 in-tree pins). Published bytes re-hash to the manifest. #2331's run with the large files removed passes. |
| `samples.shot.sources` (20 files, 179.2 MB) | Git | Release asset, next to its representations | 58: test 31, workflow 16, vaft 7, docs 3, notebooks 1 | #2328, #2331 | Workflows regenerate samples from `generation.canonical_source`, so they need the source resolvable by manifest path. | The generation command still runs with the source fetched from the Release. |
| `efit.green_tables_lfs` (4 LFS pointers, 165.1 MB objects) | Git LFS, CI restores them from an Actions cache | One baseline Release + `generated/efund` cache for other identities | 24: workflow 12, test 8, docs 2, .github 1, vaft 1 | #2330, then #2331 for CI | Three test files (`test_efit_packaged_table.py`, `test_efit_tables_workflow.py`, `test_efund_adapter.py`) skip through `test/lfs_assets.py` when a table is still a pointer. Once LFS is gone, that helper must check the resolver instead. `workflow/cluster_regeneration` hand-fetches LFS oids for hosts without git-lfs. A shot outside the baseline's era (`vest-pre-43017-pf1906`) must never resolve to the baseline. | Baseline SHA-256 equals `table.files`. The identity function includes the machine inputs and the era. A regenerated-vs-baseline comparison is reported with a tolerance ([policy §5.3](storage_policy.md#53-substitution-contract)). CI has no LFS step. |
| `efit.generated_tables_plain` (`brzgfc.dat`, `mhdout.dat`, 7.0 MB) | Git | Same baseline Release as the `.ddd` set | 8: workflow 4, test 3, vaft 1 | #2330 | These files are part of the EFUND output set but stored outside LFS. Moving only the `.ddd` files would split the baseline. | Same as the row above. |

## Historical items: archive before purge

| Item | Size | Owner | Gate before #207 / LFS cleanup |
| --- | --- | --- | --- |
| 17 superseded sample `reference_id`s, plus earlier bytes of 7 reused ids | ~1.46 GB packed (89 of 91 recorded versions in history) | #2328 | Each one is archived as a Release asset with its manifest, or retired by a decision recorded on #207. Starting with #2328, a CI check rejects a byte change that keeps its `reference_id`. |
| EFIT table generations be3be688b (2026-07, pre-#695) and 4d22d4484 (#695) | 7 LFS objects, 327 MB | #2330 | Archived under a tag distinct from the baseline, with `efit.table_records` as provenance. LFS server availability is checked first. |
| `history.retired_layouts` (`vaft/data/omas`, `vaft/data/imas`, `vaft/data/3991x.json`, `44740_imas_equilibrium.h5`, `packaging/wheel_samples`) | 261 MB packed | #207 | Each path is shown to be contained in a registered sample, or archived. 39916, 39917 and 44740 have no registered sample. |
| `history.gh_pages_backups` (`backup/gh-pages-20261021`, tag `archive/gh-pages-develop`) | 245 MB packed | #2117 | Retention decision. Deleting these refs is the only way to recover the space. |

## Groups with an open decision

These stay where they are until the decision in [policy §8](storage_policy.md#8-unresolved-decisions) is made.

| Group | Size | Consumers (files) | Owner | Note |
| --- | --- | --- | --- | --- |
| `legacy.raw_fixtures` | 83 files, 193.5 MB | 28: test 11, docs 7, notebooks 6, vaft 2, workflow 2 | #2328 | `41514.h5` (38 MB) and the 39915 camera PNGs have no textual consumer. The 45531 manifest pins the two digitizer CSVs. |
| `samples.shot.stage_products` | 15 files, 12.5 MB | 3: tutorial 2, workflow 1 | #2328 | `39915/mhd_stability`, added 2026-11-24 (#2306). |
| `notebooks.outputs` | 61 files, 53.5 MB | not searched | unowned | 279 MB of history. |
| `workflow.results` | 24 files, 20.4 MB | not searched | #2307 follow-up | Some files are provenance records; do not treat them as disposable. |

## Groups that stay in Git

| Group | Size | Consumers (files) | Why it stays |
| --- | --- | --- | --- |
| `samples.shot.manifests` | 7 files, 0.1 MB | via the sample API | The identity record the resolver verifies against. |
| `samples.shot.pinned_parts` | 8 files, 0.7 MB | 3 | Pinned by the 48224 manifest; moves only with its sample. |
| `samples.wheel_projection` | 3 files, 20.8 MB | 3 | Packaged offline sample. Two files are swapped into the wheel; `imas.nc` is repository-only. |
| `samples.tables` | 21 files, 8.4 MB | 28 | Tier A / operational-history tables with byte-exact pinned sources. |
| `samples.kinetic_efit` | 6 files, 6.9 MB | 62 | 48224 historical validation reference (#1718). |
| `efit.gfile_fixtures` | 6 files, 14.7 MB | 138 (over-counted) | Small fixtures; `g040330.*` are 4.3 MB each. |
| `efit.table_records` | 2 files, 44 kB | none textual | Provenance of the pre-#695 table generation. |
| `efit.inputs`, `data.geometry_config`, `data.small_fixtures`, `notebooks.data`, `test.fixtures` | 0.3 / 4.7 / 3.4 / 11.4 / 13.5 MB | not searched | Source, configuration and lightweight fixtures. |
| `docs.media`, `history.gh_pages_current` | 51.6 MB tree; 5 MB exclusive to gh-pages | not searched | Owned by #2117. |

## Consumers outside the data groups

| Consumer | Depends on | Risk | Owner |
| --- | --- | --- | --- |
| MCP (`vaft/mcp/_source.py`, `_tools.py`, `_paths.py`) | The sample API and local artifact reads under `VAFT_ARTIFACT_DIR` | After #2329, a sample request may trigger a download. MCP must use the resolver's offline/explicit-fetch behaviour, not download silently inside a tool call, and the artifact root must not widen to the cache. | #2329 (resolver contract), #2332 (discovery and access boundary) |
| Docs site build (`docs.yml`, `docs/build.py`) | Full clone (`fetch-depth: 0`) | Downloads all history on every run. `filter: blob:none` should be checked against `build.py`'s own reads of `origin/main` and `origin/develop`. | #2331 |
| Cluster regeneration (`workflow/cluster_regeneration`) | LFS oids fetched by hand | Breaks when LFS goes away. | #2330 |

## Work assigned to each issue

| Issue | Takes from this audit |
| --- | --- |
| #2317 (I-2) | Cache namespaces and lifecycle ([policy §6](storage_policy.md#6-local-storage-lifecycle)). The MCP artifact root is an access boundary, not storage. |
| #2328 (I-3) | Publish the representations and sources of the 7 registered samples; archive the 17 superseded `reference_id`s (or record their retirement); a CI check for byte changes that keep their `reference_id`; extend `manifest.yaml` (39915 lacks `source.kind`); decide the asset format, the `legacy.raw_fixtures` split, the stage products, and how `source.filedb_root` appears in the public catalog. |
| #2329 (I-4) | A resolver behind the unchanged sample API. A checkout hit comes first, so existing clones keep working during migration. MCP gets an explicit no-download mode. |
| #2330 (I-5) | Baseline = all six files of `table.files`, identity `e28142c7…`, valid only for era `vest-pre-43017-pf1906`. The identity function hashes only what EFUND reads, plus the machine inputs. Archive the two earlier generations under a separate tag after checking LFS server availability. Remove both `.gitattributes` LFS rules (`vaft/data/efit_table/**` names a directory that does not exist). Update `workflow/cluster_regeneration`. |
| #2331 (I-6) | The ~35 test files that skip on missing samples; `test/lfs_assets.py`; the LFS cache in `package-ci.yml`; `filter: blob:none` for the three `fetch-depth: 0` checkouts (`docs.yml` ×2, `bootstrap-ci.yml`). |
| #2117 | The retention decision for `backup/gh-pages-20261021` and `archive/gh-pages-develop` (245 MB). |
| #207 | Purge the sample representations, the sources, the retired layouts and the LFS pointers, after #2331's gate and the archive step. Estimated ~2.1 GB of the ~3.0 GB clone. Add the 5 MB size guard with its allow-list ([policy §2](storage_policy.md#size-rule-for-new-files)). |
