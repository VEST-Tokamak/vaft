# VAFT data storage policy

Status: **proposed** under #2307, for approval on #2304. Evidence: [`inventory.yaml`](inventory.yaml) and [`measurements/2026-11-25.json`](measurements/2026-11-25.json).

> Version scientific references, regenerate reproducible artifacts, cache reusable data, and keep Git focused on source code and lightweight metadata. (#2304)

## 1. Principles

1. **One authoritative location per asset, plus at most one disposable cache.** The authority is where the asset's identity is defined and from which it can be recovered. A cache can be deleted at any time without losing information.
2. **Identity is the manifest, not the shot number.** A reference is named by its `reference_id`, which carries source kind, lineage and version, together with the SHA-256 of every stored file. Two products of the same shot with different lineage are different references (#1712).
3. **A manifest identifies an artifact; it does not prove scientific equivalence.** Regenerating an artifact from the same inputs does not replace the published one until a stated comparison has passed (§5.3).
4. **History is a store too.** For superseded samples and earlier table generations, Git history is today the only copy. Nothing leaves history unless it is archived first, or unless it is retired by a decision recorded on #207 (§4).
5. **Size alone never justifies deletion.** An asset whose role is unclear stays where it is, in the decision backlog (§8), until it has been classified.
6. **Git keeps what has meaningful revision history.** Source code, configuration, manifests and small fixtures belong in Git. Large compressed files do not, because Git can neither diff nor delta them. Each new version costs about its full size: the sample source products pack at 97 % of their raw size.

## 2. Categories and placement

| Category | Authority | Cache | Groups (inventory id) |
| --- | --- | --- | --- |
| Canonical source/configuration | Git | – | `samples.shot.manifests`, `efit.inputs`, `efit.table_records`, `data.geometry_config` |
| Lightweight offline fixture | Git (and the wheel where listed) | – | `samples.wheel_projection`, `samples.shot.pinned_parts`, `samples.tables`, `efit.gfile_fixtures`, `data.small_fixtures`, `notebooks.data`, `test.fixtures` |
| Versioned scientific reference | GitHub Release asset, pinned by manifest + lockfile | `VAFT_CACHE_DIR/references/samples/`, `…/references/efit-baseline/` | `samples.shot.representations`, `samples.shot.sources`, `efit.green_tables_lfs`, `efit.generated_tables_plain` |
| Independent historical validation reference | Archived Release asset (or Git while it stays small) | `VAFT_CACHE_DIR/references/` | `samples.kinetic_efit`, `history.retired_layouts`, superseded sample versions, earlier EFIT table generations (§4) |
| Regenerable calculation artifact | The generator inputs in Git; the outputs are not stored | `VAFT_CACHE_DIR/generated/efund/` | EFUND tables for any identity other than the baseline |
| Persistent experimental/computational data | FileDB / HSDS | HSDS local cache (#964, #2317) | complete pipeline products (not in the repository) |
| External public dataset | Upstream source, pinned in `vaft/data/public` | `VAFT_PUBLIC_DATA_DIR` or `VAFT_CACHE_DIR/public` | registry only in the repository |
| Documentation media | #2117 (docs-media release, gh-pages) | – | `docs.media`, `history.gh_pages_current`, `history.gh_pages_backups` |
| Decision backlog | unchanged until decided | – | `samples.shot.stage_products`, `legacy.raw_fixtures`, `notebooks.outputs`, `workflow.results` |

Cache paths are the names #2317 proposes. The resolver itself is #2317's job.

The EFIT baseline is a **reference**, not a cache entry, even though EFUND could regenerate it. #695 showed that two table generations reconstruct differently.

### Size rule for new files

A new file of **5 MB or more**, or a set of files adding 5 MB or more to one directory, needs a category before it is committed. If it is a reference or a regenerable artifact, it goes to its Release or cache route rather than into Git. #207 adds the preventive check, a pre-commit and CI size guard with an explicit allow-list, together with the history rewrite.

The threshold is a starting point. Most fixture files are well under it. The exceptions are already in Git: `wheel_samples/39915` (`imas.nc` 10.7 MB, `omas.json.gz` 10.1 MB), `constraint_points.csv` (6.1 MB) and the 48224 kinetic files (6.9 MB in total). The allow-list names them.

## 3. Reference samples

### 3.1. What exists (develop f878d31ae)

`inventory.yaml` → `samples` checks every SHA-256 each manifest pins: representations, sources, configuration, the 45531 digitizer CSVs and the 48224 k-/g-files and equilibria. All 49 pins that resolve to a file in the tree match. Five pins name something outside the repository: `$EFITHOME/bin/efit`, or a FileDB raw dump.

The other pinning manifests are checked too (`inventory.yaml` → `pinned_manifests`). `39915/mhd_stability/manifest.yaml` (14 pins), `test/data/consolidation/case_manifest.yaml` (33) and the Tier A snapshot (2) match. `vaft/data/unified/vest_diagnostics/manifest.yaml` records SHA-256 values for 11 *source code* files (`vaft/machine_mapping/*.py`, `vest.yaml`) that have changed since the fixture was built. That is provenance drift, not data corruption, and it is #2331's to decide whether the fixture or the record is refreshed.

| Shot | `reference_id` | Source kind | Representations | Directory | Pins matched |
| --- | --- | --- | --- | ---: | ---: |
| 39915 | vest-39915-until-efit-v5 | not recorded | omas (wheel variant swapped in), imas | 77.5 MB | 7 (+1 external) |
| 40600 | vest-40600-camera-fluctuation-v2 | camera-fluctuation-sample | omas | 21.3 MB | 1 (+1 external) |
| 41524 | vest-41524-pipeline-full-imas-v6 | pipeline-until-efit | omas, imas | 105.3 MB | 8 (+1 external) |
| 41672 | vest-41672-pipeline-full-imas-v6 | pipeline-until-efit | omas, imas | 114.9 MB | 8 (+1 external) |
| 42962 | vest-42962-pipeline-full-imas-v1 | pipeline-until-efit | omas, imas | 124.7 MB | 9 |
| 45531 | vest-45531-fluctuation-v2 | fluctuation-diagnostics-sample | omas | 45.9 MB | 3 (+1 external) |
| 48224 | vest-48224-kinetic-300ms-v2 | kinetic-profile-sample | omas | 1.5 MB | 13 |

All use IMAS DD 3.41.0. 39915's manifest has no `source.kind`. The inventory records each manifest's `source.git_commit`, authority (`source.filedb_root`), `required_ids` where present, and consumers per shot. It does not record IDS occurrence coverage or validation state, because the manifests do not carry them in a common field. #1712 owns adding them.

### 3.2. Rules

- **Placement.** The representations (`omas.json.gz`, `imas.nc`) and the source products (`source/*`) move to Release assets (#2328). `manifest.yaml` stays in Git. It is what `available_samples()` lists and what the resolver (#2329) verifies against.
- **Sources are kept, not dropped.** `generation.canonical_source` is the input from which the representations are regenerated. A Release keeps them together. Being able to regenerate a representation does not make the published one disposable.
- **Immutability, enforced.** A published asset never changes. Changed bytes require a new `reference_id`. History shows that this was not followed: ten `reference_id`s, four of them current, name more than one version of a representation. So #2328 adds two CI checks:
  - **Across history.** A manifest change that alters a SHA-256 must also change `reference_id`.
  - **Across the tree.** No two manifests in the tree may give one `reference_id` different bytes.

  Each current reference is published under its current bytes.
- **The wheel projection gets its own identity.** Today `wheel_samples/39915` carries `vest-39915-until-efit-v5`, the same id as the repository sample, but holds different bytes. #2328 gives the projection a distinct id (for example `vest-39915-until-efit-v5-wheel`) that records which reference it was projected from. Then #2329 never confuses the two.
- **Packaged projection.** `setup.py` copies `wheel_samples/39915/{manifest.yaml, omas.json.gz}` over `samples/39915` in the wheel. `wheel_samples/39915/imas.nc` is repository-only. All three files stay in Git. `test/verify_dist.py` caps the wheel at 26 MiB (0.9.0 wheel: 18.1 MB).
- **Pinned parts move with their sample.** The 48224 manifest pins `efit_magnetic_server/*` and `equilibria/*` (`samples.shot.pinned_parts`). They move only if and when that sample moves.
- **Portfolio ownership.** This policy decides *where* bytes live. *Which* references exist, and which lineage they have, stays with #1712/#1718. Moving a sample to a Release must not change its content.
- **Per-shot stage products** (`39915/mhd_stability/*`, 12.5 MB in 15 files) are undecided (§8). Until decided, they stay in Git, and the §2 size rule applies to new ones.

## 4. Historical references

### 4.1. What history holds

| Kind | Content | Size (packed) |
| --- | --- | ---: |
| Superseded sample versions | 17 superseded `reference_id`s, plus earlier bytes of reused ids. All 120 representation versions recorded in manifests are still in history, some only under retired paths. | ~1.46 GB (1,876 MB of representations and sources in history, less 412 MB in the current tree) |
| Earlier EFIT table generations | 2026-07 packaged table (be3be688b, pre-#695) and the #695 regeneration (4d22d4484). 7 LFS objects, 317.4 MB, are not in today's tree; `ep129129.ddd` is unchanged since #695 and is part of the baseline. | LFS, outside the pack |
| Retired layouts | `vaft/data/omas/{39915,41524,41672}.json`, an IMAS HDF5 set under `vaft/data/imas/`, `vaft/data/{39915,39916,39917}.json`, `44740_imas_equilibrium.h5`, `packaging/wheel_samples/39915` | 261 MB |
| OMFIT-era kinetic reference | `vaft/data/kineticEfit` (48224), kept by #1718 as a historical validation reference | in the tree, 6.9 MB |

The measurement lists every item: `history.historical_sample_references`, `history.lfs_history` and `history.retired_paths`.

### 4.2. Rules

- **Archive first.** Before #207 removes an item from history, or before anyone deletes an LFS object, the item is either:
  - published as an *archived* Release asset, with a manifest recording its `reference_id` or table identity, its first commit and its SHA-256; or
  - retired by a decision recorded on #207, naming the item and the reason.

  Retirement is the exception. The default is to archive.
- **Not superseded until shown.** A retired layout counts as superseded only when its content is shown to be contained in a registered sample. 39916, 39917 (a confinement reference shot) and 44740 have no registered sample, so their files are historical references, not duplicates. `packaging/wheel_samples/39915` holds bytes that `vest-39915-until-efit-v1` recorded, so it is archived with that reference.
- **EFIT generations.** The pre-#695 table is the subject of #695's finding. Its identity record and comparison are kept in Git (`efit.table_records`: `workflow/efit_tables/reference/legacy-39915-129129.manifest.json`, `compare_bundled_vs_legacy-39915-129129.json`). Both earlier generations are archived with #2330, alongside the baseline but under a distinct archive tag, so nobody can mistake them for the baseline.
- **LFS server state is unverified.** All 11 historical LFS objects are in this machine's LFS store. Whether the LFS server still has the earlier ones was not checked. #2330 verifies this, and if needed uploads from a local store, before any LFS cleanup.

## 5. EFIT Green tables (EFUND)

### 5.1. Baseline identity and applicability

The baseline is the table set recorded in `vaft/data/efit/efund_table_manifest.json`, generated 2026-09-14 by `workflow/efit_tables/regenerate_legacy_table.py`:

| Field | Value |
| --- | --- |
| `label` | `vest-pre-43017-pf1906-129129-all-circuits` |
| `table.identity` | `e28142c72453c49c161839ea1d9be7f9e08f0aab8f8ece1d6da70c2fd4f80772` |
| Applicability | `machine.era` = `vest-pre-43017-pf1906`, `reference_shot` 43016, PF geometry 1906. The table describes all 10 PF circuits in 26 groups (#708), so any subset energised within the era needs no new table. |
| Machine inputs (`machine.static_inputs`, sha256) | coupling `1c391442762f…`, magnetics calibration `1f3566749fa2…`, magnetics geometry `e9e016cba28a…`, PF geometry `5d4d9e45a9fe…`, static geometry `2d3495f7cc39…` |
| Counts | nfcoil 530, nfsum 26, nvesel 950, magpri 64, nsilop 11 |
| `efund.config_sha256` | `b98479ce0d52…` (grid 129×129, R 0.05–1.2 m, Z ±1.5 m, `ifcoil=1 ivesel=1 igrid=1`, Gauss 8/10) |
| `efund.input` (`mhdin.dat` as run) | `af703884aa06…` |
| Executable (provenance, not identity) | `efund` build `4d10ed5-dirty`, sha256 `2d9ff9d7ed5b…` |
| Files (`table.files`) | `rv129129.ddd` 134.5 MB, `ec129129.ddd` 20.6 MB, `ep129129.ddd` 10.0 MB, `rfcoil.ddd` 15.6 kB (LFS); `brzgfc.dat` 6.9 MB, `mhdout.dat` 69 kB (plain Git) |

**The baseline is valid only for its era.** A shot outside `vest-pre-43017-pf1906`, for example one using the 2507 PF geometry, has a different identity. It never resolves to the baseline; it needs a generated table (§5.2, step 2).

Two recorded input hashes no longer match the files as shipped:

- `mhdin.dat` (#649): its `&incheck` block was replaced, and EFUND never reads `&incheck`.
- `VEST_static_geometry.json.gz` (#965): it now hashes `5a55c173fd60…`, not the recorded `2d3495f7cc39…`, because the limiter moved. This changed only the wall, which EFUND does not read.

The other four machine inputs still match. Both changes are recorded in the manifest. A whole-file identity would therefore never match the baseline again, even though the tables are unaffected. #2330's identity function must hash the *content EFUND reads* from each input (namelist groups other than `&incheck`; coil, vessel and diagnostic geometry, not the wall), and it must reproduce `e28142c7…` from today's files before anything relies on it.

### 5.2. Distribution

- The baseline set (all six files) is published **once** as a Release asset, with a lockfile entry (#2330). No other configuration is published. Earlier generations are archived separately (§4.2).
- Resolution order:
  - **Baseline identity** (`e28142c7…`): the published bytes win.
    1. A local copy whose SHA-256 matches `table.files`.
    2. Otherwise, download the baseline.
    3. A locally generated table substitutes only under §5.3.
  - **Any other identity:**
    1. A local table with exactly that identity.
    2. Otherwise, generate it with EFUND into `generated/efund/<identity>/`.
  - Otherwise, fail explicitly.

  This refines #2304 §3.3, which lists generation before download for every identity. Applied to the baseline, that order would let a regenerated table silently replace the published one.
- Generating a table requires a built `efund` (VAFT's install lives under `EFITHOME`) and a large stack (`ulimit -s` hard; it segfaults otherwise). It takes about 3 minutes (184.6 s for the baseline).

### 5.3. Substitution contract

There are two contracts, and they must not be merged:

- **Preservation.** The published bytes equal the current LFS objects. The SHA-256 values are in `table.files`.
- **Regeneration validation.** A locally generated table with the baseline identity may stand in for the baseline in a regression test only after a numerical comparison with a stated tolerance has passed and been reported. #695 showed that two table generations change EFIT reconstruction yield. Its follow-up showed that unrelated settings (`&incheck`) change acceptance counts, so the comparison must hold everything else fixed.

## 6. Local storage lifecycle

| Store | Lifecycle | Who may delete | Policy owner |
| --- | --- | --- | --- |
| FileDB (`VAFT_FILEDB_DIR`) | Persistent, authoritative pipeline products | Only through FileDB maintenance with provenance | FileDB |
| HSDS local cache (`VAFT_CACHE_DIR/hsds`) | Disposable, LRU-bounded (1 GiB default) | `vaft cache prune/clear` | #964, #2317 |
| Release references (`references/samples`, `references/efit-baseline`) | Disposable copies of immutable assets, verified by SHA-256 | Cache commands | #2317, #2329, #2330 |
| Generated EFUND tables (`generated/efund`) | Disposable, keyed by identity | Cache commands | #2317, #2330 |
| Public datasets (`VAFT_PUBLIC_DATA_DIR`) | Disposable copies of upstream data | Cache commands | #2317 |
| MCP artifact root (`VAFT_ARTIFACT_DIR`) | **Not storage.** An access boundary over user-owned files | Never deleted by VAFT | #2332 |

A cache may share a filesystem with FileDB. It is never derived from `VAFT_FILEDB_DIR`, and FileDB indexing, backup and replication must exclude it (#2317).

## 7. Manifest minimums

Every reference, historical reference or baseline asset that leaves Git records:

- logical ID and version (`reference_id` or table identity);
- source kind and lineage, with the generating commit;
- for an archived item, the commit it was first and last stored in;
- the generation command and its canonical input, with that input's SHA-256;
- IMAS DD version, where applicable;
- size and SHA-256 of every stored file;
- the Release tag and asset name, in the lockfile (#2328);
- retrieval and regeneration routes.

Server paths (`source.filedb_root`) are provenance for maintainers. #2328 decides whether the public catalog shows them.

## 8. Unresolved decisions

None of these may be treated as disposable before it is decided.

| Item | Size | Question | Proposed owner |
| --- | --- | --- | --- |
| `legacy.raw_fixtures` | 193.5 MB tree, 120 MB history | SXR digitizer CSVs (pinned by the 45531 manifest), `41514.h5`, shot JSON, camera PNGs and IMPA `.mat` files: which are references (move with #2328), which are fixtures (stay)? `41514.h5` and the PNGs have no textual consumer; check for paths built at run time first. | #2328 |
| `notebooks.outputs` | 53.5 MB tree, 279 MB history | Notebooks keep normalized outputs on purpose. Each re-run stores a full new version (`conference_operational_space_atlas.ipynb`: 10 MB per version). Neither #2117 nor #2304 owns this. | new issue, or #2117 |
| `workflow.results` | 20.4 MB | Dated study outputs (`.xlsx`, `.csv`) next to workflows: results to publish, or fixtures? Some are provenance records. | unowned (new issue) |
| `samples.shot.stage_products` | 12.5 MB | Ship with the shot's Release bundle, or stay as fixtures? | #2328 |
| Release asset format | – | #1718's 2026-10-06 benchmark: xz-compressed OMAS JSON is byte-identical and 2.2–2.9× smaller for 39915/45531, but 1.5× smaller for 41672. The loader reads only `.json.gz` (`vaft/database/_local.py`). Publishing xz would cut download size, but needs a loader change and a decision. | #2328 |
| Historical references (§4) | ~1.7 GB packed (1.46 GB superseded sample bytes + 0.26 GB retired layouts) + 317 MB LFS | For each item: archive, or retire with a recorded reason. | #2328 (samples), #2330 (tables), #207 (retired layouts) |
| gh-pages backups | 245 MB | `backup/gh-pages-20261021` and tag `archive/gh-pages-develop` hold the old site history. Keep, archive elsewhere, or delete? | #2117 |

## 9. Reconciled conflicts

- **#1718: "compact canonical bundles in Git LFS".** Superseded for *placement* by this policy. Git LFS stores every version whole, like Git, and adds a bandwidth quota. The organization exhausted it within four days in 2026-09, which is why `package-ci.yml` now restores LFS objects from an Actions cache. So no new LFS content is added, and #1718 step 5's "LFS layout" becomes the Release layout of #2328. #1718 keeps the portfolio and lineage decisions, including keeping 48224 as a historical validation reference. Its format benchmark carries over to §8.
- **#2117: documentation media.** `docs/assets`, the `docs-media` release and the gh-pages refs belong to #2117. This audit only measures them. The current `origin/gh-pages` is already one commit, with 5 MB not shared with code branches. The 245 MB is on a backup branch and an archive tag, so recovering it is a retention decision (§8), not a site-build change. Reference-sample Releases use their own tag namespace, separate from `docs-media`. If #2133 merges, they reuse its download and checksum code.
- **#964: HSDS cache lifecycle.** Unchanged. #2317 adds the shared root without changing the HSDS default path.
- **Retention vs #207.** #207 may purge only what this policy has placed outside Git, after #2331 has migrated the consumers and after §4.2's archive step. Items in §8 are excluded until decided.
