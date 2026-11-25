# VAFT data storage policy

Status: **proposed** under #2307, for approval on #2304. Evidence: [`inventory.yaml`](inventory.yaml) and [`measurements/2026-11-25.json`](measurements/2026-11-25.json).

> Version scientific references, regenerate reproducible artifacts, cache reusable data, and keep Git focused on source code and lightweight metadata. (#2304)

## 1. Principles

1. **One authoritative location per asset, plus at most one disposable cache.** The authority is where the asset's identity is defined and from which it can be recovered. A cache can be deleted at any time without losing information.
2. **Identity is the manifest, not the shot number.** A reference is named by its `reference_id`, which records source kind, lineage and version, together with the SHA-256 of each stored representation. Two products of the same shot with different lineage are different references (#1712).
3. **A manifest identifies an artifact; it does not prove scientific equivalence.** Regenerating an artifact from the same inputs does not replace the original until a stated comparison has passed (§4.3).
4. **Size alone never justifies deletion.** An asset whose role is unclear stays where it is, in the decision backlog (§7), until it has been classified.
5. **Git keeps what has meaningful revision history.** Source code, configuration, manifests and small fixtures belong in Git. Large compressed files do not, because Git can neither diff nor delta them. Each new version costs about its full size: the sample source products pack at 97 % of their raw size.

## 2. Categories and placement

| Category | Authority | Cache | Groups (inventory id) |
| --- | --- | --- | --- |
| Canonical source/configuration | Git | – | `samples.shot.manifests`, `efit.inputs`, `data.geometry_config` |
| Lightweight offline fixture | Git (and the wheel where listed) | – | `samples.wheel_projection`, `samples.tables`, `samples.kinetic_efit`, `efit.gfile_fixtures`, `data.small_fixtures`, `notebooks.data`, `test.fixtures` |
| Versioned scientific reference | GitHub Release asset, pinned by manifest + lockfile | `VAFT_CACHE_DIR/references/samples/` | `samples.shot.representations`, `samples.shot.sources` |
| Regenerable calculation artifact | One baseline Release + the generator inputs in Git | `VAFT_CACHE_DIR/references/efit-baseline/`, `VAFT_CACHE_DIR/generated/efund/` | `efit.green_tables_lfs`, `efit.generated_tables_plain` |
| Persistent experimental/computational data | FileDB / HSDS | HSDS local cache (#964, #2317) | complete pipeline products (not in the repository) |
| External public dataset | Upstream source, pinned in `vaft/data/public` | `VAFT_PUBLIC_DATA_DIR` or `VAFT_CACHE_DIR/public` | registry only in the repository |
| Documentation media | #2117 (docs-media release, gh-pages) | – | `docs.media`, `history.gh_pages` |
| Decision backlog | unchanged until decided | – | `samples.shot.stage_products`, `legacy.raw_fixtures`, `notebooks.outputs`, `workflow.results` |

Cache paths are the names #2317 proposes. The resolver itself is #2317's job.

### Size rule for new files

A new file of **5 MB or more**, or a set of files adding 5 MB or more to one directory, needs a category before it is committed. If it is a reference or a regenerable artifact, it goes to its Release or cache route rather than into Git. #207 adds the preventive check, a pre-commit and CI size guard with an explicit allow-list, together with the history rewrite. The threshold is a starting point: today, the largest files of every fixture group are under 7 MB.

## 3. Reference samples

### 3.1. What exists (develop f878d31ae)

Every registered sample's stored bytes match its manifest SHA-256 (`inventory.yaml` → `samples`).

| Shot | `reference_id` | Source kind | Representations | Directory |
| --- | --- | --- | --- | ---: |
| 39915 | vest-39915-until-efit-v5 | not recorded | omas (wheel variant swapped in), imas | 77.5 MB |
| 40600 | vest-40600-camera-fluctuation-v2 | camera-fluctuation-sample | omas | 21.3 MB |
| 41524 | vest-41524-pipeline-full-imas-v6 | pipeline-until-efit | omas, imas | 105.3 MB |
| 41672 | vest-41672-pipeline-full-imas-v6 | pipeline-until-efit | omas, imas | 114.9 MB |
| 42962 | vest-42962-pipeline-full-imas-v1 | pipeline-until-efit | omas, imas | 124.7 MB |
| 45531 | vest-45531-fluctuation-v2 | fluctuation-diagnostics-sample | omas | 45.9 MB |
| 48224 | vest-48224-kinetic-300ms-v2 | kinetic-profile-sample | omas | 1.5 MB |

All use IMAS DD 3.41.0. 39915's manifest has no `source.kind`; #2328 should fill it in when it extends the schema.

### 3.2. Rules

- **Placement.** The representations (`omas.json.gz`, `imas.nc`) and the source products (`source/*`) move to Release assets (#2328). `manifest.yaml` stays in Git. It is what `available_samples()` lists and what the resolver (#2329) verifies against.
- **Sources are kept, not dropped.** `generation.canonical_source` is the input from which the representations are regenerated. A Release keeps them together. Being able to regenerate a representation does not make the published one disposable.
- **Immutability.** A published asset never changes. Changed bytes require a new `reference_id`, a new asset and a manifest update. The Release tag and the scientific version are separate (#2304 §3.2).
- **Packaged projection.** The wheel ships the compact `wheel_samples/39915` (OMAS only), which `setup.py` swaps in for `samples/39915` at build time. It stays in Git and in the wheel. `test/verify_dist.py` caps the wheel at 26 MiB (0.9.0 wheel: 18.1 MB).
- **Portfolio ownership.** This policy decides *where* bytes live. *Which* references exist, and which lineage they have, stays with #1712/#1718. Moving a sample to a Release must not change its content.
- **Per-shot stage products** (`<shot>/mhd_stability/*`, 13 MB in 23 files) are undecided (§7). Until decided, they stay in Git, and the §2 size rule applies to new ones.

## 4. EFIT Green tables (EFUND)

### 4.1. Baseline identity

The baseline is the table set recorded in `vaft/data/efit/efund_table_manifest.json`, generated 2026-09-14 by `workflow/efit_tables/regenerate_legacy_table.py`:

| Field | Value |
| --- | --- |
| `table.identity` | `e28142c72453c49c161839ea1d9be7f9e08f0aab8f8ece1d6da70c2fd4f80772` |
| `efund.config_sha256` | `b98479ce0d52…` (grid 129×129, R 0.05–1.2 m, Z ±1.5 m, `ifcoil=1 ivesel=1 igrid=1`, Gauss 8/10) |
| `efund.input` (`mhdin.dat` as run) | `af703884aa06…` |
| Executable (provenance, not identity) | `efund` build `4d10ed5-dirty`, sha256 `2d9ff9d7ed5b…` |
| Files | `rv129129.ddd` 134.5 MB, `ec129129.ddd` 20.6 MB, `ep129129.ddd` 10.0 MB, `rfcoil.ddd` 15.6 kB (all LFS); `brzgfc.dat` 6.9 MB, `mhdout.dat` 69 kB (plain Git) |

The shipped `mhdin.dat` no longer hashes to the input recorded above. #649 replaced its `&incheck` block, which EFUND never reads. #965 moved the limiter, which EFUND also does not read. The manifest records both changes. So the identity of a table and the hash of the shipped namelist file are different things, and #2330's identity function must hash only what EFUND reads.

### 4.2. Distribution

- The baseline set (all six files, including `brzgfc.dat` and `mhdout.dat`) is published **once** as a Release asset with a lockfile entry (#2330). No other configuration is ever published.
- Resolution order:
  1. A local table with exactly the requested identity.
  2. Generate it with EFUND into `generated/efund/<identity>/`.
  3. Download the baseline, if and only if the request's identity equals `e28142c7…`.
  4. Otherwise, fail explicitly.
- Generating a table requires a built `efund` (VAFT's install lives under `EFITHOME`), a large stack (`ulimit -s` hard; it segfaults otherwise), and about 3 minutes (184.6 s for the baseline).

### 4.3. Substitution contract

There are two contracts, and they must not be merged:

- **Preservation.** The published bytes equal the current LFS objects. The SHA-256 values are in `table.files`.
- **Regeneration validation.** A locally generated table with the baseline identity may stand in for the baseline in a regression test only after a numerical comparison with a stated tolerance has passed and been reported. #695 showed that two table generations can change EFIT reconstruction yield. Its follow-up also showed that unrelated settings can change the acceptance count, so the comparison must hold everything else fixed.

## 5. Local storage lifecycle

| Store | Lifecycle | Who may delete | Policy owner |
| --- | --- | --- | --- |
| FileDB (`VAFT_FILEDB_DIR`) | Persistent, authoritative pipeline products | Only through FileDB maintenance with provenance | FileDB |
| HSDS local cache (`VAFT_CACHE_DIR/hsds`) | Disposable, LRU-bounded (1 GiB default) | `vaft cache prune/clear` | #964, #2317 |
| Release references (`references/samples`, `references/efit-baseline`) | Disposable copies of immutable assets, verified by SHA-256 | Cache commands | #2317, #2329, #2330 |
| Generated EFUND tables (`generated/efund`) | Disposable, keyed by identity | Cache commands | #2317, #2330 |
| Public datasets (`VAFT_PUBLIC_DATA_DIR`) | Disposable copies of upstream data | Cache commands | #2317 |
| MCP artifact root (`VAFT_ARTIFACT_DIR`) | **Not storage.** An access boundary over user-owned files | Never deleted by VAFT | #2332 |

A cache may share a filesystem with FileDB. It is never derived from `VAFT_FILEDB_DIR`, and FileDB indexing, backup and replication must exclude it (#2317).

## 6. Manifest minimums

Every reference or baseline asset that leaves Git records:

- logical ID and version (`reference_id` or table identity);
- source kind and lineage, with the generating commit;
- the generation command and its canonical input, with that input's SHA-256;
- IMAS DD version, where applicable;
- size and SHA-256 of every stored file;
- the Release tag and asset name, in the lockfile (#2328);
- retrieval and regeneration routes.

Server paths (`source.filedb_root`) are provenance for maintainers. #2328 decides whether the public catalog shows them.

## 7. Unresolved decisions

None of these may be treated as disposable before it is decided.

| Item | Size | Question | Proposed owner |
| --- | --- | --- | --- |
| `legacy.raw_fixtures` | 193.5 MB tree, 120 MB history | SXR digitizer CSVs (pinned by the 45531 manifest), `41514.h5`, shot JSON, camera PNGs, IMPA `.mat` files: which are references (move with #2328), which are fixtures (stay)? `41514.h5` and the PNGs have no textual consumer; check for paths built at run time first. | #2328 |
| `notebooks.outputs` | 53.5 MB tree, 279 MB history | Notebooks keep normalized outputs on purpose. Each re-run stores a full new version (`conference_operational_space_atlas.ipynb`: 10 MB per version). Neither #2117 nor #2304 owns this. | new issue, or #2117 |
| `workflow.results` | 20.4 MB | Dated study outputs (`.xlsx`, `.csv`) next to workflows: results to publish, or fixtures? | #2307 follow-up |
| `samples.shot.stage_products` | 13.2 MB | Ship with the shot's Release bundle, or stay as fixtures? | #2328 |
| Release asset format | – | #1718's 2026-10-06 benchmark: xz-compressed OMAS JSON is byte-identical and 2.2–2.9× smaller for 39915/45531, but 1.5× smaller for 41672. The loader reads only `.json.gz` (`vaft/database/_local.py`). Publishing xz would cut download size, but needs a loader change and a decision. | #2328 |
| `history.retired_sample_layouts` | 261 MB history | Confirm that each retired path (`vaft/data/omas`, `vaft/data/imas`, `vaft/data/3991x.json`, `packaging/wheel_samples`, `44740_imas_equilibrium.h5`) is superseded by a registered sample before #207 purges it. | #207 |

## 8. Reconciled conflicts

- **#1718: "compact canonical bundles in Git LFS".** Superseded for *placement* by this policy. Git LFS stores every version whole, like Git, and adds a bandwidth quota: the organization exhausted it within four days in 2026-09, which is why `package-ci.yml` now restores LFS objects from an Actions cache. So no new LFS content is added, and #1718 step 5's "LFS layout" becomes the Release layout of #2328. #1718 keeps the portfolio and lineage decisions, and its format benchmark carries over to §7.
- **#2117: documentation media.** `docs/assets`, the `docs-media` release and the gh-pages history belong to #2117. This audit only measures them (`docs.media`, `history.gh_pages`). Reference-sample Releases use their own tag namespace, separate from `docs-media`. If #2133 merges, they reuse its download and checksum code.
- **#964: HSDS cache lifecycle.** Unchanged. #2317 adds the shared root without changing the HSDS default path.
- **Retention vs #207.** #207 may purge only what this policy has placed outside Git, and only after #2331 has migrated the consumers. Items in §7 are excluded until decided.
