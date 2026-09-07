# Consolidating external soft X-ray and camera data

Soft X-ray and FAST-camera data are the two VEST diagnostics that never reached
the routine pipeline. Their raw files are not in the SQL DAQ; they arrive as
per-shot digitizer CSVs and directories of exported BMP frames, and they
accumulated across acquisition dumps, per-request copies, and working copies
inside analysis repositories. Three scripts here turn that into one archive the
machine mappings can read, and then into IDSs.

```
inventory_external_diagnostics.py   read-only scan -> inventory.json
consolidate_external_diagnostics.py inventory.json -> moves + move_manifest.jsonl
ingest_external_diagnostics.py      archive        -> one IDS product per shot
```

## Layout

```
{root}/legacy/                            replicated to the server
  soft_x_rays/{shot}/digitizer_{daq}_{shot}.csv
  soft_x_rays/_geometry/                  line-of-sight tables and grids
  camera_visible/{shot}/{shot}_{frame:08d}.bmp + {shot}_bmp.txt
  camera_visible_fluctuation/{shot}/      >= 50 kfps, reserved for issue #161
  camera_visible_fluctuation/index.json   what is reserved and why

{root}/unmapped/                          local only; no mapping reads these
  camera_visible_arranged/{shot}/         bmp_arranger output, derived
  camera_visible_mcf/{shot}.mcf           vendor container, no reader exists
  camera_ccd_2013/{shot}/                 2013-era CCD, jpg/avi
  hard_x_rays/{shot}/                     hard_x_rays is not_implemented

{root}/ods/{tree}/{shot}/                 generated IDS products + manifests
```

`legacy/` is the canonical FileDB domain that
`vaft.database.filedb.FileDB.legacy(diagnostic, shot)` resolves, and both
mappings find their files there from a data root alone:

```python
camera_visible(ods, 27134, data_root=root / "legacy" / "camera_visible")
soft_x_rays(ods, 39107, data_root=root / "legacy" / "soft_x_rays")
```

`unmapped/` is deliberately *not* a `FileDBDomain` value. Everything in it is
real data worth keeping, but handing any of it to a mapping as a data root
would either fail or, worse, quietly ingest derived frames as if they were raw.
Keeping it outside the grammar makes that impossible rather than merely
discouraged.

`ods/` is a holding area, not a canonical location: neither diagnostic has an
`OMASStage` of its own yet. Giving them stages, and a `STAGE_REPLICATION` entry
in `vaft/database/sources.py` so their IDSs reach HSDS, is issue #130's job.

## Classification

Every decision comes from names and header text, so it is reproducible and
testable (`test/test_external_diagnostic_inventory.py`) without touching the
archive:

| Signal | Meaning |
|---|---|
| `{stem}_{frame:08d}.bmp` | raw frames, original indices intact — mapper input |
| `{stem}_{time}_ms.bmp` | `bmp_arranger` output — derived, not fixture input |
| `Frame_Rate: >= 50000` in `{stem}_bmp.txt` | fluctuation-grade, reserved |
| `digitizer_{daq}_{shot}.csv` | soft X-ray |
| `digitizer_hxr_{variant}_{daq}_{shot}.csv` | hard X-ray, a different diagnostic |

Export directories carry an acquisition suffix on the directory *and* on every
file inside (`27134_50kHz/27134_50kHz_00000000.bmp`, `36976-N001/...`). The
suffix is stripped on the way in, because `camera_visible` looks for
`{shot}/{shot}_{frame:08d}.bmp`; it is preserved in each shot's
`provenance.json`.

Anything that does not classify cleanly is reported as needing review and left
where it is. Nothing is guessed at.

## Reversibility

Every operation is appended to `{root}/move_manifest.jsonl` and flushed before
the next one starts.

```bash
./consolidate_external_diagnostics.py --revert {root}/move_manifest.jsonl            # dry run
./consolidate_external_diagnostics.py --revert {root}/move_manifest.jsonl --execute
```

Two things a revert does not restore, both intentional: export residue
(`Thumbs.db`, AppleDouble `._*` sidecars), which carries no information; and the
originals of files copied out of a git checkout, which were never moved.

Moves within one filesystem are renames, so consolidating 216 GiB costs no disk
space and takes seconds. Sources on another volume are copied and hash-verified
before the original is released. Sources inside a git working tree are always
copied, never moved — a tracked file leaving its repository shows up as a
deletion in a repository this tooling has no business editing.

## Usage

```bash
./inventory_external_diagnostics.py --output inventory.json --report -
./consolidate_external_diagnostics.py --inventory inventory.json --root "$VAFT_FILEDB_DIR"
./consolidate_external_diagnostics.py --inventory inventory.json --root "$VAFT_FILEDB_DIR" --execute
./ingest_external_diagnostics.py --root "$VAFT_FILEDB_DIR" --diagnostic soft_x_rays
```

`--only` restricts consolidation to named diagnostics; `--shot`, `--limit` and
`--format` restrict and shape ingest. Routine ingest never reads
`camera_visible_fluctuation`; `--include-fluctuation` is the deliberate override.

## Replication

Only `legacy/` goes to the server:

```bash
rsync -a --partial --exclude='._*' --exclude='Thumbs.db' \
  "$VAFT_FILEDB_DIR"/legacy/{soft_x_rays,camera_visible,camera_visible_fluctuation} \
  vestuser1:/srv/vest.filedb/legacy/
```

## What the archive holds, and what it costs

Measured after the first full pass:

| | shots | ingested | products |
|---|---|---|---|
| `soft_x_rays` | 200 | 188 | 34 GB |
| `camera_visible` | 483 | 398 | 231 GB |
| `camera_visible_fluctuation` | 52 | reserved | — |

Failures are recorded per shot in `ods/{tree}/ingested_shots.json` with a
reason. They are data properties, not tooling faults:

- **12 soft X-ray shots** (39217, 39260, and the run 39606–39615) have ragged
  digitizer CSVs — a row part-way through the file has a different column
  count from the header row. The mapping rejects them, correctly.
- **85 camera shots** have no non-dark frames at all: the camera was armed but
  recorded nothing visible.

Coverage is thinner than the raw frame counts suggest, in both diagnostics.
Dark-frame rejection keeps a median of 41 frames per camera shot, and 179 of
the 200 soft X-ray shots were recorded on one digitizer rather than two, so
they yield 40 channels instead of 128. Both numbers are in each shot's
manifest; judge the archive by those, not by directory size.

## Known limits

- Camera IDS products are eight times larger than they need to be on disk:
  `vfit_camera_visible_dynamic` stores 8-bit frames as int64. 231 GB where
  ~29 GB would do. This is a *storage* cost only — the raw frame list stays
  uint8 during the build, and peak memory for the largest shot (38769, 2167
  frames at 1024×1280) measures 2.75 GB.
- Shots 26576, 26757 and 29770, which issue #161 names as required, are not
  present in any local source. 27134 and 32308 are, at 50 kfps.
- Seven camera directories have a `_bmp.txt` that is a filesystem index record
  rather than text; their frames cannot be dated and they are held for review.
- Header parsing in `camera_visible` is positional — `Frames` is pinned to line
  16, `Top Frame` to line 75 — so a header that deviates in layout fails rather
  than degrading. The inventory scanner reads the same file by field name; that
  disagreement is how the seven corrupt headers were found.
