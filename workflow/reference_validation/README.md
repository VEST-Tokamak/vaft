# Legacy VEST reference validation

This directory contains the reproducible capture and comparison procedure for
GitHub issue #88. The legacy server and `/srv/vest.filedb/public` are reference
sources only; neither is required by CI or the production workflow.

## Repository fixtures

`test/data/vest_reference` contains:

- `manifest.yaml`: workflow hashes, representative-shot status expectations,
  external artifact checksums, inventory summaries, and naming conventions;
- `config.snapshot.yaml`: a captured non-secret legacy configuration;
- `tolerances.yaml`: versioned path-specific comparison tolerances;
- `shot-39915-compact.json.gz`: a compact subset of the authoritative legacy
  `39915_combined.json` artifact.

Verify all repository-resident artifacts without contacting the server:

```python
from vaft.omas import verify_reference_artifacts

results = verify_reference_artifacts("test/data/vest_reference/manifest.yaml")
assert all(item.valid for item in results)
```

## Rebuilding the compact fixture

Download the full reference artifact to a temporary location. Do not commit it:

```bash
scp -P 2222 \
  user1@147.46.36.244:/srv/vest.filedb/public/39915/omas/39915_combined.json \
  /tmp/vaft-legacy-39915-combined.json
```

The extractor refuses a source whose checksum differs from the manifest:

```bash
python workflow/reference_validation/extract_compact_ods.py \
  /tmp/vaft-legacy-39915-combined.json
```

If the legacy reference intentionally changes, create a new `reference_id` and
review the scientific differences before updating checksums. Do not silently
replace the existing reference generation.

## Comparing a candidate ODS

```bash
python -m vaft.omas.comparison \
  test/data/vest_reference/shot-39915-compact.json.gz \
  path/to/candidate.json.gz \
  --tolerances test/data/vest_reference/tolerances.yaml \
  --scope reference \
  --json-report comparison.json \
  --markdown-report comparison.md
```

The process exits nonzero when any path is classified as an unintended
regression. Intentional improvements and unavailable legacy quantities must be
declared by path in the tolerance policy and include a rationale.

## Large external artifacts

The manifest records server paths, byte sizes, and SHA-256 checksums for the
large reference products. They stay outside Git. To validate a downloaded
artifact, preserve its manifest-relative identity or call `sha256_file()` and
compare it with the recorded checksum.

The active workflow snapshot was captured independently of the historical
artifact creation times because the legacy workflow does not record a commit
identifier in its ODS products. This limitation is explicit in the manifest.
