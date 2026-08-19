# Releasing VAFT

## One-time setup

1. On PyPI, add a GitHub Trusted Publisher for `VEST-Tokamak/vaft` with the
   workflow filename `release-pypi.yml` and environment `pypi`.
2. In GitHub repository settings, create the `pypi` environment and require a
   maintainer review before deployment.
3. Protect `main` and require the `Package CI / package` check before merging.

The workflow uses GitHub OIDC, so no PyPI API token should be stored in GitHub
secrets.

## Release procedure

1. Update `vaft/version.py` to the next PEP 440 version.
2. Update the release notes or project changelog.
3. Merge the release commit into `main` only after Package CI passes.
4. From the updated `main` branch, build and validate the distributions:

   ```bash
   python -m build
   python test/verify_dist.py dist --max-wheel-mib 25
   python -m twine check dist/*
   ```

5. Create and push a matching tag that points at the validated commit:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. Approve the `pypi` environment deployment in GitHub Actions. The workflow
   rebuilds, validates, and publishes the artifacts.

PyPI versions are immutable. If publishing fails after upload, publish a new
version rather than trying to overwrite `vX.Y.Z`.

## Distribution policy

`test/verify_dist.py` is the release gate for package contents. It requires the
runtime geometry resources, `omas/39915.json`, `legacy/sql_table.txt`, and
`legacy/diagnostic-trigger-settings.yaml`; it rejects repository-only samples
and source-distribution tests, and limits the wheel to 25 MiB.

Keep the distribution rules in `pyproject.toml`, `MANIFEST.in`, and
`test/verify_dist.py` synchronized whenever the PyPI data policy changes.
