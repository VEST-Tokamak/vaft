# Releasing VAFT

For the first publication, follow the Korean handoff guide in
[`docs/pypi-first-release.md`](docs/pypi-first-release.md).

## One-time setup

1. On PyPI, add a GitHub Trusted Publisher for `VEST-Tokamak/vaft` with the
   workflow filename `release-pypi.yml` and environment `pypi`.
2. In GitHub repository settings, create the `pypi` environment and require a
   maintainer review before deployment.
3. Protect the release branch (`develop` or `main`) and require the `Package
   CI / package` check before merging.

The workflow uses GitHub OIDC, so no PyPI API token should be stored in GitHub
secrets.

## Release procedure

1. Update `vaft/version.py` to the next PEP 440 version.
2. Update release notes or the project changelog.
3. Merge the release commit only after Package CI passes.
4. Create and push the matching tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. Approve the `pypi` environment deployment in GitHub Actions. The workflow
   rebuilds, validates, and publishes the artifacts.

PyPI versions are immutable. If publishing fails after upload, publish a new
version rather than trying to overwrite `vX.Y.Z`.

## Distribution policy

`scripts/verify_dist.py` is the release gate for package contents. It requires
runtime geometry resources, `omas/39915.json`, and `legacy/sql_table.txt`; it
rejects repository-only samples and source-distribution tests, and limits the
wheel to 25 MiB. Update that script together with `pyproject.toml` and
`MANIFEST.in` whenever the PyPI data policy changes.
