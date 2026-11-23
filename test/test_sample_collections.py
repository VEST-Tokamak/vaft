"""The repository-only sample collections hold exactly the bytes their manifests record.

``vaft/data/samples/vest_tier_a`` and ``vest_operational_history`` were moved
byte-for-byte and are read by notebooks and tutorials that trust the recorded
SHA-256. Every recorded hash is checked here, not a chosen few, and the Tier A
bytes must be protected from end-of-line conversion: a later
``samples/** text=auto`` rule would otherwise override ``-text`` and a
renormalisation would silently change them.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from vaft.data import data_path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded(collection: str) -> dict[str, str]:
    base = data_path(f"samples/{collection}")
    manifest = json.loads((base / "manifest.json").read_text())
    if "sha256" in manifest:
        return dict(manifest["sha256"])
    return {name: entry["sha256"] for name, entry in manifest["files"].items()}


@pytest.mark.parametrize("collection", ["vest_tier_a", "vest_operational_history"])
def test_every_recorded_hash_matches_the_file(collection):
    base = data_path(f"samples/{collection}")
    recorded = _recorded(collection)
    assert recorded
    wrong = {name: digest[:12] for name, digest in recorded.items() if _sha256(base / name) != digest}
    assert not wrong, f"{collection}: manifest hash disagrees with the file for {wrong}"


def test_tier_a_bytes_are_not_end_of_line_converted():
    try:
        result = subprocess.run(
            ["git", "check-attr", "text", "--", "vaft/data/samples/vest_tier_a/v1/state.csv"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert result.stdout.strip().endswith("text: unset"), result.stdout
