"""Byte parity for the files CHEASE is handed.

`vaft/code/chease.py` normalises a g-file to a fixed sign pattern before writing
CHEASE's inputs, using its own `_force_geqdsk_signs` rather than the shared COCOS
layer.  Replacing that with a declared COCOS 2 target must not change a single
byte of what CHEASE reads, so the hashes of every produced file were recorded
before the refactor and are checked here.

Parity is a red/green signal, not a claim: if these hashes move, the refactor
changed CHEASE's input and the difference has to be explained and accepted
deliberately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

MANIFEST = Path(__file__).with_name("data") / "chease_parity_manifest.json"


def _manifest() -> dict:
    if not MANIFEST.exists():  # pragma: no cover - the file ships with the repo
        pytest.skip(f"{MANIFEST} is a repository-only fixture")
    return json.loads(MANIFEST.read_text())


def _prepare(name: str, workdir: Path) -> dict[str, Path]:
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample

    prepare_chease_inputs(
        read_geqdsk(require_repository_sample(data_path(name))),
        CHEASEConfig(workdir=workdir),
    )
    return {item.name: item for item in sorted(workdir.iterdir()) if item.is_file()}


def test_the_manifest_covers_every_packaged_source():
    manifest = _manifest()
    assert set(manifest["artifacts"]) == {
        "efit/g039915.00319",
        "efit/g040330.00320",
        "kineticEfit/g048224.00300.chease",
    }
    for name, entry in manifest["artifacts"].items():
        assert {"EXPEQ", "chease_namelist", "input.geqdsk", "source.geqdsk"} <= set(entry), name


@pytest.mark.parametrize("name", [
    "efit/g039915.00319",
    "efit/g040330.00320",
    "kineticEfit/g048224.00300.chease",
])
def test_prepared_chease_inputs_are_byte_identical_to_the_recorded_hashes(name, tmp_path):
    expected = _manifest()["artifacts"][name]
    produced = _prepare(name, tmp_path)

    assert set(produced) == set(expected), (
        f"{name}: the set of files handed to CHEASE changed"
    )
    for filename, record in sorted(expected.items()):
        data = produced[filename].read_bytes()
        assert len(data) == record["bytes"], f"{name}/{filename}: size changed"
        assert hashlib.sha256(data).hexdigest() == record["sha256"], (
            f"{name}/{filename}: contents changed; CHEASE would receive different input"
        )


def test_the_sign_pattern_chease_is_given_matches_sauter_equation_23_for_cocos_2():
    """The hardcoded target is COCOS 2 with Ip < 0 and B0 > 0, not an arbitrary pattern.

    `CHEASE_COCOS02_SIGNS` was written as five independent sign choices.  They
    are not independent: Eq. 23 fixes all of them once the index and the two
    orientation signs are chosen, which is what makes replacing the bespoke
    forcing with the shared layer possible at all.
    """
    from vaft.code.chease import CHEASE_COCOS02_SIGNS
    from vaft.data.cocos import cocos_spec

    spec = cocos_spec(2)
    sigma_ip, sigma_b0 = -1, +1
    assert CHEASE_COCOS02_SIGNS["dpsi"] == spec.expected_sign("dpsi", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert CHEASE_COCOS02_SIGNS["q"] == spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert CHEASE_COCOS02_SIGNS["fpol"] == spec.expected_sign("f", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert CHEASE_COCOS02_SIGNS["current"] == sigma_ip
    assert CHEASE_COCOS02_SIGNS["bcentr"] == sigma_b0
