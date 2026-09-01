"""The CHEASE input must be unchanged by deriving its sign pattern.

`vaft/code/chease.py` normalises a g-file to a fixed sign pattern before writing
CHEASE's inputs.  That pattern used to be the literal `CHEASE_COCOS02_SIGNS`
dict; it is now derived from the COCOS index the registry declares for CHEASE
together with the orientation CHEASE requires.  The derivation must reproduce
the literal exactly, because everything downstream -- `EXPEQ`, the namelist, the
re-signed g-file -- is built from it.

The check is an A/B comparison inside one process: the same source is prepared
twice, once through the derivation and once with the legacy literal forced back
in, and every byte of every produced file is compared.  Recording hashes of the
files instead would not work, because `chease_cocos_transform.json` embeds the
absolute path of the source g-file and floating-point formatting is not
guaranteed identical across platforms; such a manifest fails on any machine but
the one that wrote it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

SOURCES = (
    "efit/g039915.00319",
    "efit/g040330.00320",
    "kineticEfit/g048224.00300.chease",
)

#: The sign pattern as it was written before it was derived, kept verbatim.
LEGACY_CHEASE_SIGNS = {
    "desired_dpsi_sign": -1,
    "desired_bcentr_sign": 1,
    "desired_current_sign": -1,
    "desired_fpol_sign": 1,
    "desired_q_sign": -1,
}


def _prepare(name: str, workdir: Path) -> dict[str, bytes]:
    from vaft.code.chease import CHEASEConfig, prepare_chease_inputs
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample

    workdir.mkdir(parents=True, exist_ok=True)
    prepare_chease_inputs(
        read_geqdsk(require_repository_sample(data_path(name))),
        CHEASEConfig(workdir=workdir),
    )
    return {
        item.name: item.read_bytes()
        for item in sorted(workdir.iterdir())
        if item.is_file()
    }


@pytest.mark.parametrize("name", SOURCES)
def test_deriving_the_sign_pattern_leaves_cheases_input_byte_identical(name, tmp_path, monkeypatch):
    """Every file CHEASE is handed must be unchanged by the refactor."""
    import vaft.code.chease as chease

    derived = _prepare(name, tmp_path / "derived")

    monkeypatch.setattr(chease, "_desired_signs_for_chease", lambda: dict(LEGACY_CHEASE_SIGNS))
    legacy = _prepare(name, tmp_path / "legacy")

    assert set(derived) == set(legacy), f"{name}: the set of files handed to CHEASE changed"
    for filename in sorted(derived):
        assert hashlib.sha256(derived[filename]).hexdigest() == hashlib.sha256(legacy[filename]).hexdigest(), (
            f"{name}/{filename}: contents differ between the derived and literal sign patterns"
        )


def test_the_derived_pattern_equals_the_literal_it_replaced():
    from vaft.code.chease import _desired_signs_for_chease

    assert _desired_signs_for_chease() == LEGACY_CHEASE_SIGNS


def test_the_sign_pattern_chease_is_given_matches_sauter_equation_23_for_cocos_2():
    """The literal was five sign choices written as though independent.

    They are not: Eq. 23 fixes all of them once the index and the two
    orientation signs are chosen, which is what makes deriving them possible.
    """
    from vaft.code.chease import CHEASE_COCOS02_SIGNS, CHEASE_ORIENTATION
    from vaft.data.cocos import convention_for, cocos_spec

    spec = cocos_spec(convention_for("chease").cocos)
    assert spec.index == 2  # Sauter Sect. IX
    sigma_ip, sigma_b0 = CHEASE_ORIENTATION["sigma_ip"], CHEASE_ORIENTATION["sigma_b0"]
    assert (sigma_ip, sigma_b0) == (-1, +1)
    assert CHEASE_COCOS02_SIGNS["dpsi"] == spec.expected_sign("dpsi", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert CHEASE_COCOS02_SIGNS["q"] == spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert CHEASE_COCOS02_SIGNS["fpol"] == spec.expected_sign("f", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    assert CHEASE_COCOS02_SIGNS["current"] == sigma_ip
    assert CHEASE_COCOS02_SIGNS["bcentr"] == sigma_b0


def test_the_comparison_would_catch_a_changed_sign_pattern(tmp_path, monkeypatch):
    """Guard against the A/B test being vacuous.

    If the derivation ever stopped matching the literal, the byte comparison has
    to fail -- so flipping one sign must produce different input for CHEASE.
    """
    import vaft.code.chease as chease

    baseline = _prepare(SOURCES[0], tmp_path / "baseline")

    wrong = dict(LEGACY_CHEASE_SIGNS, desired_q_sign=+1)
    monkeypatch.setattr(chease, "_desired_signs_for_chease", lambda: dict(wrong))
    perturbed = _prepare(SOURCES[0], tmp_path / "perturbed")

    assert any(baseline[name] != perturbed[name] for name in baseline), (
        "flipping a sign changed nothing, so the parity comparison proves nothing"
    )
