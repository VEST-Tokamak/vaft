"""The table VAFT ships, and the claim that it knows where it came from (#695).

Until #695 the packaged Green tables had no recoverable origin: which EFUND
build produced them, from which input, was not answerable. They are now the
output of a recorded run over the canonical static geometry, and the manifest
that says so sits beside them. These tests hold that shipped directory to the
two things that make the claim mean anything -- the manifest describes the
files actually present, and it names the assets it was built from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vaft.data.resources import data_path

pytest.importorskip("f90nml")

TABLE_DIRECTORY = Path(data_path("efit"))
MANIFEST = TABLE_DIRECTORY / "efund_table_manifest.json"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.is_file():
        pytest.fail(f"{MANIFEST} is missing; the packaged table must carry its provenance")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_packaged_table_records_its_own_provenance(manifest):
    from vaft.code.efit.efund import table_identity

    record = table_identity(TABLE_DIRECTORY)
    assert record["provenance"] == "manifest"
    assert record["identity"] == manifest["table"]["identity"]
    # The era and the EFUND build, which is what "recorded origin" means.
    assert manifest["machine"]["era"] == "vest-pre-43017-pf1906"
    assert manifest["efund"]["executable"]["sha256"]
    assert manifest["efund"]["config"]["grid"] == {
        "nw": 129, "nh": 129, "rleft": 0.05, "rright": 1.2, "zbotto": -1.5, "ztop": 1.5,
    }
    # Every static asset it was projected from, by hash.
    assets = manifest["machine"]["static_inputs"]
    assert set(assets) >= {"static_geometry", "pf_geometry", "magnetics_geometry"}
    for name, record in assets.items():
        assert len(record["sha256"]) == 64, name


def test_the_manifest_describes_the_files_that_are_there(manifest):
    """A manifest that does not match the bytes beside it is worse than none.

    The packaged `.ddd` files are Git LFS objects. In a checkout without them
    this reads pointer files and the hashes will not match, which is why the
    LFS guard runs first: a missing asset must skip, never fail as a
    provenance defect.
    """
    from lfs_assets import skip_unless_materialized

    skip_unless_materialized(*sorted(TABLE_DIRECTORY.glob("*.ddd")))

    for name, record in manifest["table"]["files"].items():
        path = TABLE_DIRECTORY / name
        assert path.is_file(), name
        assert path.stat().st_size == record["size"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], name


def test_the_shipped_namelist_now_matches_the_tables(manifest):
    """Reversed by #708, and it had to be.

    Until #708 the shipped `mhdin.dat` was the legacy 16-lumped-conductor file
    while the tables beside it were built from 302 filaments -- inert, because
    EFIT takes the coil response from the tables and only `nfsum` from the
    namelist, and the two agreed on that one number.

    They no longer do. The table describes 26 current groups, so a namelist
    saying 16 would have EFIT reading a different coilset from the one the
    k-file writes, which `generate_kfile` now refuses outright. The shipped
    namelist is therefore the one that built the tables.
    """
    import f90nml

    packaged = f90nml.read(str(TABLE_DIRECTORY / "mhdin.dat"))
    assert packaged["machinein"]["nfcoil"] == 530
    assert manifest["machine"]["counts"]["nfcoil"] == 530
    assert manifest["machine"]["counts"]["nfsum"] == packaged["machinein"]["nfsum"] == 26


def test_the_legacy_namelist_is_preserved_as_an_independent_reference():
    """The one thing the switch could have quietly cost, kept on purpose.

    The legacy namelist describes the same machine written by other hands, so
    `test_efund_geometry.py` uses it to check that the canonical projection is
    right rather than merely self-consistent. It could not stay in the table
    directory -- its `nfsum` is 16 and would be read by EFIT -- so it moved to
    `legacy/`, with a README saying why it is there and that it is no longer
    loadable as a table.
    """
    import f90nml

    legacy = TABLE_DIRECTORY / "legacy" / "mhdin.dat"
    assert legacy.is_file(), "the independent geometry reference must not be dropped"
    assert (TABLE_DIRECTORY / "legacy" / "README.md").is_file()

    parsed = f90nml.read(str(legacy))
    assert parsed["machinein"]["nfcoil"] == 16
    assert parsed["machinein"]["nfsum"] == 16
    # And it is genuinely a different description, not a copy of the new one.
    shipped = f90nml.read(str(TABLE_DIRECTORY / "mhdin.dat"))
    assert parsed["machinein"]["nfcoil"] != shipped["machinein"]["nfcoil"]


def test_the_acceptance_envelope_was_not_changed_by_the_switch():
    """Scope: #695 moved the table, and nothing else.

    The `&incheck` bounds EFIT reads are a separate decision (#649), still
    carried by the packaged namelist at their packaged values. A switch that
    quietly moved them too would confound the two.
    """
    import f90nml

    from vaft.code.efit.config import EFITAcceptanceEnvelope

    bundled = f90nml.read(str(TABLE_DIRECTORY / "mhdin.dat"))["incheck"]
    envelope = EFITAcceptanceEnvelope()
    for field in ("aminor_min", "aminor_max", "rcntr_min", "rcntr_max"):
        assert bundled[field] == pytest.approx(getattr(envelope, field)), field
