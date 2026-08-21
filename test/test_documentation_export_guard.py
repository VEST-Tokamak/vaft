"""Regression tests for the documentation notebook remote-write guard."""

import pytest

from scripts.export_documentation_outputs import validate_source


@pytest.mark.parametrize(
    "source",
    (
        "save_ods(data)",
        "vaft.database.ods.save_ods(data)",
        "client.save_ids(data)",
        "database.save_shot_as(data)",
        "adapter.write_to_hdf5(data)",
        "vaft.database.save(*args)",
        'hsload("/remote", "w")',
        'h5py.File("remote.h5", "r+")',
    ),
)
def test_write_guard_rejects_qualified_and_variadic_calls(source):
    with pytest.raises(RuntimeError, match="rejected remote or write-enabled API"):
        validate_source(source, "test cell")


@pytest.mark.parametrize(
    "source",
    (
        "vaft.omas.save(ods, temporary_path)",
        'Path("summary.txt").write_text(summary)',
        'h5py.File("local-read-only.h5", "r")',
    ),
)
def test_write_guard_allows_documentation_local_outputs(source):
    validate_source(source, "test cell")
