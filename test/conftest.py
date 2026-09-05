"""Global test configuration."""

import sys
from pathlib import Path

import pytest

# test/ is not a package, so the sibling module holding the develop gate's
# selection is not importable by name until its directory is on the path. The
# same shape as test/contracts/conftest.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_selection import core_paths  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--repeat",
        action="store",
        type=int,
        default=5,
        help="repeat count for the opt-in HSDS load benchmark (default: 5)",
    )


# `-m` filtering is itself a pytest_collection_modifyitems implementation, so
# the marker has to be attached before it runs or `-m core` would select
# nothing. pluggy calls conftest hooks ahead of the builtin plugins already;
# tryfirst says so out loud rather than relying on registration order, and
# test_core_selection.py collects `-m core` in a subprocess to prove it.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # omas.omas_machine defines a pytest-style ``test_machine_mapping_functions``
    # helper that leaks into test modules through ``from vaft.<pkg> import *``
    # chains; it needs omas-internal fixtures and must not be collected here.
    items[:] = [item for item in items if item.name != "test_machine_mapping_functions"]

    # `core` is what the develop gate runs. It is applied here, from the one
    # declared list in test/core_selection.py, so that the whole gate is
    # reviewable in a single diff instead of being forty scattered pytestmark
    # lines nobody can see the shape of.
    core = set(core_paths())
    for item in items:
        if Path(str(item.fspath)).resolve() in core:
            item.add_marker(pytest.mark.core)
