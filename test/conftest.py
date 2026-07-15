"""Global test configuration.

The suite must be runnable outside the VEST intranet, so force the raw-DB
layer into offline mode (zero-waveform fallbacks) unless the invoker
explicitly overrides it, e.g. ``VAFT_RAW_OFFLINE_ONLY=0 pytest test/`` on a
machine with live MySQL access.
"""

import os

os.environ.setdefault("VAFT_RAW_OFFLINE_ONLY", "1")


def pytest_collection_modifyitems(config, items):
    # omas.omas_machine defines a pytest-style ``test_machine_mapping_functions``
    # helper that leaks into test modules through ``from vaft.<pkg> import *``
    # chains; it needs omas-internal fixtures and must not be collected here.
    items[:] = [item for item in items if item.name != "test_machine_mapping_functions"]
