"""Global test configuration."""


def pytest_addoption(parser):
    parser.addoption(
        "--repeat",
        action="store",
        type=int,
        default=5,
        help="repeat count for the opt-in HSDS load benchmark (default: 5)",
    )


def pytest_collection_modifyitems(config, items):
    # omas.omas_machine defines a pytest-style ``test_machine_mapping_functions``
    # helper that leaks into test modules through ``from vaft.<pkg> import *``
    # chains; it needs omas-internal fixtures and must not be collected here.
    items[:] = [item for item in items if item.name != "test_machine_mapping_functions"]
