"""Import smoke test: every public namespace must be star-importable."""

import vaft
from vaft.process import *
from vaft.plot import *
from vaft.machine_mapping import *
from vaft.formula import *
from vaft.omas import *
from vaft.code import *
from vaft.database import *
from vaft.database.raw import *


def test_version_defined():
    assert vaft.__version__
