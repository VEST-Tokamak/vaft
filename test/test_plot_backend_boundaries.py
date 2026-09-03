"""Import boundaries of the extraction layer (issue #63).

``vaft.plot`` knows nothing of any data model; the backend imports no data
model at module level; a namespace never imports Matplotlib's pyplot just by
being imported.
"""

import subprocess
import sys

import pytest


def _modules_after(statement: str) -> set[str]:
    code = f"import sys; {statement}; print(' '.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_vaft_plot_knows_no_data_model():
    loaded = _modules_after("import vaft.plot")
    assert not {"vaft.plot.backend", "vaft.plot.backend.recipes", "vaft.omas", "vaft.imas", "omas", "imas"} & loaded


def test_the_backend_imports_no_data_model_at_module_level():
    loaded = _modules_after("import vaft.plot.backend.recipes, vaft.plot.backend.discovery, vaft.plot.backend.render")
    assert not {"omas", "imas", "vaft.omas", "vaft.imas"} & loaded


@pytest.mark.parametrize("namespace", ["vaft.omas", "vaft.imas", "vaft.database"])
def test_importing_a_namespace_does_not_import_pyplot(namespace):
    assert "matplotlib.pyplot" not in _modules_after(f"import {namespace}")
