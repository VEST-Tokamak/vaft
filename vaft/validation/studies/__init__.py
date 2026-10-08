"""Per-issue validation studies: scripts and notes, not library API (#1756).

Each folder records how one issue was investigated -- the scripts that were
run, the summaries that were written -- next to the validation layer they
exercise, instead of in a second top-level ``validation/`` directory with the
same name and a different role.  New issue studies go here, as
``vaft/validation/studies/<name>/`` (CONTRIBUTING, "Issue studies").

Rules the tests pin:

* :mod:`vaft.validation` never imports this subpackage, and the API catalog
  skips it (``scripts`` in ``docs/api_inventory.yml``): the scripts import
  optional solver toolkits at module level and must stay runnable without
  the documentation build importing them.
* Every folder is a regular package, so scripts run from an installed tree
  as ``python -m vaft.validation.studies.<name>.<script>`` instead of a
  ``PYTHONPATH=.`` path run, which imports whichever ``vaft`` is first on
  the path.
* A folder ships in the wheel only when its scripts run against the installed
  package with explicit arguments (``fixed_free_1608``); notes, run records,
  synthetic equilibria and inputs that are not in the repository never ship
  (``[tool.setuptools.packages.find] exclude``, ``MANIFEST.in``,
  ``test/verify_dist.py``).  A script that needs a file outside the package
  takes its path as a command-line argument rather than assuming a checkout.
"""
