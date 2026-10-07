"""The #891 study criteria, at the study's path.

The policy module itself is :mod:`vaft.validation._efit_study_criteria`, so a
wheel install can judge slices (``vaft.validation.equilibrium_quality`` loads
it by default); the study's scripts keep loading ``criteria.py`` from this
directory, and get the same module's names here.  Edit the policy there.
"""

from vaft.validation import _efit_study_criteria as _policy
from vaft.validation._efit_study_criteria import *  # noqa: F401,F403

CHECKS = _policy.CHECKS
CRITERIA_VERSION = _policy.CRITERIA_VERSION
__all__ = _policy.__all__
