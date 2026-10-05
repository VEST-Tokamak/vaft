"""Is a usable MITIM installation configured? (#1588 section 3).

The probe runs *in the MITIM interpreter*, as a subprocess, so asking the
question never imports MITIM into VAFT. It reports one status out of an ordered
list, the first failing check winning, plus everything it could identify.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import MITIM_PYTHON_ENV, SUPPORTED_MITIM_VERSIONS, MITIMConfig, mitim_user_config

__all__ = ["PROBE", "STATUSES", "MITIMAvailability", "mitim_availability"]

#: Statuses in the order the checks run; ``ready`` only when every check passed.
STATUSES = (
    "configuration_missing",
    "probe_timeout",
    "not_installed",
    "unsupported_version",
    "portals_unavailable",
    "gacode_unavailable",
    "ready",
)

#: Runs inside the MITIM interpreter. It prints one JSON document and never
#: raises: a failed import is a reported fact, not a crash.
PROBE = r"""
import json, platform, subprocess, sys
out = {"python": sys.executable, "python_version": platform.python_version()}
try:
    import mitim_tools
    out["version"] = getattr(mitim_tools, "__version__", None)
    root = getattr(mitim_tools, "__mitimroot__", None)
    out["root"] = None if root is None else str(root)
    import os.path
    out["templates"] = bool(root) and os.path.isfile(os.path.join(str(root), "templates", "input.neo.controls"))
    try:
        rev = subprocess.run(["git", "-C", out["root"], "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        out["revision"] = rev.stdout.strip() or None
    except Exception:
        out["revision"] = None
except Exception as error:
    out["import_error"] = f"{type(error).__name__}: {error}"
try:
    import mitim_modules.portals.PORTALSmain  # noqa: F401
    out["portals"] = True
except Exception as error:
    out["portals"] = False
    out["portals_error"] = f"{type(error).__name__}: {error}"
print(json.dumps(out))
"""


@dataclass(frozen=True)
class MITIMAvailability:
    """What the probe found. ``status`` is one of :data:`STATUSES`."""

    status: str
    detail: str = ""
    python: Optional[str] = None
    python_version: Optional[str] = None
    version: Optional[str] = None
    revision: Optional[str] = None
    root: Optional[str] = None
    portals: bool = False
    gacode_root: Optional[str] = None
    gacode_platform: Optional[str] = None
    gacode_revision: Optional[str] = None
    supported_versions: tuple[str, ...] = field(default=SUPPORTED_MITIM_VERSIONS)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["supported_versions"] = list(self.supported_versions)
        return record


def _gacode(config: MITIMConfig) -> dict[str, Optional[str]]:
    """The GACODE build MITIM will run, resolved the way VAFT's own adapters resolve it."""
    from ..gacode._runtime import find_gacode_executable, gacode_home, gacode_platform
    from ..gacode.cgyro.runner import gacode_revision

    try:
        home = gacode_home(config.gacode)
        found = {code: find_gacode_executable(config.gacode, code) for code in ("tglf", "neo")}
    except (FileNotFoundError, PermissionError) as error:
        return {"error": str(error)}
    if home is None or any(path is None for path in found.values()):
        return {"error": "no GACODE installation is configured ($GACODEHOME)"}
    try:
        platform_tag = gacode_platform(config.gacode, home=home)
    except (FileNotFoundError, ValueError) as error:
        return {"error": str(error)}
    return {"root": str(home), "platform": platform_tag,
            "revision": gacode_revision(config.gacode)}


def mitim_availability(config: MITIMConfig | None = None, *, timeout: float = 300.0) -> MITIMAvailability:
    """Probe the configured MITIM interpreter and the GACODE build it would use.

    Parameters
    ----------
    config : MITIMConfig, optional
        Names the interpreter (else ``$VAFT_MITIM_PYTHON``) and the GACODE build.
    timeout : float
        Limit on the probe itself [s]; importing MITIM loads tensorflow and torch.

    Returns
    -------
    MITIMAvailability
        ``status`` is the first failing check of :data:`STATUSES`, or ``ready``.
    """
    config = config or MITIMConfig()
    python = config.interpreter()
    if python is None:
        return MITIMAvailability(
            "configuration_missing",
            f"no MITIM interpreter: set ${MITIM_PYTHON_ENV} or MITIMConfig.python "
            "(install/install_mitim.sh prints it)")
    if not Path(python).is_file():
        return MITIMAvailability("not_installed", f"{python} does not exist", python=python)
    # MITIM reads $MITIM_CONFIG while it is being imported and fails without one, so
    # the probe gets a throwaway per-run configuration; the user site is excluded so
    # a ~/.local package can never stand in for the isolated environment's own.
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vaft-mitim-probe-") as scratch:
        config_path = Path(scratch) / "mitim_config.json"
        config_path.write_text(json.dumps(mitim_user_config(config, scratch)))
        # The caller's PYTHONPATH/PYTHONHOME would put VAFT-side (other-version)
        # packages ahead of MITIM's own; only an explicit config.env value passes.
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("PYTHONPATH", "PYTHONHOME")}
        environment.update({"MITIM_CONFIG": str(config_path), "PYTHONNOUSERSITE": "1",
                            **{str(k): str(v) for k, v in config.env.items()}})
        try:
            completed = subprocess.run([python, "-c", PROBE], capture_output=True, text=True,
                                       timeout=timeout, check=False, env=environment, cwd=scratch)
        except subprocess.TimeoutExpired:
            completed = None
        except OSError as error:
            return MITIMAvailability("not_installed", f"{python} cannot be run: {error}",
                                     python=python)
    if completed is None:
        # Importing tensorflow/torch on a loaded node can be slow; that is not a
        # broken installation, and is reported as what it is.
        return MITIMAvailability("probe_timeout", f"the probe did not finish in {timeout:g} s",
                                 python=python)
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or not lines:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-1:] or [""]
        return MITIMAvailability("not_installed", f"the probe failed: {tail[0]}", python=python)
    found = json.loads(lines[-1])
    common = dict(python=found.get("python") or python, python_version=found.get("python_version"),
                  version=found.get("version"), revision=found.get("revision"),
                  root=found.get("root"), portals=bool(found.get("portals")))
    if "import_error" in found:
        return MITIMAvailability("not_installed", found["import_error"], **common)
    if found.get("version") not in SUPPORTED_MITIM_VERSIONS:
        return MITIMAvailability(
            "unsupported_version",
            f"MITIM {found.get('version')} is not in the supported set {SUPPORTED_MITIM_VERSIONS}",
            **common)
    if not found.get("templates"):
        # A wheel install leaves __mitimroot__ in site-packages, without templates/,
        # and MITIM then fails on its first prep() (input.neo.controls).
        return MITIMAvailability(
            "not_installed",
            f"MITIM at {found.get('root')} has no templates/: install it editable from a "
            "checkout of the release tag (install/install_mitim.sh)", **common)
    if not found.get("portals"):
        return MITIMAvailability("portals_unavailable", found.get("portals_error", ""), **common)
    gacode = _gacode(config)
    if "error" in gacode:
        return MITIMAvailability("gacode_unavailable", gacode["error"], **common)
    return MITIMAvailability("ready", "", gacode_root=gacode["root"],
                             gacode_platform=gacode["platform"],
                             gacode_revision=gacode["revision"], **common)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(json.dumps(mitim_availability().as_dict(), indent=2))
    sys.exit(0)
