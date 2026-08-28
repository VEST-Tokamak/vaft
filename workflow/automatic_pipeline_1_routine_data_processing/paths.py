"""Product paths for pipeline 1, in either the legacy or the canonical layout.

Two layouts are supported through the ``layout`` config key:

``shot_first`` (default)
    The legacy server hierarchy, ``{base_dir}/{shot}/{area}/...``, which matches
    the reference output at ``/srv/vest.filedb/public`` path for path and keeps
    this pipeline directly diffable against it.

``filedb``
    The canonical OMAS-first grammar from issue #77, resolved through
    :class:`vaft.database.filedb.FileDB` so no path is reconstructed by hand.

Only the Snakefile imports this module; the stage scripts receive explicit
``--output`` paths and stay layout-agnostic.
"""

from __future__ import annotations

from pathlib import Path

from vaft.database.filedb import FileDB, GPECCode


SHOT_FIRST = "shot_first"
FILEDB = "filedb"
LAYOUTS = (SHOT_FIRST, FILEDB)

# `vaft.code.gpec`'s module key for ideal-GPEC is the literal executable name
# "gpec"; FileDB's `GPECCode` enum spells the same code "ideal-gpec" to keep
# it unambiguous next to "gpec" the whole-suite package. Translated here, at
# the workflow-script boundary, rather than renaming either side.
_GPEC_CODE_ALIASES = {"gpec": GPECCode.IDEAL_GPEC}


def _gpec_code(code: str):
    return _GPEC_CODE_ALIASES.get(code, code)


# Substituted back into Snakemake wildcards after a concrete path is resolved.
# FileDB validates shot numbers and version components, so a wildcard token can
# not be passed through the resolver directly.
_SHOT_SENTINEL = 987654321
_VERSION_SENTINEL = "VESTMACHINEVERSIONSENTINEL"
_MODE_SENTINEL = 123456789
# A real, valid `GPECCode` value used purely as a substitution placeholder --
# unlike shot/version/mode, `FileDB.gpec()` validates its `code` argument
# against the `GPECCode` enum, so an arbitrary sentinel string is rejected.
_CODE_SENTINEL = "dcon"


# Which FileDB domain owns each stage's log.
_LOG_OWNER = {
    "generate_raw_db_dump": ("raw", None),
    "generate_static_ods": ("omas", "static"),
    "generate_diagnostics_ods": ("omas", "diagnostics"),
    "generate_eddy_ods": ("omas", "eddy"),
    "generate_constraints_ods": ("omas", "efit"),
    "generate_kfile": ("efit", None),
    "run_efit": ("efit", None),
    "generate_efit_ods": ("omas", "efit"),
    "plot_raw": ("raw", None),
    "plot_diagnostics": ("omas", "diagnostics"),
    "plot_eddy": ("omas", "eddy"),
    "plot_mhd_linear": ("omas", "mhd_linear"),
    "plot_efit": ("efit", None),
    "run_chease": ("chease", None),
    "generate_chease_ods": ("omas", "chease"),
    "plot_chease": ("omas", "chease"),
    "run_gpec_suite": ("omas", "mhd_linear"),
    "build_mhd_linear": ("omas", "mhd_linear"),
}


class PipelinePaths:
    """Resolve every pipeline 1 product for the configured layout."""

    def __init__(self, base_dir: str, layout: str = SHOT_FIRST) -> None:
        if layout not in LAYOUTS:
            raise ValueError(
                f"Unknown layout {layout!r}; expected one of: {', '.join(LAYOUTS)}"
            )
        self.base_dir = str(base_dir).rstrip("/")
        self.layout = layout
        self._filedb = FileDB(self.base_dir) if layout == FILEDB else None

    @classmethod
    def from_config(cls, config) -> "PipelinePaths":
        return cls(config["base_dir"], config.get("layout", SHOT_FIRST))

    # -- internal ---------------------------------------------------------
    def _shot_dir(self, shot, area: str) -> Path:
        return Path(self.base_dir) / str(shot) / area

    def _omas(self, stage: str, shot, artifact: str) -> Path:
        return self._filedb.omas(stage, shot=shot, artifact=artifact)

    # -- raw --------------------------------------------------------------
    def raw_dump(self, shot) -> str:
        """The canonical raw DAQ dump.

        The shot number stays in the file name in both layouts so the preflight
        can keep cross-checking it against the payload.
        """
        name = f"vest_{shot}_daq_raw.json.gz"
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "diagnostics") / name)
        return str(self._filedb.raw(shot) / name)

    def raw_manifest(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "metadata") / "raw_manifest.json")
        return str(self._filedb.raw(shot) / f"vest_{shot}_daq_manifest.json")

    # -- static machine era ------------------------------------------------
    def static_ods(self, machine_version) -> str:
        if self.layout == SHOT_FIRST:
            # Mirrors the legacy `static_file_dir`, one directory per era.
            return str(
                Path(self.base_dir) / "static" / str(machine_version) / "static.json"
            )
        directory = self._filedb.omas(
            "static", machine_version=str(machine_version), artifact="output"
        )
        return str(directory / "static.json")

    def static_manifest(self, machine_version) -> str:
        if self.layout == SHOT_FIRST:
            return str(
                Path(self.base_dir) / "static" / str(machine_version) / "manifest.json"
            )
        directory = self._filedb.omas(
            "static", machine_version=str(machine_version), artifact="metadata"
        )
        return str(directory / "manifest.json")

    # -- OMAS stage products ----------------------------------------------
    def diagnostics_ods(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "omas") / f"{shot}_diagnostics.json")
        return str(self._omas("diagnostics", shot, "output") / "diagnostics.json")

    def diagnostics_manifest(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "metadata") / "diagnostics_manifest.json")
        return str(self._omas("diagnostics", shot, "metadata") / "manifest.json")

    def eddy_ods(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "omas") / f"{shot}_eddy.json")
        return str(self._omas("eddy", shot, "output") / "eddy.json")

    def eddy_manifest(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "metadata") / "eddy_manifest.json")
        return str(self._omas("eddy", shot, "metadata") / "manifest.json")

    def constraints_ods(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "omas") / f"{shot}_constraints.json")
        # Issue #77 places EFIT constraints under `omas/efit/{shot}/work`.
        return str(self._omas("efit", shot, "work") / "constraints.json")

    def efit_ods(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "omas") / f"{shot}_efit.json")
        return str(self._omas("efit", shot, "output") / "efit.json")

    def chease_ods(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "omas") / f"{shot}_chease.json")
        return str(self._omas("chease", shot, "output") / "chease.json")

    # -- external code artifacts -------------------------------------------
    def kfile_manifest(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "efit") / "kfile" / "kfiles_generated.txt")
        return str(self._filedb.efit(shot, artifact="input") / "kfiles_generated.txt")

    def gfile_manifest(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "efit") / "gfile" / "gfiles_generated.txt")
        return str(self._filedb.efit(shot, artifact="output") / "gfiles_generated.txt")

    def efit_status(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "efit") / "efit_status.txt")
        return str(self._filedb.efit(shot, artifact="metadata") / "efit_status.txt")

    def efit_artifact_manifest(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "efit") / "artifact_manifest.json")
        return str(
            self._filedb.efit(shot, artifact="metadata") / "artifact_manifest.json"
        )

    def chease_refined(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "chease") / "refined_gfiles_generated.txt")
        return str(
            self._filedb.chease(shot, artifact="output")
            / "refined_gfiles_generated.txt"
        )

    def chease_status(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "chease") / "chease_status.txt")
        return str(self._filedb.chease(shot, artifact="metadata") / "chease_status.txt")

    def chease_runs(self, shot) -> str:
        # Written by run_chease_refinement.py next to `chease_refined`, i.e.
        # `output.parent / "chease_runs.json"` -- same directory both layouts
        # resolve for chease_refined above.
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "chease") / "chease_runs.json")
        return str(self._filedb.chease(shot, artifact="output") / "chease_runs.json")

    # -- linear stability ---------------------------------------------------
    # One status/manifest pair per (shot, code, mode): `run_gpec_module`
    # writes one, retriable independently of every other cell, matching
    # FileDB's own GPEC grammar (`gpec/{code}/{shot}/n={n}/`) directly.
    def gpec_module_status(self, shot, code: str, mode: int) -> str:
        if self.layout == SHOT_FIRST:
            return str(
                self._shot_dir(shot, "linear_stability")
                / code
                / f"n={mode}"
                / "status.txt"
            )
        return str(
            self._filedb.gpec(_gpec_code(code), shot, mode, artifact="metadata")
            / "status.txt"
        )

    def gpec_module_manifest(self, shot, code: str, mode: int) -> str:
        if self.layout == SHOT_FIRST:
            return str(
                self._shot_dir(shot, "linear_stability")
                / code
                / f"n={mode}"
                / "run.json"
            )
        return str(
            self._filedb.gpec(_gpec_code(code), shot, mode, artifact="output")
            / "run.json"
        )

    def gpec_workdir(self, shot, code: str | None = None, mode: int | None = None) -> str:
        """Working directory for one independently rerunnable GPEC cell."""
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "linear_stability"))
        if code is None or mode is None:
            raise ValueError("code and mode are required for a canonical GPEC work directory")
        return str(
            self._filedb.gpec(_gpec_code(code), shot, mode, artifact="work")
        )

    def mhd_linear_ods(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "linear_stability") / "mhd_linear.json")
        return str(self._omas("mhd_linear", shot, "output") / "mhd_linear.json")

    def mhd_linear_manifest(self, shot) -> str:
        if self.layout == SHOT_FIRST:
            return str(
                self._shot_dir(shot, "linear_stability") / "mhd_linear_manifest.json"
            )
        return str(self._omas("mhd_linear", shot, "metadata") / "manifest.json")

    # -- validation plots ---------------------------------------------------
    # Validation plots are a canonical FileDB artifact class (issue #139) and
    # have no legacy shot-first equivalent, so they are only resolvable under
    # the `filedb` layout -- as with `gpec_workdir`, asking for one in the
    # legacy layout is a configuration error rather than a silent fallback.
    def _require_filedb(self, product: str) -> None:
        if self.layout != FILEDB:
            raise ValueError(
                f"{product} is a canonical FileDB artifact and has no "
                f"{SHOT_FIRST!r} equivalent; set layout: {FILEDB} in config.yaml"
            )

    def stage_plot(self, shot, stage: str, filename: str) -> str:
        """One validation plot under ``omas/{stage}/{shot}/plot/``."""
        self._require_filedb("stage_plot")
        return str(self._omas(stage, shot, "plot") / filename)

    def stage_plot_manifest(self, shot, stage: str) -> str:
        self._require_filedb("stage_plot_manifest")
        return str(self._omas(stage, shot, "metadata") / "plot_manifest.json")

    def static_plot(self, machine_version, filename: str) -> str:
        """One validation plot for a machine era's static ODS."""
        self._require_filedb("static_plot")
        directory = self._filedb.omas(
            "static", machine_version=str(machine_version), artifact="plot"
        )
        return str(directory / filename)

    def static_plot_manifest(self, machine_version) -> str:
        self._require_filedb("static_plot_manifest")
        directory = self._filedb.omas(
            "static", machine_version=str(machine_version), artifact="metadata"
        )
        return str(directory / "plot_manifest.json")

    def raw_plot(self, shot, filename: str) -> str:
        """One raw-acquisition QA plot.

        Unlike the flat `raw/{shot}` archive itself, QA figures are a derived
        artifact and use the canonical `plot` class.
        """
        self._require_filedb("raw_plot")
        return str(self._filedb.resolve("raw", shot=shot) / "plot" / filename)

    def raw_plot_manifest(self, shot) -> str:
        self._require_filedb("raw_plot_manifest")
        return str(self._filedb.resolve("raw", shot=shot) / "plot" / "plot_manifest.json")

    def code_plot(self, shot, domain: str, filename: str) -> str:
        """One validation plot under ``efit/{shot}/plot`` or ``chease/{shot}/plot``."""
        self._require_filedb("code_plot")
        return str(self._filedb.resolve(domain, shot=shot, artifact="plot") / filename)

    def code_plot_dir(self, shot, domain: str) -> str:
        self._require_filedb("code_plot_dir")
        return str(self._filedb.resolve(domain, shot=shot, artifact="plot"))

    def chease_plot_manifest(self, shot) -> str:
        """Manifest of the per-time-slice CHEASE comparison figures."""
        self._require_filedb("chease_plot_manifest")
        return str(
            self._filedb.chease(shot, artifact="plot")
            / "plot_refined_gfiles_generated.txt"
        )

    # -- batch-level and per-shot ancillary ---------------------------------
    def log(self, shot, name: str) -> str:
        """Per-shot stage log.

        Under ``filedb`` the log lands in the ``log`` artifact of the domain that
        owns the stage, so a rule's log sits beside its own products.
        """
        if self.layout == SHOT_FIRST:
            return str(self._shot_dir(shot, "logs") / f"{name}.log")
        owner = _LOG_OWNER.get(name)
        if owner is None:
            raise ValueError(f"No canonical FileDB log owner registered for stage {name!r}")
        domain, stage = owner
        if domain == "raw":
            directory = self._filedb.raw(shot)
        elif domain == "omas":
            directory = self._omas(stage, shot, "log")
        else:
            directory = self._filedb.resolve(domain, shot=shot, artifact="log")
        return str(directory / f"{name}.log")

    def static_log(self, machine_version, name: str) -> str:
        if self.layout == SHOT_FIRST:
            return str(
                Path(self.base_dir) / "static" / str(machine_version) / f"{name}.log"
            )
        directory = self._filedb.omas(
            "static", machine_version=str(machine_version), artifact="log"
        )
        return str(directory / f"{name}.log")

    def preflight_eligible(self) -> str:
        if self.layout == SHOT_FIRST:
            return str(Path(self.base_dir) / "preflight" / "eligible_shots.json")
        return str(self._filedb.pipeline("preflight", artifact="metadata") / "eligible_shots.json")

    def preflight_excluded(self) -> str:
        if self.layout == SHOT_FIRST:
            return str(Path(self.base_dir) / "preflight" / "excluded_shots.json")
        return str(self._filedb.pipeline("preflight", artifact="metadata") / "excluded_shots.json")

    # -- Snakemake wildcard patterns ----------------------------------------
    def shot_pattern(self, product: str, *args) -> str:
        """Return ``product`` with the shot replaced by a ``{shot}`` wildcard."""
        resolved = getattr(self, product)(_SHOT_SENTINEL, *args)
        return resolved.replace(str(_SHOT_SENTINEL), "{shot}")

    def version_pattern(self, product: str, *args) -> str:
        """Return ``product`` with a ``{machine_version}`` wildcard."""
        resolved = getattr(self, product)(_VERSION_SENTINEL, *args)
        return resolved.replace(_VERSION_SENTINEL, "{machine_version}")

    def gpec_module_pattern(self, product: str) -> str:
        """Return a ``gpec_module_*`` product with ``{shot}``/``{code}``/``{mode}`` wildcards."""
        resolved = getattr(self, product)(
            _SHOT_SENTINEL, _CODE_SENTINEL, _MODE_SENTINEL
        )
        resolved = resolved.replace(str(_SHOT_SENTINEL), "{shot}")
        resolved = resolved.replace(str(_MODE_SENTINEL), "{mode}")
        # `_CODE_SENTINEL` ("dcon") is a real code name, so a blind substring
        # replace could also rewrite an unrelated occurrence elsewhere in the
        # path. `code` is always emitted as a whole path segment (never glued
        # to other text), so only swap segments that match it exactly.
        segments = [
            "{code}" if segment == _CODE_SENTINEL else segment
            for segment in Path(resolved).parts
        ]
        return str(Path(*segments))


__all__ = ["FILEDB", "LAYOUTS", "SHOT_FIRST", "PipelinePaths"]
