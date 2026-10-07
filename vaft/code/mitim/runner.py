"""Launch a MITIM driver script in the isolated MITIM interpreter (#1588 stage A1).

The runner owns the run directory, the per-run ``$MITIM_CONFIG``, the GACODE
environment, the launch through VAFT's execution backend, the driver's
``result.json`` and a ``record.json`` that carries the provenance. A stop by a
time, queue or memory limit is a result (``runtime_status``), not an exception.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional

from ..execution import ExecutionRequest, resolve_backend, timeout_reason
from .availability import MITIMAvailability, mitim_availability
from .config import MITIMConfig, MITIMResult, mitim_user_config

__all__ = [
    "mitim_tglf_local_inputs",
    "run_mitim_driver",
    "run_mitim_neo",
    "run_mitim_tglf",
    "run_neo_smoke",
]

#: How far the r/a a MITIM run used (RMIN_LOC / RMIN_OVER_A it wrote) may sit from
#: the r/a requested. The comparisons match VAFT's own grid at 1e-4, so a looser
#: guard would let a shifted surface surface as a physics difference.
_RAN_TOLERANCE = 1e-4

#: The GACODE members MITIM may call; each one's ``bin`` goes on ``PATH``.
_GACODE_MEMBERS = ("neo", "tglf", "tgyro", "cgyro", "vgen")


def _environment(config: MITIMConfig, config_path: Path) -> dict[str, str]:
    from ..gacode._runtime import gacode_environment, gacode_home

    environment = {key: value for key, value in gacode_environment(config.gacode, "neo").items()
                   if key not in os.environ or os.environ[key] != value}
    home = gacode_home(config.gacode)
    if home is not None:
        extra = [str(home / member / "bin") for member in _GACODE_MEMBERS
                 if (home / member / "bin").is_dir()]
        path = gacode_environment(config.gacode, "neo").get("PATH", "")
        environment["PATH"] = os.pathsep.join([*extra, path]).rstrip(os.pathsep)
    environment["MITIM_CONFIG"] = str(config_path)
    # The isolated interpreter must not pick up the caller's user site; the
    # caller's PYTHONPATH/PYTHONHOME are removed by the command (see _command).
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update({str(k): str(v) for k, v in config.env.items()
                        if k not in ("PYTHONPATH", "PYTHONHOME")})
    return environment


def _command(config: MITIMConfig, python: str, *arguments: str) -> tuple[str, ...]:
    """Launch ``python`` with the caller's PYTHONPATH/PYTHONHOME removed.

    The execution backend overlays its ``env`` on the launching environment and
    cannot remove a variable, and a VAFT interpreter's PYTHONPATH (a worktree, a
    3.14 site-packages) would put wrong-version packages ahead of MITIM's. ``env -u``
    removes them; a value the caller set in ``config.env`` is passed explicitly.
    """
    explicit = [f"{key}={config.env[key]}" for key in ("PYTHONPATH", "PYTHONHOME") if key in config.env]
    if "PYTHONPATH" not in config.env:
        # GACODE's own launchers parse their inputs with pygacode, and VAFT's GACODE
        # environment puts $GACODE_ROOT/f2py on PYTHONPATH for exactly that; without it
        # the launcher's parse step fails silently and NEO writes no flux files (tdst,
        # 2026-10-06). Those entries are GACODE's, not the caller's, so they stay.
        from ..gacode._runtime import gacode_home

        home = gacode_home(config.gacode)
        if home is not None:
            explicit.append("PYTHONPATH=" + os.pathsep.join(
                [str(home / "f2py"), str(home / "f2py" / "pygacode")]))
    return ("env", "-u", "PYTHONPATH", "-u", "PYTHONHOME", *explicit, python, *arguments)


def _report_partial(result: MITIMResult, missing: list, shifted: Mapping[float, Any]) -> None:
    """Downgrade a run with unusable surfaces to ``partial``, on disk as in memory.

    ``result.json`` (the driver's own verdict) and ``record.json`` (the
    provenance) are rewritten with the new status, the surfaces without a usable
    result and those MITIM ran elsewhere, so a later reader of the run directory
    sees what the caller was told.
    """
    detail = {"status": "partial", "missing_r_over_a": list(missing),
              "shifted_r_over_a": dict(shifted)}
    result.result = {**(result.result or {}), **detail}
    result.record = {**result.record, "result_status": "partial",
                     "missing_r_over_a": detail["missing_r_over_a"],
                     "shifted_r_over_a": detail["shifted_r_over_a"]}
    (result.workdir / "result.json").write_text(
        json.dumps(result.result, indent=1, default=str), encoding="utf-8")
    (result.workdir / "record.json").write_text(
        json.dumps(result.record, indent=1, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_mitim_driver(
    driver: str,
    arguments: Mapping[str, Any],
    workdir: str | Path,
    config: MITIMConfig | None = None,
    *,
    availability: Optional[MITIMAvailability] = None,
) -> MITIMResult:
    """Run one packaged driver (``vaft/code/mitim/drivers/<driver>.py``) in ``workdir``.

    Parameters
    ----------
    driver : str
        Driver module name, e.g. ``"neo_smoke"``.
    arguments : mapping
        Written to ``arguments.json`` and handed to the driver.
    workdir : path
        Run directory; created if absent. The driver's previous ``result.json`` is
        removed first so an old success is never reported as this run's.
    config : MITIMConfig, optional
    availability : MITIMAvailability, optional
        A probe already taken; otherwise one is taken here. A run is refused
        (``FileNotFoundError``) unless the status is ``ready``.

    Returns
    -------
    MITIMResult
    """
    config = config or MITIMConfig()
    availability = availability or mitim_availability(
        config, timeout=max(300.0, float(config.timeout or 0.0)))
    if not availability.ready:
        raise FileNotFoundError(f"MITIM is not ready ({availability.status}): {availability.detail}")
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    source = resources.files("vaft.code.mitim.drivers").joinpath(f"{driver}.py")
    if not source.is_file():
        raise ValueError(f"no MITIM driver named {driver!r}")
    script = workdir / f"{driver}.py"
    script.write_text(source.read_text())
    (workdir / "result.json").unlink(missing_ok=True)
    argument_path = workdir / "arguments.json"
    argument_path.write_text(json.dumps(dict(arguments), indent=1, default=str))
    config_path = workdir / "mitim_config.json"
    config_path.write_text(json.dumps(mitim_user_config(config, workdir), indent=1))
    (workdir / "mitim_scratch").mkdir(exist_ok=True)

    execution = resolve_backend(config).run(
        ExecutionRequest(
            command=_command(config, availability.python, script.name, argument_path.name),
            workdir=workdir,
            env=_environment(config, config_path),
            timeout=config.timeout,
            label=f"mitim-{driver}",
        )
    )
    result = None
    result_path = workdir / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError as error:
            result = {"status": "error", "error": f"unreadable result.json: {error}"}
    stderr = execution.stderr
    if execution.timed_out:
        reason = timeout_reason("MITIM", execution, config.timeout)
        stderr = f"{stderr}\n{reason}" if stderr else reason
    record = {
        "driver": driver,
        "driver_sha256": _sha256(script),
        "arguments": json.loads(argument_path.read_text()),
        "mitim": availability.as_dict(),
        "mitim_config": json.loads(config_path.read_text()),
        "mitim_config_sha256": _sha256(config_path),
        "returncode": execution.returncode,
        "runtime_status": execution.runtime_status,
        "elapsed_s": execution.elapsed_s,
        "job_id": getattr(execution, "job_id", None),
        "result_status": None if result is None else result.get("status"),
    }
    (workdir / "record.json").write_text(json.dumps(record, indent=1, default=str))
    return MITIMResult(
        returncode=execution.returncode,
        workdir=workdir,
        stdout=execution.stdout,
        stderr=stderr,
        runtime_status=execution.runtime_status,
        elapsed_s=execution.elapsed_s,
        result=result,
        record=record,
    )


def run_neo_smoke(
    profile: Any,
    rho_tor_norm,
    workdir: str | Path,
    config: MITIMConfig | None = None,
    *,
    code_settings: Optional[str] = None,
    availability: Optional[MITIMAvailability] = None,
) -> tuple[MITIMResult, list]:
    """The stage-A1 smoke capability: MITIM ``NEOtools`` on VAFT's own ``input.gacode``.

    ``profile`` is a :class:`~vaft.code.gacode._profiles.GACODEProfile` (e.g. from
    ``prepare_gacode_profile`` or a resolved transport state). It is written with
    VAFT's writer, run by MITIM at ``rho_tor_norm``, and every NEO directory MITIM
    produced is read back with VAFT's own :func:`collect_neo_outputs`, so the
    result discovery does not depend on MITIM's in-memory objects.

    The radii are ``rho_tor_norm``, *not* the ``r/a`` VAFT's own TGLF/NEO adapters
    take: MITIM's ``NEO(rhos=...)`` converts with ``r_is_rho=True``. On 48224,
    ``rho_tor_norm`` 0.5 and 0.7 ran at ``r/a`` 0.575 and 0.789; each parsed
    output carries the ``r_over_a`` NEO actually used.

    Returns
    -------
    (MITIMResult, list of NeoOutputs)
    """
    from ..gacode._input_gacode import write_input_gacode
    from ..gacode.neo.outputs import collect_neo_outputs

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    folder = workdir / "neo"
    if folder.exists():
        shutil.rmtree(folder)  # never read a previous run's NEO output as this run's
    input_path = write_input_gacode(profile, workdir / "input.gacode")
    arguments = {"input_gacode": str(input_path), "folder": str(folder),
                 "rhos": [float(r) for r in rho_tor_norm], "rhos_coordinate": "rho_tor_norm",
                 "code_settings": code_settings,
                 "input_gacode_sha256": _sha256(input_path)}
    result = run_mitim_driver("neo_smoke", arguments, workdir, config, availability=availability)
    outputs = []
    if result.result:
        for directory in result.result.get("run_directories", []):
            parsed = collect_neo_outputs(directory)
            if parsed is not None:
                outputs.append(parsed)
    return result, outputs


def mitim_tglf_local_inputs(
    profile: Any,
    r_over_a,
    workdir: str | Path,
    config: MITIMConfig | None = None,
    *,
    code_settings: str = "SAT3",
    availability: Optional[MITIMAvailability] = None,
) -> tuple[MITIMResult, dict[float, dict]]:
    """MITIM's own TGLF local inputs for ``profile`` at exactly these ``r/a``, not run.

    The profile is written with VAFT's writer, as in :func:`run_neo_smoke`, and MITIM's
    state converter is called with ``r_is_rho=False``, so no coordinate conversion sits
    between the two inputs being compared. ``result.result["rho_tor_norm"]`` is MITIM's
    own map of the same surfaces, to check against
    :func:`vaft.code.mitim.coordinates.rho_tor_norm_at`.

    Returns
    -------
    (MITIMResult, {r/a: parsed input.tglf})
    """
    from ..gacode._input_gacode import write_input_gacode
    from .compare import read_input_tglf

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    folder = workdir / "tglf_inputs"
    if folder.exists():
        shutil.rmtree(folder)
    input_path = write_input_gacode(profile, workdir / "input.gacode")
    arguments = {"input_gacode": str(input_path), "folder": str(folder),
                 "r_over_a": [float(r) for r in r_over_a], "code_settings": code_settings,
                 "input_gacode_sha256": _sha256(input_path)}
    result = run_mitim_driver("tglf_local_inputs", arguments, workdir, config,
                              availability=availability)
    parsed: dict[float, dict] = {}
    if result.ok:
        requested = arguments["r_over_a"]
        for label, path in result.result.get("files", {}).items():
            # Keyed by the r/a VAFT asked for, never by MITIM's label for it.
            nearest = min(requested, key=lambda r: abs(r - float(label)))
            if abs(nearest - float(label)) > 1e-6:
                raise ValueError(f"MITIM wrote a surface at {label}, which was not requested")
            parsed[nearest] = read_input_tglf(path)
    return result, parsed


def run_mitim_tglf(
    profile: Any,
    r_over_a,
    workdir: str | Path,
    config: MITIMConfig | None = None,
    *,
    code_settings: str = "SAT3",
    extra_options: Optional[Mapping[str, Any]] = None,
    availability: Optional[MITIMAvailability] = None,
) -> tuple[MITIMResult, dict[float, Any]]:
    """Run TGLF through MITIM at VAFT's ``r/a`` surfaces and read the results with VAFT.

    MITIM's ``TGLF(rhos=...)`` takes ``rho_tor_norm``, so the surfaces are converted
    once with :func:`vaft.code.mitim.coordinates.rho_tor_norm_at` (VAFT's bridge, from
    the same profile), and every result is keyed back by the ``r/a`` requested.
    ``extra_options`` are MITIM ``extraOptions``: individual ``input.tglf`` keys
    applied last, e.g. to align NKY/NMODES/USE_MHD_RULE with VAFT's defaults.

    Returns
    -------
    (MITIMResult, {r/a: TglfOutputs})
    """
    from ..gacode._input_gacode import write_input_gacode
    from ..gacode.tglf.outputs import collect_tglf_outputs
    from .coordinates import rho_tor_norm_at

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    folder = workdir / "tglf"
    if folder.exists():
        shutil.rmtree(folder)
    r_over_a = [float(r) for r in r_over_a]
    rho = [float(x) for x in rho_tor_norm_at(profile, r_over_a)]
    # MITIM names each radius's files by rho to 4 decimals; two surfaces sharing a
    # label would overwrite each other's outputs.
    labels = [f"{x:.4f}" for x in rho]
    if len(set(labels)) != len(labels):
        raise ValueError(f"r/a {r_over_a} map to rho_tor_norm labels {labels} that collide")
    input_path = write_input_gacode(profile, workdir / "input.gacode")
    arguments = {"input_gacode": str(input_path), "folder": str(folder), "rho_tor_norm": rho,
                 "r_over_a": r_over_a, "code_settings": code_settings,
                 "extra_options": dict(extra_options or {}),
                 "input_gacode_sha256": _sha256(input_path)}
    result = run_mitim_driver("tglf_run", arguments, workdir, config, availability=availability)
    outputs: dict[float, Any] = {}
    shifted: dict[float, float] = {}
    if result.ok:
        from .compare import read_input_tglf

        for directory in result.result.get("run_directories", []):
            label = Path(directory).name.split("_", 1)[1]
            if label not in labels:
                continue
            index = labels.index(label)
            # MITIM converts rho back to r/a with its own map; the surface it ran is
            # the RMIN_LOC it wrote, and a shifted surface is not the one requested.
            ran = read_input_tglf(Path(directory) / "input.tglf").get("RMIN_LOC") \
                if (Path(directory) / "input.tglf").is_file() else None
            if ran is None or abs(float(ran) - r_over_a[index]) > _RAN_TOLERANCE:
                shifted[r_over_a[index]] = None if ran is None else float(ran)
                continue
            parsed = collect_tglf_outputs(directory)
            if parsed is not None and parsed.gbflux is not None and parsed.solved:
                outputs[r_over_a[index]] = parsed
        missing = [r for r in r_over_a if r not in outputs]
        if missing:
            # Some surfaces have no usable result: say so rather than return a
            # successful run with fewer surfaces than were asked for.
            _report_partial(result, missing, shifted)
    return result, outputs


def run_mitim_neo(
    profile: Any,
    r_over_a,
    workdir: str | Path,
    config: MITIMConfig | None = None,
    *,
    code_settings: Optional[str] = None,
    extra_options: Optional[Mapping[str, Any]] = None,
    availability: Optional[MITIMAvailability] = None,
) -> tuple[MITIMResult, dict[float, Any], dict[float, dict]]:
    """Run NEO through MITIM (local mode, one radius each) at VAFT's ``r/a``.

    The radii go through VAFT's bridge, colliding 4-decimal rho labels are refused,
    and each radius's ``RMIN_OVER_A`` in the ``input.neo`` MITIM ran is checked
    against the request (1e-4); a shifted or missing surface makes the result
    ``partial``. MITIM's NEO preset is ``Sonic`` (ROTATION_MODEL=2); VAFT runs
    ROTATION_MODEL=1, so pass ``extra_options={"ROTATION_MODEL": 1}`` to compare.

    Returns
    -------
    (MITIMResult, {r/a: NeoOutputs}, {r/a: parsed input.neo})
        The outputs are in MITIM's local normalisation; see
        :func:`vaft.code.mitim.compare.neo_local_to_profile_normalisation`.
    """
    from ..gacode._input_gacode import write_input_gacode
    from ..gacode.neo.outputs import collect_neo_outputs
    from .compare import read_input_neo
    from .coordinates import rho_tor_norm_at

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    folder = workdir / "neo"
    if folder.exists():
        shutil.rmtree(folder)
    r_over_a = [float(r) for r in r_over_a]
    rho = [float(x) for x in rho_tor_norm_at(profile, r_over_a)]
    labels = [f"{x:.4f}" for x in rho]
    if len(set(labels)) != len(labels):
        raise ValueError(f"r/a {r_over_a} map to rho_tor_norm labels {labels} that collide")
    input_path = write_input_gacode(profile, workdir / "input.gacode")
    arguments = {"input_gacode": str(input_path), "folder": str(folder), "rhos": rho,
                 "rhos_coordinate": "rho_tor_norm", "r_over_a": r_over_a,
                 "code_settings": code_settings, "extra_options": dict(extra_options or {}),
                 "input_gacode_sha256": _sha256(input_path)}
    result = run_mitim_driver("neo_smoke", arguments, workdir, config, availability=availability)
    outputs: dict[float, Any] = {}
    inputs: dict[float, dict] = {}
    shifted: dict[float, Optional[float]] = {}
    if result.ok:
        for directory in result.result.get("run_directories", []):
            label = Path(directory).name.split("_", 1)[1]
            if label not in labels:
                continue
            index = labels.index(label)
            path = Path(directory) / "input.neo"
            parsed_input = read_input_neo(path) if path.is_file() else {}
            ran = parsed_input.get("RMIN_OVER_A")
            if ran is None or abs(float(ran) - r_over_a[index]) > _RAN_TOLERANCE:
                shifted[r_over_a[index]] = None if ran is None else float(ran)
                continue
            parsed = collect_neo_outputs(directory)
            # NEO exits 0 on input errors; only its own success flag makes a result.
            if parsed is not None and parsed.transport is not None and parsed.solved:
                outputs[r_over_a[index]] = parsed
                inputs[r_over_a[index]] = parsed_input
        missing = [r for r in r_over_a if r not in outputs]
        if missing:
            _report_partial(result, missing, shifted)
    return result, outputs, inputs
