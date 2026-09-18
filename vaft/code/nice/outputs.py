"""Collect NICE-native output and convert it to a canonical equilibrium ODS."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np

from .config import NiceConfig, NiceResult


def _last_row(path: Path) -> np.ndarray:
    data = np.loadtxt(path, ndmin=2)
    if not data.size:
        raise ValueError("empty file")
    return np.asarray(data[-1], float)


def _parse_residuals(
    path: Path, family: str, cocos_sign: float = 1.0
) -> list[dict[str, Any]]:
    row = _last_row(path)
    count = int(row[1])
    if row.size < 2 + 3 * count:
        raise ValueError(f"expected {2 + 3 * count} fields, found {row.size}")
    measured, computed, sigma_inv = np.split(row[2 : 2 + 3 * count], 3)
    return [
        {
            "family": family,
            "index": i,
            "measured": float(cocos_sign * m),
            "reconstructed": float(cocos_sign * c),
            "residual": float(cocos_sign * (c - m)),
            "normalized_residual": float(cocos_sign * (c - m) * s),
            "residual_definition": "(reconstructed - measured) / physical uncertainty",
        }
        for i, (m, c, s) in enumerate(zip(measured, computed, sigma_inv))
    ]


def _native_psi_2d(output: Path, psi_factor: float):
    """Interpolate NICE's nodal unstructured psi onto an IMAS R-Z grid."""
    coordinates = output / "dataEqui_mesh_coord.txt"
    fields = output / "dataEqui_psi_br_bz_btor.txt"
    if not coordinates.is_file() or not fields.is_file():
        return None
    points = np.loadtxt(coordinates, ndmin=2)
    row = _last_row(fields)
    if points.ndim != 2 or points.shape[1] < 2 or row.size < 2:
        raise ValueError("invalid native mesh/field dimensions")
    count = int(row[1])
    if count != points.shape[0] or row.size < 2 + 4 * count:
        raise ValueError(
            "native psi field count does not match dataEqui_mesh_coord.txt"
        )
    # dataEqui_psi_br_bz_btor is time, count, then four node-sized blocks.
    nodal_psi = psi_factor * np.asarray(row[2 : 2 + count], float)
    if not np.all(np.isfinite(points[:, :2])) or not np.all(np.isfinite(nodal_psi)):
        raise ValueError("non-finite native mesh or psi values")
    # NICE's native mesh is unstructured, whereas the canonical overview and
    # IMAS profiles_2d representation expect separable dim1/dim2 coordinates.
    # A 129-square visualization grid retains the native output resolution.
    from scipy.interpolate import griddata

    r = np.linspace(float(points[:, 0].min()), float(points[:, 0].max()), 129)
    z = np.linspace(float(points[:, 1].min()), float(points[:, 1].max()), 129)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = griddata(points[:, :2], nodal_psi, (rr, zz), method="linear")
    # The computational contour is not rectangular. Fill only the rectangular
    # corner cells needed by profiles_2d with nearest boundary-node values so
    # the canonical ODS remains finite; the stored plasma boundary identifies
    # the physically reconstructed region in the overview.
    missing = ~np.isfinite(psi)
    if np.any(missing):
        psi[missing] = griddata(
            points[:, :2], nodal_psi, (rr[missing], zz[missing]), method="nearest"
        )
    return r, z, psi


def _native_ods(output: Path, cocos: int, sign_ip: float = 1, sign_b0: float = 1):
    from omas import ODS

    if cocos != 11:
        raise ValueError(
            "standalone native output conversion currently supports COCOS 11 only"
        )
    # aux_inout.cpp ManageCOCOS: standalone tables remain NCIpp, unlike IMAS.
    psi_factor = -sign_ip * 2 * np.pi
    ods = ODS(consistency_check=False)
    errors: list[str] = []
    global_file = output / "dataEqui_global_quantities.txt"
    if not global_file.is_file():
        return None, [f"missing {global_file.name}"]
    row = _last_row(global_file)
    if row.size < 18:
        return None, [
            f"{global_file.name}: expected at least 18 fields, found {row.size}"
        ]
    t = float(row[0])
    ods["equilibrium.time"] = np.asarray([t])
    base = "equilibrium.time_slice.0"
    ods[f"{base}.time"] = t
    for path, value in {
        "global_quantities.beta_pol": row[3],
        "global_quantities.beta_tor": row[4],
        "global_quantities.beta_normal": row[5],
        "global_quantities.ip": row[6],
        "global_quantities.li_3": row[8],
        "global_quantities.volume": row[9],
        "global_quantities.area": row[10],
        "global_quantities.surface": row[11],
        "global_quantities.psi_axis": row[12],
        "global_quantities.psi_boundary": row[13],
        "global_quantities.magnetic_axis.r": row[14],
        "global_quantities.magnetic_axis.z": row[15],
        "global_quantities.q_axis": row[16],
        "global_quantities.q_95": row[17],
    }.items():
        factor = 1.0
        if path.endswith(("psi_axis", "psi_boundary")):
            factor = psi_factor
        elif path.endswith(".ip"):
            factor = sign_ip
        elif path.endswith(("q_axis", "q_95")):
            factor = sign_ip * sign_b0
        ods[f"{base}.{path}"] = float(value * factor)
    ods[f"{base}.global_quantities.magnetic_axis.b_field_tor"] = (
        sign_b0 * float(row[2]) * float(row[1]) / float(row[14])
    )
    ods["equilibrium.vacuum_toroidal_field.r0"] = float(row[1])
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.asarray([sign_b0 * float(row[2])])

    boundary = output / "dataEqui_plasma_boundary.txt"
    if boundary.is_file():
        b = _last_row(boundary)
        n, width = int(b[1]), int(b[2])
        if b.size >= 3 + 2 * width:
            ods[f"{base}.boundary.outline.r"] = b[3 : 3 + n]
            ods[f"{base}.boundary.outline.z"] = b[3 + width : 3 + width + n]
        else:
            errors.append(f"{boundary.name}: truncated boundary row")

    profiles = (
        output
        / "dataEqui_profiles_psi_rhotornorm_pressure_f_dpdpsi_fdfdpsi_jtor_q_Ne.txt"
    )
    if profiles.is_file():
        p = _last_row(profiles)
        n = int(p[1])
        names = (
            "psi",
            "rho_tor_norm",
            "pressure",
            "f",
            "dpressure_dpsi",
            "f_df_dpsi",
            "j_tor",
            "q",
            "electrons.density_thermal",
        )
        if p.size >= 2 + len(names) * n:
            for index, name in enumerate(names):
                factor = {
                    "psi": psi_factor,
                    "dpressure_dpsi": 1 / psi_factor,
                    "f_df_dpsi": 1 / psi_factor,
                    "f": sign_b0,
                    "j_tor": sign_ip,
                    "q": sign_ip * sign_b0,
                }.get(name, 1)
                ods[f"{base}.profiles_1d.{name}"] = (
                    factor * p[2 + index * n : 2 + (index + 1) * n]
                )
        else:
            errors.append(f"{profiles.name}: truncated profile row")
    try:
        field_2d = _native_psi_2d(output, psi_factor)
        if field_2d is not None:
            r, z, psi = field_2d
            ods[f"{base}.profiles_2d.0.grid.dim1"] = r
            ods[f"{base}.profiles_2d.0.grid.dim2"] = z
            ods[f"{base}.profiles_2d.0.psi"] = psi
    except Exception as exc:
        errors.append(f"native 2-D psi: {exc}")
    ods["equilibrium.code.name"] = "NICE"
    ods["equilibrium.code.library.0.name"] = "NICE"
    ods["equilibrium.code.parameters"] = json.dumps({"cocos": int(cocos)})
    return ods, errors


def collect_nice_outputs(
    workdir: str | Path, config: Optional[NiceConfig] = None
) -> NiceResult:
    """Rebuild a NICE result from disk without importing or executing NICE."""
    base = Path(workdir).expanduser()
    output = base / "output"
    files = (
        tuple(sorted(p for p in output.glob("*") if p.is_file()))
        if output.is_dir()
        else ()
    )
    logs = tuple(sorted(p for p in base.glob("*.log") if p.is_file()))
    manifest_file = base / "nice_case_manifest.json"
    provenance: dict[str, Any] = {}
    errors: list[str] = []
    if manifest_file.is_file():
        try:
            provenance = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{manifest_file.name}: {exc}")
    cocos = int(provenance.get("cocos_out", config.cocos_out if config else 11))
    try:
        ip_b0 = (
            np.loadtxt(base / "input" / "Ip_B0.txt")
            if (base / "input" / "Ip_B0.txt").is_file()
            else [1, 1]
        )
        ods, parse_errors = _native_ods(
            output, cocos, 1 if ip_b0[0] >= 0 else -1, 1 if ip_b0[1] >= 0 else -1
        )
        errors.extend(parse_errors)
        if ods is not None and "time_s" in provenance:
            # FillDataReconstructionFromFiles hard-codes time=0; restore the
            # physical slice from the immutable preparation manifest.
            ods["equilibrium.time"] = np.asarray([float(provenance["time_s"])])
    except Exception as exc:
        ods, errors = None, errors + [f"native equilibrium: {exc}"]

    history: list[dict[str, Any]] = []
    objective: dict[str, Any] = {}
    conv = output / "dataEqui_convergence_cost.txt"
    iterations = None
    if conv.is_file():
        try:
            rows = np.loadtxt(conv, ndmin=2)
            names = (
                "relative_residual",
                "total",
                "magnetic",
                "bpol",
                "flux_loop",
                "density",
                "polarimetry",
                "stokes",
                "pressure",
                "mse",
                "regularization_a",
                "positivity_a",
                "monotonicity_a",
                "regularization_b",
                "positivity_b",
                "regularization_density",
                "positivity_density",
                "boundary",
                "current_regularization",
            )
            for row in rows:
                item = {"time": float(row[0]), "iteration": int(row[1])}
                item.update({name: float(value) for name, value in zip(names, row[2:])})
                history.append(item)
            objective = {
                key: value
                for key, value in history[-1].items()
                if key not in {"time", "iteration"}
            }
            iterations = int(history[-1]["iteration"])
        except Exception as exc:
            errors.append(f"{conv.name}: {exc}")

    residuals: list[dict[str, Any]] = []
    input_ip = next(
        (
            row.get("value")
            for row in provenance.get("diagnostic_channels", ())
            if row.get("family") == "plasma_current" and row.get("enabled")
        ),
        1.0,
    )
    sign_ip = 1.0 if float(input_ip) >= 0 else -1.0
    for filename, family in (
        ("dataEqui_bp.txt", "bpol_probe"),
        ("dataEqui_fl.txt", "flux_loop"),
    ):
        path = output / filename
        if path.is_file():
            try:
                parsed = _parse_residuals(path, family)
                channels = [
                    row
                    for row in provenance.get("diagnostic_channels", ())
                    if row.get("family") == family and row.get("enabled")
                ]
                # The pinned new-manager standalone path leaves signBp/signF
                # uninitialized. Accept tables only if their measured column
                # verifies a single unit-sign mapping to the prepared input.
                if channels:
                    expected = np.asarray([c["value"] for c in channels])
                    measured = np.asarray([p["measured"] for p in parsed])
                    if len(expected) != len(measured):
                        raise ValueError("diagnostic output channel count mismatch")
                    signs = [
                        s
                        for s in (1, -1)
                        if np.allclose(s * measured, expected, rtol=2e-5, atol=1e-12)
                    ]
                    if len(signs) != 1:
                        raise ValueError(
                            "unverifiable native diagnostic output sign; pinned signBp/signF defect"
                        )
                    parsed = _parse_residuals(path, family, signs[0])
                for item, channel in zip(parsed, channels):
                    offset = channel.get("passive_response", 0) + channel.get(
                        "active_response_correction", 0
                    )
                    item["native_conditioned_measured"] = item["measured"]
                    item["measured"] += offset
                    item["reconstructed"] += offset
                    sigma = float(channel["uncertainty"])
                    item["normalized_residual"] = item["residual"] / sigma
                    item.update(
                        {
                            "identifier": channel.get("identifier"),
                            "ods_path": channel.get("ods_path"),
                            "physical_uncertainty": channel.get("uncertainty"),
                        }
                    )
                residuals.extend(parsed)
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
    ip_channel = next(
        (
            row
            for row in provenance.get("diagnostic_channels", ())
            if row.get("family") == "plasma_current" and row.get("enabled")
        ),
        None,
    )
    if ods is not None and ip_channel is not None:
        try:
            reconstructed_ip = float(
                ods["equilibrium.time_slice.0.global_quantities.ip"]
            )
            measured_ip = float(ip_channel["value"])
            uncertainty = float(ip_channel["uncertainty"])
            residuals.append(
                {
                    "family": "plasma_current",
                    "index": 0,
                    "identifier": ip_channel.get("identifier"),
                    "ods_path": ip_channel.get("ods_path"),
                    "measured": measured_ip,
                    "reconstructed": reconstructed_ip,
                    "residual": reconstructed_ip - measured_ip,
                    "normalized_residual": (reconstructed_ip - measured_ip)
                    / uncertainty,
                    "physical_uncertainty": uncertainty,
                    "residual_definition": "(reconstructed - measured) / physical uncertainty",
                }
            )
        except Exception as exc:
            errors.append(f"plasma-current residual: {exc}")

    stdout_file, stderr_file = base / "nice.stdout.log", base / "nice.stderr.log"
    stdout = (
        stdout_file.read_text(encoding="utf-8", errors="replace")
        if stdout_file.is_file()
        else ""
    )
    stderr = (
        stderr_file.read_text(encoding="utf-8", errors="replace")
        if stderr_file.is_file()
        else ""
    )
    mesh_invalid = "===== BUG" in stdout or any(
        a != b
        for a, b in re.findall(
            r"_nBoundaryInnerNodes=(\d+)\s+_nBoundaryInnerEdges=(\d+)", stdout
        )
    )
    if not history:
        for iteration, residual in re.findall(
            r"END iter=(\d+)\s+relresidX=([^\s]+)", stdout
        ):
            history.append(
                {"iteration": int(iteration), "relative_residual": float(residual)}
            )
        if history:
            iterations = history[-1]["iteration"]
            objective["relative_residual"] = history[-1]["relative_residual"]
        else:
            begun = re.findall(r"BEGIN iter=(\d+)", stdout)
            if begun:
                iterations = int(begun[-1])
                history.append({"iteration": iterations, "completed": False})
        for name, key in (("cost", "total"), ("costM", "magnetic")):
            matches = re.findall(r"\b" + name + r"=([^\s]+)", stdout)
            if matches:
                objective[key] = float(matches[-1])
    invalid = bool(
        re.search(
            r"plasma[^\n]*(?:not valid|invalid)|non.?converg|"
            r"plasma valid\s*=\s*0|IsValid\(\)\s*=\s*0",
            stdout + "\n" + stderr,
            re.I,
        )
    )
    invalid = (
        invalid or mesh_invalid or any(not np.isfinite(v) for v in objective.values())
    )
    # VacTH-only files use -9e40 sentinels in the convergence table.  They are
    # finite, but they are not a completed nonlinear reconstruction residual.
    relative_residual = objective.get("relative_residual")
    invalid = invalid or (
        relative_residual is not None and float(relative_residual) < 0.0
    )
    invalid = invalid or bool(
        re.search(r"(?:=|\s)[+-]?(?:nan|inf)(?:\s|$)", stdout, re.I)
    )
    if ods is not None:
        invalid = invalid or any(
            not np.all(np.isfinite(v))
            for v in ods.flat().values()
            if isinstance(v, (float, int, np.ndarray))
        )
    produced = ods is not None
    converged = (
        (not invalid and produced and conv.is_file())
        if (produced or conv.is_file() or invalid)
        else None
    )
    reason = (
        "invalid plasma or non-convergence reported"
        if invalid
        else (
            "native equilibrium and convergence record produced"
            if converged
            else "outputs incomplete"
        )
    )
    if mesh_invalid:
        reason = "invalid computational mesh: boundary contour mismatch"
    if converged:
        tolerance = float(
            provenance.get("solver_tolerances", {}).get("epsStopRecon", 1e-8)
        )
        if objective.get("relative_residual", float("inf")) > tolerance:
            converged = False
            reason = "reconstruction tolerance not reached"
    active_audit = provenance.get("active_response_audit", [])
    initializer_residuals = []
    if active_audit:
        for family, prefix, sign in (
            ("bpol_probe", "Bprobes", sign_ip),
            (
                "flux_loop",
                "fluxloops",
                -sign_ip * float(provenance.get("flux_loop_input_sign", 1)),
            ),
        ):
            try:
                measured = sign * np.loadtxt(
                    output / f"vacth_{prefix}_meas.txt", ndmin=1
                )
                computed = sign * np.loadtxt(
                    output / f"vacth_{prefix}_comp.txt", ndmin=1
                )
                channels = [
                    r
                    for r in provenance.get("diagnostic_channels", ())
                    if r["family"] == family and r["enabled"]
                ]
                expected = np.asarray([r["value"] for r in channels])
                if measured.shape != expected.shape or not np.allclose(
                    measured, expected, rtol=2e-5, atol=1e-10
                ):
                    raise ValueError(
                        "effective input conversion does not match canonical diagnostics"
                    )
                for m, c, channel in zip(measured, computed, channels):
                    offset = channel.get("passive_response", 0) + channel.get(
                        "active_response_correction", 0
                    )
                    initializer_residuals.append(
                        {
                            "family": family,
                            "ods_path": channel["ods_path"],
                            "stage": "VacTH initializer, NOT final equilibrium",
                            "measured": float(m + offset),
                            "reconstructed": float(c + offset),
                            "residual": float(c - m),
                            "normalized_residual": float(
                                (c - m) / channel["uncertainty"]
                            ),
                        }
                    )
            except Exception as exc:
                errors.append(f"initializer diagnostic verification: {exc}")
    provenance["initializer_residuals"] = initializer_residuals
    if active_audit:
        verification = []
        for family, prefix, sign in (
            ("bpol_probe", "Bprobes", sign_ip),
            ("flux_loop", "fluxloops", sign_ip),
        ):
            try:
                computed = np.loadtxt(output / f"vacth_{prefix}_comp.txt", ndmin=1)
                plasma = np.loadtxt(
                    output / f"vacth_{prefix}_comp_minus_pfcoils.txt", ndmin=1
                )
                native = sign * (computed - plasma)
                rows = [r for r in active_audit if r["family"] == family]
                if len(rows) != len(native):
                    raise ValueError("native coil audit channel count mismatch")
                for row, value in zip(rows, native):
                    error = abs(value + row["correction"] - row["exact"])
                    verification.append(
                        {
                            "ods_path": row["ods_path"],
                            "error": float(error),
                            "fraction_of_sigma": float(error / row["uncertainty"]),
                        }
                    )
                if any(r["fraction_of_sigma"] >= 0.1 for r in verification):
                    errors.append("active coil response mismatch exceeds 0.1 sigma")
            except Exception as exc:
                errors.append(f"active response verification: {exc}")
        provenance["active_response_verification"] = verification
    returncode = provenance.get("process_returncode")
    process_succeeded = returncode == 0 if returncode is not None else False
    # Issue #666 stage acceptance deliberately means final numerical
    # convergence, not equilibrium-quality acceptance.  Auxiliary diagnostic
    # tables can be unusable (notably the pinned signBp/signF output defect)
    # while the equilibrium and convergence record remain valid.  Preserve
    # those diagnostics in parsing_errors for the later quality gate, but do
    # not turn them into a false numerical failure here.
    usable = bool(process_succeeded and converged and ods is not None)
    if usable and errors:
        reason = "numerically converged; auxiliary output warnings recorded"
    return NiceResult(
        returncode,
        base,
        process_succeeded,
        converged,
        usable,
        reason,
        iterations,
        objective,
        tuple(history),
        tuple(residuals),
        files,
        logs,
        stdout,
        stderr,
        tuple(errors),
        provenance,
        ods,
    )
