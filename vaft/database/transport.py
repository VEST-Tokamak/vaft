"""Checked subprocess transport for the HSDS command-line tools.

The eager IMAS loaders still materialize complete domains with ``hsget`` and
``hsload``.  Keeping that behaviour behind one runner makes failures
actionable without changing the public eager APIs.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Sequence

import h5py
import numpy as np


logger = logging.getLogger(__name__)

try:
    import h5pyd
except ImportError:  # pragma: no cover - guarded when verification is requested
    h5pyd = None


class HSDSTransportError(RuntimeError):
    """Base error raised when an HSDS command cannot complete."""


class HSDSCommandNotFoundError(HSDSTransportError):
    """Raised when a required h5pyd CLI executable is not on ``PATH``."""


class HSDSCommandError(HSDSTransportError):
    """Raised when an HSDS CLI process exits unsuccessfully."""

    def __init__(self, command: str, exit_code: int, detail: str = "") -> None:
        self.command = command
        self.exit_code = exit_code
        self.detail = detail
        message = (
            f"{command} failed with exit code {exit_code}. "
            "Check HSDS connectivity, credentials, domain ACLs, and the remote URI."
        )
        if detail:
            message += f" Detail: {detail}"
        super().__init__(message)


class HSDSTransportVerificationError(HSDSTransportError):
    """Raised when an uploaded domain differs from its local HDF5 source."""


def _require_command(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise HSDSCommandNotFoundError(
            f"Required HSDS command '{command}' was not found on PATH. "
            "Install h5pyd with its command-line tools and ensure the active "
            "Python environment's bin/Scripts directory is on PATH."
        )
    return executable


def _remote_total_size(remote_uri: str) -> int | None:
    """Best-effort domain size lookup via ``hsstat``.

    Size reporting must never turn a valid transfer into a failure, so an old
    h5pyd installation, missing ``hsstat``, or an unsupported server response
    simply yields ``None``.
    """
    executable = shutil.which("hsstat")
    if executable is None:
        return None
    domain = remote_uri.removeprefix("hdf5://")
    if not domain.startswith("/"):
        domain = "/" + domain
    try:
        result = subprocess.run(
            [executable, domain],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"^\s*total_size:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def _log_progress(
    command: str,
    remote_uri: str,
    total_size: int | None,
    staging_path: str | Path,
    status: str,
) -> None:
    size_text = str(total_size) if total_size is not None else "unknown"
    logger.info(
        "[HSDS] command=%s remote=%s total_size=%s staging=%s status=%s",
        command,
        remote_uri,
        size_text,
        Path(staging_path),
        status,
    )


def _run(
    command: str,
    arguments: Sequence[str | Path],
    *,
    remote_uri: str,
    staging_path: str | Path,
    total_size: int | None,
) -> None:
    executable = _require_command(command)
    _log_progress(command, remote_uri, total_size, staging_path, "starting")
    try:
        result = subprocess.run(
            [executable, *(str(value) for value in arguments)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise HSDSTransportError(
            f"Could not start {command}: {exc}. Check executable permissions and PATH."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise HSDSCommandError(command, result.returncode, detail)
    _log_progress(command, remote_uri, total_size, staging_path, "complete")


def run_hsget(remote_uri: str, out_path: str | Path) -> Path:
    """Materialize one remote HSDS domain as a local HDF5 file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        "hsget",
        [remote_uri, path],
        remote_uri=remote_uri,
        staging_path=path,
        total_size=_remote_total_size(remote_uri),
    )
    return path


def run_hsload(local_path: str | Path, remote_uri: str) -> str:
    """Upload one local HDF5 file as an HSDS domain."""
    path = Path(local_path)
    size = path.stat().st_size if path.exists() else None
    _run(
        "hsload",
        [path, remote_uri],
        remote_uri=remote_uri,
        staging_path=path,
        total_size=size,
    )
    return remote_uri


def _normalized_attribute(value: object) -> object:
    array = np.asarray(value)
    if array.ndim == 0:
        value = array.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).rstrip(b"\x00").decode("utf-8")
    return value


def verify_uploaded_image(
    local_path: str | Path,
    remote_uri: str,
    *,
    h5pyd_module: object | None = None,
) -> None:
    """Compare critical IMAS metadata after ``hsload`` transcoding."""
    remote_api = h5pyd_module or h5pyd
    if remote_api is None:
        raise HSDSTransportVerificationError(
            "h5pyd is required to verify an uploaded HSDS domain"
        )

    mismatches: list[str] = []
    with h5py.File(Path(local_path), "r") as local, remote_api.File(remote_uri, "r") as remote:
        local_version = local.attrs.get("HDF5_BACKEND_VERSION")
        remote_version = remote.attrs.get("HDF5_BACKEND_VERSION")
        if _normalized_attribute(local_version) != _normalized_attribute(remote_version):
            mismatches.append("HDF5_BACKEND_VERSION")

        def compare_dataset(name: str, node: object) -> None:
            if not isinstance(node, h5py.Dataset):
                return
            try:
                uploaded = remote[name]
            except Exception:
                mismatches.append(f"missing dataset {name}")
                return
            if tuple(uploaded.shape) != tuple(node.shape):
                mismatches.append(f"shape {name}")
            if np.dtype(uploaded.dtype) != np.dtype(node.dtype):
                mismatches.append(f"dtype {name}")
            if getattr(uploaded, "compression", None) != node.compression:
                mismatches.append(f"compression {name}")
            if bool(getattr(uploaded, "shuffle", False)) != bool(node.shuffle):
                mismatches.append(f"shuffle {name}")

        local.visititems(compare_dataset)

        for name in local:
            link = local.get(name, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                continue
            try:
                uploaded_link = remote.get(name, getlink=True)
            except Exception:
                mismatches.append(f"external link {name}")
                continue
            if not hasattr(uploaded_link, "filename") or not hasattr(uploaded_link, "path"):
                mismatches.append(f"external link {name}")
                continue
            if (
                Path(uploaded_link.filename).name != Path(link.filename).name
                or str(uploaded_link.path).strip("/") != str(link.path).strip("/")
            ):
                mismatches.append(f"external link target {name}")

    if mismatches:
        raise HSDSTransportVerificationError(
            f"Uploaded domain {remote_uri} failed verification: "
            + ", ".join(sorted(set(mismatches)))
        )
    logger.info("[HSDS] remote=%s status=verified", remote_uri)
