"""OMAS ODS access backed by IMAS HDF5 images stored in HSDS."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
import subprocess
import tempfile
from typing import Optional, Union

import omas

try:
    import h5pyd
except ImportError:
    h5pyd = None  # optional: pip install h5pyd==0.20.0 --no-deps

from ..imas import load_omas_imas, save_omas_imas
from .utils import _require_h5pyd, is_connect


def _download_remote_image(remote_uri: str, out_path: Path) -> Path:
    """Download one HSDS HDF5 image to a local file via hsget."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["hsget", remote_uri, str(out_path)], capture_output=False, text=True, check=True)
    return out_path


def _download_remote_shot(directory: str, shot: int, shot_dir: Path) -> list[Path]:
    """Download every IMAS HDF5 image stored for one shot."""
    _require_h5pyd()

    try:
        entries = list(h5pyd.Folder(f"/{directory}/{shot}/"))
    except Exception as exc:
        raise FileNotFoundError(f"Could not open HSDS folder /{directory}/{shot}/") from exc

    h5_files = sorted(entry for entry in entries if entry.endswith(".h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 images found in /{directory}/{shot}/")

    downloaded = []
    for filename in h5_files:
        remote_uri = f"hdf5://{directory}/{shot}/{filename}"
        local_path = shot_dir / filename
        print(f"[INFO] Downloading {remote_uri} -> {local_path}")
        downloaded.append(_download_remote_image(remote_uri, local_path))
    return downloaded


def _upload_local_image(local_path: Path, remote_uri: str) -> str:
    """Upload one local HDF5 image file to HSDS."""
    subprocess.run(["hsload", str(local_path), remote_uri], capture_output=False, text=True, check=True)
    return remote_uri


def _upload_local_shot(shot_dir: Path, directory: str, shot: int) -> list[str]:
    """Upload every generated IMAS HDF5 image for one shot to HSDS."""
    h5_files = sorted(path for path in shot_dir.iterdir() if path.is_file() and path.suffix == ".h5")
    if not h5_files:
        raise FileNotFoundError(f"No IMAS HDF5 images were generated in {shot_dir}")

    uploaded = []
    for local_path in h5_files:
        remote_uri = f"hdf5://{directory}/{shot}/{local_path.name}"
        print(f"[INFO] Uploading {local_path} -> {remote_uri}")
        uploaded.append(_upload_local_image(local_path, remote_uri))
    return uploaded


def load_ods(
    shot: Union[int, list[int]],
    directory: str = "public",
    *,
    occurrence: Optional[dict] = None,
    paths: Optional[list] = None,
    time: Optional[float] = None,
    imas_version: Optional[str] = None,
    skip_uncertainties: bool = False,
    consistency_check: bool = True,
    verbose: bool = True,
    local_dir: Optional[Union[str, Path]] = None,
) -> Union[omas.ODS, list[omas.ODS]]:
    """
    Load ODS data by reading IMAS images from ``{directory}/{shot}`` and converting via OMAS.

    Args:
        shot: One shot number or a list of shot numbers.
        directory: HSDS folder containing shot subdirectories. Defaults to ``public``.
        occurrence: Optional IDS occurrence mapping forwarded to ``load_omas_imas``.
        paths: Optional IMAS paths to fetch.
        time: Optional time slice in seconds.
        imas_version: Optional IMAS DD version for conversion.
        skip_uncertainties: Skip uncertainty loading when ``True``.
        consistency_check: Run ODS consistency checks after conversion.
        verbose: Print IMAS loading progress.
        local_dir: Optional local staging base directory. When omitted a temporary directory
            is used and cleaned up automatically.

    Returns:
        One ``omas.ODS`` or a list of ``omas.ODS`` objects.
    """
    _require_h5pyd()
    logging.getLogger().setLevel(logging.WARNING)

    occurrence = occurrence or {}

    if isinstance(shot, list):
        ods_list = []
        for s in shot:
            ods = _load_one_shot(
                int(s),
                directory=directory,
                occurrence=occurrence,
                paths=paths,
                time=time,
                imas_version=imas_version,
                skip_uncertainties=skip_uncertainties,
                consistency_check=consistency_check,
                verbose=verbose,
                local_dir=local_dir,
            )
            print("Successfully loaded ODS data for shot:", s)
            ods_list.append(ods)
        print("Successfully loaded a list of ODS data")
        return ods_list

    s = int(shot)
    ods = _load_one_shot(
        s,
        directory=directory,
        occurrence=occurrence,
        paths=paths,
        time=time,
        imas_version=imas_version,
        skip_uncertainties=skip_uncertainties,
        consistency_check=consistency_check,
        verbose=verbose,
        local_dir=local_dir,
    )
    print("Successfully loaded ODS data for shot:", s)
    return ods


def _load_one_shot(
    shot: int,
    *,
    directory: str,
    occurrence: dict,
    paths: Optional[list],
    time: Optional[float],
    imas_version: Optional[str],
    skip_uncertainties: bool,
    consistency_check: bool,
    verbose: bool,
    local_dir: Optional[Union[str, Path]],
) -> omas.ODS:
    """Load one shot from HSDS IMAS images and convert it to ODS."""
    logging.getLogger().setLevel(logging.WARNING)

    staging_ctx = (
        tempfile.TemporaryDirectory(prefix="hsds_imas_ods_")
        if local_dir is None
        else nullcontext(str(local_dir))
    )
    with staging_ctx as staging_base:
        shot_dir = Path(staging_base) / str(shot)
        shot_dir.mkdir(parents=True, exist_ok=True)
        _download_remote_shot(directory=directory, shot=shot, shot_dir=shot_dir)

        ods = load_omas_imas(
            occurrence=occurrence,
            paths=paths,
            time=time,
            imas_version=imas_version,
            skip_uncertainties=skip_uncertainties,
            consistency_check=consistency_check,
            verbose=verbose,
            uri="imas:hdf5?path=" + str(shot_dir),
        )

        ods.setdefault("dataset_description.data_entry.user", str(directory))
        ods.setdefault("dataset_description.data_entry.pulse", int(shot))
        ods.setdefault("dataset_description.data_entry.run", 0)
        return ods


def save_ods(
    ods: omas.ODS,
    shot: int,
    filename: Optional[str] = None,
    env: str = "server",
    *,
    directory: str = "public",
    path: Optional[Union[str, Path]] = None,
    occurrence: Optional[dict] = None,
    user: Optional[str] = None,
    machine: Optional[str] = None,
    run: Optional[int] = None,
    imas_version: Optional[str] = None,
    verbose: bool = True,
) -> Optional[str]:
    """
    Save ODS data by converting it to IMAS HDF5 images.

    ``filename`` is kept only for compatibility with the old API and is ignored,
    because IMAS-backed storage produces multiple ``.h5`` files per shot.

    Args:
        ods: ODS object to save.
        shot: Shot number / IMAS pulse.
        filename: Ignored compatibility parameter.
        env: ``server`` uploads to HSDS, ``local`` only writes local IMAS files.
        directory: HSDS target folder for ``env="server"``. Defaults to ``public``.
        path: Local target directory for ``env="local"``.
        occurrence: Optional IDS occurrence mapping.
        user: Optional IMAS user metadata override.
        machine: Optional IMAS machine metadata override.
        run: Optional IMAS run number. Defaults to ODS metadata or ``0``.
        imas_version: Optional IMAS DD version used during conversion.
        verbose: Print IMAS conversion progress.

    Returns:
        HSDS shot URI for ``env="server"`` or local directory path for ``env="local"``.
    """
    _ = filename
    logging.getLogger().setLevel(logging.WARNING)

    occurrence = occurrence or {}
    shot = int(shot)
    run_value = int(run if run is not None else ods.get("dataset_description.data_entry.run", 0))

    if env == "local":
        if path is None:
            path = Path(f"~/public/imasdb/VEST/3/{shot}/{run_value}").expanduser()
        local_path = Path(path).expanduser()
        local_path.mkdir(parents=True, exist_ok=True)

        save_omas_imas(
            ods,
            user=user,
            machine=machine,
            pulse=shot,
            run=run_value,
            occurrence=occurrence,
            new=not any(local_path.glob("*.h5")),
            imas_version=imas_version,
            verbose=verbose,
            uri="imas:hdf5?path=" + str(local_path),
        )
        print(f"[INFO] Saved IMAS images to: {local_path}")
        return str(local_path)

    if env != "server":
        raise ValueError(f"Unsupported env: {env!r}")

    _require_h5pyd()
    if not is_connect():
        raise ConnectionError("Connection to HSDS server failed")

    with tempfile.TemporaryDirectory(prefix="hsds_imas_ods_") as staging_base:
        shot_dir = Path(staging_base) / str(shot)
        shot_dir.mkdir(parents=True, exist_ok=True)

        save_omas_imas(
            ods,
            user=user,
            machine=machine,
            pulse=shot,
            run=run_value,
            occurrence=occurrence,
            new=True,
            imas_version=imas_version,
            verbose=verbose,
            uri="imas:hdf5?path=" + str(shot_dir),
        )
        _upload_local_shot(shot_dir=shot_dir, directory=directory, shot=shot)

    return f"hdf5://{directory}/{shot}/"
