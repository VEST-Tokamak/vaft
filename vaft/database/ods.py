"""OMAS ODS access backed by IMAS HDF5 images stored in HSDS."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time as wall_time
from datetime import datetime, timezone
from typing import Literal, Optional, Union

import omas
import numpy as np

try:
    import h5pyd
except ImportError:
    h5pyd = None  # optional: pip install 'vaft[hsds]'

from ..imas.omas_imas import load_omas_imas, save_omas_imas
from .transport import run_hsget, run_hsload, verify_uploaded_image
from .staging import requested_ids_from_paths, stage_imas_shot
from .utils import _require_h5pyd, ensure_imas_hdf5_userblock, exist_shot, is_connect


_DERIVED_CACHE_SCHEMA = 1


def _text(value) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _source_revision(directory: str, shot: int) -> tuple[str, dict[str, dict[str, str | None]]]:
    """Fingerprint canonical IMAS image revisions without reading their payloads."""
    _require_h5pyd()
    domains: dict[str, dict[str, str | None]] = {}
    names = sorted(name for name in h5pyd.Folder(f"/{directory}/{shot}/") if name.endswith(".h5"))
    for name in names:
        with h5pyd.File(f"hdf5://{directory}/{shot}/{name}", "r") as domain:
            domains[name] = {
                "last_modified": _text(getattr(domain, "modified", None)),
                "backend_version": _text(domain.attrs.get("HDF5_BACKEND_VERSION")),
            }
    if any(record["last_modified"] is None for record in domains.values()):
        raise RuntimeError("HSDS does not expose domain revision metadata; refusing to publish a stale-prone derived cache")
    payload = json.dumps(domains, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), domains


def _manifest_bytes(manifest: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8)


def _read_derived_manifest(domain) -> dict | None:
    try:
        raw = np.asarray(domain["manifest"][:], dtype=np.uint8).tobytes()
        manifest = json.loads(raw.decode("utf-8"))
        return manifest if isinstance(manifest, dict) else None
    except Exception:
        return None


def _load_derived_omas_cache(directory: str, shot: int, consistency_check: bool) -> omas.ODS | None:
    """Return a verified full-shot ODS cache, or ``None`` when stale/missing."""
    try:
        revision, _ = _source_revision(directory, shot)
        with h5pyd.File(f"hdf5://{directory}/{shot}.omas.h5", "r") as domain:
            manifest = _read_derived_manifest(domain)
            if manifest is None or manifest.get("source_revision") != revision:
                return None
            payload = np.asarray(domain["h5image"][:], dtype=np.uint8)
            expected = manifest.get("checksum")
            if expected != hashlib.sha256(payload.tobytes()).hexdigest():
                return None
        with tempfile.NamedTemporaryFile(prefix="vaft-derived-omas-", suffix=".h5") as temporary:
            temporary.write(payload.tobytes())
            temporary.flush()
            result = omas.ODS(consistency_check=consistency_check)
            result.load(temporary.name, consistency_check=consistency_check)
            return result
    except Exception:
        return None


def _publish_derived_omas_cache(
    ods: omas.ODS,
    directory: str,
    shot: int,
    imas_version: str | None,
    *,
    settle_seconds: float = 0.0,
) -> str:
    """Publish a verified compressed full-ODS materialized view beside a shot."""
    _require_h5pyd()
    from ..version import __version__

    # hsload can return after the domain is readable but before HSDS has
    # finalized the domain's ``lastModified`` value.  Capture a revision only
    # after a short post-upload settling interval; otherwise a newly written
    # derived view can be incorrectly rejected as stale on its first read.
    if settle_seconds:
        wall_time.sleep(settle_seconds)
    revision, source_domains = _source_revision(directory, shot)
    with tempfile.NamedTemporaryFile(prefix="vaft-derived-omas-", suffix=".h5") as temporary:
        ods.save(temporary.name)
        temporary.flush()
        payload = np.frombuffer(Path(temporary.name).read_bytes(), dtype=np.uint8)
    manifest = {
        "schema_version": _DERIVED_CACHE_SCHEMA,
        "source_revision": revision,
        "source_domains": source_domains,
        "imas_version": imas_version,
        "omas_version": getattr(omas, "__version__", None),
        "vaft_version": __version__,
        "checksum": hashlib.sha256(payload.tobytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    uri = f"hdf5://{directory}/{shot}.omas.h5"
    with h5pyd.File(uri, "w") as domain:
        chunk = max(1, min(int(payload.size), 1024 * 1024))
        image = domain.create_dataset(
            "h5image", shape=payload.shape, dtype=np.uint8, chunks=(chunk,),
            compression="gzip", compression_opts=1, shuffle=True,
        )
        image[:] = payload
        manifest_dataset = domain.create_dataset("manifest", shape=_manifest_bytes(manifest).shape, dtype=np.uint8)
        manifest_dataset[:] = _manifest_bytes(manifest)
    return uri


def _download_remote_image(remote_uri: str, out_path: Path) -> Path:
    """Download one HSDS HDF5 image to a local file via hsget."""
    run_hsget(remote_uri, out_path)
    ensure_imas_hdf5_userblock(out_path, out_path.parent)
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
    run_hsload(local_path, remote_uri)
    verify_uploaded_image(local_path, remote_uri)
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
    verbose: bool = False,
    path: Optional[Union[str, Path]] = None,
    local_dir: Optional[Union[str, Path]] = None,
    cache: Literal["auto", "off"] | str | Path = "auto",
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
        path: Optional existing IMAS HDF5 directory containing ``master.h5``.
            When set, files are read directly and no HSDS download is attempted.
        local_dir: Optional local staging base directory. When omitted a temporary directory
            is used and cleaned up automatically.
        cache: ``"auto"`` (default) stores validated HSDS domains under the user cache,
            ``"off"`` always downloads, or a path selects a cache base directory.

    Returns:
        One ``omas.ODS`` or a list of ``omas.ODS`` objects.
    """
    _require_h5pyd()
    logging.getLogger().setLevel(logging.WARNING)

    occurrence = occurrence or {}

    if isinstance(shot, list) and path is not None:
        raise ValueError("path can only be used when loading a single shot")

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
                path=None,
                local_dir=local_dir,
                cache=cache,
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
        path=path,
        local_dir=local_dir,
        cache=cache,
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
    path: Optional[Union[str, Path]],
    local_dir: Optional[Union[str, Path]],
    cache: Literal["auto", "off"] | str | Path,
) -> omas.ODS:
    """Load one shot from HSDS IMAS images and convert it to ODS."""
    logging.getLogger().setLevel(logging.WARNING)

    if path is not None:
        shot_dir = Path(path).expanduser()
        if not (shot_dir / "master.h5").exists():
            raise FileNotFoundError(f"No master.h5 found in IMAS HDF5 directory: {shot_dir}")
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

    if paths is None and time is None:
        derived = _load_derived_omas_cache(directory, shot, consistency_check)
        if derived is not None:
            derived.setdefault("dataset_description.data_entry.user", str(directory))
            derived.setdefault("dataset_description.data_entry.pulse", int(shot))
            derived.setdefault("dataset_description.data_entry.run", 0)
            return derived

    staging_ctx = (
        tempfile.TemporaryDirectory(prefix="hsds_imas_ods_")
        if local_dir is None
        else nullcontext(str(local_dir))
    )
    with staging_ctx as staging_base:
        shot_dir = Path(staging_base) / str(shot)
        shot_dir.mkdir(parents=True, exist_ok=True)
        requested_ids = requested_ids_from_paths(paths)
        stage_imas_shot(
            directory=directory,
            shot=shot,
            staging_dir=shot_dir,
            requested_ids=requested_ids,
            cache=cache,
        )

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
        # TODO: Revisit the selected-shot density correction from the
        # confinement_time_for_kps branch and adapt it to this IMAS-backed ODS
        # loading flow if those overestimated n_e shots still require scaling.

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
    materialize_omas_cache: bool = True,
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
        if materialize_omas_cache:
            try:
                cache_uri = _publish_derived_omas_cache(
                    ods,
                    directory,
                    shot,
                    imas_version,
                    settle_seconds=8.0,
                )
                print(f"[INFO] Published derived OMAS cache: {cache_uri}")
            except Exception as exc:
                # Canonical IMAS persistence has succeeded. A derived cache must
                # never turn that successful write into an application failure.
                logging.warning("Could not publish derived OMAS cache for shot %s: %s", shot, exc)

    return f"hdf5://{directory}/{shot}/"
