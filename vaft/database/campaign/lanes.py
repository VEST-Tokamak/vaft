"""Split a campaign's shots into lanes and chunks, and keep each lane's state.

A *lane* is one FileDB root processed by one scheduler allocation, in sequential
*chunks* (one Snakemake run each). Lanes never share a root, because the
pipeline's run-scoped checkpoints (preflight, IMPA selection) would overwrite
each other's verdicts. Shots are dealt round-robin over the sorted list so every
lane sees every era in proportion.

The on-disk layout is the one ``workflow/cluster_regeneration/run_lane.sh``
writes, so a run started by the shell tooling resumes under this API and the
other way round:

``<lane>/shots.txt``            the lane's shots, one per line
``<lane>/chunks/shots.NNN``     the chunks (``split -l N -d -a 3`` naming)
``<lane>/done_chunks.txt``      chunk names Snakemake finished, one per line
``<lane>/PIN_SHA``, ``chunk_size``   the lane's pin and chunk size, fixed for life
``<lane>/STOP``                 stop before the next chunk
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

__all__ = ["LaneState", "chunk_name", "disjoint", "plan_lanes", "read_shots", "split_chunks"]

PathLike = Union[str, Path]


def read_shots(path: PathLike) -> list[int]:
    """The shot numbers listed in a file, one per line (blank lines skipped)."""
    return [int(line) for line in Path(path).read_text(encoding="utf-8").split() if line.strip()]


def plan_lanes(shots: Iterable[int], lanes: int) -> list[list[int]]:
    """Deal the sorted, de-duplicated ``shots`` round-robin over ``lanes`` lanes."""
    if lanes < 1:
        raise ValueError(f"lanes must be >= 1, got {lanes}")
    ordered = sorted(set(int(shot) for shot in shots))
    return [ordered[index::lanes] for index in range(lanes)]


def chunk_name(index: int) -> str:
    """``shots.000`` ... ``shots.999``: the names ``split -d -a 3`` gives."""
    if not 0 <= index < 1000:
        raise ValueError(f"chunk index must be 0..999, got {index}")
    return f"shots.{index:03d}"


def split_chunks(shots: Sequence[int], size: int, directory: PathLike) -> list[Path]:
    """Write ``shots`` as chunks of ``size`` into ``directory``, published atomically.

    The chunks are written into a temporary sibling and moved in only when all
    exist, so a chunk name always means the same shots across resumes.
    Refuses a directory that already holds chunks.
    """
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    target = Path(directory)
    if target.is_dir() and any(target.glob("shots.[0-9][0-9][0-9]")):
        raise FileExistsError(f"{target} already holds chunks; their names are fixed for the lane's life")
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    names = []
    for index, start in enumerate(range(0, len(shots), size)):
        name = chunk_name(index)
        (staging / name).write_text("".join(f"{shot}\n" for shot in shots[start:start + size]), encoding="utf-8")
        names.append(name)
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        os.replace(staging / name, target / name)
    staging.rmdir()
    return [target / name for name in names]


def disjoint(*shot_sets: Iterable[int]) -> list[int]:
    """Shots listed in more than one set: what a merge of lanes would refuse."""
    seen: set[int] = set()
    twice: set[int] = set()
    for shots in shot_sets:
        for shot in set(shots):
            (twice if shot in seen else seen).add(shot)
    return sorted(twice)


@dataclass
class LaneState:
    """The resume state of one lane directory."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def chunks_dir(self) -> Path:
        return self.path / "chunks"

    def chunks(self) -> list[str]:
        return sorted(p.name for p in self.chunks_dir.glob("shots.[0-9][0-9][0-9]"))

    def chunk_shots(self, name: str) -> list[int]:
        return read_shots(self.chunks_dir / name)

    def done(self) -> list[str]:
        try:
            text = (self.path / "done_chunks.txt").read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def pending(self, max_chunks: Optional[int] = None) -> list[str]:
        """Chunks not yet done, in order; at most ``max_chunks`` of them."""
        done = set(self.done())
        remaining = [name for name in self.chunks() if name not in done]
        return remaining if max_chunks is None else remaining[:max_chunks]

    def mark_done(self, name: str) -> None:
        if name not in self.chunks():
            raise KeyError(f"{name} is not a chunk of {self.path}")
        if name in self.done():
            return
        with (self.path / "done_chunks.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{name}\n")

    @property
    def stop_requested(self) -> bool:
        return (self.path / "STOP").exists()

    def ensure(self, key: str, value: str) -> None:
        """Record ``<lane>/<key>`` once and refuse any later different value.

        For ``PIN_SHA`` and ``chunk_size``: a lane belongs to one pin and one
        split for its whole life.
        """
        file = self.path / key
        if file.exists():
            recorded = file.read_text(encoding="utf-8").strip()
            if recorded != str(value):
                raise ValueError(f"{self.path} is fixed at {key}={recorded}, not {value}")
            return
        self.path.mkdir(parents=True, exist_ok=True)
        file.write_text(f"{value}\n", encoding="utf-8")

    def shots(self) -> list[int]:
        """Every shot of every chunk (the lane's frozen list)."""
        return [shot for name in self.chunks() for shot in self.chunk_shots(name)]
