"""Command-line transforms shared by the adapters that need them.

These rewrite *what* is launched, not *how*: a stack-limit wrapper has to
travel with the command whichever execution backend runs it, local or batch.
"""

from __future__ import annotations

from typing import Sequence, Union

from .. import compat

StackSize = Union[int, str, None]


def with_stack_limit(argv: Sequence[str], stack_size_kb: StackSize, *, runner_name: str) -> list[str]:
    """Wrap ``argv`` so the child starts with a raised soft stack limit.

    ``None`` leaves ``argv`` alone. ``"hard"`` raises the soft limit to the
    hard limit. An integer asks for that many kB and falls back to the hard
    limit when the request exceeds it (macOS caps the hard limit just under
    64 MB). ``runner_name`` becomes the wrapper's ``$0``, which is what a
    process listing shows.

    On Windows ``argv`` is returned unchanged: a native image takes its stack
    reserve from the PE header, fixed by the linker (the EFIT installer passes
    ``-Wl,--stack``), and wrapping would make every run depend on an MSYS2
    bash the installers deliberately keep off PATH.
    """
    argv = [str(part) for part in argv]
    if stack_size_kb is None or compat.IS_WINDOWS:
        return argv
    if stack_size_kb == "hard":
        shell = 'ulimit -s $(ulimit -Hs) 2>/dev/null; exec "$@"'
    else:
        limit = int(stack_size_kb)
        shell = f'ulimit -s {limit} 2>/dev/null || ulimit -s $(ulimit -Hs) 2>/dev/null; exec "$@"'
    return ["bash", "-lc", shell, runner_name, *argv]
