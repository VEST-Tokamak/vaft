"""Scaffolding helpers for the guided tutorial worksheets.

The tutorial notebooks are worksheets, not demonstrations: a fresh copy must
not complete itself when a student presses *Run All*. Two pieces make that
work, and both are designed so the student sees an instruction rather than an
obscure downstream traceback.

``BLANK``
    A sentinel standing in for the answer a student has to supply. Almost
    anything you do to it -- subscripting an ODS with it, iterating it,
    comparing it, passing it to NumPy -- raises :class:`ExerciseIncomplete`
    with a readable message. That is what stops *Run All* at the first
    unanswered exercise instead of, say, ten cells later inside Matplotlib.

``require``
    A completion check for answers that a blank alone cannot catch, such as a
    variable simply assigned ``BLANK``, or a list that still contains one. Call
    it after the student's answers and before the code that uses them.

Typical use in a notebook cell::

    from exercise_support import BLANK, require

    time = ods[BLANK]      # TODO: the plasma-current time path
    signal = ods[BLANK]    # TODO: the plasma-current data path

    require(3, time=time, signal=signal)
    print("samples:", len(time))

This module lives in ``tutorial/`` rather than inside the ``vaft`` package: it
is teaching scaffolding, not library API, and it is deliberately not shipped in
the wheel.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


__all__ = ["BLANK", "ExerciseIncomplete", "is_blank", "require"]


class ExerciseIncomplete(RuntimeError):
    """Raised when an exercise still contains an unanswered blank."""


_BLANK_MESSAGE = (
    "This exercise is not complete.\n"
    "Read the instructions in the Markdown cell above, replace every BLANK in "
    "this cell with your answer, then run the cell again."
)


def _raise_incomplete(*_arguments: Any, **_keywords: Any) -> Any:
    raise ExerciseIncomplete(_BLANK_MESSAGE)


class _Blank:
    """A placeholder that refuses to be used as a value.

    ``__repr__`` deliberately stays functional so that error messages, notebook
    output, and debugger sessions can still show what is missing.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "BLANK"


# Special methods are looked up on the type, so __getattr__ cannot cover them.
# Install the raising behaviour explicitly on everything a student is plausibly
# going to trip over: dictionary/ODS keys, numeric coercion, iteration,
# comparison, arithmetic, truthiness, and attribute access.
for _name in (
    "__getattr__",
    "__hash__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__bool__",
    "__len__",
    "__iter__",
    "__contains__",
    "__call__",
    "__getitem__",
    "__setitem__",
    "__index__",
    "__int__",
    "__float__",
    "__complex__",
    "__str__",
    "__format__",
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__neg__",
    "__abs__",
    "__array__",
):
    setattr(_Blank, _name, _raise_incomplete)
del _name


#: The placeholder students replace with their answer.
BLANK = _Blank()


def is_blank(value: Any) -> bool:
    """Return whether ``value`` is (or still contains) the ``BLANK`` sentinel.

    Identity is tested with ``is``, never ``==``, because comparing to ``BLANK``
    raises. Containers are searched one level deep so that an answer such as
    ``channels = [0, BLANK]`` is caught too.
    """
    if value is BLANK:
        return True
    if isinstance(value, Mapping):
        return any(is_blank(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(is_blank(item) for item in value)
    return False


def require(exercise: int, **answers: Any) -> None:
    """Stop with a readable message while ``answers`` still hold a blank.

    Args:
        exercise: The exercise number, as printed in the Markdown instructions.
        **answers: The names the student was asked to fill in.

    Raises:
        ExerciseIncomplete: If any answer is, or contains, ``BLANK``.
    """
    missing = sorted(name for name, value in answers.items() if is_blank(value))
    if not missing:
        return
    listed = ", ".join(missing)
    raise ExerciseIncomplete(
        f"Exercise {exercise} is not complete.\n"
        f"Still unanswered: {listed}.\n"
        "Read the instructions above and fill in the TODO fields, then run the "
        "cell again."
    )


def confirm(exercise: int, message: str = "") -> None:
    """Report that an exercise's checks passed.

    Kept separate from :func:`require` so a notebook can validate an answer
    itself -- a value range, a channel count -- and still print one consistent
    confirmation line.
    """
    suffix = f" {message}" if message else ""
    print(f"Exercise {exercise} complete.{suffix}")


def check_values(exercise: int, **checks: Iterable[Any]) -> None:
    """Verify named ``(actual, expected)`` pairs and report the first mismatch.

    Args:
        exercise: The exercise number, as printed in the Markdown instructions.
        **checks: Name mapped to a two-item ``(actual, expected)`` sequence.

    Raises:
        ExerciseIncomplete: If any actual value differs from its expectation.
    """
    for name, pair in checks.items():
        actual, expected = tuple(pair)
        if actual != expected:
            raise ExerciseIncomplete(
                f"Exercise {exercise}: {name} is {actual!r}, expected {expected!r}.\n"
                "Re-read the instructions above and correct your answer."
            )
    confirm(exercise)
