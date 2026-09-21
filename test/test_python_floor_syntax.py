"""Every shipped and workflow source file parses on the oldest Python we promise.

Routine CI runs on the canonical interpreter (3.14) only, so nothing else
notices a newer-only construct in a file that ``pyproject.toml`` says works on
3.10 (cold review efit-workflows F1).
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
import warnings
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
FLOOR = (3, 10)  # pyproject.toml: requires-python = ">=3.10"
needs_fstring_tokens = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="f-strings are only tokenized from Python 3.12"
)


def _sources():
    for root in ("vaft", "workflow"):
        yield from sorted((REPOSITORY / root).rglob("*.py"))


def _quote(token_string: str) -> str:
    body = token_string.lstrip("rRbBfFuU")
    return body[:3] if body[:3] in ('"""', "'''") else body[:1]


def _pep701_lines(text: str) -> list[int]:
    """Lines where a string inside an f-string reuses the enclosing quote.

    ``ast.parse(feature_version=(3, 10))`` accepts that (the 3.12 tokenizer has
    already digested it), while 3.10 and 3.11 stop with "f-string: unterminated
    string"; a backslash or comment inside a replacement field is the same
    story but has not reached this tree.
    """
    lines: list[int] = []
    stack: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type in (tokenize.FSTRING_START, tokenize.STRING):
            quote = _quote(token.string)
            # Before 3.12 the enclosing quote character could not appear at all
            # inside a single-quoted f-string, nor the triple inside a triple.
            if any(quote[:1] == outer[:1] and len(quote) >= len(outer) for outer in stack):
                lines.append(token.start[0])
            if token.type == tokenize.FSTRING_START:
                stack.append(quote)
        elif token.type == tokenize.FSTRING_END:
            stack.pop()
    return lines


def test_the_floor_matches_pyproject():
    text = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    assert f'requires-python = ">={FLOOR[0]}.{FLOOR[1]}' in text


@needs_fstring_tokens
def test_the_quote_reuse_check_sees_the_construct_it_is_for():
    offending = "x = f\"| {'-' if a is None else f'{a[\"last\"]:.4f}'} \"\n"
    assert _pep701_lines(offending) == [1]
    assert _pep701_lines("x = f\"| {'-' if a is None else f'{a:.4f}'} {b['k']}\"\n") == []
    assert _pep701_lines('x = f"""{a["k"]} {f"{b}"}"""\n') == []


@needs_fstring_tokens
def test_no_source_reuses_the_enclosing_quote_inside_an_fstring():
    offenders = [
        f"{path.relative_to(REPOSITORY)}:{line}"
        for path in _sources()
        for line in _pep701_lines(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, "Python 3.12-only f-string quoting: " + ", ".join(offenders)


def test_every_source_parses_with_the_floor_grammar():
    failures = []
    for path in _sources():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=FLOOR)
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(REPOSITORY)}:{exc.lineno}: {exc.msg}")
    assert not failures, "not valid Python 3.10: " + "; ".join(failures)


def test_no_source_compiles_with_a_syntax_warning():
    """An invalid escape such as ``"\\p"`` is a SyntaxWarning today and is
    scheduled to become a SyntaxError; a docstring holding LaTeX needs ``r\"\"\"``.
    Found on the Python 3.14 bring-up (#1009)."""
    offenders = []
    # Tests too: a regex in pytest.raises(match="...") is where "\\s" hides.
    for path in [*_sources(), *sorted((REPOSITORY / "test").rglob("*.py"))]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        offenders += [
            f"{path.relative_to(REPOSITORY)}:{w.lineno}: {w.message}"
            for w in caught
            if issubclass(w.category, SyntaxWarning)
        ]
    assert not offenders, "; ".join(offenders)
