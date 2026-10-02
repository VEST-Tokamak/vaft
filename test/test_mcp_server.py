"""The VAFT MCP server over the real SDK and a real stdio subprocess (#1423)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_importing_vaft_and_its_mcp_package_does_not_import_the_sdk():
    """``import vaft``, ``vaft.help()`` and ``import vaft.mcp`` stay SDK-free."""
    code = (
        "import sys, vaft, vaft.mcp, vaft.mcp._tools, vaft.mcp.server, vaft.cli.mcp; vaft.help(); "
        "loaded = sorted(m for m in sys.modules if m == 'mcp' or m.startswith('mcp.')); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


mcp = pytest.importorskip("mcp")

import vaft  # noqa: E402
from vaft.mcp import _tools  # noqa: E402
from vaft.mcp.server import build_server  # noqa: E402

TREE = Path(vaft.__file__).resolve().parents[1]


def _tools_listed():
    return asyncio.run(build_server().list_tools())


def test_the_server_lists_exactly_the_curated_tools_all_read_only():
    listed = _tools_listed()
    assert [tool.name for tool in listed] == [tool.__name__ for tool in _tools.TOOLS]
    for tool in listed:
        assert tool.annotations.readOnlyHint is True, tool.name
        assert tool.annotations.destructiveHint is False, tool.name
        assert tool.annotations.openWorldHint is False, tool.name
        assert tool.description, tool.name
        properties = tool.inputSchema.get("properties", {})
        assert not any("ods" in p.lower() or "representation" in p.lower() for p in properties), tool.name
    extract = next(tool for tool in listed if tool.name == "extract_plot_data")
    assert "Allowed options keys" in extract.description and "coordinate" in extract.description


def test_the_sdk_internals_the_stdio_guard_relies_on_still_exist():
    """``_serve`` re-implements ``FastMCP.run_stdio_async`` through ``_mcp_server``.

    The SDK is pinned below 2; if a release inside the pin renames these, fail
    here, by name, rather than as a hung stdio session.
    """
    server = build_server()
    low = getattr(server, "_mcp_server", None)
    assert low is not None, "FastMCP no longer has _mcp_server: rewrite vaft.mcp.server._serve"
    assert callable(getattr(low, "run", None))
    assert callable(getattr(low, "create_initialization_options", None))
    from mcp.server.stdio import stdio_server  # noqa: F401 - the transport _serve opens


def test_a_vaft_error_becomes_a_tool_error():
    from mcp.server.fastmcp.exceptions import ToolError

    server = build_server()
    with pytest.raises(ToolError, match="no formula named"):
        asyncio.run(server.call_tool("describe_formula", {"name": "no_such_formula_anywhere"}))


def _child_env() -> dict[str, str]:
    """An environment whose interpreter imports the VAFT under test, checked before use."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH", ""), str(TREE)) if p)
    found = subprocess.run(
        [sys.executable, "-c", "import vaft; print(vaft.__file__)"],
        env=env, capture_output=True, text=True, timeout=300, check=True,
    ).stdout.strip()
    assert Path(found).resolve().is_relative_to(TREE), (
        f"the stdio child would import {found}, not the tree under test {TREE}"
    )
    return env


def _payload(result):
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


async def _session(args, calls):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable, args=args, env=_child_env())
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [tool.name for tool in (await session.list_tools()).tools]
            results = [await session.call_tool(name, arguments) for name, arguments in calls]
            return names, results


def _run(args, calls):
    return asyncio.run(asyncio.wait_for(_session(args, calls), timeout=300))


def test_a_stdio_client_discovers_and_calls_the_server():
    pytest.importorskip("omas")
    names, (capabilities, formula, missing, extracted) = _run(["-m", "vaft.mcp"], [
        ("get_capabilities", {}),
        ("describe_formula", {"name": "equilibrium.kink_safety_factor"}),
        ("describe_formula", {"name": "no_such_formula_anywhere"}),
        ("extract_plot_data", {"name": "equilibrium_time_q95", "shot": 39915, "max_points": 50}),
    ])
    assert names == [tool.__name__ for tool in _tools.TOOLS]
    assert not capabilities.isError
    assert "formula" in [row["name"] for row in _payload(capabilities)["topics"]]
    assert not formula.isError
    assert _payload(formula) == json.loads(json.dumps(_tools.describe_formula("equilibrium.kink_safety_factor")))
    assert missing.isError
    assert "no formula named" in missing.content[0].text
    assert not extracted.isError
    direct = json.loads(json.dumps(_tools.extract_plot_data("equilibrium_time_q95", max_points=50)))
    assert _payload(extracted) == direct


#: A server whose list_samples writes to stdout from Python and at the fd level,
#: the way a C or Fortran library would.
_NOISY_SERVER = textwrap.dedent(
    """
    import functools, os, sys
    from vaft.mcp import _tools

    original = _tools.list_samples

    @functools.wraps(original)
    def list_samples():
        print("python-level noise on stdout")
        sys.stdout.flush()
        os.write(1, b"fd-level noise on stdout\\n")
        return original()

    _tools.TOOLS = tuple(list_samples if f is original else f for f in _tools.TOOLS)
    from vaft.mcp.server import main
    raise SystemExit(main())
    """
)


def test_stdout_noise_from_a_tool_does_not_break_the_session():
    names, (noisy, after) = _run(["-c", _NOISY_SERVER], [("list_samples", {}), ("get_capabilities", {})])
    assert "list_samples" in names
    assert not noisy.isError
    assert 39915 in [row["shot"] for row in _payload(noisy)["items"]]
    assert not after.isError
