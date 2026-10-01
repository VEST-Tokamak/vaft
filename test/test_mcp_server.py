"""The VAFT MCP server over the real SDK and a real stdio subprocess (#1423)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_importing_vaft_and_its_mcp_package_does_not_import_the_sdk():
    """``import vaft``, ``vaft.help()`` and ``import vaft.mcp`` stay SDK-free."""
    code = (
        "import sys, vaft, vaft.mcp, vaft.mcp._tools, vaft.mcp.server, vaft.cli.mcp; vaft.help(); "
        "loaded = sorted(m for m in sys.modules if m == 'mcp' or m.startswith('mcp.')); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


mcp = pytest.importorskip("mcp")

from vaft.mcp import _tools  # noqa: E402
from vaft.mcp.server import build_server  # noqa: E402


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


def test_a_vaft_error_becomes_a_tool_error():
    from mcp.server.fastmcp.exceptions import ToolError

    server = build_server()
    with pytest.raises(ToolError, match="no formula named"):
        asyncio.run(server.call_tool("describe_formula", {"name": "no_such_formula_anywhere"}))


def _child_env() -> dict[str, str]:
    """The child must import the VAFT under test, not another installed copy."""
    import vaft

    env = dict(os.environ)
    package_root = str(Path(vaft.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH", ""), package_root) if p)
    return env


async def _round_trip():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(command=sys.executable, args=["-m", "vaft.mcp"], env=_child_env())
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [tool.name for tool in (await session.list_tools()).tools]
            capabilities = await session.call_tool("get_capabilities", {})
            formula = await session.call_tool("describe_formula", {"name": "equilibrium.kink_safety_factor"})
            missing = await session.call_tool("describe_formula", {"name": "no_such_formula_anywhere"})
            return names, capabilities, formula, missing


def _payload(result):
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


def test_a_stdio_client_discovers_and_calls_the_server():
    names, capabilities, formula, missing = asyncio.run(asyncio.wait_for(_round_trip(), timeout=300))
    assert names == [tool.__name__ for tool in _tools.TOOLS]
    assert not capabilities.isError
    assert "formula" in [row["name"] for row in _payload(capabilities)["topics"]]
    assert not formula.isError
    assert _payload(formula) == json.loads(json.dumps(_tools.describe_formula("equilibrium.kink_safety_factor")))
    assert missing.isError
    assert "no formula named" in missing.content[0].text
