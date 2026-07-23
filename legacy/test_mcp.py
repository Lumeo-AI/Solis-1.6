"""Checks for the MCP (Model Context Protocol) tool-calling layer.

Two halves:
  * pure functions — config parsing, prompt rendering, tool-call extraction,
    result flattening — tested directly;
  * the stdio transport end to end, by launching a tiny in-process MCP server
    (``_MOCK_SERVER`` below) as a subprocess and driving a real handshake +
    ``tools/list`` + ``tools/call`` through it.

Nothing here needs torch or a checkpoint; it is all serving-side plumbing.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

# Load solis/mcp.py directly rather than through `import solis`: the MCP layer
# is deliberately free of any torch/ML dependency, and loading the whole package
# would drag those in and make this suite need a full model environment to run.
_MCP_PATH = Path(__file__).resolve().parent.parent / "solis" / "mcp.py"
_spec = importlib.util.spec_from_file_location("solis_mcp", _MCP_PATH)
mcp = importlib.util.module_from_spec(_spec)
sys.modules["solis_mcp"] = mcp   # dataclasses resolves annotations via this
_spec.loader.exec_module(mcp)

MCPManager = mcp.MCPManager
ServerConfig = mcp.ServerConfig
Tool = mcp.Tool
ToolResult = mcp.ToolResult
render_tools_prompt = mcp.render_tools_prompt
parse_tool_calls = mcp.parse_tool_calls
NS_SEP = mcp.NS_SEP


# A complete, dependency-free MCP server speaking newline-framed JSON-RPC on
# stdio. It knows one tool, `echo`, and the minimum handshake.
_MOCK_SERVER = textwrap.dedent('''
    import json, sys
    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid, method = msg.get("id"), msg.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "mock", "version": "0.1"},
                "capabilities": {"tools": {}}}})
        elif method == "notifications/initialized":
            pass  # notification: no reply
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
                "name": "echo",
                "description": "echo the text back",
                "inputSchema": {"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]}}]}})
        elif method == "tools/call":
            params = msg["params"]
            if params.get("name") != "echo":
                send({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32602, "message": "unknown tool"}})
            else:
                args = params.get("arguments", {})
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": "echo: " + args.get("text", "")}],
                    "isError": False}})
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "no method " + str(method)}})
''')


def _mock_manager(tmp_path: Path) -> MCPManager:
    script = tmp_path / "mock_server.py"
    script.write_text(_MOCK_SERVER, encoding="utf-8")
    cfg = ServerConfig(name="mock", command=sys.executable, args=[str(script)])
    return MCPManager([cfg])


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_server_config_from_dict():
    stdio = ServerConfig.from_dict("fs", {"command": "npx", "args": ["a", "b"]})
    assert stdio.transport == "stdio" and stdio.args == ["a", "b"]
    http = ServerConfig.from_dict("api", {"url": "https://x/mcp",
                                          "headers": {"Authorization": "Bearer k"}})
    assert http.transport == "http" and http.headers["Authorization"] == "Bearer k"
    off = ServerConfig.from_dict("x", {"command": "c", "disabled": True})
    assert off.enabled is False
    print("server config parsing ok")


def test_tool_namespacing():
    t = Tool(server="fs", name="read", description="d", input_schema={})
    assert t.qualified == f"fs{NS_SEP}read"
    oa = t.to_openai()
    assert oa["type"] == "function" and oa["function"]["name"] == f"fs{NS_SEP}read"
    print("tool namespacing ok")


def test_parse_tool_calls_tag_form():
    text = 'sure <tool_call>{"name": "fs__read", "arguments": {"path": "/x"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "fs__read" and calls[0].arguments == {"path": "/x"}
    print("tag-form parse ok")


def test_parse_tool_calls_fallbacks_and_none():
    # Fenced JSON, no tags.
    fenced = '```json\n{"name": "t", "arguments": {"a": 1}}\n```'
    assert parse_tool_calls(fenced)[0].name == "t"
    # `parameters` accepted as an alias for `arguments`.
    alias = '<tool_call>{"name": "t", "parameters": {"a": 2}}</tool_call>'
    assert parse_tool_calls(alias)[0].arguments == {"a": 2}
    # Plain prose is never a call.
    assert parse_tool_calls("just a normal answer, no tools here") == []
    print("fallback + negative parse ok")


def test_render_tools_prompt():
    tools = [Tool("fs", "read", "read a file", {"type": "object"})]
    prompt = render_tools_prompt(tools)
    assert "fs__read" in prompt and "<tool_call>" in prompt
    assert render_tools_prompt([]) == ""
    print("prompt rendering ok")


def test_tool_result_flattening():
    r = ToolResult.from_mcp({"content": [
        {"type": "text", "text": "hello"},
        {"type": "image"},
    ], "isError": False})
    assert "hello" in r.text and "[image returned]" in r.text and not r.is_error
    err = ToolResult.from_mcp({"content": [{"type": "text", "text": "boom"}],
                               "isError": True})
    assert err.is_error
    print("tool-result flattening ok")


# --------------------------------------------------------------------------- #
# stdio transport, end to end
# --------------------------------------------------------------------------- #
def test_stdio_round_trip(tmp_path):
    mgr = _mock_manager(tmp_path)
    try:
        status = mgr.connect()
        assert status["mock"] is None, f"connect failed: {status}"

        tools = mgr.tools()
        assert [t.qualified for t in tools] == ["mock__echo"]

        result = mgr.call("mock__echo", {"text": "hi"})
        assert result.text == "echo: hi" and not result.is_error

        # A bare, unambiguous tool name resolves too.
        assert mgr.call("echo", {"text": "yo"}).text == "echo: yo"
    finally:
        mgr.close()
    print("stdio round trip ok")


def test_unknown_tool_raises(tmp_path):
    mgr = _mock_manager(tmp_path)
    try:
        mgr.connect()
        raised = False
        try:
            mgr.call("mock__nope", {})
        except mcp.MCPError:
            raised = True
        assert raised, "calling a missing tool should raise MCPError"
    finally:
        mgr.close()
    print("unknown-tool error ok")


if __name__ == "__main__":
    import tempfile
    test_server_config_from_dict()
    test_tool_namespacing()
    test_parse_tool_calls_tag_form()
    test_parse_tool_calls_fallbacks_and_none()
    test_render_tools_prompt()
    test_tool_result_flattening()
    with tempfile.TemporaryDirectory() as d:
        test_stdio_round_trip(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_unknown_tool_raises(Path(d))
    print("\nall MCP tests passed")
