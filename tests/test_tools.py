"""Checks for the Solis serving-side logic: registry, tools, search, MCP.

Torch-free by design — the registry, tool-call parsing, identity scrubbing,
web-search parsing and the MCP client have no ML dependency, so this suite runs
without a GPU or a model download.

Run: python -m pytest tests/ -q   (or: python tests/test_tools.py)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solis import registry, websearch  # noqa: E402
from solis.identity import strip_identity_leak, build_system_prompt  # noqa: E402
from solis.mcp import MCPManager, MCPError, ServerConfig, NS_SEP  # noqa: E402
from solis.toolcall import (  # noqa: E402
    ToolCallGate, normalise_tools, parse_tool_calls, strip_tool_calls)
from solis.tools import ToolBus, BUILTIN_NAMES  # noqa: E402


# --------------------------------------------------------------------------- #
# Registry — the 16 GB claims must actually hold
# --------------------------------------------------------------------------- #
def test_default_fits_and_trains_on_16gb():
    v = registry.get_variant(registry.DEFAULT_TEXT)
    assert v.base_repo == "Qwen/Qwen3-8B"
    assert v.fits_16gb(), f"{v.name} does not serve in 16 GB"
    assert v.trainable_16gb(), f"{v.name} does not QLoRA-train in 16 GB"
    print(f"default {v.name} ({v.base_repo}): serve {v.serving_vram_gb()}GB / "
          f"train {v.training_vram_gb()}GB  ok")


def test_stretch_trains_on_24gb_and_serves_on_16gb():
    """The rent-a-GPU tier only makes sense if the result runs at home."""
    v = registry.get_variant(registry.STRETCH_TRAIN)
    assert v.base_repo == "Qwen/Qwen3-14B"
    assert v.trainable_24gb(), "stretch tier must fit a rented 24 GB card"
    assert v.fits_16gb(), "a model you cannot serve at home is not worth training"
    print(f"stretch {v.name}: train {v.training_vram_gb()}GB / "
          f"serve {v.serving_vram_gb()}GB  ok")


def test_moe_memory_follows_total_not_active_params():
    """Qwen3.6-35B-A3B is 3B-active but every expert stays resident."""
    moe = registry.get_variant("solis-1.9-moe")
    assert moe.base_repo == "Qwen/Qwen3.6-35B-A3B"
    assert moe.active_b == 3.0
    weights = moe.weight_gb("nf4")
    naive = 3.0 * 1e9 * 0.55 / 1024 ** 3        # if memory tracked active only
    assert weights > 5 * naive, "MoE weights must reflect total, not active"
    assert not moe.fits_16gb(), "35B at 4-bit cannot serve on 16 GB"
    assert moe.deployment is registry.Deployment.CLOUD
    print(f"moe: {weights}GB weights (vs {naive:.1f}GB if active-only) -> cloud")


def test_no_local_variant_overruns_the_card():
    """Every variant marked LOCAL must genuinely serve inside 16 GB."""
    for v in registry.variants(deployment=registry.Deployment.LOCAL):
        assert v.fits_16gb(), (f"{v.name} is marked LOCAL but needs "
                               f"{v.serving_vram_gb()}GB")
    print("all LOCAL variants fit 16 GB")


# --------------------------------------------------------------------------- #
# Tool calls
# --------------------------------------------------------------------------- #
def test_parse_and_strip_tool_calls():
    text = ('one moment\n<tool_call>\n{"name": "web_search", '
            '"arguments": {"query": "weather"}}\n</tool_call>')
    calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "weather"}
    assert strip_tool_calls(text) == "one moment"
    assert calls[0].to_openai()["function"]["arguments"] == '{"query": "weather"}'
    # Prose is never a call.
    assert parse_tool_calls("just answering normally") == []
    print("tool-call parse ok")


def test_tool_call_gate_streaming():
    def run(chunks):
        g = ToolCallGate()
        out = [g.push(c) for c in chunks]
        out.append(g.flush())
        return "".join(out), g.calls()

    text, calls = run(["Hello ", "there."])
    assert text == "Hello there." and calls == []

    text, calls = run(['<tool_c', 'all>{"name":"web_search",',
                       '"arguments":{"query":"x"}}</tool_call>'])
    assert text == "" and len(calls) == 1 and calls[0].name == "web_search"

    # A '<' that is not a tag survives.
    assert run(["if a < b"])[0] == "if a < b"
    # Truncated mid-call emits nothing rather than raw JSON.
    assert run(['<tool_call>{"name":'])[0] == ""
    print("tool-call streaming gate ok")


def test_normalise_tools():
    out = normalise_tools([
        {"name": "a", "parameters": {"type": "object"}},
        {"type": "function", "function": {"name": "b"}},
        {"junk": 1},
    ])
    assert [t["function"]["name"] for t in out] == ["a", "b"]
    assert normalise_tools(None) is None
    print("tool normalisation ok")


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_identity_scrubs_and_leads():
    assert "Solis" in strip_identity_leak("I am Qwen, made by Alibaba Cloud.")
    p = build_system_prompt("Answer in French.")
    assert p.index("Solis") < p.index("French"), "identity must lead"
    print("identity ok")


# --------------------------------------------------------------------------- #
# Web search / fetch — parsed against fixtures, no network
# --------------------------------------------------------------------------- #
_DDG_FIXTURE = """
<div class="result">
  <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Fa">
    First &amp; best</a>
  <a class="result__snippet">A <b>snippet</b> here.</a>
</div>
"""


def test_duckduckgo_parsing_and_redirect_unwrap():
    import solis.websearch as ws
    original = ws._get
    ws._get = lambda *a, **k: _DDG_FIXTURE
    try:
        results = ws._duckduckgo("q", 5)
    finally:
        ws._get = original
    assert len(results) == 1
    assert results[0].url == "https://example.org/a", results[0].url
    assert results[0].title == "First & best"
    assert results[0].snippet == "A snippet here."
    print("duckduckgo parsing ok")


def test_fetch_page_reduces_html():
    import solis.websearch as ws
    doc = ("<html><head><script>var x=1;</script></head>"
           "<body><h1>T</h1><p>Body text.</p></body></html>")
    original = ws._get
    ws._get = lambda *a, **k: doc
    try:
        text = ws.fetch_page("https://example.com")
        assert "var x" not in text and "Body text." in text
        assert "truncated" in ws.fetch_page("https://example.com", max_chars=5)
    finally:
        ws._get = original
    print("page fetch ok")


def test_fetch_page_rejects_non_http():
    try:
        websearch.fetch_page("file:///etc/passwd")
        raise AssertionError("should have refused a non-HTTP scheme")
    except websearch.SearchError:
        pass
    print("fetch scheme guard ok")


# --------------------------------------------------------------------------- #
# Tool bus
# --------------------------------------------------------------------------- #
def test_bus_ownership_and_errors():
    bus = ToolBus(mcp=None, enable_builtins=True)
    assert set(BUILTIN_NAMES) == {"web_search", "fetch_page"}
    assert bus.owns("web_search", None) and bus.owns("fetch_page", None)
    assert not bus.owns("fs__read", None)          # no MCP configured
    client = [{"type": "function", "function": {"name": "web_search"}}]
    assert not bus.owns("web_search", client), "client tool must shadow ours"
    assert [t["function"]["name"] for t in bus.all_tools(client)].count(
        "web_search") == 1
    # Failures come back as readable results, not exceptions.
    ex = bus.execute("fetch_page", {"url": "file:///etc/passwd"})
    assert ex.is_error
    assert bus.execute("nope", {}).is_error
    print("tool bus ok")


# --------------------------------------------------------------------------- #
# MCP — real stdio round trip against a mock server
# --------------------------------------------------------------------------- #
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
            pass
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
                "name": "echo", "description": "echo it back",
                "inputSchema": {"type": "object",
                    "properties": {"text": {"type": "string"}}}}]}})
        elif method == "tools/call":
            params = msg["params"]
            if params.get("name") != "echo":
                send({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32602, "message": "unknown tool"}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "echo: "
                                 + params.get("arguments", {}).get("text", "")}],
                    "isError": False}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32601, "message": "no method"}})
''')


def _mock_manager(tmp_path: Path) -> MCPManager:
    script = tmp_path / "mock_server.py"
    script.write_text(_MOCK_SERVER, encoding="utf-8")
    return MCPManager([ServerConfig(name="mock", command=sys.executable,
                                    args=[str(script)])])


def test_mcp_stdio_round_trip(tmp_path):
    mgr = _mock_manager(tmp_path)
    try:
        assert mgr.connect()["mock"] is None
        assert [t.qualified for t in mgr.tools()] == [f"mock{NS_SEP}echo"]
        assert mgr.call("mock__echo", {"text": "hi"}).text == "echo: hi"
        assert mgr.call("echo", {"text": "yo"}).text == "echo: yo"
        assert mgr.tools()[0].to_openai()["function"]["name"] == "mock__echo"
    finally:
        mgr.close()
    print("mcp stdio round trip ok")


def test_mcp_errors_and_isolation(tmp_path):
    (tmp_path / "mock_server.py").write_text(_MOCK_SERVER, encoding="utf-8")
    mgr = MCPManager([
        ServerConfig(name="mock", command=sys.executable,
                     args=[str(tmp_path / "mock_server.py")]),
        ServerConfig(name="broken", command="definitely-not-a-real-binary"),
    ])
    try:
        status = mgr.connect()
        assert status["mock"] is None
        assert status["broken"] is not None, "broken server must record an error"
        # A healthy server keeps working alongside a broken one.
        assert [t.qualified for t in mgr.tools()] == ["mock__echo"]
        for bad in ("mock__nope", "nosuchserver__x", "unknown"):
            try:
                mgr.call(bad, {})
                raise AssertionError(f"{bad} should have raised")
            except MCPError:
                pass
    finally:
        mgr.close()
    print("mcp errors + fault isolation ok")


def test_mcp_config_parsing(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text('''{"mcpServers": {
        "a": {"command": "x", "args": ["1"]},
        "b": {"url": "https://h/mcp"},
        "c": {"command": "y", "disabled": true}}}''', encoding="utf-8")
    mgr = MCPManager.from_config(cfg)
    assert set(mgr.clients) == {"a", "b"}, "disabled server must be skipped"
    assert mgr.clients["b"].cfg.transport == "http"
    print("mcp config parsing ok")


if __name__ == "__main__":
    import tempfile

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
    print(f"\nall {len(tests)} Solis tests passed")
