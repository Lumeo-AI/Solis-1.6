"""Model Context Protocol (MCP) client for Solis.

This lets a served Solis model reach out to external **tools** — a filesystem
browser, a database, a web fetcher, anything that speaks MCP — and fold their
results back into the conversation. Nothing here trains the model; it is pure
serving-side plumbing that runs *around* generation:

    render tools -> model asks for a call -> we run it over MCP -> feed result back

MCP is JSON-RPC 2.0 over one of two transports, both implemented here with the
standard library only (no new dependencies):

  * **stdio**   — launch the server as a subprocess and exchange newline-framed
                  JSON on its stdin/stdout. This is how local servers ship.
  * **http**    — POST JSON-RPC to a URL (the "streamable HTTP" transport),
                  understanding either a JSON or an SSE (`text/event-stream`)
                  reply, and carrying the `Mcp-Session-Id` the server hands back.

Servers are described by the same ``mcpServers`` object Claude Desktop and the
other clients use, so an existing config file drops straight in:

    {
      "mcpServers": {
        "fs":   {"command": "npx", "args": ["-y",
                 "@modelcontextprotocol/server-filesystem", "/tmp"]},
        "docs": {"url": "https://example.com/mcp",
                 "headers": {"Authorization": "Bearer ..."}}
      }
    }

Tool names are namespaced ``<server>__<tool>`` so two servers can expose a tool
of the same name without colliding.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Protocol revision we advertise in `initialize`. Servers negotiate down if they
# only speak an older one; we do not hard-fail on the value they return.
PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "solis", "version": "1.1.0"}

# The separator between a server's config name and one of its tools. Kept out of
# the JSON-RPC layer — it exists only so a single flat tool namespace can be
# handed to the model and routed back to the right server.
NS_SEP = "__"

DEFAULT_TIMEOUT = float(os.environ.get("SOLIS_MCP_TIMEOUT", "30"))


class MCPError(RuntimeError):
    """A transport failure or a JSON-RPC error returned by a server."""


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    """One callable tool as advertised by a server's ``tools/list``."""

    server: str
    name: str                       # bare name as the server knows it
    description: str
    input_schema: dict              # JSON Schema for the arguments object

    @property
    def qualified(self) -> str:
        """The name the model sees and calls: ``<server>__<tool>``."""
        return f"{self.server}{NS_SEP}{self.name}"

    def to_openai(self) -> dict:
        """Render as an OpenAI `tools` entry, for clients that expect that shape."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object"},
            },
        }


@dataclass
class ServerConfig:
    """How to reach one server. Exactly one of `command` / `url` is set."""

    name: str
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    url: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ServerConfig":
        # Tolerate the two spellings ("disabled" as some configs use, and an
        # explicit "enabled") without making either one required.
        enabled = d.get("enabled", not d.get("disabled", False))
        return cls(
            name=name,
            command=d.get("command"),
            args=list(d.get("args", [])),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            cwd=d.get("cwd"),
            url=d.get("url") or d.get("endpoint"),
            headers={str(k): str(v) for k, v in (d.get("headers") or {}).items()},
            enabled=bool(enabled),
        )


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class _Transport:
    """A JSON-RPC channel: request/response, fire-and-forget notify, close."""

    def request(self, method: str, params: Optional[dict],
                timeout: float) -> Any:
        raise NotImplementedError

    def notify(self, method: str, params: Optional[dict]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class StdioTransport(_Transport):
    """JSON-RPC over a subprocess' stdin/stdout, newline-delimited.

    A background reader thread demultiplexes stdout: responses (messages with an
    ``id``) are handed to the waiting caller through a per-id event, while
    server-initiated notifications and log lines are dropped. stderr is drained
    on its own thread so a chatty server can never fill the pipe and deadlock.
    """

    def __init__(self, cfg: ServerConfig):
        env = {**os.environ, **cfg.env}
        try:
            self.proc = subprocess.Popen(
                [cfg.command, *cfg.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, cwd=cfg.cwd,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise MCPError(f"cannot launch MCP server {cfg.name!r}: "
                           f"{cfg.command!r} not found") from exc

        self._id = 0
        self._id_lock = threading.Lock()
        self._pending: dict[int, dict] = {}
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._alive = True

        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        for _ in iter(self.proc.stderr.readline, b""):
            pass

    def _read_loop(self) -> None:
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # not a JSON-RPC frame (some servers print banners)
            mid = msg.get("id")
            if mid is None:
                continue  # notification/request from the server: ignored
            with self._lock:
                self._pending[mid] = msg
                ev = self._events.get(mid)
            if ev is not None:
                ev.set()
        # Process exited: wake anyone still waiting so they fail rather than hang.
        self._alive = False
        with self._lock:
            for ev in self._events.values():
                ev.set()

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _write(self, obj: dict) -> None:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        with self._write_lock:
            if self.proc.poll() is not None:
                raise MCPError("MCP server process has exited")
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def request(self, method: str, params: Optional[dict],
                timeout: float) -> Any:
        mid = self._next_id()
        ev = threading.Event()
        with self._lock:
            self._events[mid] = ev
        self._write({"jsonrpc": "2.0", "id": mid, "method": method,
                     "params": params or {}})
        if not ev.wait(timeout):
            with self._lock:
                self._events.pop(mid, None)
                self._pending.pop(mid, None)
            raise MCPError(f"timed out after {timeout}s waiting for {method!r}")
        with self._lock:
            self._events.pop(mid, None)
            msg = self._pending.pop(mid, None)
        if msg is None:
            raise MCPError(f"MCP server closed before answering {method!r}")
        if "error" in msg:
            err = msg["error"]
            raise MCPError(f"{method} failed: {err.get('message', err)}")
        return msg.get("result")

    def notify(self, method: str, params: Optional[dict]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        self._alive = False
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class HTTPTransport(_Transport):
    """JSON-RPC over the streamable-HTTP transport.

    Each call is a POST whose body is one JSON-RPC message. The reply is either
    ``application/json`` (parsed directly) or ``text/event-stream`` (we scan the
    SSE ``data:`` lines for the JSON-RPC response). A ``Mcp-Session-Id`` handed
    back by ``initialize`` is echoed on every subsequent request.
    """

    def __init__(self, cfg: ServerConfig):
        self.url = cfg.url
        self.headers = dict(cfg.headers)
        self.session_id: Optional[str] = None
        self._id = 0
        self._id_lock = threading.Lock()

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _post(self, body: dict, timeout: float) -> Optional[Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                ctype = resp.headers.get("Content-Type", "")
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MCPError(f"HTTP {exc.code} from {self.url}: {detail[:200]}") \
                from exc
        except urllib.error.URLError as exc:
            raise MCPError(f"cannot reach {self.url}: {exc.reason}") from exc

        if "id" not in body:          # this was a notification: nothing to parse
            return None
        return _extract_jsonrpc(raw, ctype)

    def request(self, method: str, params: Optional[dict],
                timeout: float) -> Any:
        mid = self._next_id()
        msg = self._post({"jsonrpc": "2.0", "id": mid, "method": method,
                          "params": params or {}}, timeout)
        if msg is None:
            raise MCPError(f"empty response to {method!r}")
        if "error" in msg:
            err = msg["error"]
            raise MCPError(f"{method} failed: {err.get('message', err)}")
        return msg.get("result")

    def notify(self, method: str, params: Optional[dict]) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params or {}},
                   DEFAULT_TIMEOUT)

    def close(self) -> None:
        pass


def _extract_jsonrpc(raw: str, content_type: str) -> Any:
    """Pull one JSON-RPC object out of a JSON or SSE HTTP body."""
    if "text/event-stream" in content_type:
        # SSE frames: lines like `data: {...}`; take the first that parses as a
        # JSON-RPC response (has an "id").
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                return obj
        raise MCPError("no JSON-RPC response found in SSE stream")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPError(f"malformed JSON response: {raw[:200]}") from exc


# --------------------------------------------------------------------------- #
# One server
# --------------------------------------------------------------------------- #
class MCPClient:
    """A single connected server: handshake, list tools, call a tool."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.transport: Optional[_Transport] = None
        self.tools: list[Tool] = []
        self.server_info: dict = {}
        self.error: Optional[str] = None

    def connect(self, timeout: float = DEFAULT_TIMEOUT) -> "MCPClient":
        self.transport = (HTTPTransport(self.cfg) if self.cfg.transport == "http"
                          else StdioTransport(self.cfg))
        init = self.transport.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": CLIENT_INFO,
        }, timeout)
        self.server_info = (init or {}).get("serverInfo", {})
        # Required by the spec: tell the server we are ready before any real call.
        self.transport.notify("notifications/initialized", {})
        self.refresh_tools(timeout)
        return self

    def refresh_tools(self, timeout: float = DEFAULT_TIMEOUT) -> list[Tool]:
        result = self.transport.request("tools/list", {}, timeout) or {}
        self.tools = [
            Tool(server=self.cfg.name, name=t["name"],
                 description=t.get("description", ""),
                 input_schema=t.get("inputSchema") or t.get("input_schema") or {})
            for t in result.get("tools", [])
        ]
        return self.tools

    def call(self, name: str, arguments: dict,
             timeout: float = DEFAULT_TIMEOUT) -> "ToolResult":
        result = self.transport.request(
            "tools/call", {"name": name, "arguments": arguments or {}}, timeout)
        return ToolResult.from_mcp(result or {})

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
            self.transport = None


@dataclass
class ToolResult:
    """The outcome of a ``tools/call``, flattened to text the model can read."""

    text: str
    is_error: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_mcp(cls, result: dict) -> "ToolResult":
        # MCP returns `content` as a list of typed blocks. We concatenate the
        # text ones and note non-text blocks so the model at least knows they
        # were there. `structuredContent`, when present, is the machine-readable
        # form and is worth surfacing verbatim.
        blocks = result.get("content", [])
        pieces: list[str] = []
        for b in blocks:
            btype = b.get("type")
            if btype == "text":
                pieces.append(b.get("text", ""))
            elif btype in ("image", "audio"):
                pieces.append(f"[{btype} returned]")
            elif btype == "resource":
                res = b.get("resource", {})
                pieces.append(res.get("text") or f"[resource {res.get('uri','')}]")
            else:
                pieces.append(json.dumps(b))
        if not pieces and "structuredContent" in result:
            pieces.append(json.dumps(result["structuredContent"]))
        return cls(text="\n".join(p for p in pieces if p),
                   is_error=bool(result.get("isError")), raw=result)


# --------------------------------------------------------------------------- #
# Many servers
# --------------------------------------------------------------------------- #
class MCPManager:
    """Owns every configured server and presents one flat tool namespace.

    Connection is lazy and fault-isolated: a server that fails to start records
    its error and is skipped, rather than taking the whole endpoint down. Call
    :meth:`connect` once at startup (or let :meth:`tools`/:meth:`call` trigger
    it) and :meth:`close` on shutdown.
    """

    def __init__(self, configs: Iterable[ServerConfig]):
        self.clients: dict[str, MCPClient] = {
            c.name: MCPClient(c) for c in configs if c.enabled
        }
        self._connected = False
        self._lock = threading.Lock()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_config(cls, path: str | Path) -> "MCPManager":
        """Build from a JSON file using the standard ``mcpServers`` shape."""
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        servers = blob.get("mcpServers", blob.get("servers", blob))
        configs = [ServerConfig.from_dict(name, spec)
                   for name, spec in servers.items()
                   if isinstance(spec, dict)]
        return cls(configs)

    @classmethod
    def from_env(cls, var: str = "SOLIS_MCP_CONFIG") -> Optional["MCPManager"]:
        """Load from the file named by ``$SOLIS_MCP_CONFIG``, or return None."""
        path = os.environ.get(var)
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            raise MCPError(f"{var}={path!r} does not exist")
        return cls.from_config(p)

    # -- lifecycle --------------------------------------------------------- #
    def connect(self) -> dict[str, Optional[str]]:
        """Connect every server. Returns {name: error-or-None}, never raises for
        a single server's failure."""
        with self._lock:
            status: dict[str, Optional[str]] = {}
            for name, client in self.clients.items():
                if client.transport is not None:
                    status[name] = client.error
                    continue
                try:
                    client.connect()
                    status[name] = None
                except Exception as exc:  # isolate: one bad server != total loss
                    client.error = str(exc)
                    status[name] = str(exc)
            self._connected = True
            return status

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def close(self) -> None:
        for client in self.clients.values():
            try:
                client.close()
            except Exception:
                pass

    # -- tools ------------------------------------------------------------- #
    def tools(self) -> list[Tool]:
        """Every tool across all healthy servers, namespaced."""
        self._ensure_connected()
        out: list[Tool] = []
        for client in self.clients.values():
            if client.error is None:
                out.extend(client.tools)
        return out

    def find(self, qualified: str) -> tuple[MCPClient, str]:
        """Resolve ``server__tool`` (or a bare tool name if unambiguous)."""
        self._ensure_connected()
        if NS_SEP in qualified:
            server, _, bare = qualified.partition(NS_SEP)
            client = self.clients.get(server)
            if client is None:
                raise MCPError(f"no MCP server named {server!r}")
            if client.error:
                raise MCPError(f"MCP server {server!r} is unavailable: "
                               f"{client.error}")
            return client, bare
        # Bare name: accept it only if exactly one server offers it.
        matches = [(c, t.name) for c in self.clients.values()
                   if c.error is None for t in c.tools if t.name == qualified]
        if not matches:
            raise MCPError(f"unknown tool {qualified!r}")
        if len(matches) > 1:
            raise MCPError(f"tool {qualified!r} is ambiguous; qualify it as "
                           f"server{NS_SEP}{qualified}")
        return matches[0]

    def call(self, qualified: str, arguments: dict,
             timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        client, bare = self.find(qualified)
        return client.call(bare, arguments, timeout)

    def status(self) -> list[dict]:
        """Per-server health, for a /health-style report."""
        self._ensure_connected()
        return [{
            "name": name,
            "transport": client.cfg.transport,
            "connected": client.error is None and client.transport is not None,
            "error": client.error,
            "tools": [t.name for t in client.tools],
            "server_info": client.server_info,
        } for name, client in self.clients.items()]


# --------------------------------------------------------------------------- #
# Prompt rendering + tool-call parsing
# --------------------------------------------------------------------------- #
# The convention the model is asked to follow. A small from-scratch model has
# not memorised any provider's function-calling grammar, so we teach it one
# explicitly in the system prompt and keep it dead simple to parse: a single
# JSON object wrapped in <tool_call> tags.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def render_tools_prompt(tools: list[Tool]) -> str:
    """A system-prompt fragment describing the tools and the call format."""
    if not tools:
        return ""
    lines = [
        "You can call tools to help answer. Available tools:",
        "",
    ]
    for t in tools:
        schema = json.dumps(t.input_schema or {"type": "object"},
                            ensure_ascii=False)
        desc = t.description.strip().replace("\n", " ")
        lines.append(f"- {t.qualified}: {desc}")
        lines.append(f"  arguments (JSON Schema): {schema}")
    lines += [
        "",
        "To call a tool, emit exactly one JSON object wrapped in tags, and "
        "nothing else on that turn:",
        '<tool_call>{"name": "<tool>", "arguments": {<args>}}</tool_call>',
        "You will then receive the tool's output and can answer the user. "
        "If no tool is needed, just answer directly.",
    ]
    return "\n".join(lines)


@dataclass
class ParsedCall:
    name: str
    arguments: dict
    raw: str


def parse_tool_calls(text: str) -> list[ParsedCall]:
    """Extract tool calls the model emitted.

    Primary form is ``<tool_call>{...}</tool_call>``. As a fallback we also
    accept a bare fenced ```json block or a lone top-level JSON object that
    carries both a ``name`` and ``arguments`` — small models drift on the exact
    wrapper and this keeps a near-miss usable instead of silently ignored.
    """
    calls: list[ParsedCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        obj = _load_call(m.group(1))
        if obj is not None:
            calls.append(ParsedCall(obj["name"], obj.get("arguments", {}),
                                    m.group(0)))
    if calls:
        return calls
    # Fallback: a fenced or bare JSON object shaped like a call.
    for candidate in _json_candidates(text):
        obj = _load_call(candidate)
        if obj is not None:
            calls.append(ParsedCall(obj["name"], obj.get("arguments", {}),
                                    candidate))
    return calls


def _load_call(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("arguments", obj.get("parameters", {}))
        return {"name": obj["name"], "arguments": args if isinstance(args, dict)
                else {}}
    return None


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _json_candidates(text: str) -> list[str]:
    out = [m.group(1) for m in _FENCE_RE.finditer(text)]
    # A single object spanning the whole (trimmed) message.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        out.append(stripped)
    return out
