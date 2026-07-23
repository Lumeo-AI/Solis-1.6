"""The Solis tool bus — built-in tools plus every MCP server, in one namespace.

Three kinds of tool reach the model through a single flat list:

  * **built-ins** — `web_search` and `fetch_page`, implemented in-process
    (`solis/websearch.py`). These need no configuration and close the
    training-cutoff gap on anything newer than the base model.
  * **MCP tools** — whatever the configured servers advertise, namespaced
    `<server>__<tool>` (`solis/mcp.py`).
  * **client tools** — anything the caller passed in the request's `tools`
    field. A front end that owns its own tools (a filesystem, a shell) sends
    them here; those we advertise but never execute, because the client is the
    one that can actually run them.

That last distinction is the important one. The server auto-executes only the
tools it *owns* (built-ins + MCP) and hands client-owned calls straight back in
the response, so a harness keeps control of its own side effects.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from . import websearch
from .mcp import MCPManager, MCPError, NS_SEP

# --------------------------------------------------------------------------- #
# Built-in tool definitions (OpenAI function shape)
# --------------------------------------------------------------------------- #
WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this whenever the "
            "answer depends on something newer than your training data — a "
            "library version, a new API, a current error message, release "
            "notes — or when you are unsure a package or symbol exists."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The search query."},
                "max_results": {"type": "integer",
                                "description": "How many results (1-10).",
                                "default": 5},
            },
            "required": ["query"],
        },
    },
}

FETCH_PAGE = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "Fetch a web page and return its readable text. Use after "
            "web_search when a result looks authoritative and you need the "
            "actual content — documentation, a changelog, an issue thread."),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string",
                        "description": "Absolute http(s) URL to fetch."},
            },
            "required": ["url"],
        },
    },
}

BUILTIN_SPECS = [WEB_SEARCH, FETCH_PAGE]
BUILTIN_NAMES = {s["function"]["name"] for s in BUILTIN_SPECS}


def _run_web_search(args: dict) -> str:
    query = args.get("query", "")
    n = int(args.get("max_results") or websearch.MAX_RESULTS)
    results = websearch.search(query, n)
    return websearch.render_results(query, results)


def _run_fetch_page(args: dict) -> str:
    return websearch.fetch_page(args.get("url", ""))


BUILTIN_IMPLS: dict[str, Callable[[dict], str]] = {
    "web_search": _run_web_search,
    "fetch_page": _run_fetch_page,
}


# --------------------------------------------------------------------------- #
# Execution result
# --------------------------------------------------------------------------- #
@dataclass
class Execution:
    """What happened when we ran one tool call."""

    name: str
    arguments: dict
    content: str
    is_error: bool = False
    owner: str = "builtin"      # builtin | mcp | client


# --------------------------------------------------------------------------- #
# The bus
# --------------------------------------------------------------------------- #
class ToolBus:
    """Assembles the tool list and executes the calls the server owns."""

    def __init__(self, mcp: Optional[MCPManager] = None,
                 enable_builtins: bool = True):
        self.mcp = mcp
        self.enable_builtins = enable_builtins
        self._lock = threading.Lock()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "ToolBus":
        """Build from the environment.

        `SOLIS_MCP_CONFIG` points at an mcpServers JSON file; `SOLIS_NO_WEB=1`
        turns the built-in web tools off. A broken MCP config disables MCP with
        a warning rather than taking the server down.
        """
        mcp: Optional[MCPManager] = None
        try:
            mcp = MCPManager.from_env("SOLIS_MCP_CONFIG")
        except Exception as exc:
            print(f"WARNING: MCP disabled — {exc}")
        if mcp is not None:
            for name, err in mcp.connect().items():
                print(f"MCP server {name!r}: {'OK' if err is None else err}")
        enable_builtins = os.environ.get("SOLIS_NO_WEB", "").lower() not in (
            "1", "true", "yes", "on")
        return cls(mcp, enable_builtins)

    # -- advertising ------------------------------------------------------- #
    def server_tools(self) -> list[dict]:
        """Tool specs for everything *this server* can execute."""
        out: list[dict] = []
        if self.enable_builtins:
            out.extend(BUILTIN_SPECS)
        if self.mcp is not None:
            try:
                out.extend(t.to_openai() for t in self.mcp.tools())
            except Exception as exc:
                print(f"WARNING: could not list MCP tools: {exc}")
        return out

    def all_tools(self, client_tools: Optional[list[dict]]) -> Optional[list[dict]]:
        """Server tools + the caller's own, de-duplicated by name.

        Client tools win a name collision: the harness's `web_search` is the one
        it can actually service, and shadowing ours avoids a confusing double.
        """
        client = list(client_tools or [])
        seen = {t.get("function", {}).get("name") for t in client
                if isinstance(t, dict)}
        merged = client + [t for t in self.server_tools()
                           if t["function"]["name"] not in seen]
        return merged or None

    def owns(self, name: str, client_tools: Optional[list[dict]]) -> bool:
        """True if the server should execute this call itself."""
        client_names = {t.get("function", {}).get("name")
                        for t in (client_tools or []) if isinstance(t, dict)}
        if name in client_names:
            return False                      # the caller shadowed it
        if self.enable_builtins and name in BUILTIN_NAMES:
            return True
        if self.mcp is not None and NS_SEP in name:
            return True
        return False

    # -- execution --------------------------------------------------------- #
    def execute(self, name: str, arguments: dict) -> Execution:
        """Run one server-owned tool call. Never raises: a failure becomes an
        error result the model can read and react to."""
        if self.enable_builtins and name in BUILTIN_IMPLS:
            try:
                return Execution(name, arguments, BUILTIN_IMPLS[name](arguments),
                                 owner="builtin")
            except Exception as exc:
                return Execution(name, arguments, f"{type(exc).__name__}: {exc}",
                                 is_error=True, owner="builtin")
        if self.mcp is not None:
            try:
                result = self.mcp.call(name, arguments)
                return Execution(name, arguments,
                                 result.text or "(the tool returned no output)",
                                 is_error=result.is_error, owner="mcp")
            except MCPError as exc:
                return Execution(name, arguments, str(exc), is_error=True,
                                 owner="mcp")
            except Exception as exc:
                return Execution(name, arguments, f"{type(exc).__name__}: {exc}",
                                 is_error=True, owner="mcp")
        return Execution(name, arguments, f"no such tool {name!r}",
                         is_error=True, owner="builtin")

    # -- introspection ----------------------------------------------------- #
    def status(self) -> dict:
        info: dict = {
            "builtins": sorted(BUILTIN_NAMES) if self.enable_builtins else [],
            "web_search_provider": None,
            "mcp": {"enabled": self.mcp is not None},
        }
        if self.enable_builtins:
            try:
                info["web_search_provider"] = websearch.active_provider()
            except Exception as exc:
                info["web_search_provider"] = f"error: {exc}"
        if self.mcp is not None:
            servers = self.mcp.status()
            info["mcp"] = {
                "enabled": True,
                "servers": servers,
                "tool_count": sum(len(s["tools"]) for s in servers
                                  if s["connected"]),
            }
        return info


# --------------------------------------------------------------------------- #
# The agent loop
# --------------------------------------------------------------------------- #
MAX_TOOL_ROUNDS = int(os.environ.get("SOLIS_MAX_TOOL_ROUNDS", "6"))


def run_agent_loop(engine, messages: list[dict], cfg, bus: ToolBus,
                   client_tools: Optional[list[dict]] = None,
                   max_rounds: int = MAX_TOOL_ROUNDS,
                   on_event: Optional[Callable[[dict], None]] = None):
    """Generate, execute any server-owned tool calls, repeat until an answer.

    Returns ``(content, pending_calls, executions)``:

      * ``content``       — the model's final text.
      * ``pending_calls`` — calls the *client* owns, handed back untouched so
        the harness can run them and continue the conversation itself.
      * ``executions``    — what we ran on the server side, for reporting.

    The loop stops as soon as a turn produces no server-owned call, so a plain
    question costs exactly one generation.
    """
    tools = bus.all_tools(client_tools)
    work = [dict(m) for m in messages]
    executions: list[Execution] = []

    for _ in range(max(1, max_rounds)):
        result = engine.chat(work, cfg, tools=tools)
        if not result.tool_calls:
            return result.content, [], executions

        mine = [c for c in result.tool_calls if bus.owns(c.name, client_tools)]
        theirs = [c for c in result.tool_calls
                  if not bus.owns(c.name, client_tools)]

        # Anything the client owns ends the server's turn: only the caller can
        # run it, so hand back the whole request rather than half-servicing it.
        if theirs:
            return result.content, theirs, executions

        # Record the assistant's tool-request turn, then service each call.
        work.append({"role": "assistant", "content": result.content or "",
                     "tool_calls": [c.to_openai() for c in mine]})
        for call in mine:
            if on_event:
                on_event({"type": "tool_call", "name": call.name,
                          "arguments": call.arguments})
            ex = bus.execute(call.name, call.arguments)
            executions.append(ex)
            if on_event:
                on_event({"type": "tool_result", "name": ex.name,
                          "is_error": ex.is_error, "content": ex.content})
            work.append({"role": "tool", "tool_call_id": call.id,
                         "name": call.name, "content": ex.content})

    # Rounds exhausted — take one last turn with no tools so it must answer.
    final = engine.chat(work, cfg, tools=None)
    return final.content, [], executions
