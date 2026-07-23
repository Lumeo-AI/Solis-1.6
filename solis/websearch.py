"""Web search and page fetching for Solis.

An assistant with a training cutoff cannot know anything recent — this
week's news, a changed price, a new release. Giving it search closes that gap:
the model asks for `web_search`, we run the query, and the results come back as
a tool result it can read before answering.

Providers, in the order they are auto-selected:

  * **brave**   — needs ``BRAVE_API_KEY``. Best quality, generous free tier.
  * **tavily**  — needs ``TAVILY_API_KEY``. Built for LLM use; returns clean
                  summaries rather than raw SERP noise.
  * **serper**  — needs ``SERPER_API_KEY`` (Google results).
  * **duckduckgo** — no key, the default fallback. Scrapes the HTML endpoint,
                  so it is best-effort: DDG can rate-limit or change markup.

Set ``SOLIS_SEARCH_PROVIDER`` to force one. Everything here uses only the
standard library, so search adds no dependencies.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

USER_AGENT = ("Mozilla/5.0 (compatible; Solis/1.9; +https://localhost) "
              "Python-urllib")
DEFAULT_TIMEOUT = float(os.environ.get("SOLIS_SEARCH_TIMEOUT", "15"))
MAX_RESULTS = int(os.environ.get("SOLIS_SEARCH_MAX_RESULTS", "5"))
# Hard cap on fetched page text handed to the model. A 300 KB page would eat the
# whole context window and push the actual question out of it.
MAX_PAGE_CHARS = int(os.environ.get("SOLIS_FETCH_MAX_CHARS", "8000"))


class SearchError(RuntimeError):
    """A search provider failed or returned nothing usable."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def render(self) -> str:
        return f"{self.title}\n{self.url}\n{self.snippet}".strip()


def _get(url: str, headers: Optional[dict] = None,
         timeout: float = DEFAULT_TIMEOUT) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise SearchError(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}") \
            from exc
    except urllib.error.URLError as exc:
        raise SearchError(f"cannot reach {urllib.parse.urlsplit(url).netloc}: "
                          f"{exc.reason}") from exc


def _post_json(url: str, payload: dict, headers: dict,
               timeout: float = DEFAULT_TIMEOUT) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json",
                 **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raise SearchError(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}") \
            from exc
    except urllib.error.URLError as exc:
        raise SearchError(f"cannot reach {url}: {exc.reason}") from exc


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def _brave(query: str, n: int) -> list[SearchResult]:
    key = os.environ["BRAVE_API_KEY"]
    url = ("https://api.search.brave.com/res/v1/web/search?"
           + urllib.parse.urlencode({"q": query, "count": n}))
    data = json.loads(_get(url, {"Accept": "application/json",
                                 "X-Subscription-Token": key}))
    return [SearchResult(r.get("title", ""), r.get("url", ""),
                         _strip_html(r.get("description", "")))
            for r in data.get("web", {}).get("results", [])[:n]]


def _tavily(query: str, n: int) -> list[SearchResult]:
    data = _post_json("https://api.tavily.com/search",
                      {"api_key": os.environ["TAVILY_API_KEY"],
                       "query": query, "max_results": n},
                      {})
    return [SearchResult(r.get("title", ""), r.get("url", ""),
                         r.get("content", ""))
            for r in data.get("results", [])[:n]]


def _serper(query: str, n: int) -> list[SearchResult]:
    data = _post_json("https://google.serper.dev/search",
                      {"q": query, "num": n},
                      {"X-API-KEY": os.environ["SERPER_API_KEY"]})
    return [SearchResult(r.get("title", ""), r.get("link", ""),
                         r.get("snippet", ""))
            for r in data.get("organic", [])[:n]]


# DuckDuckGo's no-JS endpoint. Parsed with regex rather than a HTML library to
# keep the dependency list at zero; the markup is simple and stable enough for
# a best-effort fallback, and failures degrade to "no results" not a crash.
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL)


def _duckduckgo(query: str, n: int) -> list[SearchResult]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    body = _get(url)
    out: list[SearchResult] = []
    for m in _DDG_RESULT_RE.finditer(body):
        link = html.unescape(m.group("url"))
        # DDG wraps results in a redirect: /l/?uddg=<encoded target>
        if "uddg=" in link:
            qs = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query)
            link = qs.get("uddg", [link])[0]
        out.append(SearchResult(_strip_html(m.group("title")), link,
                                _strip_html(m.group("snippet"))))
        if len(out) >= n:
            break
    return out


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def _strip_html(s: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", s))).strip()


PROVIDERS = {
    "brave": (_brave, "BRAVE_API_KEY"),
    "tavily": (_tavily, "TAVILY_API_KEY"),
    "serper": (_serper, "SERPER_API_KEY"),
    "duckduckgo": (_duckduckgo, None),
}
# Preference order when nothing is forced: keyed providers first (better
# results), DuckDuckGo last as the always-available fallback.
_AUTO_ORDER = ["brave", "tavily", "serper", "duckduckgo"]


def active_provider() -> str:
    """Which provider will be used, honouring SOLIS_SEARCH_PROVIDER."""
    forced = os.environ.get("SOLIS_SEARCH_PROVIDER", "").strip().lower()
    if forced:
        if forced not in PROVIDERS:
            raise SearchError(f"unknown SOLIS_SEARCH_PROVIDER {forced!r}; "
                              f"known: {sorted(PROVIDERS)}")
        return forced
    for name in _AUTO_ORDER:
        _, key = PROVIDERS[name]
        if key is None or os.environ.get(key):
            return name
    return "duckduckgo"


def search(query: str, max_results: int = MAX_RESULTS) -> list[SearchResult]:
    """Run a web search with the active provider."""
    if not query or not query.strip():
        raise SearchError("empty query")
    name = active_provider()
    fn, key = PROVIDERS[name]
    if key and not os.environ.get(key):
        raise SearchError(f"provider {name!r} needs {key} in the environment")
    results = fn(query.strip(), max(1, min(max_results, 10)))
    if not results:
        raise SearchError(f"no results from {name} for {query!r}")
    return results


def render_results(query: str, results: list[SearchResult]) -> str:
    """Format results as the text the model reads back as a tool result."""
    lines = [f"Search results for: {query}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    {r.url}")
        if r.snippet:
            lines.append(f"    {r.snippet}")
        lines.append("")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Page fetching
# --------------------------------------------------------------------------- #
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_BLOCK_END_RE = re.compile(
    r"</(p|div|section|article|h[1-6]|li|tr|pre|blockquote)>",
    re.IGNORECASE)


def fetch_page(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Fetch a URL and reduce it to readable text.

    Deliberately crude — strip scripts/styles, turn block ends into newlines,
    drop the remaining tags. Good enough to read docs and issue threads, and it
    needs no HTML parser. The result is truncated so one page cannot swallow the
    context window.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise SearchError(f"refusing to fetch non-HTTP url {url!r}")
    body = _get(url)
    body = _SCRIPT_STYLE_RE.sub(" ", body)
    body = _BLOCK_END_RE.sub("\n", body)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    text = html.unescape(_TAG_RE.sub("", body))
    text = _WS_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANKS_RE.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated at {max_chars} characters]"
    return text
