"""X2: web search, either a self-hosted SearXNG or the Brave Search API.

Configured by environment (never by the model): KIN_SEARCH_URL for a SearXNG
instance (its host joins the allowlist), or KIN_BRAVE_KEY for Brave.
Without either, search is off and Kin says so.
"""

from __future__ import annotations

import html
import json
import os
import re
from urllib.parse import quote_plus, urlsplit

from kin.contracts import FactRequest

BRAVE_HOST = "api.search.brave.com"


def backend() -> str | None:
    if os.environ.get("KIN_SEARCH_URL"):
        return "searxng"
    if os.environ.get("KIN_BRAVE_KEY"):
        return "brave"
    return None


def host() -> str:
    if os.environ.get("KIN_SEARCH_URL"):
        return (urlsplit(os.environ["KIN_SEARCH_URL"]).hostname or "").lower()
    return BRAVE_HOST


def url(req: FactRequest) -> str:
    q = quote_plus(req.q or "")
    if backend() == "searxng":
        base = os.environ["KIN_SEARCH_URL"].rstrip("/")
        return f"{base}/search?q={q}&format=json"
    return f"https://{BRAVE_HOST}/res/v1/web/search?q={q}&count=5&result_filter=web"


def headers() -> dict:
    if backend() == "brave":
        return {"X-Subscription-Token": os.environ["KIN_BRAVE_KEY"], "Accept": "application/json"}
    return {"Accept": "application/json"}


_PAIR = re.compile(r'"title":\s*"((?:[^"\\]|\\.)*)".*?"(?:content|description)":\s*"((?:[^"\\]|\\.)*)"', re.S)


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", json.loads(f'"{s}"'))
    return " ".join(html.unescape(s).split())


def parse(text: str, req: FactRequest) -> dict:
    try:
        d = json.loads(text)
        rows = d.get("results") or d.get("web", {}).get("results") or []
        items = [(r.get("title", ""), r.get("content") or r.get("description", "")) for r in rows]
    except ValueError:  # cut at 16 KB: salvage the complete title/snippet pairs
        items = [(m.group(1), m.group(2)) for m in _PAIR.finditer(text)]
        items = [(_clean(a), _clean(b)) for a, b in items]

    def plain(x: str) -> str:
        return " ".join(html.unescape(re.sub(r"<[^>]+>", "", x)).split())

    results = [{"title": plain(t)[:120], "snippet": plain(s)[:200]} for t, s in items if t][:5]
    return {"q": req.q, "results": results}
