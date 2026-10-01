"""X3: allowlisted RSS feeds (BBC)."""

from __future__ import annotations

import html
import re

from kin.contracts import FactRequest

HOST = "feeds.bbci.co.uk"
FEEDS = {
    "local": "/news/world/asia/india/rss.xml",
    "tech": "/news/technology/rss.xml",
    "business": "/news/business/rss.xml",
    "sport": "/sport/rss.xml",
}
_ITEM = re.compile(r"<item>.*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", re.S)


def url(req: FactRequest) -> str:
    return f"https://{HOST}{FEEDS[req.topic or 'local']}"


def parse(text: str, req: FactRequest) -> dict:
    # Regex, not an XML parser: the response is cut at 16 KB, so the last item may be broken.
    titles = [html.unescape(t.strip()) for t in _ITEM.findall(text)]
    return {"topic": req.topic or "local", "headlines": [{"title": t[:160], "source": "BBC"} for t in titles[:5]]}
