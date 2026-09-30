"""Week 3 exit criterion: the recall demo works with Wi-Fi off.

Python networking is disabled for the whole test (any socket raises). The real
store, embedder, calendar and language model still answer from memory, and the
weather part of a mixed question degrades to "can't check right now".
"""

import datetime as dt
import socket

import pytest

from tests.conftest import DATA

pytestmark = pytest.mark.models


class NoNetwork(OSError):
    pass


def _refuse(*args, **kwargs):
    raise NoNetwork("network is off")


def test_recall_demo_with_network_off(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)

    from kin.memory.embed import Embedder
    from kin.memory.service import Memory
    from kin.memory.store import Store
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner
    from kin.reasoning.llm import LocalLLM

    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), Embedder())
    t = dt.datetime(2026, 9, 29, 18, 12).timestamp()
    mem.add(t, t + 4, "Rohan said the site visit is Thursday at 4 pm at Butibori.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    llm = LocalLLM(threads=4)
    r = Reasoner(mem, llm, Calendar(DATA / "demo_calendar.ics"), clock=lambda: dt.datetime(2026, 9, 30, 10, 40))
    try:
        a = r.handle("When is the site visit Rohan mentioned?")
        assert "Thursday" in a.text and a.citations and a.llm_used
        a = r.handle("Should I take an umbrella to the site visit Rohan mentioned?")
        assert "Butibori" in a.text and "can't check the weather right now" in a.text
        with pytest.raises(NoNetwork):
            socket.create_connection(("example.com", 443))  # really off
    finally:
        llm.unload()
        mem.store.close()
