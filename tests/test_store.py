"""Encrypted store: grouping, keyword and vector search, deletion, retention, wipe.

Uses synthetic vectors, so no models are needed.
"""

import numpy as np
import pytest

from kin.memory.store import DIM, Store
from kin.perception.segmenter import EpisodePolicy

KEY = "cd" * 32
POLICY = EpisodePolicy(max_span_s=60, max_gap_s=30)
T0 = 1_800_000_000.0


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(params=["sqlite-vec", "numpy"])
def store(tmp_path, request):
    s = Store(tmp_path / "kin.db", KEY, use_vec=request.param == "sqlite-vec")
    assert s.vec_backend == request.param
    yield s
    s.close()


def add(store, t, text, **kw):
    return store.add_utterance(t, t + 3, text, policy=POLICY, **kw)


def seed(store):
    """Three episodes: Rohan (t0), Priya (t0+200), lunch (t0+400), each embedded."""
    a = add(store, T0, "Rohan said the site visit is Thursday at 4 pm")["episode_id"]
    add(store, T0 + 10, "It is at Butibori near the depot")
    b = add(store, T0 + 200, "Priya will send the budget slides tomorrow")["episode_id"]
    c = add(store, T0 + 400, "I had pasta for lunch")["episode_id"]
    store.close_all()
    for ep, s in ((a, 1), (b, 2), (c, 3)):
        assert store.set_vector(ep, unit(s))
    return a, b, c


def test_file_is_encrypted_and_needs_the_key(tmp_path):
    s = Store(tmp_path / "kin.db", KEY)
    add(s, T0, "Priya's secret number is 4521")
    s.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert b"Priya" not in raw and b"4521" not in raw and b"SQLite format 3" not in raw
    with pytest.raises(RuntimeError, match="wrong key"):
        Store(tmp_path / "kin.db", "ef" * 32)
    again = Store(tmp_path / "kin.db", KEY)
    assert again.stats()["utterances"] == 1
    again.close()


def test_utterances_group_into_episodes(store):
    e1 = add(store, T0, "one")
    e2 = add(store, T0 + 10, "two")  # short pause: same episode
    e3 = add(store, T0 + 50, "three")  # 50 s after start, pause 37 s: new episode
    e4 = add(store, T0 + 55, "four")
    for t in range(62, 106, 7):  # steady talk, short pauses, up to 55 s after e3 began
        assert not add(store, T0 + t, "talk")["new_episode"]
    e5 = add(store, T0 + 111, "five")  # 61 s after e3 began: span reached, new episode
    assert e1["new_episode"] and not e2["new_episode"]
    assert e3["new_episode"] and e3["closed"] == [e1["episode_id"]]
    assert e4["episode_id"] == e3["episode_id"]
    assert e5["new_episode"] and e5["closed"] == [e3["episode_id"]]
    ep = store.get_episode(e1["episode_id"])
    assert ep["text"] == "one two" and ep["n_utterances"] == 2 and ep["closed"]


def test_close_due_by_pause_and_span(store):
    ep = add(store, T0, "hello")["episode_id"]
    assert store.close_due(T0 + 20, POLICY) == []
    assert store.close_due(T0 + 40, POLICY) == [ep]
    assert store.pending_embeddings() == [ep]
    store.set_vector(ep, unit(1))
    assert store.pending_embeddings() == []


def test_keyword_search_finds_exact_names_and_numbers(store):
    a, b, _ = seed(store)
    assert [r[0] for r in store.keyword_search("Butibori")] == [a]
    assert [r[0] for r in store.keyword_search("what did Priya say")] == [b]
    assert store.keyword_search("4")[0][0] == a
    assert "[Butibori]" in store.keyword_search("butibori")[0][2]
    assert store.keyword_search("the and of") == []  # only stopwords
    assert store.keyword_search('") OR 1=1 --') == []  # FTS syntax cannot be injected


def test_vector_search_ranks_by_cosine(store):
    a, b, c = seed(store)
    q = unit(2) * 0.9 + unit(1) * 0.1
    q /= np.linalg.norm(q)
    ranked = store.vector_search(q, k=3)
    assert [r[0] for r in ranked] == [b, a, c]
    assert ranked[0][1] > 0.9 > ranked[2][1]


def test_hybrid_search_reports_its_evidence(store):
    a, b, _ = seed(store)
    hits = store.search("Priya budget", unit(2), k=3)
    assert hits[0]["id"] == b
    assert hits[0]["keyword_match"] and hits[0]["vector_similarity"] > 0.95
    others = {h["id"]: h for h in hits[1:]}
    assert not others[a]["keyword_match"] and abs(others[a]["vector_similarity"]) < 0.3
    # Keyword-only (no vector) still works, e.g. before the embedder is up.
    assert store.search("Butibori", None)[0]["id"] == a


def test_delete_episode_erases_text_keywords_and_vector(store):
    a, b, _ = seed(store)
    assert store.delete_episode(a)
    assert store.get_episode(a) is None
    assert store.keyword_search("Rohan") == []
    assert a not in [r[0] for r in store.vector_search(unit(1), k=10)]
    assert store.stats()["utterances"] == 2  # seed has 4; episode A held 2
    assert not store.delete_episode(a)


def test_forget_since_is_exact(store):
    ep = add(store, T0, "keep this part")["episode_id"]
    add(store, T0 + 20, "forget this part")
    store.close_all()
    store.set_vector(ep, unit(5))
    later = add(store, T0 + 100, "and all of this")["episode_id"]
    res = store.forget_since(T0 + 20)
    assert res["utterances"] == 2 and res["episodes_removed"] == 1
    assert res["reembed"] == [ep]  # its vector described text that no longer exists
    kept = store.get_episode(ep)
    assert kept["text"] == "keep this part" and not kept["embedded"]
    assert store.get_episode(later) is None
    assert store.keyword_search("forget") == [] and store.keyword_search("keep")
    assert store.vector_search(unit(5)) == []


def test_stale_vector_is_not_stored(store):
    ep = add(store, T0, "first words")["episode_id"]
    text = store.episode_text(ep)
    add(store, T0 + 5, "more words arrive while embedding")
    assert not store.set_vector(ep, unit(1), expect_text=text)
    assert store.set_vector(ep, unit(1), expect_text=store.episode_text(ep))


def test_retention_purges_old_episodes(store):
    a, b, c = seed(store)
    assert store.purge_before(T0 + 300) == 2
    assert [e["id"] for e in store.timeline()] == [c]
    assert store.keyword_search("Rohan") == [] and store.keyword_search("Priya") == []
    assert [r[0] for r in store.vector_search(unit(1), k=5)] == [c]


def test_timeline_is_newest_first_and_pages(store):
    a, b, c = seed(store)
    assert [e["id"] for e in store.timeline()] == [c, b, a]
    page2 = store.timeline(limit=2, before=store.get_episode(b)["started_at"])
    assert [e["id"] for e in page2] == [a]
    assert [u["text"] for u in store.get_episode(a)["utterances"]][1] == "It is at Butibori near the depot"


def test_wipe_removes_everything_and_rotates_key(tmp_path, monkeypatch):
    monkeypatch.delenv("KIN_DB_KEY", raising=False)
    keys = {}
    monkeypatch.setattr("kin.memory.keys.get_key", lambda path, create: keys.setdefault("k", "12" * 32))
    monkeypatch.setattr("kin.memory.keys.delete_key", lambda path: keys.pop("k", None))
    s = Store(tmp_path / "kin.db", KEY)
    seed(s)
    s.wipe()
    assert s.stats()["episodes"] == 0 and s.timeline() == []
    add(s, T0, "fresh start")
    s.close()
    with pytest.raises(RuntimeError):
        Store(tmp_path / "kin.db", KEY)  # the old key no longer opens the new file
    assert Store(tmp_path / "kin.db", "12" * 32).stats()["utterances"] == 1


def test_command_flag_and_redaction_count_are_kept(store):
    ep = add(store, T0, "Remind me to call Rohan", is_command=True, redactions=1)["episode_id"]
    (u,) = store.get_episode(ep)["utterances"]
    assert u["command"] and u["redactions"] == 1 and u["speaker"] == "unknown"


def test_rescrub_applies_a_newer_filter_to_stored_text(store):
    from kin.perception.filters import redact

    a = add(store, T0, "My credit card number is 1234567")["episode_id"]
    b = add(store, T0 + 200, "Lunch at 1130 with Priya")["episode_id"]
    store.close_all()
    store.set_vector(a, unit(1))
    store.set_vector(b, unit(2))
    assert store.rescrub(redact) == [a]
    ep = store.get_episode(a)
    assert "1234567" not in ep["text"] and ep["utterances"][0]["redactions"] == 1
    assert not ep["embedded"] and store.pending_embeddings() == [a]  # re-indexed from clean text
    assert store.keyword_search("1234567") == []
    assert store.get_episode(b)["embedded"]  # untouched
    assert store.rescrub(redact) == []  # idempotent


def test_tasks_and_reminders(store):
    t1 = store.add_task("Call the landlord", source="command", due_at=T0 + 3600, due_has_time=True, now=T0)
    t2 = store.add_task("Send it", source="heard", status="candidate", quote="I will send it by Friday.", now=T0)
    assert [t["id"] for t in store.tasks()] == [t1, t2]  # due first, then undated
    assert store.set_task_status(t2, "open") and store.get_task(t2)["status"] == "open"
    assert store.set_task_status(t1, "done") and store.get_task(t1)["done_at"]
    assert [t["id"] for t in store.tasks(("done",))] == [t1]
    assert [t["id"] for t in store.tasks_created_between(T0 - 1, T0 + 1)] == [t1, t2]
    assert store.delete_task(t2) and store.get_task(t2) is None
    r1 = store.add_reminder("leave", T0 + 60, now=T0)
    r2 = store.add_reminder("later", T0 + 600, now=T0)
    assert store.take_due_reminders(T0 + 61) == [dict(store.reminders(("fired",))[0])]
    assert store.take_due_reminders(T0 + 62) == []  # fires once
    assert store.cancel_reminder(r2) and not store.cancel_reminder(r2)
    assert store.stats()["pending_reminders"] == 0 and r1


def test_deleting_an_episode_detaches_its_tasks(store):
    ep = add(store, T0, "I will send the drawings by Friday")
    cand = store.add_task("Send the drawings", source="heard", status="candidate", episode_id=ep["episode_id"],
                          utterance_id=ep["utterance_id"], quote="I will send the drawings by Friday")  # fmt: skip
    kept = store.add_task("Send the drawings", source="heard", status="open", episode_id=ep["episode_id"],
                          quote="I will send the drawings by Friday")  # fmt: skip
    store.delete_episode(ep["episode_id"])
    assert store.get_task(cand) is None
    assert store.get_task(kept)["quote"] is None and store.get_task(kept)["status"] == "open"


def test_rescrub_covers_tasks_and_reminders(store):
    from kin.perception.filters import redact

    t = store.add_task("Tell Kiran my card number is 1234567", source="note")
    store.add_reminder("my PIN is 4455", T0)
    store.rescrub(redact)
    assert "1234567" not in store.get_task(t)["title"]
    assert "4455" not in store.reminders()[0]["text"]
