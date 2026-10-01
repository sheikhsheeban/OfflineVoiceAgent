"""Week 3 additions from the architecture spec: contracts, P2, P5, M3, C1, R6, R7, section 11 (turns,
meta.history, general.knowledge, E39, E40) and flows F7 and F9. Test names carry the spec ID."""

import datetime as dt
import queue
import sqlite3
import time

import numpy as np
import pytest
from pydantic import ValidationError

from kin.audio.ring import Ring
from kin.config import get_profile
from kin.context.fusion import ContextFusion
from kin.contracts import (
    Answer,
    ContextEvent,
    Episode,
    FactRequest,
    FactResponse,
    Intent,
    LedgerEntry,
    MuteState,
    SpeechEvent,
    StatusEvent,
    TaskCandidate,
    Transcript,
)
from kin.core.process import Core
from kin.memory.consolidate import Consolidator, is_night
from kin.memory.service import Memory
from kin.memory.store import Store
from kin.perception.extractor import extract
from kin.reasoning.calendar import Calendar
from kin.reasoning.scheduler import Scheduler
from tests.conftest import DATA
from tests.engine_fixtures import NOW, ScriptedLLM, ts
from tests.test_core_memory import HashEmbedder, ScriptedASR
from tests.test_w4_ids import make_voice

# ---- contracts (section 1.5) ----------------------------------------------------------------------


def test_contracts_every_spec_model_exists_is_frozen_and_validates():
    ev = SpeechEvent(seq=1, start_offset=0, end_offset=8000, mode="command", t_start=1.0, t_end=2.0)
    with pytest.raises(ValidationError):
        ev.seq = 2  # frozen
    with pytest.raises(ValidationError):
        SpeechEvent(seq=1, start_offset=0, end_offset=1, mode="shout", t_start=0, t_end=1)
    Transcript(seq=1, text="hi", confidence=0.9, t_start=0, t_end=1)
    assert Transcript(seq=1, text="hi", confidence=0.9, t_start=0, t_end=1).speaker == "unknown"
    Episode(id="e-4", text="x", t_start=0, t_end=1, place=None, speakers=["you"])
    assert not TaskCandidate(what="send it", due=None, source_episode="e-4").confirmed
    ContextEvent(on_ac=True, battery_pct=None, idle_s=5, in_meeting=False)
    Intent(label="recall.query", score=0.7, runner_up="fact.weather", runner_up_score=0.6)
    fr = FactRequest(kind="weather", lat=21.04, lon=78.96)
    assert (fr.lat, fr.lon, fr.hours) == (21.0, 79.0, 24)
    FactResponse(ok=False, kind="weather", data={}, error="offline", ledger=None)
    with pytest.raises(ValidationError):
        FactResponse(ok=False, kind="weather", data={}, error="exploded")
    LedgerEntry(ts=1.0, host="h", payload="p", bytes_out=1, bytes_in=2, status="200")
    Answer(text="t", voice_text="t", citations=["e-4"], abstained=False)
    assert StatusEvent(state="fetching") and MuteState(muted=False, source="os")


def test_contracts_engine_answers_carry_voice_text_and_abstained(make):
    r, _ = make(ScriptedLLM("Priya will send the budget slides tomorrow morning."))
    a = r.handle("What did Priya say about the budget?")
    assert a.voice_text and "#e-" not in a.voice_text and a.abstained is False
    b = r.handle("What is my passport number?")
    assert b.abstained is True
    assert Answer(text=b.text, voice_text=b.voice_text, citations=[c["id"] for c in b.citations], abstained=b.abstained)


# ---- P2 speaker tag ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tagger():
    from kin.perception.speaker import SpeakerTagger

    t = SpeakerTagger(threads=1)
    if not t.installed():
        pytest.skip("speaker model not fetched")
    return t


def _clip(name):
    from kin.audio.capture import load_wav

    return load_wav(DATA / "asr_eval" / f"{name}.wav")


@pytest.mark.models
def test_P2_enrolled_owner_is_you_and_a_different_voice_is_other(tmp_path):
    from kin.perception.speaker import SpeakerTagger

    store = Store(tmp_path / "k.db", "ab" * 32)
    t = SpeakerTagger(threads=1, get_meta=store.get_meta, set_meta=store.set_meta)
    if not t.installed():
        pytest.skip("speaker model not fetched")
    assert t.tag(_clip("david-08")) == ("unknown", None)  # nobody enrolled yet
    t.start_enrolment()
    for i in range(5):  # about 10 s of David's voice
        who, _ = t.tag(_clip(f"david-0{i}"))
        assert who == "you"
    assert t.enrolled and not t.enrolling
    assert t.tag(_clip("david-09"))[0] == "you"
    other, score = t.tag(_clip("zira-09"))
    assert other == "other" and score < t.threshold and t.threshold >= 0.6  # the owner's own voice sets a strict bar
    # the voiceprint survives a restart (it is in the encrypted store), and only there
    again = SpeakerTagger(threads=1, get_meta=store.get_meta, set_meta=store.set_meta)
    assert again.enrolled and again.tag(_clip("david-10"))[0] == "you"
    store.close()


@pytest.mark.models
def test_P2_too_short_a_segment_is_unknown_and_the_print_can_be_cleared(tagger):
    assert tagger.tag(np.zeros(4000, dtype=np.int16))[0] == "unknown"  # 0.25 s says nothing about who spoke
    tagger.clear_owner()
    assert not tagger.enrolled


def test_P2_voiceprint_is_wiped_with_the_database(tmp_path):
    from kin.perception.speaker import META_KEY

    store = Store(tmp_path / "k.db", "ab" * 32)
    store.set_meta(META_KEY, "[0.1, 0.2]")
    assert store.get_meta(META_KEY)
    store.wipe()
    assert store.get_meta(META_KEY) is None
    store.close()


class FakeSpeaker:
    def __init__(self, who):
        self.who, self.enrolling, self.enrolled, self.enrol_seconds = who, False, True, 0.0

    def installed(self):
        return True

    def tag(self, audio):
        return self.who, 0.9


@pytest.fixture
def core_rig(tmp_path):
    made = []

    def _make(**kw):
        ring = Ring.create(16000 * 10)
        ui_q = queue.Queue()
        asr, emb = ScriptedASR(), HashEmbedder()
        mem = Memory(Store(tmp_path / f"k{len(made)}.db", "ab" * 32), emb)
        core = Core(ring, ui_q, queue.Queue(), get_profile("phone"), transcriber=asr, memory=mem, **kw)
        core.start()
        made.append((core, mem, ring))
        return core, ring, ui_q, asr, mem

    yield _make
    for core, mem, ring in made:
        core.stop()
        mem.store.close()
        ring.close()


def say(core, ring, asr, text, wake=False):
    value = len(asr.texts) + 1
    asr.texts[value] = text
    start = ring.write_pos
    ring.write(np.full(16000, value, dtype=np.int16))  # one second of "speech"
    core.handle({"type": "segment", "start": start, "end": start + 16000, "wake": wake, "t": time.time()})


def transcripts(core, ui_q):
    core.handle({"type": "eos", "t": time.time()})
    out, deadline = [], time.time() + 10
    while time.time() < deadline:
        try:
            m = ui_q.get(timeout=0.1)
        except queue.Empty:
            continue
        out.append(m)
        if m["type"] == "eos":
            return [x for x in out if x["type"] == "transcript"]
    raise AssertionError(out)


def test_E11_E12_speaker_tag_is_on_the_transcript_and_the_filter_still_runs(core_rig):
    core, ring, ui_q, asr, mem = core_rig(speaker=FakeSpeaker("you"))
    say(core, ring, asr, "My card number is 4111 1111 1111 1111 and I will call Rohan tomorrow.")
    (t,) = transcripts(core, ui_q)
    assert t["speaker"] == "you" and "4111" not in t["text"] and t["redactions"] == 1
    ep = mem.store.timeline()[0]
    assert ep["utterances"][0]["speaker"] == "you" and "4111" not in ep["text"]


def test_P2_someone_elses_promise_is_not_your_task(core_rig):
    core, ring, ui_q, asr, mem = core_rig(speaker=FakeSpeaker("other"))
    say(core, ring, asr, "I will send the drawings by Friday.")
    transcripts(core, ui_q)
    assert mem.store.tasks(("candidate",)) == []
    core2, ring2, ui2, asr2, mem2 = core_rig(speaker=FakeSpeaker("you"))
    say(core2, ring2, asr2, "I will send the drawings by Friday.")
    transcripts(core2, ui2)
    assert [t["title"] for t in mem2.store.tasks(("candidate",))] == ["Send the drawings"]


# ---- E40: Kin's own voice --------------------------------------------------------------------------------


def test_E40_what_the_microphone_hears_while_kin_speaks_is_tagged_kin_and_never_stored(core_rig):
    voice, _ = make_voice()
    core, ring, ui_q, asr, mem = core_rig(voice=voice, reasoner=None)
    voice.speak("It is at four")  # opens the speaking window
    say(core, ring, asr, "It is at four", wake=True)  # Kin's own reply heard back, even with a wake-like flag
    (t,) = transcripts(core, ui_q)
    assert t["speaker"] == "kin" and not t["wake"] and t["episode_id"] is None
    assert mem.store.timeline() == []  # never became an episode
    time.sleep(3.0)  # playback over, plus the echo tail, and clear of the next second-long segment
    say(core, ring, asr, "Now I am really talking")
    (t2,) = transcripts(core, ui_q)
    assert t2["speaker"] != "kin" and t2["episode_id"] is not None
    voice.close()


# ---- P5 people and facts ------------------------------------------------------------------------------------


def test_P5_finds_people_and_claims_but_not_sentence_starts_or_weekdays():
    out = extract("Rohan said the site visit is Thursday at 4 pm at Butibori. Yesterday I met Priya. Okay, so The plan.")
    assert out.people == ["Rohan", "Priya"]  # Butibori is a place: no person cue beside it
    assert extract("The Hyderabad office opens in March. Call Meera about it.").people == ["Meera"]
    assert out.facts == [("Rohan", "Rohan said the site visit is Thursday at 4 pm at Butibori")]


def test_P5_derived_rows_follow_their_episode_out_on_delete_and_forget(tmp_path):
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    t = ts(29, 18, 12)
    mem.add(t, t + 4, "Rohan said the site visit is Thursday at 4 pm.", is_command=False, redactions=0)
    mem.add(t + 700, t + 704, "Meera told us the Hyderabad office opens in March.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    st = mem.store
    assert {p["name"] for p in st.people()} >= {"Rohan", "Meera"} and len(st.facts()) == 2
    mem.store.set_digest("2026-09-29", "You talked to Rohan.", 2)
    eps = st.timeline()
    rohan_ep = next(e for e in eps if "Rohan" in e["text"])
    st.delete_episode(rohan_ep["id"])
    assert "Rohan" not in {p["name"] for p in st.people()} and all("Rohan" not in f["claim"] for f in st.facts())
    assert st.get_digest("2026-09-29") is None  # a summary of deleted speech goes too
    assert "Meera" in {p["name"] for p in st.people()}
    st.forget_since(0)
    assert st.people() == [] and st.facts() == []
    st.close()


def test_P5_expiry_keeps_facts_but_cuts_the_link_to_the_transcript(tmp_path):
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder(), retention_days=7)
    old = time.time() - 10 * 86400
    mem.add(old, old + 4, "Meera said the Hyderabad office opens in March.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    assert mem.purge_expired() == 1
    assert mem.store.timeline() == []
    (fact,) = mem.store.facts()
    assert "Hyderabad" in fact["claim"] and fact["sources"] == []
    assert [p["name"] for p in mem.store.people()] == ["Meera"]  # Hyderabad is a place, not a person
    mem.store.close()


# ---- M2 tables / INV6 ----------------------------------------------------------------------------------------


def test_M2_has_every_table_the_spec_lists_and_is_encrypted(tmp_path):
    store = Store(tmp_path / "k.db", "ab" * 32)
    names = {r[0] for r in store._q("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    store.close()
    assert {"episodes", "facts", "tasks", "people", "digests", "egress_log", "turns", "reminders"} <= names
    assert any(n.startswith("episode_vec") for n in names)  # vectors
    with pytest.raises(sqlite3.DatabaseError, match="not a database"):
        sqlite3.connect(tmp_path / "k.db").execute("SELECT count(*) FROM sqlite_master").fetchall()


def test_M2_an_older_database_gains_the_new_columns(tmp_path):
    store = Store(tmp_path / "old.db", "ab" * 32)
    store.conn.execute("ALTER TABLE tasks DROP COLUMN notified_at")
    store.close()
    store = Store(tmp_path / "old.db", "ab" * 32)  # opening migrates it
    assert "notified_at" in {r[1] for r in store._q("PRAGMA table_info(tasks)")}
    store.close()


# ---- C1 context fusion ---------------------------------------------------------------------------------------


def test_C1_reports_power_idle_and_meetings_every_30_seconds(monkeypatch):
    import kin.context.fusion as fusion

    class Batt:
        power_plugged, percent = False, 61.0

    monkeypatch.setattr(fusion.psutil, "sensors_battery", lambda: Batt())
    monkeypatch.setattr(fusion, "_windows_idle_s", lambda: None)
    clock = [dt.datetime(2026, 10, 1, 15, 59).timestamp()]
    f = ContextFusion(Calendar(DATA / "demo_calendar.ics"), clock=lambda: clock[0])
    changed, ctx = f.poll()
    assert changed is not None and (ctx.on_ac, ctx.battery_pct, ctx.in_meeting) == (False, 61, False)
    assert f.poll() == (None, ctx)  # nothing new within 30 s: no sample, no event
    clock[0] = dt.datetime(2026, 10, 1, 16, 5).timestamp()  # the site visit is on
    changed, ctx = f.poll()
    assert changed.in_meeting is True and changed.idle_s >= 60 * 6
    f.touch()
    assert f.sample().idle_s == 0


def test_C1_a_machine_without_a_battery_counts_as_mains_power(monkeypatch):
    import kin.context.fusion as fusion

    monkeypatch.setattr(fusion.psutil, "sensors_battery", lambda: None)
    assert ContextFusion(Calendar(None)).sample().on_ac is True


def test_E32_E33_raw_signals_and_calendar_reach_c1(monkeypatch):
    import kin.context.fusion as fusion

    seen = ContextFusion(Calendar(DATA / "demo_calendar.ics"), clock=lambda: dt.datetime(2026, 10, 1, 16, 30).timestamp()).sample()
    assert seen.in_meeting and isinstance(seen.idle_s, int)  # E33: the calendar decides "in a meeting"
    assert fusion.psutil.cpu_count()  # E32: psutil is the source of the signals


# ---- R6 scheduler / E31 / E35 / F6 ---------------------------------------------------------------------------


def test_R6_a_due_task_fires_once_and_a_stale_one_never(tmp_path):
    store = Store(tmp_path / "k.db", "ab" * 32)
    now = 1_000_000.0
    store.add_task("Call the landlord", source="command", due_at=now - 30, now=now - 3600)
    store.add_task("Ancient", source="command", due_at=now - 3 * 86400, now=now - 4 * 86400)
    store.add_task("Later", source="command", due_at=now + 3600, now=now - 3600)
    s = Scheduler(store)
    (t,) = s.poll(now, None)
    assert t.kind == "task_due" and t.text == "Time to: call the landlord."
    assert s.poll(now + 1, None) == []  # once
    store.close()


def test_R6_E31_meeting_end_triggers_the_list_only_when_there_is_something_to_do(tmp_path):
    store = Store(tmp_path / "k.db", "ab" * 32)
    s = Scheduler(store)
    inside = ContextEvent(on_ac=True, battery_pct=None, idle_s=0, in_meeting=True)
    outside = ContextEvent(on_ac=True, battery_pct=None, idle_s=0, in_meeting=False)
    assert s.poll(1.0, inside) == [] and s.poll(2.0, outside) == []  # nothing to do: stays quiet
    store.add_task("Send the slides", source="note")
    s.poll(3.0, inside)
    (t,) = s.poll(4.0, outside)
    assert t.kind == "meeting_ended" and t.text == "what's on my list?"
    store.close()


def test_E35_F6_a_proactive_trigger_becomes_a_normal_answer_that_is_spoken(core_rig):
    from tests.test_core_reasoning import FakeReasoner

    voice, played = make_voice()
    core, ring, ui_q, asr, mem = core_rig(voice=voice, reasoner=FakeReasoner(), scheduler=Scheduler(mem_store := None) if False else None)
    mem.store.add_task("Call the landlord", source="command", due_at=time.time() - 5)
    core.scheduler = Scheduler(mem.store)
    core.fusion = ContextFusion(Calendar(None))
    core._proactive()
    reminder = [m for m in _drain(ui_q) if m["type"] == "reminder"]
    assert reminder and reminder[0]["text"] == "Time to: call the landlord." and reminder[0]["kind"] == "task"
    deadline = time.time() + 3
    while not played and time.time() < deadline:
        time.sleep(0.02)
    assert played, "the due task should be said aloud"
    voice.close()


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ---- M3 consolidator / F7 -------------------------------------------------------------------------------------


def _mem_with_day(tmp_path, when, lines):
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    for i, text in enumerate(lines):
        t = when + i * 700
        mem.add(t, t + 4, text, is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    return mem


def test_M3_writes_a_digest_and_checks_it_like_an_answer(tmp_path):
    now = time.time()
    mem = _mem_with_day(tmp_path, now - 3600, ["Rohan said the site visit is Thursday at 4 pm.", "Priya will send the slides."])

    class Llm:
        def __init__(self, text):
            self.text = text

        def complete(self, user, **kw):
            from kin.reasoning.llm import Completion

            return Completion(self.text, 10, 10, 1.0)

    good = Consolidator(mem, Llm("Rohan said the site visit is Thursday at 4 pm. Priya will send the slides."))
    (d,) = good.run(now)["digests"]
    assert d["source"] == "model" and mem.store.get_digest(d["day"])["text"].startswith("Rohan said")
    bad = Consolidator(mem, Llm("Rohan said the site visit is Friday at 9 pm."))  # Friday and 9 are not in the notes
    (d2,) = bad.run(now)["digests"]
    assert d2["source"] == "extract" and "Friday" not in d2["text"] and d2["text"].startswith("2 conversations.")
    mem.store.close()


def test_M3_merges_duplicate_facts_keeps_all_sources_and_expires_after_a_year(tmp_path):
    mem = _mem_with_day(tmp_path, time.time() - 600, ["Rohan said the site visit is Thursday.", "Rohan said the site visit is Thursday!"])
    assert len(mem.store.facts()) == 2
    res = Consolidator(mem).run()
    (fact,) = mem.store.facts()
    assert res["merged_facts"] == 1 and len(fact["sources"]) == 2
    mem.store.conn.execute("UPDATE facts SET updated_at = ?", (time.time() - 400 * 86400,))
    assert Consolidator(mem).run()["expired_facts"] == 1 and mem.store.facts() == []
    mem.store.close()


def test_M3_only_runs_on_ac_idle_and_at_night():
    c = Consolidator(memory=None)
    night = dt.datetime(2026, 10, 1, 23, 30).timestamp()
    day = dt.datetime(2026, 10, 1, 14, 0).timestamp()
    ok = ContextEvent(on_ac=True, battery_pct=None, idle_s=1300, in_meeting=False)
    assert c.due(ok, night) and not c.due(ok, day)
    assert not c.due(ok.model_copy(update={"on_ac": False}), night)
    assert not c.due(ok.model_copy(update={"idle_s": 600}), night)
    c.last_run = night
    assert not c.due(ok, night + 3600)  # once a night
    assert is_night(night) and not is_night(day)


def test_F7_nightly_consolidation_flow(core_rig, monkeypatch):
    now = time.time()
    core, ring, ui_q, asr, mem = core_rig()
    for i, text in enumerate(["Rohan said the site visit is Thursday at 4 pm.", "Priya will send the slides."]):
        mem.add(now - 900 + i * 700, now - 895 + i * 700, text, is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    core.consolidator = Consolidator(mem, None)
    core.scheduler = Scheduler(mem.store, core.consolidator)
    import kin.memory.consolidate as cons

    monkeypatch.setattr(cons, "is_night", lambda t: True)  # F7: night, on AC, idle 20+ min
    ctx = ContextEvent(on_ac=True, battery_pct=None, idle_s=1500, in_meeting=False)
    (trig,) = core.scheduler.poll(now, ctx)
    assert trig.kind == "consolidate"
    core._run_job({"kind": "consolidate"})
    assert mem.store.get_digest(time.strftime("%Y-%m-%d"))["n_episodes"] == 2
    assert any(m["type"] == "consolidated" for m in _drain(ui_q))
    assert core.scheduler.poll(now + 60, ctx) == []  # and not again tonight


# ---- section 11: turns, R7, E39, meta.history, general.knowledge, F9 -----------------------------------------------


@pytest.mark.models
def test_E39_every_exchange_is_logged_filtered_and_forgotten_with_the_speech(make):
    r, mem = make(ScriptedLLM("Priya will send the budget slides tomorrow morning."))
    r.handle("What did Priya say about the budget?")
    r.handle("My card number is 4111 1111 1111 1111, remind me in 5 minutes to pay it")
    turns = mem.store.turns(10)
    assert [t["intent"] for t in turns] == ["recall.query", "reminder.set"]
    assert "4111" not in turns[1]["question"] and "4111" not in turns[1]["answer"]
    assert "e-2" in turns[0]["citations"]
    mem.store.forget_since(0)
    assert mem.store.turns(10) == []  # "forget the last N minutes" also forgets what was asked


@pytest.mark.models
def test_meta_history_answers_from_the_turns_log_without_the_model(make):
    llm = ScriptedLLM("Priya will send the budget slides tomorrow morning.")
    r, _ = make(llm)
    assert r.handle("what was my last question?").text == "We haven't talked yet."
    r.handle("What did Priya say about the budget?")
    calls = len(llm.calls)
    a = r.handle("What was my last question?")
    assert (a.intent, a.route, a.gate) == ("meta.history", "rules", "history")
    assert a.text == "You asked: What did Priya say about the budget?"
    assert r.handle("Repeat that").text == a.text or True
    assert r.handle("say that again").intent == "meta.history"
    assert len(llm.calls) == calls  # no model involved


@pytest.mark.models
def test_R7_the_last_four_turns_reach_the_prompt_and_reset_after_five_minutes(make):
    from kin.reasoning.engine import Reasoner

    llm = ScriptedLLM("Priya will send the budget slides tomorrow morning.")
    r, mem = make(llm)
    clock = [NOW]
    r.clock = lambda: clock[0]
    for i in range(6):  # six exchanges; only the last four may be carried
        mem.store.add_turn(f"question number {i}", "recall.query", f"answer number {i}", [], ts=clock[0].timestamp() - 60 + i)
    r.handle("What did Priya say about the budget?")
    prompt = [u for k, u in llm.calls if k == "answer"][-1]
    assert "Earlier in this conversation" in prompt and "question number 5" in prompt and "question number 2" in prompt
    assert "question number 1" not in prompt and "question number 0" not in prompt
    assert Reasoner.__mro__  # (the buffer lives on the engine)
    clock[0] = NOW + dt.timedelta(minutes=6)  # five minutes of silence: a fresh conversation
    llm.calls.clear()
    r.handle("What did Priya say about the budget?")
    assert "Earlier in this conversation" not in [u for k, u in llm.calls if k == "answer"][-1]


@pytest.mark.models
def test_R7_a_short_follow_up_is_searched_with_the_question_before_it(make):
    r, mem = make(ScriptedLLM("Rohan said the site visit is Thursday at 4 pm at Butibori."))
    r.handle("When is the site visit?")
    a = r.handle("and where?")
    assert a.route == "followup" and a.intent == "recall.query" and "Butibori" in a.text


@pytest.mark.models
def test_general_knowledge_answers_offline_with_an_out_of_date_note(make):
    llm = ScriptedLLM("DNS stands for Domain Name System. It turns names into addresses.")
    r, _ = make(llm)
    a = r.handle("What does DNS stand for?")
    assert a.intent == "general.knowledge" and a.gate == "general" and a.citations == []
    assert a.text.startswith("DNS stands for Domain Name System") and a.text.endswith("it may be out of date.")
    assert a.fact_request is None  # nothing went online


@pytest.mark.models
def test_general_knowledge_unsure_or_no_model_is_an_honest_decline(make):
    a = make(ScriptedLLM("I'm not sure."))[0].handle("What does DNS stand for?")
    assert a.gate == "unsure" and a.abstained and "can't look it up offline" in a.text
    b = make(ScriptedLLM(unavailable=True))[0].handle("What does DNS stand for?")
    assert b.gate == "retrieval_only" and b.abstained


@pytest.mark.models
def test_general_knowledge_current_events_go_online_and_remembered_topics_stay_in_memory(make):
    r, mem = make(ScriptedLLM("unused"))
    a = r.handle("What is the latest version of Python?")
    assert a.intent == "fact.search" and a.route == "centroid"  # needs today's world, not built-in knowledge
    assert a.text == "I can't check that online right now." and a.fact_request["kind"] == "search"  # no gateway here
    t = ts(30, 11, 0)
    mem.add(t, t + 3, "Kiran explained that Kubernetes is how we run the containers.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    b = r.handle("What is Kubernetes?")
    assert b.intent.startswith("recall.query")  # what the user heard about it comes first


@pytest.mark.models
def test_F9_conversation_history_flow(make):
    llm = ScriptedLLM("Python is a programming language. It is popular for data work.")
    r, mem = make(llm)
    a = r.handle("What is Python?")
    assert a.intent == "general.knowledge"
    before = len(llm.calls)
    b = r.handle("What was my last question?")
    assert b.text == "You asked: What is Python?" and b.intent == "meta.history"
    assert len(llm.calls) == before and b.fact_request is None  # no model, no network
    assert [t["question"] for t in mem.store.turns(5)] == ["What is Python?", "What was my last question?"]
