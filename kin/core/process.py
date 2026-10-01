"""kin-core: perception, memory and reasoning, with no network.

Speech events from kin-audio are transcribed from the shared ring (then the
samples are wiped), filtered for codes and card numbers, pushed to the UI as a
live transcript, and written to the encrypted memory store as episodes.

Wake-word commands are answered by the reasoning engine (router, tools, local
language model, confidence gate); commitments heard in ordinary speech become
task candidates; reminders fire from the once-a-second tick.

Also here: the speaker tag (P2), context fusion (C1), the proactive scheduler
(R6), the nightly consolidator (M3), the status controller (O1) and the voice
reply (O3). kin-core has no network: it reaches kin-gateway only over local IPC.

Every heavy model call (ASR, embeddings, the language model) goes through one
priority queue served by one worker thread, the "heavy lane":
  0  commands, answers and interactive searches
  1  ambient speech
  2  episode embeddings
  3  end-of-replay marker (after everything else)
The UI's requests (timeline, search, delete, forget, wipe) arrive on `cmd_q`.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import os
import queue
import threading
import time

import psutil

from kin.audio.ring import Overrun, Ring
from kin.config import SAMPLE_RATE, Profile, load_model_spec
from kin.sync import wait_for

log = logging.getLogger("kin.core")

PRIO_COMMAND, PRIO_AMBIENT, PRIO_BACKGROUND, PRIO_LAST = 0, 1, 2, 3
PROACTIVE_SOURCES = ("voice", "proactive")  # answers spoken aloud without being asked to
_STOP = (99, 0, None)
PURGE_EVERY_S = 3600


class Core:
    def __init__(
        self,
        ring: Ring,
        ui_q,
        ctl_q,
        profile: Profile,
        transcriber=None,
        memory=None,
        asr_model=None,
        reasoner=None,
        voice=None,
        speaker=None,
        status=None,
        fusion=None,
        scheduler=None,
        consolidator=None,
        audio_cmd_q=None,
    ):
        from kin.perception.asr import Transcriber, wake_anywhere_pattern, wake_phrase_pattern, wake_text_pattern

        self.ring = ring
        self.ui_q = ui_q
        self.ctl_q = ctl_q
        self.profile = profile
        self.asr = transcriber or Transcriber(threads=profile.asr_threads, model=asr_model)
        self.memory = memory
        self.reasoner = reasoner
        self.voice = voice
        self.speaker = speaker
        self.status = status
        self.fusion = fusion
        self.scheduler = scheduler
        self.consolidator = consolidator
        self.audio_cmd_q = audio_cmd_q
        self.muted = False
        self._llm_loaded = False
        self.wake_re = wake_phrase_pattern(load_model_spec()["wake_word"]["phrase"])
        self.wake_text_re = wake_text_pattern(load_model_spec()["wake_word"]["phrase"])
        self.wake_any_re = wake_anywhere_pattern(load_model_spec()["wake_word"]["phrase"])
        self.work: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._ids = itertools.count(1)
        self._embed_queued: set[int] = set()
        self._embed_lock = threading.Lock()
        self._last_purge = 0.0
        self.worker = threading.Thread(target=self._work_loop, name="heavy-lane", daemon=True)

    def start(self) -> None:
        self.asr.warm_up()
        self.worker.start()
        if self.memory is not None:
            # Text stored under an older filter is re-filtered (and re-indexed).
            changed = self.memory.rescrub_if_needed()
            if changed:
                log.info("filter: removed sensitive numbers from %d stored episode(s)", len(changed))
            n = self.memory.backfill_derived_if_needed()
            if n:
                log.info("people and facts derived for %d earlier episode(s)", n)
            # Episodes left open by a previous run are finished; embed any that lack a vector.
            self.memory.store.close_all()
            self.tick(force_purge=True)

    def _put(self, prio: int, job: dict) -> None:
        self.work.put((prio, next(self._seq), job))

    def handle(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "segment":
            self._put(PRIO_COMMAND if ev.get("wake") else PRIO_AMBIENT, {"kind": "asr", "ev": ev})
        elif kind == "eos":
            # Delivered after everything queued before it, including the embeddings
            # of the episodes this closes.
            if self.memory is not None:
                self._put(PRIO_AMBIENT, {"kind": "close_all"})
            self._put(PRIO_LAST, {"kind": "eos", "ev": ev})
        else:  # status, wake, audio_ready, mute: to the UI, and the ones that matter to O1 first
            if kind == "wake":
                if self.voice is not None:
                    self.voice.stop()  # barge-in: a new command cuts the reply off
                if self.status is not None:
                    self.status.capture_started()
                if self.fusion is not None:
                    self.fusion.touch()
            elif kind == "mute":
                self.muted = bool(ev["muted"])
                if self.status is not None:
                    self.status.set("muted", self.muted)
                if self.muted and self.voice is not None:
                    self.voice.stop()
            self.ui_q.put(ev)

    def run_heavy(self, fn, prio: int = PRIO_COMMAND, timeout: float = 30.0):
        """Run `fn` on the heavy lane (one model call at a time) and wait for it."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._put(prio, {"kind": "call", "fn": fn, "future": fut})
        return fut.result(timeout=timeout)

    def queue_embeddings(self, ids) -> None:
        with self._embed_lock:
            for ep_id in ids:
                if ep_id not in self._embed_queued:
                    self._embed_queued.add(ep_id)
                    self._put(PRIO_BACKGROUND, {"kind": "embed", "id": ep_id})

    def tick(self, force_purge: bool = False) -> None:
        """Called about once a second: close finished episodes, apply retention."""
        if self.memory is None:
            return
        self._fire_reminders()
        self._release_idle_model()
        self._proactive()
        if self.status is not None:
            self.status.tick()
        if self.voice is not None:
            self.voice.maybe_unload()
        changed = bool(self.memory.close_due())
        self.queue_embeddings(self.memory.store.pending_embeddings())
        now = time.time()
        if force_purge or now - self._last_purge > PURGE_EVERY_S:
            self._last_purge = now
            n = self.memory.purge_expired(now)
            if n:
                log.info("retention: deleted %d expired episodes", n)
                changed = True
        if changed:
            self._notify("episode_closed")

    def _fire_reminders(self) -> None:
        now = time.time()
        for r in self.memory.store.take_due_reminders(now):
            late = now - r["fire_at"] > 60  # missed while Kin was not running
            log.info("reminder: %s%s", r["text"], " (late)" if late else "")
            self.ui_q.put({"type": "reminder", "id": r["id"], "text": r["text"], "fire_at": r["fire_at"],
                           "late": late, "t": now})  # fmt: skip
            if self.voice is not None and not self.muted:
                self.voice.speak(f"Reminder: {r['text']}" + (", which was due earlier." if late else "."))
            self._notify("tasks")

    def _proactive(self) -> None:
        """C1 samples the laptop every 30 s; R6 turns changes and due times into triggers (E31, E35)."""
        if self.scheduler is None or self.fusion is None:
            return
        changed, ctx = self.fusion.poll()
        if changed is not None:
            self.ui_q.put({"type": "context", **changed.model_dump(), "t": time.time()})
        for trig in self.scheduler.poll(time.time(), ctx):
            if trig.kind == "task_due":
                log.info("task due: %s", trig.text)
                self.ui_q.put({"type": "reminder", "id": None, "text": trig.text, "fire_at": time.time(),
                               "late": False, "kind": "task", "t": time.time()})  # fmt: skip
                if self.voice is not None and not self.muted:
                    self.voice.speak(trig.text)
                self._notify("tasks")
            elif trig.kind == "meeting_ended" and self.reasoner is not None and not self.muted:
                self._put(PRIO_COMMAND, {"kind": "answer", "question": trig.text, "source": "proactive"})
            elif trig.kind == "consolidate" and self.consolidator is not None:
                self._put(PRIO_BACKGROUND, {"kind": "consolidate"})

    def shed_memory(self) -> list[str]:
        """Give back memory that can be reloaded later: the voice model and the language model, if
        neither is in use right now. Called by the watchdog before it would stop kin-core."""
        import gc

        released = []
        eng = getattr(self.voice, "engine", None)
        if eng is not None and getattr(eng, "loaded", False):
            eng.unload()
            released.append("voice model")
        llm = self.reasoner.llm if self.reasoner is not None else None
        if llm is not None and llm.loaded and llm.unload_if_idle():
            released.append("language model")
            self._llm_state()
        gc.collect()
        return released

    def _release_idle_model(self) -> None:
        llm = self.reasoner.llm if self.reasoner is not None else None
        if llm is not None and llm.maybe_unload():
            self._llm_state()

    def _llm_state(self) -> None:
        llm = self.reasoner.llm if self.reasoner is not None else None
        loaded = bool(llm and llm.loaded)
        if loaded != self._llm_loaded:
            self._llm_loaded = loaded
            self.ui_q.put({"type": "llm_state", "loaded": loaded, "t": time.time()})

    def answer(self, question: str, source: str, speak: bool | None = None) -> dict:
        """Heavy: run on the heavy lane. Pushes the answer to the UI, says it if it was asked by voice,
        and returns it."""
        if self.reasoner is None:
            raise RuntimeError("reasoning is not available (memory store missing)")
        if self.status is not None:
            self.status.set("capturing", False)
        msg = self.reasoner.handle(question).to_dict()
        msg["source"] = source
        if speak is None:
            speak = source in PROACTIVE_SOURCES
        v = self.voice
        msg["spoken"] = bool(speak and v is not None and not self.muted and v.speak(msg["voice_text"]))
        self.ui_q.put(msg)
        if msg["task"] or msg["reminder"]:
            self._notify("tasks")
        self._llm_state()
        return msg

    def _commitments(self, text: str, utterance_id: int, episode_id: int, t: float) -> None:
        import datetime as dt

        from kin.reasoning.commitments import extract

        found = extract(text, dt.datetime.fromtimestamp(t))
        for c in found:
            self.memory.store.add_task(
                c.title, source="heard", status="candidate", due_at=c.due_at.timestamp() if c.due_at else None,
                due_has_time=c.has_time, episode_id=episode_id, utterance_id=utterance_id, quote=c.sentence, now=t,
            )  # fmt: skip
        if found:
            log.info("heard %d commitment(s): %s", len(found), "; ".join(c.title for c in found))
            self._notify("tasks")

    def _send_audio(self, cmd: dict) -> None:
        if self.audio_cmd_q is not None:
            self.audio_cmd_q.put(cmd)

    def _notify(self, reason: str) -> None:
        self.ui_q.put({"type": "memory_changed", "reason": reason, "t": time.time()})

    def stop(self) -> None:
        if self.worker.is_alive():
            self.work.put(_STOP)
            self.worker.join(timeout=5)

    def _work_loop(self) -> None:
        while True:
            _, _, job = self.work.get()
            if job is None:
                return
            try:
                self._run_job(job)
            except Exception as e:  # keep the lane alive; report and move on
                log.exception("heavy-lane job failed: %s", job.get("kind"))
                if job.get("kind") == "call":
                    job["future"].set_exception(e)

    def _run_job(self, job: dict) -> None:
        kind = job["kind"]
        if kind == "asr":
            self._transcribe(job["ev"])
        elif kind == "embed":
            with self._embed_lock:
                self._embed_queued.discard(job["id"])
            if self.memory is not None and self.memory.embed_episode(job["id"]):
                self._notify("episode_embedded")
        elif kind == "call":
            job["future"].set_result(job["fn"]())
        elif kind == "close_all":
            self.queue_embeddings(self.memory.store.close_all())
            self._notify("episode_closed")
        elif kind == "answer":
            self.answer(job["question"], job["source"])
        elif kind == "consolidate":
            res = self.consolidator.run()
            self.ui_q.put({"type": "consolidated", "digests": len(res["digests"]), "merged": res["merged_facts"],
                           "expired": res["expired_episodes"], "t": time.time()})  # fmt: skip
            self._notify("consolidated")
        elif kind == "warm":
            if self.reasoner is not None:
                self.reasoner.warm()
                self._llm_state()
        elif kind == "eos":
            self.ui_q.put(job["ev"])
            self.ctl_q.put(job["ev"])

    def _transcribe(self, ev: dict) -> None:
        from kin.perception.asr import strip_wake_phrase
        from kin.perception.filters import redact

        start, end = ev["start"], ev["end"]
        queued_ms = (time.time() - ev["t"]) * 1000
        # E40: while Kin is speaking, what the microphone hears is Kin. Tag it, never store or answer it.
        seg_end = ev["t"]
        seg_start = seg_end - (end - start) / SAMPLE_RATE
        is_kin = self.voice is not None and self.voice.speaking_between(seg_start, seg_end)
        if self.fusion is not None and not is_kin:
            self.fusion.touch()
        try:
            audio = self.ring.read(start, end)
        except Overrun:
            log.warning("segment %d-%d overwritten before ASR", start, end)
            self.ui_q.put({"type": "dropped", "start_s": start / SAMPLE_RATE, "reason": "overrun"})
            return
        res = self.asr.transcribe(audio)
        speaker = "kin" if is_kin else self._tag_speaker(audio)
        # Privacy invariant: raw audio does not outlive its decode.
        del audio
        self.ring.zero(start, end)

        filtered = redact(res.text)  # codes and card numbers never reach the UI or the store
        text = filtered.text
        wake = bool(ev.get("wake")) and not is_kin
        if not wake and not is_kin and self.wake_text_re.match(text):
            # Safety net: the acoustic wake word missed (it does for some voices and microphones) but the
            # recogniser heard the phrase at the start of an ambient sentence. Reroute it as a command.
            wake = True
            ev = {**ev, "wake_miss": True}
            log.warning("wake_miss: %r heard as ambient; rerouted as a command", text[:60])
        command = strip_wake_phrase(text, self.wake_re, self.wake_any_re) if wake else None

        episode_id = None
        stored = None if is_kin else (command if wake else text)
        if self.memory is not None and stored:
            t_end = ev["t"]
            r = self.memory.add(t_end - res.audio_s, t_end, stored, is_command=wake, redactions=filtered.redactions,
                                speaker=speaker)  # fmt: skip
            episode_id = r["episode_id"]
            self.queue_embeddings(r["closed"])
            self._notify("utterance")
            if not wake and speaker != "other":  # someone else's promise is not yours
                self._commitments(stored, r["utterance_id"], episode_id, t_end)

        msg = {
            "type": "transcript",
            "id": next(self._ids),
            "text": text,
            "wake": wake,
            "mode": "command" if wake else "ambient",
            "wake_miss": bool(ev.get("wake_miss")),
            "speaker": speaker,
            "command": command,
            "redactions": filtered.redactions,
            "episode_id": episode_id,
            "forced": ev.get("forced", False),
            "start_s": round(start / SAMPLE_RATE, 2),
            "audio_s": round(res.audio_s, 2),
            "asr_ms": round(res.decode_ms, 1),
            "rtf": round(res.rtf, 3),
            "latency_ms": round(queued_ms + res.decode_ms, 1),  # end of speech -> text
            "t": time.time(),
        }
        if text:
            log.info("[%s] %s  (%.0f ms)", "CMD" if wake else "amb", text, res.decode_ms)
        self.ui_q.put(msg)
        if command and self.reasoner is not None:
            # Answer next, ahead of any ambient speech still waiting.
            self._put(PRIO_COMMAND, {"kind": "answer", "question": command, "source": "voice"})

    def _tag_speaker(self, audio) -> str:
        """P2: 'you' / 'other' / 'unknown'. While enrolling, the segment teaches the voiceprint."""
        sp = self.speaker
        if sp is None or not sp.installed():
            return "unknown"
        was_enrolling = sp.enrolling
        try:
            who, score = sp.tag(audio)
        except Exception:  # noqa: BLE001 - a speaker-model failure must not lose the transcript
            log.exception("speaker tag failed")
            return "unknown"
        if was_enrolling:
            done = sp.enrolled and not sp.enrolling
            self.ui_q.put({"type": "enrol", "seconds": round(sp.enrol_seconds, 1), "done": done,
                           "t": time.time()})  # fmt: skip
        return who

    # -- requests from kin-ui ---------------------------------------------------------

    def rpc(self, method: str, params: dict):
        if self.memory is None:
            raise RuntimeError("memory store is not available")
        store = self.memory.store
        if method == "timeline":
            return store.timeline(limit=int(params.get("limit", 50)), before=params.get("before"))
        if method == "episode":
            return store.get_episode(int(params["id"]))
        if method == "stats":
            return store.stats()
        if method == "search":
            q = str(params.get("q", "")).strip()[:200]
            if not q:
                return []
            qvec = self.run_heavy(lambda: self.memory.embedder.embed_one(q))
            return self.memory.search(q, k=int(params.get("k", 5)), qvec=qvec)
        if method == "delete_episode":
            ok = store.delete_episode(int(params["id"]))
            self._notify("deleted")
            return {"deleted": ok}
        if method == "forget":
            res = self.memory.forget_minutes(float(params.get("minutes", 10)))
            self.queue_embeddings(res["reembed"])
            self._notify("forgot")
            return res
        if method == "ask":
            q = str(params.get("q", "")).strip()[:300]
            if not q:
                raise ValueError("empty question")
            from kin.perception.asr import strip_wake_phrase

            if self.wake_re.match(q):
                q = strip_wake_phrase(q, self.wake_re) or q  # "hey jarvis, when is ..." typed like it is said
            speak = bool(params.get("speak"))
            return self.run_heavy(lambda: self.answer(q, "typed", speak=speak), timeout=60)
        if method == "tasks":
            return {"tasks": store.tasks(("candidate", "open")), "done": store.tasks(("done",))[-10:],
                    "reminders": store.reminders(("pending",))}  # fmt: skip
        if method == "task_set":
            ok = store.set_task_status(int(params["id"]), str(params["status"]))
            self._notify("tasks")
            return {"ok": ok}
        if method == "task_delete":
            ok = store.delete_task(int(params["id"]))
            self._notify("tasks")
            return {"ok": ok}
        if method == "reminder_cancel":
            ok = store.cancel_reminder(int(params["id"]))
            self._notify("tasks")
            return {"ok": ok}
        if method == "note_add":
            text = str(params.get("text", "")).strip()[:300]
            if not text:
                raise ValueError("empty note")
            task_id = store.add_task(text, source="note")
            self._notify("tasks")
            return store.get_task(task_id)
        if method == "ledger":
            return {"entries": store.egress(int(params.get("limit", 100))), "count": store.egress_count()}
        if method == "people":
            return {"people": store.people(), "facts": store.facts()}
        if method == "turns":
            return store.turns(int(params.get("limit", 20)))
        if method == "digest":
            day = str(params.get("day") or time.strftime("%Y-%m-%d"))
            return store.get_digest(day)
        if method == "consolidate":
            if self.consolidator is None:
                raise RuntimeError("consolidation is not available")
            res = self.run_heavy(lambda: self.consolidator.run(), prio=PRIO_BACKGROUND, timeout=120)
            self._notify("consolidated")
            return res
        if method == "mute":
            self._send_audio({"type": "mute", "muted": bool(params.get("muted")), "source": "ui"})
            return {"requested": bool(params.get("muted"))}
        if method == "voice":
            if self.voice is not None:
                self.voice.enabled = bool(params.get("enabled"))
                if not self.voice.enabled:
                    self.voice.stop()
            v = self.voice
            return {"enabled": bool(v and v.enabled), "available": bool(v and v.available)}
        if method == "speak":
            ok = bool(self.voice and self.voice.speak(str(params.get("text", ""))[:400]))
            return {"spoken": ok}
        if method == "enrol_start":
            if self.speaker is None or not self.speaker.installed():
                raise RuntimeError("the speaker model is not installed (run `uv run kin fetch-models`)")
            self.speaker.start_enrolment()
            return {"enrolling": True}
        if method == "enrol_cancel":
            if self.speaker is not None:
                self.speaker.cancel_enrolment()
            return {"enrolling": False}
        if method == "enrol_clear":
            if self.speaker is not None:
                self.speaker.clear_owner()
            return {"enrolled": False}
        if method == "net_probe":
            from kin import guard

            return guard.probe()
        if method == "wipe":
            if params.get("confirm") != "wipe":
                raise ValueError("wipe needs confirm='wipe'")
            store.wipe()
            if self.speaker is not None:
                self.speaker.owner = None  # the voiceprint went with the database
            self._notify("wiped")
            return {"wiped": True}
        raise ValueError(f"unknown method {method!r}")


def _serve_rpc(core: Core, cmd_q, ui_q, stop_evt) -> None:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rpc")

    def answer(req: dict) -> None:
        try:
            ui_q.put({"type": "rpc_result", "id": req["id"], "result": core.rpc(req["method"], req.get("params", {}))})
        except Exception as e:  # noqa: BLE001 - errors go back to the caller
            log.warning("rpc %s failed: %s", req.get("method"), e)
            ui_q.put({"type": "rpc_result", "id": req["id"], "error": str(e)})

    while not stop_evt.is_set():
        try:
            req = cmd_q.get(timeout=0.2)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break
        pool.submit(answer, req)
    pool.shutdown(wait=False, cancel_futures=True)


def _memory_watchdog(limit_mb: int, ui_q, stop_evt, shed=None) -> None:
    """Phone profile: fail loudly if private memory passes the limit.

    The language model is memory-mapped: its ~700 MB of weights are file-backed
    pages the OS can drop at any time, and Windows counts them in RSS. The budget
    ("<=1.1 GB thinking, mostly mmap") is therefore enforced on private resident
    memory (USS). RSS is checked first because USS takes ~30 ms to measure.
    """
    proc = psutil.Process()
    while not wait_for(stop_evt, 1.0):
        if proc.memory_info().rss / 2**20 <= limit_mb:
            continue
        private = proc.memory_full_info().uss / 2**20
        if private > limit_mb and shed is not None:
            # Graceful first: let go of what is idle (the voice model, the language model) and look again.
            released = shed()
            if released:
                log.warning("memory pressure (%.0f MB private): released %s", private, ", ".join(released))
                time.sleep(0.5)
                private = proc.memory_full_info().uss / 2**20
        if private > limit_mb:
            log.critical("private memory %.0f MB exceeds phone-profile limit %d MB", private, limit_mb)
            ui_q.put(
                {"type": "error", "text": f"kin-core private memory {private:.0f} MB > {limit_mb} MB (phone profile)"}
            )
            time.sleep(0.2)
            os._exit(3)


def _cap_native_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)


def open_reasoner(memory, profile: Profile, calendar_path: str | None, facts=None, on_state=None):
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner
    from kin.reasoning.llm import LocalLLM

    return Reasoner(memory, LocalLLM(threads=profile.core_threads), Calendar(calendar_path), facts=facts,
                    on_state=on_state)  # fmt: skip


def open_memory(db_path: str | None):
    from kin.memory.embed import Embedder
    from kin.memory.service import Memory
    from kin.memory.store import Store

    return Memory(Store.open(db_path), Embedder(threads=1))


def run(
    ring_name: str,
    events_q,
    ui_q,
    ctl_q,
    cmd_q,
    stop_evt,
    profile: Profile,
    db_path: str | None = None,
    asr_model: str | None = None,
    calendar_path: str | None = None,
    warm_llm: bool = False,
    session: str | None = None,
    authkey: bytes | None = None,
    audio_cmd_q=None,
    voice_enabled: bool = False,
) -> None:
    from kin import guard

    guard.install()  # kin-core has no network (INV3); it reaches kin-gateway over local IPC only

    from kin.logs import setup

    setup("kin-core")
    _cap_native_threads(profile.core_threads)
    ui_q.cancel_join_thread()
    ctl_q.cancel_join_thread()

    ring = Ring.attach(ring_name)
    try:
        memory = open_memory(db_path)
        st = memory.store.stats()
        log.info("memory: %d episodes, %s, vectors in %s", st["episodes"], st["cipher"], st["vector_backend"])
    except Exception as e:  # noqa: BLE001 - keep transcribing, but say so loudly
        log.critical("memory store unavailable: %s", e)
        ui_q.put({"type": "error", "text": f"Memory store unavailable: {e}"})
        memory = None

    from kin.context.fusion import ContextFusion
    from kin.memory.consolidate import Consolidator
    from kin.output.status import StatusController
    from kin.output.voice import Voice
    from kin.perception.speaker import SpeakerTagger
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.facts import FactService
    from kin.reasoning.scheduler import Scheduler

    status = StatusController(ui_q, audio_cmd_q)
    voice = Voice(enabled=voice_enabled, on_state=lambda st: ui_q.put({"type": "voice", "state": st, "t": time.time()}))
    speaker = consolidator = scheduler = reasoner = None
    fusion = ContextFusion(Calendar(calendar_path))
    if memory is not None:

        def record(resp) -> None:  # CL3: kin-core writes every LedgerEntry to egress_log; the UI shows it
            if resp.ledger is not None:
                entry = resp.ledger.model_dump()
                memory.store.add_egress(entry)
                ui_q.put({"type": "ledger", **entry, "kind": resp.kind, "ok": resp.ok})

        facts = FactService(session, authkey, record=record)
        reasoner = open_reasoner(memory, profile, calendar_path, facts=facts, on_state=status.set)
        consolidator = Consolidator(memory, reasoner.llm)
        scheduler = Scheduler(memory.store, consolidator)
        speaker = SpeakerTagger(threads=1, get_meta=memory.store.get_meta, set_meta=memory.store.set_meta)
    core = Core(ring, ui_q, ctl_q, profile, memory=memory, asr_model=asr_model, reasoner=reasoner, voice=voice,
                speaker=speaker, status=status, fusion=fusion, scheduler=scheduler, consolidator=consolidator,
                audio_cmd_q=audio_cmd_q)  # fmt: skip
    core.start()
    if warm_llm and reasoner is not None:
        core._put(PRIO_BACKGROUND, {"kind": "warm"})  # the demo's first answer skips the load
    threading.Thread(target=_serve_rpc, args=(core, cmd_q, ui_q, stop_evt), name="rpc", daemon=True).start()
    if profile.core_rss_limit_mb:
        threading.Thread(
            target=_memory_watchdog, args=(profile.core_rss_limit_mb, ui_q, stop_evt, core.shed_memory), daemon=True
        ).start()
    log.info("ready (profile=%s, asr=%s, asr_threads=%d)", profile.name, core.asr.model, profile.asr_threads)
    net = guard.probe()
    ui_q.put({"type": "core_info", "asr": core.asr.model, "profile": profile.name,
              "llm": reasoner.llm.path.name if reasoner and reasoner.llm.installed() else None,
              "calendar": str(calendar_path) if calendar_path else None,
              "voice": {"available": voice.available, "enabled": voice.enabled},
              "speaker": {"installed": bool(speaker and speaker.installed()),
                          "enrolled": bool(speaker and speaker.enrolled)},
              "gateway": session is not None, "net": net,
              "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "DO_NOT_TRACK")}})  # fmt: skip
    ctl_q.put({"type": "core_ready", "t": time.time(), "net": net})
    next_tick = time.monotonic() + 1.0
    try:
        while not stop_evt.is_set():
            try:
                core.handle(events_q.get(timeout=0.2))
            except queue.Empty:
                pass
            if time.monotonic() >= next_tick:
                next_tick = time.monotonic() + 1.0
                core.tick()
    finally:
        voice.close()
        core.stop()
        if memory is not None:
            memory.store.close()
        ring.close()
        log.info("stopped")
