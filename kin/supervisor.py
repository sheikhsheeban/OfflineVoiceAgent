"""Starts and watches the Kin processes, and measures their CPU and RAM.

Week 1 runs kin-audio, kin-core and kin-ui. kin-gateway joins in week 4.
A crashed process is restarted within 2 s (at most 5 times a minute).
"""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import queue
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import psutil

from kin.audio.ring import Ring
from kin.config import RING_SECONDS, SAMPLE_RATE, Settings, asr_model_name, default_calendar_path

log = logging.getLogger("kin.supervisor")


@dataclass
class Child:
    name: str
    target: Callable
    args: tuple
    proc: mp.Process | None = None
    ps: psutil.Process | None = None
    restarts: list[float] = field(default_factory=list)

    def start(self, ctx) -> None:
        self.proc = ctx.Process(target=self.target, args=self.args, name=self.name, daemon=True)
        self.proc.start()
        self.ps = psutil.Process(self.proc.pid)
        self.ps.cpu_percent(None)  # prime the counter


class Supervisor:
    def __init__(
        self,
        settings: Settings,
        source: str = "mic",
        speed: float = 1.0,
        ui: bool = True,
        token: str | None = None,
        metrics_csv: Path | None = None,
    ):
        self.settings = settings
        self.source = source
        self.ui = ui
        self.token = token or secrets.token_urlsafe(16)
        self.ctx = mp.get_context("spawn")  # same behaviour on every OS
        self.ring = Ring.create(RING_SECONDS * SAMPLE_RATE)
        self.events_q = self.ctx.Queue()
        self.ui_q = self.ctx.Queue()
        self.ctl_q = self.ctx.Queue()
        self.cmd_q = self.ctx.Queue()  # kin-ui -> kin-core requests
        self.stop_evt = self.ctx.Event()
        self.messages: list[dict] = []  # headless mode: what the UI would have shown
        self.stats_history: list[dict] = []
        self.metrics_csv = metrics_csv
        self._csv = None
        self._stopping = False
        self._next_stats = 0.0
        self._samples = 0
        self._private: dict[str, float] = {}

        self.asr_model = asr_model_name(settings.profile.name, settings.asr_model)

        from kin.audio import process as audio_proc
        from kin.core import process as core_proc

        self.children = [
            Child(
                "kin-audio",
                audio_proc.run,
                (self.ring.name, self.events_q, self.stop_evt, settings.audio, source, speed),
            ),
            Child(
                "kin-core",
                core_proc.run,
                (
                    self.ring.name,
                    self.events_q,
                    self.ui_q,
                    self.ctl_q,
                    self.cmd_q,
                    self.stop_evt,
                    settings.profile,
                    settings.memory_db,
                    self.asr_model,
                    settings.calendar_path or default_calendar_path(),
                    settings.warm_llm,
                ),
            ),
        ]
        if ui:
            from kin.ui import app as ui_app

            self.children.append(
                Child(
                    "kin-ui",
                    ui_app.run,
                    (self.ui_q, self.cmd_q, self.stop_evt, settings.ui_host, settings.ui_port, self.token),
                )
            )

    @property
    def ui_url(self) -> str:
        return f"http://{self.settings.ui_host}:{self.settings.ui_port}/?token={self.token}"

    def start(self, ready_timeout: float = 90.0) -> None:
        # kin-core loads its models before kin-audio starts, so no speech is
        # queued against a core that cannot transcribe it yet.
        by_name = {c.name: c for c in self.children}
        by_name["kin-core"].start(self.ctx)
        if "kin-ui" in by_name:
            by_name["kin-ui"].start(self.ctx)
        self.wait_ready(ready_timeout)
        by_name["kin-audio"].start(self.ctx)
        if self.metrics_csv:
            self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            f = open(self.metrics_csv, "w", newline="")
            self._csv = (f, csv.writer(f))
            self._csv[1].writerow(["t", "process", "cpu_pct_core", "rss_mb"])

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Block until kin-core has loaded its models."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.ctl_q.get(timeout=0.5).get("type") == "core_ready":
                    return
            except queue.Empty:
                for c in self.children:
                    if c.proc is not None and not c.proc.is_alive():
                        raise RuntimeError(f"{c.name} exited during start-up (code {c.proc.exitcode})") from None
        raise TimeoutError("kin-core did not become ready")

    def run(self, until_eos: bool = False, duration: float | None = None, on_tick: Callable | None = None) -> None:
        """Supervise until stopped, end of replay, or `duration` seconds."""
        t_end = time.monotonic() + duration if duration else None
        try:
            while True:
                self._check_children()
                if not self.ui:
                    self._drain_ui()
                try:
                    ev = self.ctl_q.get(timeout=0.1)
                    if ev.get("type") == "eos" and until_eos:
                        self._drain_ui()
                        return
                except queue.Empty:
                    pass
                now = time.monotonic()
                if now >= self._next_stats:
                    self._sample_stats()
                    self._next_stats = now + 1.0
                    if on_tick:
                        on_tick(self)
                if t_end and now >= t_end:
                    return
        except KeyboardInterrupt:
            log.info("stopping")

    def rpc(self, method: str, params: dict | None = None, timeout: float = 30.0):
        """Call kin-core directly (headless mode only; with a UI, kin-ui owns ui_q)."""
        if self.ui:
            raise RuntimeError("rpc() is for headless supervisors")
        req_id = secrets.token_hex(8)
        self.cmd_q.put({"id": req_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = self.ui_q.get(timeout=0.1)
            except queue.Empty:
                self._check_children()
                continue
            if msg.get("type") == "rpc_result" and msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]
            self.messages.append(msg)
        raise TimeoutError(f"kin-core did not answer {method}")

    def _drain_ui(self) -> None:
        while True:
            try:
                self.messages.append(self.ui_q.get_nowait())
            except queue.Empty:
                return

    def _check_children(self) -> None:
        if self._stopping:
            return
        for c in self.children:
            if c.proc is not None and not c.proc.is_alive():
                now = time.monotonic()
                c.restarts = [t for t in c.restarts if now - t < 60]
                if len(c.restarts) >= 5:
                    raise RuntimeError(f"{c.name} keeps crashing (exit code {c.proc.exitcode})")
                log.error("%s exited with code %s; restarting", c.name, c.proc.exitcode)
                c.restarts.append(now)
                c.start(self.ctx)

    def _sample_stats(self) -> None:
        procs = {}
        t = time.time()
        self._samples += 1
        for c in self.children:
            alive = c.proc is not None and c.proc.is_alive()
            cpu = rss = 0.0
            if alive and c.ps:
                try:
                    cpu = c.ps.cpu_percent(None)
                    rss = c.ps.memory_info().rss / 2**20
                    # Private resident memory excludes the memory-mapped model file; it takes
                    # tens of ms to measure, so every 5 s.
                    if self._samples % 5 == 1 or c.name not in self._private:
                        self._private[c.name] = c.ps.memory_full_info().uss / 2**20
                except psutil.Error:
                    alive = False
            procs[c.name] = {"cpu": round(cpu, 1), "rss_mb": round(rss, 1), "alive": alive,
                             "private_mb": round(self._private.get(c.name, rss), 1)}  # fmt: skip
            if self._csv:
                self._csv[1].writerow([round(t, 2), c.name, round(cpu, 2), round(rss, 1)])
        if self._csv:
            self._csv[0].flush()
        total_cpu = sum(p["cpu"] for p in procs.values())
        stats = {
            "type": "stats",
            "t": t,
            "profile": self.settings.profile.name,
            "procs": procs,
            "cpu_machine_pct": round(total_cpu / (psutil.cpu_count() or 1), 2),
            "rss_mb_total": round(sum(p["rss_mb"] for p in procs.values()), 1),
            "private_mb_total": round(sum(p["private_mb"] for p in procs.values()), 1),
        }
        self.stats_history.append(stats)
        self.ui_q.put(stats)

    def stop(self) -> None:
        self._stopping = True
        self.stop_evt.set()
        for c in self.children:
            if c.proc is not None:
                c.proc.join(timeout=5)
                if c.proc.is_alive():
                    c.proc.terminate()
                    c.proc.join(timeout=2)
        if not self.ui:
            self._drain_ui()
        if self._csv:
            self._csv[0].close()
        for q in (self.events_q, self.ui_q, self.ctl_q, self.cmd_q):
            q.cancel_join_thread()
            q.close()
        self.ring.close()

    def __enter__(self) -> "Supervisor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
