"""R6 proactive scheduler: things Kin says without being asked.

A `poll` looks at the clock, the task list and the latest ContextEvent (C1) and
returns triggers; kin-core turns each into a normal answer (E35 -> R1 -> the F2
path) or, for consolidation, a background job. Kin never speaks on its own
judgement about what it overheard: only for a task the user set a time on, a
reminder, the end of a meeting with things to do, and the nightly housekeeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kin.contracts import ContextEvent
from kin.memory.consolidate import Consolidator


@dataclass
class Trigger:
    kind: str  # "task_due" | "meeting_ended" | "consolidate"
    text: str = ""
    data: dict = field(default_factory=dict)


class Scheduler:
    def __init__(self, store, consolidator: Consolidator | None = None):
        self.store = store
        self.consolidator = consolidator
        self._prev: ContextEvent | None = None

    def poll(self, now: float | None, ctx: ContextEvent | None) -> list[Trigger]:
        now = time.time() if now is None else now
        out: list[Trigger] = []
        for t in self.store.take_due_tasks(now):
            out.append(Trigger("task_due", f"Time to: {t['title'][0].lower() + t['title'][1:]}.", {"task": t}))
        if ctx is not None:
            prev, self._prev = self._prev, ctx
            if prev is not None and prev.in_meeting and not ctx.in_meeting and self.store.tasks(("open",)):
                out.append(Trigger("meeting_ended", "what's on my list?"))
            if self.consolidator is not None and self.consolidator.due(ctx, now):
                out.append(Trigger("consolidate"))
        return out
