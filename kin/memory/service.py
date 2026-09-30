"""Memory as kin-core uses it: store + embedder + episode policy + retention."""

from __future__ import annotations

import time

from kin.memory.store import Store
from kin.perception.segmenter import EpisodePolicy

RETENTION_DAYS = 7  # transcripts; digests and facts (a year) arrive with consolidation


class Memory:
    def __init__(self, store: Store, embedder, policy: EpisodePolicy = EpisodePolicy(), retention_days=RETENTION_DAYS):
        self.store = store
        self.embedder = embedder
        self.policy = policy
        self.retention_s = retention_days * 86400

    # write path
    def add(self, t_start: float, t_end: float, text: str, *, is_command: bool, redactions: int) -> dict:
        return self.store.add_utterance(
            t_start, t_end, text, policy=self.policy, is_command=is_command, redactions=redactions
        )

    def close_due(self, now: float | None = None) -> list[int]:
        return self.store.close_due(time.time() if now is None else now, self.policy)

    def embed_episode(self, ep_id: int) -> bool:
        """Heavy: call only from kin-core's heavy lane."""
        text = self.store.episode_text(ep_id)
        if not text:
            return False
        return self.store.set_vector(ep_id, self.embedder.embed_one(text), expect_text=text)

    def rescrub_if_needed(self) -> list[int]:
        """Apply the current write-path filter to text stored under an older one."""
        from kin.perception.filters import FILTER_VERSION, redact

        if self.store.get_meta("filter_version") == str(FILTER_VERSION):
            return []
        changed = self.store.rescrub(redact)
        self.store.set_meta("filter_version", str(FILTER_VERSION))
        return changed

    def purge_expired(self, now: float | None = None) -> int:
        return self.store.purge_before((time.time() if now is None else now) - self.retention_s)

    # read path
    def search(self, q: str, k: int = 5, qvec=None) -> list[dict]:
        return self.store.search(q, qvec, k=k)

    def forget_minutes(self, minutes: float) -> dict:
        return self.store.forget_since(time.time() - minutes * 60)
