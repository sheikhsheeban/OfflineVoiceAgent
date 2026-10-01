"""Memory as kin-core uses it: store + embedder + episode policy + retention."""

from __future__ import annotations

import time

from kin.memory.store import Store
from kin.perception.segmenter import EpisodePolicy

RETENTION_DAYS = 7  # transcripts; digests, facts and people last a year (see consolidate.py)


class Memory:
    def __init__(self, store: Store, embedder, policy: EpisodePolicy = EpisodePolicy(), retention_days=RETENTION_DAYS):
        self.store = store
        self.embedder = embedder
        self.policy = policy
        self.retention_s = retention_days * 86400

    # write path
    def add(
        self, t_start: float, t_end: float, text: str, *, is_command: bool, redactions: int, speaker: str = "unknown"
    ) -> dict:
        return self.store.add_utterance(
            t_start, t_end, text, policy=self.policy, is_command=is_command, redactions=redactions, speaker=speaker
        )

    def close_due(self, now: float | None = None) -> list[int]:
        return self.store.close_due(time.time() if now is None else now, self.policy)

    def embed_episode(self, ep_id: int) -> bool:
        """Heavy: call only from kin-core's heavy lane."""
        text = self.store.episode_text(ep_id)
        if not text:
            return False
        if not self.store.set_vector(ep_id, self.embedder.embed_one(text), expect_text=text):
            return False
        self.derive(ep_id, text)
        return True

    def derive(self, ep_id: int, text: str) -> None:
        """P5: people and claims from a closed episode (removed again if its text is deleted or changed)."""
        from kin.perception.extractor import extract

        found = extract(text)
        now = time.time()
        self.store.add_people(found.people, ep_id, now)
        for subject, claim in found.facts:
            self.store.add_fact(subject, claim, ep_id, now)

    def backfill_derived_if_needed(self) -> int:
        """People and facts are derived when an episode is indexed; episodes indexed before that
        existed get theirs once, at the next start-up."""
        if self.store.get_meta("derived_version") == "1":
            return 0
        n = 0
        for ep in self.store.timeline(limit=100_000):
            if ep["closed"] and ep["text"]:
                self.derive(ep["id"], ep["text"])
                n += 1
        self.store.set_meta("derived_version", "1")
        return n

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
