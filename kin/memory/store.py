"""Encrypted memory store: SQLCipher + FTS5 + sqlite-vec.

One connection, shared by kin-core's threads behind a lock. Utterances are
written as soon as they are transcribed and grouped into ~60 s episodes; each
closed episode gets one int8 vector. Search fuses keyword (FTS5/bm25) and
vector (cosine) rankings with reciprocal rank fusion.

Deletes are real: `secure_delete` overwrites freed pages, FTS5 runs in
secure-delete mode, and a full wipe also destroys the key.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from kin.config import ROOT, SPEC_DIR
from kin.memory.embed import quantize

DATA_DIR = Path(os.environ.get("KIN_DATA_DIR", ROOT / "data"))
SCHEMA_VERSION = "1"
DIM = 384
RRF_K = 60
_STOPWORDS = set(
    "a an and are as at be but by did do does for from had has have he her him his i if in is it its me my "
    "of on or our she so that the their them they this to up us was we were what when where which who why "
    "will with you your about said say tell told".split()
)


def default_path() -> Path:
    return DATA_DIR / "kin.db"


def _fts_query(q: str) -> str | None:
    terms = [t for t in re.findall(r"[\w']+", q.lower()) if t not in _STOPWORDS]
    terms = [t.replace("'", "") for t in terms if t.replace("'", "")]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class Store:
    def __init__(self, path: str | Path, key: str, use_vec: bool = True):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", key):
            raise ValueError("key must be 64 hex characters")
        self.path = Path(path)
        self._key = key
        self._want_vec = use_vec
        self._lock = threading.RLock()
        self._connect()

    @classmethod
    def open(cls, path: str | Path | None = None, use_vec: bool = True) -> "Store":
        from kin.memory.keys import get_key

        path = Path(path or default_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        key = get_key(str(path), create=not path.exists())
        return cls(path, key, use_vec=use_vec)

    # -- connection -----------------------------------------------------------------

    def _connect(self) -> None:
        import sqlcipher3

        self.conn = sqlcipher3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.conn.execute(f"PRAGMA key = \"x'{self._key}'\"")  # must be the first statement
        try:
            self.conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlcipher3.DatabaseError as e:
            self.conn.close()
            raise RuntimeError(f"cannot decrypt {self.path}: wrong key") from e
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA secure_delete = ON")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript((SPEC_DIR / "schema.sql").read_text(encoding="utf-8"))
        self.conn.execute("INSERT INTO utterances_fts(utterances_fts, rank) VALUES ('secure-delete', 1)")
        self.conn.execute(
            "INSERT OR IGNORE INTO meta VALUES ('schema_version', ?), ('created_at', ?)",
            (SCHEMA_VERSION, str(time.time())),
        )
        self.vec_backend = "numpy"
        if self._want_vec:
            try:
                import sqlite_vec

                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS episode_vec USING vec0("
                    f"episode_id INTEGER PRIMARY KEY, embedding int8[{DIM}] distance_metric=cosine)"
                )
                self.vec_backend = "sqlite-vec"
            except Exception:  # noqa: BLE001 - any load failure means: use the fallback
                self.vec_backend = "numpy"
        if self.vec_backend == "numpy":
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS episode_vec_blob (episode_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
            )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _q(self, sql: str, args: tuple = ()) -> list[tuple]:
        return self.conn.execute(sql, args).fetchall()

    # -- write path -----------------------------------------------------------------

    def add_utterance(
        self,
        t_start: float,
        t_end: float,
        text: str,
        *,
        policy,
        is_command: bool = False,
        speaker: str = "unknown",
        redactions: int = 0,
    ) -> dict:
        """Store one utterance, in the open episode or a new one (per `policy`)."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._q("SELECT id, started_at, ended_at FROM episodes WHERE closed = 0 ORDER BY id DESC LIMIT 1")
                new, closed = True, []
                if row and policy.continues(row[0][1], row[0][2], t_start):
                    ep_id, new = row[0][0], False
                else:
                    for (old,) in self._q("SELECT id FROM episodes WHERE closed = 0"):
                        self.conn.execute("UPDATE episodes SET closed = 1 WHERE id = ?", (old,))
                        closed.append(old)
                    ep_id = self.conn.execute(
                        "INSERT INTO episodes (started_at, ended_at) VALUES (?, ?)", (t_start, t_end)
                    ).lastrowid
                utt_id = self.conn.execute(
                    "INSERT INTO utterances (episode_id, t_start, t_end, speaker, is_command, redactions, text)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ep_id, t_start, t_end, speaker, int(is_command), redactions, text),
                ).lastrowid
                self._refresh_episode(ep_id)
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
        return {"utterance_id": utt_id, "episode_id": ep_id, "new_episode": new, "closed": closed}

    def _refresh_episode(self, ep_id: int) -> bool:
        rows = self._q("SELECT t_start, t_end, text FROM utterances WHERE episode_id = ? ORDER BY t_start", (ep_id,))
        if not rows:
            self._delete_episode_rows(ep_id)
            return False
        self.conn.execute(
            "UPDATE episodes SET started_at = ?, ended_at = ?, n_utterances = ?, text = ? WHERE id = ?",
            (rows[0][0], max(r[1] for r in rows), len(rows), " ".join(r[2] for r in rows), ep_id),
        )
        return True

    def close_due(self, now: float, policy) -> list[int]:
        """Close episodes whose time is up; returns their ids (they now need an embedding)."""
        with self._lock:
            due = [
                r[0]
                for r in self._q("SELECT id, started_at, ended_at FROM episodes WHERE closed = 0")
                if policy.should_close(r[1], r[2], now)
            ]
            for ep_id in due:
                self.conn.execute("UPDATE episodes SET closed = 1 WHERE id = ?", (ep_id,))
            return due

    def close_all(self) -> list[int]:
        with self._lock:
            ids = [r[0] for r in self._q("SELECT id FROM episodes WHERE closed = 0")]
            self.conn.execute("UPDATE episodes SET closed = 1 WHERE closed = 0")
            return ids

    def pending_embeddings(self) -> list[int]:
        with self._lock:
            return [r[0] for r in self._q("SELECT id FROM episodes WHERE closed = 1 AND embedded = 0 ORDER BY id")]

    def episode_text(self, ep_id: int) -> str | None:
        with self._lock:
            row = self._q("SELECT text FROM episodes WHERE id = ?", (ep_id,))
            return row[0][0] if row else None

    def set_vector(self, ep_id: int, vec: np.ndarray, expect_text: str | None = None) -> bool:
        """Store an episode's vector. If `expect_text` no longer matches (edited by a
        forget while embedding), the vector is stale and is not stored."""
        q = vec if vec.dtype == np.int8 else quantize(vec)
        with self._lock:
            row = self._q("SELECT text FROM episodes WHERE id = ?", (ep_id,))
            if not row or (expect_text is not None and row[0][0] != expect_text):
                return False
            self._delete_vector(ep_id)
            if self.vec_backend == "sqlite-vec":
                self.conn.execute(
                    "INSERT INTO episode_vec (episode_id, embedding) VALUES (?, vec_int8(?))", (ep_id, q.tobytes())
                )
            else:
                self.conn.execute("INSERT INTO episode_vec_blob VALUES (?, ?)", (ep_id, q.tobytes()))
            self.conn.execute("UPDATE episodes SET embedded = 1 WHERE id = ?", (ep_id,))
            return True

    # -- read path ------------------------------------------------------------------

    def _episodes(self, where: str, args: tuple, order: str = "e.started_at DESC", limit: int = 50) -> list[dict]:
        eps = self._q(
            "SELECT e.id, e.started_at, e.ended_at, e.closed, e.embedded, e.n_utterances, e.text FROM episodes e"
            f" WHERE {where} ORDER BY {order} LIMIT ?",
            (*args, limit),
        )
        if not eps:
            return []
        ids = [e[0] for e in eps]
        utts: dict[int, list[dict]] = {i: [] for i in ids}
        marks = ",".join("?" * len(ids))
        for u in self._q(
            "SELECT id, episode_id, t_start, t_end, speaker, is_command, redactions, text FROM utterances"
            f" WHERE episode_id IN ({marks}) ORDER BY t_start",
            tuple(ids),
        ):
            utts[u[1]].append(
                {
                    "id": u[0],
                    "t_start": u[2],
                    "t_end": u[3],
                    "speaker": u[4],
                    "command": bool(u[5]),
                    "redactions": u[6],
                    "text": u[7],
                }  # fmt: skip
            )
        return [
            {
                "id": e[0],
                "started_at": e[1],
                "ended_at": e[2],
                "closed": bool(e[3]),
                "embedded": bool(e[4]),
                "n_utterances": e[5],
                "text": e[6],
                "utterances": utts[e[0]],
            }  # fmt: skip
            for e in eps
        ]

    def timeline(self, limit: int = 50, before: float | None = None) -> list[dict]:
        with self._lock:
            if before is None:
                return self._episodes("1", (), limit=limit)
            return self._episodes("e.started_at < ?", (before,), limit=limit)

    def get_episode(self, ep_id: int) -> dict | None:
        with self._lock:
            eps = self._episodes("e.id = ?", (ep_id,), limit=1)
            return eps[0] if eps else None

    def keyword_search(self, q: str, limit: int = 20) -> list[tuple[int, float, str]]:
        """(episode_id, bm25 score (lower is better), snippet), best first."""
        fq = _fts_query(q)
        if fq is None:
            return []
        with self._lock:
            return self._q(
                "SELECT episode_id, MIN(score), snip FROM ("
                "  SELECT u.episode_id AS episode_id, bm25(utterances_fts) AS score,"
                "         snippet(utterances_fts, 0, '[', ']', '…', 12) AS snip"
                "  FROM utterances_fts JOIN utterances u ON u.id = utterances_fts.rowid"
                "  WHERE utterances_fts MATCH ? ORDER BY score LIMIT 200)"
                " GROUP BY episode_id ORDER BY MIN(score) LIMIT ?",
                (fq, limit),
            )

    def vector_search(self, qvec: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        """(episode_id, cosine similarity), best first."""
        q = quantize(qvec)
        with self._lock:
            if self.vec_backend == "sqlite-vec":
                n = self._q("SELECT count(*) FROM episode_vec")[0][0]
                if n == 0:
                    return []
                rows = self._q(
                    "SELECT episode_id, distance FROM episode_vec WHERE embedding MATCH vec_int8(?) AND k = ?",
                    (q.tobytes(), min(k, n)),
                )
                return [(r[0], 1.0 - r[1]) for r in rows]
            rows = self._q("SELECT episode_id, embedding FROM episode_vec_blob")
        if not rows:
            return []
        ids = np.array([r[0] for r in rows])
        m = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.int8).reshape(len(rows), DIM).astype(np.float32)
        qf = q.astype(np.float32)
        sims = (m @ qf) / (np.linalg.norm(m, axis=1) * np.linalg.norm(qf) + 1e-9)
        top = np.argsort(-sims)[:k]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def _similarity(self, ep_ids: list[int], qvec: np.ndarray) -> dict[int, float]:
        if not ep_ids:
            return {}
        table = "episode_vec" if self.vec_backend == "sqlite-vec" else "episode_vec_blob"
        marks = ",".join("?" * len(ep_ids))
        with self._lock:
            rows = self._q(f"SELECT episode_id, embedding FROM {table} WHERE episode_id IN ({marks})", tuple(ep_ids))
        q = quantize(qvec).astype(np.float32)
        out = {}
        for ep_id, blob in rows:
            v = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
            out[ep_id] = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q) + 1e-9))
        return out

    def search(self, q: str, qvec: np.ndarray | None, k: int = 5) -> list[dict]:
        """Hybrid search. Each hit reports its keyword and vector evidence separately,
        so a later confidence gate can decide whether memory really has an answer."""
        kw = self.keyword_search(q, limit=20)
        vs = self.vector_search(qvec, k=20) if qvec is not None else []
        scores: dict[int, float] = {}
        for rank, (ep_id, *_rest) in enumerate(kw):
            scores[ep_id] = scores.get(ep_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (ep_id, _sim) in enumerate(vs):
            scores[ep_id] = scores.get(ep_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        sims = dict(vs)
        if qvec is not None:
            sims.update(self._similarity([i for i in best if i not in sims], qvec))
        snippets = {r[0]: r[2] for r in kw}
        bm25 = {r[0]: r[1] for r in kw}
        hits = []
        for ep_id in best:
            ep = self.get_episode(ep_id)
            if ep is None:
                continue
            ep.update(
                score=round(scores[ep_id], 5),
                vector_similarity=round(sims[ep_id], 3) if ep_id in sims else None,
                keyword_match=ep_id in bm25,
                bm25=round(bm25[ep_id], 3) if ep_id in bm25 else None,
                snippet=snippets.get(ep_id),
            )
            hits.append(ep)
        return hits

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._q("SELECT value FROM meta WHERE key = ?", (key,))
            return row[0][0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def rescrub(self, redact_fn) -> list[int]:
        """Re-run the write-path filter over everything stored (after the filter
        improves). Changed episodes lose their vector and are re-embedded; the WAL
        is checkpointed so the old text does not linger there."""
        with self._lock:
            rows = self._q("SELECT id, episode_id, text, redactions FROM utterances")
            changed: set[int] = set()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for utt_id, ep_id, text, n in rows:
                    f = redact_fn(text)
                    if f.text != text:
                        self.conn.execute(
                            "UPDATE utterances SET text = ?, redactions = ? WHERE id = ?",
                            (f.text, n + f.redactions, utt_id),
                        )
                        changed.add(ep_id)
                for table, col in (("tasks", "title"), ("tasks", "quote"), ("reminders", "text")):
                    for row_id, text in self._q(f"SELECT id, {col} FROM {table} WHERE {col} IS NOT NULL"):
                        f = redact_fn(text)
                        if f.text != text:
                            self.conn.execute(f"UPDATE {table} SET {col} = ? WHERE id = ?", (f.text, row_id))
                for ep_id in changed:
                    self._delete_vector(ep_id)
                    self.conn.execute("UPDATE episodes SET embedded = 0 WHERE id = ?", (ep_id,))
                    self._refresh_episode(ep_id)
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            if changed:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return sorted(changed)

    # -- deletion -------------------------------------------------------------------

    def _delete_vector(self, ep_id: int) -> None:
        table = "episode_vec" if self.vec_backend == "sqlite-vec" else "episode_vec_blob"
        self.conn.execute(f"DELETE FROM {table} WHERE episode_id = ?", (ep_id,))

    def _detach_tasks(self, where: str, args: tuple) -> None:
        """Speech is being deleted: drop commitments Kin merely heard from it, and
        strip the quoted sentence from tasks the user kept."""
        self.conn.execute(f"DELETE FROM tasks WHERE status IN ('candidate', 'dismissed') AND ({where})", args)
        self.conn.execute(f"UPDATE tasks SET quote = NULL, episode_id = NULL, utterance_id = NULL WHERE {where}", args)

    def _delete_episode_rows(self, ep_id: int) -> None:
        self._detach_tasks("episode_id = ?", (ep_id,))
        self.conn.execute("DELETE FROM utterances WHERE episode_id = ?", (ep_id,))  # fires the FTS delete trigger
        self._delete_vector(ep_id)
        self.conn.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))

    def delete_episode(self, ep_id: int) -> bool:
        with self._lock:
            if not self._q("SELECT 1 FROM episodes WHERE id = ?", (ep_id,)):
                return False
            self.conn.execute("BEGIN IMMEDIATE")
            self._delete_episode_rows(ep_id)
            self.conn.execute("COMMIT")
            return True

    def forget_since(self, t: float) -> dict:
        """Hard-delete everything heard at or after `t` (text and vectors).

        Episodes that straddle `t` keep their earlier utterances; their vector is
        dropped and they are returned in `reembed`.
        """
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            affected = [r[0] for r in self._q("SELECT DISTINCT episode_id FROM utterances WHERE t_end >= ?", (t,))]
            self._detach_tasks("utterance_id IN (SELECT id FROM utterances WHERE t_end >= ?)", (t,))
            n_utt = self.conn.execute("DELETE FROM utterances WHERE t_end >= ?", (t,)).rowcount
            reembed, removed = [], 0
            for ep_id in affected:
                self._delete_vector(ep_id)
                self.conn.execute("UPDATE episodes SET embedded = 0 WHERE id = ?", (ep_id,))
                if self._refresh_episode(ep_id):
                    closed = self._q("SELECT closed FROM episodes WHERE id = ?", (ep_id,))[0][0]
                    if closed:
                        reembed.append(ep_id)
                else:
                    removed += 1
            self.conn.execute("COMMIT")
        return {"utterances": n_utt, "episodes_removed": removed, "reembed": reembed}

    def purge_before(self, t: float) -> int:
        """Retention: drop episodes (text and vectors) that ended before `t`."""
        with self._lock:
            ids = [r[0] for r in self._q("SELECT id FROM episodes WHERE ended_at < ?", (t,))]
            if ids:
                self.conn.execute("BEGIN IMMEDIATE")
                for ep_id in ids:
                    self._delete_episode_rows(ep_id)
                self.conn.execute("COMMIT")
            return len(ids)

    def wipe(self) -> None:
        """Delete every file of the store and destroy its key; start empty with a new key."""
        from kin.memory.keys import delete_key, get_key

        with self._lock:
            self.conn.close()
            for suffix in ("", "-wal", "-shm", "-journal"):
                p = Path(str(self.path) + suffix)
                if p.exists():
                    p.unlink()
            delete_key(str(self.path))
            self._key = get_key(str(self.path), create=True)
            self._connect()

    # -- tasks and reminders --------------------------------------------------------

    def add_task(
        self,
        title: str,
        *,
        source: str,
        status: str = "open",
        due_at: float | None = None,
        due_has_time: bool = False,
        episode_id: int | None = None,
        utterance_id: int | None = None,
        quote: str | None = None,
        now: float | None = None,
    ) -> int:
        with self._lock:
            return self.conn.execute(
                "INSERT INTO tasks (title, due_at, due_has_time, status, source, episode_id, utterance_id, quote,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (title, due_at, int(due_has_time), status, source, episode_id, utterance_id, quote,
                 time.time() if now is None else now),
            ).lastrowid  # fmt: skip

    def _task_rows(self, where: str, args: tuple = (), order: str = "coalesce(due_at, 9e18), id") -> list[dict]:
        cols = "id, title, due_at, due_has_time, status, source, episode_id, quote, created_at, done_at"
        rows = self._q(f"SELECT {cols} FROM tasks WHERE {where} ORDER BY {order}", args)
        keys = [c.strip() for c in cols.split(",")]
        out = [dict(zip(keys, r, strict=True)) for r in rows]
        for t in out:
            t["due_has_time"] = bool(t["due_has_time"])
        return out

    def tasks(self, statuses: tuple[str, ...] = ("candidate", "open")) -> list[dict]:
        with self._lock:
            marks = ",".join("?" * len(statuses))
            return self._task_rows(f"status IN ({marks})", statuses)

    def get_task(self, task_id: int) -> dict | None:
        with self._lock:
            rows = self._task_rows("id = ?", (task_id,))
            return rows[0] if rows else None

    def tasks_created_between(self, t0: float, t1: float) -> list[dict]:
        with self._lock:
            return self._task_rows("created_at >= ? AND created_at < ?", (t0, t1), order="created_at")

    def set_task_status(self, task_id: int, status: str, now: float | None = None) -> bool:
        if status not in ("candidate", "open", "done", "dismissed"):
            raise ValueError(f"bad task status {status!r}")
        done_at = (time.time() if now is None else now) if status == "done" else None
        with self._lock:
            n = self.conn.execute("UPDATE tasks SET status = ?, done_at = ? WHERE id = ?", (status, done_at, task_id))
            return n.rowcount == 1

    def delete_task(self, task_id: int) -> bool:
        with self._lock:
            return self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,)).rowcount == 1

    def add_reminder(self, text: str, fire_at: float, now: float | None = None) -> int:
        with self._lock:
            return self.conn.execute(
                "INSERT INTO reminders (text, fire_at, created_at) VALUES (?, ?, ?)",
                (text, fire_at, time.time() if now is None else now),
            ).lastrowid

    def reminders(self, statuses: tuple[str, ...] = ("pending",)) -> list[dict]:
        with self._lock:
            marks = ",".join("?" * len(statuses))
            rows = self._q(
                f"SELECT id, text, fire_at, status, created_at, fired_at FROM reminders WHERE status IN ({marks})"
                " ORDER BY fire_at",
                statuses,
            )
        keys = ("id", "text", "fire_at", "status", "created_at", "fired_at")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def take_due_reminders(self, now: float) -> list[dict]:
        """Pending reminders whose time has come, marked fired in the same step."""
        with self._lock:
            due = [r for r in self.reminders() if r["fire_at"] <= now]
            for r in due:
                self.conn.execute("UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ?", (now, r["id"]))
                r.update(status="fired", fired_at=now)
            return due

    def cancel_reminder(self, reminder_id: int) -> bool:
        with self._lock:
            n = self.conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'", (reminder_id,)
            )
            return n.rowcount == 1

    def episodes_between(self, t0: float, t1: float) -> int:
        with self._lock:
            return self._q("SELECT count(*) FROM episodes WHERE started_at >= ? AND started_at < ?", (t0, t1))[0][0]

    # -- info -----------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            n_ep, n_open, n_emb = self._q(
                "SELECT count(*), coalesce(sum(closed = 0), 0), coalesce(sum(embedded), 0) FROM episodes"
            )[0]
            n_utt = self._q("SELECT count(*) FROM utterances")[0][0]
            oldest = self._q("SELECT min(started_at) FROM episodes")[0][0]
            n_tasks = self._q("SELECT count(*) FROM tasks WHERE status IN ('candidate', 'open')")[0][0]
            n_rem = self._q("SELECT count(*) FROM reminders WHERE status = 'pending'")[0][0]
            cipher = self._q("PRAGMA cipher_version")[0][0]
        size = sum(Path(str(self.path) + s).stat().st_size for s in ("", "-wal") if Path(str(self.path) + s).exists())
        return {
            "episodes": n_ep,
            "open_episodes": n_open,
            "embedded": n_emb,
            "utterances": n_utt,
            "oldest": oldest,
            "bytes": size,
            "cipher": f"SQLCipher {cipher}",
            "vector_backend": self.vec_backend,
            "open_tasks": n_tasks,
            "pending_reminders": n_rem,
        }
