-- Kin memory store. Shared with the Android build.
-- Opened through SQLCipher: every page, including the WAL, is encrypted at rest.
-- Times are unix seconds (REAL).

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ~60 s of conversation. An episode stays open while speech keeps arriving,
-- then closes and gets one embedding.
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY,
    started_at   REAL    NOT NULL,
    ended_at     REAL    NOT NULL,
    closed       INTEGER NOT NULL DEFAULT 0,
    embedded     INTEGER NOT NULL DEFAULT 0,
    n_utterances INTEGER NOT NULL DEFAULT 0,
    text         TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS episodes_started ON episodes(started_at);

-- One transcribed speech segment, already filtered for OTPs and card numbers.
CREATE TABLE IF NOT EXISTS utterances (
    id         INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    t_start    REAL    NOT NULL,
    t_end      REAL    NOT NULL,
    speaker    TEXT    NOT NULL DEFAULT 'unknown',  -- 'you' | 'other' | 'unknown'
    is_command INTEGER NOT NULL DEFAULT 0,
    redactions INTEGER NOT NULL DEFAULT 0,
    text       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS utterances_episode ON utterances(episode_id);
CREATE INDEX IF NOT EXISTS utterances_start ON utterances(t_start);

-- Keyword index for exact names and numbers (external content: text lives once).
CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
    text, content = 'utterances', content_rowid = 'id', tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS utterances_ai AFTER INSERT ON utterances BEGIN
    INSERT INTO utterances_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS utterances_ad AFTER DELETE ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS utterances_au AFTER UPDATE OF text ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO utterances_fts(rowid, text) VALUES (new.id, new.text);
END;

-- The episode vector table is created by the store, because its form depends on
-- whether the sqlite-vec extension loads:
--   sqlite-vec: CREATE VIRTUAL TABLE episode_vec USING vec0(
--                   episode_id INTEGER PRIMARY KEY, embedding int8[384] distance_metric=cosine);
--   fallback:   CREATE TABLE episode_vec_blob(episode_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
--               (int8 bytes, searched with numpy; fine below ~100k episodes)

-- To-dos: added by command, confirmed from commitments Kin heard ("candidate"),
-- or saved as notes. A heard commitment keeps the sentence it came from until
-- that speech is forgotten or deleted.
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY,
    title        TEXT    NOT NULL,
    due_at       REAL,
    due_has_time INTEGER NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL DEFAULT 'open',  -- candidate | open | done | dismissed
    source       TEXT    NOT NULL,                 -- command | heard | note | ui
    episode_id   INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    utterance_id INTEGER REFERENCES utterances(id) ON DELETE SET NULL,
    quote        TEXT,
    created_at   REAL    NOT NULL,
    done_at      REAL
);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status, due_at);

CREATE TABLE IF NOT EXISTS reminders (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    fire_at    REAL NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending | fired | cancelled
    created_at REAL NOT NULL,
    fired_at   REAL
);
CREATE INDEX IF NOT EXISTS reminders_due ON reminders(status, fire_at);
