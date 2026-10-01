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
    done_at      REAL,
    notified_at  REAL                                -- when Kin last told the user it was due
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

-- People named in speech (P5). Rows exist only while some stored episode mentions them;
-- deleting or forgetting that speech removes them. Expiry of old transcripts keeps them.
CREATE TABLE IF NOT EXISTS people (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    aliases    TEXT NOT NULL DEFAULT '',
    first_seen REAL NOT NULL,
    last_seen  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS person_mentions (
    person_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    episode_id INTEGER            -- NULL once the transcript has expired
);
CREATE INDEX IF NOT EXISTS person_mentions_ep ON person_mentions(episode_id);

-- Claims someone made ("Rohan said the site visit is Thursday"), with the episodes they came from.
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY,
    subject    TEXT NOT NULL,
    claim      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_sources (
    fact_id    INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    episode_id INTEGER
);
CREATE INDEX IF NOT EXISTS fact_sources_ep ON fact_sources(episode_id);

-- One summary per day, written overnight (M3). Kept a year; deleted with the speech it summarises.
CREATE TABLE IF NOT EXISTS digests (
    day        TEXT PRIMARY KEY,   -- YYYY-MM-DD, local
    text       TEXT NOT NULL,
    n_episodes INTEGER NOT NULL,
    created_at REAL NOT NULL
);

-- Every exchange with Kin (spec section 11). Kept 7 days like transcripts.
CREATE TABLE IF NOT EXISTS turns (
    id        INTEGER PRIMARY KEY,
    ts        REAL NOT NULL,
    question  TEXT NOT NULL,
    intent    TEXT NOT NULL,
    answer    TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS turns_ts ON turns(ts);

-- Every online call, with the exact request that was sent (CL3: the single source of truth).
CREATE TABLE IF NOT EXISTS egress_log (
    id        INTEGER PRIMARY KEY,
    ts        REAL NOT NULL,
    host      TEXT NOT NULL,
    payload   TEXT NOT NULL,
    bytes_out INTEGER NOT NULL,
    bytes_in  INTEGER NOT NULL,
    status    TEXT NOT NULL
);
