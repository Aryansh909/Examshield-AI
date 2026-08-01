"""
EXAMSHIELD AI — Database Persistence Layer
SQLite-backed event log and session summary storage.
Enables historical analysis and cross-session reporting.
"""

import sqlite3
import datetime
import uuid
from config import PATHS

DB_PATH = PATHS["database"]


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and indexes if they do not exist."""
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            event_type  TEXT    NOT NULL,
            session_ts  REAL    NOT NULL,
            score       REAL    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id               TEXT    NOT NULL UNIQUE,
            started_at               TEXT    NOT NULL,
            ended_at                 TEXT,
            duration_seconds         REAL,
            avg_suspicion_score      REAL,
            max_suspicion_score      REAL,
            high_risk_periods        INTEGER,
            total_events             INTEGER,
            phone_violations         INTEGER,
            multiple_face_violations INTEGER,
            head_violations          INTEGER,
            gaze_violations          INTEGER,
            mouth_violations         INTEGER,
            identity_violations      INTEGER
        )
    """)

    # Index for fast per-session event lookups
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_session
        ON events (session_id)
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialised at {DB_PATH}")


# ── Active session ID (set once per server start or reset) ────────────────────
CURRENT_SESSION_ID = str(uuid.uuid4())[:8]
SESSION_START_TIME = datetime.datetime.now().isoformat()

def reset_session_db():
    """Generate a new session ID and start timestamp for a fresh session."""
    global CURRENT_SESSION_ID, SESSION_START_TIME
    CURRENT_SESSION_ID = str(uuid.uuid4())[:8]
    SESSION_START_TIME = datetime.datetime.now().isoformat()
    print(f"[DB] Session reset — new ID: {CURRENT_SESSION_ID}")
    return CURRENT_SESSION_ID



def log_event_db(event_type: str, session_ts: float, score: float):
    """Persist a single violation event."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO events (session_id, event_type, session_ts, score) VALUES (?,?,?,?)",
        (CURRENT_SESSION_ID, event_type, session_ts, score)
    )
    conn.commit()
    conn.close()


def save_session_summary_db(summary: dict):
    """Persist session summary on shutdown or reset."""
    conn = _get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO sessions
        (session_id, started_at, ended_at, duration_seconds,
         avg_suspicion_score, max_suspicion_score, high_risk_periods,
         total_events, phone_violations, multiple_face_violations,
         head_violations, gaze_violations, mouth_violations, identity_violations)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        CURRENT_SESSION_ID,
        SESSION_START_TIME,
        datetime.datetime.now().isoformat(),
        summary.get("duration_seconds"),
        summary.get("avg_suspicion_score"),
        summary.get("max_suspicion_score"),
        summary.get("high_risk_periods"),
        summary.get("total_events"),
        summary.get("phone_violations"),
        summary.get("multiple_face_violations"),
        summary.get("head_violations"),
        summary.get("gaze_violations"),
        summary.get("mouth_violations"),
        summary.get("identity_violations"),
    ))
    conn.commit()
    conn.close()


def get_all_sessions():
    """Retrieve all past session summaries, most recent first."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_events(session_id: str):
    """Retrieve all events for a given session."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM events WHERE session_id=? ORDER BY session_ts",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_by_id(session_id: str):
    """Retrieve a single session summary by ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id=?",
        (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# Auto-init on import
init_db()

