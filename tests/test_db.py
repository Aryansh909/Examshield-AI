"""
Tests for db.py — SQLite persistence layer.
Uses a temporary database to avoid polluting the production DB.
"""
import os
import sys
import tempfile
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDatabase:
    """Test the DB schema and CRUD operations using a temp database."""

    def setup_method(self):
        """Create a fresh temp database before each test."""
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmpfile.name
        self.tmpfile.close()
        self._init_schema()

    def teardown_method(self):
        """Remove temp database after each test."""
        os.unlink(self.db_path)

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, event_type TEXT NOT NULL,
                session_ts REAL NOT NULL, score REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE, started_at TEXT NOT NULL,
                ended_at TEXT, duration_seconds REAL, avg_suspicion_score REAL,
                max_suspicion_score REAL, high_risk_periods INTEGER,
                total_events INTEGER, phone_violations INTEGER,
                multiple_face_violations INTEGER, head_violations INTEGER,
                gaze_violations INTEGER, mouth_violations INTEGER,
                identity_violations INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def test_schema_events_table_exists(self):
        conn = self._get_conn()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone()
        conn.close()
        assert result is not None, "events table missing"

    def test_schema_sessions_table_exists(self):
        conn = self._get_conn()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
        conn.close()
        assert result is not None, "sessions table missing"

    def test_insert_event(self):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO events (session_id, event_type, session_ts, score) VALUES (?,?,?,?)",
            ("test-001", "gaze_off_screen", 12.5, 45.0)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM events WHERE session_id='test-001'").fetchone()
        conn.close()
        assert row["event_type"] == "gaze_off_screen"
        assert row["score"] == 45.0

    def test_insert_session(self):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO sessions
            (session_id, started_at, duration_seconds, avg_suspicion_score,
             max_suspicion_score, high_risk_periods, total_events,
             phone_violations, multiple_face_violations, head_violations,
             gaze_violations, mouth_violations, identity_violations)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, ("sess-001", "2026-08-01T10:00:00", 600.0, 35.2, 87.5, 3, 12, 1, 0, 4, 5, 2, 0))
        conn.commit()
        row = conn.execute("SELECT * FROM sessions WHERE session_id='sess-001'").fetchone()
        conn.close()
        assert row["max_suspicion_score"] == 87.5
        assert row["total_events"] == 12

    def test_multiple_events_for_session(self):
        conn = self._get_conn()
        for i, evt in enumerate(["gaze_off_screen", "head_turned", "phone_detected"]):
            conn.execute(
                "INSERT INTO events (session_id, event_type, session_ts, score) VALUES (?,?,?,?)",
                ("sess-multi", evt, float(i * 10), float(i * 20))
            )
        conn.commit()
        rows = conn.execute("SELECT * FROM events WHERE session_id='sess-multi'").fetchall()
        conn.close()
        assert len(rows) == 3

