"""
Smoke tests for the Flask REST API.
Tests use a mocked Camera so no real webcam is required.
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock Camera before importing app ────────────────────────────────

def _make_mock_camera():
    cam = MagicMock()
    cam.suspicion_score = 55.5
    cam.risk_level = "MEDIUM"
    cam.behaviour_state = "SUSPICIOUS"
    cam.fps = 28.4
    cam.face_count = 1
    cam.phone_detected = False
    cam.head_direction = "forward"
    cam.gaze_direction = "on-screen"
    cam.mouth_state = "closed"
    cam.hand_near_face = False
    cam.identity_mismatch = False
    cam.enrollment_complete = True
    cam.phone_violations = 0
    cam.multiple_face_violations = 0
    cam.head_violation_count = 2
    cam.gaze_violations = 3
    cam.mouth_violations = 1
    cam.identity_violations = 0
    cam.hand_violations = 0
    cam.session_start = time.time() - 120
    cam.score_history = [45.0, 50.0, 55.5]
    cam.frame_count = 300   # explicit int — MagicMock attributes are not JSON serialisable
    cam.timeline_data = [
        {"event_type": "gaze_off_screen", "timestamp": 10.0, "score": 40.0, "time": "10:00:10"}
    ]
    cam.event_log = cam.timeline_data.copy()
    cam.get_top_suspicion_signals.return_value = [("gaze_off_screen", 8)]
    cam.generate_session_summary.return_value = {
        "duration_seconds": 120.0, "total_events": 1,
        "avg_suspicion_score": 50.0, "max_suspicion_score": 55.5,
    }
    cam.simulation_overrides = {}
    return cam


@patch("camera.Camera", return_value=_make_mock_camera())
class TestFlaskAPI:
    """Smoke tests for all public Flask endpoints."""

    def _get_client(self, mock_camera_cls):
        from app import app
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-key"
        return app.test_client()

    def test_index_returns_200(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_stats_returns_json(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "suspicion_score" in data
        assert "risk_level" in data
        assert "fps" in data

    def test_status_alias_works(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_events_returns_list(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/events")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_report_returns_json(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/report")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "events" in data
        assert "total_events" in data

    def test_session_summary_returns_json(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/session_summary")
        assert resp.status_code == 200

    def test_enroll_status_returns_json(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/enroll_status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "enrollment_complete" in data

    def test_admin_redirects_to_login(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/admin")
        # Should redirect to login (302)
        assert resp.status_code in (302, 301)

    def test_export_requires_auth(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/export")
        assert resp.status_code == 403

    def test_history_requires_auth(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.get("/history")
        assert resp.status_code == 403

    def test_reset_returns_ok(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_admin_login_rejects_bad_password(self, mock_camera_cls):
        client = self._get_client(mock_camera_cls)
        resp = client.post(
            "/admin/login",
            json={"username": "admin", "password": "wrong"},
            content_type="application/json",
        )
        assert resp.status_code == 401

