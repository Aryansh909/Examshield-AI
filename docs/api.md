# ExamShield AI — API Reference

Base URL: `http://localhost:5000`

All JSON responses use `Content-Type: application/json`.

---

## Public Endpoints

### `GET /`
Serves the candidate monitoring dashboard (`index.html`).

### `GET /video`
MJPEG stream of the annotated webcam feed.
- **MIME**: `multipart/x-mixed-replace; boundary=frame`
- Use in `<img src="/video" />` for live feed

### `GET /stats` or `GET /status`
Real-time telemetry snapshot. Both routes return identical data.

**Response:**
```json
{
  "suspicion_score": 45.2,
  "risk_level": "MEDIUM",
  "behaviour_state": "SUSPICIOUS",
  "fps": 28.4,
  "face_count": 1,
  "phone_detected": false,
  "head_direction": "forward",
  "gaze_direction": "on-screen",
  "mouth_state": "closed",
  "hand_near_face": false,
  "identity_mismatch": false,
  "enrollment_complete": true,
  "identity_enrolled": true,
  "phone_violations": 1,
  "multiple_face_violations": 0,
  "head_violation_count": 2,
  "gaze_violations": 3,
  "mouth_violations": 1,
  "identity_violations": 0,
  "hand_violations": 0,
  "session_time": 142.5,
  "score_history": [40.0, 42.1, 45.2],
  "top_signals": [{"label": "gaze_off_screen", "value": 8}],
  "violations": {
    "phone_detected": 1,
    "multiple_faces": 0,
    "head_turned": 2,
    "gaze_off_screen": 3,
    "mouth_open": 1,
    "identity_mismatch": 0
  }
}
```

### `GET /events`
Last 100 violation events.
```json
[
  {
    "event_type": "gaze_off_screen",
    "type": "gaze_off_screen",
    "time": "14:32:10",
    "timestamp": 42.5,
    "score": 45.2
  }
]
```

### `GET /report`
Session summary with full event log.
```json
{
  "session_time": 142.5,
  "total_events": 12,
  "events": [ ... ]
}
```

### `GET /timeline`
Last 200 chronological events (alias of `/events` with larger window).

### `GET /session_summary`
Aggregate statistics for the current session.
```json
{
  "duration_seconds": 600.0,
  "total_events": 24,
  "avg_suspicion_score": 38.5,
  "max_suspicion_score": 87.2,
  "high_risk_periods": 3,
  "phone_violations": 1,
  "multiple_face_violations": 0,
  "head_violations": 5,
  "gaze_violations": 12,
  "mouth_violations": 4,
  "identity_violations": 0
}
```

### `GET /enroll_status`
Identity enrollment state.
```json
{
  "enrollment_complete": true,
  "frame_count": 120
}
```

### `GET /snapshots`
List of saved evidence snapshots (last 20).
```json
{
  "snapshots": ["evidence_20260801_143210_score87.jpg", ...]
}
```

### `GET /snapshot/<filename>`
Serve a specific evidence snapshot image.

### `POST /reset`
Reset the current session and restart identity enrollment.
```json
// Response
{"status": "ok", "message": "Session reset — re-enrolling identity"}
```

---

## Admin-Only Endpoints
Require admin session (authenticate via `POST /admin/login` first).

### `GET /admin`
Admin multi-student dashboard.

### `GET /admin/login` (GET)
Serves the admin login page.

### `POST /admin/login`
```json
// Request
{"username": "admin", "password": "admin"}

// Success
{"status": "ok", "message": "Authenticated"}

// Failure (401)
{"status": "error", "message": "Invalid password"}
```

### `GET /admin/logout`
Clears admin session, redirects to login.

### `GET /export` 🔒
Download the current session as a CSV file.
- **Content-Type**: `text/csv`
- **Auth required**: Yes

### `GET /history` 🔒
All past session summaries from SQLite.
```json
{"sessions": [{"session_id": "abc12345", "started_at": "...", ...}]}
```

### `GET /history/<session_id>/events` 🔒
All events for a past session by ID.

### `POST /simulate` 🔒
Inject testing overrides without physical violations.
```json
// Set a signal
{"field": "phone_detected", "value": true}

// Clear all overrides
{"field": "clear"}

// Response
{"status": "ok", "overrides": {"phone_detected": true}}
```

---

## WebSocket Events

Connect via Socket.IO: `io('http://localhost:5000')`

| Event | Direction | Payload |
|-------|-----------|--------|
| `stats_update` | Server → Client | Same as `/stats` JSON |
| `alert` | Server → Client | `{"message": "HIGH RISK", "score": 87.2}` |

```javascript
const socket = io('http://localhost:5000');
socket.on('stats_update', (data) => {
  console.log('Score:', data.suspicion_score);
});
socket.on('alert', (data) => {
  alert(data.message);
});
```
