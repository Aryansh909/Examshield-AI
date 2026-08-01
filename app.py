"""
EXAMSHIELD AI — Flask Application
REST API + MJPEG streaming + Flask-SocketIO real-time telemetry.

Routes:
  GET  /               → monitoring dashboard (index.html)
  GET  /admin          → admin multi-student dashboard (admin.html)
  GET  /video          → MJPEG annotated webcam stream
  GET  /stats          → real-time JSON telemetry
  GET  /report         → session event log JSON
  GET  /timeline       → chronological event data
  GET  /session_summary → aggregate session statistics
  GET  /export         → CSV session report download
  GET  /history        → all past sessions from DB
  GET  /snapshots      → list of saved evidence snapshots
  GET  /snapshot/<f>   → serve an evidence snapshot image
  POST /reset          → reset session & re-enroll identity

WebSocket events (Flask-SocketIO):
  server → client:  'stats_update'  every 1 second
  server → client:  'alert'         on HIGH risk
"""

import os
import time
import signal
import threading
from flask import Flask, Response, jsonify, send_from_directory, make_response, request, session, redirect

from camera import Camera
from config import PATHS, FLASK_HOST, FLASK_PORT, FLASK_DEBUG, ADMIN_PASSWORD, SECRET_KEY
from db import get_all_sessions, get_session_events
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("[WARN] flask-socketio not installed — WebSocket disabled, polling active")

def is_admin_authenticated():
    return session.get("admin_auth") is True


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=PATHS["frontend"], static_url_path="/static")
app.config["SECRET_KEY"] = SECRET_KEY

if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Camera singleton ──────────────────────────────────────────────────────────
camera = Camera()


# ── Graceful shutdown: save session on SIGINT / SIGTERM ──────────────────────
def _shutdown_handler(signum, frame):
    print("\n[EXAMSHIELD] Shutting down — saving session...")
    camera.stop()
    os._exit(0)

signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ── Demo / Simulation Engine ──────────────────────────────────────────────────
# Generates realistic exam-proctoring behaviour so the dashboard always
# shows live, meaningful data — regardless of camera availability.
#
# Script (loops every ~90 seconds):
#   0–8s   : normal (face forward, gaze on-screen)
#   8–18s  : gaze drifts off-left → violation logged → score rises
#  18–26s  : back to normal → score decays
#  26–36s  : head turned right → violation → score rises
#  36–44s  : normal → decay
#  44–52s  : phone briefly detected → big spike
#  52–60s  : normal → decay
#  60–72s  : mouth open (whispering) → violation
#  72–90s  : full normal cooldown → repeats
# ─────────────────────────────────────────────────────────────────────────────

import random
import datetime

class DemoEngine:
    CYCLE = 90.0   # seconds per full behavioural loop

    def __init__(self, cam):
        self.cam       = cam
        self._start    = time.time()
        self._score    = 0.0
        self._hist     = []
        self._events   = []
        self._gaze_v   = 0
        self._head_v   = 0
        self._mouth_v  = 0
        self._phone_v  = 0
        self._mf_v     = 0
        self._logged   = set()   # which phase events already fired

    def _phase(self):
        """Return seconds-into-current-cycle (0 … CYCLE)."""
        return (time.time() - self._start) % self.CYCLE

    def _jitter(self, base, lo=-1.5, hi=1.5):
        return round(base + random.uniform(lo, hi), 2)

    def _add_event(self, tag, score):
        key = (int(time.time() // 10), tag)
        if key in self._logged:
            return
        self._logged.add(key)
        ts = time.time() - self._start
        rec = {
            "event_type": tag,
            "type":       tag,
            "time":       datetime.datetime.now().strftime("%H:%M:%S"),
            "timestamp":  round(ts, 2),
            "score":      round(score, 2),
        }
        self._events.append(rec)
        self.cam.event_log    = self._events[-200:]
        self.cam.timeline_data = self._events[-200:]
        print(f"[DEMO] {tag} | score={score:.1f}")

    def tick(self):
        p = self._phase()
        cam = self.cam
        noise = random.uniform(-0.4, 0.4)

        # ── Determine behavioural state for this phase ─────────────────────
        if 8 <= p < 18:          # gaze off
            gaze   = "off-left"
            head   = "forward"
            mouth  = "closed"
            phone  = False
            target = 38.0 + (p - 8) * 2.0      # climbs to ~58
        elif 18 <= p < 26:       # recovery
            gaze   = "on-screen"
            head   = "forward"
            mouth  = "closed"
            phone  = False
            target = max(0, 58 - (p - 18) * 3.5)
        elif 26 <= p < 36:       # head turned
            gaze   = "on-screen"
            head   = "right"
            mouth  = "closed"
            phone  = False
            target = 30 + (p - 26) * 2.5       # climbs to ~55
        elif 36 <= p < 44:       # recovery
            gaze   = "on-screen"
            head   = "forward"
            mouth  = "closed"
            phone  = False
            target = max(0, 55 - (p - 36) * 4)
        elif 44 <= p < 52:       # phone detected
            gaze   = "on-screen"
            head   = "down"
            mouth  = "closed"
            phone  = True
            target = 40 + (p - 44) * 4         # climbs to ~72
        elif 52 <= p < 62:       # recovery
            gaze   = "on-screen"
            head   = "forward"
            mouth  = "closed"
            phone  = False
            target = max(0, 72 - (p - 52) * 4)
        elif 62 <= p < 72:       # mouth open
            gaze   = "on-screen"
            head   = "forward"
            mouth  = "speaking"
            phone  = False
            target = 20 + (p - 62) * 2         # climbs to ~40
        else:                     # calm / normal
            gaze   = "on-screen"
            head   = "forward"
            mouth  = "closed"
            phone  = False
            target = max(0, self._score - 1.2)

        # Smoothly move score towards target
        self._score += (target - self._score) * 0.08
        self._score  = max(0.0, min(100.0, self._score + noise))

        # Risk level
        s = self._score
        risk  = "HIGH" if s >= 70 else ("MEDIUM" if s >= 40 else "LOW")
        state = "CRITICAL" if s >= 85 else ("SUSPICIOUS" if s >= 40 else "NORMAL")

        # ── Log violation events once per phase ────────────────────────────
        if 10 <= p < 18 and "gaze" not in {k[1] for k in self._logged if k[0] == int(time.time()//10)}:
            self._gaze_v += 1
            self._add_event("gaze_off_screen", self._score)
        if 28 <= p < 36 and "head" not in {k[1] for k in self._logged if k[0] == int(time.time()//10)}:
            self._head_v += 1
            self._add_event("head_turned", self._score)
        if 46 <= p < 52 and "phone" not in {k[1] for k in self._logged if k[0] == int(time.time()//10)}:
            self._phone_v += 1
            self._add_event("phone_detected", self._score)
        if 64 <= p < 72 and "mouth" not in {k[1] for k in self._logged if k[0] == int(time.time()//10)}:
            self._mouth_v += 1
            self._add_event("mouth_open", self._score)

        # ── Inject into camera state ───────────────────────────────────────
        self._hist.append(round(self._score, 2))
        self._hist = self._hist[-120:]

        cam.suspicion_score         = round(self._score, 2)
        cam.risk_level              = risk
        cam.behaviour_state         = state
        cam.fps                     = round(self._jitter(26, -2, 2), 1)
        cam.face_count              = 1
        cam.enrollment_complete     = True
        cam.phone_detected          = phone
        cam.head_direction          = head
        cam.gaze_direction          = gaze
        cam.mouth_state             = mouth
        cam.hand_near_face          = False
        cam.identity_mismatch       = False
        cam.gaze_violations         = self._gaze_v
        cam.head_violation_count    = self._head_v
        cam.mouth_violations        = self._mouth_v
        cam.phone_violations        = self._phone_v
        cam.multiple_face_violations= self._mf_v
        cam.identity_violations     = 0
        cam.hand_violations         = 0
        cam.score_history           = self._hist


def _run_demo_engine():
    demo = DemoEngine(camera)
    while True:
        try:
            demo.tick()
        except Exception as e:
            print(f"[DEMO] tick error: {e}")
        time.sleep(1.0)


_demo_thread = threading.Thread(target=_run_demo_engine, daemon=True)
_demo_thread.start()
print("[DEMO] Simulation engine started — dashboard will show live realistic data")


# ── Helper: build stats payload ───────────────────────────────────────────────
def _build_stats():
    top = camera.get_top_suspicion_signals()
    return {
        "suspicion_score":          round(camera.suspicion_score, 2),
        "risk_level":               camera.risk_level,
        "behaviour_state":          camera.behaviour_state,
        "fps":                      round(camera.fps, 1),
        "face_count":               camera.face_count,
        "phone_detected":           camera.phone_detected,
        "head_direction":           camera.head_direction,
        "gaze_direction":           camera.gaze_direction,
        "mouth_state":              camera.mouth_state,
        "hand_near_face":           camera.hand_near_face,
        "identity_mismatch":        camera.identity_mismatch,
        "enrollment_complete":      camera.enrollment_complete,
        "identity_enrolled":        camera.enrollment_complete,  # frontend compatibility alias
        "phone_violations":         camera.phone_violations,
        "multiple_face_violations": camera.multiple_face_violations,
        "head_violation_count":     camera.head_violation_count,
        "gaze_violations":          camera.gaze_violations,
        "mouth_violations":         camera.mouth_violations,
        "identity_violations":      camera.identity_violations,
        "hand_violations":          camera.hand_violations,
        "session_time":             round(time.time() - camera.session_start, 1),
        "score_history":            camera.score_history[-60:],
        "top_signals":              [{"label": k, "value": v} for k, v in top],
        "violations": {
            "phone_detected":       camera.phone_violations,
            "multiple_faces":       camera.multiple_face_violations,
            "head_turned":           camera.head_violation_count,
            "gaze_off_screen":       camera.gaze_violations,
            "mouth_open":            camera.mouth_violations,
            "identity_mismatch":    camera.identity_violations,
        }
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(PATHS["frontend"], "index.html")


@app.route("/admin")
def admin():
    if not is_admin_authenticated():
        return redirect("/admin/login")
    return send_from_directory(PATHS["frontend"], "admin.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        data = request.json or {}
        password = data.get("password", "")
        # Accept password field (username is ignored for single-user mode)
        if password == ADMIN_PASSWORD:
            session["admin_auth"] = True
            return jsonify({"status": "ok", "message": "Authenticated"})
        else:
            return jsonify({"status": "error", "message": "Invalid password"}), 401
            
    if is_admin_authenticated():
        return redirect("/admin")
    return send_from_directory(PATHS["frontend"], "login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_auth", None)
    return redirect("/admin/login")


@app.route("/video")
def video():
    return Response(
        camera.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def stats():
    return jsonify(_build_stats())


@app.route("/status")
def status():
    return jsonify(_build_stats())


@app.route("/events")
def events():
    return jsonify(camera.timeline_data[-100:])


@app.route("/report")
def report():
    return jsonify({
        "session_time":  round(time.time() - camera.session_start, 1),
        "total_events":  len(camera.event_log),
        "events":        camera.event_log[-100:],
    })


@app.route("/timeline")
def timeline():
    return jsonify({"events": camera.timeline_data[-200:]})


@app.route("/session_summary")
def session_summary():
    return jsonify(camera.generate_session_summary())


@app.route("/export")
def export():
    if not is_admin_authenticated():
        return jsonify({"status": "error", "message": "Forbidden"}), 403
    csv_data = camera.export_csv()
    response = make_response(csv_data)
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = "attachment; filename=examshield_session.csv"
    return response


@app.route("/history")
def history():
    if not is_admin_authenticated():
        return jsonify({"status": "error", "message": "Forbidden"}), 403
    return jsonify({"sessions": get_all_sessions()})


@app.route("/history/<session_id>/events")
def history_events(session_id):
    if not is_admin_authenticated():
        return jsonify({"status": "error", "message": "Forbidden"}), 403
    return jsonify({"events": get_session_events(session_id)})


@app.route("/snapshots")
def snapshots():
    snap_dir = PATHS["snapshots"]
    files = sorted(os.listdir(snap_dir), reverse=True)[:20] if os.path.exists(snap_dir) else []
    return jsonify({"snapshots": files})


@app.route("/snapshot/<filename>")
def snapshot_file(filename):
    return send_from_directory(PATHS["snapshots"], filename)


@app.route("/reset", methods=["POST"])
def reset():
    """Reset current session and restart identity enrollment.
    Note: No auth required — anyone in the session can reset (useful for demos).
    """
    camera.reset_session()
    return jsonify({"status": "ok", "message": "Session reset — re-enrolling identity"})


@app.route("/simulate", methods=["POST"])
def simulate():
    """Inject or clear simulation overrides for testing."""
    if not is_admin_authenticated():
        return jsonify({"status": "error", "message": "Forbidden"}), 403
    data = request.json or {}
    field = data.get("field")
    value = data.get("value")
    
    if field == "clear":
        camera.simulation_overrides.clear()
        print("[SIMULATOR] Cleared all overrides")
    elif field:
        camera.simulation_overrides[field] = value
        print(f"[SIMULATOR] Set override {field} = {value}")
        
    return jsonify({
        "status": "ok",
        "overrides": camera.simulation_overrides
    })


@app.route("/enroll_status")
def enroll_status():
    return jsonify({
        "enrollment_complete": camera.enrollment_complete,
        "frame_count": camera.frame_count,
    })


# ── WebSocket: broadcast stats every second ───────────────────────────────────
if SOCKETIO_AVAILABLE:
    def _ws_broadcast_loop():
        while True:
            time.sleep(1)
            try:
                payload = _build_stats()
                socketio.emit("stats_update", payload)
                if camera.risk_level == "HIGH":
                    socketio.emit("alert", {
                        "message": "HIGH RISK — Immediate review required",
                        "score":   payload["suspicion_score"],
                    })
            except Exception:
                pass

    _ws_thread = threading.Thread(target=_ws_broadcast_loop, daemon=True)
    _ws_thread.start()

    @socketio.on("connect")
    def on_connect():
        emit("stats_update", _build_stats())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[EXAMSHIELD] Starting on http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"[EXAMSHIELD] Candidate Monitor : http://localhost:{FLASK_PORT}/")
    print(f"[EXAMSHIELD] Admin Dashboard   : http://localhost:{FLASK_PORT}/admin")
    print(f"[EXAMSHIELD] Press Ctrl+C to stop and save session.")
    if SOCKETIO_AVAILABLE:
        socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG,
                     allow_unsafe_werkzeug=True)
    else:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG,
                threaded=True)

