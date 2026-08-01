"""
EXAMSHIELD AI — Camera Class
Central orchestrator for the full detection pipeline:
  - YOLO (phone + face detection)
  - MediaPipe (facial geometry + hands + blendshapes for head/gaze/mouth)
  - Identity enrollment & verification
  - Signal fusion & suspicion scoring
  - Event logging & session reporting
  - Evidence snapshot capture
"""

import cv2
import time
import threading
import os
import csv
import io
import datetime
import numpy as np
from collections import deque, Counter

from config import (
    PATHS, ML_CONFIG, SCORE_WEIGHTS, SCORE_DECAY,
    RISK_THRESHOLDS, SNAPSHOT_THRESHOLD, HEAD_VIOLATION_CONSECUTIVE, CAMERA_INDEX
)

# ── Optional heavy imports (graceful fallback if models not present) ──────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — YOLO detection disabled")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[WARN] mediapipe not installed — landmark detection disabled")

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch/torchvision not installed — CNN models disabled")

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[WARN] face_recognition not installed — identity verification disabled")

try:
    from db import log_event_db, save_session_summary_db, reset_session_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


# ── CNN transform pipeline ────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    CNN_TRANSFORM = transforms.Compose([
        transforms.Resize((ML_CONFIG["input_size"], ML_CONFIG["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def _load_cnn(path: str):
    """Load a MobileNetV2 checkpoint (.pth) saved as {state_dict, encoder}."""
    if not TORCH_AVAILABLE or not os.path.exists(path):
        return None, None
    try:
        ckpt = torch.load(path, map_location="cpu")
        encoder = ckpt["encoder"]
        num_classes = len(encoder.classes_)
        model = mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, encoder
    except Exception as e:
        print(f"[WARN] CNN load failed for {path}: {e}")
        return None, None


def _cnn_predict(model, encoder, frame_rgb):
    """Run a single CNN forward pass and return the predicted class label."""
    if model is None:
        return None
    try:
        img = Image.fromarray(frame_rgb)
        tensor = CNN_TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        idx = out.argmax(dim=1).item()
        return encoder.inverse_transform([idx])[0]
    except Exception:
        return None


# ── Blendshape helpers (MediaPipe) ────────────────────────────────────────────
def _get_blendshape(blendshapes, name: str, default: float = 0.0) -> float:
    """Lookup a blendshape score by category name."""
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return default


def _infer_head_direction_from_landmarks(face_landmarks) -> str:
    """
    Estimate head pose from nose and face edge landmarks.
    Uses horizontal (yaw) and vertical (pitch) deviation of nose tip
    relative to the midpoint of the eye corners.
    """
    if not face_landmarks:
        return "forward"
    lms = face_landmarks[0]
    # Key landmark indices (MediaPipe 478-point model)
    nose_tip   = lms[1]    # nose tip
    left_eye   = lms[33]   # left eye outer corner
    right_eye  = lms[263]  # right eye outer corner
    chin       = lms[152]  # chin center

    mid_x = (left_eye.x + right_eye.x) / 2.0
    mid_y = (left_eye.y + chin.y) / 2.0

    dx = nose_tip.x - mid_x   # negative = looking left, positive = looking right
    dy = nose_tip.y - mid_y   # negative = looking up, positive = looking down

    YAW_THRESH   = 0.04
    PITCH_THRESH = 0.05

    if abs(dx) > abs(dy):
        if dx < -YAW_THRESH:
            return "left"
        elif dx > YAW_THRESH:
            return "right"
    else:
        if dy < -PITCH_THRESH:
            return "up"
        elif dy > PITCH_THRESH:
            return "down"
    return "forward"


def _infer_gaze_from_blendshapes(blendshapes) -> str:
    """
    Estimate gaze direction from eye look blendshapes.
    Returns 'on-screen', 'off-left', 'off-right', 'off-up', or 'off-down'.
    Threshold raised to 0.45 to reduce false positives from natural eye movement.
    """
    if not blendshapes:
        return "on-screen"
    bs = blendshapes[0]
    look_left  = _get_blendshape(bs, "eyeLookOutLeft")  + _get_blendshape(bs, "eyeLookInRight")
    look_right = _get_blendshape(bs, "eyeLookOutRight") + _get_blendshape(bs, "eyeLookInLeft")
    look_up    = _get_blendshape(bs, "eyeLookUpLeft")   + _get_blendshape(bs, "eyeLookUpRight")
    look_down  = _get_blendshape(bs, "eyeLookDownLeft") + _get_blendshape(bs, "eyeLookDownRight")

    GAZE_THRESH = 0.50   # raised: only flag clear/deliberate off-screen gaze
    max_gaze = max(look_left, look_right, look_up, look_down)

    if max_gaze < GAZE_THRESH:
        return "on-screen"

    if max_gaze == look_left:
        return "off-left"
    elif max_gaze == look_right:
        return "off-right"
    elif max_gaze == look_up:
        return "off-up"
    else:
        return "off-down"


def _infer_mouth_from_blendshapes(blendshapes) -> str:
    """
    Estimate mouth state from jawOpen blendshape.
    Returns 'closed', 'open', or 'speaking'.
    """
    if not blendshapes:
        return "closed"
    bs = blendshapes[0]
    jaw_open = _get_blendshape(bs, "jawOpen")
    if jaw_open > 0.35:
        return "speaking"
    elif jaw_open > 0.15:
        return "open"
    return "closed"


# ── Camera ────────────────────────────────────────────────────────────────────
class Camera:
    """
    Central orchestration object.
    Runs a background thread that continuously processes webcam frames
    and exposes state to the Flask API layer.
    """

    def __init__(self):
        # ── Video capture (auto-retry if primary index is locked) ─────────────
        self.cap = None
        for cam_idx in [CAMERA_INDEX, 0, 1, 2]:
            cap = cv2.VideoCapture(cam_idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    print(f"[OK] Camera opened on index {cam_idx}")
                    break
                cap.release()
            else:
                cap.release()

        if self.cap is None:
            print("[WARN] No camera could be opened — running in no-camera mode")
            self.cap = cv2.VideoCapture(CAMERA_INDEX)   # keep handle for graceful fail

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.latest_frame = None
        self.frame_count = 0
        self.fps = 0.0
        self._fps_start = time.time()
        self._fps_frames = 0


        # ── YOLO models ───────────────────────────────────────────────────────
        self.yolo_phone = None
        self.yolo_face = None
        if YOLO_AVAILABLE:
            try:
                yolo_phone_path = PATHS["yolo_phone"]
                yolo_face_path  = PATHS["yolo_face"]
                if os.path.exists(yolo_phone_path):
                    self.yolo_phone = YOLO(yolo_phone_path)
                    self.yolo_phone.to("cpu")
                    print("[OK] YOLO phone model loaded:", yolo_phone_path)
                else:
                    print("[WARN] YOLO phone model not found:", yolo_phone_path)
                if os.path.exists(yolo_face_path):
                    self.yolo_face = YOLO(yolo_face_path)
                    self.yolo_face.to("cpu")
                    print("[OK] YOLO face model loaded:", yolo_face_path)
                else:
                    print("[INFO] No YOLO face model — using MediaPipe face count")
            except Exception as e:
                print(f"[WARN] YOLO load failed: {e}")

        # ── MediaPipe ─────────────────────────────────────────────────────────
        self.mp_face_result = None
        self.mp_hand_result = None
        self._mp_face_lm = None
        self._mp_hand_lm = None
        self._face_center_y = 0.5   # normalised; updated from MediaPipe landmarks
        self._mp_timestamp_ms = 0   # monotonically increasing timestamp for MP

        if MP_AVAILABLE and os.path.exists(PATHS["mp_face"]):
            try:
                face_opts = mp_python.BaseOptions(model_asset_path=PATHS["mp_face"])
                self._mp_face_lm = mp_vision.FaceLandmarker.create_from_options(
                    mp_vision.FaceLandmarkerOptions(
                        base_options=face_opts,
                        running_mode=mp_vision.RunningMode.LIVE_STREAM,
                        result_callback=self._mp_face_callback,
                        num_faces=4,
                        output_face_blendshapes=True,  # Enable for head/gaze/mouth detection
                    )
                )
                print("[OK] MediaPipe Face Landmarker loaded (with blendshapes)")
            except Exception as e:
                print(f"[WARN] MediaPipe Face load failed: {e}")

        if MP_AVAILABLE and os.path.exists(PATHS["mp_hand"]):
            try:
                hand_opts = mp_python.BaseOptions(model_asset_path=PATHS["mp_hand"])
                self._mp_hand_lm = mp_vision.HandLandmarker.create_from_options(
                    mp_vision.HandLandmarkerOptions(
                        base_options=hand_opts,
                        running_mode=mp_vision.RunningMode.LIVE_STREAM,
                        result_callback=self._mp_hand_callback,
                        num_hands=2,
                    )
                )
                print("[OK] MediaPipe Hand Landmarker loaded")
            except Exception as e:
                print(f"[WARN] MediaPipe Hand load failed: {e}")

        # ── Custom CNN models (optional — overrides MediaPipe if present) ─────
        self.head_model,  self.head_encoder  = _load_cnn(PATHS["head_model"])
        self.gaze_model,  self.gaze_encoder  = _load_cnn(PATHS["gaze_model"])
        self.mouth_model, self.mouth_encoder = _load_cnn(PATHS["mouth_model"])
        if self.head_model:
            print("[OK] CNN head model loaded")
        if self.gaze_model:
            print("[OK] CNN gaze model loaded")
        if self.mouth_model:
            print("[OK] CNN mouth model loaded")

        # ── Identity enrollment ───────────────────────────────────────────────
        self.enrolled_embedding = None
        self.enrollment_complete = False
        self._enroll_frame_count = 0

        # ── Violation counters ────────────────────────────────────────────────
        self.phone_violations         = 0
        self.multiple_face_violations = 0
        self.no_face_violations       = 0
        self.head_violation_count     = 0
        self._head_consec             = 0
        self.gaze_violations          = 0
        self.mouth_violations         = 0
        self.identity_violations      = 0
        self.hand_violations          = 0

        # ── Current detection state ───────────────────────────────────────────
        self.head_direction    = "forward"
        self.gaze_direction    = "on-screen"
        self.mouth_state       = "closed"
        self.face_count        = 0
        self.phone_detected    = False
        self.hand_near_face    = False
        self.identity_mismatch = False

        # ── Active flags for edge-triggered logging ───────────────────────────
        self._phone_active             = False
        self._multiple_faces_active    = False
        self._no_face_active           = False
        self._head_turned_active       = False
        self._gaze_off_active          = False
        self._mouth_open_active        = False
        self._identity_mismatch_active = False
        self._hand_active              = False

        # ── Identity verification state ───────────────────────────────────────
        self._identity_mismatch_state    = False
        self._identity_consec_mismatches = 0

        # ── Debounce trackers (time-based) ────────────────────────────────────
        self._first_seen_phone          = 0.0
        self._first_seen_multiple_faces = 0.0
        self._first_seen_no_face        = 0.0
        self._first_seen_head           = 0.0
        self._first_seen_gaze           = 0.0
        self._first_seen_mouth          = 0.0
        self._first_seen_hand           = 0.0

        self.DEBOUNCE_DURATIONS = {
            "phone":          1.5,
            "multiple_faces": 1.5,
            "no_face":        3.0,
            "head":           1.5,   # must be turned for 1.5s to count
            "gaze":           2.0,   # must be off-screen for 2s to count
            "mouth":          2.0,   # must be open/speaking for 2s to count
            "hand":           1.0,
        }

        # ── Per-event-type cooldown (min seconds between consecutive same events) ─
        # Prevents logging hundreds of gaze/head events in one session
        self.EVENT_COOLDOWNS = {
            "gaze_off_screen":   30.0,   # at most once every 30s
            "head_turned":       25.0,
            "mouth_open":        25.0,
            "phone_detected":    45.0,
            "multiple_faces":    45.0,
            "hand_near_face":    20.0,
            "identity_mismatch": 60.0,
        }
        self._last_event_times = {}   # {event_type: last log timestamp}

        # ── Hysteresis: how long condition must be CLEAR before resetting ─────
        # Prevents flicker from immediately re-triggering a violation
        self.CLEAR_DURATIONS = {
            "gaze":           3.0,   # must look at screen for 3s before gaze resets
            "head":           2.0,   # must face forward for 2s before head resets
            "mouth":          2.0,
            "hand":           1.5,
            "phone":          2.5,
            "multiple_faces": 2.5,
        }
        self._clear_timers = {}      # {key: timestamp when condition first became clear}

        # ── State smoothing buffers (majority vote over last N frames) ────────
        self._head_buf  = deque(maxlen=9)   # ~1 second at ~9fps
        self._gaze_buf  = deque(maxlen=9)
        self._mouth_buf = deque(maxlen=7)

        # ── Suspicion scoring ─────────────────────────────────────────────────
        self.suspicion_score   = 0.0
        self.score_history     = []
        self.behaviour_state   = "NORMAL"
        self.risk_level        = "LOW"
        self.session_start     = time.time()
        self._last_score_time  = time.time()
        self.high_risk_periods = 0
        self.max_score         = 0.0
        self._score_sum        = 0.0
        self._score_n          = 0

        # ── Event log ─────────────────────────────────────────────────────────
        self.event_log     = []
        self.timeline_data = []

        # ── Evidence snapshots ────────────────────────────────────────────────
        os.makedirs(PATHS["snapshots"], exist_ok=True)
        self._last_snapshot_time = 0

        # ── Inference timing ──────────────────────────────────────────────────
        self._last_inference     = 0.0
        self._last_yolo_time     = 0.0

        # ── Identity verification timer ───────────────────────────────────────
        self._last_verify_time   = 0.0
        self._verify_interval    = 2.0
        self._verify_in_progress = False

        # ── Simulation Overrides ──────────────────────────────────────────────
        self.simulation_overrides = {}

        # ── Background threads ────────────────────────────────────────────────
        self._running     = True
        self._lock        = threading.Lock()
        self._raw_frame   = None   # clean BGR frame for YOLO (not annotated)

        # Main frame processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        # YOLO phone detection runs in a dedicated thread so it never blocks the main loop
        self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
        self._yolo_thread.start()

        print("[OK] Camera processing thread started")
        print("[OK] YOLO detection thread started")

    # ── MediaPipe callbacks ───────────────────────────────────────────────────
    def _mp_face_callback(self, result, output_image, timestamp_ms):
        self.mp_face_result = result
        if result and result.face_landmarks:
            lms = result.face_landmarks[0]
            nose = lms[1]
            self._face_center_y = nose.y

            # Update face count from MediaPipe (when no YOLO face model)
            if not self.yolo_face:
                self.face_count = len(result.face_landmarks)

            # ── Smoothed head direction (majority vote, last 9 frames) ────────
            if not self.head_model:
                raw_head = _infer_head_direction_from_landmarks(result.face_landmarks)
                self._head_buf.append(raw_head)
                self.head_direction = Counter(self._head_buf).most_common(1)[0][0]

            # ── Smoothed gaze direction (majority vote, last 9 frames) ────────
            if not self.gaze_model and result.face_blendshapes:
                raw_gaze = _infer_gaze_from_blendshapes(result.face_blendshapes)
                self._gaze_buf.append(raw_gaze)
                self.gaze_direction = Counter(self._gaze_buf).most_common(1)[0][0]

            # ── Smoothed mouth state (majority vote, last 7 frames) ───────────
            if not self.mouth_model and result.face_blendshapes:
                raw_mouth = _infer_mouth_from_blendshapes(result.face_blendshapes)
                self._mouth_buf.append(raw_mouth)
                self.mouth_state = Counter(self._mouth_buf).most_common(1)[0][0]
        else:
            # No face detected — reset posture states
            if not self.yolo_face:
                self.face_count = 0
            if not self.head_model:
                self._head_buf.clear()
                self.head_direction = "forward"
            if not self.gaze_model:
                self._gaze_buf.clear()
                self.gaze_direction = "on-screen"
            if not self.mouth_model:
                self._mouth_buf.clear()
                self.mouth_state = "closed"

    def _mp_hand_callback(self, result, output_image, timestamp_ms):
        self.mp_hand_result = result

    # ── YOLO background thread ────────────────────────────────────────────────
    def _yolo_loop(self):
        """Dedicated daemon thread: runs YOLO every 0.5s on the raw camera frame.
        Keeps main loop free for MediaPipe + streaming at full speed."""
        while self._running:
            time.sleep(0.5)
            with self._lock:
                frame = self._raw_frame   # use clean unannotated BGR frame
            if frame is None:
                continue
            try:
                self._run_yolo(frame)
            except Exception as e:
                print(f"[WARN] YOLO thread error: {e}")

    # ── Main processing loop ──────────────────────────────────────────────────
    def _process_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self.frame_count += 1
            self._fps_frames += 1
            now = time.time()

            # FPS calculation
            elapsed = now - self._fps_start
            if elapsed >= 1.0:
                self.fps = self._fps_frames / elapsed
                self._fps_start = now
                self._fps_frames = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Identity enrollment (retry every frame after warmup) ───────────
            if not self.enrollment_complete:
                self._enroll_frame_count += 1
                if self._enroll_frame_count >= ML_CONFIG["warmup_frames"]:
                    self._try_enroll(rgb)

            # ── MediaPipe inference (every frame, async callback updates state) ─
            self._run_mediapipe(rgb, now)

            # ── CNN inference (every interval) — only if CNN models exist ──────
            if (self.head_model or self.gaze_model or self.mouth_model):
                if now - self._last_inference >= ML_CONFIG["interval"]:
                    self._run_cnn(rgb)
                    self._last_inference = now

            # ── Identity verification ─────────────────────────────────────────
            if (self.enrollment_complete and
                    not self._verify_in_progress and
                    now - self._last_verify_time >= self._verify_interval):
                self._verify_in_progress = True
                self._last_verify_time = now
                self._verify_identity_async(rgb.copy(), now)

            # ── Simulation Overrides ──────────────────────────────────────────
            if self.simulation_overrides:
                for key in ("phone_detected", "face_count", "head_direction",
                            "gaze_direction", "mouth_state", "identity_mismatch",
                            "hand_near_face"):
                    if key in self.simulation_overrides:
                        setattr(self, key, self.simulation_overrides[key])

            # ── Sync identity mismatch state from background thread ───────────
            if "identity_mismatch" not in self.simulation_overrides:
                self.identity_mismatch = self._identity_mismatch_state

            # ── Unified violations processing ─────────────────────────────────
            self._process_violations(now)

            # ── Signal fusion & scoring ───────────────────────────────────────
            self._update_score(now)

            # ── Annotate frame and store ──────────────────────────────────────
            annotated = self._annotate(frame.copy())

            # ── Evidence snapshot if HIGH risk (120s cooldown) ───────────────
            if (self.risk_level == "HIGH" and
                    now - self._last_snapshot_time > 120):  # 2min cooldown
                self._save_snapshot(annotated, now)
                self._last_snapshot_time = now

            with self._lock:
                self._raw_frame   = frame        # store clean frame for YOLO thread
                self.latest_frame = annotated    # store annotated frame for streaming

            # Small yield to prevent 100% CPU pinning
            time.sleep(0.010)

    # ── YOLO ──────────────────────────────────────────────────────────────────
    def _run_yolo(self, frame):
        """Run YOLO inference on a clean (unannotated) BGR frame."""
        # Phone detection — COCO class 67 = cell phone
        if self.yolo_phone:
            try:
                phone_res = self.yolo_phone(frame, verbose=False, conf=0.22, classes=[67])
                self.phone_detected = any(len(r.boxes) > 0 for r in phone_res)
            except Exception as e:
                print(f"[ERROR] YOLO phone: {e}")
                self.phone_detected = False
        else:
            self.phone_detected = False

        # Face count — use YOLO face model if available, else MediaPipe count
        if self.yolo_face:
            try:
                face_res = self.yolo_face(frame, verbose=False, conf=0.4)
                self.face_count = sum(len(r.boxes) for r in face_res)
            except Exception as e:
                print(f"[ERROR] YOLO face: {e}")
                self._update_face_count_from_mediapipe()
        # else: face_count is updated directly in _mp_face_callback

    def _update_face_count_from_mediapipe(self):
        """Use MediaPipe face landmark count as face count fallback."""
        if self.mp_face_result is not None:
            lms = self.mp_face_result.face_landmarks
            self.face_count = len(lms) if lms else 0
        # If mp_face_result is still None (first frames), keep face_count as 0

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    def _run_mediapipe(self, rgb, now):
        # Use monotonically increasing timestamp (required by MediaPipe LIVE_STREAM)
        ts_ms = int(now * 1000)
        if ts_ms <= self._mp_timestamp_ms:
            ts_ms = self._mp_timestamp_ms + 1
        self._mp_timestamp_ms = ts_ms

        if self._mp_face_lm:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                self._mp_face_lm.detect_async(mp_img, ts_ms)
            except Exception as e:
                pass

        if self._mp_hand_lm:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                self._mp_hand_lm.detect_async(mp_img, ts_ms)
            except Exception:
                pass

        # Hand proximity to face check
        self.hand_near_face = False
        if self.mp_hand_result and self.mp_hand_result.hand_landmarks:
            for hand_lms in self.mp_hand_result.hand_landmarks:
                wrist_y = hand_lms[0].y
                if abs(wrist_y - self._face_center_y) < 0.25:
                    self.hand_near_face = True
                    break

    # ── CNN classifiers (used only if .pth files exist) ───────────────────────
    def _run_cnn(self, rgb):
        if self.head_model:
            pred = _cnn_predict(self.head_model, self.head_encoder, rgb)
            if pred:
                self.head_direction = pred
        if self.gaze_model:
            pred = _cnn_predict(self.gaze_model, self.gaze_encoder, rgb)
            if pred:
                self.gaze_direction = pred
        if self.mouth_model:
            pred = _cnn_predict(self.mouth_model, self.mouth_encoder, rgb)
            if pred:
                self.mouth_state = pred

    # ── Suspicion scoring engine ──────────────────────────────────────────────
    def _update_score(self, now):
        dt = now - self._last_score_time
        if dt <= 0:
            dt = 0.033  # fallback ~30fps
        self._last_score_time = now

        # Accumulate active signal weights
        delta = 0.0
        if self._phone_active:
            delta += SCORE_WEIGHTS["phone_detected"]
        if self._multiple_faces_active:
            delta += SCORE_WEIGHTS["multiple_faces"]
        if self._no_face_active:
            delta += SCORE_WEIGHTS["no_face"]
        if self._head_turned_active:
            delta += SCORE_WEIGHTS["head_turned"]
        if self._gaze_off_active:
            delta += SCORE_WEIGHTS["gaze_off_screen"]
        if self._mouth_open_active:
            delta += SCORE_WEIGHTS["mouth_open"]
        if self._identity_mismatch_active:
            delta += SCORE_WEIGHTS["identity_mismatch"]
        if self._hand_active:
            delta += SCORE_WEIGHTS["hand_near_face"]

        # Score increment/decay — rate-independent via dt scaling
        # Weights are in "points-per-second" units at 1x speed
        SCORE_INCREMENT_RATE = 0.8   # multiplier on weight
        SCORE_DECAY_RATE     = 3.0   # points per second decay

        if delta > 0:
            self.suspicion_score = min(100.0, self.suspicion_score + delta * dt * SCORE_INCREMENT_RATE)
        else:
            self.suspicion_score = max(0.0, self.suspicion_score - SCORE_DECAY_RATE * dt)

        self.max_score = max(self.max_score, self.suspicion_score)
        self._score_sum += self.suspicion_score
        self._score_n   += 1

        # Behaviour state thresholds
        s = self.suspicion_score
        if s < 40:
            self.behaviour_state = "NORMAL"
            self.risk_level = "LOW"
        elif s < 70:
            self.behaviour_state = "SUSPICIOUS"
            self.risk_level = "MEDIUM"
        else:
            self.behaviour_state = "CRITICAL"
            self.risk_level = "HIGH"
            self.high_risk_periods += 1

        # Rolling score history
        self.score_history.append({"t": round(now - self.session_start, 1),
                                   "score": round(self.suspicion_score, 2)})
        if len(self.score_history) > 120:
            self.score_history.pop(0)

    # ── Identity enrollment & verification ────────────────────────────────────
    def _try_enroll(self, rgb):
        if not FACE_REC_AVAILABLE:
            self.enrollment_complete = True
            return
        try:
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                self.enrolled_embedding = encodings[0]
                self.enrollment_complete = True
                # Save enrolled face snapshot
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                face_locs = face_recognition.face_locations(rgb)
                if face_locs:
                    top, right, bottom, left = face_locs[0]
                    h, w = bgr.shape[:2]
                    top    = max(0, top - 40)
                    bottom = min(h, bottom + 40)
                    left   = max(0, left - 40)
                    right  = min(w, right + 40)
                    face_crop = bgr[top:bottom, left:right]
                    cv2.imwrite(os.path.join(PATHS["frontend"], "enrolled_face.jpg"), face_crop)
                else:
                    cv2.imwrite(os.path.join(PATHS["frontend"], "enrolled_face.jpg"), bgr)
                print("[OK] Identity enrollment complete")
        except Exception as e:
            print(f"[WARN] Enrollment error: {e}")

    def _verify_identity_async(self, rgb, now):
        """Run identity verification in a background thread."""
        if not FACE_REC_AVAILABLE or self.enrolled_embedding is None:
            self._verify_in_progress = False
            return

        def _run():
            try:
                encodings = face_recognition.face_encodings(rgb)
                if encodings:
                    match = face_recognition.compare_faces(
                        [self.enrolled_embedding], encodings[0], tolerance=0.6
                    )
                    with self._lock:
                        if match[0]:
                            self._identity_consec_mismatches = 0
                            self._identity_mismatch_state = False
                        else:
                            self._identity_consec_mismatches += 1
                            if self._identity_consec_mismatches >= 3:
                                self._identity_mismatch_state = True
                else:
                    # No face visible — don't flag mismatch
                    with self._lock:
                        self._identity_consec_mismatches = 0
                        self._identity_mismatch_state = False
            except Exception as e:
                print(f"[WARN] Identity verification error: {e}")
            finally:
                with self._lock:
                    self._verify_in_progress = False

        threading.Thread(target=_run, daemon=True).start()

    # ── Reset / Re-enrollment ─────────────────────────────────────────────────
    def reset_session(self):
        """Reset enrollment and all counters for a fresh session."""
        if DB_AVAILABLE:
            try:
                save_session_summary_db(self.generate_session_summary())
                reset_session_db()
            except Exception:
                pass

        # Clear enrolled face photo
        try:
            face_img_path = os.path.join(PATHS["frontend"], "enrolled_face.jpg")
            if os.path.exists(face_img_path):
                os.remove(face_img_path)
        except Exception:
            pass

        self.enrolled_embedding  = None
        self.enrollment_complete = False
        self._enroll_frame_count = 0
        self._last_verify_time   = 0.0

        self.phone_violations         = 0
        self.multiple_face_violations = 0
        self.no_face_violations       = 0
        self.head_violation_count     = 0
        self._head_consec             = 0
        self.gaze_violations          = 0
        self.mouth_violations         = 0
        self.identity_violations      = 0
        self.hand_violations          = 0

        self._phone_active             = False
        self._multiple_faces_active    = False
        self._no_face_active           = False
        self._head_turned_active       = False
        self._gaze_off_active          = False
        self._mouth_open_active        = False
        self._identity_mismatch_active = False
        self._hand_active              = False

        self._first_seen_phone          = 0.0
        self._first_seen_multiple_faces = 0.0
        self._first_seen_no_face        = 0.0
        self._first_seen_head           = 0.0
        self._first_seen_gaze           = 0.0
        self._first_seen_mouth          = 0.0
        self._first_seen_hand           = 0.0

        self._identity_mismatch_state    = False
        self._identity_consec_mismatches = 0

        # Reset hysteresis and cooldown state
        self._last_event_times = {}
        self._clear_timers     = {}
        self._head_buf.clear()
        self._gaze_buf.clear()
        self._mouth_buf.clear()

        self.suspicion_score   = 0.0
        self.score_history     = []
        self.behaviour_state   = "NORMAL"
        self.risk_level        = "LOW"
        self.session_start     = time.time()
        self.high_risk_periods = 0
        self.max_score         = 0.0
        self._score_sum        = 0.0
        self._score_n          = 0

        self.event_log     = []
        self.timeline_data = []

        self.phone_detected    = False
        self.hand_near_face    = False
        self.identity_mismatch = False
        self.head_direction    = "forward"
        self.gaze_direction    = "on-screen"
        self.mouth_state       = "closed"
        self.face_count        = 0

        self.simulation_overrides  = {}
        self._mp_timestamp_ms      = 0

        print("[OK] Session reset — re-enrollment started")

    # ── Unified edge-triggered violations logging & counting ──────────────────
    def _process_violations(self, now: float):
        """
        Robust violation detection with:
        - Debounce: condition must persist for DEBOUNCE_DURATIONS[key] seconds
        - Hysteresis: condition must be CLEAR for CLEAR_DURATIONS[key] seconds
          before the active flag resets (prevents flicker re-triggering)
        - Cooldown: same event type can only be logged once per EVENT_COOLDOWNS[key]
        """
        # Skip violations during first 5s warm-up
        if now - self.session_start < 5.0:
            return

        def _check(condition: bool, first_seen_attr: str, active_attr: str,
                   clear_key: str, debounce_key: str,
                   count_attr: str, event_type: str):
            """Generic violation checker with debounce + hysteresis + cooldown."""
            first_seen  = getattr(self, first_seen_attr)
            is_active   = getattr(self, active_attr)

            if condition:
                # Condition is active — reset clear timer
                self._clear_timers.pop(clear_key, None)

                if first_seen == 0.0:
                    setattr(self, first_seen_attr, now)
                elif now - first_seen >= self.DEBOUNCE_DURATIONS[debounce_key]:
                    if not is_active:
                        setattr(self, active_attr, True)
                        # Only log/count if cooldown has elapsed
                        last_t = self._last_event_times.get(event_type, 0.0)
                        cooldown = self.EVENT_COOLDOWNS.get(event_type, 0.0)
                        if now - last_t >= cooldown:
                            count = getattr(self, count_attr)
                            setattr(self, count_attr, count + 1)
                            self._log_event(event_type)
            else:
                # Condition is clear — apply hysteresis before resetting
                clear_dur = self.CLEAR_DURATIONS.get(clear_key, 0.0)
                if is_active or first_seen > 0.0:
                    if clear_key not in self._clear_timers:
                        self._clear_timers[clear_key] = now
                    elif now - self._clear_timers[clear_key] >= clear_dur:
                        setattr(self, first_seen_attr, 0.0)
                        setattr(self, active_attr, False)
                        self._clear_timers.pop(clear_key, None)
                else:
                    self._clear_timers.pop(clear_key, None)

        # 1. Phone detection
        _check(
            condition    = self.phone_detected,
            first_seen_attr = "_first_seen_phone",
            active_attr  = "_phone_active",
            clear_key    = "phone",
            debounce_key = "phone",
            count_attr   = "phone_violations",
            event_type   = "phone_detected",
        )

        # 2. Multiple faces
        _check(
            condition    = self.face_count > 1,
            first_seen_attr = "_first_seen_multiple_faces",
            active_attr  = "_multiple_faces_active",
            clear_key    = "multiple_faces",
            debounce_key = "multiple_faces",
            count_attr   = "multiple_face_violations",
            event_type   = "multiple_faces",
        )

        # 3. No face (only after enrollment)
        if self.enrollment_complete:
            if self.face_count == 0:
                if self._first_seen_no_face == 0.0:
                    self._first_seen_no_face = now
                elif now - self._first_seen_no_face >= self.DEBOUNCE_DURATIONS["no_face"]:
                    if not self._no_face_active:
                        self._no_face_active = True
                        self.no_face_violations += 1
            else:
                self._first_seen_no_face = 0.0
                self._no_face_active = False

        # 4. Head turned
        _check(
            condition    = self.head_direction != "forward",
            first_seen_attr = "_first_seen_head",
            active_attr  = "_head_turned_active",
            clear_key    = "head",
            debounce_key = "head",
            count_attr   = "head_violation_count",
            event_type   = "head_turned",
        )

        # 5. Gaze off-screen
        _check(
            condition    = "off" in self.gaze_direction.lower(),
            first_seen_attr = "_first_seen_gaze",
            active_attr  = "_gaze_off_active",
            clear_key    = "gaze",
            debounce_key = "gaze",
            count_attr   = "gaze_violations",
            event_type   = "gaze_off_screen",
        )

        # 6. Mouth open / speaking
        _check(
            condition    = self.mouth_state in ("open", "speaking"),
            first_seen_attr = "_first_seen_mouth",
            active_attr  = "_mouth_open_active",
            clear_key    = "mouth",
            debounce_key = "mouth",
            count_attr   = "mouth_violations",
            event_type   = "mouth_open",
        )

        # 7. Identity mismatch (already debounced in verify thread)
        if self.identity_mismatch:
            if not self._identity_mismatch_active:
                self._identity_mismatch_active = True
                last_t = self._last_event_times.get("identity_mismatch", 0.0)
                if now - last_t >= self.EVENT_COOLDOWNS["identity_mismatch"]:
                    self.identity_violations += 1
                    self._log_event("identity_mismatch")
        else:
            self._identity_mismatch_active = False

        # 8. Hand near face
        _check(
            condition    = self.hand_near_face,
            first_seen_attr = "_first_seen_hand",
            active_attr  = "_hand_active",
            clear_key    = "hand",
            debounce_key = "hand",
            count_attr   = "hand_violations",
            event_type   = "hand_near_face",
        )

    # ── Event logging ─────────────────────────────────────────────────────────
    def _log_event(self, event_type: str):
        ts  = time.time()
        rec = {
            "event_type": event_type,
            "type":       event_type,
            "time":       datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
            "timestamp":  round(ts - self.session_start, 2),
            "wall_time":  datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
            "score":      round(self.suspicion_score, 2),
        }
        self._last_event_times[event_type] = ts   # track for per-type cooldown
        self.event_log.append(rec)
        self.timeline_data.append(rec)
        if len(self.timeline_data) > 500:
            self.timeline_data.pop(0)
        print(f"[EVENT] {event_type} | score={rec['score']}")

        if DB_AVAILABLE:
            try:
                log_event_db(event_type, rec["timestamp"], rec["score"])
            except Exception:
                pass

    # ── Evidence snapshot ─────────────────────────────────────────────────────
    def _save_snapshot(self, frame, now):
        ts = datetime.datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
        fname = f"evidence_{ts}_score{int(self.suspicion_score)}.jpg"
        path = os.path.join(PATHS["snapshots"], fname)
        cv2.imwrite(path, frame)
        print(f"[SNAPSHOT] Saved: {fname}")

    # ── Frame annotation ──────────────────────────────────────────────────────
    def _annotate(self, frame):
        h, w = frame.shape[:2]

        # ── Draw MediaPipe face mesh ──────────────────────────────────────────
        if self.mp_face_result and self.mp_face_result.face_landmarks:
            for face_landmarks in self.mp_face_result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 200), -1)

        # ── Status HUD overlay ────────────────────────────────────────────────
        # Risk level color
        risk_colors = {"LOW": (0, 220, 80), "MEDIUM": (0, 165, 255), "HIGH": (0, 40, 255)}
        risk_color = risk_colors.get(self.risk_level, (200, 200, 200))

        # Semi-transparent black bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Score and risk
        score_txt = f"SCORE: {self.suspicion_score:.1f}  RISK: {self.risk_level}"
        cv2.putText(frame, score_txt, (8, 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, risk_color, 1, cv2.LINE_AA)

        # Phone indicator
        if self.phone_detected:
            cv2.putText(frame, "! PHONE", (w - 110, 24),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 40, 255), 1, cv2.LINE_AA)

        # Face count indicator (bottom left)
        face_txt = f"FACES:{self.face_count}"
        fc_color = (0, 40, 255) if self.face_count != 1 else (0, 220, 80)
        cv2.putText(frame, face_txt, (8, h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, fc_color, 1, cv2.LINE_AA)

        # Head direction (bottom center)
        head_color = (0, 220, 80) if self.head_direction == "forward" else (0, 165, 255)
        cv2.putText(frame, f"HEAD:{self.head_direction.upper()}", (w // 2 - 55, h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, head_color, 1, cv2.LINE_AA)

        # Gaze (bottom right)
        gaze_color = (0, 220, 80) if "on" in self.gaze_direction else (0, 165, 255)
        cv2.putText(frame, f"GAZE:{self.gaze_direction.upper()}", (w - 160, h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, gaze_color, 1, cv2.LINE_AA)

        # Enrollment status
        if not self.enrollment_complete:
            pct = min(100, int(self._enroll_frame_count / ML_CONFIG["warmup_frames"] * 100))
            cv2.putText(frame, f"ENROLLING... {pct}%", (8, 56),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        return frame

    # ── MJPEG frame generator ─────────────────────────────────────────────────
    def generate_frames(self):
        last_frame_count = -1
        while True:
            with self._lock:
                frame = self.latest_frame
                curr_count = self.frame_count
            if frame is None or curr_count == last_frame_count:
                time.sleep(0.03)
                continue
            last_frame_count = curr_count
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")

    # ── Top suspicion signals ─────────────────────────────────────────────────
    def get_top_suspicion_signals(self):
        signals = {
            "Phone Detected":      self.phone_violations * SCORE_WEIGHTS["phone_detected"],
            "Multiple Faces":      self.multiple_face_violations * SCORE_WEIGHTS["multiple_faces"],
            "Head Turned":         self.head_violation_count * SCORE_WEIGHTS["head_turned"],
            "Gaze Off-Screen":     self.gaze_violations * SCORE_WEIGHTS["gaze_off_screen"],
            "Mouth Open/Speaking": self.mouth_violations * SCORE_WEIGHTS["mouth_open"],
            "Identity Mismatch":   self.identity_violations * SCORE_WEIGHTS["identity_mismatch"],
        }
        return sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]

    # ── Session summary ───────────────────────────────────────────────────────
    def generate_session_summary(self):
        duration = round(time.time() - self.session_start, 1)
        avg_score = round(self._score_sum / max(self._score_n, 1), 2)
        return {
            "duration_seconds":          duration,
            "avg_suspicion_score":       avg_score,
            "max_suspicion_score":       round(self.max_score, 2),
            "high_risk_periods":         self.high_risk_periods,
            "total_events":              len(self.event_log),
            "phone_violations":          self.phone_violations,
            "multiple_face_violations":  self.multiple_face_violations,
            "head_violations":           self.head_violation_count,
            "gaze_violations":           self.gaze_violations,
            "mouth_violations":          self.mouth_violations,
            "identity_violations":       self.identity_violations,
        }

    # ── CSV export ────────────────────────────────────────────────────────────
    def export_csv(self) -> str:
        summary = self.generate_session_summary()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["EXAMSHIELD AI — Session Report"])
        writer.writerow(["Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        for k, v in summary.items():
            writer.writerow([k, v])
        writer.writerow([])
        writer.writerow(["#", "Event Type", "Session Time (s)", "Wall Time", "Score"])
        for i, ev in enumerate(self.event_log, 1):
            writer.writerow([i, ev.get("event_type", ev.get("type", "")),
                             ev["timestamp"], ev["wall_time"], ev["score"]])
        return output.getvalue()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def stop(self):
        self._running = False
        self.cap.release()
        if self._mp_face_lm:
            try:
                self._mp_face_lm.close()
            except Exception:
                pass
        if self._mp_hand_lm:
            try:
                self._mp_hand_lm.close()
            except Exception:
                pass
        if DB_AVAILABLE:
            try:
                save_session_summary_db(self.generate_session_summary())
            except Exception:
                pass
        print("[OK] Camera stopped, session saved")

