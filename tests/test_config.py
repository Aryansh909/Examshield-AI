"""
Tests for config.py — ensures all required keys and types are present.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestPaths:
    REQUIRED_PATH_KEYS = [
        "yolo_phone", "yolo_face", "mp_face", "mp_hand",
        "head_model", "gaze_model", "mouth_model",
        "frontend", "snapshots", "database",
    ]

    def test_all_path_keys_present(self):
        for key in self.REQUIRED_PATH_KEYS:
            assert key in config.PATHS, f"Missing PATHS key: {key}"

    def test_paths_are_absolute(self):
        for key, path in config.PATHS.items():
            assert os.path.isabs(path), f"PATHS['{key}'] is not absolute: {path}"

    def test_frontend_dir_exists(self):
        assert os.path.isdir(config.PATHS["frontend"]), "frontend/ directory missing"


class TestMLConfig:
    def test_ml_config_keys(self):
        assert "input_size" in config.ML_CONFIG
        assert "interval" in config.ML_CONFIG
        assert "warmup_frames" in config.ML_CONFIG

    def test_ml_config_values(self):
        assert config.ML_CONFIG["input_size"] > 0
        assert 0 < config.ML_CONFIG["interval"] < 5.0
        assert config.ML_CONFIG["warmup_frames"] > 0


class TestScoreWeights:
    REQUIRED_WEIGHT_KEYS = [
        "phone_detected", "multiple_faces", "no_face",
        "head_turned", "gaze_off_screen", "mouth_open",
        "identity_mismatch", "hand_near_face",
    ]

    def test_all_weight_keys_present(self):
        for key in self.REQUIRED_WEIGHT_KEYS:
            assert key in config.SCORE_WEIGHTS, f"Missing SCORE_WEIGHTS key: {key}"

    def test_weights_are_positive(self):
        for key, val in config.SCORE_WEIGHTS.items():
            assert val > 0, f"SCORE_WEIGHTS['{key}'] must be positive"

    def test_weight_sum_in_range(self):
        # Total weight should be expressible in 0-100 range
        assert sum(config.SCORE_WEIGHTS.values()) <= 200


class TestFlaskConfig:
    def test_flask_port_valid(self):
        assert 1024 <= config.FLASK_PORT <= 65535

    def test_flask_debug_is_bool(self):
        assert isinstance(config.FLASK_DEBUG, bool)

    def test_camera_index_is_int(self):
        assert isinstance(config.CAMERA_INDEX, int)
        assert config.CAMERA_INDEX >= 0
