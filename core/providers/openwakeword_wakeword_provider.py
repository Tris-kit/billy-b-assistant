from __future__ import annotations

from pathlib import Path

import numpy as np

from ..logger import logger
from ..wakeword_provider import WakeWordBackend


def _resolve_path(root_dir: str, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(root_dir) / path
    return path


def _resolve_model_path(root_dir: str, value: str) -> Path:
    if not value:
        return Path("")

    raw = Path(value)
    if raw.is_absolute() or raw.parent != Path("."):
        return _resolve_path(root_dir, value)

    candidate = Path(root_dir) / "wakewords" / raw.name
    if candidate.exists():
        return candidate
    return Path(root_dir) / "wakewords" / raw.name


class OpenWakeWordBackend(WakeWordBackend):
    def __init__(
        self,
        *,
        root_dir: str,
        openwakeword_model_path: str,
        openwakeword_threshold: float,
        openwakeword_inference_framework: str,
    ):
        self.root_dir = root_dir
        self.model_path_raw = (openwakeword_model_path or "").strip()
        self.threshold = max(0.0, min(1.0, float(openwakeword_threshold)))
        self.inference_framework = (
            (openwakeword_inference_framework or "onnx").strip().lower()
        )
        if self.inference_framework not in {"onnx", "tflite"}:
            logger.warning(
                "Invalid WAKE_WORD_OPENWAKEWORD_INFERENCE_FRAMEWORK, using 'onnx'.",
                "⚠️",
            )
            self.inference_framework = "onnx"

        self._model = None
        self._keyword_label = ""
        self._model_key = ""

    @property
    def backend_name(self) -> str:
        return "openwakeword"

    @property
    def keyword_label(self) -> str:
        return self._keyword_label

    @property
    def sample_rate(self) -> int:
        # openWakeWord expects 16kHz PCM input.
        return 16000

    @property
    def frame_length(self) -> int:
        # openWakeWord models consume 80ms chunks at 16kHz.
        return 1280

    def initialize(self) -> bool:
        if not self.model_path_raw:
            logger.warning(
                "WAKE_WORD_OPENWAKEWORD_MODEL_PATH is empty; wake-word listener disabled.",
                "⚠️",
            )
            return False

        model_path = _resolve_model_path(self.root_dir, self.model_path_raw)
        if not model_path.exists():
            logger.warning(f"openWakeWord model file not found: {model_path}", "⚠️")
            return False

        try:
            from openwakeword.model import Model
        except ImportError:
            logger.warning(
                "openwakeword is not installed. Install requirements to enable wake-word.",
                "⚠️",
            )
            return False

        try:
            self._model = Model(
                wakeword_models=[str(model_path)],
                inference_framework=self.inference_framework,
            )
        except TypeError:
            # Backward compatibility for older openwakeword releases.
            self._model = Model(wakeword_models=[str(model_path)])
        except Exception as e:
            logger.warning(f"Failed to initialize openWakeWord model: {e}", "⚠️")
            return False

        self._keyword_label = model_path.stem
        self._model_key = model_path.stem.lower()
        return True

    def process(self, frame: np.ndarray) -> bool:
        if self._model is None:
            return False

        try:
            prediction = self._model.predict(frame)
        except Exception as e:
            logger.warning(f"openWakeWord inference failed: {e}", "⚠️")
            return False

        if not isinstance(prediction, dict):
            return False

        score = None
        for key, value in prediction.items():
            key_norm = str(key).strip().lower()
            if key_norm == self._model_key:
                score = value
                break

        if score is None:
            numeric_scores = [
                float(value)
                for value in prediction.values()
                if isinstance(value, (int, float))
            ]
            if not numeric_scores:
                return False
            score = max(numeric_scores)

        return float(score) >= self.threshold

    def close(self):
        self._model = None
