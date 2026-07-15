"""
Dashboard + odometer region detector.

Stable interface: `Detector.detect(image) -> DetectionResult`.
If a trained ONNX model is present at the given path, real inference runs.
Otherwise falls back to a mock that returns the center ~40% of the image as
the "odometer" box, so the rest of the pipeline is testable before Phase 2
(YOLO training) is complete.

Swap in the real model by training with training/train_detector.py,
exporting via training/export_onnx.py, and pointing `model_path` at the
resulting .onnx file. No other code changes needed.
"""
import os
from dataclasses import dataclass

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass
class DetectionResult:
    dashboard_box: tuple | None   # (x1, y1, x2, y2) in pixel coords, or None
    odometer_box: tuple           # (x1, y1, x2, y2)
    confidence: float
    used_mock: bool


class Detector:
    def __init__(self, model_path: str = "models/detector.onnx", conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.session = None
        if ort is not None and os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path)

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self.session is not None:
            return self._detect_real(image)
        return self._detect_mock(image)

    def _detect_mock(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        x1, y1 = int(w * 0.30), int(h * 0.40)
        x2, y2 = int(w * 0.70), int(h * 0.60)
        return DetectionResult(
            dashboard_box=(int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)),
            odometer_box=(x1, y1, x2, y2),
            confidence=0.50,   # deliberately mid so downstream confidence logic is exercised
            used_mock=True,
        )

    def _detect_real(self, image: np.ndarray) -> DetectionResult:
        """
        Placeholder for real YOLOv11 ONNX inference: preprocess (letterbox,
        normalize) -> session.run -> NMS -> pick highest-confidence
        dashboard/odometer boxes. Fill this in once training/export_onnx.py
        has produced models/detector.onnx and you've confirmed its exact
        input/output tensor shapes (they depend on your export settings).
        """
        raise NotImplementedError(
            "Real detector inference not wired up yet. "
            "Implement pre/post-processing to match your exported YOLOv11 ONNX model, "
            "or delete/rename models/detector.onnx to keep using the mock."
        )
