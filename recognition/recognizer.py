"""
Digit-sequence recognizer (PARSeq-style: predicts the whole sequence at once,
not digit-by-digit).

Stable interface: `Recognizer.recognize(cropped_gray) -> RecognitionResult`.
Falls back to a mock that returns a random plausible-looking reading with
per-digit confidences, so validation/confidence-engine code can be built
and tested before Phase 4 (recognizer training) is complete.
"""
import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass
class RecognitionResult:
    reading: str
    digit_confidences: list[float]
    overall_confidence: float
    used_mock: bool


class Recognizer:
    def __init__(self, model_path: str = "models/recognizer.onnx"):
        self.session = None
        if ort is not None and os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path)

    def recognize(self, cropped_gray: np.ndarray) -> RecognitionResult:
        if self.session is not None:
            return self._recognize_real(cropped_gray)
        return self._recognize_mock(cropped_gray)

    def _recognize_mock(self, cropped_gray: np.ndarray) -> RecognitionResult:
        length = random.choice([5, 6, 6, 6, 7])
        digits = [random.randint(0, 9) for _ in range(length)]
        # simulate one occasionally-uncertain digit, like a real model would produce
        confidences = [round(random.uniform(0.96, 0.999), 4) for _ in digits]
        if random.random() < 0.3:
            idx = random.randrange(length)
            confidences[idx] = round(random.uniform(0.55, 0.85), 4)

        return RecognitionResult(
            reading="".join(str(d) for d in digits),
            digit_confidences=confidences,
            overall_confidence=round(min(confidences) * 100, 2),
            used_mock=True,
        )

    def _recognize_real(self, cropped_gray: np.ndarray) -> RecognitionResult:
        """
        Placeholder for real PARSeq/CRNN ONNX inference: normalize input to
        the model's expected shape -> session.run -> decode sequence output
        (CTC-decode or attention-decode depending on architecture) -> per-step
        softmax max as digit confidence. Fill in once
        training/train_recognizer.py + export_onnx.py produce
        models/recognizer.onnx.
        """
        raise NotImplementedError(
            "Real recognizer inference not wired up yet. "
            "Implement pre/post-processing to match your exported recognizer ONNX model, "
            "or delete/rename models/recognizer.onnx to keep using the mock."
        )
