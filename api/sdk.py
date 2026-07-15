"""
Reference implementation of the full pipeline, mirroring the eventual
Android/Flutter API shape (`sdk.read(image) -> SDKResponse`). This is what
the Android Kotlin SDK and Flutter plugin are wrappers around conceptually
(Android will use ONNX Runtime Android directly rather than calling Python).
"""
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from detection.detector import Detector
from preprocessing.pipeline import enhance
from preprocessing.quality_checks import run_quality_checks
from recognition.recognizer import Recognizer
from validation.business_rules import validate_reading
from validation.confidence_engine import ConfidenceAction, evaluate_confidence


@dataclass
class SDKResponse:
    reading: str | None
    confidence: float
    is_valid: bool
    action: str
    digit_confidences: list
    issues: list
    validation_message: str | None
    used_mock_models: bool


class OdometerSDK:
    def __init__(self, detector_model_path="models/detector.onnx",
                 recognizer_model_path="models/recognizer.onnx"):
        self.detector = Detector(detector_model_path)
        self.recognizer = Recognizer(recognizer_model_path)

    def read(self, image: np.ndarray, previous_reading: int | None = None) -> SDKResponse:
        # 1. quality gate
        quality = run_quality_checks(image)
        if not quality.passed:
            return SDKResponse(
                reading=None, confidence=0.0, is_valid=False, action="ask_retake",
                digit_confidences=[], issues=quality.issues,
                validation_message=None, used_mock_models=False,
            )

        # 2. detect dashboard/odometer region
        detection = self.detector.detect(image)
        x1, y1, x2, y2 = detection.odometer_box
        cropped = image[y1:y2, x1:x2]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        # 3. enhance (perspective correction + CLAHE + gamma + denoise + sharpen + resize)
        enhanced = enhance(image, corners=corners)

        # 4. recognize full sequence
        recognition = self.recognizer.recognize(enhanced)

        # 5. confidence decision
        decision = evaluate_confidence(recognition.digit_confidences)

        # 6. business validation (only meaningful once we trust the reading)
        validation_message = None
        is_valid = decision.action == ConfidenceAction.ACCEPT
        if is_valid:
            try:
                reading_int = int(recognition.reading)
                result = validate_reading(reading_int, previous_reading)
                is_valid = result.is_valid
                validation_message = result.message
            except ValueError:
                is_valid = False
                validation_message = "reading was not a valid integer"

        return SDKResponse(
            reading=recognition.reading,
            confidence=recognition.overall_confidence,
            is_valid=is_valid,
            action=decision.action.value,
            digit_confidences=recognition.digit_confidences,
            issues=quality.issues,
            validation_message=validation_message,
            used_mock_models=detection.used_mock or recognition.used_mock,
        )


def read_image_path(sdk: OdometerSDK, image_path: str, previous_reading=None) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    response = sdk.read(image, previous_reading=previous_reading)
    return json.loads(json.dumps(asdict(response)))  # dataclass -> plain dict
