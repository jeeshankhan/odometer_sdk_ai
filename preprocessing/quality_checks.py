"""
Image quality gate: run before detection to reject bad frames early
(fast, cheap, no model needed) and give the user actionable feedback
("too blurry", "too dark", "glare detected", etc).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityReport:
    passed: bool
    blur_score: float
    brightness: float
    glare_ratio: float
    issues: list = field(default_factory=list)


def check_blur(gray: np.ndarray, threshold: float = 100.0) -> tuple[float, bool]:
    """Variance of Laplacian. Lower = blurrier. Threshold is empirical;
    tune against your own real-world dataset once you have one."""
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, score >= threshold


def check_brightness(gray: np.ndarray, low: float = 40.0, high: float = 220.0):
    mean_val = float(gray.mean())
    return mean_val, low <= mean_val <= high


def check_glare(gray: np.ndarray, blown_out_threshold: int = 245, max_ratio: float = 0.03):
    """Fraction of pixels that are near-white (blown out highlight)."""
    blown = np.sum(gray >= blown_out_threshold)
    ratio = blown / gray.size
    return ratio, ratio <= max_ratio


def check_perspective_skew(gray: np.ndarray, max_angle_deg: float = 25.0):
    """
    Rough skew estimate using the dominant Hough line angle.
    Returns (angle, ok). If no strong lines are found, assumes ok=True
    (detector/perspective-correction stage will handle fine-grained cases).
    """
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=gray.shape[1] * 0.3, maxLineGap=20)
    if lines is None:
        return 0.0, True

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = float(np.median(angles))
    return median_angle, abs(median_angle) <= max_angle_deg


def run_quality_checks(image: np.ndarray) -> QualityReport:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    blur_score, blur_ok = check_blur(gray)
    brightness, brightness_ok = check_brightness(gray)
    glare_ratio, glare_ok = check_glare(gray)
    skew_angle, skew_ok = check_perspective_skew(gray)

    issues = []
    if not blur_ok:
        issues.append("image_too_blurry")
    if not brightness_ok:
        issues.append("bad_lighting")
    if not glare_ok:
        issues.append("glare_detected")
    if not skew_ok:
        issues.append("excessive_perspective_skew")

    return QualityReport(
        passed=len(issues) == 0,
        blur_score=blur_score,
        brightness=brightness,
        glare_ratio=glare_ratio,
        issues=issues,
    )
