"""
Enhancement pipeline applied after odometer-region detection and before
sequence recognition:

    perspective correction -> CLAHE -> gamma correction -> denoise -> sharpen -> resize

Each step is a standalone function so you can A/B test which combination
actually helps your recognizer (this often matters more than swapping
recognizer architectures).
"""
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_correct(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    corners: 4x2 array of (x, y) points for the odometer region, in any order.
    If you only have an axis-aligned bounding box (no true corner detection
    yet), just pass its 4 corners - this becomes a no-op crop+warp.
    """
    rect = order_points(np.array(corners, dtype="float32"))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_gamma(gray: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def resize_for_recognizer(gray: np.ndarray, target_height: int = 64) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = target_height / h
    return cv2.resize(gray, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)


def enhance(image: np.ndarray, corners: np.ndarray | None = None) -> np.ndarray:
    """Full pipeline: perspective correct (if corners given) -> grayscale ->
    CLAHE -> gamma -> denoise -> sharpen -> resize. Returns a single-channel
    image ready for the recognizer."""
    working = perspective_correct(image, corners) if corners is not None else image
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY) if working.ndim == 3 else working
    gray = apply_clahe(gray)
    gray = apply_gamma(gray)
    gray = denoise(gray)
    gray = sharpen(gray)
    gray = resize_for_recognizer(gray)
    return gray
