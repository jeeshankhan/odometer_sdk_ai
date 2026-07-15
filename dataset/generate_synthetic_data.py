"""
Generate synthetic odometer dashboard images for testing the pipeline
before you have a real labeled dataset.

Each image contains:
  - a fake "dashboard" background rectangle
  - a fake "odometer" digit strip inside it (random 5-7 digit number)
  - optional degradations: blur, low brightness, glare patch, slight rotation

Also writes a YOLO-format label file (class 0 = dashboard, class 1 = odometer)
and a ground_truth.json mapping filename -> digit string, so the same
synthetic set can be used to sanity-check the detector, the recognizer,
and the validation rules.

Usage:
    python generate_synthetic_data.py --count 20 --out dataset/synthetic
"""
import argparse
import json
import os
import random

import cv2
import numpy as np


def make_dashboard_image(width=800, height=600, digits_len=None, degrade=True):
    digits_len = digits_len or random.choice([5, 6, 6, 6, 7])
    digit_string = "".join(str(random.randint(0, 9)) for _ in range(digits_len))

    img = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    # subtle texture so flat synthetic backgrounds don't read as "blurry"
    # under variance-of-Laplacian the way a real (never perfectly flat) photo wouldn't
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # fake dashboard panel
    dash_x1, dash_y1 = int(width * 0.15), int(height * 0.2)
    dash_x2, dash_y2 = int(width * 0.85), int(height * 0.8)
    cv2.rectangle(img, (dash_x1, dash_y1), (dash_x2, dash_y2), (60, 60, 65), -1)

    # fake odometer strip inside the dashboard
    odo_x1 = dash_x1 + int((dash_x2 - dash_x1) * 0.25)
    odo_y1 = dash_y1 + int((dash_y2 - dash_y1) * 0.55)
    odo_x2 = dash_x1 + int((dash_x2 - dash_x1) * 0.75)
    odo_y2 = odo_y1 + 60
    cv2.rectangle(img, (odo_x1, odo_y1), (odo_x2, odo_y2), (10, 10, 10), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness = 3
    text_size = cv2.getTextSize(digit_string, font, font_scale, thickness)[0]
    text_x = odo_x1 + ((odo_x2 - odo_x1) - text_size[0]) // 2
    text_y = odo_y1 + ((odo_y2 - odo_y1) + text_size[1]) // 2
    cv2.putText(img, digit_string, (text_x, text_y), font, font_scale,
                (220, 220, 220), thickness, cv2.LINE_AA)

    boxes = {
        "dashboard": (dash_x1, dash_y1, dash_x2, dash_y2),
        "odometer": (odo_x1, odo_y1, odo_x2, odo_y2),
    }

    if degrade:
        img, boxes = _apply_random_degradation(img, boxes)

    return img, digit_string, boxes


def _apply_random_degradation(img, boxes):
    h, w = img.shape[:2]

    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.3:
        factor = random.uniform(0.4, 0.7)
        img = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

    if random.random() < 0.3:
        glare = np.zeros_like(img)
        gx, gy = random.randint(0, w), random.randint(0, h)
        cv2.circle(glare, (gx, gy), random.randint(40, 100), (255, 255, 255), -1)
        glare = cv2.GaussianBlur(glare, (51, 51), 0)
        img = cv2.addWeighted(img, 1.0, glare, 0.5, 0)

    if random.random() < 0.3:
        angle = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(30, 30, 30))
        # NOTE: boxes are not rotated here for simplicity; for real training
        # use albumentations with bbox-aware transforms instead.

    return img, boxes


def to_yolo_label(boxes, img_w, img_h):
    """class 0 = dashboard, class 1 = odometer"""
    lines = []
    for cls_id, name in enumerate(["dashboard", "odometer"]):
        x1, y1, x2, y2 = boxes[name]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=str, default="dataset/synthetic")
    args = parser.parse_args()

    img_dir = os.path.join(args.out, "images")
    label_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    ground_truth = {}
    for i in range(1, args.count + 1):
        img, digits, boxes = make_dashboard_image()
        fname = f"img_{i:04d}.png"
        cv2.imwrite(os.path.join(img_dir, fname), img)

        label_path = os.path.join(label_dir, fname.replace(".png", ".txt"))
        with open(label_path, "w") as f:
            f.write(to_yolo_label(boxes, img.shape[1], img.shape[0]))

        ground_truth[fname] = digits

    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {args.count} synthetic images in {args.out}")
    print(f"  images/       -> raw images")
    print(f"  labels/       -> YOLO-format detection labels")
    print(f"  ground_truth.json -> filename -> digit string (for recognizer testing)")


if __name__ == "__main__":
    main()
