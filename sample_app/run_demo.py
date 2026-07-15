"""
End-to-end demo: runs a synthetic image through the full SDK pipeline
and prints the JSON response, same shape the Android/Flutter layers expose.

Usage:
    python dataset/generate_synthetic_data.py --count 5 --out dataset/synthetic
    python sample_app/run_demo.py --image dataset/synthetic/images/img_0001.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.sdk import OdometerSDK, read_image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an odometer image")
    parser.add_argument("--previous_reading", type=int, default=None)
    args = parser.parse_args()

    sdk = OdometerSDK()
    response = read_image_path(sdk, args.image, previous_reading=args.previous_reading)
    print(json.dumps(response, indent=2))

    if response["used_mock_models"]:
        print("\n[note] detector/recognizer are running in MOCK mode "
              "(no trained ONNX models found at models/*.onnx).")


if __name__ == "__main__":
    main()
