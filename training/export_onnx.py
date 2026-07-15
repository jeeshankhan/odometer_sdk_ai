"""
Export trained detector or recognizer weights to ONNX for ONNX Runtime
inference on Android/iOS/Python.

Usage:
    python training/export_onnx.py --weights runs/detector/weights/best.pt \
        --out models/detector.onnx --type detector

    python training/export_onnx.py --weights runs/recognizer/recognizer_best.pt \
        --out models/recognizer.onnx --type recognizer
"""
import argparse
import os


def export_detector(weights_path: str, out_path: str, imgsz: int = 640):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True)
    exported = weights_path.rsplit(".", 1)[0] + ".onnx"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.replace(exported, out_path)
    print(f"Detector exported to {out_path}")


def export_recognizer(weights_path: str, out_path: str, input_height: int = 64, input_width: int = 256):
    import torch
    from train_recognizer import build_model

    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, input_height, input_width)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model, dummy_input, out_path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {3: "width"}, "logits": {1: "seq_len"}},
        opset_version=17,
    )
    print(f"Recognizer exported to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--type", choices=["detector", "recognizer"], required=True)
    args = parser.parse_args()

    if args.type == "detector":
        export_detector(args.weights, args.out)
    else:
        export_recognizer(args.weights, args.out)


if __name__ == "__main__":
    main()
