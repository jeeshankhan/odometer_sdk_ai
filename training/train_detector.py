"""
Train the YOLOv11 dashboard/odometer detector.

Expects data in YOLO format (see dataset/generate_synthetic_data.py for the
exact label format, or export from CVAT/Label Studio in YOLO format):

    dataset/
        images/
            img_0001.png
            ...
        labels/
            img_0001.txt   # class cx cy w h, normalized 0-1
            ...

Requires `pip install ultralytics`.

Usage:
    python training/train_detector.py \
        --data dataset/real/data.yaml \
        --epochs 100 \
        --imgsz 640 \
        --out training/runs/detector
"""
import argparse


def write_data_yaml(images_dir: str, out_path: str, class_names=("dashboard", "odometer")):
    """Convenience helper: generates the data.yaml ultralytics expects,
    assuming images_dir/train and images_dir/val subfolders exist."""
    content = f"""\
train: {images_dir}/train
val: {images_dir}/val
nc: {len(class_names)}
names: {list(class_names)}
"""
    with open(out_path, "w") as f:
        f.write(content)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolo11n.pt", help="Base checkpoint: n/s/m/l/x sizes available")
    parser.add_argument("--out", default="training/runs/detector")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Run: pip install ultralytics"
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.out,
        name="detector",
        patience=20,          # early stopping
        augment=True,
        mosaic=1.0,
        mixup=0.1,
    )

    print(f"\nTraining complete. Best weights: {args.out}/detector/weights/best.pt")
    print("Next: python training/export_onnx.py --weights "
          f"{args.out}/detector/weights/best.pt --out models/detector.onnx")


if __name__ == "__main__":
    main()
