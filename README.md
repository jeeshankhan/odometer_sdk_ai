# OdometerSDK

A from-scratch, commercial-grade odometer OCR SDK: dashboard/odometer detection (YOLO) →
perspective correction & OpenCV enhancement → sequence recognition (PARSeq/CRNN) →
confidence scoring → business validation → clean Android/Flutter API.

This repo is a **working scaffold**. Every module that doesn't depend on a trained
model runs today against synthetic data. Modules that need a trained model
(`detection/`, `recognition/`) fall back to a mock so the pipeline is runnable
end-to-end from day one, and swap to real ONNX models with zero API changes.

## Directory structure

```
OdometerSDK/
├── dataset/            synthetic data generator (stand-in until you have real images)
├── training/           YOLOv11 detector + recognizer training scripts, ONNX export
├── preprocessing/       fully functional OpenCV pipeline (quality checks, enhancement, perspective correction)
├── detection/          ONNX Runtime wrapper for dashboard/odometer detection (mock fallback)
├── recognition/         ONNX Runtime wrapper for digit-sequence recognition (mock fallback)
├── validation/          confidence engine + business rules (fully functional)
├── api/                 Python reference SDK tying everything together
├── android/             Standalone Kotlin SDK reference (ONNX Runtime Android) - for pure-native
│                        Android apps that aren't using Flutter. NOT what Flutter builds.
├── flutter/              Flutter plugin - this is what `flutter pub get` actually compiles.
│                        flutter/android/ has its OWN copy of the Kotlin source files
│                        (Detector.kt, Recognizer.kt, etc.) because a Flutter plugin module
│                        can't depend on a sibling Gradle project outside its own folder.
│                        ⚠️ If you edit SDK logic, update BOTH android/ and flutter/android/,
│                        or better: delete android/ and treat flutter/android/ as the only copy.
├── sample_app/           end-to-end demo script using synthetic images
└── documentation/        phase-by-phase build notes
```

## Quickstart (works right now, no trained models needed)

```bash
pip install -r requirements.txt
python dataset/generate_synthetic_data.py --count 20 --out dataset/synthetic
python sample_app/run_demo.py --image dataset/synthetic/img_0001.png
```

This runs quality checks → mock detection → preprocessing → mock recognition →
confidence scoring → validation, and prints a full JSON SDK response — so you
can build/test the Android & Flutter layers before a single model is trained.

## Roadmap (see documentation/ROADMAP.md for detail)

1. Collect & annotate 50k–100k real odometer images
2. Train YOLOv11 detector (`training/train_detector.py`)
3. Train PARSeq/CRNN recognizer (`training/train_recognizer.py`)
4. Export both to ONNX (`training/export_onnx.py`)
5. Drop `.onnx` files into `models/` — `detection/` and `recognition/` auto-switch
   from mock mode to real inference
6. Wire the Android SDK (`android/`) and Flutter plugin (`flutter/`) to the real SDK
7. Add continuous retraining from field data (see documentation/CONTINUOUS_LEARNING.md)

## Why this structure

Preprocessing and validation are the two components that work with zero ML and
still carry a large share of real-world accuracy, so they're built out fully
and testable today. Everything ML-dependent is isolated behind a stable
interface (`detection.Detector`, `recognition.Recognizer`) so training progress
never forces API rewrites downstream.
