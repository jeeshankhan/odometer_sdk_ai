# Build Roadmap

Status legend: ✅ scaffolded & runnable today · 🔧 stubbed, needs your work · ⏳ not started

| Phase | What | Status | Where |
|---|---|---|---|
| 0 | Project structure, preprocessing, validation, mock pipeline | ✅ | whole repo |
| 1 | Real dataset: 50k–100k images, annotated | ⏳ | `dataset/` (currently only synthetic) |
| 2 | Train YOLOv11 dashboard/odometer detector | 🔧 | `training/train_detector.py` |
| 3 | Image enhancement pipeline | ✅ | `preprocessing/pipeline.py` |
| 4 | Train PARSeq/CRNN sequence recognizer | 🔧 | `training/train_recognizer.py` |
| 5 | Confidence scoring + retry logic | ✅ | `validation/confidence_engine.py` |
| 6 | Business validation rules | ✅ | `validation/business_rules.py` |
| 7 | Android SDK | 🔧 | `android/` (compiles conceptually; real inference methods raise `NotImplementedError` until models exist) |
| 8 | Flutter plugin | 🔧 | `flutter/` (Dart API + Android bridge done; iOS not started) |
| 9 | Continuous learning loop | ⏳ | see `CONTINUOUS_LEARNING.md` |

## Suggested order of work

1. **Get the mock pipeline running** (already works — see repo root README Quickstart).
   Use this to build and test the Android/Flutter layers against realistic
   JSON shapes before any model exists.
2. **Learn OpenCV preprocessing hands-on** by tuning `preprocessing/pipeline.py`
   and `preprocessing/quality_checks.py` against a handful of real phone
   photos of odometers (even 20-30 images will teach you a lot about your
   actual blur/glare/brightness thresholds).
3. **Collect + annotate a first small real dataset** (500-2,000 images is
   enough for a first detector iteration — you don't need 50k to start
   testing the training loop end-to-end). Use CVAT or Label Studio, exported
   in YOLO format, matching the structure `generate_synthetic_data.py`
   produces.
4. **Train the detector** (`training/train_detector.py`). Even a rough model
   beats the mock's fixed center-crop box.
5. **Crop odometer regions with the trained detector**, hand-correct/label
   the digit strings, and use that as your recognizer training set.
6. **Train the recognizer baseline (CRNN+CTC)** (`training/train_recognizer.py`).
   Get this working end-to-end (data → train → export → Android inference)
   before attempting PARSeq — PARSeq is a strict upgrade later, not a
   prerequisite.
7. **Export both to ONNX**, drop into `models/`, implement the
   `_detect_real` / `_recognize_real` methods (Python) and their Kotlin
   equivalents to match your exact export shapes.
8. **Scale the dataset** toward the 50k-100k target, prioritizing diversity
   (day/night, rain, blur, phone models, vehicle types) over raw volume once
   you have a few thousand images — diversity moves accuracy more than
   volume alone at this stage.
9. **Add iOS** once Android is proven in the field.
