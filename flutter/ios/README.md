# iOS SDK — not yet implemented

The Flutter Dart API (`lib/odometer_sdk.dart`) and MethodChannel name
(`odometer_sdk`) are platform-agnostic and already correct for iOS.

To add iOS support:

1. Build a native iOS SDK (Swift) mirroring `android/app/.../OdometerSDK.kt`:
   `QualityChecker`, `Detector`, `Recognizer`, `ImageEnhancer`, `ConfidenceEngine`,
   `BusinessRules` — same structure, same thresholds, same JSON-ish result shape.
   Use ONNX Runtime's iOS pod (`onnxruntime-objc` or `onnxruntime-c`) for inference
   and OpenCV's iOS framework for image processing — same model files
   (`detector.onnx`, `recognizer.onnx`) work on both platforms unchanged.
2. Implement `OdometerSdkPlugin.swift` handling the same `"read"` method call
   with the same argument/result shape as `OdometerSdkPlugin.kt`.
3. Register it in `ios/odometer_sdk.podspec` (not yet created).

Until this exists, calling `OdometerSdk.read()` from a Flutter app running on
iOS will fail with a "MissingPluginException".
