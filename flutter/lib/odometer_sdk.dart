library odometer_sdk;

import 'dart:async';
import 'package:flutter/services.dart';

/// Confidence-driven next action for a reading.
enum ConfidenceAction { accept, retryPreprocessing, askRetake }

ConfidenceAction _actionFromString(String value) {
  switch (value) {
    case 'accept':
      return ConfidenceAction.accept;
    case 'retry_preprocessing':
      return ConfidenceAction.retryPreprocessing;
    default:
      return ConfidenceAction.askRetake;
  }
}

/// Result of an odometer read. Mirrors SDKResponse in the Python reference
/// implementation and OdometerResult in the Android SDK, so behavior is
/// consistent no matter which layer you're debugging.
class OdometerResult {
  final String? reading;
  final double confidence;
  final bool isValid;
  final ConfidenceAction action;
  final List<double> digitConfidences;
  final List<String> issues;
  final String? validationMessage;
  final bool usedMockModels;

  OdometerResult({
    required this.reading,
    required this.confidence,
    required this.isValid,
    required this.action,
    required this.digitConfidences,
    required this.issues,
    required this.validationMessage,
    required this.usedMockModels,
  });

  factory OdometerResult.fromMap(Map<dynamic, dynamic> map) {
    return OdometerResult(
      reading: map['reading'] as String?,
      confidence: (map['confidence'] as num).toDouble(),
      isValid: map['isValid'] as bool,
      action: _actionFromString(map['action'] as String),
      digitConfidences: (map['digitConfidences'] as List)
          .map((e) => (e as num).toDouble())
          .toList(),
      issues: List<String>.from(map['issues'] as List),
      validationMessage: map['validationMessage'] as String?,
      usedMockModels: map['usedMockModels'] as bool,
    );
  }
}

/// Flutter developers shouldn't need to know anything about ONNX or OpenCV
/// internally - this is the entire public surface.
class OdometerSdk {
  static const MethodChannel _channel = MethodChannel('odometer_sdk');

  /// [imagePath] - path to a JPEG/PNG odometer photo.
  /// [previousReading] - last known reading, used for plausibility checks.
  static Future<OdometerResult> read(
    String imagePath, {
    int? previousReading,
  }) async {
    final result = await _channel.invokeMethod<Map<dynamic, dynamic>>('read', {
      'imagePath': imagePath,
      'previousReading': previousReading,
    });
    if (result == null) {
      throw PlatformException(
        code: 'NULL_RESULT',
        message: 'OdometerSDK returned no result',
      );
    }
    return OdometerResult.fromMap(result);
  }
}
