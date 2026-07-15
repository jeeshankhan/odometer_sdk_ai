package com.odometersdk

/** Mirrors api.sdk.SDKResponse in the Python reference implementation,
 * so behavior/output stays consistent across platforms. */
data class OdometerResult(
    val reading: String?,
    val confidence: Double,
    val isValid: Boolean,
    val action: ConfidenceAction,
    val digitConfidences: List<Double>,
    val issues: List<String>,
    val validationMessage: String?,
    val usedMockModels: Boolean
)

enum class ConfidenceAction {
    ACCEPT, RETRY_PREPROCESSING, ASK_RETAKE
}

data class QualityReport(
    val passed: Boolean,
    val blurScore: Double,
    val brightness: Double,
    val glareRatio: Double,
    val issues: List<String>
)

data class DetectionBox(val x1: Int, val y1: Int, val x2: Int, val y2: Int)

data class DetectionResult(
    val dashboardBox: DetectionBox?,
    val odometerBox: DetectionBox,
    val confidence: Double,
    val usedMock: Boolean
)

data class RecognitionResult(
    val reading: String,
    val digitConfidences: List<Double>,
    val overallConfidence: Double,
    val usedMock: Boolean
)
