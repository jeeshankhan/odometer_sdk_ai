package com.odometersdk

import android.content.Context
import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat

/**
 * Public entry point.
 *
 * val sdk = OdometerSDK(context)
 * val result = sdk.read(bitmap, previousReading = 45210)
 * result.reading        // "45892"
 * result.confidence      // 98.7
 * result.isValid          // true
 * result.action           // ConfidenceAction.ACCEPT
 */
class OdometerSDK(private val context: Context) {

    companion object {
        init {
            System.loadLibrary("opencv_java4")
        }
    }

    private val qualityChecker = QualityChecker()
    private val detector = Detector(context)
    private val recognizer = Recognizer(context)

    fun read(bitmap: Bitmap, previousReading: Int? = null): OdometerResult {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        val quality = qualityChecker.run(mat)
        if (!quality.passed) {
            return OdometerResult(
                reading = null, confidence = 0.0, isValid = false,
                action = ConfidenceAction.ASK_RETAKE, digitConfidences = emptyList(),
                issues = quality.issues, validationMessage = null, usedMockModels = false
            )
        }

        val detection = detector.detect(mat)
        val box = detection.odometerBox
        val cropped = Mat(mat, org.opencv.core.Rect(
            box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1
        ))
        val enhanced = ImageEnhancer.enhance(cropped)

        val recognition = recognizer.recognize(enhanced)
        val action = ConfidenceEngine.evaluate(recognition.digitConfidences)

        var isValid = action == ConfidenceAction.ACCEPT
        var validationMessage: String? = null
        if (isValid) {
            val readingInt = recognition.reading.toIntOrNull()
            if (readingInt == null) {
                isValid = false
                validationMessage = "reading was not a valid integer"
            } else {
                val result = BusinessRules.validateReading(readingInt, previousReading)
                isValid = result.isValid
                validationMessage = result.message
            }
        }

        return OdometerResult(
            reading = recognition.reading,
            confidence = recognition.overallConfidence,
            isValid = isValid,
            action = action,
            digitConfidences = recognition.digitConfidences,
            issues = quality.issues,
            validationMessage = validationMessage,
            usedMockModels = detection.usedMock || recognition.usedMock
        )
    }

    fun close() {
        detector.close()
        recognizer.close()
    }
}
