package com.odometersdk

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Mat
import java.io.File

/** Mirrors detection/detector.py: falls back to a mock center-crop box when
 * no detector.onnx is bundled yet, so the rest of the SDK is testable
 * before Phase 2 training is complete. */
class Detector(context: Context, modelFileName: String = "detector.onnx") {

    private var session: OrtSession? = null
    private val env = OrtEnvironment.getEnvironment()

    init {
        val modelFile = File(context.filesDir, modelFileName)
        if (modelFile.exists()) {
            session = env.createSession(modelFile.absolutePath)
        }
    }

    fun detect(image: Mat): DetectionResult {
        val activeSession = session
        return if (activeSession != null) {
            detectReal(image, activeSession)
        } else {
            detectMock(image)
        }
    }

    private fun detectMock(image: Mat): DetectionResult {
        val w = image.cols()
        val h = image.rows()
        val odometerBox = DetectionBox(
            x1 = (w * 0.30).toInt(), y1 = (h * 0.40).toInt(),
            x2 = (w * 0.70).toInt(), y2 = (h * 0.60).toInt()
        )
        val dashboardBox = DetectionBox(
            x1 = (w * 0.1).toInt(), y1 = (h * 0.1).toInt(),
            x2 = (w * 0.9).toInt(), y2 = (h * 0.9).toInt()
        )
        return DetectionResult(dashboardBox, odometerBox, confidence = 0.50, usedMock = true)
    }

    private fun detectReal(image: Mat, session: OrtSession): DetectionResult {
        // TODO: letterbox-resize `image` to the model's input size, normalize,
        // build an OnnxTensor, run session, decode YOLO output + NMS.
        // Exact pre/post-processing depends on your export settings in
        // training/export_onnx.py - confirm input/output tensor shapes with
        // `session.inputInfo` / `session.outputInfo` before implementing.
        throw NotImplementedError(
            "Wire up real detector inference to match your exported ONNX model's I/O shapes."
        )
    }

    fun close() {
        session?.close()
    }
}
