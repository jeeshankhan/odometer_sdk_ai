package com.odometersdk

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Mat
import java.io.File
import kotlin.random.Random

/** Mirrors recognition/recognizer.py: mock mode returns a plausible random
 * reading with per-digit confidences until recognizer.onnx is bundled. */
class Recognizer(context: Context, modelFileName: String = "recognizer.onnx") {

    private var session: OrtSession? = null
    private val env = OrtEnvironment.getEnvironment()

    init {
        val modelFile = File(context.filesDir, modelFileName)
        if (modelFile.exists()) {
            session = env.createSession(modelFile.absolutePath)
        }
    }

    fun recognize(croppedGray: Mat): RecognitionResult {
        val activeSession = session
        return if (activeSession != null) {
            recognizeReal(croppedGray, activeSession)
        } else {
            recognizeMock()
        }
    }

    private fun recognizeMock(): RecognitionResult {
        val length = listOf(5, 6, 6, 6, 7).random()
        val digits = (1..length).map { Random.nextInt(0, 10) }
        val confidences = digits.map { 0.96 + Random.nextDouble() * 0.039 }.toMutableList()
        if (Random.nextDouble() < 0.3) {
            val idx = Random.nextInt(length)
            confidences[idx] = 0.55 + Random.nextDouble() * 0.30
        }
        return RecognitionResult(
            reading = digits.joinToString(""),
            digitConfidences = confidences,
            overallConfidence = (confidences.min() * 100),
            usedMock = true
        )
    }

    private fun recognizeReal(croppedGray: Mat, session: OrtSession): RecognitionResult {
        // TODO: normalize croppedGray to the model's expected input shape,
        // run session, CTC/attention-decode the output sequence, and use
        // per-timestep softmax max as each digit's confidence.
        throw NotImplementedError(
            "Wire up real recognizer inference to match your exported ONNX model's I/O shapes."
        )
    }

    fun close() {
        session?.close()
    }
}
