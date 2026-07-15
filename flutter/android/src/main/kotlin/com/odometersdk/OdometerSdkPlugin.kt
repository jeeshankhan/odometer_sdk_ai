package com.odometersdk

import android.graphics.BitmapFactory
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

/** Bridges the Dart `OdometerSdk.read()` call to the native OdometerSDK
 * Kotlin implementation in android/. */
class OdometerSdkPlugin : FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel
    private var sdk: OdometerSDK? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "odometer_sdk")
        channel.setMethodCallHandler(this)
        sdk = OdometerSDK(binding.applicationContext)
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "read" -> {
                val imagePath = call.argument<String>("imagePath")
                val previousReading = call.argument<Int>("previousReading")

                if (imagePath == null) {
                    result.error("INVALID_ARGS", "imagePath is required", null)
                    return
                }

                val bitmap = BitmapFactory.decodeFile(imagePath)
                if (bitmap == null) {
                    result.error("DECODE_FAILED", "Could not decode image at $imagePath", null)
                    return
                }

                try {
                    val odometerResult = sdk!!.read(bitmap, previousReading)
                    result.success(odometerResult.toMap())
                } catch (e: Exception) {
                    result.error("SDK_ERROR", e.message, null)
                }
            }
            else -> result.notImplemented()
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        sdk?.close()
    }
}

/** Converts the result to the plain Map the Dart side's fromMap() expects. */
private fun OdometerResult.toMap(): Map<String, Any?> = mapOf(
    "reading" to reading,
    "confidence" to confidence,
    "isValid" to isValid,
    "action" to when (action) {
        ConfidenceAction.ACCEPT -> "accept"
        ConfidenceAction.RETRY_PREPROCESSING -> "retry_preprocessing"
        ConfidenceAction.ASK_RETAKE -> "ask_retake"
    },
    "digitConfidences" to digitConfidences,
    "issues" to issues,
    "validationMessage" to validationMessage,
    "usedMockModels" to usedMockModels
)
