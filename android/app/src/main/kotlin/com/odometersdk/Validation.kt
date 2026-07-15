package com.odometersdk

/** Mirrors validation/confidence_engine.py */
object ConfidenceEngine {
    fun evaluate(
        digitConfidences: List<Double>,
        acceptThreshold: Double = 0.97,
        retryThreshold: Double = 0.80,
        maxLowDigitsForRetry: Int = 2
    ): ConfidenceAction {
        val lowIndices = digitConfidences.indices.filter { digitConfidences[it] < acceptThreshold }
        if (lowIndices.isEmpty()) return ConfidenceAction.ACCEPT

        val veryLow = lowIndices.filter { digitConfidences[it] < retryThreshold }
        return if (veryLow.isNotEmpty() || lowIndices.size > maxLowDigitsForRetry) {
            ConfidenceAction.ASK_RETAKE
        } else {
            ConfidenceAction.RETRY_PREPROCESSING
        }
    }
}

/** Mirrors validation/business_rules.py */
data class ValidationResult(val isValid: Boolean, val message: String)

object BusinessRules {
    fun validateReading(
        currentReading: Int,
        previousReading: Int?,
        maxPlausibleDailyKm: Int = 1500,
        daysSincePrevious: Double = 1.0
    ): ValidationResult {
        if (previousReading == null) {
            return ValidationResult(true, "no previous reading to compare against")
        }
        if (currentReading < previousReading) {
            return ValidationResult(
                false,
                "current reading $currentReading is lower than previous $previousReading"
            )
        }
        val delta = currentReading - previousReading
        val maxPlausibleDelta = maxPlausibleDailyKm * maxOf(daysSincePrevious, 1.0)
        if (delta > maxPlausibleDelta) {
            return ValidationResult(
                false,
                "jump of $delta exceeds plausible max of $maxPlausibleDelta"
            )
        }
        return ValidationResult(true, "within plausible range")
    }
}
