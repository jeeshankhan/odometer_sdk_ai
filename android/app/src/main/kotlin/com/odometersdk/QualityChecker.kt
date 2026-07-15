package com.odometersdk

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/** Direct Kotlin port of preprocessing/quality_checks.py - keep both in sync
 * if you tune thresholds; they should behave identically. */
class QualityChecker(
    private val blurThreshold: Double = 100.0,
    private val brightnessLow: Double = 40.0,
    private val brightnessHigh: Double = 220.0,
    private val glareBlownOutThreshold: Int = 245,
    private val glareMaxRatio: Double = 0.03
) {
    fun run(bgrImage: Mat): QualityReport {
        val gray = Mat()
        Imgproc.cvtColor(bgrImage, gray, Imgproc.COLOR_BGR2GRAY)

        val blurScore = laplacianVariance(gray)
        val brightness = Core.mean(gray).`val`[0]
        val glareRatio = blownOutRatio(gray)

        val issues = mutableListOf<String>()
        if (blurScore < blurThreshold) issues.add("image_too_blurry")
        if (brightness < brightnessLow || brightness > brightnessHigh) issues.add("bad_lighting")
        if (glareRatio > glareMaxRatio) issues.add("glare_detected")

        return QualityReport(
            passed = issues.isEmpty(),
            blurScore = blurScore,
            brightness = brightness,
            glareRatio = glareRatio,
            issues = issues
        )
    }

    private fun laplacianVariance(gray: Mat): Double {
        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
        val mean = MatOfInt()
        val stdDev = org.opencv.core.MatOfDouble()
        Core.meanStdDev(laplacian, org.opencv.core.MatOfDouble(), stdDev)
        val std = stdDev.toArray()[0]
        return std * std
    }

    private fun blownOutRatio(gray: Mat): Double {
        val mask = Mat()
        Imgproc.threshold(gray, mask, glareBlownOutThreshold.toDouble(), 255.0, Imgproc.THRESH_BINARY)
        val blownPixels = Core.countNonZero(mask)
        return blownPixels.toDouble() / (gray.rows() * gray.cols())
    }
}
