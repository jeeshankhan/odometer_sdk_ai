package com.odometersdk

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** Mirrors preprocessing/pipeline.py: CLAHE -> gamma -> denoise -> sharpen ->
 * resize. Perspective correction is applied earlier via the detection box
 * corners in a full implementation; simplified here to a straight crop,
 * matching the current mock detector's axis-aligned box. Extend with
 * Imgproc.getPerspectiveTransform/warpPerspective once the real detector
 * returns oriented corners. */
object ImageEnhancer {

    fun enhance(croppedBgr: Mat, targetHeight: Int = 64): Mat {
        val gray = Mat()
        Imgproc.cvtColor(croppedBgr, gray, Imgproc.COLOR_BGR2GRAY)

        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        val afterClahe = Mat()
        clahe.apply(gray, afterClahe)

        val gammaCorrected = applyGamma(afterClahe, 1.2)

        val denoised = Mat()
        org.opencv.photo.Photo.fastNlMeansDenoising(gammaCorrected, denoised, 10f)

        val sharpened = sharpen(denoised)

        return resizeToHeight(sharpened, targetHeight)
    }

    private fun applyGamma(gray: Mat, gamma: Double): Mat {
        val lut = Mat(1, 256, CvType.CV_8U)
        val invGamma = 1.0 / gamma
        for (i in 0 until 256) {
            val value = (Math.pow(i / 255.0, invGamma) * 255.0)
            lut.put(0, i, value)
        }
        val result = Mat()
        Core_LUT(gray, lut, result)
        return result
    }

    private fun Core_LUT(src: Mat, lut: Mat, dst: Mat) {
        org.opencv.core.Core.LUT(src, lut, dst)
    }

    private fun sharpen(gray: Mat): Mat {
        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, 0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0)
        val result = Mat()
        Imgproc.filter2D(gray, result, -1, kernel)
        return result
    }

    private fun resizeToHeight(gray: Mat, targetHeight: Int): Mat {
        val scale = targetHeight.toDouble() / gray.rows()
        val newWidth = (gray.cols() * scale).toInt()
        val result = Mat()
        Imgproc.resize(gray, result, Size(newWidth.toDouble(), targetHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
        return result
    }
}
