package com.example.compiledmodeldemo

import android.graphics.Bitmap
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YOLOX preprocess aligned with yolox_tflite_reference_v3.py:
 * - Center letterbox to 640x640, pad value 114
 * - RGB channel order (Android Bitmap is already RGB)
 * - Normalize: (RGB - mean) / std
 * - Feed float32 NHWC
 */
data class PreprocessResult(
  val inputFloats: FloatArray,
  /** Letterbox scale: resized / original. */
  val scale: Float,
  val padLeft: Int,
  val padTop: Int,
  val inputWidth: Int,
  val inputHeight: Int,
)

class YoloPreprocessor {

  fun preprocess(
    bitmap: Bitmap,
    targetWidth: Int = INPUT_SIZE,
    targetHeight: Int = INPUT_SIZE,
  ): PreprocessResult {
    require(targetWidth > 0 && targetHeight > 0) { "Invalid target size." }
    val w = bitmap.width
    val h = bitmap.height
    require(w > 0 && h > 0) { "Invalid source bitmap." }

    // letterbox_center_bgr in v3 (Bitmap path skips BGR; already RGB).
    val scale = min(targetWidth / w.toFloat(), targetHeight / h.toFloat())
    val resizedW = (w * scale).roundToInt().coerceAtLeast(1)
    val resizedH = (h * scale).roundToInt().coerceAtLeast(1)
    val padW = (targetWidth - resizedW) / 2.0f
    val padH = (targetHeight - resizedH) / 2.0f
    // Match OpenCV: top = round(pad_h - 0.1), bottom = round(pad_h + 0.1)
    val padTop = (padH - 0.1f).roundToInt()
    val padBottom = (padH + 0.1f).roundToInt()
    val padLeft = (padW - 0.1f).roundToInt()
    val padRight = (padW + 0.1f).roundToInt()

    val resized = Bitmap.createScaledBitmap(bitmap, resizedW, resizedH, true)
    val temp = IntArray(resizedW * resizedH)
    resized.getPixels(temp, 0, resizedW, 0, 0, resizedW, resizedH)
    if (resized !== bitmap) {
      resized.recycle()
    }

    val padGray = 114
    val padColor = (0xff shl 24) or (padGray shl 16) or (padGray shl 8) or padGray
    val canvas = IntArray(targetWidth * targetHeight) { padColor }
    for (y in 0 until resizedH) {
      val srcRow = y * resizedW
      val dstRow = (y + padTop) * targetWidth + padLeft
      System.arraycopy(temp, srcRow, canvas, dstRow, resizedW)
    }

    // (RGB - mean) / std, NHWC float32
    val out = FloatArray(targetWidth * targetHeight * 3)
    var dst = 0
    for (p in canvas) {
      val r = ((p shr 16) and 0xff).toFloat()
      val g = ((p shr 8) and 0xff).toFloat()
      val b = (p and 0xff).toFloat()
      out[dst++] = (r - MEAN_R) / STD_R
      out[dst++] = (g - MEAN_G) / STD_G
      out[dst++] = (b - MEAN_B) / STD_B
    }

    // Sanity: letterbox border sizes should reconstruct 640.
    require(padLeft + resizedW + padRight == targetWidth) {
      "letterbox width mismatch: $padLeft+$resizedW+$padRight != $targetWidth"
    }
    require(padTop + resizedH + padBottom == targetHeight) {
      "letterbox height mismatch: $padTop+$resizedH+$padBottom != $targetHeight"
    }

    return PreprocessResult(
      inputFloats = out,
      scale = scale,
      padLeft = padLeft,
      padTop = padTop,
      inputWidth = targetWidth,
      inputHeight = targetHeight,
    )
  }

  companion object {
    const val INPUT_SIZE = 640

    // ImageNet-style mean/std used by yolox_tflite_reference_v3.py
    private const val MEAN_R = 123.675f
    private const val MEAN_G = 116.28f
    private const val MEAN_B = 103.53f
    private const val STD_R = 58.395f
    private const val STD_G = 57.12f
    private const val STD_B = 57.375f
  }
}
