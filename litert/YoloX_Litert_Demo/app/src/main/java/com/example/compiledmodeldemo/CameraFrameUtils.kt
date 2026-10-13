package com.example.compiledmodeldemo

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

object CameraFrameUtils {

  private const val TAG = "CameraFrameUtils"

  fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val format = image.format
    val rotation = image.imageInfo.rotationDegrees
    Log.d(TAG, "frame ${image.width}x${image.height} format=$format rotation=$rotation")

    // For YUV_420_888, prefer our NV21 path. ImageProxy.toBitmap() on some AML
    // external cameras can return wrong colors and kill YOLO scores.
    val base = when (format) {
      ImageFormat.YUV_420_888 -> yuvToBitmap(image)
      else -> tryConvertDirect(image) ?: yuvToBitmap(image)
    }

    if (rotation == 0) {
      return ensureArgb8888(base)
    }
    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    val rotated = Bitmap.createBitmap(base, 0, 0, base.width, base.height, matrix, true)
    if (rotated !== base) {
      base.recycle()
    }
    return ensureArgb8888(rotated)
  }

  private fun tryConvertDirect(image: ImageProxy): Bitmap? {
    return runCatching { image.toBitmap() }.getOrNull()
  }

  private fun ensureArgb8888(bitmap: Bitmap): Bitmap {
    if (bitmap.config == Bitmap.Config.ARGB_8888 && !bitmap.isMutable) {
      // Immutable ARGB is fine for preprocess getPixels.
      return bitmap
    }
    if (bitmap.config == Bitmap.Config.ARGB_8888) return bitmap
    return bitmap.copy(Bitmap.Config.ARGB_8888, false).also { bitmap.recycle() }
  }

  private fun yuvToBitmap(image: ImageProxy): Bitmap {
    val nv21 = yuv420888ToNv21(image)
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    // Higher JPEG quality reduces chroma loss before YOLO.
    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 95, out)
    val jpegBytes = out.toByteArray()
    return android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
      ?: error("Failed to decode camera frame")
  }

  private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
    val width = image.width
    val height = image.height
    val ySize = width * height
    val nv21 = ByteArray(ySize + ySize / 2)

    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]

    val yBuffer = yPlane.buffer.duplicate()
    val uBuffer = uPlane.buffer.duplicate()
    val vBuffer = vPlane.buffer.duplicate()
    yBuffer.rewind()
    uBuffer.rewind()
    vBuffer.rewind()

    val yRowStride = yPlane.rowStride
    val yPixelStride = yPlane.pixelStride
    var outputOffset = 0
    for (row in 0 until height) {
      val yRowStart = row * yRowStride
      for (col in 0 until width) {
        nv21[outputOffset++] = yBuffer.get(yRowStart + col * yPixelStride)
      }
    }

    val chromaHeight = height / 2
    val chromaWidth = width / 2
    val uRowStride = uPlane.rowStride
    val vRowStride = vPlane.rowStride
    val uPixelStride = uPlane.pixelStride
    val vPixelStride = vPlane.pixelStride

    // Fast path: NV21 already (pixelStride=2, V then U interleaved).
    if (vPixelStride == 2 && uPixelStride == 2 &&
      vRowStride == uRowStride &&
      vBuffer.get(1) == uBuffer.get(0)
    ) {
      // Common case on many devices: VU interlaced; copy row by row carefully.
    }

    for (row in 0 until chromaHeight) {
      for (col in 0 until chromaWidth) {
        val uIndex = row * uRowStride + col * uPixelStride
        val vIndex = row * vRowStride + col * vPixelStride
        nv21[outputOffset++] = vBuffer.get(vIndex)
        nv21[outputOffset++] = uBuffer.get(uIndex)
      }
    }
    return nv21
  }
}
