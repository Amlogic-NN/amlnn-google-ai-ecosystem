package com.example.compiledmodeldemo

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

class YoloDrawer {

  private val boxColors = intArrayOf(
    Color.RED,
    Color.rgb(0, 128, 255),
    Color.rgb(0, 200, 70),
    Color.rgb(255, 140, 0),
    Color.rgb(185, 40, 255),
    Color.rgb(0, 220, 220),
    Color.rgb(255, 215, 0),
    Color.rgb(80, 200, 120),
  )

  fun draw(
    source: Bitmap,
    detections: List<Detection>,
    classes: List<String>,
  ): Bitmap {
    val output = source.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(output)
    val strokeWidth = max(2f, min(source.width, source.height) / 200f)
    val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      style = Paint.Style.STROKE
      this.strokeWidth = strokeWidth
    }
    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.WHITE
      textSize = max(24f, min(source.width, source.height) / 18f)
      style = Paint.Style.FILL
    }
    val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      style = Paint.Style.FILL
    }

    for (det in detections) {
      val color = boxColors[(det.classId.coerceAtLeast(0)) % boxColors.size]
      boxPaint.color = color
      bgPaint.color = color
      val rect = RectF(det.x1, det.y1, det.x2, det.y2)
      canvas.drawRect(rect, boxPaint)

      val label = buildString {
        append(classes.getOrElse(det.classId) { "unknown" })
        append(' ')
        append(String.format("%.2f", det.score))
      }
      val textWidth = textPaint.measureText(label)
      val textHeight = textPaint.fontMetrics.run { bottom - top }
      val labelLeft = rect.left.coerceAtLeast(0f)
      val labelTop = (rect.top - textHeight - 12f).coerceAtLeast(0f)
      val labelRect = RectF(
        labelLeft,
        labelTop,
        min(source.width.toFloat(), labelLeft + textWidth + 16f),
        min(source.height.toFloat(), labelTop + textHeight + 8f),
      )
      canvas.drawRect(labelRect, bgPaint)
      canvas.drawText(label, labelRect.left + 8f, labelRect.bottom - 8f, textPaint)
    }

    return output
  }
}
