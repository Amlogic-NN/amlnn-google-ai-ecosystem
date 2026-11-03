package com.example.compiledmodeldemo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Draws YOLO boxes on top of PreviewView.
 * Do NOT use PorterDuff.Mode.CLEAR here — it punches through and erases the
 * camera PreviewView underneath.
 */
class DetectionOverlayView @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
) : View(context, attrs) {

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

  private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.STROKE
    strokeWidth = 4f
  }
  private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.WHITE
    textSize = 36f
    style = Paint.Style.FILL
  }
  private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.FILL
  }
  private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.GREEN
    textSize = 40f
    style = Paint.Style.FILL
  }

  @Volatile
  private var detections: List<Detection> = emptyList()

  @Volatile
  private var sourceWidth: Int = 0

  @Volatile
  private var sourceHeight: Int = 0

  @Volatile
  private var frameId: Int = 0

  init {
    setWillNotDraw(false)
    // Transparent so PreviewView shows through; system clears this view on invalidate.
    setBackgroundColor(Color.TRANSPARENT)
  }

  fun setDetections(
    detections: List<Detection>,
    sourceWidth: Int,
    sourceHeight: Int,
    frameId: Int = 0,
  ) {
    this.detections = detections
    this.sourceWidth = sourceWidth
    this.sourceHeight = sourceHeight
    this.frameId = frameId
    postInvalidate()
  }

  fun clearDetections() {
    detections = emptyList()
    sourceWidth = 0
    sourceHeight = 0
    frameId = 0
    postInvalidate()
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)

    canvas.drawText("f=$frameId dets=${detections.size}", 24f, 48f, hudPaint)

    val srcW = sourceWidth
    val srcH = sourceHeight
    val current = detections
    if (srcW <= 0 || srcH <= 0 || current.isEmpty()) return

    val viewW = width.toFloat()
    val viewH = height.toFloat()
    val scale = min(viewW / srcW, viewH / srcH)
    val offsetX = (viewW - srcW * scale) / 2f
    val offsetY = (viewH - srcH * scale) / 2f

    for (det in current) {
      val color = boxColors[(det.classId.coerceAtLeast(0)) % boxColors.size]
      boxPaint.color = color
      bgPaint.color = color

      val rect = RectF(
        det.x1 * scale + offsetX,
        det.y1 * scale + offsetY,
        det.x2 * scale + offsetX,
        det.y2 * scale + offsetY,
      )
      canvas.drawRect(rect, boxPaint)

      val label = buildString {
        append(YoloPostprocessor.CLASSES.getOrElse(det.classId) { "unknown" })
        append(' ')
        append(String.format("%.2f", det.score))
      }
      val textWidth = textPaint.measureText(label)
      val textHeight = textPaint.fontMetrics.run { bottom - top }
      val labelLeft = rect.left.coerceAtLeast(0f)
      val labelTop = (rect.top - textHeight - 8f).coerceAtLeast(0f)
      val labelRect = RectF(
        labelLeft,
        labelTop,
        min(viewW, labelLeft + textWidth + 16f),
        min(viewH, labelTop + textHeight + 8f),
      )
      canvas.drawRect(labelRect, bgPaint)
      canvas.drawText(label, labelRect.left + 8f, labelRect.bottom - 8f, textPaint)
    }
  }
}
