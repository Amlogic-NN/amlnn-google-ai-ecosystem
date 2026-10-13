package com.example.compiledmodeldemo

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class Detection(
  val x1: Float,
  val y1: Float,
  val x2: Float,
  val y2: Float,
  val score: Float,
  val classId: Int,
)

data class PostprocessDebug(
  val outputSizes: List<Int>,
  val maxScore: Float,
  val rawDetCount: Int,
  val finalDetCount: Int,
  val mapping: String,
)

/**
 * YOLOX postprocess aligned with yolox_tflite_reference_v3.py:
 * - decode strides 8/16/32
 * - score = objectness * class score
 * - map boxes: (box - pad) / scale
 * - class-wise NMS (scoreThr=0.25, nmsThr=0.45)
 */
class YoloPostprocessor(
  private val inputSize: Int = YoloPreprocessor.INPUT_SIZE,
  private val numClasses: Int = 80,
  private val scoreThreshold: Float = 0.25f,
  private val nmsThreshold: Float = 0.45f,
) {

  var lastDebug: PostprocessDebug? = null
    private set

  fun process(
    rawOutput: FloatArray,
    preprocess: PreprocessResult,
  ): List<Detection> {
    val dim = 5 + numClasses
    require(rawOutput.isNotEmpty() && rawOutput.size % dim == 0) {
      "Unexpected YOLOX output size=${rawOutput.size}, expect N*$dim"
    }

    // Default LiteRT flatten of [1,8400,85] is row-major [N,85].
    // v3 also handles [85,N] via transpose when shape[0]==85.
    var raw = rawOutput
    var numAnchors = rawOutput.size / dim
    var layout = "[N,$dim]"
    val expectedAnchors = expectedNumAnchors(preprocess.inputWidth, preprocess.inputHeight)
    if (numAnchors != expectedAnchors && rawOutput.size / expectedAnchors == dim) {
      // Flattened as [85, N]
      val transposed = FloatArray(rawOutput.size)
      for (r in 0 until dim) {
        for (c in 0 until expectedAnchors) {
          transposed[c * dim + r] = rawOutput[r * expectedAnchors + c]
        }
      }
      raw = transposed
      numAnchors = expectedAnchors
      layout = "transposed_[85,N]->[N,85]"
    }
    require(numAnchors == expectedAnchors) {
      "Output rows $numAnchors != YOLOX grid points $expectedAnchors"
    }

    val (proposals, maxScore) = decodeYolox(
      raw = raw,
      numAnchors = numAnchors,
      dim = dim,
      scale = preprocess.scale,
      padLeft = preprocess.padLeft,
      padTop = preprocess.padTop,
    )

    val finalDets = nmsByClass(proposals)
    lastDebug = PostprocessDebug(
      outputSizes = listOf(rawOutput.size),
      maxScore = maxScore,
      rawDetCount = proposals.size,
      finalDetCount = finalDets.size,
      mapping = "v3 strides=8/16/32 N=$numAnchors layout=$layout pad=(${preprocess.padLeft},${preprocess.padTop})",
    )
    return finalDets
  }

  private fun expectedNumAnchors(inputW: Int, inputH: Int): Int {
    var count = 0
    for (stride in intArrayOf(8, 16, 32)) {
      count += (inputW / stride) * (inputH / stride)
    }
    return count
  }

  private fun decodeYolox(
    raw: FloatArray,
    numAnchors: Int,
    dim: Int,
    scale: Float,
    padLeft: Int,
    padTop: Int,
  ): Pair<MutableList<Detection>, Float> {
    val proposals = mutableListOf<Detection>()
    val strides = intArrayOf(8, 16, 32)
    var index = 0
    var maxScore = 0f
    val invScale = 1f / scale
    val padXF = padLeft.toFloat()
    val padYF = padTop.toFloat()

    for (stride in strides) {
      val gridW = inputSize / stride
      val gridH = inputSize / stride
      val strideF = stride.toFloat()
      for (gy in 0 until gridH) {
        for (gx in 0 until gridW) {
          if (index >= numAnchors) break
          val base = index * dim
          val cx = (raw[base] + gx) * strideF
          val cy = (raw[base + 1] + gy) * strideF
          val bw = exp(raw[base + 2]) * strideF
          val bh = exp(raw[base + 3]) * strideF

          val obj = raw[base + 4]
          var bestClass = 0
          var bestClassScore = 0f
          for (c in 0 until numClasses) {
            val s = raw[base + 5 + c]
            if (s > bestClassScore) {
              bestClassScore = s
              bestClass = c
            }
          }

          val score = obj * bestClassScore
          if (score > maxScore) maxScore = score
          if (score >= scoreThreshold) {
            // v3: (box - pad) / scale
            val left = (cx - bw * 0.5f - padXF) * invScale
            val top = (cy - bh * 0.5f - padYF) * invScale
            val right = (cx + bw * 0.5f - padXF) * invScale
            val bottom = (cy + bh * 0.5f - padYF) * invScale
            proposals.add(Detection(left, top, right, bottom, score, bestClass))
          }
          index++
        }
      }
    }
    return proposals to maxScore
  }

  private fun nmsByClass(input: List<Detection>): List<Detection> {
    val result = mutableListOf<Detection>()
    input.groupBy { it.classId }.forEach { (_, detections) ->
      val sorted = detections.sortedByDescending { it.score }.toMutableList()
      while (sorted.isNotEmpty()) {
        val best = sorted.removeAt(0)
        result.add(best)
        val it = sorted.iterator()
        while (it.hasNext()) {
          val other = it.next()
          if (iou(best, other) > nmsThreshold) {
            it.remove()
          }
        }
      }
    }
    return result.sortedByDescending { it.score }
  }

  private fun iou(a: Detection, b: Detection): Float {
    val x1 = max(a.x1, b.x1)
    val y1 = max(a.y1, b.y1)
    val x2 = min(a.x2, b.x2)
    val y2 = min(a.y2, b.y2)
    val interW = max(0f, x2 - x1)
    val interH = max(0f, y2 - y1)
    val inter = interW * interH
    val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
    val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
    return inter / max(1e-6f, areaA + areaB - inter)
  }

  companion object {
    val CLASSES = listOf(
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
      "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
      "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
      "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "doughnut", "cake",
      "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
      "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
      "toothbrush",
    )
  }
}
