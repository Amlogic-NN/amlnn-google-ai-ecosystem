package com.example.compiledmodeldemo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Surface
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.compiledmodeldemo.databinding.ActivityMainBinding
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.system.measureTimeMillis

/**
 * Live YOLO:
 * Preview + ImageAnalysis -> Bitmap -> preprocess -> LiteRT run -> overlay boxes.
 * Model is created once; frames are dropped while inference is busy.
 */
class MainActivity : AppCompatActivity() {

  private lateinit var binding: ActivityMainBinding
  private val preprocessor = YoloPreprocessor()
  private val postprocessor = YoloPostprocessor()

  private var env: Environment? = null
  private var compiledModel: CompiledModel? = null
  private var inputBuffers: List<TensorBuffer> = emptyList()
  private var outputBuffers: List<TensorBuffer> = emptyList()

  private var cameraProvider: ProcessCameraProvider? = null
  private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
  private var analysisBound = false
  private var detecting = false

  private val inferenceBusy = AtomicBoolean(false)
  @Volatile private var busySinceMs: Long = 0L
  private val modelLock = Any()
  private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
  private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()

  private var frameCount = 0
  private var dropCount = 0
  private var lastLatencyMs = 0L
  private var lastDetCount = 0
  private var lastError: String? = null
  private var lastPreviewFrameW = 0
  private var lastPreviewFrameH = 0
  private var lastInHash = 0
  private var lastOutHash = 0

  private val permissionLauncher =
    registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) {
        prepareModelThenCamera()
      } else {
        binding.statusText.text = getString(R.string.status_camera_denied)
        Toast.makeText(this, R.string.status_camera_denied, Toast.LENGTH_LONG).show()
      }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)

    binding.previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
    // Letterbox to keep camera aspect in landscape (no crop mismatch with overlay).
    binding.previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

    binding.startButton.setOnClickListener { startDetection() }
    binding.stopButton.setOnClickListener { stopDetection() }

    binding.logText.text = buildStartupHint()

    when {
      ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED -> prepareModelThenCamera()
      else -> permissionLauncher.launch(Manifest.permission.CAMERA)
    }
  }

  override fun onDestroy() {
    stopDetection()
    cameraExecutor.shutdown()
    inferenceExecutor.shutdown()
    releaseModel()
    super.onDestroy()
  }

  private fun prepareModelThenCamera() {
    binding.statusText.text = getString(R.string.status_loading_model)
    binding.startButton.isEnabled = false

    inferenceExecutor.execute {
      val ok = runCatching { ensureModelLoaded() }.onFailure {
        Log.e(TAG, "Model load failed", it)
      }.isSuccess

      runOnUiThread {
        if (!ok) {
          binding.statusText.text = "Model load failed"
          binding.logText.text = "FAILED to load ${BuildConfig.MODEL_ASSET}"
          binding.startButton.isEnabled = false
          return@runOnUiThread
        }
        bindCameraPreviewOnly()
        binding.statusText.text = "Model ready. Tap Start for live detection."
        binding.startButton.isEnabled = true
        binding.stopButton.isEnabled = false
      }
    }
  }

  private fun ensureModelLoaded() {
    if (compiledModel != null) return

    val modelAsset = BuildConfig.MODEL_ASSET
    val useNpu = BuildConfig.USE_NPU
    val isDispatchModel = modelAsset.contains("dispatch", ignoreCase = true)
    val options =
      if (useNpu) {
        CompiledModel.Options(Accelerator.NPU)
      } else {
        CompiledModel.Options.CPU
      }

    val createdEnv = createEnvironmentIfNeeded(useNpu)
    env = createdEnv
    val model = createCompiledModel(modelAsset, options, createdEnv, isDispatchModel)
    compiledModel = model

    // Create IO buffers once and reuse. Do NOT recreate outputs on the same
    // CompiledModel (AML: LiteRT count becomes 6 vs NPU 3). Stale-output on
    // reuse needs a dispatch.so fix (proper output detach / host sync).
    inputBuffers = model.createInputBuffers()
    outputBuffers = model.createOutputBuffers()
    require(inputBuffers.isNotEmpty()) { "Model has no inputs." }
    require(outputBuffers.size == 1) {
      "Unexpected output count=${outputBuffers.size}, expected 1 YOLOX output [1,8400,85]"
    }
    Log.i(
      TAG,
      "Model loaded: $modelAsset inputs=${inputBuffers.size} outputs=${outputBuffers.size} (YOLOX float)",
    )
  }

  private fun releaseModel() {
    runCatching {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      compiledModel?.close()
      env?.close()
    }
    inputBuffers = emptyList()
    outputBuffers = emptyList()
    compiledModel = null
    env = null
  }

  private fun bindCameraPreviewOnly() {
    val providerFuture = ProcessCameraProvider.getInstance(this)
    providerFuture.addListener(
      {
        try {
          val provider = providerFuture.get()
          cameraProvider = provider
          provider.unbindAll()
          analysisBound = false

          cameraSelector = resolveCameraSelector(provider)
          val rotation = currentDisplayRotation()
          val preview = Preview.Builder()
            .setTargetRotation(rotation)
            .build()
            .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }
          provider.bindToLifecycle(this, cameraSelector, preview)
          Log.i(TAG, "Preview bound with selector=$cameraSelector rotation=$rotation")
        } catch (t: Throwable) {
          Log.e(TAG, "bindCameraPreviewOnly failed", t)
          binding.statusText.text = "Camera bind failed: ${t.message}"
          binding.startButton.isEnabled = false
          Toast.makeText(this, "Camera bind failed", Toast.LENGTH_LONG).show()
        }
      },
      ContextCompat.getMainExecutor(this),
    )
  }

  private fun currentDisplayRotation(): Int {
    return binding.previewView.display?.rotation
      ?: @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation
  }

  /**
   * Keep preview host match_parent. Resizing it to huge pixel sizes (e.g.
   * 2240x1680) has caused RenderThread SIGSEGV on this device.
   * FIT_CENTER on PreviewView + overlay mapping is enough for aspect.
   */
  private fun updatePreviewHostAspect(frameW: Int, frameH: Int) {
    if (frameW <= 0 || frameH <= 0) return
    if (frameW == lastPreviewFrameW && frameH == lastPreviewFrameH) return
    lastPreviewFrameW = frameW
    lastPreviewFrameH = frameH
    Log.i(TAG, "Camera frame ${frameW}x${frameH} (preview host stays match_parent)")
  }

  /**
   * AML / external USB cameras often report LENS_FACING as null and
   * HARDWARE_LEVEL_EXTERNAL, so DEFAULT_BACK_CAMERA cannot resolve them.
   */
  private fun resolveCameraSelector(provider: ProcessCameraProvider): CameraSelector {
    val available = provider.availableCameraInfos
    Log.i(TAG, "availableCameraInfos=${available.size}")
    available.forEachIndexed { index, info ->
      Log.i(
        TAG,
        "camera[$index] facing=${safeLensFacing(info)} implType=${info.implementationType}",
      )
    }

    val candidates = listOf(
      CameraSelector.DEFAULT_BACK_CAMERA,
      CameraSelector.DEFAULT_FRONT_CAMERA,
      CameraSelector.Builder()
        .requireLensFacing(CameraSelector.LENS_FACING_EXTERNAL)
        .build(),
      // Accept whatever CameraX enumerated (covers facing=null external cams).
      CameraSelector.Builder()
        .addCameraFilter { cameraInfos ->
          if (cameraInfos.isEmpty()) emptyList() else listOf(cameraInfos.first())
        }
        .build(),
    )

    for (selector in candidates) {
      if (runCatching { provider.hasCamera(selector) }.getOrDefault(false)) {
        Log.i(TAG, "Using camera selector: $selector")
        return selector
      }
    }
    error("No usable camera found on this device")
  }

  private fun safeLensFacing(info: CameraInfo): String {
    return runCatching { info.lensFacing.toString() }.getOrElse { "null/unknown" }
  }

  private fun startDetection() {
    if (compiledModel == null) {
      Toast.makeText(this, "Model not ready", Toast.LENGTH_SHORT).show()
      return
    }
    if (detecting) return
    detecting = true
    frameCount = 0
    dropCount = 0
    lastPreviewFrameW = 0
    lastPreviewFrameH = 0
    binding.overlayView.clearDetections()
    binding.startButton.isEnabled = false
    binding.stopButton.isEnabled = true
    binding.statusText.text = getString(R.string.status_running)
    bindCameraWithAnalysis()
  }

  private fun stopDetection() {
    detecting = false
    analysisBound = false
    inferenceBusy.set(false)
    cameraProvider?.let { provider ->
      provider.unbindAll()
      val rotation = currentDisplayRotation()
      val preview = Preview.Builder()
        .setTargetRotation(rotation)
        .build()
        .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }
      runCatching {
        provider.bindToLifecycle(this, cameraSelector, preview)
      }.onFailure { t ->
        Log.e(TAG, "Rebind preview after stop failed", t)
      }
    }
    binding.overlayView.clearDetections()
    binding.startButton.isEnabled = compiledModel != null
    binding.stopButton.isEnabled = false
    binding.statusText.text = "Stopped"
  }

  private fun bindCameraWithAnalysis() {
    val provider = cameraProvider ?: return
    provider.unbindAll()

    // EXTERNAL / AML boards often already deliver landscape buffers with
    // rotationDegrees=0. Forcing display rotation can scramble analysis frames.
    val rotation = Surface.ROTATION_0
    val preview = Preview.Builder()
      .setTargetRotation(currentDisplayRotation())
      .build()
      .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

    val resolutionSelector = ResolutionSelector.Builder()
      .setResolutionStrategy(
        ResolutionStrategy(
          Size(1280, 720),
          ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
        ),
      )
      .build()

    val analysis = ImageAnalysis.Builder()
      .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
      .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
      .setResolutionSelector(resolutionSelector)
      .setTargetRotation(rotation)
      .build()

    // Run convert+infer on one thread. Avoids "first frame ok, then busy stuck"
    // when cameraExecutor holds the busy flag during slow JPEG convert while
    // NPU run hangs on a second thread.
    analysis.setAnalyzer(inferenceExecutor) { imageProxy ->
      if (!detecting) {
        imageProxy.close()
        return@setAnalyzer
      }

      maybeRecoverStuckBusy()

      if (!inferenceBusy.compareAndSet(false, true)) {
        dropCount++
        imageProxy.close()
        maybeRefreshHud()
        return@setAnalyzer
      }
      busySinceMs = System.currentTimeMillis()

      var bitmap: Bitmap? = null
      try {
        bitmap = CameraFrameUtils.imageProxyToBitmap(imageProxy)
      } catch (t: Throwable) {
        Log.e(TAG, "Frame convert failed", t)
        lastError = "convert: ${t.message}"
        inferenceBusy.set(false)
        return@setAnalyzer
      } finally {
        imageProxy.close()
      }

      try {
        runFrame(bitmap!!)
      } catch (t: Throwable) {
        Log.e(TAG, "Frame inference failed", t)
        lastError = t.message
        runOnUiThread {
          binding.statusText.text =
            "Inference error · frames=$frameCount drop=$dropCount · ${t.message}"
        }
      } finally {
        bitmap?.let {
          if (!it.isRecycled) it.recycle()
        }
        inferenceBusy.set(false)
        busySinceMs = 0L
      }
    }

    provider.bindToLifecycle(
      this,
      cameraSelector,
      preview,
      analysis,
    )
    analysisBound = true
  }

  private fun maybeRecoverStuckBusy() {
    val since = busySinceMs
    if (since == 0L || !inferenceBusy.get()) return
    val waited = System.currentTimeMillis() - since
    if (waited < 3_000L) return
    Log.e(TAG, "inferenceBusy stuck for ${waited}ms — force reset")
    lastError = "busy-stuck-${waited}ms"
    inferenceBusy.set(false)
    busySinceMs = 0L
  }

  private fun runFrame(bitmap: Bitmap) {
    val model = compiledModel
    val inputs = inputBuffers
    val outputs = outputBuffers
    if (model == null || inputs.isEmpty() || outputs.size != 1) {
      lastError = "model/buffers not ready"
      return
    }

    val prep = preprocessor.preprocess(bitmap, targetWidth = 640, targetHeight = 640)
    val inHash = hashFloats(prep.inputFloats)

    val latencyMs: Long
    val rawOutput: FloatArray
    val outHash: Int
    synchronized(modelLock) {
      // YOLOX v3: NHWC RGB float after (x-mean)/std
      inputs[0].writeFloat(prep.inputFloats.copyOf())
      latencyMs = measureTimeMillis {
        model.run(inputs, outputs)
      }
      val view = outputs[0].readFloat()
      rawOutput = FloatArray(view.size) { idx -> view[idx] }
      outHash = hashFloats(rawOutput)
    }

    val detections = postprocessor.process(rawOutput, prep)
      .map { det ->
        det.copy(
          x1 = det.x1.coerceIn(0f, bitmap.width.toFloat()),
          y1 = det.y1.coerceIn(0f, bitmap.height.toFloat()),
          x2 = det.x2.coerceIn(0f, bitmap.width.toFloat()),
          y2 = det.y2.coerceIn(0f, bitmap.height.toFloat()),
        )
      }

    frameCount++
    lastLatencyMs = latencyMs
    lastDetCount = detections.size
    lastInHash = inHash
    lastOutHash = outHash
    lastError = null

    val srcW = bitmap.width
    val srcH = bitmap.height
    val dbg = postprocessor.lastDebug
    Log.i(
      TAG,
      "frame=$frameCount latency=${latencyMs}ms dets=${detections.size} size=${srcW}x${srcH} " +
        "drop=$dropCount inHash=$inHash outHash=$outHash maxScore=${dbg?.maxScore}",
    )

    val uiDets = detections.toList()
    val uiFrame = frameCount
    runOnUiThread {
      if (!detecting) return@runOnUiThread
      updatePreviewHostAspect(srcW, srcH)
      binding.overlayView.setDetections(uiDets, srcW, srcH, uiFrame)
      val maxScoreText = dbg?.maxScore?.let { String.format(Locale.US, "%.3f", it) } ?: "?"
      binding.statusText.text =
        "Live f=$uiFrame · ${latencyMs}ms · dets=$lastDetCount · max=$maxScoreText · in=$inHash out=$outHash"
      binding.logText.text = buildLiveLog(uiDets, dbg)
    }
  }

  private fun hashFloats(data: FloatArray): Int {
    var h = 1
    val step = maxOf(1, data.size / 64)
    var i = 0
    while (i < data.size) {
      h = 31 * h + data[i].toBits()
      i += step
    }
    h = 31 * h + data.size
    return h
  }

  private fun maybeRefreshHud() {
    if (dropCount % 15 != 0) return
    runOnUiThread {
      if (!detecting) return@runOnUiThread
      val err = lastError?.let { " · err=$it" } ?: ""
      binding.statusText.text =
        "Live · ${lastLatencyMs}ms · dets=$lastDetCount · frames=$frameCount drop=$dropCount$err"
    }
  }

  private fun createEnvironmentIfNeeded(useNpu: Boolean): Environment? {
    if (!useNpu) return null
    val libDir = applicationInfo.nativeLibraryDir
    Log.i(TAG, "Creating Environment with explicit dirs: libDir=$libDir")
    return Environment.create(
      mapOf(
        Environment.Option.DispatchLibraryDir to libDir,
        Environment.Option.CompilerPluginLibraryDir to libDir,
      ),
    )
  }

  private fun createCompiledModel(
    modelAsset: String,
    options: CompiledModel.Options,
    env: Environment?,
    isDispatchModel: Boolean,
  ): CompiledModel {
    return if (env != null) {
      if (isDispatchModel) {
        CompiledModel.create(assets, modelAsset, options, env)
      } else {
        CompiledModel.create(copyAssetToInternalFile(modelAsset), options, env)
      }
    } else {
      if (isDispatchModel) {
        CompiledModel.create(assets, modelAsset, options)
      } else {
        CompiledModel.create(copyAssetToInternalFile(modelAsset), options)
      }
    }
  }

  private fun copyAssetToInternalFile(assetName: String): String {
    val outFile = File(cacheDir, assetName)
    outFile.parentFile?.mkdirs()
    if (!outFile.exists() || outFile.length() == 0L) {
      assets.open(assetName).use { input ->
        outFile.outputStream().use { output ->
          input.copyTo(output)
        }
      }
    }
    return outFile.absolutePath
  }

  private fun buildStartupHint(): String {
    return buildString {
      appendLine("model=${BuildConfig.MODEL_ASSET}")
      appendLine("mode=LIVE YOLOX camera")
      appendLine("preprocess=v3 center-letterbox RGB (x-mean)/std")
      appendLine("post=v3 score=0.25 nms=0.45")
      appendLine("accelerator=${if (BuildConfig.USE_NPU) "NPU" else "CPU"}")
    }
  }

  private fun buildLiveLog(
    detections: List<Detection>,
    dbg: PostprocessDebug?,
  ): String {
    return buildString {
      appendLine("model=${BuildConfig.MODEL_ASSET}")
      appendLine(
        String.format(
          Locale.US,
          "latency_ms=%d frames=%d dropped=%d dets=%d maxScore=%.3f",
          lastLatencyMs,
          frameCount,
          dropCount,
          detections.size,
          dbg?.maxScore ?: -1f,
        ),
      )
      appendLine("outSizes=${dbg?.outputSizes} map=${dbg?.mapping}")
      if (detections.isEmpty()) {
        appendLine("No boxes. If maxScore<<0.25, input/output mismatch is likely.")
      }
      detections.take(8).forEachIndexed { index, det ->
        val label = YoloPostprocessor.CLASSES.getOrElse(det.classId) { "unknown" }
        appendLine(
          "#$index $label ${String.format(Locale.US, "%.2f", det.score)} " +
            "[${det.x1.toInt()},${det.y1.toInt()},${det.x2.toInt()},${det.y2.toInt()}]",
        )
      }
    }
  }

  companion object {
    private const val TAG = "CompiledModelDemo"
  }
}
