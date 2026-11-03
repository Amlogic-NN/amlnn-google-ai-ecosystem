package com.example.interpreterdemo

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
import com.example.interpreterdemo.databinding.ActivityMainBinding
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.system.measureTimeMillis
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.AmlDelegate

/**
 * Live YOLOX on the Interpreter API.
 *
 * The source model is the plain yolox_s_w8a16_f16.tflite. AmlDelegate.model_path
 * is the app cache directory (the only path this APK can write). cache_path stays
 * empty, so the first run JIT-compiles to cache/<model_name>.adla and later
 * launches reuse that file.
 */
class MainActivity : AppCompatActivity() {

  private lateinit var binding: ActivityMainBinding
  private val preprocessor = YoloPreprocessor()
  private val postprocessor = YoloPostprocessor()

  private var amlDelegate: AmlDelegate? = null
  private var interpreter: Interpreter? = null
  private var inputBuffer: ByteBuffer? = null
  private var outputBuffer: ByteBuffer? = null
  private var jitAdlaPath: String = ""

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
      val error = runCatching { ensureModelLoaded() }.exceptionOrNull()
      runOnUiThread {
        if (error != null) {
          Log.e(TAG, "Model load failed", error)
          binding.statusText.text = "Model load failed"
          binding.logText.text = buildString {
            appendLine(buildStartupHint())
            appendLine("FAILED: ${error.message}")
          }
          binding.startButton.isEnabled = false
          return@runOnUiThread
        }
        bindCameraPreviewOnly()
        binding.statusText.text = "Model ready. Tap Start for live detection."
        binding.logText.text = buildStartupHint()
        binding.startButton.isEnabled = true
        binding.stopButton.isEnabled = false
      }
    }
  }

  private fun ensureModelLoaded() {
    if (interpreter != null) return

    preloadNpuLibs()

    val modelName = BuildConfig.MODEL_NAME
    // App-private cache. The process can create files here; /data/vendor/nn cannot.
    val jitDir = cacheDir.apply { mkdirs() }
    val adlaFile = File(jitDir, "$modelName.adla")
    jitAdlaPath = adlaFile.absolutePath
    Log.i(
      TAG,
      "JIT model=$modelName dir=${jitDir.absolutePath} adlaExists=${adlaFile.exists()} " +
        "adlaBytes=${adlaFile.length()}",
    )

    val modelBuffer = loadMappedModel(BuildConfig.MODEL_ASSET)
    val amlOptions = AmlDelegate.Options()
    // model_path is the directory AmlContextCompile writes <model_name>.adla into.
    // Native ResolveAdlaFilePath appends '/' + model_name + ".adla".
    amlOptions.model_path = jitDir.absolutePath
    amlOptions.model_name = modelName
    // cache_path is an existing .adla file, not a directory. A non-empty value
    // skips JIT and fails when the file is missing, so leave it unset.
    amlOptions.cache_path = null
    amlOptions.mode_buffer = null

    val delegate = AmlDelegate(amlOptions)
    try {
      val options = Interpreter.Options()
        .setNumThreads(2)
        .setUseXNNPACK(false)
        .setUseNNAPI(false)
        .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_APPLICATION_ONLY)
        .addDelegate(delegate)
      val created = Interpreter(modelBuffer, options)
      val inputTensor = created.getInputTensor(0)
      val outputTensor = created.getOutputTensor(0)
      Log.i(
        TAG,
        "tensors in=${inputTensor.shape().contentToString()} ${inputTensor.dataType()} " +
          "out=${outputTensor.shape().contentToString()} ${outputTensor.dataType()}",
      )
      require(inputTensor.dataType() == DataType.FLOAT32) {
        "Expected FLOAT32 input, got ${inputTensor.dataType()} " +
          "shape=${inputTensor.shape().contentToString()}"
      }
      require(outputTensor.dataType() == DataType.FLOAT32) {
        "Expected FLOAT32 output, got ${outputTensor.dataType()} " +
          "shape=${outputTensor.shape().contentToString()}"
      }
      val expectedInBytes = YoloPreprocessor.INPUT_SIZE * YoloPreprocessor.INPUT_SIZE * 3 * 4
      require(inputTensor.numBytes() == expectedInBytes) {
        "Input bytes=${inputTensor.numBytes()} expected=$expectedInBytes"
      }

      inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
      outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
      amlDelegate = delegate
      interpreter = created
    } catch (t: Throwable) {
      delegate.close()
      throw t
    }
  }

  private fun preloadNpuLibs() {
    // Load before AmlDelegate so dlopen("libnnsdk.so") hits an already-mapped library.
    // adla pulls teec; nnsdk pulls adla.
    System.loadLibrary("teec")
    System.loadLibrary("adla")
    System.loadLibrary("nnsdk")
    Log.i(TAG, "Preloaded teec/adla/nnsdk from ${applicationInfo.nativeLibraryDir}")
  }

  private fun loadMappedModel(assetName: String): MappedByteBuffer {
    assets.openFd(assetName).use { afd ->
      FileInputStream(afd.fileDescriptor).use { input ->
        return input.channel.map(
          FileChannel.MapMode.READ_ONLY,
          afd.startOffset,
          afd.declaredLength,
        )
      }
    }
  }

  private fun releaseModel() {
    runCatching {
      interpreter?.close()
      amlDelegate?.close()
    }
    interpreter = null
    amlDelegate = null
    inputBuffer = null
    outputBuffer = null
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

  private fun updatePreviewHostAspect(frameW: Int, frameH: Int) {
    if (frameW <= 0 || frameH <= 0) return
    if (frameW == lastPreviewFrameW && frameH == lastPreviewFrameH) return
    lastPreviewFrameW = frameW
    lastPreviewFrameH = frameH
    Log.i(TAG, "Camera frame ${frameW}x${frameH} (preview host stays match_parent)")
  }

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
    if (interpreter == null) {
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
    binding.startButton.isEnabled = interpreter != null
    binding.stopButton.isEnabled = false
    binding.statusText.text = "Stopped"
  }

  private fun bindCameraWithAnalysis() {
    val provider = cameraProvider ?: return
    provider.unbindAll()

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

      val bitmap = try {
        CameraFrameUtils.imageProxyToBitmap(imageProxy)
      } catch (t: Throwable) {
        Log.e(TAG, "Frame convert failed", t)
        lastError = "convert: ${t.message}"
        inferenceBusy.set(false)
        return@setAnalyzer
      } finally {
        imageProxy.close()
      }

      try {
        runFrame(bitmap)
      } catch (t: Throwable) {
        Log.e(TAG, "Frame inference failed", t)
        lastError = t.message
        runOnUiThread {
          binding.statusText.text =
            "Inference error · frames=$frameCount drop=$dropCount · ${t.message}"
        }
      } finally {
        if (!bitmap.isRecycled) bitmap.recycle()
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
    val runtime = interpreter
    val input = inputBuffer
    val output = outputBuffer
    if (runtime == null || input == null || output == null) {
      lastError = "model/buffers not ready"
      return
    }

    val prep = preprocessor.preprocess(bitmap, targetWidth = 640, targetHeight = 640)
    val inHash = hashFloats(prep.inputFloats)
    require(prep.inputFloats.size * 4 == input.capacity()) {
      "preprocess floats=${prep.inputFloats.size} buffer=${input.capacity()}"
    }

    val latencyMs: Long
    val rawOutput: FloatArray
    val outHash: Int
    synchronized(modelLock) {
      input.clear()
      input.asFloatBuffer().put(prep.inputFloats)
      input.rewind()
      output.clear()
      latencyMs = measureTimeMillis {
        runtime.run(input, output)
      }
      output.rewind()
      rawOutput = FloatArray(output.capacity() / 4)
      output.asFloatBuffer().get(rawOutput)
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

  private fun buildStartupHint(): String {
    val adla = File(jitAdlaPath)
    return buildString {
      appendLine("model=${BuildConfig.MODEL_ASSET}")
      appendLine("api=Interpreter + AmlDelegate")
      appendLine("compile=JIT (cache_path empty)")
      appendLine("adla=${if (jitAdlaPath.isEmpty()) "(not loaded)" else jitAdlaPath}")
      if (jitAdlaPath.isNotEmpty()) {
        appendLine("adlaExists=${adla.exists()} bytes=${adla.length()}")
      }
      appendLine("preprocess=v3 center-letterbox RGB (x-mean)/std")
      appendLine("post=v3 score=0.25 nms=0.45")
    }
  }

  private fun buildLiveLog(
    detections: List<Detection>,
    dbg: PostprocessDebug?,
  ): String {
    return buildString {
      appendLine("model=${BuildConfig.MODEL_ASSET}")
      appendLine("api=Interpreter + AmlDelegate JIT")
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
    private const val TAG = "InterpreterDemo"
  }
}
