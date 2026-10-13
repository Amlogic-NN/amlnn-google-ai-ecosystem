/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.soundclassifier

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.media.AudioRecord
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.WindowManager
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.os.HandlerCompat
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.examples.soundclassifier.databinding.ActivityMainBinding
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter
import java.util.*
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.Classifications
import java.nio.FloatBuffer
import kotlin.collections.ArrayList
import java.nio.charset.Charset
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
  private val probabilitiesAdapter by lazy { ProbabilitiesAdapter() }

  private var audioClassifier: AudioClassifier? = null
  private var audioRecord: AudioRecord? = null
  private var classificationInterval = 500L // how often should classification run in milli-secs
  private lateinit var handler: Handler // background thread handler to run classification

  // global val
  private var tfLiteModel: MappedByteBuffer? = null
  private var tfLiteOptions: Interpreter.Options? = null
  private var tfLite: Interpreter? = null
  private var tfLite_1: Interpreter? = null
  private var tfLite_cpu: Interpreter? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    Log.d(TAG, "test 1")
    val binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)

    with(binding) {
      recyclerView.apply {
        setHasFixedSize(false)
        adapter = probabilitiesAdapter
      }

      // Input switch to turn on/off classification
      keepScreenOn(inputSwitch.isChecked)
      inputSwitch.setOnCheckedChangeListener { _, isChecked ->
        if (isChecked) startAudioClassification() else stopAudioClassification()
        keepScreenOn(isChecked)
      }
      // Slider which control how often the classification task should run
      classificationIntervalSlider.value = classificationInterval.toFloat()
      classificationIntervalSlider.setLabelFormatter { value: Float ->
        "${value.toInt()} ms"
      }
      classificationIntervalSlider.addOnChangeListener { _, value, _ ->
        classificationInterval = value.toLong()
        stopAudioClassification()
        startAudioClassification()
      }

    }

    // Create a handler to run classification in a background thread
    val handlerThread = HandlerThread("backgroundThread")
    handlerThread.start()
    handler = HandlerCompat.createAsync(handlerThread.looper)

    // Request microphone permission and start running classification
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestMicrophonePermission()
    } else {
      startAudioClassification()
    }
  }

  private fun startAudioClassification() {
    // If the audio classifier is initialized and running, do nothing.
    if (audioClassifier != null) return;

    // Initialize the audio classifier
    val classifier = AudioClassifier.createFromFile(this, MODEL_FILE)
    val audioTensor = classifier.createInputTensorAudio()

    // Initialize the audio recorder
    val record = classifier.createAudioRecord()
    record.startRecording()

    val context:Context = this;
    val modelFile:MappedByteBuffer = loadModelFile(context.assets,"yamnet_1.tflite");
    val options_1:Interpreter.Options = Interpreter.Options();
    options_1.setUseNNAPI(false);
//    options_1.setUseXNNPACK(true);
    options_1.setNumThreads(4);
    tfLite = Interpreter(modelFile,options_1)//正常构建了第一个interpreter，确定XNNPACk
    Log.d(TAG, "Create XNNPACK interpreter")

    val modelFile_1:MappedByteBuffer = loadModelFile(context.assets,"yamnet_2.tflite");
    val options_2:Interpreter.Options = Interpreter.Options();
    options_2.setUseNNAPI(true);
    options_2.setUseXNNPACK(false);
//    options_2.setUseNNAPI(false);
//    options_2.setUseXNNPACK(true);
    options_2.setNumThreads(4);
    tfLite_1 = Interpreter(modelFile_1,options_2)//正常构建了第二个interpreter，确定NNAPI
    Log.d(TAG, "Create NNAPI interpreter")

    val modelFile_2:MappedByteBuffer = loadModelFile(context.assets,"yamnet.tflite");
    val options_3:Interpreter.Options = Interpreter.Options();
    options_3.setUseNNAPI(false);
//    options_1.setUseXNNPACK(true);
    options_3.setNumThreads(4);
    tfLite_cpu = Interpreter(modelFile_2,options_3)//正常构建了第一个interpreter，确定XNNPACk

    val assetManager: AssetManager = context.assets
    val resultList = mutableListOf<Pair<Int,String>>()
    assetManager.open("label_yamnet.txt").bufferedReader().useLines { lines ->
      lines.forEachIndexed { index, line ->
        val pair = Pair(index+1,line)
        resultList.add(pair)
      }
    }
    //test for npu input
//    val inputStream = assetManager.open("yamnet_27.bin")
//    val fileSize = inputStream.available()
//    val test_byteBuffer = ByteBuffer.allocateDirect(fileSize)
//    test_byteBuffer.order(ByteOrder.nativeOrder())
//    inputStream.read(test_byteBuffer.array())
//    test_byteBuffer.rewind()


    // Define the classification runnable
    val run = object : Runnable {
      override fun run() {
        val startTime = System.currentTimeMillis()

        // Load the latest audio sample
        audioTensor.load(record)
//        val output = classifier.classify(audioTensor) // for 后处理
        val byteBuffer: ByteBuffer = audioTensor.tensorBuffer.buffer;
        var out_byteBuffer:ByteBuffer =ByteBuffer.allocate(24576)
        var out_byteBuffer_1:ByteBuffer = ByteBuffer.allocate(2084)
        var out_byteBuffer_cpu:ByteBuffer = ByteBuffer.allocate(2084)
        // 保存输入文件
        Log.d(TAG, "sound buffer size = ${audioTensor.tensorBuffer.flatSize}")
        val path="/data/vendor/nn/sound_input_cpu.bin"
        val path_NN="/data/vendor/nn/sound_input_npu.bin"
        val path_out="/data/vendor/nn/sound_output.bin"
//        saveByteBufferToFile(byteBuffer,path)
//        Log.d(TAG, "Save sound input_CPU OK")
        byteBuffer.rewind()

        tfLite!!.run(byteBuffer,out_byteBuffer)
//        Log.d(TAG, "XNNPACK test yamnet_1 OK")

        out_byteBuffer.rewind()
//        saveByteBufferToFile(out_byteBuffer,path_NN)
//        Log.d(TAG, "Save sound input_NPU OK")


        tfLite_1!!.run(out_byteBuffer,out_byteBuffer_1)
//        tfLite_1!!.run(test_byteBuffer,out_byteBuffer_1)
//        Log.d(TAG, "NNAPI test yamnet_2 OK")

        out_byteBuffer_1.order(ByteOrder.nativeOrder())
        out_byteBuffer_1.rewind()
//        saveByteBufferToFile(out_byteBuffer_1,path_out)
//        Log.d(TAG, "Save sound Output OK")//NPU output目前拿不到正确的输出,模型精度测试OK

//        val floatArray = FloatArray(out_byteBuffer_1.remaining() / 4)
//        out_byteBuffer_1.asFloatBuffer().get(floatArray)

//        byteBuffer.rewind()
//        tfLite_cpu!!.run(byteBuffer,out_byteBuffer_cpu)
//        out_byteBuffer_cpu.order(ByteOrder.nativeOrder())
//        out_byteBuffer_cpu.rewind()

        val floatArray = FloatArray(out_byteBuffer_1.remaining() / 4)
        out_byteBuffer_1.asFloatBuffer().get(floatArray)
//        val maxValue = floatArray.maxOrNull()
//        Log.d(TAG, "maxValue = ${maxValue}")
//        val maxValueIndex = maxValue?.let { floatArray.indexOfFirst { it == it } }
//        Log.d(TAG, "maxValueIndex = ${maxValueIndex}")
//        Log.d(TAG, "resultFloatArray[0] = ${floatArray[0]}")
//        Log.d(TAG, "resultFloatArray[0].label = ${resultList[0].second}")// label = resultList[num].second

//        save the highest score
//        val category:Category = Category.create(resultList[maxValueIndex!!].second,"",maxValue,resultList[maxValueIndex!!].first)
//        val list_test = listOf<Category>(category)
//        Log.d(TAG, "list_test.Categories = ${category}")

        val list_test = mutableListOf<Category>()
        for (i in 0..520)
        {
          var category:Category = Category.create(resultList[i].second,"",floatArray[i],resultList[i].first)
          list_test.add(category)
        }
        val filteredModelOutput = list_test.filter{
          it.score > MINIMUM_DISPLAY_THRESHOLD
        }.sortedBy {
          -it.score
        }


//         Filter out results above a certain threshold, and sort them descendingly
//        val filteredModelOutput = output[0].categories.filter {
//          it.score > MINIMUM_DISPLAY_THRESHOLD
//        }.sortedBy {
//          -it.score
//        }
//        Log.d(TAG, "filteredModelOutput.Categories = ${filteredModelOutput}")

        val finishTime = System.currentTimeMillis()

        Log.d(TAG, "Latency = ${finishTime - startTime}ms")

        // Updating the UI
        runOnUiThread {
//          probabilitiesAdapter.categoryList = filteredModelOutput
          probabilitiesAdapter.categoryList = filteredModelOutput
          probabilitiesAdapter.notifyDataSetChanged()
        }

        // Rerun the classification after a certain interval
        handler.postDelayed(this, classificationInterval)
      }
    }

    // Start the classification process
    handler.post(run)

    // Save the instances we just created for use later
    audioClassifier = classifier
    audioRecord = record
  }


  private fun loadModelFile( assets:AssetManager, modelFilename:String): MappedByteBuffer {
      val fileDescriptor: AssetFileDescriptor =assets.openFd(modelFilename);
      val inputStream: FileInputStream = FileInputStream(fileDescriptor.fileDescriptor);
      val fileChannel:FileChannel = inputStream.channel;
      val startOffset = fileDescriptor.getStartOffset();
      val declaredLength = fileDescriptor.getDeclaredLength();
      val modelFile:MappedByteBuffer =fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
      return modelFile;
  }

  private fun saveByteBufferToFile(byteBuffer: ByteBuffer,filePath:String):Boolean{
    // 创建文件对象
    val file = File(filePath)

    // 创建文件输出流
    val fos = FileOutputStream(file)

    fos.channel.write(byteBuffer)

    // 关闭流
    fos.close()
    return true
  }

  private fun stopAudioClassification() {
    handler.removeCallbacksAndMessages(null)
    audioRecord?.stop()
    audioRecord = null
    audioClassifier = null
  }

  override fun onTopResumedActivityChanged(isTopResumedActivity: Boolean) {
    // Handles "top" resumed event on multi-window environment
    if (isTopResumedActivity && isRecordAudioPermissionGranted()) {
      startAudioClassification()
    } else {
      stopAudioClassification()
    }
  }

  override fun onRequestPermissionsResult(
          requestCode: Int,
          permissions: Array<out String>,
          grantResults: IntArray
  ) {
    if (requestCode == REQUEST_RECORD_AUDIO) {
      if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Log.i(TAG, "Audio permission granted :)")
        startAudioClassification()
      } else {
        Log.e(TAG, "Audio permission not granted :(")
      }
    }
  }

  @RequiresApi(Build.VERSION_CODES.M)
  private fun requestMicrophonePermission() {
    if (isRecordAudioPermissionGranted()) {
      startAudioClassification()
    } else {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
    }
  }

  private fun isRecordAudioPermissionGranted(): Boolean {
      return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO
      ) == PackageManager.PERMISSION_GRANTED
  }

  private fun keepScreenOn(enable: Boolean) =
    if (enable) {
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    } else {
      window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

  companion object {
    const val REQUEST_RECORD_AUDIO = 1337
    private const val TAG = "AudioDemo"
    private const val MODEL_FILE = "yamnet.tflite"
    private const val MINIMUM_DISPLAY_THRESHOLD: Float = 0.3f
  }
}
