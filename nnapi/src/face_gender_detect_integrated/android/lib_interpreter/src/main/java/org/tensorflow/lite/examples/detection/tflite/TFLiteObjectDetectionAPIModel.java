/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import static java.lang.Math.exp;
import static java.lang.Math.max;
import static java.lang.Math.min;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.util.LogPrinter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.tensorflow.lite.DataType;
//import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.*;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.examples.detection.tflite.env.Logger;
import org.tensorflow.lite.support.common.FileUtil;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API: -
 * https://github.com/tensorflow/models/tree/master/research/object_detection where you can find the
 * training code.
 *
 * <p>To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithInterpreter";
  private static final Logger LOGGER = new Logger();
  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private final List<String> labels = new ArrayList<>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;
  static float[][] delta_s3 = new float[1][1];
  static float[][] delta_s2 = new float[1][1];
  static float[][] delta_s1 = new float[1][1];

  static float[][] local_delta_stage3 = new float[1][3];
  static float[][] local_delta_stage2 = new float[1][3];
  static float[][] local_delta_stage1 = new float[1][3];

  static float[][] pred_age_stage3 = new float[1][3];
  static float[][] pred_age_stage2 = new float[1][3];
  static float[][] pred_age_stage1 = new float[1][3];

  private ByteBuffer imgData;

  private MappedByteBuffer tfLiteModel;
  private Interpreter.Options tfLiteOptions;
  private Interpreter tfLite;

  private TensorImage inputImageBuffer;
  /** Image size along the x axis. */
  private final int imageSizeX = 640;

  /** Image size along the y axis. */
  private final int imageSizeY = 384;

  public static class sortable_bbox_retina {
    int index;
    int classId;
    float probs;
    sortable_bbox_retina(
            int index, int classId, float probs) {
      this.index = index;
      this.classId = classId;
      this.probs = probs;
    }
  }

  class SortByProb implements Comparator<sortable_bbox_retina> {
    @Override
    public int compare(sortable_bbox_retina o1,sortable_bbox_retina o2){
      if(o1.probs > o2.probs){
        return -1;
      }
      else if(o1.probs < o2.probs){
        return 1;
      }
      else{
        return 0;
      }
    }
  }

  public static class box{

    float x;
    float y;
    float w;
    float h;
    box(float x , float y, float w, float h){
      this.x = x;
      this.y = y;
      this.w = w;
      this.h = h;
    }
  }
  public static class detBox{
    float x;
    float y;
    float w;
    float h;
    float score;
    detBox(float x , float y, float w, float h, float score){
      this.x = x;
      this.y = y;
      this.w = w;
      this.h = h;
      this.score = score;
    }
  }

  public static class face_detect_out_t{
    int detNum;
    detBox[] pBox;
    public face_detect_out_t(int detNum, detBox[] pBox){
      this.detNum = detNum;
      this.pBox = pBox;
    }
  }

  /** post process array **/
  float[][][] bbox_32 = new float[12][20][8];
  float[][][] bbox_16 = new float[24][40][8];
  float[][][] bbox_8 = new float[48][80][8];
  // prob score,input
  float[][][] prob_32 = new float[240][2][2];
  float[][][] prob_16 = new float[960][2][2];
  float[][][] prob_8 = new float[3840][2][2];

  static float[][] prob32 = new float[480][1];
  static float[][] prob16 = new float[1920][1];
  static float[][] prob8 = new float[7680][1];

  static sortable_bbox_retina[] s1 = new sortable_bbox_retina[480];
  static sortable_bbox_retina[] s2 = new sortable_bbox_retina[1920];
  static sortable_bbox_retina[] s3 = new sortable_bbox_retina[7680];
  private final int g_detect_number = 230;

  float[] pprob_32 = new float[960];
  float[] pprob_16 = new float[3840];
  float[] pprob_8 = new float[15360];
  float[] pbox_32 = new float[1920];
  float[] pbox_16 = new float[7680];
  float[] pbox_8 = new float[30720];

  byte[] inputs0 = new byte[960];//960 u8
  byte[] inputs1 = new byte[1920];
  byte[] inputs3 = new byte[3840];//3840 u8
  byte[] inputs4 = new byte[7680];
  byte[] inputs6 = new byte[15360];//15360 u8
  byte[] inputs7 = new byte[30720];

  detBox[] pBox = new detBox[230];

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
//  public static Detector genderCreate(Context context,String model_name) throws IOException {
//    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();
//    MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(context,model_name);
//    NnApiDelegate nnApiDelegate = new NnApiDelegate();
//    try{
//      Interpreter.Options options = new Interpreter.Options();
//      options.setNumThreads(NUM_THREADS);
//      options.setUseXNNPACK(false);
////      options.setUseNNAPI(true);
//      options.addDelegate(nnApiDelegate);
//      d.tfLite = new Interpreter(tfliteModel,options);
//      d.tfLiteModel = tfliteModel;
//    }catch (Exception e) {
//      throw new RuntimeException(e);
//    }
//    return d;
//  }
  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param modelFilename The model file path relative to the assets folder
   * @param labelFilename The label file path relative to the assets folder
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelFilename);
//    MetadataExtractor metadata = new MetadataExtractor(modelFile);
//    try (BufferedReader br =
//        new BufferedReader(
//            new InputStreamReader(
//                metadata.getAssociatedFile(labelFilename), Charset.defaultCharset())))
//    {
//      String line;
//      while ((line = br.readLine()) != null) {
//        Log.w("TFOD", line);
//        d.labels.add(line);
//      }
//    }

    d.inputSize = inputSize;

    try {
      Interpreter.Options options = new Interpreter.Options();
      options.setNumThreads(NUM_THREADS);
      options.setUseNNAPI(true);
      options.setUseXNNPACK(false);
      d.tfLite = new Interpreter(modelFile, options);
      d.tfLiteModel = modelFile;
      d.tfLiteOptions = options;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
//    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
//    d.imgData.order(ByteOrder.nativeOrder());
//    d.intValues = new int[d.inputSize * d.inputSize];
//    d.imgData = ByteBuffer.allocateDirect(1 * 640 * 384 * 3 * numBytesPerChannel);
//    d.imgData.order(ByteOrder.nativeOrder());
//    d.intValues = new int[384 * 640];

//    d.outputLocations = new float[1][NUM_DETECTIONS][4];
//    d.outputClasses = new float[1][NUM_DETECTIONS];
//    d.outputScores = new float[1][NUM_DETECTIONS];
//    d.numDetections = new float[1];
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.


    DataType imageDataType = tfLite.getInputTensor(0).dataType();
    inputImageBuffer = new TensorImage(imageDataType);
    inputImageBuffer = loadImage(bitmap, 0,640,384);

    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");


    Object[] inputArray = {inputImageBuffer.getBuffer()};
    int[] OutputShape;
    DataType OutputDataType;
    ByteBuffer x;
    Map<Integer,Object> outputs = new HashMap<>();
    for(int i =0;i<tfLite.getOutputTensorCount();i++){
      OutputShape = tfLite.getOutputTensor(i).shape();
      OutputDataType = tfLite.getOutputTensor(i).dataType();
      x= TensorBuffer.createFixedSize(OutputShape,OutputDataType).getBuffer();
      outputs.put(i,x);
    }
    Trace.endSection();
    // Run the inference call.
    Trace.beginSection("run");
    long startTime = System.nanoTime();
    tfLite.runForMultipleInputsOutputs(inputArray, outputs);
    long endTime = System.nanoTime();
    Trace.endSection();
    long duration = endTime - startTime;
    double durationMs = duration / 1e6;
    LOGGER.d("face detection inference = %f ms",durationMs);

    ByteBuffer x0,x1,x3,x4,x6,x7 ;
    x0 = (ByteBuffer) outputs.get(0);
    x1 = (ByteBuffer) outputs.get(1);
    x3 = (ByteBuffer) outputs.get(3);
    x4 = (ByteBuffer) outputs.get(4);
    x6 = (ByteBuffer) outputs.get(6);
    x7 = (ByteBuffer) outputs.get(7);
    x0.flip();
    x1.flip();
    x3.flip();
    x4.flip();
    x6.flip();
    x7.flip();
    x0.get(inputs0,0,960);
    x1.get(inputs1,0,1920);
    x3.get(inputs3,0,3840);
    x4.get(inputs4,0,7480);
    x6.get(inputs6,0,15360);
    x7.get(inputs7,0,30720);

    int n =0,m=0,l=0,o=0,p=0,q=0;
    float q0 = 0.00390625f;             //prob_8,prob_16,prob_32 scale
    float q1 = 0.0031159173231571913f;  //box_32 scale
    float q2 = 0.0032531728502362967f;  //box_16 scale
    float q3 = 0.0027853690553456545f;  //box_8  scale

    for(int i=0;i<960;i++){
      int input_buffer = inputs0[i] & 0xFF;
      pprob_32[i] = (float)(input_buffer)*q0;
    }
    for(int i=0;i<240;i++){
      for(int j = 0;j<2;j++) {
        for (int k = 0; k < 2; k++) {
          prob_32[i][j][k] = pprob_32[4*i + 2 * j + k];//length 960 ,
        }
      }
    }

    for(int i=0;i<3840;i++){
      int input_buffer = inputs3[i] & 0xFF;
      pprob_16[i] = (float)(input_buffer)*q0;
    }
    for(int i=0;i<960;i++){
      for(int j = 0;j<2;j++) {
        for (int k = 0; k < 2; k++) {
          prob_16[i][j][k] = pprob_16[4 * i + 2 * j + k];
        }
      }
    }

    for(int i=0;i<15360;i++){
      int input_buffer = inputs6[i] & 0xFF;
      pprob_8[i] = (float)(input_buffer)*q0;
    }
    for(int i=0;i<3840;i++){
      for(int j = 0;j<2;j++) {
        for (int k = 0; k < 2; k++) {
          prob_8[i][j][k] = pprob_8[4 * i + 2 * j +  k];//填充prob_32,prob_16,prob_8
        }
      }
    }


    for(int i=0;i<1920;i++){
      int input_buffer = inputs1[i] & 0xFF;
      pbox_32[i] = (float)(input_buffer-135)*q1;
    }
    for(int i=0;i<12;i++){
      for(int j = 0;j<20;j++) {
        for (int k = 0; k < 8; k++) {
          bbox_32[i][j][k] = pbox_32[160 * i + 8 * j +  k];
        }
      }
    }
    for(int i=0;i<7480;i++){
      int input_buffer = inputs4[i] & 0xFF;
      pbox_16[i] = (float)(input_buffer-104)*q2;
    }
    for(int i=0;i<24;i++){
      for(int j = 0;j<40;j++) {
        for (int k = 0; k < 8; k++) {
          bbox_16[i][j][k] = pbox_16[320 * i + 8 * j +  k];
        }
      }
    }
    for(int i=0;i<30720;i++){
      int input_buffer = inputs7[i] & 0xFF;
      pbox_8[i] = (float)(input_buffer-97)*q3;
    }
    for(int i=0;i<48;i++){
      for(int j = 0;j<80;j++) {
        for (int k = 0; k < 8; k++) {
          bbox_8[i][j][k] = pbox_8[640 * i + 8 * j +  k];
        }
      }
    }

    //后处理
    int detNum = 0;
    detNum = face_detect_postprocess(pBox);
    face_detect_out_t pface_detect_out_t = new face_detect_out_t(detNum,pBox);
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    for(int i = 0; i < pface_detect_out_t.detNum;i++){
      float left = pface_detect_out_t.pBox[i].x;
      float top = pface_detect_out_t.pBox[i].y;
      float right = pface_detect_out_t.pBox[i].x + pface_detect_out_t.pBox[i].w;
      float bot = pface_detect_out_t.pBox[i].y + pface_detect_out_t.pBox[i].h;
      if (left < 0) left = 0.000001f;
      if (right < 0) right = 0.000001f;
      if (top < 0) top = 0.000001f;
      if (bot < 0 ) bot = 0.000001f;
      if(left > 1) left = 0.99999f;
      if(right > 1) right = 0.99999f;
      if(top > 1) top = 0.99999f;
      if(bot > 1) bot = 0.99999f;
      RectF location = new RectF(left,top,right,bot);
      Recognition recognition = new Recognition("","face",pface_detect_out_t.pBox[i].score,location);
      recognitions.add(recognition);
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  public float genderRecognizeImage(final Bitmap bitmap){
    // Log this method so that it can be analyzed with systrace.

    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.


    DataType imageDataType = tfLite.getInputTensor(0).dataType();
    inputImageBuffer = new TensorImage(imageDataType);
    inputImageBuffer = loadImage(bitmap, 0,48,48);


    Object[] inputArray = {inputImageBuffer.getBuffer()};
    int[] OutputShape;
    DataType OutputDataType;
    ByteBuffer x;
    Map<Integer,Object> outputs = new HashMap<>();
    for(int i =0;i<tfLite.getOutputTensorCount();i++){
      OutputShape = tfLite.getOutputTensor(i).shape();
      OutputDataType = tfLite.getOutputTensor(i).dataType();
      x= TensorBuffer.createFixedSize(OutputShape,OutputDataType).getBuffer();
      outputs.put(i,x);
    }

    tfLite.runForMultipleInputsOutputs(inputArray, outputs);


    ByteBuffer x0 ;
    x0 = (ByteBuffer) outputs.get(0);
    x0.flip();
    byte[] gender = new byte[2];
    x0.get(gender,0,2);
    LOGGER.d("gender[0] = %d",gender[0]);
    float q0 = 0.00390625f;
    float[] gender_if = new float[2];
    for(int i =0;i<2;i++){
      int buffer = gender[i] & 0xFF;
      LOGGER.d("gender[%d] = %d",i,buffer);
      gender_if[i] = buffer * q0;
    }
    LOGGER.d("gender_if[0] = %f",gender_if[0]);
    return gender_if[0];
  }

  public float ageRecognizationImage(final Bitmap bitmap){
    DataType imageDataType = tfLite.getInputTensor(0).dataType();
    inputImageBuffer = new TensorImage(imageDataType);
    inputImageBuffer = loadImage(bitmap, 0,64,64);


    Object[] inputArray = {inputImageBuffer.getBuffer()};
    Object[] bgrArray = convertRGBToBGR(inputArray);

    int[] OutputShape;
    DataType OutputDataType;
    ByteBuffer x;
    Map<Integer,Object> outputs = new HashMap<>();
    for(int i =0;i<tfLite.getOutputTensorCount();i++){
      OutputShape = tfLite.getOutputTensor(i).shape();
      OutputDataType = tfLite.getOutputTensor(i).dataType();
      x= TensorBuffer.createFixedSize(OutputShape,OutputDataType).getBuffer();
      outputs.put(i,x);
    }

//    tfLite.runForMultipleInputsOutputs(inputArray, outputs);
    tfLite.runForMultipleInputsOutputs(bgrArray, outputs);

    ByteBuffer x0,x1,x2,x3,x4,x5,x6,x7,x8 ;
    x0 = (ByteBuffer) outputs.get(0);
    x1 = (ByteBuffer) outputs.get(1);
    x2 = (ByteBuffer) outputs.get(2);
    x3 = (ByteBuffer) outputs.get(3);
    x4 = (ByteBuffer) outputs.get(4);
    x5 = (ByteBuffer) outputs.get(5);
    x6 = (ByteBuffer) outputs.get(6);
    x7 = (ByteBuffer) outputs.get(7);
    x8 = (ByteBuffer) outputs.get(8);
    x0.flip();
    x1.flip();
    x2.flip();
    x3.flip();
    x4.flip();
    x5.flip();
    x6.flip();
    x7.flip();
    x8.flip();
    byte[] delta3 = new byte[4];
    byte[] delta2 = new byte[4];
    byte[] delta1 = new byte[4];
    byte[] local_delta3 = new byte[12];
    byte[] local_delta2 = new byte[12];
    byte[] local_delta1 = new byte[12];
    byte[] pred3 = new byte[3];
    byte[] pred2 = new byte[3];
    byte[] pred1 = new byte[3];
    x2.get(delta1,0,4);
    x1.get(delta2,0,4);
    x0.get(delta3,0,4);
    x5.get(local_delta1,0,12);
    x4.get(local_delta2,0,12);
    x3.get(local_delta3,0,12);
    x8.get(pred1,0,3);
    x7.get(pred2,0,3);
    x6.get(pred3,0,3);

    delta_s3[0][0] = byteToFloat(delta3);
    delta_s2[0][0] = byteToFloat(delta2);
    delta_s1[0][0] = byteToFloat(delta1);
    LOGGER.d("delta_s3[0][0]= %f",delta_s3[0][0]);
    LOGGER.d("delta_s2[0][0]= %f",delta_s2[0][0]);
    LOGGER.d("delta_s1[0][0]= %f",delta_s1[0][0]);

    int j = 0;
    for(int i =0;i<12;i = i+4){
      local_delta_stage3[0][j] = byteArrayToFloat(local_delta3,i);
      local_delta_stage2[0][j] = byteArrayToFloat(local_delta2,i);
      local_delta_stage1[0][j] = byteArrayToFloat(local_delta1,i);
      LOGGER.d("local_delta_stage3[0][%d]= %f",i,local_delta_stage3[0][j]);
      LOGGER.d("local_delta_stage2[0][%d]= %f",i,local_delta_stage2[0][j]);
      LOGGER.d("local_delta_stage1[0][%d]= %f",i,local_delta_stage1[0][j]);
      j++;
    }
    float q3 = 0.0009277688222937286f;
    float q2 = 0.00040087723755277693f;
    float q1 = 0.00021605855727102607f;

    for(int i = 0;i<3;i++){
      int buffer3 = pred3[i] & 0xFF;
      int buffer2 = pred2[i] & 0xFF;
      int buffer1 = pred1[i] & 0xFF;
      pred_age_stage3[0][i] = (float)buffer3 ;
      pred_age_stage2[0][i] = (float)buffer2 ;
      pred_age_stage1[0][i] = (float)buffer1 ;
      pred_age_stage3[0][i] = (float)buffer3 * q3;
      pred_age_stage2[0][i] = (float)buffer2 * q2;
      pred_age_stage1[0][i] = (float)buffer1 * q1;
      LOGGER.d("pred_age_stage3[0][%d]= %f",i,pred_age_stage3[0][i]);
      LOGGER.d("pred_age_stage2[0][%d]= %f",i,pred_age_stage2[0][i]);
      LOGGER.d("pred_age_stage1[0][%d]= %f",i,pred_age_stage1[0][i]);
    }

    int S1=3;
    int S2=3;
    int S3=3;
    int lambda_local =1;
    int lambda_d =1;
    float a=0;
    float b=0;
    float c=0;
    for (int i = 0;i < S1;i++)
    {
      a = a + (float)(i + lambda_local * local_delta_stage1[0][i]) * pred_age_stage1[0][i];
    }
    a = a /(S1*(1 + lambda_d * delta_s1[0][0]));
    LOGGER.d("a= %f",a);
    for (int i = 0;i < S2;i++)
    {
      b = b + (float)(i + lambda_local * local_delta_stage2[0][i]) * pred_age_stage2[0][i];
    }
    b = b /(S1*(1 + lambda_d * delta_s1[0][0])) / (S2*(1+lambda_d*delta_s2[0][0])) ;
    LOGGER.d("b= %f",b);
    for (int i = 0;i < S3;i++)
    {
      c = c + (float)(i + lambda_local * local_delta_stage3[0][i]) * pred_age_stage3[0][i];
    }
    c = c /(S1*(1 + lambda_d * delta_s1[0][0])) / (S2*(1+lambda_d*delta_s2[0][0])) / (S3*(1+lambda_d* delta_s3[0][0]));
    LOGGER.d("c= %f",c);
    float age = (a+b+c)*101;
    LOGGER.d("age = %f",age);
    return age;
  }
  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (tfLite != null) {
      tfLite.close();
      tfLite = null;
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    if (tfLite != null) {
      tfLiteOptions.setNumThreads(numThreads);
      recreateInterpreter();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) {
      tfLiteOptions.setUseNNAPI(true);
      recreateInterpreter();
    }
  }

  private void recreateInterpreter() {
    tfLite.close();
    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
  }

  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation,int width ,int height) {
    // Loads bitmap into a TensorImage.
    inputImageBuffer.load(bitmap);

    // Creates processor for the TensorImage.
    int targetHeight = height;
    int tragetWidth = width;
    int cropSize = min(bitmap.getWidth(), bitmap.getHeight());
//    int numRotation = sensorOrientation / 90;
    // TODO(b/143564309): Fuse ops inside ImageProcessor.
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
//            .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                    // TODO(b/169379396): investigate the impact of the resize algorithm on accuracy.
                    // To get the same inference results as lib_task_api, which is built on top of the Task
                    // Library, use ResizeMethod.BILINEAR.
                    .add(new ResizeOp(targetHeight, tragetWidth, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
//            .add(new Rot90Op(numRotation))
//            .add(getPreprocessNormalizeOp())
                    .build();
    return imageProcessor.process(inputImageBuffer);
  }

  public static float byteToFloat(byte[] arr){
    int accum = 0;
    accum = accum|(arr[0] & 0xff) << 0 ;
    accum = accum|(arr[1] & 0xff) << 8 ;
    accum = accum|(arr[2] & 0xff) << 16 ;
    accum = accum|(arr[3] & 0xff) << 24 ;
    return Float.intBitsToFloat(accum);
  }
  public static float byteArrayToFloat(byte[] arr,int i){
    int accum = 0;
    accum = accum|(arr[i] & 0xff) << 0 ;
    accum = accum|(arr[i+1] & 0xff) << 8 ;
    accum = accum|(arr[i+2] & 0xff) << 16 ;
    accum = accum|(arr[i+3] & 0xff) << 24 ;
    return Float.intBitsToFloat(accum);
  }

  public static Object[] convertRGBToBGR(Object[] inputArray) {
    // 获取图像数据缓冲区
    ByteBuffer rgbBuffer = (ByteBuffer) inputArray[0];

    // 创建新的缓冲区用于存储 BGR 数据
    ByteBuffer bgrBuffer = ByteBuffer.allocate(rgbBuffer.capacity());

    for (int i = 0; i < rgbBuffer.capacity(); i += 3) {
      // 获取 RGB 值
      byte r = rgbBuffer.get(i);
      byte g = rgbBuffer.get(i + 1);
      byte b = rgbBuffer.get(i + 2);

      // 将 RGB 转换为 BGR
      bgrBuffer.put(b);
      bgrBuffer.put(g);
      bgrBuffer.put(r);
    }

    // 重置缓冲区位置
    bgrBuffer.flip();

    // 返回新的 BGR 数据缓冲区
    return new Object[]{bgrBuffer};
  }


  public float retina_overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1;
    float l2 = x2;
    float left = max(l1,l2);
    float r1 = x1 + w1;
    float r2 = x2 + w2;
    float right = min(r1,r2);
    return right - left;
  }

  public float retina_box_intersection(box a, box b){
    float area =0;
    float w = retina_overlap(a.x, a.w, b.x, b.w);
    float h = retina_overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0){
      return 0;
    }
    area = w * h;
    return area;
  }

  public float retina_box_union(box a, box b){
    float i = retina_box_intersection(a,b);
    float u = a.w * a.h + b.w * b.h -i;
    return u;
  }

  public float retina_box_iou(box a, box b){
    return retina_box_intersection(a,b)/retina_box_union(a,b);
  }

  public int retina_nms_comparator(sortable_bbox_retina a, sortable_bbox_retina b){
    float diff = a.probs - b.probs;
    if(diff < 0) return 1;
    else if(diff >0) return -1;
    return 0;
  }

  public void do_global_sort(box[] boxe1, box[] boxe2, float[][] prob1, float[][] prob2, int len1, int len2, float thresh){
    int i,j;
    for(i = 0; i < len1 ;++i){
      if(prob1[i][0] > thresh){
        for(j = 0;j < len2;j++){
          if(prob2[j][0] > thresh){
            if(retina_box_iou(boxe1[i], boxe2[j]) > 0.1){
              if(prob2[j][0] > prob1[i][0] ){
                prob1[i][0] = 0;
              }
              else{
                prob2[j][0] = 0;
              }
            }
          }
        }
      }
    }
  }

  public void retina_do_nms_sort(box[] boxes, float[][] probs, int total, float thresh){ //NMS未消去同样的框
    int i,j;
    int x=0,y=0;
    sortable_bbox_retina[] s = new sortable_bbox_retina[total];//类数组的赋值,需要给一个实例类
    for(i = 0;i < total;i++){
      sortable_bbox_retina si = new sortable_bbox_retina(i,0,probs[i][0]);
      s[i] = si;
    }

    //排序
    Arrays.sort(s, 0, total,new SortByProb());

    for(i = 0;i < total; i++){
      x = s[i].index;
      if(probs[x][0]>= 0.01){
        for(j = i+1; j < total; j++){
          y = s[j].index;
          if(probs[y][0]>=0.01){
            if(retina_box_iou(boxes[x],boxes[y]) > thresh){
              probs[y][0] = 0.00f;
            }
          }
        }
      }
    }
  }

  public int retina_result(int num, float thresh, box[] boxes, float[][] probs, detBox[] pBox, int detNum){
    int i;
    int detect_num = detNum;
    for(i = 0;i < num; i++){
      float prob = probs[i][0];
      if(detect_num < g_detect_number)
      {
        if(prob > thresh)
        {
          float x,y,w,h;
          x = boxes[i].x / 640;
          y = boxes[i].y / 384;
          w = boxes[i].w / 640;
          h = boxes[i].h / 384;

          pBox[detect_num] = new detBox(x,y,w,h,prob);

          detect_num++;
        }
      }
    }
    return detect_num;
  }

  public int face_detect_postprocess(detBox[] pBox){
    int i = 0, j = 0, k = 0, x, y, m;
    int idx;
    int h32, w32, h16, w16, h8, w8;
    float ctrx, ctry, pred_ctrx, pred_ctry;
    float predw, predh;
    int valid_8 = 0, valid_16 = 0, valid_32 =0;
    int detNum = 0;
    //全局判断置信度，小于0.7的进行置0处理prob[i][j][0]为正确
    for(i = 0; i < 240; i++){
      for(j = 0; j < 2; j++){
        prob32[k][0] = prob_32[i][j][1];//prob32验证完毕
        if(prob32[k][0] < 0.7){
          prob32[k][0] = 0;
        }
        else{
          valid_32 =1;
        }
        k++;
      }
    }

    k = 0;
    for(i = 0; i < 960; i++){
      for(j = 0; j < 2; j++){
        prob16[k][0] = prob_16[i][j][1];
        if(prob16[k][0] < 0.7){
          prob16[k][0] = 0;
        }
        else{
          valid_16 =1;
        }
        k++;
      }
    }
    k = 0;
    for(i = 0; i < 3840; i++){
      for(j = 0; j < 2; j++){
        prob8[k][0] = prob_8[i][j][1];
        if(prob8[k][0] < 0.7){
          prob8[k][0] = 0;
        }
        else{
          valid_8 =1;
        }
        k++;
      }
    }
    int sum32 = 0;
    box[] pbox32 = new box[480];
    // 填充box32,box16,box8
    if(valid_32 == 1){
      for(y = 0; y < 12; y++){
        for(x = 0; x < 20; x++){
          for(idx = 0; idx < 2; idx++){
            if(prob32[40*y + 2*x + idx][0] <= 0.01) {
              box boxi = new box(0,0,0,0);
              pbox32[sum32] = boxi;
              sum32++;
            }
            else {
              if (idx == 0) h32 = w32 = 512;
              else h32 = w32 = 256;
              ctrx = (float) (32 * x + 7.5);
              ctry = (float) (32 * y + 7.5);
              pred_ctrx = bbox_32[y][x][4 * idx] + ctrx;
              pred_ctry = bbox_32[y][x][4 * idx + 1] + ctry;
              predw = (float) (exp((double) bbox_32[y][x][4 * idx + 2])) * w32;
              predh = (float) (exp((double) bbox_32[y][x][4 * idx + 3])) * h32;
              float xi = (float) (pred_ctrx - 0.5 * (predw - 1));
              float yi = (float) (pred_ctry - 0.5 * (predh - 1));
              box boxi = new box(xi, yi, predw, predh);
              pbox32[sum32] = boxi;
              sum32++;
            }
          }
        }
      }
    }

    int sum16 = 0;
    box[] pbox16 = new box[1920];
    if(valid_16 == 1){
      for(y = 0; y < 24; y++){
        for(x = 0; x < 40; x++){
          for(idx = 0; idx < 2; idx++){
            if(prob16[80*y + 2*x + idx][0] <= 0.01) {
              box boxi = new box(0,0,0,0);
              pbox16[sum16] = boxi;
              sum16++;
            }
            else {
              if (idx == 0) h16 = w16 = 128;
              else h16 = w16 = 64;
              ctrx = (float) (16 * x + 7.5);
              ctry = (float) (16 * y + 7.5);
              pred_ctrx = bbox_16[y][x][4 * idx] + ctrx;
              pred_ctry = bbox_16[y][x][4 * idx + 1] + ctry;

              predw = (float) exp((double) bbox_16[y][x][4 * idx + 2]) * w16;
              predh = (float) exp((double) bbox_16[y][x][4 * idx + 3]) * h16;
              float xi = (float) (pred_ctrx - (predw - 1) / 2);
              float yi = (float) (pred_ctry - (predh - 1) / 2);
              box boxi = new box(xi, yi, predw, predh);
              pbox16[sum16] = boxi;
              sum16++;
            }
          }
        }
      }
    }

    int sum8 = 0;
    box[] pbox8 = new box[7680];
    if(valid_8 == 1){
      for(y = 0; y < 48; y++){
        for(x = 0; x < 80; x++){
          for(idx = 0; idx < 2; idx++){
            if(prob8[160*y + 2*x + idx][0] <= 0.01) {
              box boxi = new box(0,0,0,0);
              pbox8[sum8] = boxi;
              sum8++;
            }
            else {
              if (idx == 0) h8 = w8 = 32;
              else h8 = w8 = 16;
              ctrx = (float) (8 * x + 7.5);
              ctry = (float) (8 * y + 7.5);
              pred_ctrx = bbox_8[y][x][4 * idx] + ctrx;
              pred_ctry = bbox_8[y][x][4 * idx + 1] + ctry;

              predw = (float) exp((double) bbox_8[y][x][4 * idx + 2]) * w8;
              predh = (float) exp((double) bbox_8[y][x][4 * idx + 3]) * h8;
              float xi = (float) (pred_ctrx - (predw - 1) / 2);
              float yi = (float) (pred_ctry - (predh - 1) / 2);
              box boxi = new box(xi, yi, predw, predh);
              pbox8[sum8] = boxi;
              sum8++;
            }
          }
        }
      }
    }

    // nms result
    if(valid_32 == 1){
      retina_do_nms_sort(pbox32,prob32,480,(float)0.1);
      if(valid_16 == 1){
        retina_do_nms_sort(pbox16,prob16,1920,(float)0.1);
        do_global_sort(pbox32,pbox16,prob32,prob16,480,1920,(float)0.7);
        if(valid_8 == 1){
          retina_do_nms_sort(pbox8,prob8,7680,(float)0.1);
          do_global_sort(pbox16,pbox8,prob16,prob8,1920,7680,(float)0.7);
          detNum = retina_result(480,(float)0.8,pbox32,prob32,pBox,detNum);
          detNum = retina_result(1920,(float)0.8,pbox16,prob16,pBox,detNum);
          detNum = retina_result(7680,(float)0.8,pbox8,prob8,pBox,detNum);
        }
        else{
          detNum = retina_result(480,(float)0.8,pbox32,prob32,pBox,detNum);
          detNum = retina_result(1920,(float)0.8,pbox16,prob16,pBox,detNum);
        }
      }
      else{
        detNum = retina_result(480,(float)0.8,pbox32,prob32,pBox,detNum);
      }
    }
    if(valid_16 == 1 && valid_32 == 0){
      retina_do_nms_sort(pbox16,prob16,1920,(float)0.1);
      if(valid_8 == 1){
        retina_do_nms_sort(pbox8,prob8,7680,(float)0.1);
        do_global_sort(pbox16,pbox8,prob16,prob8,1920,7680,(float)0.8);
        detNum = retina_result(1920,(float)0.8,pbox16,prob16,pBox,detNum);
        detNum = retina_result(7680,(float)0.8,pbox8,prob8,pBox,detNum);
      }
      else {
        detNum = retina_result(1920, (float) 0.8, pbox16, prob16, pBox,detNum);
      }
    }
    if (valid_8 == 1 && valid_16 == 0 && valid_32 == 0 )
    {
      retina_do_nms_sort(pbox8, prob8, 7680, (float)0.2);
      detNum = retina_result(7680, (float)0.4, pbox8, prob8, pBox,detNum);
    }
    return detNum;
  }
}
