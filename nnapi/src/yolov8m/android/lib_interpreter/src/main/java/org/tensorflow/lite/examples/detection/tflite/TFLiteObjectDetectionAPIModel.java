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

import static java.lang.Math.max;
import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;
import android.util.Size;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.security.AllPermission;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import java.io.InputStream;
import android.os.Environment;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

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

    // Only return this many results.
    private static final int NUM_DETECTIONS = 10;
    // Float model
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;
    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private static final boolean IS_INT8 = true;
    private static final Size INPNUT_SIZE = new Size(320, 320);
    private static final int[] OUTPUT_SIZE = new int[]{1, 84, 3549};
    private final float DETECT_THRESHOLD = 0.25f;
    private final float IOU_THRESHOLD = 0.45f;
    private final float IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f;

    static MetadataExtractor.QuantizationParams input5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);
    MetadataExtractor.QuantizationParams output5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.006305381190031767f, 5);
    private boolean isModelQuantized;
    // Config values.
    private int inputSize;
    // Pre-allocated buffers.
    private List<String> labels = new ArrayList<>();
    private int[] intValues;
    /*
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
    private float[] numDetections;*/
    private TensorBuffer outputBuffer;
    private TensorImage inputImage;

    private ByteBuffer imgData;

    private MappedByteBuffer tfLiteModel;
    private Interpreter.Options tfLiteOptions;
    private Interpreter tfLite;

    private TensorImage inputImageBuffer;

    /**
     * output process object
     */
    private static float[][][] big_output = new float[52][52][144];
    private static float[][][] medium_output = new float[26][26][144];
    private static float[][][] small_output = new float[13][13][144];

    byte[] input0 = new byte[389376];//960 u8
    byte[] input1 = new byte[97344];
    byte[] input2 = new byte[24336];//3840 u8
    private static float[] float_output0 = new float[389376];
    private static float[] float_output1= new float[97344];
    private static float[] float_output2 = new float[24336];

    byte[] byte_list = new byte[298116];
    float[][] yolov8m_output = new float[84][3549];

    private final int g_detect_number = 230;
    detBox[] pBox = new detBox[230];
    int detNum = 0;
    int yolov3_box_num_after_filter = 0;
    int box1 = 169;
    box[] boxes_1 = new box[box1];
    box[] boxes_2 = new box[box1*4];
    box[] boxes_3 = new box[box1*16];
    box[] boxes = new box[box1*21];

    float[][] probs_1 = new float[box1][81];
    float[][] probs_2 = new float[box1*4][81];
    float[][] probs_3 = new float[box1*16][81];

    int loop =0;
    static final String[] coco_names = {
        "person","bicycle","car","motorbike","aeroplane","bus","train",
                "truck","boat","traffic light","fire hydrant","stop sign","parking meter",
                "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
                "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
                "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
                "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
                "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
                "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
                "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
                "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };

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
        float prob_obj;
        box(float x , float y, float w, float h,float prob_obj){
            this.x = x;
            this.y = y;
            this.w = w;
            this.h = h;
            this.prob_obj = prob_obj;
        }
    }
    public static class detBox{
        float x;
        float y;
        float w;
        float h;
        float score;
        int objectClass;
        detBox(float x , float y, float w, float h, float score , int objectClass){
            this.x = x;
            this.y = y;
            this.w = w;
            this.h = h;
            this.score = score;
            this.objectClass = objectClass;
        }
    }

    public static class Object_Detect_Out{
        int detNum;
        detBox[] pBox;
        public Object_Detect_Out(int detNum, detBox[] pBox){
            this.detNum = detNum;
            this.pBox = pBox;
        }
    }

    private TFLiteObjectDetectionAPIModel() {
    }

    /**
     * Memory-map the model file in Assets.
     */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param modelFilename The model file path relative to the assets folder
     * @param labelFilename The label file path relative to the assets folder
     * @param inputSize     The size of image input
     * @param isQuantized   Boolean representing model is quantized or not
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
        /*MetadataExtractor metadata = new MetadataExtractor(modelFile);
        try (BufferedReader br =
                     new BufferedReader(
                             new InputStreamReader(
                                     metadata.getAssociatedFile(labelFilename), Charset.defaultCharset()))) {
            String line;
            while ((line = br.readLine()) != null) {
                Log.w("TFOD", line);
                d.labels.add(line);
            }
        }*/
        d.labels = FileUtil.loadLabels(context, labelFilename);

        d.inputSize = inputSize;

        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(NUM_THREADS);
            options.setUseXNNPACK(true);
            options.setUseNNAPI(true);
            options.addDelegate(new NnApiDelegate());
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

//        float_output0_test = readFloatArrayFromAssets(context, "nnsdk_output_1.txt");
//        float_output1_test = readFloatArrayFromAssets(context, "nnsdk_output_0.txt");
//        float_output2_test = readFloatArrayFromAssets(context, "nnsdk_output_2.txt");
        /*d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());*/
//        d.intValues = new int[d.inputSize * d.inputSize];
//
//        d.inputImage = new TensorImage(DataType.UINT8);
//
//        d.outputBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.UINT8);
        /*d.outputBuffer = ByteBuffer.allocateDirect(1 * 6300 * 85 * numBytesPerChannel);
        d.outputBuffer.order(ByteOrder.nativeOrder());*/

        /*d.outputLocations = new float[1][NUM_DETECTIONS][4];
        d.outputClasses = new float[1][NUM_DETECTIONS];
        d.outputScores = new float[1][NUM_DETECTIONS];
        d.numDetections = new float[1];*/
        return d;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        long startTime_1 = System.currentTimeMillis();
        DataType imageDataType = tfLite.getInputTensor(0).dataType();
        inputImageBuffer = new TensorImage(imageDataType);
        inputImageBuffer = loadImage(bitmap, 0,416,416);
        long endTime_1 = System.currentTimeMillis();

        Object[] inputArray = {inputImageBuffer.getBuffer()}; // byteBuffer
        //test 0227
//        ByteBuffer test = readBinFile("/data/vendor/nn/input_tensor.bin");
//        Object[] inputArray = {test}; // byteBuffer

        int[] OutputShape;
        DataType OutputDataType;
        ByteBuffer in;
        Map<Integer,Object> outputs = new HashMap<>();
        for(int i =0;i<tfLite.getOutputTensorCount();i++){
            OutputShape = tfLite.getOutputTensor(i).shape();
            OutputDataType = tfLite.getOutputTensor(i).dataType();
            in= TensorBuffer.createFixedSize(OutputShape,OutputDataType).getBuffer();
            outputs.put(i,in);
        }
        Trace.endSection();
        // Run the inference call.
        long invoke_startTime = System.currentTimeMillis();
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputs);
        Trace.endSection();
        long invoke_endTime = System.currentTimeMillis();
        Log.i("dwh", "Invoke Execution time: " + (invoke_endTime - invoke_startTime) + " ms");

        ByteBuffer x0;
        x0 = (ByteBuffer) outputs.get(0);
        x0.flip();
        x0.get(byte_list,0,298116);

        float scale = 0.004429318942129612f;
        int zp = 5;
        int index =0;
        for (int i=0;i<84;i++)
        {
            for (int j =0;j<3549;j++)
            {
                int input_buffer = byte_list[index] & 0xFF;
                yolov8m_output[i][j] = (input_buffer-zp) * scale;
                index++;
            }
        }

//        Log.i("dwh", "Save image ");
//        saveBitmapToJpg(bitmap, "bitmap_image");
//        saveTensorImageToJpg(inputImageBuffer,"input");
//        saveByteBufferToBin(inputImageBuffer.getBuffer(),"input_tensor.bin");
//        saveByteBufferToBin(x0,"output_tensor.bin");
//        saveFloatArrayToTxt(yolov8m_output, "yolov8m_output.txt");

        int filter_num =0 ;

        for (int i = 0; i < OUTPUT_SIZE[2]; i++) {
            float x = yolov8m_output[0][i];
            float y = yolov8m_output[1][i];
            float w = yolov8m_output[2][i];
            float h = yolov8m_output[3][i];

            float[] classScores = new float[80];
            for ( int j =0;j<80;j++)
            {
                classScores[j] = (yolov8m_output[j+4][i] > 0.55f) ? yolov8m_output[j+4][i] : 0.00f; // 小于0.7的全部置0
//                classScores[j] = yolov8m_output[j+4][i]; // 小于0.7的全部置0
            }

            int labelId = 0;
            float maxLabelScores = 0.f;
            for (int j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j];
                    labelId = j;
                }
            } // 找到最大label的ID和置信度

            if (maxLabelScores > 0.55f)
            {
                pBox[filter_num] = new detBox(x,y,w,h,maxLabelScores,labelId);
//                Log.i("dwh", "maxLabelScores: " + maxLabelScores + " , labelId : " + coco_names[labelId] + " cooperation x = " + x + " y = " + y+ " w = "  +  w + " output_yolo.get(i).h = " + h );
                filter_num = filter_num + 1;
            }
        }
        //IOU Filter here
        // 进行排序（按照 prob 降序）
        Arrays.sort(pBox, 0, filter_num, new Comparator<detBox>() {
            @Override
            public int compare(detBox b1, detBox b2) {
                return Float.compare(b2.score, b1.score); // 降序
            }
        });

        Log.i("dwh", "filter_num: " + filter_num);

        List<detBox> output_yolo = nonMaximumSuppression(pBox,filter_num,0.35f);

        Log.i("dwh", "After filter_num: " + output_yolo.size());
        ArrayList<Recognition> allRecognitions = new ArrayList<>();
        for (int i=0;i<output_yolo.size();i++)
        {
//            Log.i("dwh", "maxLabelScores: " + output_yolo.get(i).score + " , labelId : " + coco_names[output_yolo.get(i).objectClass] + " cooperation x = " + output_yolo.get(i).x + " y = " + output_yolo.get(i).y + " w = "  + output_yolo.get(i).w + " output_yolo.get(i).h = " +output_yolo.get(i).h);
            float left = Math.max(0, 416 * (output_yolo.get(i).x - output_yolo.get(i).w / 2));
            float top = Math.max(0, 416 * (output_yolo.get(i).y - output_yolo.get(i).h / 2));
            float right = Math.min(416, 416 * (output_yolo.get(i).x + output_yolo.get(i).w / 2));
            float bottom = Math.min(416, 416 * (output_yolo.get(i).y + output_yolo.get(i).h / 2));
//            float left = (pBox[i].x - pBox[i].w/2);
//            float top = (pBox[i].y - pBox[i].h/2);
//            float right = (pBox[i].x + pBox[i].w/2);
//            float bottom = (pBox[i].y + pBox[i].h/2);
//            Log.i("dwh", "maxLabelScores: " + output_yolo.get(i).score + " , labelId : " + coco_names[output_yolo.get(i).objectClass] + "cooperation left = " + left + " right = " + right + " top = "  + top + " bottom = " +bottom);
            Recognition r = new Recognition(
                    output_yolo.get(i).objectClass,
                    coco_names[output_yolo.get(i).objectClass],
                    output_yolo.get(i).score,
                    new RectF(left,top,right,bottom));
            allRecognitions.add(r);
        }
        //test
//        android.os.Process.killProcess(android.os.Process.myPid()); // 杀死当前进程
//        System.exit(0); // 退出 JVM
        return allRecognitions;
    }

    public List<Recognition> Test_for_post_process(float[] yolov8m_output_origin) {

//        for(int i=0;i<10;i++)
//        {
//            Log.i("dwh", "yolov8m_output_origin: " + yolov8m_output_origin[i]);
//        }
        for (int i=0;i<84;i++)
        {
            for (int j =0;j<3549;j++)
            {
                yolov8m_output[i][j] = yolov8m_output_origin[ j + 3549 * i];
            }
        }

        int filter_num =0 ;

        for (int i = 0; i < OUTPUT_SIZE[2]; i++) {
            float x = yolov8m_output[0][i];
            float y = yolov8m_output[1][i];
            float w = yolov8m_output[2][i];
            float h = yolov8m_output[3][i];

            float[] classScores = new float[80];
            for ( int j =0;j<80;j++)
            {
                classScores[j] = (yolov8m_output[j+4][i] > 0.45f) ? yolov8m_output[j+4][i] : 0.00f; // 小于0.7的全部置0
//                classScores[j] = yolov8m_output[j+4][i]; // 小于0.7的全部置0
            }

            int labelId = 0;
            float maxLabelScores = 0.f;
            for (int j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j];
                    labelId = j;
                }
            } // 找到最大label的ID和置信度


            if (maxLabelScores > 0.45f && filter_num < g_detect_number)
            {
                pBox[filter_num] = new detBox(x,y,w,h,maxLabelScores,labelId);
                filter_num = filter_num + 1;
            }
        }
        //IOU Filter here
        // 进行排序（按照 prob 降序）
        Arrays.sort(pBox, 0, filter_num, new Comparator<detBox>() {
            @Override
            public int compare(detBox b1, detBox b2) {
                return Float.compare(b2.score, b1.score); // 降序
            }
        });
//        for (int i=0;i<filter_num;i++)
//        {
//            Log.i("dwh", "maxLabelScores: " + pBox[i].score + " , labelId : " + pBox[i].objectClass);
//        }//排序
        List<detBox> output_yolo = nonMaximumSuppression(pBox,filter_num,0.55f);


        ArrayList<Recognition> allRecognitions = new ArrayList<>();
        for (int i=0;i<output_yolo.size();i++)
        {
            Log.i("dwh", "maxLabelScores: " + output_yolo.get(i).score + " , labelId : " + coco_names[output_yolo.get(i).objectClass] + " cooperation x = " + output_yolo.get(i).x + " y = " + output_yolo.get(i).y + " w = "  + output_yolo.get(i).w + " output_yolo.get(i).h = " +output_yolo.get(i).h);
            float left = Math.max(0, 416 * (output_yolo.get(i).x - output_yolo.get(i).w / 2));
            float top = Math.max(0, 416 * (output_yolo.get(i).y - output_yolo.get(i).h / 2));
            float right = Math.min(416, 416 * (output_yolo.get(i).x + output_yolo.get(i).w / 2));
            float bottom = Math.min(416, 416 * (output_yolo.get(i).y + output_yolo.get(i).h / 2));

            Recognition r = new Recognition(
                    output_yolo.get(i).objectClass,
                    coco_names[output_yolo.get(i).objectClass],
                    output_yolo.get(i).score,
                    new RectF(left,top,right,bottom));
            allRecognitions.add(r);
        }
//        Recognition r = new Recognition(
//                0,
//                "0000",
//                0.00f,
//                new RectF(0, 0, 0, 0));
//        allRecognitions.add(r);

        return allRecognitions;
    }

    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation,int width ,int height) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int targetHeight = height;
        int tragetWidth = width;
        int cropSize = Math.max(bitmap.getWidth(), bitmap.getHeight());
//    int numRotation = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        // TODO(b/169379396): investigate the impact of the resize algorithm on accuracy.
                        // To get the same inference results as lib_task_api, which is built on top of the Task
                        // Library, use ResizeMethod.BILINEAR.
                        .add(new ResizeOp(targetHeight, tragetWidth, ResizeOp.ResizeMethod.BILINEAR))// NEAREST_NEIGHBOR
//            .add(new Rot90Op(numRotation))
//            .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    public static void saveFloatArrayToTxt(float[][] data, String fileName) {
        String path = "/data/vendor/nn/" + fileName;
        File file = new File(path);

        try (FileWriter writer = new FileWriter(file)) {
            for (float[] row : data) {
                for (float value : row) {
                    writer.write(value + "\n");
                }
//                writer.write("\n");
            }
            writer.flush();
            System.out.println("Float[][] 文件保存成功: " + path);
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("写入失败，可能是权限问题: " + e.getMessage());
        }
    }

    // 直接保存 ByteBuffer 到 /data/vendor/nn/
    public static void saveByteBufferToBin(ByteBuffer buffer, String fileName) {
        String path = "/data/vendor/nn/" + fileName;
        File file = new File(path);

        try (FileOutputStream fos = new FileOutputStream(file)) {
            // 确保 buffer 位置正确
            buffer.rewind();

            // 创建 byte 数组
            byte[] bytes = new byte[buffer.remaining()];
            buffer.get(bytes);

            // 写入文件
            fos.write(bytes);
            fos.flush();
            System.out.println("ByteBuffer 文件保存成功: " + path);

        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("写入失败，可能是权限问题: " + e.getMessage());
        }
    }

    public static void saveBitmapToJpg(Bitmap bitmap, String fileName) {
        System.out.println("saveBitmapToJpg " );
        // 目标路径（确保该路径有写入权限）
        String dirPath = "/data/vendor/nn/";
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs(); // 创建目录
        }

        // 目标文件
        File file = new File(dir, fileName + ".jpg");

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(file);
            // 将 Bitmap 压缩为 JPG 格式，质量 90（0-100）
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos);
            fos.flush();
            System.out.println("Bitmap 保存成功: " + file.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("保存失败: " + e.getMessage());
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static void saveTensorImageToJpg(TensorImage tensorImage, String fileName) {
        // 1️⃣ 获取 Bitmap
        Bitmap bitmap = tensorImage.getBitmap();

        // 2️⃣ 目标路径
        String dirPath = "/data/vendor/nn/";
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs(); // 创建目录（需要 root 权限）
        }

        // 3️⃣ 目标文件
        File file = new File(dir, fileName + ".jpg");

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(file);
            // 4️⃣ 将 Bitmap 压缩为 JPG 格式，质量 90（0-100）
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos);
            fos.flush();
            System.out.println("Tensor Image保存成功: " + file.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("保存失败: " + e.getMessage());
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static ByteBuffer readBinFile(String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            System.out.println("文件不存在: " + filePath);
            return null;
        }

        FileInputStream fis = null;
        try {
            // 1️⃣ 读取文件到 byte[]
            byte[] data = new byte[(int) file.length()];
            fis = new FileInputStream(file);
            fis.read(data);

            // 2️⃣ 转换为 ByteBuffer（默认使用 LITTLE_ENDIAN）
            ByteBuffer buffer = ByteBuffer.wrap(data);
            buffer.order(ByteOrder.LITTLE_ENDIAN);  // 设置字节顺序
            return buffer;

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        } finally {
            try {
                if (fis != null) fis.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private float computeIoU(detBox box1, detBox box2) {
        float x1 = Math.max(box1.x - box1.w/2, box2.x - box2.w/2);
        float y1 = Math.max(box1.y - box1.h/2, box2.y - box2.h/2);
        float x2 = Math.min(box1.x + box1.w/2, box2.x + box2.w/2);
        float y2 = Math.min(box1.y + box1.h/2, box2.y + box2.h/2);

        float interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float box1Area = box1.w * box1.h;
        float box2Area = box2.w * box2.h;

        return interArea / (box1Area + box2Area - interArea);
    }

    private List<detBox> nonMaximumSuppression(detBox[] pBox, int filter_num, float iouThreshold) {
        List<detBox> result = new ArrayList<>();
        boolean[] suppressed = new boolean[filter_num]; // 标记被抑制的框

        for (int i = 0; i < filter_num; i++) {
            if (suppressed[i]) continue; // 如果当前框已经被抑制，跳过
            result.add(pBox[i]); // 保留当前最高置信度的框

            for (int j = i + 1; j < filter_num; j++) {
                if (computeIoU(pBox[i], pBox[j]) > iouThreshold) {
//                    Log.i("dwh", "Iou: " + computeIoU(pBox[i], pBox[j]));
                    suppressed[j] = true; // 抑制 IoU 过高的框
                }
            }
        }
        return result;
    }

    /**
     * output 从byte转float
     */
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

    /**
     * 拼接float数组
     */
    public static float[][] concat2DArrays(float[][] arr1, float[][] arr2, float[][] arr3)
    {
        int totalRows = arr1.length + arr2.length + arr3.length;
        float[][] result = new float[totalRows][81]; // 列数固定为 81

        int rowIndex = 0;

        // 复制第一个数组
        System.arraycopy(arr1, 0, result, rowIndex, arr1.length);
        rowIndex += arr1.length;

        // 复制第二个数组
        System.arraycopy(arr2, 0, result, rowIndex, arr2.length);
        rowIndex += arr2.length;

        // 复制第三个数组
        System.arraycopy(arr3, 0, result, rowIndex, arr3.length);

        return result;
    }

    /**
     * 拼接box数组
     */
    public static box[] concatBoxArrays(box[] arr1, box[] arr2, box[] arr3) {
        int totalLength = arr1.length + arr2.length + arr3.length;
        box[] result = new box[totalLength];

        System.arraycopy(arr1, 0, result, 0, arr1.length);
        System.arraycopy(arr2, 0, result, arr1.length, arr2.length);
        System.arraycopy(arr3, 0, result, arr1.length + arr2.length, arr3.length);

        return result;
    }


    /**
     * logistic的实现
     */
    float logistic_activate(float x)
    {
        return (float) (1.0/(1.0 + Math.exp(-x)));
    }
    /**
     * 求重叠的长度
     */
    public float retina_overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1;
        float l2 = x2;
        float left = max(l1,l2);
        float r1 = x1 + w1;
        float r2 = x2 + w2;
        float right = min(r1,r2);
        return right - left;
    }

    /**
     * 获取对应的box信息
     */
    public box get_region_box(float[] x, int index, int i, int j, int w, int h,float max_prob)
    {
        float prob_obj = (max_prob > 0.35f) ? max_prob : 0.00f;

        if (prob_obj == 0.00f)
        {
            box b = new box(0.00f,0.00f,0.00f,0.00f,0.00f);
            return b;
        }
        else
        {
            float[] tmp =new float[4];
            for (int k = 0; k < 4; k++)
            {
                float sum = 0;
                for(int m = 0; m < 16; m++)
                {
                    x[index + k * 16 + m] = (float) Math.exp(x[index + k * 16 + m]);
                    sum += x[index + k * 16 + m];
                }
//                Log.i("dwh", "get_region_box sum =  " +sum );
                for(int m = 0; m < 16; m++)
                {
                    tmp[k] += m * x[index + k * 16 + m] / sum;
                }
            }
            float x_ = (float) ((j + 0.5 - tmp[0]) / w);
            float y_ = (float) ((i + 0.5 - tmp[1]) / h);
            float w_ = (float) ((j + 0.5 + tmp[2]) / w);
            float h_ = (float) ((i + 0.5 + tmp[3]) / h);
            w_ = w_- x_;
            h_ = h_ - y_;
            x_ = x_ + w_ / 2;
            y_ = y_ + h_ / 2;
            box b = new box(x_,y_,w_,h_,prob_obj);
            return b;
        }
//        box b = new box(0,0,0,0,0);
//        b.prob_obj = (max_prob > 0.35f) ? max_prob : 0;
//
//        if (b.prob_obj ==0)
//        {
//            b.w = 0;
//            b.h = 0;
//            b.x = 0;
//            b.y = 0;
//        }
//        else
//        {
//            float[] tmp =new float[4];
//            for (int k = 0; k < 4; k++)
//            {
//                float sum = 0;
//                for(int m = 0; m < 16; m++)
//                {
//                    x[index + k * 16 + m] = (float) Math.exp(x[index + k * 16 + m]);
//                    sum += x[index + k * 16 + m];
//                }
////                Log.i("dwh", "get_region_box sum =  " +sum );
//                for(int m = 0; m < 16; m++)
//                {
//                    tmp[k] += m * x[index + k * 16 + m] / sum;
//                }
//            }
////            Log.i("dwh", "get_region_box tmp[0] =  " +tmp[0] + " tmp[1] = " + tmp[1] + " tmp[2] = "+ tmp[2] +" tmp[3] =  " +tmp[3] );
//            b.x = (float) ((j + 0.5 - tmp[0]) / w);
//            b.y = (float) ((i + 0.5 - tmp[1]) / h);
//            b.w = (float) ((j + 0.5 + tmp[2]) / w);
//            b.h = (float) ((i + 0.5 + tmp[3]) / h);
//            b.w = b.w - b.x;
//            b.h = b.h - b.y;
//            b.x = b.x + b.w / 2;
//            b.y = b.y + b.h / 2;
//        }
//        Log.i("dwh", "get_region_box box x =  " +b.x + " y = " + b.y + " w = "+ b.w +" h = " +b.h );
    }

    /**
     * 求重叠框的面积
     */
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

    /**
     * 求两个重叠框覆盖总共的面积
     */
    public float retina_box_union(box a, box b){
        float i = retina_box_intersection(a,b);
        float u = a.w * a.h + b.w * b.h -i;
        return u;
    }

    /**
     * 求最大值的索引
     */
    public int max_index(float[] a, int n)
    {
        int i, max_i = 0;
        float max = a[0];

        if (n <= 0)
            return -1;

        for (i = 1; i < n; ++i)
        {
            if (a[i] > max)
            {
                max = a[i];
                max_i = i;
            }
        }
        return max_i;
    }
    /**
     * 求两个框IOU比
     */
    public float retina_box_iou(box a, box b){
        return retina_box_intersection(a,b)/retina_box_union(a,b);
    }

    public int retina_nms_comparator(sortable_bbox_retina a, sortable_bbox_retina b){
        float diff = a.probs - b.probs;
        if(diff < 0) return 1;
        else if(diff >0) return -1;
        return 0;
    }

    /**
     * 做全局的sort
     */
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

    /**
     * 做nms的sort
     */
    public void retina_do_nms_sort(box[] boxes, float[][] probs, int total,int classes, float thresh){ //NMS未消去同样的框
        int i,j,k;
        int x=0,y=0;
        sortable_bbox_retina[] s = new sortable_bbox_retina[total];//类数组的赋值,需要给一个实例类
        for(i = 0;i < total;i++){
            sortable_bbox_retina si = new sortable_bbox_retina(i,0,probs[i][0]);
            s[i] = si;
        }
        for (k=0;k<classes;k++)
        {
            for (i = 0; i < total; ++i)
            {
                s[i].classId = k;
            }
            //排序
            Arrays.sort(s, 0, total,new SortByProb());

            for(i = 0;i < total; i++){
                x = s[i].index;
                if(probs[x][k]>= 0.01){
                    for(j = i+1; j < total; j++){
                        y = s[j].index;
                        if(probs[y][k]>=0.01){
                            if(retina_box_iou(boxes[x],boxes[y]) > thresh){
                                probs[y][k] = 0.00f;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * 获取最终的结果
     */
    public int retina_result(int num, float thresh, box[] boxes, float[][] probs,int classes, detBox[] pBox, int detNum){
        int i;
        int detect_num = detNum;
        for(i = 0;i < num; i++){
            int classId = max_index(probs[i], classes);
            float prob = probs[i][classId];
            if(detect_num < g_detect_number)
            {
                if(prob > thresh)
                {
//                    Log.i("dwh", "Test prob =  " + prob);
                    float x,y,w,h;
                    x = boxes[i].x ;
                    y = boxes[i].y ;
                    w = boxes[i].w ;
                    h = boxes[i].h ;

                    pBox[detect_num] = new detBox(x,y,w,h,prob,classId);

                    detect_num = detect_num + 1;
                }
            }
        }
        return detect_num;
    }

    public int yolo_v3_post_process_onescale(float[] predictions , int[] input_size,box[] boxes,float[][] probs,float threshold_in)
    {
        int i,j,k,index;
        int num_class = 80;
        int coords = 64;
        int bb_size = coords + num_class;
        int modelWidth = input_size[0];
        int modelHeight = input_size[1];
        float threshold = threshold_in;
        float max_prob;
//        Log.i("dwh", "Before yolov3_box_num_after_filter: " + yolov3_box_num_after_filter);
        for (i = 0; i < modelHeight; ++i)
        {
            for (j = 0; j < modelWidth; ++j)
            {
                index = i * modelHeight + j;
                max_prob = 0;

                for (k = 0; k < num_class; ++k)
                {
                    int index_p = index * bb_size +k;
                    float prob = logistic_activate(predictions[index_p]);
                    probs[index][k] = (prob > threshold) ? prob : 0;
                    max_prob = (prob > threshold && prob > max_prob) ? prob : max_prob;
//                    probs[index][k] = Math.max(prob, threshold) == threshold ? 0 : prob;
//                    max_prob = Math.max(max_prob, Math.max(prob, threshold) == threshold ? max_prob : prob);
                }
//                boxes[index].prob_obj = (max_prob > threshold) ? max_prob : 0;
//                boxes[index].prob_obj = Math.max(max_prob, threshold) == threshold ? 0 : max_prob;;

                if (max_prob > threshold)
                {
                    int box_index = index * bb_size + num_class;
                    boxes[index] = get_region_box(predictions, box_index, i, j, modelWidth, modelHeight,max_prob);
                    yolov3_box_num_after_filter = yolov3_box_num_after_filter + 1;
                }
                else
                {
                    boxes[index] = new box(0,0,0,0,0);
                }
            }
        }
//        Log.i("dwh", "After yolov3_box_num_after_filter: " + yolov3_box_num_after_filter);
        return 0;

    }

    public void yolo_process(float[] big_output, float[] medium_output,float[] small_output)
    {
        int nn_width,nn_height, nn_channel;
        nn_width = 416;
        nn_height = 416;
        nn_channel = 3;
        int[] size={nn_width/32, nn_height/32,80 + 64};
        yolov3_box_num_after_filter = 0;
        detNum =0;

        int j, k, index;
        int num_class = 80;
        float threshold = (float) 0.5;
        float iou_threshold = (float) 0.45;

        int[] size2 = {size[0]*2,size[1]*2,size[2]};
        int[] size4 = {size[0]*4,size[1]*4,size[2]};
        int len1 = size[0]*size[1]*size[2];
        int box1 = len1/(num_class+64);

        long concat_startTime = System.currentTimeMillis();
        yolo_v3_post_process_onescale(small_output, size, boxes_1, probs_1, threshold);
        yolo_v3_post_process_onescale(medium_output, size2, boxes_2, probs_2, threshold);
        yolo_v3_post_process_onescale(big_output, size4, boxes_3, probs_3, threshold);
        long concat_endTime = System.currentTimeMillis();
        Log.i("dwh", "yolov3_onescale time: " + (concat_endTime - concat_startTime) + " ms");

        float[][] probs = concat2DArrays(probs_1, probs_2, probs_3);
        boxes = concatBoxArrays(boxes_1,boxes_2,boxes_3);

        box[] tmp_boxes = new box[yolov3_box_num_after_filter];
//        for (int i = 0; i < yolov3_box_num_after_filter; i++) {
//            tmp_boxes[i] = new box(0,0,0,0,0);  // 初始化每个 Box 对象
//        }
        float[][] tmp_probs = new float[yolov3_box_num_after_filter][81];

        k =0;
        for(index =0;index<box1 *(1+4+16);index++)
        {
            if(boxes[index].prob_obj > 0)
            {
                tmp_probs[k] = probs[index];
                tmp_boxes[k] = boxes[index];
                k=k+1;
            }
        }
//        Log.i("dwh", "tmp_boxes size: " + k);
//        for(int i =0;i<10;i++)
//        {
//            Log.i("dwh", "tmp_boxes.prob_obj = " + tmp_boxes[i].prob_obj + " x =" + tmp_boxes[i].x + "y = "+ tmp_boxes[i].y );
//        }
        retina_do_nms_sort(tmp_boxes, tmp_probs, yolov3_box_num_after_filter, num_class, iou_threshold);
//        yolov3_result(yolov3_box_num_after_filter, threshold, tmp_boxes, tmp_probs, num_class, dectout);
        detNum = retina_result(yolov3_box_num_after_filter, threshold, tmp_boxes, tmp_probs,num_class, pBox, detNum);
//        Log.i("dwh", "detNum: " + detNum);
//        for (int i =0;i<detNum;i++)
//        {
//            Log.i("dwh", "pBox[i].x = " + pBox[i].x +" y = " + pBox[i].y + " w = " + pBox[i].w + " h = "+ pBox[i].h + " Class =" + pBox[i].objectClass + "prob = " + pBox[i].score);
//        }

    }

    public static float[] readFloatArrayFromAssets(Context context, String fileName) {
        ArrayList<Float> floatList = new ArrayList<>();
        try {
            // 1. 获取 AssetManager
            AssetManager assetManager = context.getAssets();
            InputStream inputStream = assetManager.open(fileName);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

            // 2. 读取文本文件
            String line;
            while ((line = reader.readLine()) != null) {
                // 3. 按空格或逗号拆分
                String[] values = line.trim().split("[ ,]+");
                for (String value : values) {
                    try {
                        floatList.add(Float.parseFloat(value));
                    } catch (NumberFormatException e) {
                        Log.e("AssetFileReader", "Invalid float value: " + value);
                    }
                }
            }
            reader.close();
            inputStream.close();
        } catch (Exception e) {
            Log.e("AssetFileReader", "Error reading asset file", e);
        }

        // 4. 转换为 float 数组
        float[] floatArray = new float[floatList.size()];
        for (int i = 0; i < floatList.size(); i++) {
            floatArray[i] = floatList.get(i);
        }

        return floatArray;
    }

    // 保存 Bitmap 为 JPG 文件
    public static boolean saveBitmapToJPG(Bitmap bitmap, String fileName) {
        // 获取外部存储目录
        File storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        File file = new File(storageDir, fileName + ".jpg");

        FileOutputStream out = null;
        try {
            out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out); // 质量 90
            out.flush();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        } finally {
            try {
                if (out != null) out.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    @Override
    public void enableStatLogging(final boolean logStats) {
    }

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
}
