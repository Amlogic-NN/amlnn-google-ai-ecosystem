/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.*;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "det_retinaface_int8_uint8.tflite";
  private static final String gender_model_file = "sdk_gender_int8_uint8.tflite";
  private static final String age_model_file = "sdk_age_quant.acuity.tflite";

  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(1280, 720);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Detector detector;
  private Detector gender_detector;
  private Detector age_detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Bitmap genderBitmap = null;
  private Bitmap rgbCopyBitmap = null;
  private Bitmap ageBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;



    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              this,
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      gender_detector =
              TFLiteObjectDetectionAPIModel.create(
                      this,
                      gender_model_file,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
      age_detector =
              TFLiteObjectDetectionAPIModel.create(
                      this,
                      age_model_file,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      LOGGER.d("face detector create");
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing Detector!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }


    previewWidth = size.getWidth();
    previewHeight = size.getHeight();


    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(640, 384, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            640, 384,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");
    LOGGER.i("-------------------"+String.valueOf(previewWidth));
    LOGGER.i("-------------------"+String.valueOf(previewHeight));
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }
    rgbCopyBitmap = rgbFrameBitmap.copy(Config.ARGB_8888,true);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
//            saveImageToGallery(rgbFrameBitmap);
            final List<Detector.Recognition> results = detector.recognizeImage(rgbFrameBitmap);
//            final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap);
            LOGGER.d("Detect.Det_Num : %s",results.size());
            LOGGER.v("Detect: %s", results);
            LOGGER.d("Detect: %s", results);

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
//            final Canvas canvas = new Canvas(rgbFrameBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Detector.Recognition> mappedRecognitions =
                new ArrayList<Detector.Recognition>();

            for (final Detector.Recognition result : results) {
              final RectF location = result.getLocation();
              LOGGER.i("-------------------"+String.valueOf(previewWidth));
              LOGGER.i("-------------------"+String.valueOf(previewHeight));
              LOGGER.i("-------------------"+String.valueOf(location.left));
              LOGGER.i("-------------------"+String.valueOf(location.right));
              LOGGER.i("-------------------"+String.valueOf(location.top));
              LOGGER.i("-------------------"+String.valueOf(location.bottom));

              genderBitmap = rgbCopyBitmap.copy(Config.ARGB_8888,true);
              int genderWidth = rgbCopyBitmap.getWidth();
              int genderHeight = rgbCopyBitmap.getHeight();
              float width = (location.right - location.left)*genderWidth;
              float height= (location.bottom - location.top)*genderHeight;
              float delta = 0.5f * (height - width);

              if ((genderWidth * location.left- delta) > 0 && (genderWidth * location.left- delta + 1.2 * height)<genderWidth && (0.9 * genderHeight * location.top + 1.2 * height)<genderHeight)
              {
//                genderBitmap = Bitmap.createBitmap(genderBitmap,(int)(genderWidth * location.left-delta),(int)(0.9 * genderHeight * location.top),(int)(1.2 * height),(int)(1.2 * height),null,false);
                genderBitmap = Bitmap.createBitmap(genderBitmap,(int)(genderWidth * location.left-delta),(int)(genderHeight * location.top),(int)(height),(int)(height),null,false);

                location.left = location.left * 640;
                location.right = location.right * 640;
                location.bottom=location.bottom*384;
                location.top=location.top*384;

//              genderBitmap = Bitmap.createBitmap(genderBitmap,(int)(genderWidth * location.left),(int)(0.8 * genderHeight * location.top),(int)(1.2 * width),(int)(1.4 * height),null,false);
                float gender = gender_detector.genderRecognizeImage(genderBitmap);
                LOGGER.d("gender = %f",gender);
                if (gender > 0.5) {
                  LOGGER.d("This is Female");
                  result.setGender("Female");
                }
                else {
//                  saveImageToGallery(genderBitmap);
                  LOGGER.d("This is Male");
                  result.setGender("Male");
                }
                if ( width>64 && height >64)
                {
//                  ageBitmap = Bitmap.createBitmap(genderBitmap,0,0,64,64,null,false);
//                  ageBitmap = genderBitmap.copy(Config.ARGB_8888,true);
//                  Bitmap bgrAgeBitmap = convertToBGR(ageBitmap);
//                  saveImageToGallery(bgrAgeBitmap);

//                  float age = age_detector.ageRecognizationImage(bgrAgeBitmap);
                  float age = age_detector.ageRecognizationImage(genderBitmap);
                  LOGGER.d("age = %f",age);
                  result.setAge(age);
                }
              }

              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }

//              saveImageToGallery(genderBitmap);
            }
            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(
        () -> {
          try {
            detector.setUseNNAPI(isChecked);
            gender_detector.setUseNNAPI(isChecked);
          } catch (UnsupportedOperationException e) {
            LOGGER.e(e, "Failed to set \"Use NNAPI\".");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
//    runInBackground(() -> gender_detector.setNumThreads(numThreads));
  }

  public int saveImageToGallery(Bitmap bmp) {
    //target path
    String root = "/data/vendor/nn/";
    String dirName = "camera_picture";
    File appDir = new File(root , dirName);
    if (!appDir.exists()) {
      appDir.mkdirs();
    }

    long timeStamp = System.currentTimeMillis();

    String sd = "camera";
    String fileName = sd + ".jpg";

    //get file
    File file = new File(appDir, fileName);
    FileOutputStream fos = null;
    try {
      fos = new FileOutputStream(file);
      bmp.compress(Bitmap.CompressFormat.JPEG, 100, fos);
      fos.flush();

      return 2;
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      try {
        if (fos != null) {
          fos.close();
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    return -1;
  }

  public static Bitmap convertToBGR(Bitmap bmp) {
    int width = bmp.getWidth();
    int height = bmp.getHeight();

    Bitmap bgrBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = bmp.getPixel(x, y);
        int alpha = Color.alpha(pixel);
        int red = Color.red(pixel);
        int green = Color.green(pixel);
        int blue = Color.blue(pixel);

        int bgrPixel = Color.argb(alpha, blue, green, red);
        bgrBitmap.setPixel(x, y, bgrPixel);
      }
    }
    return bgrBitmap;
  }

}
