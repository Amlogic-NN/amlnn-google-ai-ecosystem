# MediaPipe

[MediaPipe](https://developers.google.com/edge/mediapipe/solutions/guide) is Google's on-device framework for vision, text, and audio ML. The Tasks API wraps each task in a ready-to-call interface and runs inference on the device. Common vision tasks include gesture recognition, hand landmarks, face detection and landmarks, pose, object detection, image classification, and image segmentation. Models use the `.task` format.

The MediaPipe sample in this tree packages that stack for Android. `Android/Libs` holds the prebuilt Tasks AARs an app links against. `Android/Example` holds camera demos for gesture recognition, hand landmarks, and pose landmarks. The gesture demos include float and quantized builds.

## Structure

- `Android/Libs/` — prebuilt MediaPipe Tasks AARs, grouped by ABI.
- `Android/Libs/arm64/` — `tasks_core.aar` (LiteRT 2.3.0) and `task_vision.aar`.
- `Android/Example/` — prebuilt demo APKs.
- `Android/Example/gesture_recognizer.apk` — Gesture Recognizer. Bundles the float and quantized models.
- `Android/Example/gesture_recoginizer_float.apk` — Gesture Recognizer with the preprocessed float32 model.
- `Android/Example/gesture_recoginizer_quant.apk` — Gesture Recognizer with the quantized hand model.
- `Android/Example/hand_landmarker.apk` — Hand Landmarker.
- `Android/Example/pose_landmarker.apk` — Pose Landmarker. Bundles lite, full, and heavy models.

## Libs

Prebuilt MediaPipe Tasks libraries for Android. A vision app depends on both AARs. The Android demo uses `arm64`.

| Directory | Contents                            |
| --------- | ----------------------------------- |
| `arm64`   | `tasks_core.aar`, `task_vision.aar` |

- `tasks_core.aar`: Tasks runtime, built against LiteRT **2.3.0**. It contains the graph and packet framework, `TaskRunner`, `BaseOptions` (CPU, GPU, or NPU), the shared result types, and `libmediapipe_tasks_jni.so`. Task loading and execution go through this library.
- `task_vision.aar`: Java API for vision tasks. It depends on `tasks_core.aar` and exposes `GestureRecognizer`, `HandLandmarker`, `FaceDetector`, `FaceLandmarker`, `PoseLandmarker`, `ObjectDetector`, `ImageClassifier`, and `ImageSegmenter`.

The prebuilt stack targets ADLA **2.0.2.7.6**. Other ADLA versions are not supported. A different ADLA version requires a rebuild.

## Example

### Models

A `.task` bundle holds one or more `.tflite` graphs. In the upstream float graphs, each weight is a float16 constant. A `DEQUANTIZE` op expands it to float32, and the following float operators consume that float32 tensor. AML support for this float16-constant + `DEQUANTIZE` pattern is limited, so the graphs are converted before the `.task` is packed: the constants are stored as float32, and the float operators read them directly.

### Gesture Recognizer

Real-time [gesture recognition](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer). The camera frame is classified into a gesture, and the result includes handedness and hand landmarks.

- `gesture_recognizer.apk`: bundles `assets/gesture_recognizer.task` and `assets/gesture_recognizer_quant.task`.
- `gesture_recoginizer_float.apk`: float model, preprocessed to float32 weights. Bundles `assets/gesture_recognizer.task`.
- `gesture_recoginizer_quant.apk`: quantized model. Bundles `assets/gesture_recognizer.task`.

### Hand Landmarker

Real-time [hand landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker). The camera frame is tracked for hand landmarks and handedness.

- `hand_landmarker.apk`: bundles `assets/hand_landmarker.task`.

### Pose Landmarker

Real-time [pose landmarks](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker). The camera frame is tracked for body pose landmarks.

- `pose_landmarker.apk`: bundles `assets/pose_landmarker_lite.task`, `assets/pose_landmarker_full.task`, and `assets/pose_landmarker_heavy.task`.
