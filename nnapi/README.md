# NNAPI

本目录是 Android NNAPI 相关示例。`src` 是可编译工程，`example` 是对应的预编译 APK。

这些 Demo 基于 TensorFlow Lite，摄像头或麦克风推理时可以切换到 NNAPI。

## src

用 Android Studio 打开各工程下的 `android` 目录编译运行。部分工程还带 `ios`、`raspberry_pi`，本目录重点是 Android。

| 目录 | 说明 |
|------|------|
| `face_gender_detect_integrated` | 摄像头实时人脸检测（`det_retinaface_int8_uint8.tflite`），再对裁出的人脸做性别（`sdk_gender_int8_uint8.tflite`）和年龄（`sdk_age_quant.acuity.tflite`）识别。界面可切换 NNAPI。 |
| `hand_gesture` | 摄像头实时检测，模型为 `hand_landmark_full_unt8.tflite`，界面可切换 NNAPI。 |
| `pose_estimation` | 人体姿态估计。支持 PoseNet、MoveNet Lightning / Thunder（单人）和 MoveNet MultiPose（最多 6 人）。设备可选 CPU、NNAPI、GPU。 |
| `sound_classification` | 麦克风声音分类，模型为 YAMNet。工程里分别用 XNNPACK 和 NNAPI 创建 Interpreter。 |
| `yolov8m` | 摄像头实时目标检测。模型为 `yolov8m_float32_int8.tflite`（输入 416，COCO 标签），界面可切换 NNAPI。 |

## example

预编译 APK，可直接装到设备上，与 `src` 一一对应：

| APK | 对应源码 |
|-----|----------|
| `face_detect.apk` | `src/face_gender_detect_integrated` |
| `hand_gesture.apk` | `src/hand_gesture` |
| `posenet.apk` | `src/pose_estimation` |
| `sound_classification.apk` | `src/sound_classification` |
| `yolov8m.apk` | `src/yolov8m` |
