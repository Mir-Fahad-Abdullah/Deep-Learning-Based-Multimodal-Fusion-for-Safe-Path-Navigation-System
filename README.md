# Deep Learning-Based Multimodal Fusion for Safe Path Navigation System


> A low-cost, edge-AI navigation assistant that sees obstacles, reasons about danger, and tells the user what to do through short voice instructions.


![Platform](https://img.shields.io/badge/Edge%20AI-NVIDIA%20Jetson%20Nano-76B900?logo=nvidia&logoColor=white)
![Controller](https://img.shields.io/badge/MCU-Arduino%20Nano-00979D?logo=arduino&logoColor=white)
![Model](https://img.shields.io/badge/Detector-YOLO--based%20(MobileNetV3)-blue)
![Runtime](https://img.shields.io/badge/Deploy-ONNX%20%E2%86%92%20TensorRT%20FP16-orange)
![Language](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)

---

## 📖 Overview

In Bangladesh there are an estimated **750,000–950,000 blind people**, and more than **27 million** people live with some form of vision loss. The most common aid, the white cane, is passive: it only senses what it touches, and it struggles in crowded streets, traffic, and rough terrain. Many existing electronic aids rely on low-power microcontrollers and a single sensor, and the more capable commercial devices are out of reach for most people.

**WalkMate** is a proactive alternative. It is a six-wheel rover that combines **computer vision**, **multi-sensor fusion**, and **voice-guided feedback** to detect hazards ahead of the user and give clear, concise spoken instructions such as *"stop, vehicle ahead"* or *"move right, pothole left"*. The whole system runs **on-device**, with no internet connection or cloud inference, and was built for about **47,669 BDT**.

## ✨ Key Features

- **Real-time object detection at the edge**: a lightweight YOLO-based detector (MobileNetV3 backbone, PAN neck, extra P2 head for small near-ground objects) exported through ONNX to a TensorRT FP16 engine on the Jetson Nano.
- **Multi-sensor fusion**: camera vision combined with an ultrasonic sensor, two VL53L1X time-of-flight (LiDAR) sensors, and an MPU-6050 IMU for close-range obstacle detection and incline/tilt awareness.
- **Proximity and time-to-contact (TTC) reasoning**: distance is estimated from bounding-box size and its change over time, then stabilized with EMA smoothing, hysteresis, and cooldowns.
- **Hazard scoring**: each detection gets a unified score from proximity, approach speed, and class priority. The single most critical hazard drives the instruction.
- **Contextual voice guidance**: class-aware instruction templates with direction (left/center/right) and urgency, spoken offline via Piper or pyttsx3. Only one short message is spoken at a time, so the user is not overwhelmed.
- **Rough-terrain mobility**: a rocker-bogie chassis, inspired by Mars rovers, helps the robot handle uneven ground and obstacles close to the wheel diameter.
- **Bangladesh-focused dataset**: trained on SafeWalkBD plus 300 custom street-scene images collected in Bangladesh.
- **Production-style deployment**: runs as a `systemd` service with auto-restart, structured logging, and a health probe.

## 🏗️ System Architecture

WalkMate uses a **two-controller design**:

| Controller | Role |
|---|---|
| **NVIDIA Jetson Nano (4 GB)** | AI perception and decision-making: camera capture, object detection, TTC and hazard reasoning, instruction generation, text-to-speech |
| **Arduino Nano** | Low-level control: reads the ultrasonic, ToF, and IMU sensors, drives the motors, and updates the OLED display |

```
 USB Webcam ──► Jetson Nano ──► Detector ──► Post-processing ──► Proximity / TTC ──► Hazard scoring
                                (TensorRT)   (Soft-NMS, tracking)                          │
                                                                                           ▼
 Ultrasonic + ToF + IMU ──► Arduino Nano ──► Motor control            JSON {closest, farthest, hazard}
                                                                                           │
                                                                                           ▼
                                                                          Instruction templates ──► TTS ──► 🔊 Speaker
```

### Decision output

The reasoning stack hands the instruction layer a compact JSON payload:

```json
{
  "closest":  { "class": "person",        "score": 0.86, "prox_proxy": 0.42, "ttc": 1.7,  "sector": "center" },
  "farthest": { "class": "traffic_light", "score": 0.78, "prox_proxy": 0.05, "ttc": null, "sector": "right"  },
  "hazard":   { "class": "pothole",       "score": 0.91, "prox_proxy": 0.37, "ttc": 0.9,  "sector": "left", "hazard_score": 1.23 }
}
```

## 🔧 Hardware

| Component | Qty | Purpose |
|---|:-:|---|
| NVIDIA Jetson Nano Developer Kit (4 GB) | 1 | AI processing |
| Arduino Nano (ATmega328) | 1 | Sensor interface and motor control |
| Logitech C270 USB webcam (720p, 30 FPS) | 1 | Vision |
| HC-SR04 ultrasonic sensor | 1 | Close-range obstacle detection (backup) |
| VL53L1X time-of-flight sensor | 2 | LiDAR-style ranging, up to ~2 m |
| GY-521 MPU-6050 (6-DoF IMU) | 1 | Tilt, incline, and pothole awareness |
| 128×32 OLED display | 1 | Status display |
| BTS7960 motor driver | 2 | High-current motor control |
| 6 V 100 RPM DC gear motors + 6WD rocker-bogie chassis | 6 | Locomotion |
| DC-DC buck converter, Li-ion packs, power bank, USB speaker | – | Power and audio |

<img width="2632" height="2988" alt="Gemini_Generated_Image_e3cdape3cdape3cd (1)" src="https://github.com/user-attachments/assets/57c7903c-c212-498c-9226-4406efaee0ac" />


## 🧠 Pipeline

**Dataset**
- **SafeWalkBD** (Roboflow Universe): 10,000 annotated images, 29 classes covering living agents, road-surface hazards, and traffic infrastructure.
- **Custom set**: 300 street-scene images from Bangladesh, annotated in YOLO format with the same class taxonomy and merged into the training data.

**Preprocessing and imbalance handling**
- EXIF auto-orientation and light augmentation (flip, 15% grayscale, ±15% brightness, slight blur).
- Label sanitization: box clamping, malformed or empty rows dropped, class IDs validated.
- An **oversampled manifest** repeats images with under-represented classes (up to a capped multiplier) to improve recall on long-tail classes.

**Staged training**: Warm-up (448 px) → Stage 1a (frozen backbone, 512 px) → Stage 1b (unfrozen backbone) → Stage 2 (multi-scale, 512–640 px), with mixed precision, gradient accumulation, and OOM-safe dynamic batch and image size.

**Post-processing**: Soft-NMS, lightweight IoU-based tracking, and temporal smoothing.

## 📊 Results

### Detector performance

| Metric | Warm-up | Stage 1a | Stage 1b | **Stage 2 (final)** |
|---|:-:|:-:|:-:|:-:|
| mAP@0.5 | 0.019 | 0.369 | 0.544 | **0.568** |
| Peak F1 | 0.02 | 0.33 | 0.51 | **0.49** |
| Max recall | 0.13 | 0.67 | 0.70 | **0.67** |
| Resolution (px) | 448 | 512 | 512 | **512–640** |
| Backbone | Frozen | Frozen | Unfrozen | **Unfrozen** |

Progressive training raised mAP@0.5 from 0.019 to **0.568** while keeping recall high for near-ground hazards.

### Sensor-based obstacle response

On the Arduino side, the robot reacted to obstacles with a consistent **~220 ms reaction time**. It moves forward when the path is clear (ultrasonic ≥ 40 cm) and turns left or right as ultrasonic and ToF readings drop below ~25 cm.


## ⚠️ Limitations

- Low-cost sensors are affected by reflective surfaces, environmental noise, and lighting.
- Testing was done in controlled and semi-structured environments. Crowded scenes and extreme weather were not tested thoroughly.
- Feedback is not yet personalized to individual users' walking speed or preferences.
- Voice output is currently **English only**.

## 🚀 Future Work

- Enhanced sensor fusion with depth cameras or additional IMU data to reduce false detections.
- Path planning for **directional guidance** instead of obstacle warnings only.
- Further optimization for lower latency and power use.
- Adaptive, user-centered feedback modes (including Bangla voice output).
- Field trials with visually impaired users.

## 📁 Repository Structure

> *Update this section to match your repo.*

```
WalkMate/
├── firmware/        # Arduino Nano code (sensors, motors, OLED)
├── perception/      # Detector, tracking, TTC and hazard scoring
├── training/        # Dataset prep, staged training scripts
├── deployment/      # ONNX/TensorRT export, systemd service, configs
├── docs/            # Report, figures, circuit schematic
└── README.md
```
