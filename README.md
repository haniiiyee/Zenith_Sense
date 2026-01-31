# Zenith_Sense: On-Board Wildfire Detection Module 🛰️

## 🚀 Project Overview
Zenith_Sense is a resource-constrained Deep Learning module designed for Low-Earth Orbit (LEO) satellites. It performs real-time **On-Board Processing (OBP)** to detect wildfires from satellite imagery, reducing downlink bandwidth usage by transmitting only critical alerts rather than raw data.

## 🎯 Key Engineering Metrics
| Metric | Value | Constraint (Space-Grade) | Status |
| :--- | :--- | :--- | :--- |
| **Model Size** | **1.19 MB** | < 30 MB | ✅ PASS |
| **Peak RAM** | **~2-4 MB** | < 50 MB | ✅ PASS |
| **Precision** | **INT8 (Quantized)** | 32-bit Float | ✅ PASS |
| **Architecture**| MobileNetV3-Small | Low Compute | ✅ PASS |

## 🛠️ Technology Stack
* **Framework:** TensorFlow / TensorFlow Lite
* **Optimization:** Post-Training Quantization (INT8)
* **Hardware Target:** ARM Cortex-M / Rad-Hard DSPs
* **Language:** Python 3.x

## ⚙️ How It Works
1.  **Acquisition:** Satellite captures Earth observation imagery.
2.  **Inference:** The `zenith_int8.tflite` model processes the image in-memory.
3.  **Decision:** * **Fire Detected (>50%):** Trigger high-priority downlink packet.
    * **No Fire:** Discard image to save bandwidth and battery.

## 📂 Project Structure
* `01_download_data.py`: Automates dataset retrieval (Lite Version).
* `02_train_model.py`: Trains MobileNetV3 on wildfire data.
* `03_quantize_model.py`: Converts model to TFLite (INT8) for edge deployment.
* `04_benchmark.py`: Validates RAM usage and inference speed.
* `05_predict.py`: Runs visual inference on test images.

## 🔬 Results
The system successfully identifies wildfire signatures with high confidence while maintaining a memory footprint suitable for CubeSat and MicroSat architectures.