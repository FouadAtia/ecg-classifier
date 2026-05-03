# 🩺 AI-Driven ECG Diagnostic Assistant
### **Explainable Neuro-Symbolic System for Myocardial Infarction Detection**

---

## 📌 Project Overview
This system provides high-precision classification of ECG images into four clinical categories:
*   **Myocardial Infarction (MI)**
*   **History of MI**
*   **Abnormal Heartbeat**
*   **Normal**

By combining **Deep Learning** (MobileNetV2) with **Evolutionary Optimization** (Genetic Algorithms) and **Symbolic Logic**, this project bridges the gap between "Black Box" AI and clinical trust.

---

## 🏗️ Technical Architecture

### 1. Vision Engine
*   **Feature Extraction**: Utilizes a pre-trained **MobileNetV2** (PyTorch) to convert raw ECG images into 1,280-dimensional feature vectors.
*   **Preprocessing**: Implements **Otsu’s Binarization** to remove background grid noise and isolate the ECG signal.

### 2. Search Heuristics (Comparison Study)
We compared two methods for feature selection to optimize the model's performance:
*   **Genetic Algorithm (GA)**: A global stochastic search that evolved a population of masks.
*   **Sequential Forward Selection (SFS)**: A greedy local search heuristic used as a baseline.

### 3. Ensemble Tournament
The project implements a "Super-Learner" architecture using **Stacking**.

| Model | Type | Accuracy |
| :--- | :--- | :--- |
| **Stacking Ensemble** | **Production Winner** | **96.24%** |
| Random Forest (RF) | Base Model | 95.16% |
| Soft Voting | Ensemble | 94.62% |
| Genetic Algorithm | Search Method | 91.40% |
| Heuristic (SFS) | Search Method | 81.81% |

---

## 🚀 Deployment

### Running the API
The system is deployed via **Flask** with full model versioning and logging.
```bash
# Start the production server
python api_server.py
