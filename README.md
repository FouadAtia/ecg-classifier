AI-Driven ECG Diagnostic Assistant
An Explainable Neuro-Symbolic System for Myocardial Infarction Detection
📌 Project Overview
This project is a high-precision medical diagnostic system designed to classify ECG images into four clinical categories:

Myocardial Infarction (MI)

History of MI

Abnormal Heartbeat

Normal

The system utilizes a Neuro-Symbolic architecture, combining deep learning feature extraction with evolutionary optimization and a rule-based logic layer to provide both a prediction and a clinical justification.

🏗️ Technical Architecture
1. Vision Engine (Deep Learning)
Model: Pre-trained MobileNetV2 (PyTorch).

Function: Performs deep feature extraction, converting raw ECG images into 1,280-dimensional feature vectors.

2. Optimization Layer (Heuristic Search)
Two search strategies were compared to prune the 1,280 features and identify the most significant diagnostic markers:

Genetic Algorithm (GA): A global stochastic meta-heuristic that evolved a population of feature masks to find optimal feature interactions.

Sequential Forward Selection (SFS): A greedy heuristic search used as a baseline to evaluate local optimization performance.

3. Classification Ensemble (The Tournament)
The project implements a "Super-Learner" ensemble using three base models: Random Forest (RF), Support Vector Machine (SVM), and Gradient Boosting (GB).

Hard & Soft Voting: Consensus-based decision making.

Stacking (Winner): A meta-classifier that learns how to weigh the strengths of each base model for maximum accuracy.

4. Explainability & Reasoning (XAI)
Neuro-Symbolic Logic: A custom ExplanationEngine that applies medical "If-Then" rules to model probabilities.

Interpretability: Integration of SHAP and LIME to visualize which parts of the ECG signal influenced the AI's decision.

📈 Tournament Results & Comparison
Our evaluation process yielded the following performance metrics:

Model / Strategy	Type	Accuracy
Stacking Ensemble	Production Winner	96.24%
Random Forest (RF)	Base Model	95.16%
Soft Voting	Ensemble	94.62%
Genetic Algorithm	Global Search	91.40% (Avg)
Heuristic (SFS)	Local Search	81.81%
Key Finding: Stacking provided a significant lift over individual models, proving that meta-learning effectively handles complex medical edge cases.

🚀 Deployment & API
The system is deployed as a working Flask Web Application featuring model versioning and automated logging.

Installation
Bash
pip install flask numpy torch torchvision opencv-python scikit-learn pygad joblib shap lime
Running the API
Bash
python api_server.py
Endpoint: /predict (POST)

Versioning: Automatically loads the latest optimized models from models/v1.0/.

Logging: All diagnostic requests and performance metrics are recorded in logs/api.log.

Example API Request (cURL)
Bash
curl -X POST -F "file=@path_to_ecg_image.png" http://127.0.0.1:5000/predict
📂 Project Structure
src/: Core logic (Ingestion, Features, Heuristic Search, Reasoning).

models/v1.0/: Saved champion models and optimized feature masks.

ui/: Streamlit dashboard for real-time demonstrations.

logs/: API transaction logs for medical auditing and monitoring.

main_pipeline.py: The "Tournament" script used to train, compare, and save the best models.

Project Discussion
In our analysis, the Genetic Algorithm outperformed the Sequential Forward Selection heuristic by capturing complex synergies between features that a greedy approach missed. By combining this with a Stacking Ensemble, the system achieves a state-of-the-art accuracy of 96.24%. The inclusion of a Reasoning Layer ensures that every prediction is backed by transparent, clinical logic, bridging the gap between "Black Box" AI and clinical trust.