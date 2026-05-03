import os
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime

# Local imports from your existing scripts
from src.ingestion import preprocess_ecg_image
from src.features import ECGFeatureExtractor
from src.reasoning import ExplanationEngine

app = Flask(__name__)

# --- 1. Logging Setup ---
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# --- 2. Model Versioning Configuration ---
MODEL_VERSION = "v1.0"
MODEL_PATH = f"models/{MODEL_VERSION}/final_ensemble.pkl"
MASK_PATH = f"models/{MODEL_VERSION}/ga_feature_mask.pkl"

# Load resources once when the server starts
ensemble = joblib.load(MODEL_PATH)
ga_mask = joblib.load(MASK_PATH)
extractor = ECGFeatureExtractor()
class_names = ["MI", "History of MI", "Abnormal", "Normal"]
explainer = ExplanationEngine(class_names)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint: Receives an image file, returns diagnosis and reasoning.
    Input: Multipart Form Data (file)
    Output: JSON
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    save_path = f"temp_{file.filename}"
    file.save(save_path)

    try:
        # --- 3. Input -> Output Workflow ---
        # A. Preprocess
        img = preprocess_ecg_image(save_path)
        
        # B. Feature Extraction
        raw_features = extractor.extract(np.expand_dims(img, axis=0))
        
        # C. Apply GA Mask (Optimization)
        selected_features = np.where(ga_mask == 1)[0]
        optimized_features = raw_features[:, selected_features]
        
        # D. Ensemble Prediction
        probs = ensemble.predict_proba(optimized_features)[0]
        pred_idx = np.argmax(probs)
        
        # E. Reasoning Layer
        explanation, recommendation = explainer.generate_explanation(pred_idx, probs)

        # --- 4. Success Logging ---
        logging.info(f"Version: {MODEL_VERSION} | Prediction: {class_names[pred_idx]} | Conf: {max(probs):.2f}")

        return jsonify({
            "status": "success",
            "model_version": MODEL_VERSION,
            "prediction": class_names[pred_idx],
            "confidence": float(max(probs)),
            "all_probabilities": {name: float(p) for name, p in zip(class_names, probs)},
            "explanation": explanation,
            "recommendation": recommendation
        })

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)