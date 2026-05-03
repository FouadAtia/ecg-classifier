import numpy as np

class ExplanationEngine:
    def __init__(self, class_names):
        self.class_names = class_names

    def generate_explanation(self, prediction_idx, probabilities):
        """
        Applies symbolic logic rules to the ML probability outputs to generate 
        human-readable reasoning and medical safety warnings.
        """
        # --- THE FIX: Force conversion to NumPy array ---
        probabilities = np.array(probabilities)
        
        primary_class = self.class_names[prediction_idx]
        primary_prob = probabilities[prediction_idx]
        
        # Sort probabilities to find the "runner up" class
        sorted_indices = probabilities.argsort()[::-1]
        secondary_idx = sorted_indices[1]
        secondary_class = self.class_names[secondary_idx]
        secondary_prob = probabilities[secondary_idx]

        explanation = []
        recommendation = []

        # RULE 1: High Confidence Consensus
        if primary_prob >= 0.85:
            explanation.append(f"🟢 **Strong Consensus:** The ensemble is highly confident ({primary_prob*100:.1f}%) in identifying '{primary_class}'. The Random Forest, SVM, and Gradient Boosting models are in strict agreement based on the optimized features.")
        
        # RULE 2: Low Confidence / Borderline (Safety Override)
        elif primary_prob < 0.60:
            explanation.append(f"🔴 **Low Confidence Warning:** The primary prediction is '{primary_class}', but confidence is low ({primary_prob*100:.1f}%). The models in the ensemble are disagreeing.")
            recommendation.append("⚠️ ACTION: Manual doctor review heavily advised. Do not rely solely on automated diagnosis for this patient.")
        else:
            explanation.append(f"🟡 **Moderate Consensus:** The ensemble predicts '{primary_class}' with {primary_prob*100:.1f}% confidence. The models generally agree, but lack the strict >85% consensus required for a strong rating. Minor overlapping features from other classes were detected.")    

        # RULE 3: The "History vs. Active MI" Conflict Rule
        if primary_class == "Myocardial Infarction (MI)" and secondary_class == "History of MI" and secondary_prob > 0.20:
            explanation.append(f"🟡 **Diagnostic Nuance:** While active MI is the primary diagnosis, there is a significant secondary probability ({secondary_prob*100:.1f}%) of Historical MI.")
            recommendation.append("🔍 CLINICAL NOTE: Check patient records for past heart attacks. The deep features show overlapping markers (e.g., pathological Q-waves) that blur active and past conditions.")

        # RULE 4: Abnormal but not MI
        if primary_class == "Abnormal Heartbeat" and probabilities[0] < 0.05: # Index 0 is MI
            explanation.append("🔵 **Anomaly Assessed:** An abnormal rhythm is detected, but active MI markers are virtually absent (under 5% probability).")
            recommendation.append("🩺 CLINICAL NOTE: Likely an arrhythmia. Proceed with standard Holter monitoring or further non-emergency diagnostic testing.")

        return "\n\n".join(explanation), "\n\n".join(recommendation)

if __name__ == "__main__":
    # Local Test
    engine = ExplanationEngine(["MI", "History of MI", "Abnormal", "Normal"])
    probs = [0.55, 0.35, 0.05, 0.05] # Simulating a confusing case
    exp, rec = engine.generate_explanation(0, probs)
    print("Explanation:\n", exp)
    print("\nRecommendation:\n", rec)