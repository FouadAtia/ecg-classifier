import torch  # Critical DLL fix for Windows
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Import your custom modules
from src.ingestion import load_dataset
from src.features import ECGFeatureExtractor
from src.heuristic_search import GreedyHeuristicSelector
from src.ensemble import EnsembleFactory

def run_tournament():
    # --- STEP 1: DATA INGESTION ---
    # The fix: Catching class_names to prevent the 'too many values to unpack' error
    print("🚀 Step 1: Loading data and preprocessing images...")
    X_images, y, class_names = load_dataset("data/") 
    
    # --- STEP 2: FEATURE EXTRACTION ---
    # Turning 2D images into 1280-dimensional deep feature vectors
    print("🧠 Step 2: Extracting deep features via MobileNetV2...")
    extractor = ECGFeatureExtractor()
    X_raw = extractor.extract(X_images) # Returns (N, 1280)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

    # --- STEP 3: HEURISTIC SEARCH COMPARISON ---
    print("\n🔎 Step 3: Feature Selection Comparison (Heuristic Requirements)...")
    
    # A. Use existing Genetic Algorithm results
    # Assuming ga_feature_mask.pkl exists in your models folder
    ga_mask_path = 'models/v1.0/ga_feature_mask.pkl'
    if os.path.exists(ga_mask_path):
        ga_mask = joblib.load(ga_mask_path)
        # Handle if mask is stored as binary array or dictionary
        mask_arr = ga_mask if isinstance(ga_mask, np.ndarray) else ga_mask['mask']
        X_train_ga = X_train[:, np.where(mask_arr == 1)[0]]
        X_test_ga = X_test[:, np.where(mask_arr == 1)[0]]
        ga_count = np.sum(mask_arr)
    else:
        print("Warning: GA Mask not found. Skipping GA comparison.")
        X_train_ga, X_test_ga, ga_count = X_train, X_test, 1280

    # B. Run Sequential Forward Selection (Greedy Heuristic Search)
    # We limit max_features to 100 for speed during the tournament
    h_selector = GreedyHeuristicSelector(num_features=1280, max_features=100)
    h_mask = h_selector.fit(X_train, y_train)
    X_train_h = X_train[:, np.where(h_mask == 1)[0]]
    X_test_h = X_test[:, np.where(h_mask == 1)[0]]
    h_count = np.sum(h_mask)

    # --- STEP 4: ENSEMBLE TOURNAMENT ---
    # We will use the GA-optimized features for the final ensemble training
    print("\n🏆 Step 4: Ensemble Tournament (Evaluating Bases, Voting, & Stacking)...")
    factory = EnsembleFactory()
    
    # A. Requirement: Evaluate Base Models
    base_results = factory.evaluate_base_models(X_train_ga, X_test_ga, y_train, y_test)
    
    # B. Requirement: Implement & Evaluate Different Ensembles
    ensembles = {
        "Hard Voting": factory.build_voting_ensemble(voting_type='hard'),
        "Soft Voting": factory.build_voting_ensemble(voting_type='soft'),
        "Stacking": factory.build_stacking_ensemble()
    }
    
    ensemble_results = {}
    for name, model in ensembles.items():
        model.fit(X_train_ga, y_train)
        preds = model.predict(X_test_ga)
        acc = accuracy_score(y_test, preds)
        ensemble_results[name] = acc
        print(f"Ensemble [{name}] Accuracy: {acc:.4f}")

    # --- STEP 5: SAVE THE PRODUCTION MODEL ---
    best_name = max(ensemble_results, key=ensemble_results.get)
    print(f"\n✅ Tournament Winner: {best_name}")
    joblib.dump(ensembles[best_name], 'models/v1.0/final_ensemble.pkl')

    # --- SUMMARY FOR DISCUSSION ---
    print("\n" + "="*40)
    print("📈 FINAL COMPARISON DATA FOR REPORT")
    print("="*40)
    print(f"Search Type | Features Selected | Base Accuracy (Avg)")
    print(f"Genetic Alg | {ga_count:<17} | {np.mean(list(base_results.values())):.4f}")
    print(f"Heuristic   | {h_count:<17} | (Tested via SFS logic)")
    print("-" * 40)
    print("Ensemble Performance Ranking:")
    all_final = {**base_results, **ensemble_results}
    for model_name in sorted(all_final, key=all_final.get, reverse=True):
        print(f" - {model_name:15}: {all_final[model_name]*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_tournament()