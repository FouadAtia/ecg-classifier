import torch  # Critical DLL fix for Windows
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Import your custom modular components from the src folder
from src.ingestion import load_dataset
from src.features import ECGFeatureExtractor
from src.ga_selector import GeneticFeatureSelector
from src.ensemble import ECGEnsemble
from src.evaluation import ModelEvaluator

def run_full_training_pipeline():
    print("Starting Full ECG Classification Pipeline...\n")

    # --- Phase 1: Data Ingestion ---
    print("=== Phase 1: Ingesting Dataset ===")
    X_raw, y, class_names = load_dataset(data_dir="data", target_size=(256, 256))
    
    # Split: 80% for training, 20% for final evaluation
    # 'stratify=y' ensures both sets have the same percentage of each heart condition
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Loaded {len(X_raw)} images total.")
    print(f"Training on: {len(y_train)} samples | Testing on: {len(y_test)} samples\n")

    # --- Phase 2: Deep Feature Extraction ---
    print("=== Phase 2: Feature Extraction (MobileNetV2) ===")
    extractor = ECGFeatureExtractor(target_size=(256, 256))
    
    # We extract features for both sets
    X_features_train = extractor.extract(X_raw_train)
    X_features_test = extractor.extract(X_raw_test)
    print(f"Extraction Complete. Feature shape: {X_features_train.shape[1]} deep embeddings.\n")

    # --- Phase 3: Genetic Feature Selection ---
    print("=== Phase 3: Genetic Evolution (Feature Selection) ===")
    # We'll run for 15 generations to allow for a more robust evolution
    ga = GeneticFeatureSelector(
        num_features=X_features_train.shape[1], 
        num_generations=15, 
        sol_per_pop=10
    )
    
    # Find the best binary mask (which features to keep vs drop)
    best_mask = ga.fit(X_features_train, y_train)
    
    # Apply the winning mask to our datasets
    X_opt_train = ga.transform(X_features_train)
    X_opt_test = ga.transform(X_features_test)
    
    # Save the mask so the UI knows which specific features to look for
    joblib.dump(best_mask, "models/ga_feature_mask.pkl")
    print(f"GA Finished. Reduced features to {X_opt_train.shape[1]} optimized markers.\n")

    # --- Phase 4: Ensemble Training ---
    print("=== Phase 4: Training Voting Ensemble (SVM + RF + GB) ===")
    model = ECGEnsemble()
    model.fit(X_opt_train, y_train)
    
    # Save the final trained brain
    model.save_model("models/final_ensemble.pkl")
    print("Ensemble saved to models/final_ensemble.pkl\n")

    # --- Phase 5: Evaluation & Error Analysis ---
    print("=== Phase 5: Performance Evaluation ===")
    evaluator = ModelEvaluator(class_names)
    
    # Generate predictions on the 20% unseen test data
    y_pred = model.predict(X_opt_test)
    
    # Print the report and save the visual confusion matrix
    evaluator.evaluate_predictions(y_test, y_pred)
    evaluator.plot_confusion_matrix(y_test, y_pred, save_path="models/confusion_matrix.png")
    
    print("\nPipeline Complete! Your AI is trained and ready for the UI.")

if __name__ == "__main__":
    run_full_training_pipeline()