import torch
import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

class ECGEnsemble:
    def __init__(self):
        """
        Initializes the Soft Voting Classifier with three diverse base models.
        """
        print("Initializing Soft Voting Ensemble (RF + GB + SVM)...")
        # 1. The Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 2. The Booster
        gb = HistGradientBoostingClassifier(random_state=42)
        
        # 3. The Boundary Finder (probability=True is required for soft voting)
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Wrap them in the Voting Classifier
        self.ensemble = VotingClassifier(
            estimators=[('Random Forest', rf), ('Gradient Boosting', gb), ('SVM', svm)],
            voting='soft'
        )

    def fit(self, X_train, y_train):
        """Trains the ensemble on the GA-optimized dataset."""
        print("Training the ensemble. This will just take a few seconds...")
        self.ensemble.fit(X_train, y_train)
        print("Ensemble training complete.")

    def predict(self, X):
        """Generates final class predictions."""
        return self.ensemble.predict(X)
        
    def predict_proba(self, X):
        """Returns the exact confidence percentages for each class."""
        return self.ensemble.predict_proba(X)

    def save_model(self, filepath="models/final_ensemble.pkl"):
        """Saves the trained model to disk for the Streamlit UI."""
        joblib.dump(self.ensemble, filepath)
        print(f"Model successfully saved to {filepath}")

if __name__ == "__main__":
    # --- Local Testing Block ---
    from ingestion import load_dataset
    from features import ECGFeatureExtractor
    from ga_selector import GeneticFeatureSelector
    from sklearn.model_selection import train_test_split # <--- We import the shuffler
    
    print("Starting Ensemble Pipeline Test...\n")
    
    # 1. Load Data 
    X_raw, y_raw, classes = load_dataset(data_dir="data", target_size=(256, 256))
    
    if len(X_raw) > 0:
        # 2. Grab 50 randomly shuffled images instead of just the top 50
        X_train, X_test_batch, y_train, y_test_batch = train_test_split(
            X_raw, y_raw, test_size=50, random_state=42, stratify=y_raw
        )
        
        # 3. Extract deep features
        extractor = ECGFeatureExtractor()
        X_features = extractor.extract(X_test_batch)
        
        # 4. Run a hyper-fast 1-generation GA just to get a feature mask
        ga = GeneticFeatureSelector(num_features=X_features.shape[1], num_generations=1)
        ga.fit(X_features, y_test_batch)
        X_optimized = ga.transform(X_features)
        
        # 5. Train the Ensemble
        model = ECGEnsemble()
        model.fit(X_optimized, y_test_batch)
        
        # 6. Test Prediction and Save
        sample_prediction = model.predict(X_optimized[0:1])
        print(f"\nPrediction for first sample: Class {sample_prediction[0]}")
        model.save_model()