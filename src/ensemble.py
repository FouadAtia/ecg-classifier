import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

class EnsembleFactory:
    def __init__(self, random_state=42):
        self.random_state = random_state
        # 1. Define Base Models
        self.rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.svm = SVC(probability=True, kernel='linear', random_state=random_state)
        self.gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        
        self.base_models = [('rf', self.rf), ('svm', self.svm), ('gb', self.gb)]

    def evaluate_base_models(self, X_train, X_test, y_train, y_test):
        """Requirement: Evaluate performance on the base models individually."""
        results = {}
        print("--- Individual Base Model Performance ---")
        for name, model in self.base_models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc
            print(f"{name.upper()} Accuracy: {acc:.4f}")
        return results

    def build_voting_ensemble(self, voting_type='soft'):
        """Requirement: Implement Voting (Hard/Soft)."""
        return VotingClassifier(estimators=self.base_models, voting=voting_type)

    def build_stacking_ensemble(self):
        """Requirement: Implement Stacking."""
        # Meta-learner is typically a simple model like Logistic Regression
        meta_learner = LogisticRegression()
        return StackingClassifier(
            estimators=self.base_models, 
            final_estimator=meta_learner, 
            cv=5
        )

def run_ensemble_comparison(X_train, X_test, y_train, y_test):
    factory = EnsembleFactory()
    
    # Evaluate Bases
    factory.evaluate_base_models(X_train, X_test, y_train, y_test)
    
    # Compare Ensembles
    ensembles = {
        "Hard Voting": factory.build_voting_ensemble(voting_type='hard'),
        "Soft Voting": factory.build_voting_ensemble(voting_type='soft'),
        "Stacking": factory.build_stacking_ensemble()
    }
    
    for name, model in ensembles.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"\n--- {name} Results ---")
        print(classification_report(y_test, preds))
        
        # Save the best one (usually Stacking or Soft Voting)
        if name == "Stacking":
            joblib.dump(model, 'models/v1.0/final_ensemble.pkl')