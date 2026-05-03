import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

class ModelExplainer:
    def __init__(self, ensemble_model, X_train_optimized, class_names):
        """
        Initializes the SHAP and LIME explainers.
        - ensemble_model: Your trained soft-voting classifier.
        - X_train_optimized: A sample of your training data AFTER the GA mask is applied.
                             (Needed to build the background distributions).
        - class_names: ["MI", "History of MI", "Abnormal", "Normal"]
        """
        self.model = ensemble_model
        self.class_names = class_names
        
        # We use a small background dataset (e.g., 100 samples) to speed up SHAP
        background_data = shap.sample(X_train_optimized, 100)
        
        print("Initializing SHAP Kernel Explainer (This might take a moment)...")
        # KernelExplainer is used because a VotingClassifier is a complex "black box"
        self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
        
        print("Initializing LIME Tabular Explainer...")
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_optimized,
            feature_names=[f"DeepFeature_{i}" for i in range(X_train_optimized.shape[1])],
            class_names=self.class_names,
            mode='classification'
        )

    def generate_shap_plot(self, X_instance):
        """
        Generates a SHAP force plot for a single patient.
        """
        print("\nCalculating SHAP values...")
        # Calculate SHAP values for the single instance
        shap_values = self.shap_explainer.shap_values(X_instance)
        
        # Plotting the summary plot for this instance
        shap.summary_plot(shap_values, X_instance, feature_names=[f"Feature_{i}" for i in range(X_instance.shape[1])])
        # Note: In a Streamlit app, you would save this plot using plt.savefig() to display it.

    def generate_lime_explanation(self, X_instance_1d):
        """
        Generates a LIME explanation for a single patient.
        Requires a 1D array (a single row).
        """
        print("\nCalculating LIME explanation...")
        # LIME needs the prediction function that outputs probabilities
        predict_fn = lambda x: self.model.predict_proba(x)
        
        exp = self.lime_explainer.explain_instance(
            data_row=X_instance_1d, 
            predict_fn=predict_fn, 
            num_features=10 # Show the top 10 most influential features
        )
        
        # Print text explanation to the terminal
        print("\n--- LIME Top 10 Influential Features ---")
        for feature_rule, weight in exp.as_list():
            print(f"{feature_rule}: {weight:.4f}")
            
        # Optional: Save it as an HTML file so your friends/professor can view it
        exp.save_to_file('lime_explanation.html')
        print("Saved LIME visual to 'lime_explanation.html'")

if __name__ == "__main__":
    # --- Local Test Block ---
    import joblib
    
    # 1. Mock some data (In reality, load your X_train_optimized and trained model)
    print("Running XAI Test...")
    dummy_X_train = np.random.rand(200, 650) # 200 images, 650 optimized features
    dummy_instance = dummy_X_train[0:1]      # 1 single image to test (2D array)
    dummy_instance_1d = dummy_instance[0]    # 1D array for LIME
    
    # Let's pretend we have a basic model for the test
    from sklearn.ensemble import RandomForestClassifier
    mock_model = RandomForestClassifier().fit(dummy_X_train, np.random.randint(0, 4, 200))
    
    # 2. Run the Explainer
    explainer = ModelExplainer(mock_model, dummy_X_train, ["MI", "History of MI", "Abnormal", "Normal"])
    explainer.generate_lime_explanation(dummy_instance_1d)
    explainer.generate_shap_plot(dummy_instance) # Uncomment to pop open the plot