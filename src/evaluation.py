import torch # Keep the DLL fix consistent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names

    def evaluate_predictions(self, y_true, y_pred):
        """Prints standard classification metrics (Precision, Recall, F1)."""
        print("\n--- Final Model Evaluation ---")
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {acc * 100:.2f}%\n")
        
        print("Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return acc

    def plot_confusion_matrix(self, y_true, y_pred, save_path="models/confusion_matrix.png"):
        """Generates a heatmap to visualize error patterns."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('ECG Classification Error Analysis')
        plt.ylabel('Actual Condition')
        plt.xlabel('Predicted Condition')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nConfusion matrix visualization saved to {save_path}")

    def get_misclassified_examples(self, y_true, y_pred, X_raw_test):
        """Identifies specific images the model got wrong for further inspection."""
        errors = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                errors.append({
                    'index': i,
                    'true': self.class_names[y_true[i]],
                    'pred': self.class_names[y_pred[i]]
                })
        return errors

if __name__ == "__main__":
    # Quick dummy test to ensure the plotting works
    print("Testing Evaluation Module...")
    classes = ['MI', 'History of MI', 'Abnormal', 'Normal']
    evaluator = ModelEvaluator(classes)
    
    # Fake data for testing
    y_true = [0, 1, 2, 3, 0]
    y_pred = [0, 2, 2, 3, 1]
    
    evaluator.evaluate_predictions(y_true, y_pred)
    evaluator.plot_confusion_matrix(y_true, y_pred)