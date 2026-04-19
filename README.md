# AI-Powered ECG Classifier 🫀

An end-to-end pipeline using **Transfer Learning (MobileNetV2)**, **Genetic Algorithms** for feature selection, and an **Ensemble Classifier** to detect heart conditions with 95% accuracy.

## Features
- **Preprocessing:** Grayscale conversion and noise reduction.
- **Deep Learning:** PyTorch-based feature extraction.
- **Optimization:** Genetic Algorithm prunes 1280 features down to ~600.
- **Ensemble:** Voting logic between SVM, Random Forest, and Gradient Boosting.
- **UI:** Interactive Streamlit dashboard.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python main_pipeline.py`
3. Launch UI: `streamlit run ui/app.py`