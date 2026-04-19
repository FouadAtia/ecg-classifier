import os
import cv2
import numpy as np

def preprocess_ecg_image(image_path, target_size=(256, 256)):
    """
    Loads an ECG image, converts it to grayscale, resizes it, 
    and applies thresholding to remove the background grid.
    """
    # 1. Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    # 2. Resize to standardize the input shape for feature extraction
    img_resized = cv2.resize(img, target_size)

    # 3. Apply Otsu's Binarization
    # The grid is usually light gray, and the ECG ink is dark. 
    # This mathematical threshold turns the background pure black (0) 
    # and the signal pure white (255), dropping all the noisy grid lines.
    _, img_binarized = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return img_binarized

def load_dataset(data_dir="data", target_size=(256, 256)):
    """
    Iterates through the subfolders in the data directory, 
    processes the images, and automatically assigns numerical labels.
    """
    images = []
    labels = []
    class_names = []
    
    print(f"Scanning directory: {data_dir}...")
    
    # Iterate through each subfolder (e.g., 'Normal Person ECG', 'ECG Images of Myocardial...')
    for label_idx, folder_name in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, folder_name)
        
        # Skip files that aren't directories
        if not os.path.isdir(folder_path):
            continue
            
        class_names.append(folder_name)
        print(f"Processing folder: {folder_name} (Label: {label_idx})")
        
        # Process each image inside the subfolder
        valid_extensions = ('.png', '.jpg', '.jpeg')
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(valid_extensions):
                img_path = os.path.join(folder_path, file_name)
                
                processed_img = preprocess_ecg_image(img_path, target_size)
                
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(label_idx)
                    
    # Convert lists to NumPy arrays so Scikit-Learn and PyGAD can use them
    X = np.array(images)
    y = np.array(labels)
    
    return X, y, class_names

if __name__ == "__main__":
    # --- Local Testing Block ---
    # This only runs if you execute this specific file directly.
    print("Starting Ingestion Pipeline Test...\n")
    
    # Assuming your terminal is in the root 'ecg_classifier_project' folder
    X, y, classes = load_dataset(data_dir="data", target_size=(256, 256))
    
    print("\n--- Ingestion Complete ---")
    print(f"Total Images Loaded: {len(X)}")
    print(f"Total Labels Loaded: {len(y)}")
    print(f"Total Classes Found: {len(classes)}")
    
    for i, class_name in enumerate(classes):
        # Count how many images belong to each class
        count = np.sum(y == i)
        print(f"  - {class_name}: {count} images")
        
    if len(X) > 0:
        print(f"\nShape of the full dataset (X): {X.shape}")
        print(f"Shape of a single processed image: {X[0].shape}")
        print("Data type:", X.dtype)