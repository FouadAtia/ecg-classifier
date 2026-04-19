import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

class ECGFeatureExtractor:
    def __init__(self, target_size=(256, 256)):
        """
        Initializes the feature extractor using a pre-trained PyTorch MobileNetV2.
        """
        self.target_size = target_size
        
        print("Loading pre-trained PyTorch MobileNetV2 model...")
        # Load the pre-trained model
        weights = models.MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)
        
        # MobileNetV2's features are extracted before the classifier.
        # We add an AdaptiveAvgPool2d to flatten the 2D feature maps into a 1D vector (1280 features)
        self.model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Set to evaluation mode (turns off training-specific layers like dropout)
        self.model.eval()
        print("Feature Extractor ready.")

    def extract(self, images):
        """
        Takes the binarized 2D grayscale images, converts them to 3-channel PyTorch tensors,
        applies ImageNet normalization, and extracts deep features.
        """
        print(f"Extracting features for {len(images)} images...")
        
        # 1. Duplicate the grayscale channel 3 times to fake an RGB image
        images_rgb = np.stack((images,) * 3, axis=-1)
        
        # 2. Convert to PyTorch Tensor format: (Batch, Channels, Height, Width)
        # PyTorch expects images in [0, 1] range rather than [0, 255]
        tensor_images = torch.tensor(images_rgb).permute(0, 3, 1, 2).float() / 255.0
        
        # 3. Apply standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor_images = (tensor_images - mean) / std
        
        # 4. Create a DataLoader to process in batches (saves memory)
        dataset = TensorDataset(tensor_images)
        dataloader = DataLoader(dataset, batch_size=32)
        
        features_list = []
        
        # 5. Extract features without tracking gradients (saves memory & speeds up)
        with torch.no_grad():
            for batch in dataloader:
                batch_features = self.model(batch[0])
                features_list.append(batch_features.numpy())
                
        return np.vstack(features_list)

if __name__ == "__main__":
    # --- Local Testing Block ---
    from ingestion import load_dataset
    
    print("Starting Feature Extraction Test...\n")
    
    X_raw, y_raw, classes = load_dataset(data_dir="data", target_size=(256, 256))
    
    if len(X_raw) > 0:
        extractor = ECGFeatureExtractor(target_size=(256, 256))
        X_features = extractor.extract(X_raw)
        
        print("\n--- Feature Extraction Complete ---")
        print(f"Original Image Shape: {X_raw.shape}")
        print(f"Extracted Features Shape: {X_features.shape}")
        print(f"Each image has been converted into {X_features.shape[1]} deep numerical features.")
    else:
        print("No images found to process. Check your data directory.")