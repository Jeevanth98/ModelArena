"""
Advanced Model Architectures for Deepfake Detection
- XceptionNet (Primary - Best for deepfakes)
- EfficientNetV2 (Secondary - Fast and accurate)
- Support for RGB + Frequency features (5 channels)
- Optimized for small datasets (600 videos)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List, Tuple


class InputAdapter(nn.Module):
    """
    Adapt 5-channel input (RGB + DCT + FFT) to 3-channel for pre-trained models
    """
    
    def __init__(self, in_channels: int = 5, out_channels: int = 3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Initialize to approximately preserve RGB channels
        with torch.no_grad():
            # First 3 channels pass through
            self.adapter.weight[:3, :3] = torch.eye(3).unsqueeze(-1).unsqueeze(-1)
            # Last 2 channels (frequency) contribute slightly
            if in_channels > 3:
                self.adapter.weight[:, 3:] = torch.randn(3, in_channels-3, 1, 1) * 0.01
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class XceptionNetDeepfake(nn.Module):
    """
    XceptionNet architecture for deepfake detection
    State-of-the-art for deepfake detection on FaceForensics++
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.5,
        input_channels: int = 3
    ):
        """
        Args:
            num_classes: Number of output classes (1 for binary)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate for regularization
            input_channels: Number of input channels (3 for RGB, 5 for RGB+freq)
        """
        super().__init__()
        
        self.input_channels = input_channels
        
        # Load Xception backbone
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,  # Remove head
            global_pool=''
        )
        
        # Add input adapter if using frequency features
        if input_channels > 3:
            self.input_adapter = InputAdapter(input_channels, 3)
        else:
            self.input_adapter = None
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head with attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(dropout)
        
        # Multi-layer head for better feature transformation
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W) where C=3 or 5
            
        Returns:
            Logits (B, 1)
        """
        # Adapt input if necessary
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        
        # Extract features
        features = self.backbone(x)  # (B, feature_dim, H', W')
        
        # Global pooling
        features = self.global_pool(features)  # (B, feature_dim, 1, 1)
        features = torch.flatten(features, 1)  # (B, feature_dim)
        
        # Classification head
        x = self.dropout1(features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for ensemble)"""
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        
        features = self.backbone(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        return features


class EfficientNetV2Deepfake(nn.Module):
    """
    EfficientNetV2 for deepfake detection
    Good balance of accuracy and speed
    """
    
    def __init__(
        self,
        model_name: str = 'tf_efficientnetv2_s',
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.4,
        input_channels: int = 3
    ):
        """
        Args:
            model_name: EfficientNetV2 variant
            num_classes: Number of classes
            pretrained: Use pretrained weights
            dropout: Dropout rate
            input_channels: Input channels (3 or 5)
        """
        super().__init__()
        
        self.input_channels = input_channels
        
        # Load backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Input adapter for frequency features
        if input_channels > 3:
            self.input_adapter = InputAdapter(input_channels, 3)
        else:
            self.input_adapter = None
        
        # Feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        
        features = self.backbone(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        x = self.dropout1(features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features"""
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        
        features = self.backbone(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        return features


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models with weighted fusion
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        learnable_weights: bool = False
    ):
        """
        Args:
            models: List of model instances
            weights: Fusion weights (None = equal weights)
            learnable_weights: Whether weights are learnable
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        # Initialize fusion weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.fusion_weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32),
            requires_grad=learnable_weights
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through ensemble
        
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Ensemble prediction (B, 1)
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_models, B, 1)
        
        # Apply softmax to weights
        weights = F.softmax(self.fusion_weights, dim=0)
        weights = weights.view(-1, 1, 1)  # (num_models, 1, 1)
        
        # Weighted sum
        output = (predictions * weights).sum(dim=0)
        
        return output


def create_model(
    model_name: str = 'xception',
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.5,
    input_channels: int = 3
) -> nn.Module:
    """
    Factory function for creating models
    
    Args:
        model_name: 'xception' or 'efficientnetv2'
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        input_channels: 3 for RGB, 5 for RGB+frequency
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'xception':
        model = XceptionNetDeepfake(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            input_channels=input_channels
        )
    elif model_name in ['efficientnetv2', 'efficientnet']:
        model = EfficientNetV2Deepfake(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            input_channels=input_channels
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False
) -> nn.Module:
    """
    Load model weights from checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict, strict=strict)
    
    print(f"Loaded weights from {checkpoint_path}")
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Print model summary"""
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test model creation
    print("Testing XceptionNet...")
    xception = create_model('xception', input_channels=3, pretrained=False)
    print_model_summary(xception, "XceptionNet")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = xception(dummy_input)
    print(f"Output shape: {output.shape}")
    
    print("\nTesting EfficientNetV2...")
    effnet = create_model('efficientnetv2', input_channels=3, pretrained=False)
    print_model_summary(effnet, "EfficientNetV2")
    
    # Test with frequency features
    print("\nTesting with frequency features (5 channels)...")
    xception_freq = create_model('xception', input_channels=5, pretrained=False)
    dummy_input_freq = torch.randn(2, 5, 224, 224)
    output_freq = xception_freq(dummy_input_freq)
    print(f"Output shape with freq features: {output_freq.shape}")
