"""
ModelArena - Deepfake Video Classification System

A comprehensive deepfake detection system using ensemble of XceptionNet and EfficientNetV2
with 5-fold cross-validation, Test-Time Augmentation, and advanced preprocessing.

Modules:
- preprocessing_advanced: Video preprocessing with MTCNN face detection
- dataset: PyTorch datasets with heavy augmentation and MixUp
- models_advanced: XceptionNet and EfficientNetV2 architectures
- training_utils: Training utilities, losses, and metrics
- train_cv: 5-fold cross-validation training pipeline
- inference: Ensemble inference with TTA
"""

__version__ = '1.0.0'
__author__ = 'ModelArena Team'

# Import main classes for convenience
try:
    from .preprocessing_advanced import VideoPreprocessor, FrequencyFeatureExtractor
    from .models_advanced import XceptionNetDeepfake, EfficientNetV2Deepfake, create_model
    from .dataset import DeepfakeFrameDataset, DeepfakeVideoDataset
    from .training_utils import FocalLoss, MetricsCalculator, train_one_epoch, evaluate
    from .train_cv import CrossValidationTrainer
    from .inference import EnsemblePredictor, InferencePipeline
    
    __all__ = [
        'VideoPreprocessor',
        'FrequencyFeatureExtractor',
        'XceptionNetDeepfake',
        'EfficientNetV2Deepfake',
        'create_model',
        'DeepfakeFrameDataset',
        'DeepfakeVideoDataset',
        'FocalLoss',
        'MetricsCalculator',
        'train_one_epoch',
        'evaluate',
        'CrossValidationTrainer',
        'EnsemblePredictor',
        'InferencePipeline',
    ]
except ImportError:
    # Imports might fail if dependencies not installed
    pass
