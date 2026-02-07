"""
ModelArena - Deepfake Video Classification System

A comprehensive deepfake detection system using ensemble of XceptionNet and EfficientNetV2
with 5-fold cross-validation, Test-Time Augmentation, and advanced preprocessing.

Modules:
- preprocessing_advanced: Video preprocessing with MTCNN face detection
- dataset: PyTorch datasets with heavy augmentation and MixUp
- models_advanced: XceptionNet and EfficientNetV2 architectures
- training_utils: Training utilities, losses, and metrics
- train_cv: 5-fold cross-validation training pipeline (WARNING: Frame-level split - data leakage)
- train_single_split: 70:30 train/val split with VIDEO-level splitting (NO data leakage)
- inference: Ensemble inference with TTA
"""

__version__ = '1.1.0'
__author__ = 'ModelArena Team'

# Import main classes for convenience
try:
    from .preprocessing_advanced import VideoPreprocessor, FrequencyFeatureExtractor
    from .models_advanced import XceptionNetDeepfake, EfficientNetV2Deepfake, create_model
    from .dataset import DeepfakeFrameDataset, DeepfakeVideoDataset
    from .training_utils import FocalLoss, MetricsCalculator, train_one_epoch, evaluate
    from .train_cv import CrossValidationTrainer
    from .train_single_split import SingleSplitTrainer
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
        'SingleSplitTrainer',
        'EnsemblePredictor',
        'InferencePipeline',
    ]
except ImportError:
    # Imports might fail if dependencies not installed
    pass
