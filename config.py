"""
Configuration file for training and inference
Modify these parameters to customize your training
"""

# ==================== DATA CONFIGURATION ====================
DATA_CONFIG = {
    # Paths
    'train_video_dir': 'data/train',
    'processed_data_dir': 'data/processed',
    'test_video_dir': 'data/test',  # Update with your test data path
    
    # Preprocessing
    'num_frames_per_video': 16,      # Number of frames to extract (10-20 recommended)
    'image_size': 224,                # Image size (224 for ImageNet models)
    'face_margin': 40,                # Margin around detected face
    'extract_frequency': False,       # Extract DCT/FFT frequency features
    
    # Data splits
    'val_split': 0.2,                 # Validation split (only for single split, not CV)
    'random_seed': 42,
}

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    # Primary model
    'primary_model': 'xception',      # 'xception' or 'efficientnetv2'
    'secondary_model': 'efficientnetv2',
    
    # Model parameters
    'input_channels': 3,              # 3 for RGB, 5 for RGB+frequency
    'dropout': 0.5,                   # Dropout rate (0.4-0.5 recommended)
    'pretrained': True,               # Use ImageNet pretrained weights
}

# ==================== TRAINING CONFIGURATION ====================
TRAINING_CONFIG = {
    # Training strategy
    'use_cv': True,                   # Use cross-validation (highly recommended)
    'n_folds': 5,                     # Number of CV folds
    
    # Training hyperparameters
    'num_epochs': 30,                 # Maximum epochs per fold
    'batch_size': 32,                 # Batch size (reduce to 16 if OOM)
    'learning_rate': 1e-4,            # Initial learning rate
    'weight_decay': 1e-5,             # L2 regularization
    'gradient_accumulation_steps': 1,  # Gradient accumulation (increase if OOM)
    
    # Loss function
    'use_focal_loss': True,           # Use Focal Loss (vs standard BCE)
    'focal_alpha': 0.25,              # Focal loss alpha parameter
    'focal_gamma': 2.0,               # Focal loss gamma parameter
    'label_smoothing': 0.0,           # Label smoothing (0.0 = disabled)
    
    # Augmentation
    'use_mixup': True,                # Use MixUp augmentation
    'mixup_alpha': 0.2,               # MixUp alpha parameter
    
    # Optimizer
    'optimizer': 'adamw',             # 'adam' or 'adamw'
    'scheduler': 'cosine',            # 'cosine', 'step', or 'plateau'
    
    # Early stopping
    'early_stopping': True,
    'early_stopping_patience': 7,     # Epochs to wait before stopping
    'early_stopping_metric': 'auc_roc',  # Metric to monitor
    
    # Checkpointing
    'save_best_only': True,
    'save_frequency': 1,              # Save every N epochs
    
    # Computation
    'device': 'cuda',                 # 'cuda' or 'cpu'
    'num_workers': 2,                 # Data loading workers (2-4 for Colab)
    'pin_memory': True,
}

# ==================== INFERENCE CONFIGURATION ====================
INFERENCE_CONFIG = {
    # Ensemble
    'use_ensemble': True,             # Combine multiple models
    'ensemble_weights': {             # Weights for ensemble
        'xception': 0.6,              # Primary model weight
        'efficientnetv2': 0.4,        # Secondary model weight
    },
    'ensemble_method': 'weighted_average',  # 'average', 'weighted_average', 'voting'
    
    # Test-Time Augmentation
    'use_tta': True,                  # Use TTA (recommended for best performance)
    'tta_steps': 5,                   # Number of TTA augmentations
    
    # Video aggregation
    'frame_aggregation': 'median',    # 'mean', 'median', 'max', 'voting'
    
    # Prediction threshold
    'threshold': 0.5,                 # Classification threshold
    
    # Batch processing
    'inference_batch_size': 16,
}

# ==================== OUTPUT CONFIGURATION ====================
OUTPUT_CONFIG = {
    'output_dir': 'models',
    'results_dir': 'results',
    'logs_dir': 'logs',
    'submission_dir': 'submission',
    
    # Logging
    'verbose': True,
    'save_metrics': True,
    'save_predictions': True,
    'save_visualizations': False,     # Save prediction visualizations (slower)
}

# ==================== EXPERIMENT TRACKING ====================
EXPERIMENT_CONFIG = {
    'experiment_name': 'deepfake_detection_v1',
    'use_wandb': False,               # Use Weights & Biases (requires wandb login)
    'wandb_project': 'deepfake-detection',
    'use_tensorboard': False,         # Use TensorBoard
}

# ==================== ABLATION STUDY CONFIGS ====================
# Predefined configurations for common experiments

# Config 1: Fast baseline (for quick testing)
FAST_CONFIG = {
    **DATA_CONFIG,
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    'num_epochs': 15,
    'n_folds': 1,
    'batch_size': 32,
    'num_frames_per_video': 10,
    'use_mixup': False,
    'early_stopping_patience': 5,
}

# Config 2: Best performance (for competition)
BEST_CONFIG = {
    **DATA_CONFIG,
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    'num_epochs': 40,
    'n_folds': 5,
    'batch_size': 32,
    'num_frames_per_video': 16,
    'use_mixup': True,
    'use_focal_loss': True,
    'early_stopping_patience': 10,
    'extract_frequency': False,  # Can enable for potential boost
}

# Config 3: Memory-efficient (for Colab Free tier)
MEMORY_EFFICIENT_CONFIG = {
    **DATA_CONFIG,
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    'batch_size': 16,
    'num_workers': 2,
    'gradient_accumulation_steps': 2,
    'num_frames_per_video': 12,
    'extract_frequency': False,
}

# Config 4: With frequency features
FREQUENCY_CONFIG = {
    **DATA_CONFIG,
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    'extract_frequency': True,
    'input_channels': 5,
    'num_frames_per_video': 12,  # Fewer frames to save memory
}

# ==================== HELPER FUNCTIONS ====================
def get_config(config_name='default'):
    """
    Get configuration by name
    
    Args:
        config_name: 'default', 'fast', 'best', 'memory_efficient', or 'frequency'
        
    Returns:
        Dictionary with configuration
    """
    if config_name == 'fast':
        return FAST_CONFIG
    elif config_name == 'best':
        return BEST_CONFIG
    elif config_name == 'memory_efficient':
        return MEMORY_EFFICIENT_CONFIG
    elif config_name == 'frequency':
        return FREQUENCY_CONFIG
    else:
        return {
            **DATA_CONFIG,
            **MODEL_CONFIG,
            **TRAINING_CONFIG,
            **INFERENCE_CONFIG,
            **OUTPUT_CONFIG,
            **EXPERIMENT_CONFIG,
        }

def print_config(config):
    """Pretty print configuration"""
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example: Print default config
    config = get_config('default')
    print_config(config)
    
    #  Example: Print best config
    print("\nBest Performance Config:")
    best_config = get_config('best')
    print_config(best_config)
