"""
Traditional 70:30 Train/Val Split with Video-Level Splitting
- Prevents data leakage by splitting at VIDEO level (not frame level)
- Ensures all frames from same video stay in same split
- Stratified split to maintain class balance
- Single model training (no cross-validation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import time
import sys

# Handle imports for both direct execution and module import
try:
    from src.models_advanced import create_model
    from src.dataset import DeepfakeFrameDataset, get_training_augmentation, get_validation_augmentation
    from src.training_utils import (
        FocalLoss, train_one_epoch, evaluate, 
        save_checkpoint, EarlyStopping, MetricsCalculator
    )
except ImportError:
    from models_advanced import create_model
    from dataset import DeepfakeFrameDataset, get_training_augmentation, get_validation_augmentation
    from training_utils import (
        FocalLoss, train_one_epoch, evaluate, 
        save_checkpoint, EarlyStopping, MetricsCalculator
    )


class SingleSplitTrainer:
    """
    Single Train/Validation Split Trainer
    Splits data at VIDEO level to prevent data leakage
    """
    
    def __init__(
        self,
        model_name: str = 'xception',
        train_split: float = 0.7,
        num_epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        output_dir: str = 'models',
        early_stopping_patience: int = 7,
        use_focal_loss: bool = True,
        use_mixup: bool = True,
        input_channels: int = 3,
        random_seed: int = 42
    ):
        """
        Args:
            model_name: Model architecture ('xception' or 'efficientnetv2')
            train_split: Fraction of data for training (default 0.7 = 70%)
            num_epochs: Maximum epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            device: Device for training
            output_dir: Directory to save models
            early_stopping_patience: Patience for early stopping
            use_focal_loss: Use Focal Loss (vs BCE)
            use_mixup: Use MixUp augmentation
            input_channels: Input channels (3 or 5)
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.train_split = train_split
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.early_stopping_patience = early_stopping_patience
        self.use_focal_loss = use_focal_loss
        self.use_mixup = use_mixup
        self.input_channels = input_channels
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SingleSplitTrainer Initialized")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Split: {int(train_split*100)}% train / {int((1-train_split)*100)}% validation")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"Random Seed: {random_seed}")
        print(f"{'='*70}\n")
        
    def create_video_level_split(
        self,
        video_data: List[Dict],
        random_seed: int = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create VIDEO-LEVEL train/val split to prevent data leakage
        
        This ensures all frames from the same video stay together in either
        train or validation set, preventing the model from seeing different
        frames of the same video during training and validation.
        
        Args:
            video_data: List of video metadata dictionaries
            random_seed: Random seed (uses instance seed if None)
            
        Returns:
            Tuple of (train_videos, val_videos)
        """
        if random_seed is None:
            random_seed = self.random_seed
            
        # Filter videos with faces detected
        valid_videos = [v for v in video_data if v.get('faces_detected', 0) > 0]
        
        # Extract video identifiers and labels
        video_ids = [Path(v['video_path']).stem for v in valid_videos]
        labels = [v['label'] for v in valid_videos]
        
        print(f"\n{'='*70}")
        print("Creating Video-Level Split (Prevents Data Leakage)")
        print(f"{'='*70}")
        print(f"Total videos: {len(valid_videos)}")
        print(f"  Real: {sum(1 for l in labels if l == 0)}")
        print(f"  Fake: {sum(1 for l in labels if l == 1)}")
        
        # Stratified split at VIDEO level
        train_indices, val_indices = train_test_split(
            list(range(len(valid_videos))),
            test_size=(1 - self.train_split),
            stratify=labels,
            random_state=random_seed
        )
        
        train_videos = [valid_videos[i] for i in train_indices]
        val_videos = [valid_videos[i] for i in val_indices]
        
        # Calculate frame counts
        train_frames = sum(v.get('faces_detected', 0) for v in train_videos)
        val_frames = sum(v.get('faces_detected', 0) for v in val_videos)
        
        train_labels = [v['label'] for v in train_videos]
        val_labels = [v['label'] for v in val_videos]
        
        print(f"\nTrain Set:")
        print(f"  Videos: {len(train_videos)} (Real: {sum(1 for l in train_labels if l == 0)}, "
              f"Fake: {sum(1 for l in train_labels if l == 1)})")
        print(f"  Frames: {train_frames}")
        
        print(f"\nValidation Set:")
        print(f"  Videos: {len(val_videos)} (Real: {sum(1 for l in val_labels if l == 0)}, "
              f"Fake: {sum(1 for l in val_labels if l == 1)})")
        print(f"  Frames: {val_frames}")
        
        print(f"\n✅ Video-level split ensures NO DATA LEAKAGE")
        print(f"   (All frames from same video stay in same split)")
        print(f"{'='*70}\n")
        
        return train_videos, val_videos
    
    def check_data_leakage(
        self,
        train_dataset: DeepfakeFrameDataset,
        val_dataset: DeepfakeFrameDataset
    ) -> bool:
        """
        Verify no data leakage between train and validation sets
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            True if no leakage detected, False otherwise
        """
        # Get unique video names from each set
        train_videos = set(frame['video_name'] for frame in train_dataset.frame_list)
        val_videos = set(frame['video_name'] for frame in val_dataset.frame_list)
        
        # Check for overlap
        overlap = train_videos.intersection(val_videos)
        
        print(f"\n{'='*70}")
        print("Data Leakage Check")
        print(f"{'='*70}")
        print(f"Train videos: {len(train_videos)}")
        print(f"Val videos: {len(val_videos)}")
        print(f"Overlapping videos: {len(overlap)}")
        
        if len(overlap) > 0:
            print(f"\n❌ DATA LEAKAGE DETECTED!")
            print(f"   {len(overlap)} videos appear in both train and validation:")
            for vid in list(overlap)[:5]:
                print(f"   - {vid}")
            if len(overlap) > 5:
                print(f"   ... and {len(overlap) - 5} more")
            print(f"{'='*70}\n")
            return False
        else:
            print(f"\n✅ NO DATA LEAKAGE - All videos properly separated")
            print(f"{'='*70}\n")
            return True
    
    def train(
        self,
        metadata_file: str,
        data_root: str
    ) -> Dict:
        """
        Train model with single train/val split
        
        Args:
            metadata_file: Path to preprocessing metadata JSON
            data_root: Root directory for processed frames
            
        Returns:
            Training results dictionary
        """
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        video_data = metadata.get('videos', [])
        
        # Create video-level split (prevents data leakage)
        train_videos, val_videos = self.create_video_level_split(video_data)
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = DeepfakeFrameDataset(
            video_data=train_videos,
            data_root=data_root,
            transform=get_training_augmentation(),
            is_training=True,
            use_frequency=False,
            num_frames_per_video=16
        )
        
        val_dataset = DeepfakeFrameDataset(
            video_data=val_videos,
            data_root=data_root,
            transform=get_validation_augmentation(),
            is_training=False,
            use_frequency=False,
            num_frames_per_video=16
        )
        
        # Verify no data leakage
        leakage_free = self.check_data_leakage(train_dataset, val_dataset)
        if not leakage_free:
            raise ValueError("Data leakage detected! Please fix the split.")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Create model
        print(f"\nCreating {self.model_name} model...")
        model = create_model(
            model_name=self.model_name,
            num_classes=1,
            pretrained=True,
            input_channels=self.input_channels
        )
        model = model.to(self.device)
        
        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print("Using Focal Loss (alpha=0.25, gamma=2.0)")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using BCE Loss")
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            mode='max',  # Maximize AUC-ROC
            min_delta=0.001
        )
        
        # Training loop
        best_auc = 0.0
        best_epoch = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}\n")
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"{'─'*70}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_loss, train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                mixup_fn=None if not self.use_mixup else lambda x, y: x,
                gradient_accumulation_steps=1
            )
            
            # Validate
            val_loss, val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=self.device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Print metrics
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
            print(f"  Val F1-Score: {val_metrics['f1_score']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print(f"Epoch Time: {time.time() - epoch_start:.1f}s")
            
            # Save best model
            if val_metrics['auc_roc'] > best_auc:
                best_auc = val_metrics['auc_roc']
                best_epoch = epoch
                
                checkpoint_path = self.output_dir / f"{self.model_name}_best.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    filepath=str(checkpoint_path)
                )
                print(f"✓ Best model saved (AUC-ROC: {best_auc:.4f})")
            
            # Early stopping
            should_stop = early_stopping(val_metrics['auc_roc'])
            if should_stop:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Training complete
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Best Epoch: {best_epoch + 1}/{epoch + 1}")
        print(f"Best AUC-ROC: {best_auc:.4f}")
        print(f"Model saved to: {self.output_dir / f'{self.model_name}_best.pth'}")
        print(f"{'='*70}\n")
        
        # Save training results
        results = {
            'model_name': self.model_name,
            'best_epoch': best_epoch + 1,
            'best_auc': float(best_auc),
            'final_metrics': history['val_metrics'][best_epoch],
            'history': history,
            'config': {
                'train_split': self.train_split,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'use_focal_loss': self.use_focal_loss,
                'use_mixup': self.use_mixup,
                'random_seed': self.random_seed
            }
        }
        
        results_file = self.output_dir / f"{self.model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}\n")
        
        return results


if __name__ == "__main__":
    # Example usage
    trainer = SingleSplitTrainer(
        model_name='xception',
        train_split=0.7,  # 70% train, 30% validation
        num_epochs=30,
        batch_size=32,
        learning_rate=1e-4,
        device='cuda',
        output_dir='models/xception_single_split',
        use_focal_loss=True,
        use_mixup=True,
        input_channels=3,
        random_seed=42
    )
    
    results = trainer.train(
        metadata_file='data/processed/preprocessing_metadata.json',
        data_root='data/processed'
    )
