"""
5-Fold Cross-Validation Training Pipeline
- Stratified K-Fold for balanced splits
- Train multiple models with CV for robust evaluation
- Save best models from each fold
- Aggregate results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
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


class CrossValidationTrainer:
    """
    5-Fold Cross-Validation Trainer
    """
    
    def __init__(
        self,
        model_name: str = 'xception',
        n_folds: int = 5,
        num_epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        output_dir: str = 'models',
        early_stopping_patience: int = 7,
        use_focal_loss: bool = True,
        use_mixup: bool = True,
        input_channels: int = 3
    ):
        """
        Args:
            model_name: Model architecture ('xception' or 'efficientnetv2')
            n_folds: Number of CV folds
            num_epochs: Maximum epochs per fold
            batch_size: Batch size
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            device: Device for training
            output_dir: Directory to save models
            early_stopping_patience: Patience for early stopping
            use_focal_loss: Use Focal Loss (vs BCE)
            use_mixup: Use MixUp augmentation
            input_channels: Input channels (3 or 5)
        """
        self.model_name = model_name
        self.n_folds = n_folds
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
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.fold_results = []
        
    def create_fold_splits(
        self,
        dataset: DeepfakeFrameDataset,
        random_seed: int = 42
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create stratified K-fold splits
        
        Args:
            dataset: Full dataset
            random_seed: Random seed for reproducibility
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Get labels for stratification
        labels = np.array([dataset.frame_list[i]['label'] for i in range(len(dataset))])
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=random_seed)
        
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
            # Print split info
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            print(f"Fold {fold_idx + 1}:")
            print(f"  Train: {len(train_idx)} samples (Real: {(train_labels==0).sum()}, Fake: {(train_labels==1).sum()})")
            print(f"  Val:   {len(val_idx)} samples (Real: {(val_labels==0).sum()}, Fake: {(val_labels==1).sum()})")
        
        return splits
    
    def train_fold(
        self,
        fold_idx: int,
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Dict:
        """
        Train one fold
        
        Args:
            fold_idx: Fold index (0-based)
            train_dataset: Training dataset subset
            val_dataset: Validation dataset subset
            
        Returns:
            Dictionary with fold results
        """
        print(f"\n{'='*70}")
        print(f"Training Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'='*70}\n")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Create model
        model = create_model(
            model_name=self.model_name,
            num_classes=1,
            pretrained=True,
            dropout=0.5 if self.model_name == 'xception' else 0.4,
            input_channels=self.input_channels
        )
        model = model.to(self.device)
        
        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
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
        fold_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_loss, train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                mixup_fn=None if not self.use_mixup else lambda x, y: x,  # Placeholder
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
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_metrics'].append(val_metrics)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f} | Val Precision: {val_metrics['precision']:.4f}")
            print(f"Epoch Time: {time.time() - epoch_start:.1f}s")
            
            # Save best model
            if val_metrics['auc_roc'] > best_auc:
                best_auc = val_metrics['auc_roc']
                best_epoch = epoch
                
                checkpoint_path = self.output_dir / f"{self.model_name}_fold{fold_idx + 1}_best.pth"
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
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        # Fold summary
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1} Complete!")
        print(f"  Best Epoch: {best_epoch + 1}")
        print(f"  Best AUC-ROC: {best_auc:.4f}")
        print(f"{'='*70}\n")
        
        return {
            'fold': fold_idx + 1,
            'best_epoch': best_epoch,
            'best_auc': best_auc,
            'history': fold_history,
            'final_metrics': fold_history['val_metrics'][best_epoch] if best_epoch < len(fold_history['val_metrics']) else val_metrics
        }
    
    def train_cross_validation(
        self,
        metadata_file: str,
        data_root: str
    ) -> Dict:
        """
        Run full cross-validation training
        
        Args:
            metadata_file: Path to preprocessing metadata JSON
            data_root: Root directory for processed frames
            
        Returns:
            Dictionary with all fold results
        """
        print("\n" + "="*70)
        print("Starting 5-Fold Cross-Validation Training")
        print("="*70 + "\n")
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            video_data = data['videos']
        
        # Create full dataset
        full_dataset = DeepfakeFrameDataset(
            video_data=video_data,
            data_root=data_root,
            transform=get_training_augmentation(),
            is_training=True,
            use_frequency=(self.input_channels > 3)
        )
        
        print(f"Total dataset size: {len(full_dataset)} frames\n")
        
        # Create fold splits
        splits = self.create_fold_splits(full_dataset)
        
        # Train each fold
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            # Create training subset
            train_subset = Subset(full_dataset, train_indices)
            
            # Create validation dataset with proper transform
            # Filter video_data for validation indices
            val_frames = [full_dataset.frame_list[i] for i in val_indices]
            val_video_indices = list(set([f['video_name'] for f in val_frames]))
            val_videos = [v for v in video_data if Path(v['video_path']).stem in val_video_indices]
            
            val_dataset = DeepfakeFrameDataset(
                video_data=val_videos,
                data_root=data_root,
                transform=get_validation_augmentation(),
                is_training=False,
                use_frequency=(self.input_channels > 3)
            )
            
            # Use subset for validation to match exact indices
            # Map val_indices to val_dataset indices
            val_frame_mapping = {}
            for i, frame in enumerate(val_dataset.frame_list):
                key = (frame['video_name'], frame['frame_idx'])
                val_frame_mapping[key] = i
            
            val_subset_indices = []
            for idx in val_indices:
                frame = full_dataset.frame_list[idx]
                key = (frame['video_name'], frame['frame_idx'])
                if key in val_frame_mapping:
                    val_subset_indices.append(val_frame_mapping[key])
            
            val_subset = Subset(val_dataset, val_subset_indices)
            
            # Train fold
            fold_result = self.train_fold(fold_idx, train_subset, val_subset)
            self.fold_results.append(fold_result)
        
        # Aggregate results
        cv_results = self.aggregate_results()
        
        # Save CV results
        results_file = self.output_dir / f"{self.model_name}_cv_results.json"
        with open(results_file, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        print(f"\nCross-validation results saved: {results_file}")
        
        return cv_results
    
    def aggregate_results(self) -> Dict:
        """Aggregate results across all folds"""
        print(f"\n{'='*70}")
        print("Cross-Validation Results Summary")
        print(f"{'='*70}\n")
        
        # Collect metrics from all folds
        auc_scores = [fold['best_auc'] for fold in self.fold_results]
        accuracies = [fold['final_metrics']['accuracy'] for fold in self.fold_results]
        f1_scores = [fold['final_metrics']['f1_score'] for fold in self.fold_results]
        
        # Calculate statistics
        cv_results = {
            'model_name': self.model_name,
            'n_folds': self.n_folds,
            'auc_roc': {
                'mean': float(np.mean(auc_scores)),
                'std': float(np.std(auc_scores)),
                'min': float(np.min(auc_scores)),
                'max': float(np.max(auc_scores)),
                'folds': auc_scores
            },
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'folds': accuracies
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'folds': f1_scores
            },
            'fold_details': self.fold_results
        }
        
        # Print summary
        print(f"AUC-ROC:  {cv_results['auc_roc']['mean']:.4f} ± {cv_results['auc_roc']['std']:.4f}")
        print(f"Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"F1-Score: {cv_results['f1_score']['mean']:.4f} ± {cv_results['f1_score']['std']:.4f}")
        print(f"\n{'='*70}\n")
        
        return cv_results


if __name__ == "__main__":
    # Example usage
    trainer = CrossValidationTrainer(
        model_name='xception',
        n_folds=5,
        num_epochs=30,
        batch_size=32,
        learning_rate=1e-4,
        device='cuda',
        output_dir='models/xception_cv',
        use_focal_loss=True,
        use_mixup=True,
        input_channels=3
    )
    
    results = trainer.train_cross_validation(
        metadata_file='data/processed/preprocessing_metadata.json',
        data_root='data/processed'
    )
