"""
Training Utilities
- Focal Loss for handling hard examples
- Comprehensive metrics (Accuracy, AUC-ROC, F1, Precision, Recall)
- Training and evaluation functions
- Learning rate schedulers
- Early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import time


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    Handles class imbalance and focuses on hard examples
    
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor (0-1) for class balance
            gamma: Focusing parameter (higher = focus more on hard examples)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, 1) - logits
            targets: Ground truth labels (B, 1) - binary
            
        Returns:
            Loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing
    Prevents overconfidence
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Smoothing factor (0.1 = smooth labels to 0.1 and 0.9)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits)
            targets: Binary labels
            
        Returns:
            Loss
        """
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Standard BCE
        loss = F.binary_cross_entropy_with_logits(inputs, targets)
        
        return loss


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics
    """
    
    @staticmethod
    def calculate_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            predictions: Predicted probabilities (N,)
            targets: Ground truth labels (N,)
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Convert to binary predictions
        binary_preds = (predictions >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, binary_preds)
        metrics['precision'] = precision_score(targets, binary_preds, zero_division=0)
        metrics['recall'] = recall_score(targets, binary_preds, zero_division=0)
        metrics['f1_score'] = f1_score(targets, binary_preds, zero_division=0)
        
        # AUC-ROC (requires probabilities)
        try:
            metrics['auc_roc'] = roc_auc_score(targets, predictions)
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, binary_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Balanced accuracy
            metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        """Pretty print metrics"""
        print(f"\n{prefix}Metrics:")
        print("=" * 60)
        print(f"  Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print(f"  Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"  AUC-ROC:            {metrics.get('auc_roc', 0):.4f}")
        print(f"  Precision:          {metrics.get('precision', 0):.4f}")
        print(f"  Recall:             {metrics.get('recall', 0):.4f}")
        print(f"  F1-Score:           {metrics.get('f1_score', 0):.4f}")
        print(f"  Specificity:        {metrics.get('specificity', 0):.4f}")
        
        if 'true_positives' in metrics:
            print(f"\n  Confusion Matrix:")
            print(f"    TP: {metrics['true_positives']}  FP: {metrics['false_positives']}")
            print(f"    FN: {metrics['false_negatives']}  TN: {metrics['true_negatives']}")
        print("=" * 60)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0001,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (lower is better) or 'max'
 (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if should stop
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    mixup_fn: Optional[callable] = None,
    gradient_accumulation_steps: int = 1
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch
    
    Args:
        model: Model to train
        dataloader: Training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        mixup_fn: MixUp function (optional)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply MixUp if provided
        if mixup_fn is not None and np.random.rand() < 0.5:
            # Mini-batch MixUp
            indices = torch.randperm(images.size(0))
            images2 = images[indices]
            labels2 = labels[indices]
            
            lam = np.random.beta(0.2, 0.2)
            images = lam * images + (1 - lam) * images2
            labels = lam * labels + (1 - lam) * labels2
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every N steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        running_loss += loss.item() * gradient_accumulation_steps
        
        # Collect predictions (convert to probabilities)
        with torch.no_grad():
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(probs.flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
        
        # Update progress bar
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    metrics = MetricsCalculator.calculate_metrics(
        np.array(all_predictions),
        np.array(all_targets)
    )
    
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model
    
    Args:
        model: Model to evaluate
        dataloader: Validation/test data
        criterion: Loss function
        device: Device
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        
        # Collect predictions
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_predictions.extend(probs.flatten())
        all_targets.extend(labels.cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': running_loss / (len(pbar) + 1)})
    
    avg_loss = running_loss / len(dataloader)
    metrics = MetricsCalculator.calculate_metrics(
        np.array(all_predictions),
        np.array(all_targets)
    )
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


if __name__ == "__main__":
    # Test focal loss
    print("Testing Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Dummy data
    inputs = torch.randn(32, 1)
    targets = torch.randint(0, 2, (32, 1)).float()
    
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test metrics
    print("\nTesting Metrics...")
    preds = np.random.rand(100)
    targets = np.random.randint(0, 2, 100)
    
    metrics = MetricsCalculator.calculate_metrics(preds, targets)
    MetricsCalculator.print_metrics(metrics, prefix="Test ")
