"""
Ensemble Inference Pipeline with Test-Time Augmentation (TTA)
- Load multiple trained models
- Apply TTA for robust predictions
- Aggregate predictions from ensemble
- Generate submission file
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import cv2
import sys

# Handle imports for both direct execution and module import
try:
    from src.models_advanced import create_model, EnsembleModel
    from src.dataset import DeepfakeVideoDataset, get_tta_augmentation, get_validation_augmentation
    from src.preprocessing_advanced import VideoPreprocessor
except ImportError:
    from models_advanced import create_model, EnsembleModel
    from dataset import DeepfakeVideoDataset, get_tta_augmentation, get_validation_augmentation
    from preprocessing_advanced import VideoPreprocessor


class EnsemblePredictor:
    """
    Ensemble predictor with TTA support
    """
    
    def __init__(
        self,
        model_configs: List[Dict],
        device: str = 'cuda',
        use_tta: bool = True,
        tta_steps: int = 5,
        ensemble_method: str = 'weighted_average'
    ):
        """
        Args:
            model_configs: List of model config dicts
                [{
                    'model_name': 'xception',
                    'checkpoint_path': 'path/to/model.pth',
                    'weight': 0.6,  # Ensemble weight
                    'input_channels': 3
                }, ...]
            device: Device for inference
            use_tta: Whether to use TTA
            tta_steps: Number of TTA augmentations
            ensemble_method: 'average', 'weighted_average', or 'voting'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_tta = use_tta
        self.tta_steps = tta_steps
        self.ensemble_method = ensemble_method
        
        # Load models
        self.models = []
        self.model_weights = []
        
        print(f"\nLoading {len(model_configs)} models for ensemble...")
        for config in model_configs:
            model = self._load_model(config)
            self.models.append(model)
            self.model_weights.append(config.get('weight', 1.0 / len(model_configs)))
        
        # Normalize weights
        total_weight = sum(self.model_weights)
        self.model_weights = [w / total_weight for w in self.model_weights]
        
        print(f"Ensemble weights: {[f'{w:.3f}' for w in self.model_weights]}")
        print(f"Using TTA: {use_tta} ({tta_steps} augmentations)" if use_tta else "No TTA")
    
    def _load_model(self, config: Dict) -> nn.Module:
        """Load a single model from checkpoint"""
        model = create_model(
            model_name=config['model_name'],
            num_classes=1,
            pretrained=False,
            input_channels=config.get('input_channels', 3)
        )
        
        # Load weights
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        
        print(f"  ✓ Loaded {config['model_name']} from {Path(config['checkpoint_path']).name}")
        
        return model
    
    @torch.no_grad()
    def predict_frame(
        self,
        image: torch.Tensor,
        apply_tta: bool = True
    ) -> float:
        """
        Predict single frame with ensemble and optional TTA
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            apply_tta: Whether to apply TTA
            
        Returns:
            Prediction probability (0-1, fake probability)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim
        
        image = image.to(self.device)
        
        all_predictions = []
        
        # TTA Loop
        tta_iterations = self.tta_steps if (apply_tta and self.use_tta) else 1
        
        for tta_idx in range(tta_iterations):
            # Apply TTA transform if enabled
            if tta_idx > 0 and apply_tta:
                # Simple TTA: horizontal flip, brightness adjustment, etc.
                augmented = self._apply_tta_transform(image, tta_idx)
            else:
                augmented = image
            
            # Ensemble predictions
            model_preds = []
            for model, weight in zip(self.models, self.model_weights):
                output = model(augmented)
                prob = torch.sigmoid(output)
                model_preds.append(prob.item() * weight)
            
            # Weighted ensemble
            ensemble_pred = sum(model_preds)
            all_predictions.append(ensemble_pred)
        
        # Average across TTA iterations
        final_pred = np.mean(all_predictions)
        
        return final_pred
    
    def _apply_tta_transform(self, image: torch.Tensor, tta_idx: int) -> torch.Tensor:
        """
        Apply simple TTA transforms
        
        Args:
            image: Input tensor (B, C, H, W)
            tta_idx: TTA iteration index
            
        Returns:
            Augmented tensor
        """
        if tta_idx == 1:
            # Horizontal flip
            return torch.flip(image, dims=[-1])
        elif tta_idx == 2:
            # Brightness adjustment
            return image * 1.1
        elif tta_idx == 3:
            # Darken
            return image * 0.9
        elif tta_idx == 4:
            # Center crop + resize
            b, c, h, w = image.shape
            crop_size = int(h * 0.95)
            start = (h - crop_size) // 2
            cropped = image[:, :, start:start+crop_size, start:start+crop_size]
            return nn.functional.interpolate(cropped, size=(h, w), mode='bilinear')
        else:
            return image
    
    @torch.no_grad()
    def predict_video(
        self,
        frames: List[torch.Tensor],
        aggregation: str = 'median'
    ) -> Tuple[float, List[float]]:
        """
        Predict video from multiple frames
        
        Args:
            frames: List of frame tensors
            aggregation: 'mean', 'median', 'max', or 'voting'
            
        Returns:
            Tuple of (video_prediction, frame_predictions)
        """
        frame_preds = []
        
        for frame in frames:
            pred = self.predict_frame(frame, apply_tta=self.use_tta)
            frame_preds.append(pred)
        
        # Aggregate frame predictions to video-level
        if aggregation == 'mean':
            video_pred = np.mean(frame_preds)
        elif aggregation == 'median':
            video_pred = np.median(frame_preds)
        elif aggregation == 'max':
            video_pred = np.max(frame_preds)
        elif aggregation == 'voting':
            # Majority vote (threshold at 0.5)
            votes = [1 if p > 0.5 else 0 for p in frame_preds]
            video_pred = 1.0 if sum(votes) > len(votes) / 2 else 0.0
        else:
            video_pred = np.mean(frame_preds)
        
        return video_pred, frame_preds


class InferencePipeline:
    """
    Complete inference pipeline from raw videos to predictions
    """
    
    def __init__(
        self,
        ensemble_predictor: EnsemblePredictor,
        preprocessor: VideoPreprocessor,
        batch_size: int = 16
    ):
        """
        Args:
            ensemble_predictor: Ensemble predictor instance
            preprocessor: Video preprocessor
            batch_size: Batch size for frame prediction
        """
        self.predictor = ensemble_predictor
        self.preprocessor = preprocessor
        self.batch_size = batch_size
    
    def predict_video_file(
        self,
        video_path: str,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Predict single video file
        
        Args:
            video_path: Path to video file
            video_name: Video name (for output)
            
        Returns:
            Dictionary with prediction results
        """
        if video_name is None:
            video_name = Path(video_path).stem
        
        # Preprocess video
        frames, metadata = self.preprocessor.process_video(video_path)
        
        if len(frames) == 0:
            return {
                'video_name': video_name,
                'prediction': 0.0,  # Default to real if no faces detected
                'confidence': 0.0,
                'num_frames': 0,
                'error': 'No faces detected'
            }
        
        # Convert frames to tensors
        transform = get_validation_augmentation()
        frame_tensors = []
        
        for frame in frames:
            # Handle RGB only (3 channels)
            if frame.shape[-1] > 3:
                frame = frame[:, :, :3]  # Take only RGB channels
            
            augmented = transform(image=frame)
            frame_tensor = augmented['image']
            frame_tensors.append(frame_tensor)
        
        # Predict
        video_pred, frame_preds = self.predictor.predict_video(
            frames=frame_tensors,
            aggregation='median'
        )
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(video_pred - 0.5) * 2
        
        return {
            'video_name': video_name,
            'prediction': float(video_pred),
            'confidence': float(confidence),
            'num_frames': len(frames),
            'frame_predictions': [float(p) for p in frame_preds],
            'label': 1 if video_pred > 0.5 else 0
        }
    
    def predict_dataset(
        self,
        video_paths: List[str],
        output_csv: str = 'predictions.csv'
    ) -> pd.DataFrame:
        """
        Predict entire dataset and save results
        
        Args:
            video_paths: List of video file paths
            output_csv: Output CSV file path
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        print(f"\nPredicting {len(video_paths)} videos...")
        
        for video_path in tqdm(video_paths):
            video_name = Path(video_path).stem
            
            try:
                result = self.predict_video_file(video_path, video_name)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
                results.append({
                    'video_name': video_name,
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'label': 0,
                    'error': str(e)
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save predictions
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        # Save submission format (filename, label, probability)
        submission_path = output_path.parent / f"submission_{output_path.stem}.csv"
        submission_df = df[['video_name', 'label', 'prediction']].copy()
        submission_df.columns = ['filename', 'label', 'probability']
        submission_df['filename'] = submission_df['filename'] + '.mp4'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved to: {submission_path}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"  Total videos: {len(df)}")
        print(f"  Predicted as REAL: {(df['label'] == 0).sum()}")
        print(f"  Predicted as FAKE: {(df['label'] == 1).sum()}")
        print(f"  Average confidence: {df['confidence'].mean():.3f}")
        
        return df


def create_ensemble_from_cv_models(
    model_name: str,
    cv_dir: str,
    n_folds: int = 5,
    device: str = 'cuda',
    input_channels: int = 3
) -> List[Dict]:
    """
    Create ensemble config from CV-trained models
    
    Args:
        model_name: Model architecture name
        cv_dir: Directory containing CV model checkpoints
        n_folds: Number of folds
        device: Device
        input_channels: Number of input channels
        
    Returns:
        List of model configs for ensemble
    """
    cv_dir = Path(cv_dir)
    model_configs = []
    
    # Equal weight for all folds
    weight_per_fold = 1.0 / n_folds
    
    for fold_idx in range(1, n_folds + 1):
        checkpoint_path = cv_dir / f"{model_name}_fold{fold_idx}_best.pth"
        
        if checkpoint_path.exists():
            model_configs.append({
                'model_name': model_name,
                'checkpoint_path': str(checkpoint_path),
                'weight': weight_per_fold,
                'input_channels': input_channels
            })
    
    return model_configs


if __name__ == "__main__":
    # Example: Create ensemble from Xception CV models
    model_configs = create_ensemble_from_cv_models(
        model_name='xception',
        cv_dir='models/xception_cv',
        n_folds=5,
        input_channels=3
    )
    
    # Create predictor
    predictor = EnsemblePredictor(
        model_configs=model_configs,
        device='cuda',
        use_tta=True,
        tta_steps=5
    )
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(
        num_frames=16,
        img_size=224,
        extract_frequency=False
    )
    
    # Create pipeline
    pipeline = InferencePipeline(
        ensemble_predictor=predictor,
        preprocessor=preprocessor
    )
    
    # Example prediction
    # results = pipeline.predict_dataset(
    #     video_paths=['test_video.mp4'],
    #     output_csv='predictions.csv'
    # )
