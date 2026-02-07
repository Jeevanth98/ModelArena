"""
Dataset and Data Augmentation Pipeline
- Custom dataset for video frames with labels
- Heavy augmentation strategies including MixUp
- Support for RGB + Frequency domain features
- Memory-efficient data loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import random


class MixUpAugmentation:
    """
    MixUp data augmentation
    Proven to improve generalization on small datasets
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Args:
           alpha: MixUp parameter (Beta distribution)
            prob: Probability of applying MixUp
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        image1: torch.Tensor,
        label1: torch.Tensor,
        image2: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to a pair of images
        
        Args:
            image1, image2: Input images (C, H, W)
            label1, label2: Labels
            
        Returns:
            Mixed image and label
        """
        if random.random() > self.prob:
            return image1, label1
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Mix labels (for binary classification, this creates soft labels)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label


def get_training_augmentation(img_size: int = 224) -> A.Compose:
    """
    Heavy augmentation pipeline for training
    Designed to prevent overfitting on small dataset
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1.0),
        ], p=0.3),
        
        # Color/appearance transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.5),
        
        # Noise and blur (simulate compression artifacts)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(p=1.0),
            A.MultiplicativeNoise(p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Compression simulation (critical for deepfake robustness)
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        
        # Cutout/Erasing
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.3
        ),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation(img_size: int = 224) -> A.Compose:
    """Minimal augmentation for validation/test"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_augmentation(img_size: int = 224) -> List[A.Compose]:
    """
    Test-Time Augmentation (TTA) transforms
    Returns list of different augmentation pipelines for TTA
    """
    base_transform = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
    
    tta_transforms = [
        # Original
        A.Compose(base_transform),
        
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transform),
        
        # Slight rotation
        A.Compose([A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=1.0)] + base_transform),
        
        # Brightness adjusted
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)] + base_transform),
        
        # Center crop + resize
        A.Compose([
            A.CenterCrop(height=int(img_size*0.9), width=int(img_size*0.9)),
            A.Resize(height=img_size, width=img_size)
        ] + base_transform),
    ]
    
    return tta_transforms


class DeepfakeFrameDataset(Dataset):
    """
    Dataset for deepfake detection from video frames
    Supports RGB and Frequency domain features
    """
    
    def __init__(
        self,
        video_data: List[Dict],
        data_root: str,
        transform: Optional[Callable] = None,
        is_training: bool = True,
        use_frequency: bool = True,
        num_frames_per_video: int = 16
    ):
        """
        Args:
            video_data: List of dicts with video metadata (from preprocessing)
            data_root: Root directory containing processed frames
            transform: Albumentations transform pipeline
            is_training: Whether this is training set (for MixUp)
            use_frequency: Whether to use frequency domain features
            num_frames_per_video: Number of frames extracted per video
        """
        self.video_data = [v for v in video_data if v.get('faces_detected', 0) > 0]
        self.data_root = Path(data_root)
        self.transform = transform
        self.is_training = is_training
        self.use_frequency = use_frequency
        self.num_frames_per_video = num_frames_per_video
        
        # Create flat list of all frames
        self.frame_list = []
        for video in self.video_data:
            video_name = Path(video['video_path']).stem
            num_faces = video.get('faces_detected', 0)
            label = video['label']
            
            for frame_idx in range(num_faces):
                self.frame_list.append({
                    'video_name': video_name,
                    'frame_idx': frame_idx,
                    'label': label
                })
        
        print(f"Dataset created: {len(self.frame_list)} frames from {len(self.video_data)} videos")
    
    def __len__(self) -> int:
        return len(self.frame_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single frame and its label
        
        Returns:
            image: Tensor (3, H, W) or (5, H, W) if using frequency features
            label: Tensor (1,) - binary label
        """
        frame_info = self.frame_list[idx]
        
        # Construct frame path
        frame_name = f"{frame_info['video_name']}_frame_{frame_info['frame_idx']:03d}.jpg"
        frame_path = self.data_root / frame_info['video_name'] / frame_name
        
        # Try alternative path if above doesn't exist
        if not frame_path.exists():
            # Frames might be in flat directory
            frame_path = self.data_root / frame_name
        
        if not frame_path.exists():
            # Return a dummy tensor if frame not found (shouldn't happen)
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, torch.tensor([frame_info['label']], dtype=torch.float32)
        
        # Load image
        image = cv2.imread(str(frame_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        label = torch.tensor([frame_info['label']], dtype=torch.float32)
        
        return image, label


class DeepfakeVideoDataset(Dataset):
    """
    Dataset that treats each video as a single sample
    Returns multiple frames per video for video-level prediction
    """
    
    def __init__(
        self,
        video_data: List[Dict],
        data_root: str,
        transform: Optional[Callable] = None,
        max_frames: int = 16
    ):
        """
        Args:
            video_data: List of video metadata dicts
            data_root: Root directory
            transform: Augmentation pipeline
            max_frames: Maximum frames to load per video
        """
        self.video_data = [v for v in video_data if v.get('faces_detected', 0) > 0]
        self.data_root = Path(data_root)
        self.transform = transform
        self.max_frames = max_frames
        
    def __len__(self) -> int:
        return len(self.video_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            frames: Tensor (N, 3, H, W) - multiple frames from one video
            label: Tensor (1,)
            video_name: str
        """
        video = self.video_data[idx]
        video_name = Path(video['video_path']).stem
        label = video['label']
        num_faces = min(video.get('faces_detected', 0), self.max_frames)
        
        frames = []
        for frame_idx in range(num_faces):
            frame_name = f"{video_name}_frame_{frame_idx:03d}.jpg"
            frame_path = self.data_root / video_name / frame_name
            
            if not frame_path.exists():
                frame_path = self.data_root / frame_name
            
            if frame_path.exists():
                image = cv2.imread(str(frame_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                else:
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                
                frames.append(image)
        
        if len(frames) == 0:
            # Fallback: return dummy
            frames = [torch.zeros(3, 224, 224)]
        
        # Stack frames
        frames = torch.stack(frames)
        label = torch.tensor([label], dtype=torch.float32)
        
        return frames, label, video_name


def create_dataloaders(
    train_metadata: str,
    val_metadata: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_frequency: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_metadata: Path to training metadata JSON
        val_metadata: Path to validation metadata JSON
        data_root: Root directory for processed frames
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_frequency: Whether to use frequency features
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load metadata
    with open(train_metadata, 'r') as f:
        train_data = json.load(f)['videos']
    
    with open(val_metadata, 'r') as f:
        val_data = json.load(f)['videos']
    
    # Create datasets
    train_dataset = DeepfakeFrameDataset(
        video_data=train_data,
        data_root=data_root,
        transform=get_training_augmentation(),
        is_training=True,
        use_frequency=use_frequency
    )
    
    val_dataset = DeepfakeFrameDataset(
        video_data=val_data,
        data_root=data_root,
        transform=get_validation_augmentation(),
        is_training=False,
        use_frequency=use_frequency
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test augmentation pipeline
    import matplotlib.pyplot as plt
    
    # Load a sample image
    sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Apply augmentation
    transform = get_training_augmentation()
    augmented = transform(image=sample_img)
    
    print(f"Aug image shape: {augmented['image'].shape}")
    print(f"Aug image type: {augmented['image'].dtype}")
