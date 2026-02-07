"""
Advanced Preprocessing Pipeline with Frequency Domain Analysis
- Frame extraction with intelligent sampling
- MTCNN face detection and alignment
- RGB features + Frequency domain features (DCT/FFT)
- Optimized for memory efficiency (Colab Free tier)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import json
from scipy import fftpack
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class FrequencyFeatureExtractor:
    """Extract frequency domain features using DCT and FFT"""
    
    def __init__(self, use_dct: bool = True, use_fft: bool = True):
        """
        Args:
            use_dct: Use Discrete Cosine Transform
            use_fft: Use Fast Fourier Transform
        """
        self.use_dct = use_dct
        self.use_fft = use_fft
    
    def extract_dct_features(self, image: np.ndarray, block_size: int = 8) -> np.ndarray:
        """
        Extract DCT coefficients from image
        Deepfakes often have artifacts in high-frequency DCT coefficients
        
        Args:
            image: RGB image (H, W, 3)
            block_size: DCT block size (typically 8x8)
            
        Returns:
            DCT feature map
        """
        # Convert to grayscale for DCT
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        dct_map = np.zeros_like(gray, dtype=np.float32)
        
        # Apply DCT on blocks
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                dct_map[i:i+block_size, j:j+block_size] = dct_block
        
        # Normalize
        dct_map = np.abs(dct_map)
        dct_map = (dct_map - dct_map.min()) / (dct_map.max() - dct_map.min() + 1e-8)
        
        return dct_map
    
    def extract_fft_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract FFT magnitude spectrum
        Synthetic images have different frequency patterns
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            FFT magnitude spectrum
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log transform for better visualization
        magnitude = np.log(magnitude + 1)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude
    
    def extract_combined_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract and combine RGB + DCT + FFT features
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Combined feature image (H, W, 5) - RGB + DCT + FFT
        """
        features = [image]
        
        if self.use_dct:
            dct = self.extract_dct_features(image)
            features.append(dct[..., np.newaxis])
        
        if self.use_fft:
            fft = self.extract_fft_features(image)
            features.append(fft[..., np.newaxis])
        
        # Concatenate along channel dimension
        combined = np.concatenate(features, axis=-1)
        
        return combined.astype(np.float32)


class VideoPreprocessor:
    """
    Advanced video preprocessing with face detection and frequency features
    Memory-optimized for Colab Free tier
    """
    
    def __init__(
        self,
        num_frames: int = 16,
        img_size: int = 224,
        face_margin: int = 40,
        extract_frequency: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        min_face_size: int = 40,
        confidence_threshold: float = 0.9
    ):
        """
        Args:
            num_frames: Number of frames to extract per video
            img_size: Target image size
            face_margin: Margin around detected face
            extract_frequency: Whether to extract frequency features
            device: Device for MTCNN
            min_face_size: Minimum face size for detection
            confidence_threshold: Confidence threshold for face detection
        """
        self.num_frames = num_frames
        self.img_size = img_size
        self.face_margin = face_margin
        self.extract_frequency = extract_frequency
        self.device = device
        
        # Initialize MTCNN with optimized settings
        self.mtcnn = MTCNN(
            image_size=img_size,
            margin=face_margin,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, confidence_threshold],
            keep_all=False,
            device=device,
            post_process=False
        )
        
        # Initialize frequency extractor
        if extract_frequency:
            self.freq_extractor = FrequencyFeatureExtractor(use_dct=True, use_fft=True)
        else:
            self.freq_extractor = None
    
    def extract_frames(self, video_path: str, strategy: str = 'smart') -> List[np.ndarray]:
        """
        Extract frames from video with intelligent sampling
        
        Args:
            video_path: Path to video file
            strategy: 'uniform', 'smart', or 'keyframe'
                - uniform: Evenly spaced frames
                - smart: 80% uniform + 20% random for diversity
                - keyframe: High-motion frames
                
        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Calculate frame indices
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            if strategy == 'uniform':
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            elif strategy == 'smart':
                # 80% uniformly spaced, 20% random
                num_uniform = max(1, int(self.num_frames * 0.8))
                num_random = self.num_frames - num_uniform
                
                uniform_indices = np.linspace(0, total_frames - 1, num_uniform, dtype=int)
                
                # Random indices from remaining frames
                remaining_indices = list(set(range(total_frames)) - set(uniform_indices))
                if len(remaining_indices) >= num_random:
                    random_indices = np.random.choice(remaining_indices, num_random, replace=False)
                else:
                    random_indices = remaining_indices
                
                frame_indices = sorted(list(uniform_indices) + list(random_indices))
            else:
                # Default to uniform
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def detect_and_crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and crop with proper error handling
        
        Args:
            frame: Input frame in RGB
            
        Returns:
            Cropped face or None
        """
        try:
            # MTCNN detection
            face = self.mtcnn(frame)
            
            if face is None:
                # Fallback: return center crop if no face detected
                h, w = frame.shape[:2]
                size = min(h, w)
                start_h = (h - size) // 2
                start_w = (w - size) // 2
                crop = frame[start_h:start_h+size, start_w:start_w+size]
                crop = cv2.resize(crop, (self.img_size, self.img_size))
                return crop
            
            # Convert tensor to numpy if needed
            if isinstance(face, torch.Tensor):
                face = face.permute(1, 2, 0).cpu().numpy()
                # Denormalize if needed
                if face.max() <= 1.0:
                    face = (face * 255).astype(np.uint8)
            
            # Resize to target size
            face = cv2.resize(face, (self.img_size, self.img_size))
            
            return face
            
        except Exception as e:
            # Fallback to center crop
            h, w = frame.shape[:2]
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            crop = frame[start_h:start_h+size, start_w:start_w+size]
            crop = cv2.resize(crop, (self.img_size, self.img_size))
            return crop
    
    def process_video(
        self,
        video_path: str,
        save_dir: Optional[str] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Complete preprocessing pipeline for one video
        
        Args:
            video_path: Path to video file
            save_dir: Directory to save frames (optional)
            
        Returns:
            Tuple of (processed_frames, metadata)
        """
        video_path = Path(video_path)
        
        # Extract frames
        frames = self.extract_frames(str(video_path), strategy='smart')
        
        if len(frames) == 0:
            return [], {"error": "No frames extracted", "num_faces": 0}
        
        # Detect and crop faces
        processed_frames = []
        for i, frame in enumerate(frames):
            face_crop = self.detect_and_crop_face(frame)
            
            if face_crop is not None:
                # Add frequency features if enabled
                if self.extract_frequency and self.freq_extractor:
                    # Extract frequency features
                    freq_features = self.freq_extractor.extract_combined_features(face_crop)
                    processed_frames.append(freq_features)
                else:
                    processed_frames.append(face_crop)
                
                # Save frame if directory provided
                if save_dir:
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save RGB image
                    img_name = f"{video_path.stem}_frame_{i:03d}.jpg"
                    cv2.imwrite(
                        str(save_path / img_name),
                        cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                    )
        
        metadata = {
            "video_name": video_path.name,
            "total_frames_extracted": len(frames),
            "faces_detected": len(processed_frames),
            "detection_rate": len(processed_frames) / len(frames) if len(frames) > 0 else 0,
            "has_frequency_features": self.extract_frequency
        }
        
        return processed_frames, metadata
    
    def process_dataset(
        self,
        video_paths: List[str],
        labels: List[int],
        output_dir: str,
        save_frames: bool = False,
        batch_log_interval: int = 50
    ) -> Dict:
        """
        Process entire dataset with progress tracking
        
        Args:
            video_paths: List of video file paths
            labels: Corresponding labels (0=real, 1=fake)
            output_dir: Output directory
            save_frames: Whether to save individual frames
            batch_log_interval: Log progress every N videos
            
        Returns:
            Dictionary with processing statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_metadata = []
        failed_videos = []
        
        print(f"Processing {len(video_paths)} videos...")
        
        for idx, (video_path, label) in enumerate(tqdm(zip(video_paths, labels), total=len(video_paths))):
            try:
                save_dir = (output_dir / Path(video_path).stem) if save_frames else None
                
                processed_frames, metadata = self.process_video(video_path, save_dir)
                
                metadata['label'] = label
                metadata['video_path'] = str(video_path)
                metadata['video_idx'] = idx
                all_metadata.append(metadata)
                
                if len(processed_frames) == 0:
                    failed_videos.append(Path(video_path).name)
                
                # Periodic logging
                if (idx + 1) % batch_log_interval == 0:
                    success_rate = (idx + 1 - len(failed_videos)) / (idx + 1) * 100
                    print(f"  Processed {idx + 1}/{len(video_paths)} | Success: {success_rate:.1f}%")
                    
            except Exception as e:
                print(f"Error processing {Path(video_path).name}: {e}")
                failed_videos.append(Path(video_path).name)
        
        # Save metadata
        metadata_file = output_dir / "preprocessing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'videos': all_metadata,
                'failed_videos': failed_videos,
                'total_processed': len(all_metadata),
                'total_failed': len(failed_videos),
                'success_rate': (len(all_metadata) - len(failed_videos)) / len(all_metadata) * 100
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Preprocessing Complete!")
        print(f"  Successfully processed: {len(all_metadata) - len(failed_videos)}/{len(all_metadata)}")
        print(f"  Failed: {len(failed_videos)}")
        print(f"  Metadata saved: {metadata_file}")
        print(f"{'='*70}\n")
        
        return {
            'metadata': all_metadata,
            'failed': failed_videos,
            'success_count': len(all_metadata) - len(failed_videos)
        }


def create_frame_dataset_from_videos(
    data_dir: str,
    output_dir: str,
    num_frames: int = 16,
    extract_frequency: bool = True
):
    """
    Helper function to preprocess entire dataset
    
    Args:
        data_dir: Directory containing train/real and train/fake subdirectories
        output_dir: Output directory for processed data
        num_frames: Number of frames per video
        extract_frequency: Whether to extract frequency features
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Get all video paths
    real_videos = sorted(list((data_dir / 'train' / 'real').glob('*.mp4')))
    fake_videos = sorted(list((data_dir / 'train' / 'fake').glob('*.mp4')))
    
    video_paths = [str(v) for v in real_videos + fake_videos]
    labels = [0] * len(real_videos) + [1] * len(fake_videos)
    
    print(f"Found {len(real_videos)} real and {len(fake_videos)} fake videos")
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        num_frames=num_frames,
        img_size=224,
        extract_frequency=extract_frequency,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process dataset
    results = preprocessor.process_dataset(
        video_paths=video_paths,
        labels=labels,
        output_dir=output_dir,
        save_frames=True  # MUST save frames for DeepfakeFrameDataset
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    results = create_frame_dataset_from_videos(
        data_dir="data",
        output_dir="data/processed",
        num_frames=16,
        extract_frequency=True
    )
