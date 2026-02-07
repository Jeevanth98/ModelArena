"""
Quick Dataset Analysis Script
Run this to understand video properties before training
"""

import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

def analyze_video(video_path):
    """Extract metadata from a single video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        cap.release()
        
        return {
            'filename': video_path.name,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration_sec': duration,
            'file_size_mb': file_size_mb,
            'resolution': f"{width}x{height}"
        }
    except Exception as e:
        print(f"Error analyzing {video_path.name}: {e}")
        return None

def analyze_dataset(data_dir, sample_size=20):
    """
    Analyze dataset videos
    
    Args:
        data_dir: Path to data directory containing train/real and train/fake
        sample_size: Number of videos to sample from each class
    """
    data_dir = Path(data_dir)
    
    results = {
        'real': [],
        'fake': []
    }
    
    # Analyze real videos
    real_dir = data_dir / 'train' / 'real'
    fake_dir = data_dir / 'train' / 'fake'
    
    print("Analyzing REAL videos...")
    real_videos = list(real_dir.glob("*.mp4"))[:sample_size]
    for video_path in tqdm(real_videos):
        info = analyze_video(video_path)
        if info:
            results['real'].append(info)
    
    print("\nAnalyzing FAKE videos...")
    fake_videos = list(fake_dir.glob("*.mp4"))[:sample_size]
    for video_path in tqdm(fake_videos):
        info = analyze_video(video_path)
        if info:
            results['fake'].append(info)
    
    # Compute statistics
    print("\n" + "="*70)
    print("DATASET ANALYSIS RESULTS")
    print("="*70)
    
    for label in ['real', 'fake']:
        videos = results[label]
        if len(videos) == 0:
            continue
            
        print(f"\n{label.upper()} Videos (sampled {len(videos)}):")
        print("-" * 70)
        
        # Resolution statistics
        resolutions = [v['resolution'] for v in videos]
        unique_res = list(set(resolutions))
        print(f"  Resolutions: {unique_res}")
        
        # Width/Height stats
        widths = [v['width'] for v in videos]
        heights = [v['height'] for v in videos]
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")
        
        # FPS stats
        fps_values = [v['fps'] for v in videos]
        print(f"  FPS: min={min(fps_values):.1f}, max={max(fps_values):.1f}, avg={np.mean(fps_values):.1f}")
        
        # Duration stats
        durations = [v['duration_sec'] for v in videos]
        print(f"  Duration: min={min(durations):.1f}s, max={max(durations):.1f}s, avg={np.mean(durations):.1f}s")
        
        # Frame count stats
        frames = [v['frame_count'] for v in videos]
        print(f"  Frames: min={min(frames)}, max={max(frames)}, avg={np.mean(frames):.0f}")
        
        # File size stats
        sizes = [v['file_size_mb'] for v in videos]
        print(f"  File Size: min={min(sizes):.2f}MB, max={max(sizes):.2f}MB, avg={np.mean(sizes):.2f}MB")
    
    # Save detailed results
    output_file = data_dir / 'video_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Detailed analysis saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Recommendations
    print("📋 RECOMMENDATIONS:")
    all_videos = results['real'] + results['fake']
    avg_frames = np.mean([v['frame_count'] for v in all_videos])
    avg_duration = np.mean([v['duration_sec'] for v in all_videos])
    
    # Suggest frame sampling
    if avg_frames < 50:
        sample_frames = int(avg_frames * 0.5)
    elif avg_frames < 150:
        sample_frames = 15
    else:
        sample_frames = 20
    
    print(f"  ✓ Suggested frames to extract: {sample_frames} per video")
    print(f"  ✓ This will create ~{600 * sample_frames:,} training images")
    print(f"  ✓ Estimated preprocessing time: {(600 * avg_duration / 60):.1f} minutes")
    
    return results

if __name__ == "__main__":
    # Run analysis
    data_dir = Path("data")
    
    if not (data_dir / "train" / "real").exists():
        print("Error: Cannot find data/train/real directory")
        print("Please ensure videos are in the correct location")
    else:
        results = analyze_dataset(data_dir, sample_size=30)
