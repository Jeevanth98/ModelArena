# 🚀 Google Colab Setup - Deepfake Detection Training
# Copy each cell into Google Colab and run sequentially

---

## 📊 **What You'll Get:**

| Output File | Contains | Format |
|-------------|----------|--------|
| **predictions.csv** | Detailed predictions with probabilities | `video_name, prediction, label, confidence, num_frames` |
| **submission_predictions.csv** | Submission format with probabilities | `filename, label, probability` |
| **TRAINED_MODEL.pth** | Best model weights | PyTorch state dict (~84MB) |
| **cv_results.json** | Cross-validation metrics | JSON with fold-wise results |
| **validation_results.csv** | Validation set predictions | `video_name, true_label, predicted_label, probability, confidence` |

**Key Output Columns:**
- **prediction**: Probability score (0.0-1.0, where 0=REAL, 1=FAKE)
- **label**: Binary prediction (0=REAL, 1=FAKE)
- **confidence**: How confident the model is (0.0-1.0)

---

## Cell 1: Environment Setup (2 minutes)
```python
# Check GPU availability
!nvidia-smi

# Install all dependencies
!pip install -q torch torchvision timm facenet-pytorch albumentations 
!pip install -q opencv-python opencv-contrib-python scikit-learn scipy
!pip install -q tqdm pandas matplotlib seaborn

import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

---

## Cell 2: Mount Drive & Upload Code (5 minutes)
```python
from google.colab import drive
drive.mount('/content/drive')

# Option A: Upload ZIP file from local
from google.colab import files
uploaded = files.upload()  # Upload ModelArena.zip
!unzip -q ModelArena.zip
%cd ModelArena

# Option B: Clone from GitHub (if you pushed it)
# !git clone https://github.com/YOUR_USERNAME/ModelArena.git
# %cd ModelArena

# Verify structure
!ls -la
!ls -la src/
!ls -la data/train/real/ | head -5
!ls -la data/train/fake/ | head -5
```

---

## Cell 3: Verify Imports (30 seconds)
```python
import sys
sys.path.append('/content/ModelArena')

# Test all critical imports
try:
    from src.models_advanced import create_model, XceptionNetDeepfake
    from src.dataset import DeepfakeFrameDataset, get_training_augmentation
    from src.training_utils import FocalLoss, MetricsCalculator
    from src.train_cv import CrossValidationTrainer
    from src.preprocessing_advanced import VideoPreprocessor, create_frame_dataset_from_videos
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    raise
```

---

## Cell 4: Run Preprocessing (30-60 minutes) ⏳
```python
from src.preprocessing_advanced import create_frame_dataset_from_videos
from pathlib import Path

# Count videos
real_count = len(list(Path('data/train/real').glob('*.mp4')))
fake_count = len(list(Path('data/train/fake').glob('*.mp4')))
print(f"Found {real_count} real + {fake_count} fake = {real_count + fake_count} total videos")

# Start preprocessing
print("\n🔄 Starting preprocessing...")
results = create_frame_dataset_from_videos(
    data_dir='data',
    output_dir='data/processed',
    num_frames=16,  # Extract 16 frames per video
    extract_frequency=False  # Set True for frequency features (slower)
)

# Verify results
print(f"\n✅ Preprocessing complete!")
print(f"  Success: {results['success_count']}/{real_count + fake_count}")
print(f"  Failed: {len(results['failed'])}")

# Check saved frames
saved_frames = list(Path('data/processed').glob('*/*.jpg'))
print(f"  Total frames saved: {len(saved_frames)}")
print(f"  Expected: ~{(real_count + fake_count) * 16}")

# Save results to Drive (backup)
!mkdir -p /content/drive/MyDrive/ModelArena_backup
!cp -r data/processed /content/drive/MyDrive/ModelArena_backup/
print("\n💾 Preprocessed data backed up to Drive")
```

---

## Cell 5: Test Dataset Loading (1 minute)
```python
from src.dataset import DeepfakeFrameDataset, get_training_augmentation
from pathlib import Path
import json

# Load metadata
with open('data/processed/preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Metadata loaded: {metadata['total_processed']} videos")

# Create test dataset
test_dataset = DeepfakeFrameDataset(
    video_data=metadata['videos'][:10],  # Test with first 10 videos
    data_root=Path('data/processed'),
    transform=get_training_augmentation(),
    is_training=True,
    use_frequency=False
)

print(f"✅ Dataset created: {len(test_dataset)} frames")

# Test loading one sample
image, label = test_dataset[0]
print(f"✅ Sample loaded: shape={image.shape}, label={label.item()}")
```

---

## Cell 6: Train XceptionNet with 5-Fold CV (4-5 hours) ⏳⏳⏳
```python
from src.train_cv import CrossValidationTrainer
import torch

# Initialize trainer
trainer = CrossValidationTrainer(
    model_name='xception',           # Primary model
    n_folds=5,                       # 5-fold cross-validation
    num_epochs=30,                   # 30 epochs per fold
    batch_size=32,                   # Safe for 12GB GPU
    learning_rate=1e-4,              # Conservative LR
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dir='models/xception_cv',
    early_stopping_patience=7,       # Stop if no improvement for 7 epochs
    use_focal_loss=True,             # Handle class imbalance
    use_mixup=True,                  # MixUp augmentation
    input_channels=3                 # RGB only (no frequency)
)

# Start training
print("\n🚀 Starting XceptionNet 5-Fold CV Training...")
print("⏱️  Estimated time: 4-5 hours")
print("="*70)

xception_results = trainer.train_cross_validation(
    metadata_file='data/processed/preprocessing_metadata.json',
    data_root='data/processed'
)

# Print final results
print("\n" + "="*70)
print("📊 XceptionNet Final Results:")
print("="*70)
for fold_idx, metrics in enumerate(xception_results['fold_results']):
    print(f"Fold {fold_idx + 1}:")
    print(f"  Best Accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"  Best AUC-ROC:  {metrics['best_val_auc_roc']:.4f}")

print(f"\nMean Accuracy: {xception_results['mean_accuracy']:.4f} ± {xception_results['std_accuracy']:.4f}")
print(f"Mean AUC-ROC:  {xception_results['mean_auc_roc']:.4f} ± {xception_results['std_auc_roc']:.4f}")

# Backup to Drive
!cp -r models/xception_cv /content/drive/MyDrive/ModelArena_backup/
print("\n💾 Models backed up to Drive")
```

---

## Cell 7: Train EfficientNetV2 (Optional - 3-4 hours) ⏳⏳
```python
from src.train_cv import CrossValidationTrainer
import torch

# Initialize trainer
trainer2 = CrossValidationTrainer(
    model_name='efficientnet_v2_s',  # Secondary model (faster)
    n_folds=5,
    num_epochs=30,
    batch_size=48,                   # Can use larger batch size
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dir='models/efficientnet_cv',
    early_stopping_patience=7,
    use_focal_loss=True,
    use_mixup=True,
    input_channels=3
)

# Start training
print("\n🚀 Starting EfficientNetV2 5-Fold CV Training...")
efficientnet_results = trainer2.train_cross_validation(
    metadata_file='data/processed/preprocessing_metadata.json',
    data_root='data/processed'
)

# Print results
print(f"\nMean Accuracy: {efficientnet_results['mean_accuracy']:.4f}")
print(f"Mean AUC-ROC:  {efficientnet_results['mean_auc_roc']:.4f}")

# Backup
!cp -r models/efficientnet_cv /content/drive/MyDrive/ModelArena_backup/
```

---

## Cell 8: Create Ensemble & Generate Predictions (10 minutes)
```python
from src.inference import EnsemblePredictor, InferencePipeline
from pathlib import Path
import torch

# Create ensemble from trained models
model_paths = []

# Add XceptionNet models (5 folds)
for fold in range(5):
    model_path = f'models/xception_cv/fold_{fold}/best_model.pth'
    if Path(model_path).exists():
        model_paths.append((model_path, 'xception'))

# Add EfficientNet models (5 folds) - if trained
for fold in range(5):
    model_path = f'models/efficientnet_cv/fold_{fold}/best_model.pth'
    if Path(model_path).exists():
        model_paths.append((model_path, 'efficientnet_v2_s'))

print(f"✅ Found {len(model_paths)} trained models for ensemble")

# Initialize inference pipeline
pipeline = InferencePipeline(
    model_paths=model_paths,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_tta=True,  # Test-Time Augmentation
    tta_num_augmentations=5
)

# Generate predictions on test set
# Assuming test videos are in data/test/
test_videos = sorted(list(Path('data/test').glob('*.mp4')))
print(f"Found {len(test_videos)} test videos")

# Run inference
predictions_df = pipeline.predict_dataset(
    video_paths=[str(v) for v in test_videos],
    output_csv='predictions.csv'
)

print("\n✅ Predictions saved!")
print("\nSample predictions:")
print(predictions_df[['video_name', 'prediction', 'label', 'confidence']].head(10))

print("\n📊 Output files created:")
print("  1. predictions.csv - Detailed results (5 columns)")
print("  2. submission_predictions.csv - Submission format (3 columns: filename, label, probability)")

print("\n📊 Output columns in predictions.csv:")
print("  - video_name: Video filename (without .mp4)")
print("  - prediction: Probability score (0.0-1.0, higher = more likely FAKE)")
print("  - label: Binary prediction (0=REAL, 1=FAKE)")
print("  - confidence: Prediction confidence (0.0-1.0)")
print("  - num_frames: Number of frames processed")

# Copy to Drive
!cp predictions.csv /content/drive/MyDrive/ModelArena_backup/
!cp submission_predictions.csv /content/drive/MyDrive/ModelArena_backup/
!cp models/xception_cv/fold_0/best_model.pth /content/drive/MyDrive/ModelArena_backup/TRAINED_MODEL.pth
print("\n💾 All files backed up to Drive")
```

---

## Cell 9: Evaluate on Validation Set (5 minutes)
```python
from src.inference import InferencePipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd
import json

# Load train labels for validation
train_df = pd.read_csv('data/train_labels.csv')

# Use last fold as final validation
val_videos = train_df.tail(120)  # Last 20% as validation

# Get predictions
val_predictions = []
val_labels = []
val_results = []

for _, row in val_videos.iterrows():
    video_path = f"data/train/{'fake' if row['label'] == 1 else 'real'}/{row['video_name']}.mp4"
    if Path(video_path).exists():
        result = pipeline.predict_video_file(video_path, row['video_name'])
        val_predictions.append(result['prediction'])  # Probability
        val_labels.append(row['label'])
        val_results.append({
            'video_name': row['video_name'],
            'true_label': row['label'],
            'predicted_label': result['label'],
            'probability': result['prediction'],
            'confidence': result['confidence']
        })

# Calculate metrics
accuracy = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_predictions])
auc_roc = roc_auc_score(val_labels, val_predictions)

print("\n" + "="*70)
print("📊 FINAL VALIDATION METRICS (On Held-out Data)")
print("="*70)
print(f"Accuracy:  {accuracy:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_report(val_labels, [1 if p > 0.5 else 0 for p in val_predictions]))

# Show some example predictions
print("\n" + "="*70)
print("SAMPLE VALIDATION PREDICTIONS:")
print("="*70)
val_df = pd.DataFrame(val_results)
for idx, row in val_df.head(10).iterrows():
    true_text = "FAKE" if row['true_label'] == 1 else "REAL"
    pred_text = "FAKE" if row['predicted_label'] == 1 else "REAL"
    match = "✅" if row['true_label'] == row['predicted_label'] else "❌"
    print(f"{match} {row['video_name']}: True={true_text}, Pred={pred_text} (prob={row['probability']:.4f}, conf={row['confidence']:.4f})")

# Save validation results
val_df.to_csv('validation_results.csv', index=False)
print("\n💾 Validation results saved to validation_results.csv")
```

---

## Cell 10: Visualize Predictions (Optional - 2 minutes)
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_csv('predictions.csv')

# Display sample predictions
print("=" * 80)
print("SAMPLE PREDICTIONS:")
print("=" * 80)
for idx, row in df.head(10).iterrows():
    label_text = "FAKE" if row['label'] == 1 else "REAL"
    print(f"{row['video_name']}: {label_text} (probability={row['prediction']:.4f}, confidence={row['confidence']:.4f})")

print(f"\n" + "=" * 80)
print(f"SUMMARY: {len(df)} videos")
print(f"  REAL (label=0): {(df['label'] == 0).sum()} videos")
print(f"  FAKE (label=1): {(df['label'] == 1).sum()} videos")
print(f"  Average confidence: {df['confidence'].mean():.3f}")
print("=" * 80)

# Plot probability distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of probabilities
axes[0].hist(df['prediction'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
axes[0].set_xlabel('Prediction Probability (0=Real, 1=Fake)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Prediction Probabilities')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Histogram of confidence scores
axes[1].hist(df['confidence'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Confidence Score')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Prediction Confidence')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualization saved as 'prediction_analysis.png'")

# Show most confident predictions
print("\n🔥 Most Confident FAKE Predictions:")
fake_confident = df[df['label'] == 1].nlargest(5, 'confidence')
for idx, row in fake_confident.iterrows():
    print(f"  {row['video_name']}: prob={row['prediction']:.4f}, conf={row['confidence']:.4f}")

print("\n✅ Most Confident REAL Predictions:")
real_confident = df[df['label'] == 0].nlargest(5, 'confidence')
for idx, row in real_confident.iterrows():
    print(f"  {row['video_name']}: prob={row['prediction']:.4f}, conf={row['confidence']:.4f}")

print("\n⚠️  Least Confident Predictions (Uncertain):")
uncertain = df.nsmallest(5, 'confidence')
for idx, row in uncertain.iterrows():
    label_text = "FAKE" if row['label'] == 1 else "REAL"
    print(f"  {row['video_name']}: {label_text}, prob={row['prediction']:.4f}, conf={row['confidence']:.4f}")
```

---

## Cell 11: Download Results (1 minute)
```python
from google.colab import files
import shutil

# Create submission package
!mkdir -p submission
!cp models/xception_cv/fold_0/best_model.pth submission/TRAINED_MODEL.pth
!cp submission_predictions.csv submission/predictions.csv
!cp predictions.csv submission/predictions_detailed.csv

# Copy this notebook as INFERENCE_PIPELINE.ipynb
# (You'll need to download the notebook manually: File → Download → .ipynb)

# Copy documentation
!cp README.md submission/
!cp ERROR_FIXES.md submission/

# Create ZIP
!zip -r submission.zip submission/

# Download
files.download('submission.zip')

print("✅ Download complete!")
print("\n📦 Submission package contents:")
print("  ✅ TRAINED_MODEL.pth - Best model weights")
print("  ✅ predictions.csv - Submission format (filename, label, probability)")
print("  ✅ predictions_detailed.csv - Full format (with confidence, num_frames)")
print("  ⚠️  INFERENCE_PIPELINE.ipynb - Download this notebook manually")
print("  📄 Documentation (README, ERROR_FIXES)")
```

---

## ⚠️ Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution:** Run Cell 3 to add ModelArena to Python path

### Issue: "FileNotFoundError: data/processed/..."
**Solution:** Re-run Cell 4 (preprocessing). Check if frames were saved.

### Issue: "CUDA out of memory"
**Solution:** Reduce batch_size from 32 to 16 in Cell 6

### Issue: "Too slow on CPU"
**Solution:** Runtime → Change runtime type → GPU

### Issue: "Session disconnected"
**Solution:** Models are saved to Drive every fold. Resume from last checkpoint.

---

## ⏱️ Total Time Estimate

| Stage | Time | Can Skip? |
|-------|------|-----------|
| Setup (Cells 1-3) | 5 min | No |
| Preprocessing (Cell 4) | 45 min | No |
| XceptionNet Training (Cell 6) | 5 hours | No |
| EfficientNet Training (Cell 7) | 4 hours | Yes (optional) |
| Inference (Cell 8) | 10 min | No |
| Evaluation (Cell 9) | 5 min | No |

**Minimum time:** ~6 hours (XceptionNet only)  
**Full pipeline:** ~10 hours (XceptionNet + EfficientNet ensemble)

---

## 📄 **Output Files Explained**

### **predictions.csv** (Detailed results)
```csv
video_name,prediction,label,confidence,num_frames
video_001,0.9234,1,0.8468,16
video_002,0.1567,0,0.6866,16
video_003,0.7891,1,0.5782,16
```

**Columns:**
- **video_name**: Video filename (without .mp4 extension)
- **prediction**: Probability score (0.0 to 1.0)
  - `0.0` = 100% confident REAL
  - `0.5` = Uncertain
  - `1.0` = 100% confident FAKE
- **label**: Binary prediction
  - `0` = REAL video
  - `1` = FAKE video (deepfake)
- **confidence**: Confidence level (0.0 to 1.0)
  - Distance from 0.5, scaled: `abs(prediction - 0.5) * 2`
  - `1.0` = Very confident, `0.0` = Uncertain
- **num_frames**: Number of frames successfully processed

### **submission_predictions.csv** (Submission format)
```csv
filename,label,probability
video_001.mp4,1,0.9234
video_002.mp4,0,0.1567
video_003.mp4,1,0.7891
```
**Three-column format for submission:**
- **filename**: Video name with .mp4 extension
- **label**: Binary classification (0=REAL, 1=FAKE)
- **probability**: Prediction score (0.0 to 1.0)

---

## 🎯 Expected Performance

| Metric | Expected | Competitive |
|--------|----------|-------------|
| Accuracy | 94-96% | >93% |
| AUC-ROC | 0.96-0.98 | >0.95 |
| F1-Score | 0.94-0.96 | >0.93 |

---

*Ready to start? Copy Cell 1 into Google Colab and go! 🚀*
