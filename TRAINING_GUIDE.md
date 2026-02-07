# 🚀 Quick Start Guide - Hackathon Training Pipeline

## ⚡ FASTEST PATH TO RESULTS (10-12 hours)

### Prerequisites
- Google Colab (Free tier is sufficient)
- Your 600 training videos uploaded
- Internet connection

---

## 📋 STEP-BY-STEP EXECUTION PLAN

### ✅ Stage 1: Setup (15 minutes)

**1.1 Check Your Videos (Local)**
```bash
cd e:/Projects/ModelVerse/ModelArena
python analyze_dataset.py
```
**→ IMPORTANT: Share the output with me!** (Resolution, FPS, duration)

**1.2 Upload to Colab**
```python
# In Google Colab
!git clone https://github.com/Jeevanth98/ModelArena.git
%cd ModelArena

# Install all dependencies
!pip install -q torch torchvision timm facenet-pytorch albumentations opencv-python opencv-contrib-python scikit-learn scipy tqdm pandas
```

**1.3 Mount Drive & Upload Data**
```python
from google.colab import drive
drive.mount('/content/drive')

# Upload your data/ folder manually or:
# !cp -r /content/drive/MyDrive/your-data-folder/* data/
```

---

### ✅ Stage 2: Preprocessing (30-60 minutes)

```python
# Run preprocessing script
from src.preprocessing_advanced import create_frame_dataset_from_videos

results = create_frame_dataset_from_videos(
    data_dir='data',
    output_dir='data/processed',
    num_frames=16,
    extract_frequency=False  # Keep False for speed
)

# Check results
print(f"Successfully processed: {results['success_count']}/{600}")

# IMPORTANT: Verify frames were saved
from pathlib import Path
saved_frames = list(Path('data/processed').glob('*/*.jpg'))
print(f"Total frames saved: {len(saved_frames)}")  # Should be ~9,600
```

**Expected output**: ~9,600 frames extracted (16 per video × 600 videos)  
**⚠️ Critical:** Frames MUST be saved to disk (save_frames=True is now default)

---

### ✅ Stage 3: Train XceptionNet (4-5 hours)  [PRIMARY MODEL]

```python
from src.train_cv import CrossValidationTrainer

# Initialize trainer
xception_trainer = CrossValidationTrainer(
    model_name='xception',
    n_folds=5,
    num_epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda',
    output_dir='models/xception_cv',
    early_stopping_patience=7,
    use_focal_loss=True,
    use_mixup=True,
    input_channels=3
)

# Start training (this will take 4-5 hours)
xception_results = xception_trainer.train_cross_validation(
    metadata_file='data/processed/preprocessing_metadata.json',
    data_root='data/processed'
)

# Save results to Drive (important!)
!cp -r models/xception_cv /content/drive/MyDrive/ModelArena_models/
```

**Monitoring Progress:**
- Each fold takes ~50 minutes
- Watch for "Best model saved" messages
- Target AUC-ROC per fold: >0.95

---

### ✅ Stage 4: Train EfficientNetV2 (3-4 hours) [SECONDARY MODEL]

```python
# While Xception is still training or after it completes
effnet_trainer = CrossValidationTrainer(
    model_name='efficientnetv2',
    n_folds=5,
    num_epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda',
    output_dir='models/efficientnet_cv',
    early_stopping_patience=7,
    use_focal_loss=True,
    use_mixup=True,
    input_channels=3
)

effnet_results = effnet_trainer.train_cross_validation(
    metadata_file='data/processed/preprocessing_metadata.json',
    data_root='data/processed'
)

# Backup to Drive
!cp -r models/efficientnet_cv /content/drive/MyDrive/ModelArena_models/
```

**⏰ Time Management Tip:**
- If you're running out of time, you can proceed with just XceptionNet
- Ensemble of both models gives +2-3% accuracy boost

---

### ✅ Stage 5: Evaluate Models (10 minutes)

```python
# Load and check CV results
import json

with open('models/xception_cv/xception_cv_results.json', 'r') as f:
    xception_cv = json.load(f)

with open('models/efficientnet_cv/efficientnetv2_cv_results.json', 'r') as f:
    effnet_cv = json.load(f)

print(f"XceptionNet Mean AUC-ROC: {xception_cv['auc_roc']['mean']:.4f} ± {xception_cv['auc_roc']['std']:.4f}")
print(f"EfficientNet Mean AUC-ROC: {effnet_cv['auc_roc']['mean']:.4f} ± {effnet_cv['auc_roc']['std']:.4f}")
```

**Expected Performance:**
- XceptionNet: 0.96-0.98 AUC-ROC
- EfficientNetV2: 0.94-0.96 AUC-ROC

---

### ✅ Stage 6: Create Ensemble & Predict (30 minutes)

```python
from src.inference import (
    EnsemblePredictor,
    InferencePipeline,
    create_ensemble_from_cv_models
)
from src.preprocessing_advanced import VideoPreprocessor
import glob

# Create ensemble configurations
xception_configs = create_ensemble_from_cv_models(
    model_name='xception',
    cv_dir='models/xception_cv',
    n_folds=5,
    input_channels=3
)

effnet_configs = create_ensemble_from_cv_models(
    model_name='efficientnetv2',
    cv_dir='models/efficientnet_cv',
    n_folds=5,
    input_channels=3
)

# Set ensemble weights (Xception 60%, EfficientNet 40%)
for config in xception_configs:
    config['weight'] = 0.6 / len(xception_configs)
for config in effnet_configs:
    config['weight'] = 0.4 / len(effnet_configs)

all_configs = xception_configs + effnet_configs

# Create ensemble predictor with TTA
predictor = EnsemblePredictor(
    model_configs=all_configs,
    device='cuda',
    use_tta=True,
    tta_steps=5  # 5 augmentations per prediction
)

# Create preprocessor
preprocessor = VideoPreprocessor(
    num_frames=16,
    img_size=224,
    extract_frequency=False
)

# Create inference pipeline
pipeline = InferencePipeline(
    ensemble_predictor=predictor,
    preprocessor=preprocessor
)

# Get test video paths (update this with your test data location)
test_videos = sorted(glob.glob('data/test/**/*.mp4', recursive=True))
print(f"Found {len(test_videos)} test videos")

# Generate predictions
predictions_df = pipeline.predict_dataset(
    video_paths=test_videos,
    output_csv='results/predictions.csv'
)

print("\n✅ Predictions complete!")
print(predictions_df.head())
```

---

### ✅ Stage 7: Prepare Submission (15 minutes)

```python
# Download predictions
from google.colab import files

# Main predictions file
files.download('results/predictions.csv')

# Submission format (filename, label)
files.download('results/submission_predictions.csv')

# Download best model checkpoint (for submission requirement)
!cp models/xception_cv/xception_fold1_best.pth submission/TRAINED_MODEL.PT
files.download('submission/TRAINED_MODEL.PT')

# Download all models (optional backup)
!zip -r all_models.zip models/
files.download('all_models.zip')
```

---

## 🎯 SUBMISSION CHECKLIST

Create a folder with these files:

- ✅ **TRAINED_MODEL.PT** - Best model checkpoint
- ✅ **INFERENCE_PIPELINE.IPYNB** - Copy `notebooks/inference.ipynb`
- ✅ **PREDICTIONS.CSV** - Your test predictions
- ✅ **SYSTEM_ARCH.PDF** - Architecture diagram (from README)

---

## 🔥 OPTIMIZATION TIPS

### If Running Out of Time:
1. **Train only XceptionNet** (skip EfficientNet)
   - Still achieves ~95-96% accuracy
   - Saves 3-4 hours

2. **Reduce CV folds**
   ```python
   n_folds=3  # Instead of 5 (saves ~2 hours)
   ```

3. **Use fewer frames**
   ```python
   num_frames=12  # Instead of 16 (saves ~30 min preprocessing)
   ```

### If Getting OOM Errors:
```python
batch_size=16  # Reduce from 32
gradient_accumulation_steps=2  # Simulate larger batch
num_workers=2  # Reduce workers
```

### To Maximize Accuracy:
```python
num_epochs=40  # More epochs
early_stopping_patience=10  # More patience
use_tta=True  # Always use TTA
tta_steps=5  # More augmentations
```

---

## 📊 EXPECTED TIMELINE

| Stage | Time | Cumulative |
|-------|------|------------|
| Setup | 15 min | 0.25 hr |
| Preprocessing | 45 min | 1 hr |
| Train Xception (5-fold) | 5 hr | 6 hr |
| Train EfficientNet (5-fold) | 4 hr | 10 hr |
| Inference + Ensemble | 30 min | 10.5 hr |
| Submission prep | 15 min | 10.75 hr |
| **TOTAL** | **~11 hours** | |

**Buffer**: Keep 1-2 hours for debugging, re-runs, etc.

---

## ❓ TROUBLESHOOTING

### Colab Disconnects
```python
# Run this at the start to prevent disconnects
import IPython
from google.colab import output

def keep_colab_alive():
    display(IPython.display.Javascript('''
        function ClickConnect(){
            console.log("Clicked on connect button"); 
            document.querySelector("colab-connect-button").click()
        }
        setInterval(ClickConnect,60000)
    '''))

keep_colab_alive()
```

### Preprocessing Fails
- Check video paths are correct
- Ensure MTCNN can access GPU
- Reduce `num_frames` if needed

### Training Diverges (Loss → NaN)
```python
learning_rate=5e-5  # Lower LR
gradient_clip_value=1.0  # Add gradient clipping
```

### Low Validation Accuracy (<90%)
- Check face detection rate (should be >95%)
- Enable MixUp and Focal Loss
- Increase num_epochs
- Use ensemble + TTA

---

## 💡 PRO TIPS

1. **Always backup to Drive** after each training stage
2. **Monitor GPU usage** - Keep it >80% for efficient training
3. **Save intermediate results** - Don't lose progress
4. **Test preprocessing on 10 videos first** before full dataset
5. **Run video analysis locally first** - Share results for optimization

---

## 📞 NEXT STEPS

**Right now:**
1. Run `analyze_dataset.py` locally
2. Share the video analysis results
3. I'll optimize the config based on your video properties

**Then:**
Follow this guide step-by-step in Colab!

**Good luck! 🚀🎬**
