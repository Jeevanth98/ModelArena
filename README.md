# 🎬 Deepfake Video Classification - ModelArena

**Advanced deepfake detection system using ensemble of Vision Transformers and CNNs**

## 🚀 Quick Start (Google Colab)

### Step 1: Run Video Analysis (Local - Optional)
```bash
python analyze_dataset.py
```
This analyzes your video properties (resolution, FPS, duration) - **share results with me if you run this!**

### Step 2: Upload to Colab & Install Dependencies

```python
# In Colab
!git clone https://github.com/Jeevanth98/ModelArena.git
%cd ModelArena

# Install requirements
!pip install -q torch torchvision timm facan et-pytorch albumentations opencv-python scikit-learn scipy tqdm pandas
```

### Step 3: Upload Your Dataset to Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Option A: Upload directly to Colab (if dataset is small)
from google.colab import files
# Upload your data/ folder

# Option B: Use Google Drive (recommended for larger datasets)
# Copy data to /content/drive/MyDrive/ModelArena/data/
```

### Step 4: Preprocessing (Extract Frames & Detect Faces)

```python
# Run preprocessing
from src.preprocessing_advanced import create_frame_dataset_from_videos

results = create_frame_dataset_from_videos(
    data_dir='data',
    output_dir='data/processed',
    num_frames=16,  # Extract 16 frames per video
    extract_frequency=False  # Set True to use RGB+Frequency features
)

# This will take ~30-60 minutes for 600 videos
```

### Step 5: Train Models with 5-Fold CV

**Option A: Train XceptionNet (Primary Model - ~4-5 hours)**

```python
from src.train_cv import CrossValidationTrainer

xception_trainer = CrossValidationTrainer(
    model_name='xception',
    n_folds=5,
    num_epochs=30,
    batch_size=32,  # Reduce to 16 if OOM
    learning_rate=1e-4,
    device='cuda',
    output_dir='models/xception_cv',
    use_focal_loss=True,
    use_mixup=True,
    input_channels=3
)

xception_results = xception_trainer.train_cross_validation(
    metadata_file='data/processed/preprocessing_metadata.json',
    data_root='data/processed'
)
```

**Option B: Train EfficientNetV2 (Secondary Model - ~3-4 hours)**

```python
effnet_trainer = CrossValidationTrainer(
    model_name='efficientnetv2',
    n_folds=5,
    num_epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda',
    output_dir='models/efficientnet_cv',
    use_focal_loss=True,
    use_mixup=True,
    input_channels=3
)

effnet_results = effnet_trainer.train_cross_validation(
    metadata_file='data/processed/preprocessing_metadata.json',
    data_root='data/processed'
)
```

### Step 6: Create Ensemble & Make Predictions

```python
from src.inference import (
    EnsemblePredictor, InferencePipeline,
    create_ensemble_from_cv_models
)
from src.preprocessing_advanced import VideoPreprocessor

# Create ensemble from trained models
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

# Combine with custom weights (Xception 60%, EfficientNet 40%)
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
    tta_steps=5
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

# Predict test videos
import glob
test_videos = glob.glob('data/test/**/*.mp4', recursive=True)

predictions_df = pipeline.predict_dataset(
    video_paths=test_videos,
    output_csv='results/predictions.csv'
)

# Two CSV files are generated:
# 1. predictions.csv - Detailed results with 5 columns:
#    - video_name, prediction, label, confidence, num_frames
#
# 2. submission_predictions.csv - Submission format with 3 columns:
#    - filename, label, probability
#    where:
#      - filename = video name with .mp4 extension
#      - label = 0 (REAL) or 1 (FAKE)
#      - probability = prediction score (0.0-1.0)
```

### Step 7: Download Results

```python
from google.colab import files

# Download predictions
files.download('results/predictions.csv')              # Detailed (5 columns)
files.download('results/submission_predictions.csv')   # Submission format (3 columns: filename, label, probability)

# Download trained models
!zip -r models.zip models/
files.download('models.zip')
```

---

## 📊 System Architecture

```
Input: .mp4 Videos (600 train, N test)
    ↓
[Preprocessing Pipeline]
    ├─ Extract 16 frames/video (uniform + random sampling)
    ├─ MTCNN face detection & cropping
    ├─ Resize to 224x224
    └─ Optional: DCT/FFT frequency features
    ↓
[Data Augmentation]
    ├─ Geometric: flip, rotation, distortion
    ├─ Color: brightness, contrast, hue
    ├─ Noise: Gaussian, ISO noise
    ├─ Compression simulation (JPEG 60-100%)
    ├─ MixUp (α=0.2)
    └─ Cutout/CoarseDropout
    ↓
[Model Training - 5-Fold CV]
    ├─ Model 1: XceptionNet (60% weight)
    │   ├─ ImageNet pre-trained
    │   ├─ Focal Loss (α=0.25, γ=2.0)
    │   ├─ AdamW optimizer (lr=1e-4)
    │   └─ CosineLR scheduler
    ├─ Model 2: EfficientNetV2-S (40% weight)
    │   └─ Same training config
    └─ Early stopping (patience=7)
    ↓
[Ensemble Inference]
    ├─ Load 5 folds per model (10 total models)
    ├─ Test-Time Augmentation (5 variants)
    ├─ Weighted ensemble fusion
    └─ Video-level aggregation (median)
    ↓
Output: Predictions.csv (filename, label, confidence)
```

## 🎯 Key Features

### ✅ Optimized for Small Datasets (600 videos)
- **Heavy augmentation** (10+ techniques)
- **5-fold cross-validation** for robust evaluation
- **Focal Loss** for hard example mining
- **MixUp** for regularization
- **Label smoothing** to prevent overconfidence

### ✅ State-of-the-Art Architecture
- **XceptionNet**: Best performer on FaceForensics++ benchmark
- **EfficientNetV2**: Modern, efficient CNN
- **Ensemble fusion**: Combines strengths of multiple models
- **Test-Time Augmentation**: 5 augmentations per prediction

### ✅ Comprehensive Metrics
- Accuracy, Balanced Accuracy
- AUC-ROC (primary metric for imbalanced data)
- Precision, Recall, F1-Score
- Specificity, Confusion Matrix

### ✅ Production-Ready
- Modular, clean codebase
- Extensive error handling
- Progress tracking & logging
- Memory-efficient (works on Colab Free)

---

## 📁 Project Structure

```
ModelArena/
├── data/
│   ├── train/
│   │   ├── real/          # 300 real videos
│   │   └── fake/          # 300 fake videos
│   ├── train_labels.csv
│   └── processed/         # Extracted frames (generated)
├── src/
│   ├── preprocessing_advanced.py    # Video → frames + faces
│   ├── dataset.py                   # PyTorch datasets + augmentation
│   ├── models_advanced.py           # XceptionNet, EfficientNetV2
│   ├── training_utils.py            # Focal Loss, metrics, training loops
│   ├── train_cv.py                  # 5-fold CV training
│   └── inference.py                 # Ensemble + TTA prediction
├── models/                          # Saved model checkpoints
├── results/                         # Predictions & metrics
├── notebooks/                       # Colab notebooks
├── analyze_dataset.py               # Video analysis script
├── requirements.txt
└── README.md
```

---

## 🔧 Advanced Options

### Using Frequency Domain Features

```python
# During preprocessing
results = create_frame_dataset_from_videos(
    data_dir='data',
    output_dir='data/processed_freq',
    num_frames=16,
    extract_frequency=True  # ← Enable frequency features
)

# During training (update input_channels)
trainer = CrossValidationTrainer(
    model_name='xception',
    input_channels=5,  # ← RGB (3) + DCT (1) + FFT (1)
    ...
)
```

### Hyperparameter Tuning

```python
# Aggressive augmentation for even better generalization
trainer = CrossValidationTrainer(
    num_epochs=40,           # More epochs
    batch_size=16,           # Smaller batch (more updates)
    learning_rate=5e-5,      # Lower LR
    weight_decay=1e-4,       # Stronger regularization
    early_stopping_patience=10,
    ...
)
```

### Single-Fold Quick Training (for testing)

```python
# Train on single split (faster, ~45 min)
from src.train_cv import CrossValidationTrainer

trainer = CrossValidationTrainer(
    model_name='xception',
    n_folds=1,  # ← Just  1 fold for quick test
    num_epochs=15,
    ...
)
```

---

## 📈 Expected Performance

Based on similar deepfake detection benchmarks:

| Model | Val Accuracy | Val AUC-ROC | F1-Score |
|-------|-------------|-------------|----------|
| XceptionNet (5-fold CV) | 92-95% | 0.96-0.98 | 0.92-0.95 |
| EfficientNetV2 (5-fold CV) | 90-93% | 0.94-0.96 | 0.90-0.93 |
| **Ensemble (Xception + EfficientNet)** | **94-97%** | **0.97-0.99** | **0.94-0.97** |
| **+ Test-Time Augmentation** | **+1-2%** | **+0.01-0.02** | **+1-2%** |

---

## ⏱️ Time Estimates (Colab Free Tier - T4 GPU)

| Task | Time | Notes |
|------|------|-------|
| Preprocessing (600 videos) | ~30-60 min | Face detection is slowest |
| Train Xception (5-fold) | ~4-5 hours | ~50 min per fold |
| Train EfficientNetV2 (5-fold) | ~3-4 hours | ~40 min per fold |
| Inference (100 videos) | ~10-15 min | With TTA |
| **Total Pipeline** | **~8-10 hours** | Both models + inference |

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Error
```python
# Reduce batch size
batch_size=16  # or even 8

# Or disable frequency features
extract_frequency=False
input_channels=3
```

### Slow Preprocessing
```python
# Reduce frames per video
num_frames=10  # instead of 16

# Or disable frequency extraction
extract_frequency=False
```

### Low Accuracy
- Ensure faces are detected properly (check metadata)
- Increase num_epochs (try 40-50)
- Enable MixUp and Focal Loss
- Use ensemble of both models
- Enable TTA

---

## 📝 Submission Checklist

For hackathon submission, ensure you have:

- [ ] **TRAINED_MODEL.PT** - Best model checkpoint(s)
- [ ] **INFERENCE_PIPELINE.IPYNB** - Notebook showing inference process
- [ ] **PREDICTIONS.CSV** - Test set predictions (filename, label, probability)
- [ ] **SYSTEM_ARCH.PDF** - Architecture diagram (see above)
- [ ] **(Optional)** Training logs, CV results, ablation studies

---

## 🎓 Citation & References

- **XceptionNet**: Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions"
- **FaceForensics++**: Rössler et al. (2019). "FaceForensics++: Learning to Detect Manipulated Facial Images"
- **EfficientNetV2**: Tan & Le (2021). "EfficientNetV2: Smaller Models and Faster Training"
- **Focal Loss**: Lin et al. (2017). "Focal Loss for Dense Object Detection"

---

## 💬 Questions?

Run the video analysis first and share results:
```bash
python analyze_dataset.py
```

Then I can help optimize the preprocessing and training parameters specifically for your dataset!

**Good luck with your hackathon! 🚀**
