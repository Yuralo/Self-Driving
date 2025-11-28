# Self-Driving Training Project

This project demonstrates **Imitation Learning** (Behavior Cloning) for autonomous driving using real-world datasets and a complex CNN + Transformer architecture.

## Quick Start

1. **Install Requirements**:
   ```bash
   pip install torch torchvision pandas pillow matplotlib opencv-python gdown
   ```

2. **Run Training**:
   ```bash
   python3 run.py
   ```
   This script will:
   - Inspect and print sample datapoints from all datasets.
   - Combine Udacity and Comma2k19 datasets.
   - Train the model for 2 epochs.

## Architecture

- **Spatial Encoder**: Nvidia-style CNN for extracting features from each frame
- **Temporal Encoder**: Transformer Encoder to process sequences of 5 frames
- **Policy Head**: MLP to predict steering angles

## Datasets

### 1. Udacity Self-Driving Car Dataset
- **Format**: Image sequences + CSV
- **Size**: ~8,000 samples
- **Location**: `data/data/`

### 2. Comma2k19 Dataset
- **Format**: HEVC Video + NumPy logs
- **Size**: ~83,000 samples
- **Location**: `data/comma2k19/`

## Project Structure

```
/Users/bahaa/dev/RL/training/
├── run.py                      # Main entry point
├── src/
│   ├── data/
│   │   ├── comma_dataset.py    # Comma2k19 loader
│   │   ├── udacity_dataset.py  # Udacity loader
│   │   └── ...
│   ├── models/
│   │   └── policy.py           # CNN + Transformer policy
│   ├── train_multi.py          # Training logic
│   └── visualization/
│       └── ...
├── data/                       # Dataset storage
└── models/                     # Saved checkpoints
```

## Visualization

### Training Progress
```bash
PYTHONPATH=. python3 src/visualization/plot_progress.py
```

### Dataset Distribution
```bash
PYTHONPATH=. python3 src/visualization/dataset_stats.py
```
