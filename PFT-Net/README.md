# PFT-Net: Prototypes Filtering and Transformation for Few-shot Medical Image Segmentation

This repository contains the implementation of **PFT-Net**, a few-shot learning framework for medical image segmentation.

## Overview

PFT-Net is designed for few-shot medical image segmentation tasks. Key features:
- **Prototype Filtering**: Iteratively filters and selects high-quality prototypes based on feature similarity
- **Prototype Transformation**: Uses Optimal Transport to transform and align prototypes with query features
- **Adaptive Thresholding**: Learns instance-specific thresholds for segmentation

Supported datasets: CHAOST2 (abdominal MRI), CMR (cardiac MRI), and SABS (abdominal CT).

## Requirements

- Python 3.7+
- PyTorch 1.0+
- CUDA-capable GPU (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize datasets in the following structure:

```
data/
├── CHAOST2/
│   └── chaos_MR_T2_normalized/
│       ├── image_*.nii.gz
│       ├── label_*.nii.gz
│       └── supervoxels_5000/
│           └── supervoxel_*.nii.gz
├── CMR/
│   └── cmr_MR_normalized/
│       ├── image_*.nii.gz
│       ├── label_*.nii.gz
│       └── supervoxels_1000/
│           └── supervoxel_*.nii.gz
└── SABS/
    └── sabs_CT_normalized/
        ├── image_*.nii.gz
        ├── label_*.nii.gz
        └── supervoxels_5000/
            └── supervoxel_*.nii.gz
```

**Label Encoding:**
- CHAOST2: Liver (1), Right Kidney (2), Left Kidney (3), Spleen (4)
- CMR: LV-MYO (1), LV-BP (2), RV (3)
- SABS: Multiple organs (1-13)

## Usage

### Training

```bash
# Using shell scripts
bash exps/train_CHAOST2_setting1.sh
bash exps/train_CMR.sh
bash exps/train_SABS_setting1.sh
```

Or run directly:
```bash
python train_PFT.py with \
  mode='train' \
  dataset='CHAOST2' \
  eval_fold=0 \
  n_steps=35000 \
  n_shot=1 \
  use_gt=False \
  test_label=[1,2,3,4] \
  seed=2021
```

### Testing

```bash
# Using shell script
bash exps/test_CHAOST2.sh
```

Or run directly:
```bash
python test_main.py with \
  mode='test' \
  dataset='CHAOST2' \
  eval_fold=0 \
  supp_idx=2 \
  n_part=3 \
  reload_model_path='path/to/checkpoint.pth' \
  test_label=[1,2,3,4]
```

### Key Parameters

- `dataset`: 'CHAOST2', 'CMR', or 'SABS'
- `n_shot`: Number of support samples (default: 1)
- `eval_fold`: Cross-validation fold (0-4)
- `use_gt`: Use ground truth (True) or supervoxels (False) for training
- `n_sv`: Number of supervoxels (5000 for CHAOST2/SABS, 1000 for CMR)

## Project Structure

```
PFT-Net/
├── config.py                 # Configuration file
├── train_PFT.py              # Training script
├── test_main.py              # Testing script
├── utils.py                  # Utility functions
├── models/
│   ├── fewshot_PFT.py        # Main model with OT attention
│   ├── encoder_sort.py       # ResNet101 encoder
│   └── loss.py               # Loss functions
├── dataloaders/
│   ├── datasets.py           # Dataset classes
│   ├── dataset_specifics.py  # Dataset configurations
│   └── image_transforms.py   # Data augmentation
└── exps/
    ├── train_*.sh            # Training scripts
    └── test_*.sh              # Testing scripts
```

## Evaluation Metrics

The model reports Dice Score and IoU (Intersection over Union) per class and averaged across test cases.
