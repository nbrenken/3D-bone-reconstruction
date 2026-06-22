# X-ray Pose Estimation Models

This repository contains the code, pretrained model checkpoints, data-generation pipeline, evaluation scripts, and visualization tools used to train and compare five model architectures for X-ray-based prediction tasks.

## Repository Structure

```text
.
├── 3DVisualization/
├── AdditionalDAXData/
├── DataExample/
├── Datageneration/
├── Evaluation/
├── Models/
└── Training/
```

## Folders

### `3DVisualization/`

Contains Python code for visualizing model predictions in 3D and comparing them with the corresponding ground-truth pose or geometry.

Use this folder to inspect qualitative model performance, including prediction-to-ground-truth alignment and spatial deviations.

### `DataExample/`

Contains one fully processed example case (`1.3.6.1.4.1.9328.50.4.0001`) that can be used to test the pipeline without running data generation first:

```text
DataExample/
├── CT_Scan/               # Raw CT volume and segmentation mask (.nii.gz)
├── Simulated_x_ray/       # Generated DRRs, vertebra masks, and mesh files (.png, .npz)
├── VoxelShape/            # Precomputed voxel cache (.npy)
└── 3D_reconstruction/     # Example 3D reconstruction overlay (.png)
```

To use it, point `SPINE_DATA_DIR` and `SPINE_TEST_DIR` to `DataExample/Simulated_x_ray/` and `SPINE_CACHE_DIR` to `DataExample/VoxelShape/`.

### `AdditionalDAXData/`

Contains backbone definitions and supporting code required for the DAX-based models.

These components were not developed in this project. They originate from the DAX repository, which adapts DINO-style architectures for X-ray data:

- DAX repository: https://github.com/JoshuaScheuplein/DAX

Please consult the original DAX repository for implementation details, licensing, and attribution requirements.

### `Datageneration/`

Contains the Python script used to generate the training and evaluation data.

Data generation is based on **DiffDRR**, a differentiable digitally reconstructed radiograph framework:

- DiffDRR repository: https://github.com/eigenvivek/DiffDRR

Please consult the original DiffDRR repository for setup instructions, implementation details, and licensing information.

### `Evaluation/`

Contains five Python evaluation scripts. Each script evaluates one trained model:

| Model | Evaluation script |
|---|---|
| DAX18 | `EvaluationDAX18.py` |
| DAX50 | `EvaluationDAX50.py` |
| ResNet18 (non-pretrained) | `EvaluationNP18.py` |
| ResNet50 (non-pretrained) | `EvaluationNP50.py` |
| ViT8 | `EvaluationVit8.py` |

The scripts load the corresponding checkpoint from `Models/` and compute the evaluation metrics defined in the code.

### `Models/`

Contains the saved model checkpoints in `.pth` format:

| Checkpoint | Description |
|---|---|
| `DAX18` | DAX model with a ResNet18-style backbone |
| `DAX50` | DAX model with a ResNet50-style backbone |
| `ResNet18` | Non-pretrained ResNet18 baseline |
| `ResNet50` | Non-pretrained ResNet50 baseline |
| `ViT8` | Vision Transformer baseline with patch size/configuration `8` |

> The exact checkpoint filenames should be referenced in the corresponding training and evaluation scripts.

### `Training/`

Contains five Python training scripts, one for each architecture:

| Model | Training script |
|---|---|
| DAX18 | `Dax18_Training.py` |
| DAX50 | `DAX50_Training.py` |
| ResNet18 (non-pretrained) | `NP18_Training.py` |
| ResNet50 (non-pretrained) | `NP50_Training.py` |
| ViT8 | `ViT8_Training.py` |

Each training script is responsible for configuring the data pipeline, initializing the relevant architecture, training the model, and saving the resulting checkpoint.

## Models Compared

The repository compares the following models:

1. **DAX18**
2. **DAX50**
3. **Non-pretrained ResNet18**
4. **Non-pretrained ResNet50**
5. **ViT8**

The DAX models use components adapted from the external DAX project, while the ResNet and ViT models serve as comparison baselines.

## Setup

### Environment variables

All scripts use environment variables for paths to external data that cannot ship in the repository. Set them before running any script.

| Variable | Required | Description |
|---|---|---|
| `SPINE_CT_ROOT` | Yes (Datageneration) | Root of the 1K Spine CT dataset — must contain `data/` and `label/` subdirectories. |
| `SPINE_DATA_DIR` | Yes (Training, 3DVisualization) | Path to the DRR output folder produced by `Datageneration/Datageneration.py`. |
| `SPINE_TEST_DIR` | Yes (Evaluation) | Path to the folder containing the test-split DRR data. |
| `SPINE_OUTPUT_DIR` | No | Where `Datageneration.py` writes its output. Defaults to `<repo>/output`. |
| `SPINE_CACHE_DIR` | No | Voxel cache directory used by training and evaluation. Defaults to `<repo>/voxel_cache`. |
| `SPINE_MODEL_DIR` | No | Directory where training scripts save model checkpoints. Defaults to `<repo>/Models`. |
| `SPINE_DAX_CKPT` | No | Override path to a DAX backbone checkpoint. Defaults to the matching file in `AdditionalDAXData/`. |
| `SPINE_MODEL_CKPT` | No | Override path to a trained reconstruction model checkpoint. Defaults to the matching file in `Models/`. |
| `SPINE_OUTPUT_CSV` | No | Override path for the evaluation results CSV. Defaults to `Evaluation/results/metrics_<model>.csv`. |
| `SPINE_CT_VOLUMES` | Yes (3DVisualization) | CT volumes directory (e.g. `.../data/colon`). |
| `SPINE_CT_LABELS` | Yes (3DVisualization) | CT labels directory (e.g. `.../label/colon`). |

### ViT8 note

`ViT8_Training.py` and `EvaluationVit8.py` import `vision_transformer_dax` from `AdditionalDAXData/`. Both files insert that directory onto `sys.path` automatically at import time, so no manual `PYTHONPATH` change is required.

### Dependencies

Install all required packages before running any script. Run:

```bash
pip install -r requirements.txt
```

## Typical Workflow

**Quick start (using the included example data):**

```text
1. Point env vars at DataExample/
   SPINE_DATA_DIR  → DataExample/Simulated_x_ray/
   SPINE_TEST_DIR  → DataExample/Simulated_x_ray/
   SPINE_CACHE_DIR → DataExample/VoxelShape/

2. Train or evaluate directly — skip data generation
```

**Full workflow (your own dataset):**

```text
1. Generate data
   Datageneration/  (requires SPINE_CT_ROOT)

2. Train one of the five models
   Training/        (requires SPINE_DATA_DIR)

3. Save or load checkpoints
   Models/

4. Evaluate the trained model
   Evaluation/      (requires SPINE_TEST_DIR)

5. Inspect predictions visually against ground truth
   3DVisualization/ (requires SPINE_CT_VOLUMES, SPINE_CT_LABELS, SPINE_DATA_DIR)
```

## External Dependencies

This project depends on external code and frameworks, including:

- DAX: https://github.com/JoshuaScheuplein/DAX
- DiffDRR: https://github.com/eigenvivek/DiffDRR

## Notes

- Set the required environment variables (see **Setup** above) before running any script. No hardcoded paths remain in the codebase.
- Run the data-generation pipeline before training if the required generated dataset is not already available.
- Use the matching evaluation script and `.pth` checkpoint for each architecture.
- The DAX-related code in `AdditionalDAXData/` should retain the relevant attribution to the original DAX repository.
