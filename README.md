# X-ray Pose Estimation Models

This repository contains the code, pretrained model checkpoints, data-generation pipeline, evaluation scripts, and visualization tools used to train and compare five model architectures for X-ray-based prediction tasks.

## Repository Structure

```text
.
├── 3DVisualization/
├── AdditionalDAXData/
├── Datageneration/
├── Evaluation/
├── Models/
└── Training/
```

## Folders

### `3DVisualization/`

Contains Python code for visualizing model predictions in 3D and comparing them with the corresponding ground-truth pose or geometry.

Use this folder to inspect qualitative model performance, including prediction-to-ground-truth alignment and spatial deviations.

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
| DAX18 | DAX18 evaluation script |
| DAX50 | DAX50 evaluation script |
| ResNet18 (non-pretrained) | ResNet18 evaluation script |
| ResNet50 (non-pretrained) | ResNet50 evaluation script |
| ViT8 | ViT8 evaluation script |

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
| DAX18 | DAX18 training script |
| DAX50 | DAX50 training script |
| ResNet18 (non-pretrained) | ResNet18 training script |
| ResNet50 (non-pretrained) | ResNet50 training script |
| ViT8 | ViT8 training script |

Each training script is responsible for configuring the data pipeline, initializing the relevant architecture, training the model, and saving the resulting checkpoint.

## Models Compared

The repository compares the following models:

1. **DAX18**
2. **DAX50**
3. **Non-pretrained ResNet18**
4. **Non-pretrained ResNet50**
5. **ViT8**

The DAX models use components adapted from the external DAX project, while the ResNet and ViT models serve as comparison baselines.

## Typical Workflow

```text
1. Generate data
   Datageneration/

2. Train one of the five models
   Training/

3. Save or load checkpoints
   Models/

4. Evaluate the trained model
   Evaluation/

5. Inspect predictions visually against ground truth
   3DVisualization/
```

## External Dependencies

This project depends on external code and frameworks, including:

- DAX: https://github.com/JoshuaScheuplein/DAX
- DiffDRR: https://github.com/eigenvivek/DiffDRR

Install the dependencies required by the individual scripts before training, evaluating, or generating data. The exact package versions should be documented in a `requirements.txt` or environment file if reproducibility is required.

## Notes

- Ensure that model paths and dataset paths in the scripts are adjusted to match your local environment.
- Run the data-generation pipeline before training if the required generated dataset is not already available.
- Use the matching evaluation script and `.pth` checkpoint for each architecture.
- The DAX-related code in `AdditionalDAXData/` should retain the relevant attribution to the original DAX repository.
