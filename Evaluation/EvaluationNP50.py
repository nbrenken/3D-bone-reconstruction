import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import your existing classes/functions from the training file.
# Rename this import to match your actual filename.
#
# Example:
# if your training file is ThirdModelOWNResnetTraining.py:
from ThirdModelOWNResnetTraining import (
    SpineDataset,
    SpineDataloader,
    ResNetFeatureExtractor,
    Refiner,
    ReconstructionModel,
)


def segmentation_metrics_per_sample_from_logits(
    logits,
    targets,
    threshold=0.75,
    smooth=1e-5,
):
    """
    Computes several binary 3D segmentation metrics per sample.

    Parameters
    ----------
    logits : torch.Tensor
        Shape: (B, 1, D, H, W)
        Raw model outputs before sigmoid.

    targets : torch.Tensor
        Shape: (B, 1, D, H, W)
        Ground-truth binary voxel masks.

    threshold : float
        Probability threshold used to binarize predictions.

    smooth : float
        Small value to avoid division by zero.

    Returns
    -------
    metrics : dict
        Dictionary of numpy arrays, each of shape (B,).

    preds_np : np.ndarray
        Binary predictions as numpy array of shape (B, 1, D, H, W).
    """

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    dims = (1, 2, 3, 4)

    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - preds) * targets).sum(dim=dims)
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=dims)

    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)

    iou = (tp + smooth) / (tp + fp + fn + smooth)

    precision = (tp + smooth) / (tp + fp + smooth)

    recall = (tp + smooth) / (tp + fn + smooth)

    specificity = (tn + smooth) / (tn + fp + smooth)

    false_positive_rate = (fp + smooth) / (fp + tn + smooth)

    false_negative_rate = (fn + smooth) / (fn + tp + smooth)

    volume_error = (pred_sum - target_sum) / (target_sum + smooth)

    abs_volume_error = torch.abs(volume_error)

    metrics = {
        "dice": dice.detach().cpu().numpy(),
        "iou": iou.detach().cpu().numpy(),
        "precision": precision.detach().cpu().numpy(),
        "recall": recall.detach().cpu().numpy(),
        "specificity": specificity.detach().cpu().numpy(),
        "false_positive_rate": false_positive_rate.detach().cpu().numpy(),
        "false_negative_rate": false_negative_rate.detach().cpu().numpy(),
        "volume_error": volume_error.detach().cpu().numpy(),
        "abs_volume_error": abs_volume_error.detach().cpu().numpy(),
    }

    return metrics, preds.detach().cpu().numpy()


def summarize_metric(values, metric_name):
    """
    Prints summary statistics for one metric.
    """

    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        print(f"{metric_name:30s}: no values")
        return

    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    median_value = float(np.median(values))
    min_value = float(np.min(values))
    max_value = float(np.max(values))

    print(f"{metric_name:30s}:")
    print(f"  Mean   : {mean_value:.6f}")
    print(f"  Std    : {std_value:.6f}")
    print(f"  Median : {median_value:.6f}")
    print(f"  Min    : {min_value:.6f}")
    print(f"  Max    : {max_value:.6f}")


def evaluate_test_set(
    model,
    dataloader,
    dataset,
    device,
    projection_matrices,
    min_dice_to_keep=0.3,
    threshold=0.75,
    output_csv=None,
):
    model.eval()

    all_rows = []

    all_metrics = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "false_positive_rate": [],
        "false_negative_rate": [],
        "volume_error": [],
        "abs_volume_error": [],
    }

    sample_offset = 0

    with torch.no_grad():
        for batch_idx, (drr_images, voxels, vertebra_idx) in enumerate(dataloader):
            drr_images = drr_images.to(device)
            voxels = voxels.to(device).float()
            vertebra_idx = vertebra_idx.to(device)

            target = voxels.unsqueeze(1)  # (B, 1, 80, 80, 80)

            logits = model(
                drr_images,
                projection_matrices,
                vertebra_idx,
                grid_size=80,
            )

            metrics, preds_np = segmentation_metrics_per_sample_from_logits(
                logits,
                target,
                threshold=threshold,
            )

            batch_size = drr_images.shape[0]

            for local_i in range(batch_size):
                global_i = sample_offset + local_i
                sample = dataset.samples[global_i]

                gt_np = voxels[local_i].detach().cpu().numpy().astype(bool)
                pred_np = preds_np[local_i, 0].astype(bool)

                row = {
                    "index": global_i,
                    "patient": sample["patient"],
                    "vertebra": sample["vertebra"],

                    "dice": float(metrics["dice"][local_i]),
                    "iou": float(metrics["iou"][local_i]),
                    "precision": float(metrics["precision"][local_i]),
                    "recall": float(metrics["recall"][local_i]),
                    "specificity": float(metrics["specificity"][local_i]),
                    "false_positive_rate": float(metrics["false_positive_rate"][local_i]),
                    "false_negative_rate": float(metrics["false_negative_rate"][local_i]),
                    "volume_error": float(metrics["volume_error"][local_i]),
                    "abs_volume_error": float(metrics["abs_volume_error"][local_i]),

                    "gt_voxels": int(gt_np.sum()),
                    "pred_voxels": int(pred_np.sum()),
                    "gt_fill_percent": float(100.0 * gt_np.mean()),
                    "pred_fill_percent": float(100.0 * pred_np.mean()),
                }

                all_rows.append(row)

                for metric_name in all_metrics.keys():
                    all_metrics[metric_name].append(row[metric_name])

                print(
                    f"[{global_i + 1:04d}/{len(dataset):04d}] "
                    f"{row['patient']} / {row['vertebra']} | "
                    f"Dice = {row['dice']:.4f} | "
                    f"IoU = {row['iou']:.4f} | "
                    f"Precision = {row['precision']:.4f} | "
                    f"Recall = {row['recall']:.4f} | "
                    f"VolErr = {row['volume_error']:.4f} | "
                    f"GT fill = {row['gt_fill_percent']:.2f}% | "
                    f"Pred fill = {row['pred_fill_percent']:.2f}%"
                )

            sample_offset += batch_size

    all_dice = np.asarray(all_metrics["dice"], dtype=np.float64)

    kept_mask = all_dice >= min_dice_to_keep
    excluded_mask = all_dice < min_dice_to_keep

    kept_dice = all_dice[kept_mask]
    excluded_dice = all_dice[excluded_mask]

    mean_dice_all = float(np.mean(all_dice)) if len(all_dice) else float("nan")
    std_dice_all = float(np.std(all_dice, ddof=1)) if len(all_dice) > 1 else 0.0

    mean_dice_kept = float(np.mean(kept_dice)) if len(kept_dice) else float("nan")
    std_dice_kept = float(np.std(kept_dice, ddof=1)) if len(kept_dice) > 1 else 0.0

    print("\n" + "=" * 70)
    print("TEST SET SUMMARY")
    print("=" * 70)
    print(f"Voxel probability threshold : {threshold}")
    print(f"Minimum Dice to keep        : {min_dice_to_keep}")
    print()
    print(f"All samples                 : {len(all_dice)}")
    print(f"Kept samples                : {len(kept_dice)}")
    print(f"Excluded samples            : {len(excluded_dice)}")
    print()

    print("DICE FILTER SUMMARY")
    print("-" * 70)
    print(f"Mean Dice, all samples      : {mean_dice_all:.6f}")
    print(f"Std Dice, all samples       : {std_dice_all:.6f}")
    print(f"Min Dice, all samples       : {np.min(all_dice):.6f}")
    print(f"Max Dice, all samples       : {np.max(all_dice):.6f}")
    print()
    print(f"Mean Dice, kept samples     : {mean_dice_kept:.6f}")
    print(f"Std Dice, kept samples      : {std_dice_kept:.6f}")
    print()

    print("ALL-SAMPLE METRIC SUMMARY")
    print("-" * 70)

    for metric_name, values in all_metrics.items():
        summarize_metric(values, metric_name)

    print("=" * 70)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "patient",
                    "vertebra",

                    "dice",
                    "iou",
                    "precision",
                    "recall",
                    "specificity",
                    "false_positive_rate",
                    "false_negative_rate",
                    "volume_error",
                    "abs_volume_error",

                    "gt_voxels",
                    "pred_voxels",
                    "gt_fill_percent",
                    "pred_fill_percent",
                ],
            )
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\nSaved per-sample metric values to:")
        print(output_csv)

    return mean_dice_all, std_dice_all, all_rows


def main():
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    test_dir = Path(
        r"C:\Users\karlo\OneDrive\Desktop\DinoVert\ShirleySTUFF\test_set"
    )

    cache_dir = Path(
        r"C:\Users\karlo\OneDrive\Desktop\DinoVert\ShirleySTUFF\voxel_cache"
    )

    model_checkpoint = Path(
        r"C:\Users\karlo\OneDrive\Desktop\DinoVert\ShirleySTUFF\models_ownresnet50\best_model.pth"
    )

    output_csv = Path(
        r"C:\Users\karlo\OneDrive\Desktop\DinoVert\ShirleySTUFF\EVALNP50\test_segmentation_metrics_resultsOWNRES.csv"
    )

    # Same threshold as in your existing evaluate_dice function.
    threshold = 0.75
    min_dice_to_keep = 0.3

    batch_size = 1
    num_workers = 0  # safer on Windows, especially with preloaded image cache

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Checks
    # -------------------------------------------------------------------------
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    if not cache_dir.exists():
        raise FileNotFoundError(f"Voxel cache directory not found: {cache_dir}")

    if not model_checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    # -------------------------------------------------------------------------
    # Dataset: use ALL samples in test_set, no train/val/test split
    # -------------------------------------------------------------------------
    dataset = SpineDataset(test_dir, cache_dir)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No valid test samples found in {test_dir}. "
            "Check folder structure, DRR files, .npz files, projection matrices, and voxel cache names."
        )

    print(f"Found {len(dataset)} test samples.")

    pin_memory = device.type == "cuda"

    test_loader = SpineDataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    feature_extractor = ResNetFeatureExtractor(
        resnet_type="resnet50",
        device=device,
        train_backbone=True,
    ).to(device)

    refiner = Refiner().to(device)

    model = ReconstructionModel(feature_extractor, refiner).to(device)

    print(f"Loading trained model from: {model_checkpoint}")
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)

    projection_matrices = dataset.projection_matrices.to(device)

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    evaluate_test_set(
        model=model,
        dataloader=test_loader,
        dataset=dataset,
        device=device,
        projection_matrices=projection_matrices,
        min_dice_to_keep=min_dice_to_keep,
        threshold=threshold,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()