# Non pretrained ResNet18 Model Training

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
import numpy as np
import trimesh
from pathlib import Path
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL.Image as Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_fill_holes
from datetime import datetime, timedelta


def plot_voxel_grid_3d(voxels: np.ndarray, max_points: int = 20000):
    """Plot a voxel grid as a sampled point cloud for interactive speed."""
    filled = voxels.astype(bool)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    filled_count = int(filled.sum())
    if filled_count == 0:
        print("Voxel grid is empty; nothing to plot.")
        return

    coords = np.argwhere(filled)
    if coords.shape[0] > max_points:
        sample_idx = np.random.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[sample_idx]
        print(f"Showing {max_points}/{filled_count} occupied voxels for faster rendering.")

    ax.scatter(
        coords[:, 0].tolist(),
        coords[:, 1].tolist(),
        coords[:, 2].tolist(),
        s=1,
        c="steelblue",
        alpha=0.8,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(filled.shape)
    plt.tight_layout()
    plt.show()


def plot_voxel_fixed_view_3d(
    voxels: np.ndarray,
    max_points: int = 200000,
    elev: float = 20,
    azim: float = 35,
    save_path: str = "voxel_fixed_view.png",
    show: bool = False,
):
    """Render a fixed-angle 3D voxel view and optionally save it without interaction."""
    filled = np.squeeze(voxels).astype(bool)
    if filled.ndim != 3:
        raise ValueError(f"plot_voxel_fixed_view_3d expects a 3D voxel grid after squeeze, got shape {filled.shape}")

    filled_count = int(filled.sum())
    if filled_count == 0:
        print("Voxel grid is empty; nothing to plot.")
        return

    coords = np.argwhere(filled)
    if coords.shape[0] > max_points:
        sample_idx = np.random.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[sample_idx]
        print(f"Showing {max_points}/{filled_count} occupied voxels for faster rendering.")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords[:, 0].tolist(),
        coords[:, 1].tolist(),
        coords[:, 2].tolist(),
        s=1,
        c="steelblue",
        alpha=0.8,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("Voxel Grid (Fixed 3D View)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(filled.shape)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved fixed-view 3D plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_voxel_comparison_fixed_view_3d(
    prediction_voxels: np.ndarray,
    ground_truth_voxels: np.ndarray,
    max_points: int = 200000,
    elev: float = 20,
    azim: float = 35,
    save_path: str = "voxel_comparison_fixed_view.png",
):
    """Render prediction and ground truth side-by-side in one non-interactive figure."""
    pred = np.squeeze(prediction_voxels).astype(bool)
    gt = np.squeeze(ground_truth_voxels).astype(bool)

    if pred.ndim != 3:
        raise ValueError(f"plot_voxel_comparison_fixed_view_3d expects prediction to be 3D after squeeze, got shape {pred.shape}")
    if gt.ndim != 3:
        raise ValueError(f"plot_voxel_comparison_fixed_view_3d expects ground truth to be 3D after squeeze, got shape {gt.shape}")

    pred_coords = np.argwhere(pred)
    gt_coords = np.argwhere(gt)

    if pred_coords.shape[0] == 0 and gt_coords.shape[0] == 0:
        print("Both voxel grids are empty; nothing to plot.")
        return

    if pred_coords.shape[0] > max_points:
        sample_idx = np.random.choice(pred_coords.shape[0], size=max_points, replace=False)
        pred_coords = pred_coords[sample_idx]
        print(f"Prediction: showing {max_points}/{int(pred.sum())} occupied voxels for faster rendering.")

    if gt_coords.shape[0] > max_points:
        sample_idx = np.random.choice(gt_coords.shape[0], size=max_points, replace=False)
        gt_coords = gt_coords[sample_idx]
        print(f"Ground truth: showing {max_points}/{int(gt.sum())} occupied voxels for faster rendering.")

    fig = plt.figure(figsize=(14, 7))
    ax_pred = fig.add_subplot(121, projection="3d")
    ax_gt = fig.add_subplot(122, projection="3d")

    if pred_coords.shape[0] > 0:
        ax_pred.scatter(
            pred_coords[:, 0].tolist(),
            pred_coords[:, 1].tolist(),
            pred_coords[:, 2].tolist(),
            s=1,
            c="tomato",
            alpha=0.8,
        )

    if gt_coords.shape[0] > 0:
        ax_gt.scatter(
            gt_coords[:, 0].tolist(),
            gt_coords[:, 1].tolist(),
            gt_coords[:, 2].tolist(),
            s=1,
            c="steelblue",
            alpha=0.8,
        )

    ax_pred.set_title("Prediction")
    ax_gt.set_title("Ground Truth")

    for ax, shape in ((ax_pred, pred.shape), (ax_gt, gt.shape)):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect(shape)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved side-by-side fixed-view plot to {save_path}")
    plt.close(fig)


class BackprojectionModule(nn.Module):
    def __init__(self, grid_size=80, world_size=125.0):
        super().__init__()
        lin = torch.linspace(-world_size/2, world_size/2, grid_size)
        x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')
        X_hom = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1), torch.ones(grid_size**3)], dim=0)
        self.register_buffer('X_hom', X_hom)
        self.grid_size = grid_size

    def forward(self, features, projection_matrices, image_size=256):
        N, C, H, W = features.shape
        view_grids = []
        for i in range(N):
            P = projection_matrices[i]
            feat = features[i]
            X_proj = P @ self.X_hom
            depth = X_proj[2].clamp(min=1e-8)
            p_x = (X_proj[0] / depth / (image_size - 1)) * 2.0 - 1.0
            p_y = (X_proj[1] / depth / (image_size - 1)) * 2.0 - 1.0
            sample_grid = torch.stack([p_x, p_y], dim=-1).reshape(1, 1, -1, 2)
            sampled = F.grid_sample(feat.unsqueeze(0), sample_grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=True).squeeze(0).squeeze(1)
            view_grids.append(sampled.reshape(C, self.grid_size, self.grid_size, self.grid_size))
        return torch.stack(view_grids, dim=0)


class SpineDataset(Dataset):
    def __init__(self, drr_dir: Path, cache_dir: Path, transform=None):
        self.drr_dir = drr_dir
        self.cache_dir = cache_dir
        self.transform = transform or transforms.ToTensor()
        self.samples = []
        self.vertebra_to_idx = {}
        current_idx = 0

        # --- STEP 1: LOAD SAMPLES ---
        for patient in self.drr_dir.iterdir():
            if patient.is_dir():
                for vertebra in patient.iterdir():
                    if vertebra.is_dir():
                        if vertebra.name == "L5":
                            continue
                        mesh_files = list(vertebra.glob("*.npz"))
                        if not mesh_files:
                            continue

                        cache_path = self.cache_dir / f"{patient.name}_{vertebra.name}.npy"
                        if not cache_path.exists():
                            continue

                        drr_files = sorted(vertebra.glob("*full.png"))
                        if not drr_files:
                            continue

                        self.samples.append({
                            "patient": patient.name,
                            "vertebra": vertebra.name,
                            "mesh_file": mesh_files[0],
                            "drr_files": drr_files,
                        })

        # --- STEP 2: BUILD VERTEBRA INDEX ---
        self.vertebra_to_idx = {}
        current_idx = 0

        for sample in self.samples:
            v = sample["vertebra"]
            if v not in self.vertebra_to_idx:
                self.vertebra_to_idx[v] = current_idx
                current_idx += 1

        print("Vertebra mapping:", self.vertebra_to_idx)

        for sample in self.samples:
            v = sample["vertebra"]
            if v not in self.vertebra_to_idx:
                self.vertebra_to_idx[v] = current_idx
                current_idx += 1

        # --- STEP 3: LOAD PROJECTION MATRICES FROM DISK  [FIX] ---
        #
        # Original code built P matrices analytically here, placing the X-ray
        # source at Z=-1200 mm. diffdrr actually places it at Y=+1200 mm (AP
        # orientation). The mismatch caused backprojection to read features from
        # completely wrong pixels.
        #
        # Datageneration2 computes P matrices using diffdrr's perspective_projection
        # and saves {view}_P.npy to drr_dir. We load them here, in the same order
        # as the DRR files (alphabetically sorted by glob("*full.png")), so the
        # i-th P matrix always corresponds to the i-th DRR image.

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found — cannot load P matrices.")

        p_list = []
        for drr_path in self.samples[0]["drr_files"]:
            view_name = drr_path.stem.replace("_full", "")
            p_path = drr_dir / f"{view_name}_P.npy"
            if not p_path.exists():
                raise FileNotFoundError(
                    f"Projection matrix not found: {p_path}\n"
                    f"Run Datageneration2 first to generate the P matrix files."
                )
            p_list.append(np.load(p_path))

        self.projection_matrices = torch.from_numpy(np.stack(p_list)).float()  # (9, 3, 4)
        print(f"Loaded {len(p_list)} projection matrices from {drr_dir}")

        # Preload all DRR images into RAM so training reads from memory, not disk.
        # Cost: one slow Drive pass at startup (~1-2 min). Benefit: zero disk reads per batch.
        print(f"Preloading {len(self.samples)} × 9 DRR images into RAM...")
        self.image_cache = {}  # (patient, vertebra, view_idx) -> (3, H, W) float32 tensor
        for si, sample in enumerate(self.samples):
            if si % 50 == 0:
                print(f"  [{si}/{len(self.samples)}]")
            for vi, drr_path in enumerate(sample["drr_files"]):
                img = np.array(Image.open(drr_path)).astype(np.float32) / 65535.0
                tensor = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)
                self.image_cache[(sample["patient"], sample["vertebra"], vi)] = tensor
        print("Preload complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        cache_path = self.cache_dir / f"{sample['patient']}_{sample['vertebra']}.npy"
        voxels = np.load(cache_path)

        drr_images = [
            self.image_cache[(sample["patient"], sample["vertebra"], vi)]
            for vi in range(len(sample["drr_files"]))
        ]

        vertebra_idx = self.vertebra_to_idx[sample["vertebra"]]

        return torch.stack(drr_images, dim=0), voxels, vertebra_idx


class SpineDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

class Down3d(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.pool = nn.MaxPool3d(stride)
        self.block = ResBlock3D(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))

class Up3d(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=stride)
        self.block = ResBlock3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        cat_dim = 1 if x.ndim == 5 else 0
        x = torch.cat([x, skip], dim=cat_dim)
        return self.block(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(1,2,3,4))
        union = probs.sum(dim=(1,2,3,4)) + targets.sum(dim=(1,2,3,4))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ViewAttentionFusion(nn.Module):
    def __init__(self, channels, num_views):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(channels * num_views, channels, 1),
            nn.ReLU(),
            nn.Conv3d(channels, num_views, 1)
        )

    def forward(self, x):
        B, V, C, D, H, W = x.shape

        x_flat = x.view(B, V*C, D, H, W)
        attn = self.attn(x_flat)  # (B, V, D, H, W)
        attn = torch.softmax(attn, dim=1)

        out = (x * attn.unsqueeze(2)).sum(dim=1)
        return out


class ResNetFeatureExtractor(nn.Module):
    """
    Trainable ResNet feature extractor initialized from scratch.

    This replaces the pretrained DAX backbone.

    Output:
        (B, 64, 80, 80, 80)

    Explanation:
        - Extracts 4 ResNet feature levels:
            layer1: 256 channels
            layer2: 512 channels
            layer3: 1024 channels
            layer4: 2048 channels
        - Reduces each to 16 channels
        - Backprojects each level into 3D
        - Fuses 9 views with ViewAttentionFusion
        - Concatenates the 4 levels:
            4 * 16 = 64 channels
    """

    def __init__(self, resnet_type="resnet18", device=None, train_backbone=True):
        super().__init__()

        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_backbone = train_backbone

        # Own ResNet, randomly initialized.
        # No ImageNet weights, no DAX checkpoint.
        self.resnet = models.__dict__[resnet_type.lower()](weights=None)

        # Remove classification head conceptually.
        # We do not use avgpool or fc.
        del self.resnet.avgpool
        del self.resnet.fc

        self.fusion = ViewAttentionFusion(channels=16, num_views=9)

        self.reducer1 = nn.Conv2d(64, 16, kernel_size=1)
        self.reducer2 = nn.Conv2d(128, 16, kernel_size=1)
        self.reducer3 = nn.Conv2d(256, 16, kernel_size=1)
        self.reducer4 = nn.Conv2d(512, 16, kernel_size=1)

        self.backprojector = BackprojectionModule(grid_size=80, world_size=125.0)

        self.to(self.device)

    def forward_resnet_features(self, x):
        """
        Return ResNet intermediate feature maps.

        Input:
            x: (B*V, 3, H, W)

        Output:
            x1, x2, x3, x4
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def forward(self, x, P_matrices, grid_size=80):
        x = x.to(self.device)
        P_matrices = P_matrices.to(self.device)

        if x.ndim == 4:
            x = x.unsqueeze(0)

        B, V, C, H, W = x.shape
        image_size = H

        if P_matrices.ndim == 3:
            P_matrices = P_matrices.unsqueeze(0).expand(B, -1, -1, -1)

        x_flat = x.reshape(B * V, C, H, W)

        # Important:
        # No torch.no_grad() here.
        # The ResNet is trained jointly with the reconstruction model.
        x1, x2, x3, x4 = self.forward_resnet_features(x_flat)

        f1 = self.reducer1(x1)
        f2 = self.reducer2(x2)
        f3 = self.reducer3(x3)
        f4 = self.reducer4(x4)

        def backproject_batch(features, P_mats):
            _, C_f, H_f, W_f = features.shape
            features = features.reshape(B, V, C_f, H_f, W_f)

            grids = []
            for b in range(B):
                vol = self.backprojector(features[b], P_mats[b], image_size)
                grids.append(vol)

            return torch.stack(grids, dim=0)

        def fuse(grid):
            return self.fusion(grid)

        grid1 = fuse(backproject_batch(f1, P_matrices))
        grid2 = fuse(backproject_batch(f2, P_matrices))
        grid3 = fuse(backproject_batch(f3, P_matrices))
        grid4 = fuse(backproject_batch(f4, P_matrices))

        return torch.cat([grid1, grid2, grid3, grid4], dim=1)

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_ch)

        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)


class Refiner(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv1 = ResBlock3D(96, 32)
        self.down1 = Down3d(32, 64, stride=2)
        self.down2 = Down3d(64, 128, stride=2)
        self.bottleneck = Down3d(128, 256, stride=2)
        self.up1 = Up3d(256, 128, stride=2)
        self.up2 = Up3d(128, 64, stride=2)
        self.up3 = Up3d(64, 32, stride=2)
        self.final = nn.Conv3d(32, 1, kernel_size=1)
        self.drop = nn.Dropout3d(dropout)

    def forward(self, feature_grid):
        x = self.conv1(feature_grid)

        s1 = F.relu(x)

        s2 = self.down1(s1)
        s3 = self.drop(self.down2(s2))

        b = self.drop(self.bottleneck(s3))

        x = self.up1(b, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)

        x = self.final(x)

        return x


class ReconstructionModel(nn.Module):
    def __init__(self, feature_extractor: ResNetFeatureExtractor, refiner: Refiner):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.refiner = refiner
        self.embedding = nn.Embedding(num_embeddings=50, embedding_dim=32)

    def forward(self, drr_images, projection_matrices, vertebra_idx, grid_size=80):
        features = self.feature_extractor(drr_images, projection_matrices, grid_size=grid_size)
        emb = self.embedding(vertebra_idx)  # (B, 32)

        emb = emb[:, :, None, None, None]
        emb = emb.expand(-1, -1, features.shape[2], features.shape[3], features.shape[4])

        features = torch.cat([features, emb], dim=1)

        refined_output = self.refiner(features)
        return refined_output


def evaluate(model, dataloader, criterion, device, projection_matrices):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for drr_images, voxels, vertebra_idx in dataloader:
            drr_images = drr_images.to(device)
            voxels = voxels.to(device).float()
            vertebra_idx = vertebra_idx.to(device)

            outputs = model(drr_images, projection_matrices, vertebra_idx, grid_size=80)
            target = voxels.unsqueeze(1)
            loss = criterion(outputs, target)

            total_loss += loss.item()
            count += 1

    return total_loss / count


def evaluate_dice(model, dataloader, device, projection_matrices, threshold=0.75, smooth=1e-5):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for drr_images, voxels, vertebra_idx in dataloader:
            drr_images = drr_images.to(device)
            voxels = voxels.to(device).float()
            vertebra_idx = vertebra_idx.to(device)

            outputs = model(drr_images, projection_matrices, vertebra_idx, grid_size=80)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            target = voxels.unsqueeze(1)

            intersection = (preds * target).sum(dim=(1, 2, 3, 4))
            union = preds.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_scores.extend(dice.detach().cpu().numpy())

    return float(np.mean(dice_scores)) if len(dice_scores) > 0 else float("nan")


def train(lr, epochs, train_dataloader, val_dataloader, model, save_dir, device, projection_matrices, criterion=nn.BCEWithLogitsLoss()):
    print("Starting training...")
    save_dir.mkdir(parents=True, exist_ok=True)
    score_dir = save_dir / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_log.csv"

    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,epoch,train_loss,val_loss,lr,dice_mean,bce_mean\n")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_dice = 0.0
    early_stop_patience = 12
    no_improve_count = 0
    epoch_durations = []
    print("Entering training loop...")
    for epoch in range(epochs):
        epoch_start = datetime.now()
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        total_batches = len(train_dataloader)
        progress_interval = max(1, total_batches // 4)
        print(f"Epoch {epoch+1}/{epochs} started at {epoch_start.strftime('%H:%M:%S')} | batches: {total_batches}")
        dice_scores = []
        bce_scores = []
        for batch_idx, (drr_images, voxels, vertebra_idx) in enumerate(train_dataloader):
            drr_images = drr_images.to(device)
            voxels = voxels.to(device).float()
            vertebra_idx = vertebra_idx.to(device)

            optimizer.zero_grad()
            outputs = model(drr_images, projection_matrices, vertebra_idx, grid_size=80)

            target = voxels.unsqueeze(1)

            probs = torch.sigmoid(outputs)

            intersection = (probs * target).sum(dim=(1,2,3,4))
            union = probs.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
            dice = (2 * intersection + 1e-5) / (union + 1e-5)

            bce_log = F.binary_cross_entropy_with_logits(outputs, target, reduction='none')
            bce_log = bce_log.mean(dim=(1,2,3,4))

            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dice_scores.extend(dice.detach().cpu().numpy())
            bce_scores.extend(bce_log.detach().cpu().numpy())

            epoch_loss += loss.item()
            batch_count += 1
            if ((batch_idx + 1) % progress_interval == 0) or (batch_idx + 1 == total_batches):
                running_avg = epoch_loss / batch_count
                print(
                    f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} | "
                    f"Batch Loss: {loss.item():.4f} | Running Avg: {running_avg:.4f}"
                )

        avg_loss = epoch_loss / len(train_dataloader)

        avg_val_loss = evaluate(model, val_dataloader, criterion, device, projection_matrices)
        val_dice = evaluate_dice(model, val_dataloader, device, projection_matrices)
        scheduler.step(avg_val_loss)

        train_dice_mean = float(np.mean(dice_scores)) if dice_scores else float("nan")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Train Dice: {train_dice_mean:.4f} | Val Loss: {avg_val_loss:.4f}, Val Dice: {val_dice:.4f} | LR: {current_lr:.6f}")

        epoch_seconds = (datetime.now() - epoch_start).total_seconds()
        epoch_durations.append(epoch_seconds)
        avg_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = epochs - (epoch + 1)
        eta = datetime.now() + timedelta(seconds=remaining_epochs * avg_epoch_seconds)
        print(
            f"Epoch duration: {epoch_seconds/60:.2f} min | "
            f"Estimated remaining: {(remaining_epochs * avg_epoch_seconds)/60:.2f} min | "
            f"ETA: {eta.strftime('%H:%M:%S')}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            no_improve_count = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"New best model saved (val dice: {best_dice:.4f})")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count}/{early_stop_patience} epochs")
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break
        chunk_size = 100

        dice_avg = [
            np.mean(dice_scores[i:i+chunk_size])
            for i in range(0, len(dice_scores), chunk_size)
        ]

        bce_avg = [
            np.mean(bce_scores[i:i+chunk_size])
            for i in range(0, len(bce_scores), chunk_size)
        ]

        plt.figure()
        plt.plot(dice_avg, label="Dice")
        plt.plot(bce_avg, label="BCE")
        plt.xlabel("Chunk (100 vertebrae)")
        plt.ylabel("Score")
        plt.title(f"Epoch {epoch+1}")
        plt.legend()

        file_name = score_dir / f"scores_epoch_{epoch+1}.png"
        plt.savefig(file_name)
        plt.close()

        dice_mean = float(np.mean(dice_scores)) if len(dice_scores) > 0 else float("nan")
        bce_mean = float(np.mean(bce_scores)) if len(bce_scores) > 0 else float("nan")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat(timespec='seconds')},{epoch+1},{avg_loss:.6f},{avg_val_loss:.6f},{current_lr:.8f},{dice_mean:.6f},{bce_mean:.6f}\n"
            )

        print(f"Saved progress plot: {file_name}")
        print(f"Updated training log: {log_path}")

def precompute_voxel_cache(drr_dir, cache_dir, grid_size=80):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    samples = []
    for patient in Path(drr_dir).iterdir():
        if patient.is_dir():
            for vertebra in patient.iterdir():
                if vertebra.is_dir():
                    mesh_files = list(vertebra.glob("*.npz"))
                    if mesh_files:
                        samples.append((patient.name, vertebra.name, mesh_files[0]))

    for i, (patient, vertebra, mesh_file) in enumerate(samples):
        cache_path = cache_dir / f"{patient}_{vertebra}.npy"
        if cache_path.exists():
            continue

        if i % 10 == 0:
            print(f"Voxelizing [{i}/{len(samples)}]")

        data = np.load(mesh_file)
        gs = grid_size
        world_half = 62.5  # mm — must match BackprojectionModule world_size/2

        if 'voxels_gt' in data:
            voxels = data['voxels_gt'].astype(bool)
        elif 'verts_world' in data:
            voxel_pitch = (2 * world_half) / gs  # 1.5625 mm/voxel
            verts = data['verts_world']
            grid_idx = np.clip(
                ((verts + world_half) / voxel_pitch).astype(int), 0, gs - 1
            )
            surface = np.zeros((gs, gs, gs), dtype=bool)
            surface[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]] = True
            voxels = binary_fill_holes(surface)
        else:
            # Legacy path (old meshes without verts_world).
            # Re-run Datageneration2 to get world-aligned ground truth.
            mesh = trimesh.Trimesh(vertices=data['verts'], faces=data['faces'])
            bounds = mesh.bounds
            center = bounds.mean(axis=0)
            size = (bounds[1] - bounds[0]).max()
            mesh.vertices = (mesh.vertices - center) / (size * 1.2)

            voxels_raw = binary_fill_holes(mesh.voxelized(pitch=1.0 / gs).matrix)
            x, y, z = voxels_raw.shape
            pad_x, pad_y, pad_z = max(0, gs - x), max(0, gs - y), max(0, gs - z)
            voxels = np.pad(voxels_raw, (
                (pad_x // 2, pad_x - pad_x // 2),
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_z // 2, pad_z - pad_z // 2),
            ), mode='constant', constant_values=False)

        if voxels.shape != (gs, gs, gs):
            print("WARNING: unexpected voxel shape:", voxels.shape)

        voxels = voxels.astype(bool)
        np.save(cache_path, voxels)

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight, dice_weight=6, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = bce * ((1 - p_t) ** self.gamma)
        focal = focal.mean()

        return focal + self.dice_weight * self.dice(logits, targets)


def debug_backprojection(dataset, dax):
    print("\n" + "="*60)
    print("DEBUG: Backprojection")
    print("="*60)

    view_names = ["AP_Caudal", "AP_Cranial", "AP", "Lat_L", "Lat_R",
                  "Oblique_L", "Oblique_R", "Pedicle_L", "Pedicle_R"]
    P_all = dataset.projection_matrices  # (9, 3, 4)
    bp = dax.backprojector
    image_size = 256

    print("\n  [1] Projection Sanity")
    print(f"      Image size: {image_size}x{image_size}, principal point: (128, 128)")
    print(f"      World grid spans ±62.5 mm, SDD=1200 mm\n")

    test_points = {
        "origin      (0,0,0)":    torch.tensor([0.,    0.,   0.,   1.]),
        "corner X+   (62.5,0,0)": torch.tensor([62.5,  0.,   0.,   1.]),
        "corner X-  (-62.5,0,0)": torch.tensor([-62.5, 0.,   0.,   1.]),
        "corner Y+   (0,62.5,0)": torch.tensor([0.,   62.5,  0.,   1.]),
        "corner Z+   (0,0,62.5)": torch.tensor([0.,    0.,  62.5,  1.]),
    }

    for v_idx, v_name in enumerate(view_names):
        P = P_all[v_idx]
        print(f"      {v_name}")
        for pt_name, pt in test_points.items():
            proj = P @ pt
            depth = proj[2].item()
            if depth <= 0:
                print(f"        {pt_name}: BEHIND CAMERA (depth={depth:.1f})")
                continue
            px = (proj[0] / depth).item()
            py = (proj[1] / depth).item()
            in_bounds = 0 <= px <= 255 and 0 <= py <= 255
            flag = "" if in_bounds else " <- OUT OF BOUNDS"
            print(f"        {pt_name}: pixel=({px:6.1f}, {py:6.1f}){flag}")
        print()

    print("\n  [2] Grid Coverage  (% of 80^3 voxels projecting inside image)")
    X_hom = bp.X_hom

    for v_idx, v_name in enumerate(view_names):
        P = P_all[v_idx].to(X_hom.device)
        X_proj = P @ X_hom
        depth  = X_proj[2].clamp(min=1e-8)
        px = (X_proj[0] / depth)
        py = (X_proj[1] / depth)

        in_bounds = ((px >= 0) & (px <= image_size - 1) &
                     (py >= 0) & (py <= image_size - 1))
        pct = in_bounds.float().mean().item() * 100

        neg_depth = (X_proj[2] < 0).float().mean().item() * 100

        flag = ""
        if pct < 70:
            flag = " <- LOW: many voxels will get zero features"
        if neg_depth > 1:
            flag += f" | {neg_depth:.1f}% behind camera"

        print(f"      {v_name:15s}: {pct:5.1f}% in bounds{flag}")

    print("\n\n  [3] Backprojected Volume Quality  (using sample 0)")

    drr_images, _, _ = dataset[0]
    x = drr_images.to(dax.device)

    with torch.no_grad():
        x1, x2, x3, x4 = dax.DAX(x)
        f1 = dax.reducer1(x1)

    P_dev = P_all.to(dax.device)
    vol = bp(f1, P_dev, image_size)

    print(f"      Volume shape: {tuple(vol.shape)}  (views, channels, D, H, W)")
    print()

    for v_idx, v_name in enumerate(view_names):
        v = vol[v_idx]
        mean  = v.mean().item()
        std   = v.std().item()
        zeros = (v == 0).float().mean().item()

        flag = ""
        if zeros > 0.9:
            flag = " <- >90% zeros: coverage problem or features all zero"
        elif std < 0.01:
            flag = " <- near-zero variance: no spatial structure"

        print(f"      {v_name:15s}: mean={mean:+.4f}  std={std:.4f}  zeros={zeros:.3f}{flag}")

    print("\n      Cross-view correlation at channel 0 (AP_Caudal as reference):")
    ref = vol[0, 0].reshape(-1).float()
    for v_idx, v_name in enumerate(view_names[1:], 1):
        other = vol[v_idx, 0].reshape(-1).float()
        ref_c   = ref   - ref.mean()
        other_c = other - other.mean()
        denom = (ref_c.norm() * other_c.norm()).clamp(min=1e-8)
        corr = (ref_c * other_c).sum() / denom
        flag = " <- uncorrelated: views may not be seeing same structure" if corr < 0.1 else ""
        print(f"      AP_Caudal vs {v_name:15s}: corr={corr:.4f}{flag}")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60 + "\n")


def debug_drr_dax(dataset, dax, num_samples=3):
    print("\n" + "="*60)
    print("DEBUG: DRR + DAX Feature Extraction")
    print("="*60)

    view_names = ["AP_Caudal", "AP_Cranial", "AP", "Lat_L", "Lat_R", "Oblique_L", "Oblique_R", "Pedicle_L", "Pedicle_R"]

    for sample_idx in range(min(num_samples, len(dataset))):
        drr_images, voxels, vertebra_idx = dataset[sample_idx]
        sample_info = dataset.samples[sample_idx]
        print(f"\n--- Sample {sample_idx}: {sample_info['patient']} / {sample_info['vertebra']} ---")

        print(f"\n  [1] DRR Images  (shape: {drr_images.shape})")
        print(f"      Expected: (9 views, 3 channels, H, W)")

        if drr_images.shape[0] != 9:
            print(f"      WARNING: expected 9 views, got {drr_images.shape[0]}. "
                  f"Files found: {[f.name for f in sample_info['drr_files']]}")

        for v_idx, v_name in enumerate(view_names[:drr_images.shape[0]]):
            img = drr_images[v_idx, 0]
            mn, mx, mean = img.min().item(), img.max().item(), img.mean().item()
            all_zero = mx < 1e-6
            saturated = mn > 0.99
            flag = " <- ALL ZERO (bad file?)" if all_zero else (" <- SATURATED" if saturated else "")
            print(f"      {v_name:15s}: min={mn:.4f}  max={mx:.4f}  mean={mean:.4f}{flag}")

        print(f"\n  [2] DAX Backbone Features")
        print(f"      (each layer: mean, std, fraction of dead neurons)")

        x = drr_images.to(dax.device)
        with torch.no_grad():
            x1, x2, x3, x4 = dax.DAX(x)

        layer_names = ["Layer1 (256ch, 64x64)", "Layer2 (512ch, 32x32)",
                       "Layer3 (1024ch,16x16)", "Layer4 (2048ch, 8x8)"]
        for name, feat in zip(layer_names, [x1, x2, x3, x4]):
            mean = feat.mean().item()
            std  = feat.std().item()
            dead = (feat == 0).float().mean().item()
            flag = ""
            if std < 0.01:
                flag = " <- LOW VARIANCE: features may be collapsed"
            elif dead > 0.8:
                flag = " <- HIGH DEAD RATIO: ReLU killing most neurons"
            print(f"      {name}: mean={mean:.4f}  std={std:.4f}  dead={dead:.3f}{flag}")

        print(f"\n  [3] View Diversity  (cosine similarity between views at Layer1)")

        ap_feat = x1[2].reshape(-1)
        for v_idx, v_name in enumerate(view_names):
            if v_idx == 2:
                continue
            other_feat = x1[v_idx].reshape(-1)
            cos_sim = F.cosine_similarity(ap_feat.unsqueeze(0), other_feat.unsqueeze(0)).item()
            flag = " <- nearly identical to AP" if cos_sim > 0.99 else ""
            print(f"      AP vs {v_name:15s}: cosine_sim={cos_sim:.4f}{flag}")

        print(f"\n  [4] After 1x1 Reducers  (16ch each)")

        with torch.no_grad():
            r1 = dax.reducer1(x1)
            r2 = dax.reducer2(x2)
            r3 = dax.reducer3(x3)
            r4 = dax.reducer4(x4)

        for name, feat in zip(["reducer1", "reducer2", "reducer3", "reducer4"], [r1, r2, r3, r4]):
            mean = feat.mean().item()
            std  = feat.std().item()
            flag = " <- near-zero output: reducer not yet trained (OK at epoch 0)" if std < 0.01 else ""
            print(f"      {name}: mean={mean:.4f}  std={std:.4f}{flag}")

        print(f"\n  [5] Ground Truth Voxels  (shape: {voxels.shape})")
        fill_ratio = voxels.astype(float).mean()
        print(f"      Filled voxels: {voxels.sum()} / {voxels.size}  ({fill_ratio*100:.2f}%)")
        if fill_ratio < 0.005:
            print(f"      WARNING: voxel grid is nearly empty — check voxelization / cache")
        elif fill_ratio > 0.5:
            print(f"      WARNING: >50% filled — voxelization may have failed (hollow mesh?)")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60 + "\n")


def debug_predictions(model, dataset, val_idx, device, projection_matrices, save_dir, num_samples=5):
    """Diagnose model quality: logit stats, fill ratios, threshold sweep, and saved visualizations."""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(val_idx), size=min(num_samples, len(val_idx)), replace=False)

    print("\n" + "=" * 60)
    print("DEBUG PREDICTIONS")
    print("=" * 60)

    with torch.no_grad():
        for rank, si in enumerate(chosen):
            real_idx = int(val_idx[si])
            drr_images, voxels, vertebra_idx = dataset[real_idx]

            drr_batch = drr_images.unsqueeze(0).to(device)
            vidx_batch = torch.tensor([vertebra_idx], device=device)

            outputs = model(drr_batch, projection_matrices, vidx_batch, grid_size=80)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            logits = outputs.squeeze().cpu().numpy()
            gt = voxels.astype(bool)

            gt_fill = gt.mean()
            sample_info = dataset.samples[real_idx]
            print(f"\nSample {rank+1}: {sample_info['patient']} / {sample_info['vertebra']}")
            print(f"  Logits : min={logits.min():.3f}  max={logits.max():.3f}  mean={logits.mean():.3f}  std={logits.std():.3f}")
            print(f"  Probs  : min={probs.min():.4f}  max={probs.max():.4f}  mean={probs.mean():.4f}")
            print(f"  GT fill: {gt_fill*100:.2f}%  ({gt.sum()} / {gt.size} voxels)")
            print(f"  Threshold sweep:")
            best_dice, best_thresh = 0.0, 0.5
            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                pred = probs > t
                inter = (pred & gt).sum()
                d = (2 * inter + 1e-5) / (pred.sum() + gt.sum() + 1e-5)
                marker = " <-- best" if d > best_dice else ""
                print(f"    t={t:.1f}  fill={pred.mean()*100:.2f}%  Dice={d:.4f}{marker}")
                if d > best_dice:
                    best_dice, best_thresh = d, t

            pred_best = probs > best_thresh
            save_path = save_dir / f"pred_vs_gt_{rank+1}_{sample_info['vertebra']}.png"
            plot_voxel_comparison_fixed_view_3d(
                prediction_voxels=pred_best,
                ground_truth_voxels=gt,
                save_path=str(save_path),
            )
            print(f"  Saved : {save_path}")

    print("\n" + "=" * 60)
    print("DEBUG PREDICTIONS COMPLETE")
    print("=" * 60 + "\n")


def main():

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        print("MPS fallback is enabled: unsupported ops will run on CPU.")

    project_root = Path(__file__).resolve().parent
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    ######################################### Hyperparameters and setup #########################################
    # Set SPINE_DATA_DIR to the DRR output folder from Datageneration/Datageneration.py.
    _raw_data = os.environ.get("SPINE_DATA_DIR")
    if not _raw_data:
        raise EnvironmentError(
            "Set SPINE_DATA_DIR to the folder produced by Datageneration/Datageneration.py."
        )
    data_dir = Path(_raw_data)
    cache_dir = Path(os.environ.get("SPINE_CACHE_DIR") or (project_root.parent / "voxel_cache"))
    model_dir = Path(os.environ.get("SPINE_MODEL_DIR") or (project_root.parent / "Models"))
    mode = "train"  # "train", "plot", or "debug"
    pretrained_checkpoint = False

    lr = 1e-4
    imbalance_param = 5.0
    epochs = 40
    batch_size = 2
    num_workers = 2
    #############################################################################################################

    if not data_dir.exists():
        raise FileNotFoundError(f"DRR data directory not found: {data_dir}")

    precompute_voxel_cache(data_dir, cache_dir)
    dataset = SpineDataset(data_dir, cache_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid samples found in {data_dir}. Check DRR output structure and generated files.")

    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.333, random_state=seed)

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = SpineDataloader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_loader = SpineDataloader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = SpineDataloader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    feature_extractor = ResNetFeatureExtractor(
    resnet_type="resnet18",
    device=device,
    train_backbone=True,).to(device)

    refiner = Refiner().to(device)

    reconstructor = ReconstructionModel(feature_extractor, refiner).to(device)


    print("Model initialized.")

    if pretrained_checkpoint:
        reconst_checkpoint = model_dir / "best_model.pth"
        state = torch.load(reconst_checkpoint, map_location=device)
        reconstructor.load_state_dict(state)

    pos_weight = torch.tensor([imbalance_param], device=device)

    criterion = CombinedLoss(pos_weight)

    if mode == "debug":
        debug_drr_dax(dataset, dax, num_samples=3)
        debug_backprojection(dataset, dax)
        return

    if mode == "predict":
        model_path = model_dir / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model at {model_path} — run mode='train' first.")
        reconstructor.load_state_dict(torch.load(model_path, map_location=device))
        debug_predictions(reconstructor, dataset, val_idx, device, dataset.projection_matrices, model_dir / "predictions")
        return

    if mode == "train":
        train(lr, epochs, train_loader, val_loader, reconstructor, model_dir, device, dataset.projection_matrices, criterion)
        model_path = model_dir / "best_model.pth"
        reconstructor.load_state_dict(torch.load(model_path, map_location=device))
        test_loss = evaluate(reconstructor, test_loader, criterion, device, dataset.projection_matrices)
        print(f"Final test loss: {test_loss:.4f}")

    if mode == "plot":
        with torch.no_grad():
            example = reconstructor(dataset[0][0].to(device), dataset.projection_matrices.to(device), grid_size=80)
            probs = torch.sigmoid(example)
            print(f"Logits  - min: {example.min():.3f}, max: {example.max():.3f}, mean: {example.mean():.3f}")
            print(f"Probs   - min: {probs.min():.3f}, max: {probs.max():.3f}, mean: {probs.mean():.3f}")
            print(f"Filled  - {(probs > 0.3).sum().item()} / {probs.numel()} = {(probs > 0.3).float().mean():.4f}")

        voxels = (probs > 0.3).squeeze(0).cpu().detach().numpy()
        plot_voxel_comparison_fixed_view_3d(
            prediction_voxels=voxels,
            ground_truth_voxels=dataset[0][1],
            save_path="prediction_vs_ground_truth_fixed_view.png",
        )

    pass

if __name__ == "__main__":
    main()