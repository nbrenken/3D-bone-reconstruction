"""
Interactive single-sample 3D plot for the DAX-based model.


What this script does:
  1. Loads one CT and segmentation mask
  2. Extracts only one Lumbar vertebra
  3. Generates the 9 DRRs
  4. Runs the trained DAX-based reconstruction model once
  5. Shows an interactive rotatable 3D Plotly plot
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gc
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import torchio as tio
import runpy

from skimage.measure import marching_cubes
from diffdrr.data import read
from diffdrr.drr import DRR

import plotly.graph_objects as go
from plotly.subplots import make_subplots


#Example
CASE_ID = "1.3.6.1.4.1.9328.50.4.0001"
VERT_LABEL = 23
VERT_NAME = "L4"


# ── Paths

PROJECT_ROOT = Path(__file__).resolve().parent

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Set SPINE_CT_VOLUMES to the .../data/colon (or relevant) directory of the CT dataset.
_raw_vol = os.environ.get("SPINE_CT_VOLUMES")
if not _raw_vol:
    raise EnvironmentError("Set SPINE_CT_VOLUMES to the CT volumes directory (e.g. .../data/colon).")
VOLUMES_DIR = Path(_raw_vol)

# Set SPINE_CT_LABELS to the .../label/colon directory of the CT dataset.
_raw_lbl = os.environ.get("SPINE_CT_LABELS")
if not _raw_lbl:
    raise EnvironmentError("Set SPINE_CT_LABELS to the CT labels directory (e.g. .../label/colon).")
LABELS_DIR = Path(_raw_lbl)

# Directory containing the *_P.npy projection matrices (output of Datageneration).
_raw_pmat = os.environ.get("SPINE_DATA_DIR")
if not _raw_pmat:
    raise EnvironmentError("Set SPINE_DATA_DIR to the DRR output directory containing *_P.npy files.")
P_MATRIX_DIR = Path(_raw_pmat)

# Training script is in this repo.
TRAINING_SCRIPT = _REPO_ROOT / "Training" / "DAX50_Training.py"

MODEL_CKPT = Path(
    os.environ.get("SPINE_MODEL_CKPT") or (_REPO_ROOT / "Models" / "best_modelDAX50.pth")
)

DAX_CKPT = Path(
    os.environ.get("SPINE_DAX_CKPT") or
    (_REPO_ROOT / "AdditionalDAXData" / "dax-checkpoint-resnet50-version-b.pth")
)


# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_TO_NAME = {
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
}

NAME_TO_IDX = {
    "L3": 0,
    "L4": 1,
    "L2": 2,
    "L1": 3,
}

GRID_SIZE = 80
WORLD_HALF = 62.5
VOXEL_PITCH = (2 * WORLD_HALF) / GRID_SIZE

SDD = 1200
HEIGHT, WIDTH = 256, 256
DELX, DELY = 0.5, 0.5
PARAMETERIZATION = "euler_angles"
CONVENTION = "ZYX"

VIEWS_SORTED = [
    ("AP_Caudal",     [0,  -10, 0]),
    ("AP_Cranial",    [0,   10, 0]),
    ("AP",            [0,    0, 0]),
    ("LateralLeft",   [90,   0, 0]),
    ("LateralRight",  [-90,  0, 0]),
    ("Oblique_Left",  [45,   0, 0]),
    ("Oblique_Right", [-45,  0, 0]),
    ("Pedicle_Left",  [30,   5, 0]),
    ("Pedicle_Right", [-30,  5, 0]),
]


# ── Device ────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ── Load DAX-based model ──────────────────────────────────────────────────────

if not MODEL_CKPT.exists():
    raise FileNotFoundError(
        f"\nModel checkpoint not found:\n{MODEL_CKPT}\n"
    )

if not DAX_CKPT.exists():
    raise FileNotFoundError(
        f"\nDAX checkpoint not found:\n{DAX_CKPT}\n"
    )

if not TRAINING_SCRIPT.exists():
    raise FileNotFoundError(
        f"\nTraining script not found:\n{TRAINING_SCRIPT}\n"
    )

ns = runpy.run_path(str(TRAINING_SCRIPT))

FeatureExtractor = ns["FeatureExtractor"]
Refiner = ns["Refiner"]
ReconstructionModel = ns["ReconstructionModel"]

dax = FeatureExtractor(
    DAX_CKPT,
    device=device,
).to(device)

refiner = Refiner().to(device)

model = ReconstructionModel(
    dax,
    refiner,
).to(device)

state = torch.load(
    MODEL_CKPT,
    map_location=device,
)

model.load_state_dict(state)
model.eval()

print("DAX-based model loaded.")


# ── Load projection matrices ──────────────────────────────────────────────────

p_list = []

for view_name, _ in VIEWS_SORTED:
    p_path = P_MATRIX_DIR / f"{view_name}_P.npy"

    if not p_path.exists():
        raise FileNotFoundError(f"P matrix not found: {p_path}")

    p_list.append(np.load(p_path))

projection_matrices = torch.from_numpy(
    np.stack(p_list)
).float().to(device)

print(f"Loaded {len(p_list)} projection matrices.")


# ── Helper functions ──────────────────────────────────────────────────────────

def compute_bbox(mask, margin=250):
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise ValueError("Mask is empty; cannot compute bounding box.")

    mn, mx = coords.min(axis=0), coords.max(axis=0)

    lo = np.maximum(mn - margin, 0)
    hi = np.minimum(mx + margin, np.array(mask.shape))

    return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]


def center_subvolume(submask, ct_crop):
    """
    Shift CT and mask so that the vertebra centroid is centered in the volume.
    """

    coords = np.argwhere(submask > 0)

    if coords.size == 0:
        raise ValueError("Submask is empty; cannot center volume.")

    center_voxel = coords.mean(axis=0)
    vol_center = np.array(submask.shape) / 2.0
    shift = (vol_center - center_voxel).astype(int)

    ct_out = np.zeros_like(ct_crop)
    mask_out = np.zeros_like(submask)

    for arr_in, arr_out in [
        (ct_crop, ct_out),
        (submask.astype(ct_crop.dtype), mask_out),
    ]:
        sx0 = max(0, -shift[0])
        sx1 = min(arr_in.shape[0], arr_in.shape[0] - shift[0])

        sy0 = max(0, -shift[1])
        sy1 = min(arr_in.shape[1], arr_in.shape[1] - shift[1])

        sz0 = max(0, -shift[2])
        sz1 = min(arr_in.shape[2], arr_in.shape[2] - shift[2])

        dx0 = max(0, shift[0])
        dy0 = max(0, shift[1])
        dz0 = max(0, shift[2])

        arr_out[
            dx0:dx0 + (sx1 - sx0),
            dy0:dy0 + (sy1 - sy0),
            dz0:dz0 + (sz1 - sz0),
        ] = arr_in[
            sx0:sx1,
            sy0:sy1,
            sz0:sz1,
        ]

    return mask_out.astype(np.uint8), ct_out


def build_gt_voxels(mask_centered, zooms):
    """
    Build the ground-truth voxel grid for visualization only.
    """

    verts, _, _, _ = marching_cubes(
        mask_centered,
        level=0.5,
        spacing=tuple(zooms),
    )

    centroid = verts.mean(axis=0)

    mask_coords = np.argwhere(mask_centered > 0)
    coords_mm = mask_coords * np.array(zooms) - centroid

    grid_idx = ((coords_mm + WORLD_HALF) / VOXEL_PITCH).astype(int)

    valid = np.all(
        (grid_idx >= 0) & (grid_idx < GRID_SIZE),
        axis=1,
    )

    grid_idx = grid_idx[valid]

    world_grid = np.zeros(
        (GRID_SIZE, GRID_SIZE, GRID_SIZE),
        dtype=bool,
    )

    if grid_idx.shape[0] > 0:
        world_grid[
            grid_idx[:, 0],
            grid_idx[:, 1],
            grid_idx[:, 2],
        ] = True

    return world_grid


def generate_drrs(ct_centered, spacing):
    """
    Generate 9 DRR images in memory.

    Returns:
        drr_np: shape (9, HEIGHT, WIDTH), float32 in [0, 1]
    """

    bone_threshold = 15

    ct_bone = ct_centered.copy()
    ct_bone[ct_bone < bone_threshold] = 0

    drr_imgs = []

    tmp_dir = tempfile.mkdtemp(prefix="single_drr_plot_dax_")

    try:
        affine_drr = np.diag(
            [spacing[0], spacing[1], spacing[2], 1.0]
        )

        full_path = os.path.join(tmp_dir, "full.nii.gz")

        nib.save(
            nib.Nifti1Image(ct_bone, affine_drr),
            full_path,
        )

        drr_renderer = "trilinear" if device.type == "mps" else "siddon"

        renderer = DRR(
            read(full_path),
            sdd=SDD,
            height=HEIGHT,
            width=WIDTH,
            delx=DELX,
            dely=DELY,
            renderer=drr_renderer,
        ).to(device)

        for _, angles_deg in VIEWS_SORTED:
            rot = torch.from_numpy(
                np.array(
                    [np.deg2rad(angles_deg)],
                    dtype=np.float32,
                )
            ).to(device)

            trans = torch.tensor(
                [[0.0, float(SDD), 0.0]],
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                img = renderer(
                    rot,
                    trans,
                    parameterization=PARAMETERIZATION,
                    convention=CONVENTION,
                )

            img_np = img.squeeze().detach().cpu().numpy().astype(np.float32)

            rng = img_np.max() - img_np.min()

            if rng > 1e-8:
                img_np = (img_np - img_np.min()) / rng

            drr_imgs.append(img_np)

        del renderer
        gc.collect()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return np.stack(drr_imgs, axis=0)


def run_inference(drr_np, vertebra_name):
    """
    Run model inference for one vertebra.

    Returns:
        probs: shape (GRID_SIZE, GRID_SIZE, GRID_SIZE)
    """

    tensors = [
        torch.tensor(drr_np[i]).unsqueeze(0).repeat(3, 1, 1)
        for i in range(drr_np.shape[0])
    ]

    drr_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    vidx = torch.tensor(
        [NAME_TO_IDX[vertebra_name]],
        device=device,
    )

    with torch.no_grad():
        logits = model(
            drr_tensor,
            projection_matrices,
            vidx,
            grid_size=GRID_SIZE,
        )

        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    return probs


def interactive_3d_plot(
    pred_vox,
    gt_vox,
    max_points=20000,
    title="Interactive 3D plot",
):
    """
    Interactive rotatable 3D plot using Plotly.
    Nothing is saved.
    """

    def sample_points(voxels, n=max_points):
        pts = np.argwhere(voxels)

        if len(pts) > n:
            idx = np.random.choice(len(pts), n, replace=False)
            pts = pts[idx]

        return pts

    pred_pts = sample_points(pred_vox)
    gt_pts = sample_points(gt_vox)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[
            [
                {"type": "scene"},
                {"type": "scene"},
                {"type": "scene"},
            ]
        ],
        subplot_titles=[
            "Prediction",
            "Ground Truth",
            "Overlay",
        ],
    )

    if len(pred_pts):
        fig.add_trace(
            go.Scatter3d(
                x=pred_pts[:, 0],
                y=pred_pts[:, 1],
                z=pred_pts[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    opacity=0.7,
                    color="red",
                ),
                name="Prediction",
            ),
            row=1,
            col=1,
        )

    if len(gt_pts):
        fig.add_trace(
            go.Scatter3d(
                x=gt_pts[:, 0],
                y=gt_pts[:, 1],
                z=gt_pts[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    opacity=0.7,
                    color="blue",
                ),
                name="Ground Truth",
            ),
            row=1,
            col=2,
        )

    if len(gt_pts):
        fig.add_trace(
            go.Scatter3d(
                x=gt_pts[:, 0],
                y=gt_pts[:, 1],
                z=gt_pts[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    opacity=0.35,
                    color="blue",
                ),
                name="GT overlay",
            ),
            row=1,
            col=3,
        )

    if len(pred_pts):
        fig.add_trace(
            go.Scatter3d(
                x=pred_pts[:, 0],
                y=pred_pts[:, 1],
                z=pred_pts[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    opacity=0.35,
                    color="red",
                ),
                name="Prediction overlay",
            ),
            row=1,
            col=3,
        )

    axis_settings = dict(
        range=[0, GRID_SIZE],
        title="",
        showbackground=True,
    )

    scene_settings = dict(
        xaxis=axis_settings,
        yaxis=axis_settings,
        zaxis=axis_settings,
        aspectmode="cube",
    )

    fig.update_layout(
        title=title,
        width=1500,
        height=650,
        showlegend=True,
        scene=scene_settings,
        scene2=scene_settings,
        scene3=scene_settings,
    )

    fig.show()


# ── Main: one case, one vertebra only ─────────────────────────────────────────

ct_path = VOLUMES_DIR / f"{CASE_ID}.nii.gz"
mask_path = LABELS_DIR / f"{CASE_ID}_seg.nii.gz"

if not ct_path.exists():
    raise FileNotFoundError(f"CT not found: {ct_path}")

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

print(f"\nCase: {CASE_ID}")
print(f"Vertebra: {VERT_NAME}")

ct_img = tio.ScalarImage(str(ct_path))
mask_img = tio.LabelMap(str(mask_path))

ct_img = tio.ToCanonical()(ct_img)
mask_img = tio.ToCanonical()(mask_img)
mask_img = tio.Resample(ct_img)(mask_img)

ct_np = ct_img.data.squeeze().numpy().astype(np.float32)
mask_np = mask_img.data.squeeze().numpy().astype(np.int16)
zooms = ct_img.spacing

labels_present = set(np.unique(mask_np).tolist())

if VERT_LABEL not in labels_present:
    raise ValueError(
        f"Label {VERT_LABEL} for {VERT_NAME} was not found in the mask."
    )

vert_mask = (mask_np == VERT_LABEL).astype(np.uint8)

print("Cropping and centering L2...")

x0, x1, y0, y1, z0, z1 = compute_bbox(vert_mask)

submask = vert_mask[x0:x1, y0:y1, z0:z1]
ct_crop = ct_np[x0:x1, y0:y1, z0:z1]

mask_centered, ct_centered = center_subvolume(
    submask,
    ct_crop,
)

print("Building GT voxel grid for plotting...")

gt_voxels = build_gt_voxels(
    mask_centered,
    zooms,
)

print("Generating DRRs...")

drr_np = generate_drrs(
    ct_centered,
    zooms,
)

print("Running inference...")

probs = run_inference(
    drr_np,
    VERT_NAME,
)

# Fixed threshold for visualization only.
# No threshold sweep, no Dice, no Hausdorff.
pred_voxels = probs > 0.5

print("Opening interactive 3D plot...")

interactive_3d_plot(
    pred_voxels,
    gt_voxels,
    max_points=20000,
    title=f"DAX model / {CASE_ID} / {VERT_NAME} / threshold = 0.5",
)

print("Done.")