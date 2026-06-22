# Datageneration2

import os
import tempfile
import numpy as np
import torch
import nibabel as nib
import torchio as tio
import imageio
import gc

from diffdrr.data import read, load_example_ct
from diffdrr.drr import DRR
from diffdrr.pose import convert as diffdrr_convert  # [FIX Bug 1]
from skimage.measure import marching_cubes


# ============================================================
# SETTINGS
# ============================================================

root = "C:/Users/karlo/OneDrive/Desktop/1kSpineDataxray3"
base_out_dir = "C:/Users/karlo/OneDrive/Desktop/DinoVert/ShirleySTUFF/output"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

data_dir = data_dir = os.path.join(root)  # No 'raw_data', directly use the root folder


# ============================================================
# LOAD CASES
# ============================================================

volumes_dir = os.path.normpath(os.path.join(data_dir, "data"))  # Change "volumes" to "data"
target_folders = sorted(
    folder for folder in os.listdir(volumes_dir)
    if os.path.isdir(os.path.join(volumes_dir, folder))
)
target_folders.remove("liver")
print(target_folders)


cases = []
for folder in target_folders:
    for root_dir, _, files in os.walk(os.path.join(volumes_dir, folder)):
        for f in files:
            if f.endswith(".nii.gz"):
                cases.append(os.path.join(root_dir, f))

cases = sorted(cases)
print("Total CT files:", len(cases))


# ============================================================
# FIND MASK
# ============================================================

def find_mask(ct_path):
    # Get the relative path to the volumes directory
    rel_path = os.path.relpath(ct_path, volumes_dir)

    # Define the label directory where the segmentation files are stored
    mask_dir = os.path.join(data_dir, "label", os.path.dirname(rel_path))
    base = os.path.basename(ct_path).replace(".nii.gz", "")

    # Look for corresponding segmentation file in the 'label' subfolders
    for cand in [
        base + "_seg.nii.gz",  # expected name for segmentation file
        base.replace("_ct", "") + "_seg.nii.gz",  # alternative
        base.replace("_ct", "") + ".nii.gz",  # another variation
    ]:
        path = os.path.join(mask_dir, cand)
        if os.path.exists(path):
            return path

    # If no matching segmentation file found, return None
    return None


# ============================================================
# SAVE IMAGE
# ============================================================

def save_png16(path, img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    imageio.imwrite(path, (img * 65535).astype(np.uint16))


# ============================================================
# DRR SETTINGS
# ============================================================

height = 256
width = 256

sdd = 1200
delx = 0.5
dely = 0.5

parameterization = "euler_angles"
convention = "ZYX"
drr_renderer = "trilinear" if device.type == "mps" else "siddon"


# ============================================================
# VIEWS  [FIX Bug 2: jitter removed]
#
# Original used jitter([...]) which added random ±2–5° noise at script
# startup. Those exact angles were never saved, so P matrices in training
# always used the wrong nominal angles. Now using exact nominal angles.
# ============================================================

views_deg = {
    "AP":            [0,    0, 0],
    "LateralLeft":   [90,   0, 0],
    "LateralRight":  [-90,  0, 0],
    "Oblique_Left":  [45,   0, 0],
    "Oblique_Right": [-45,  0, 0],
    "Pedicle_Left":  [30,   5, 0],
    "Pedicle_Right": [-30,  5, 0],
    "AP_Cranial":    [0,   10, 0],
    "AP_Caudal":     [0,  -10, 0],
}

view_names = list(views_deg.keys())

translation = torch.tensor([[0.0, 1200, 0]], dtype=torch.float32, device=device)
translations = translation.repeat(len(view_names), 1)

rotations = torch.from_numpy(
    np.array([np.deg2rad(v) for v in views_deg.values()], dtype=np.float32)
).to(device)


# ============================================================
# HELPERS
# ============================================================

def compute_bbox(mask, margin=250):
    coords = np.argwhere(mask > 0)
    min_x, min_y, min_z = coords.min(axis=0)
    max_x, max_y, max_z = coords.max(axis=0)

    return (
        max(min_x - margin, 0),
        min(max_x + margin, mask.shape[0]),
        max(min_y - margin, 0),
        min(max_y + margin, mask.shape[1]),
        max(min_z - margin, 0),
        min(max_z + margin, mask.shape[2]),
    )


label_to_name = {
    20: "L1",
    21: "L2",
    22: "L3",
    23: "L4",
    24: "L5",
}


# ============================================================
# DLT SOLVER  [FIX Bug 1: new]
# ============================================================

def dlt_solve(pts3d, pts2d):
    """
    Solve for 3x4 projection matrix P using Direct Linear Transform.

    pts3d : (N, 3) world-space coordinates (mm, bone centroid at origin)
    pts2d : (N, 2) pixel coordinates (u, v) in [0, image_size-1]

    Returns P (3, 4) such that:
        x = P @ [X, Y, Z, 1]^T
        u = x[0] / x[2]   (pixel column)
        v = x[1] / x[2]   (pixel row)
    """
    N = pts3d.shape[0]
    X = np.hstack([pts3d, np.ones((N, 1))])  # homogeneous (N, 4)
    A = np.zeros((2 * N, 12))
    for i in range(N):
        u, v = float(pts2d[i, 0]), float(pts2d[i, 1])
        Xi = X[i]
        A[2*i]     = [0, 0, 0, 0,
                      -Xi[0], -Xi[1], -Xi[2], -Xi[3],
                      v*Xi[0],  v*Xi[1],  v*Xi[2],  v*Xi[3]]
        A[2*i + 1] = [Xi[0],  Xi[1],  Xi[2],  Xi[3],
                      0, 0, 0, 0,
                      -u*Xi[0], -u*Xi[1], -u*Xi[2], -u*Xi[3]]
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    # DLT sign ambiguity: SVD returns p or -p with equal validity.
    # The world origin (0,0,0) is always in front of the camera (1200mm away),
    # so P[2,3] = depth of origin must be positive.
    if (P[2] @ np.array([0., 0., 0., 1.])) < 0:
        P = -P
    return P


# ============================================================
# P MATRIX COMPUTATION  [FIX Bug 1: new]
# ============================================================

def compute_all_P_matrices(renderer, world_size=125.0, n_grid=8, image_size=256):
    """
    Compute one 3x4 projection matrix per view using diffdrr's own
    perspective_projection. This guarantees axis consistency — the P matrices
    describe exactly the same geometry used when rendering the DRRs.

    Uses a grid of world-space test points (±56mm, bone centroid at origin)
    to get 3D→2D correspondences, then solves for P via DLT.

    P is the same for all cases because:
      - camera geometry (sdd, delx, dely) is fixed
      - view angles are now fixed (jitter removed)
      - canonicalize() always places the bone centroid at world origin

    Returns: dict  view_name -> P (3, 4) float32 numpy array
    """
    half = world_size * 0.45  # 90% of grid to avoid corners
    g = np.linspace(-half, half, n_grid)
    gx, gy, gz = np.meshgrid(g, g, g, indexing='ij')
    pts3d = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    pts_torch = torch.tensor(pts3d, device=device).unsqueeze(0)  # (1, N, 3)

    trans = torch.tensor([[0.0, 1200.0, 0.0]], dtype=torch.float32, device=device)

    P_matrices = {}
    for view_name, angles_deg in views_deg.items():
        rot = torch.tensor([np.deg2rad(angles_deg)], dtype=torch.float32, device=device)
        pose = diffdrr_convert(rot, trans, parameterization="euler_angles", convention="ZYX")

        with torch.no_grad():
            pixels = renderer.perspective_projection(pose, pts_torch)  # (1, N, 2)
        pixels = pixels.squeeze(0).cpu().numpy()  # (N, 2)

        u, v = pixels[:, 0], pixels[:, 1]
        valid = (np.isfinite(u) & np.isfinite(v) &
                 (u >= 0) & (u <= image_size - 1) &
                 (v >= 0) & (v <= image_size - 1))

        n_valid = int(valid.sum())
        if n_valid < 20:
            print(f"  WARNING {view_name}: only {n_valid} valid correspondences — P may be inaccurate")

        P = dlt_solve(pts3d[valid], pixels[valid])
        P_matrices[view_name] = P.astype(np.float32)
        print(f"  {view_name}: {n_valid}/{len(pts3d)} valid correspondences → P computed")

    return P_matrices


# ============================================================
# COMPUTE AND SAVE P MATRICES  [FIX Bug 1: new block]
#
# Run this once before the main loop. Uses a lightweight dummy renderer
# (content of the volume doesn't matter — perspective_projection only
# uses camera geometry, not the voxel data).
# P files are saved to base_out_dir and loaded by ThirdModelTraining.
# ============================================================

print("\nComputing projection matrices via diffdrr perspective_projection ...")
_dummy_renderer = DRR(
    load_example_ct(),
    sdd=sdd, height=height, width=width, delx=delx, dely=dely,
    renderer=drr_renderer,
).to(device)

P_mats = compute_all_P_matrices(_dummy_renderer)
del _dummy_renderer
gc.collect()

os.makedirs(base_out_dir, exist_ok=True)
for view_name, P in P_mats.items():
    out_path = os.path.join(base_out_dir, f"{view_name}_P.npy")
    np.save(out_path, P)

print(f"Projection matrices saved to {base_out_dir}\n")


# ============================================================
# TEMP DIR
# ============================================================

tmp_dir = tempfile.mkdtemp(prefix="drr_tmp_")


# ============================================================
# MAIN LOOP  (unchanged from Datageneration)
# ============================================================

for ct_path in cases:

    case_id = os.path.basename(ct_path).replace(".nii.gz", "")

    mask_path = find_mask(ct_path)
    #print(f"Mask for {case_id}: {mask_path}")
    if mask_path is None:
        continue

    subject_out_dir = os.path.join(base_out_dir, case_id)

    # Check if every vertebra for this case is already fully processed.
    # A vertebra is done when it has all 9 *_full.png files.
    all_done = all(
        os.path.isdir(os.path.join(subject_out_dir, vname)) and
        len([f for f in os.listdir(os.path.join(subject_out_dir, vname))
             if f.endswith("_full.png")]) == len(view_names)
        for vname in label_to_name.values()
    )
    if all_done:
        print(f"Skipping {case_id} (already processed)")
        continue

    print("\nProcessing:", case_id)
    os.makedirs(subject_out_dir, exist_ok=True)

    ct_img = tio.ScalarImage(ct_path)
    mask_img = tio.LabelMap(mask_path)

    ct_img = tio.ToCanonical()(ct_img)
    mask_img = tio.ToCanonical()(mask_img)
    mask_img = tio.Resample(ct_img)(mask_img)

    ct_np = ct_img.data.squeeze().numpy().astype(np.float32)
    mask_np = mask_img.data.squeeze().numpy().astype(np.int16)

    labels_present = np.unique(mask_np)

    if not any(l in labels_present for l in label_to_name):
        continue

    affine = ct_img.affine.copy()

    for label, name in label_to_name.items():

        if label not in labels_present:
            continue

        # Skip this vertebra if already fully rendered
        vert_out_dir = os.path.join(subject_out_dir, name)
        if (os.path.isdir(vert_out_dir) and
                len([f for f in os.listdir(vert_out_dir)
                     if f.endswith("_full.png")]) == len(view_names)):
            print(f"   Skipping {name} (already processed)")
            continue

        print("  ", name)
        vert_mask = (mask_np == label)

        min_x, max_x, min_y, max_y, min_z, max_z = compute_bbox(vert_mask)

        # FIRST crop
        submask = vert_mask[min_x:max_x, min_y:max_y, min_z:max_z]
        ct_crop = ct_np[min_x:max_x, min_y:max_y, min_z:max_z]

        # ====================================================
        # HARD CENTERING IN VOXEL SPACE
        # ====================================================

        coords = np.argwhere(submask > 0)
        center_voxel = coords.mean(axis=0)

        vol_center = np.array(submask.shape) / 2.0
        shift = (vol_center - center_voxel).astype(int)

        ct_centered = np.zeros_like(ct_crop)
        mask_centered = np.zeros_like(submask)

        src_x0 = max(0, -shift[0])
        src_x1 = min(submask.shape[0], submask.shape[0] - shift[0])

        src_y0 = max(0, -shift[1])
        src_y1 = min(submask.shape[1], submask.shape[1] - shift[1])

        src_z0 = max(0, -shift[2])
        src_z1 = min(submask.shape[2], submask.shape[2] - shift[2])

        dst_x0 = max(0, shift[0])
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        dst_y0 = max(0, shift[1])
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        dst_z0 = max(0, shift[2])
        dst_z1 = dst_z0 + (src_z1 - src_z0)

        ct_centered[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] = \
            ct_crop[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]

        mask_centered[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] = \
            submask[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]

        ct_crop = ct_centered
        submask = mask_centered
        bone_threshold = 150

        ct_bone = ct_crop.copy()
        ct_bone[ct_bone < bone_threshold] = 0

        vert_crop = ct_bone * submask
        ct_crop = ct_bone


        # ====================================================
        # MESH + PCA ALIGNMENT
        # ====================================================

        try:
            verts, faces, _, _ = marching_cubes(submask, level=0.5, spacing=ct_img.spacing)

            verts -= verts.mean(axis=0)

            # Save world-coordinate vertices (mm, centroid at origin) BEFORE any
            # PCA rotation or unit-sphere normalization.  precompute_voxel_cache uses
            # these to rasterize the GT into the same ±62.5 mm world grid as
            # BackprojectionModule, so prediction and GT share one coordinate system.
            verts_world = verts.copy()

            cov = np.cov(verts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            verts = verts @ eigvecs

            for i in range(3):
                if verts[:, i].mean() < 0:
                    verts[:, i] *= -1

            verts /= np.max(np.linalg.norm(verts, axis=1))

            mesh_path = os.path.join(subject_out_dir, name, f"{name}_mesh.npz")
            os.makedirs(os.path.dirname(mesh_path), exist_ok=True)

            np.savez_compressed(mesh_path, verts=verts, faces=faces, verts_world=verts_world)

        except Exception as e:
            print("Mesh failed:", e)

        # ====================================================
        # DRR
        # ====================================================

        full_path = os.path.join(tmp_dir, f"{case_id}_{name}_full.nii.gz")
        vert_path = os.path.join(tmp_dir, f"{case_id}_{name}_vert.nii.gz")

        nib.save(nib.Nifti1Image(ct_crop, affine), full_path)
        nib.save(nib.Nifti1Image(vert_crop, affine), vert_path)

        renderer_full = DRR(read(full_path), sdd=sdd,
                    height=height, width=width,
                    delx=delx, dely=dely,
                    renderer=drr_renderer).to(device)

        renderer_vert = DRR(read(vert_path), sdd=sdd,
                    height=height, width=width,
                    delx=delx, dely=dely,
                    renderer=drr_renderer).to(device)

        full_list, vert_list = [], []

        for i in range(len(view_names)):

            rot = rotations[i:i+1]
            trans = translations[i:i+1]

            with torch.no_grad():
                full_batch = renderer_full(rot, trans,
                                           parameterization=parameterization,
                                           convention=convention)

                vert_batch = renderer_vert(rot, trans,
                                           parameterization=parameterization,
                                           convention=convention)

            full_list.append(full_batch.cpu())
            vert_list.append(vert_batch.cpu())

        full = torch.cat(full_list).numpy()
        vert = torch.cat(vert_list).numpy()

        out_dir = os.path.join(subject_out_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        for i, view in enumerate(view_names):
            f = full[i].squeeze()
            v = vert[i].squeeze()
            m = (v > 0.1).astype(np.float32)

            save_png16(os.path.join(out_dir, f"{view}_full.png"), f)
            save_png16(os.path.join(out_dir, f"{view}_vert.png"), v)
            imageio.imwrite(os.path.join(out_dir, f"{view}_mask.png"), (m * 255).astype(np.uint8))

        del renderer_full, renderer_vert
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


print("\n DONE")