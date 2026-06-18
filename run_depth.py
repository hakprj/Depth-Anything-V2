# =========================
# Depth Anything V2 (ViT-L) -> depth PNGs (uint16)
# =========================
import os, sys, subprocess, argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# --------- paths ----------
_parser = argparse.ArgumentParser()
_parser.add_argument("--root", type=str, default=None,
                     help="Root folder containing 'images/'. Output goes to <root>/depths")
_args, _ = _parser.parse_known_args()

if _args.root:
    _root = Path(os.path.expanduser(_args.root))
    img_dir = _root / "images"
    out_dir = _root / "depths"
else:
    img_dir = Path("/mnt/c/Users/hayra/Projects/3d/inputs/thin_casual_recording/images")
    out_dir = Path("/mnt/c/Users/hayra/Projects/3d/inputs/thin_casual_recording/depths")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[DepthAnythingV2] images: {img_dir}")
print(f"[DepthAnythingV2] depths: {out_dir}")

# --------- install Depth Anything V2 ----------
repo_dir = Path("/home/hayrap/repos/gl_image_to_3d/external/Depth-Anything-V2")


# Make import work
sys.path.append(str(repo_dir))

from depth_anything_v2.dpt import DepthAnythingV2

# --------- download best checkpoint (vitl) ----------
ckpt_dir = repo_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / "depth_anything_v2_vitl.pth"

if not ckpt_path.exists():
    subprocess.run([
        "bash", "-lc",
        "wget -q -O /home/hayrap/repos/gl_image_to_3d/external/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth "
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    ], check=True)

# --------- load model ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For DA-v2, these constructor args are used in the official usage examples
model = DepthAnythingV2(
    encoder="vitl",
    features=256,
    out_channels=[256, 512, 1024, 1024],
).to(DEVICE).eval()

state = torch.load(str(ckpt_path), map_location="cpu")
model.load_state_dict(state)

print("✅ Loaded Depth Anything V2 ViT-L on", DEVICE)

# --------- run inference ----------
exts = (".jpg", ".jpeg", ".png")
img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
print("Found images:", len(img_paths))

if len(img_paths) == 0:
    raise RuntimeError(f"No images found in: {img_dir}")

# recommended for stability vs per-image min/max: percentile normalization
use_percentile_norm = True
p_lo, p_hi = 2, 98

for i, p in enumerate(img_paths, 1):
    bgr = cv2.imread(str(p))
    if bgr is None:
        print("Skip unreadable:", p)
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        depth = model.infer_image(rgb, 518)  # float32-like, HxW

    d = depth.astype(np.float32)

    if use_percentile_norm:
        lo, hi = np.percentile(d, [p_lo, p_hi])
        d = np.clip(d, lo, hi)
        d = (d - lo) / (hi - lo + 1e-8)
    else:
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)

    depth_u16 = (d * 65535.0).astype(np.uint16)

    out_path = out_dir / (p.stem + ".png")
    cv2.imwrite(str(out_path), depth_u16)

    if i % 25 == 0 or i == len(img_paths):
        print(f"Saved {i}/{len(img_paths)}")

print("✅ Done. Depth maps in:", out_dir)
