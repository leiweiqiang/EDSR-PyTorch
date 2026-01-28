python main.py --model EDSR --scale 4 --patch_size 192 \
  --data_test Demo --dir_demo "/workspace/EDSR-Lite/test/5/5_lr_x4" \
  --pre_train "/workspace/EDSR-Lite/models/edsr_baseline_x4-6b446fab.pt" \
  --test_only --save_results --cpu --save edsr_baseline_x4_fp32_demo

python - <<'PY'
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

sr_dir = Path("/workspace/EDSR-Lite/experiment/edsr_baseline_x4_fp32_demo/results-Demo")
gt_dir = Path("/workspace/EDSR-Lite/test/5/5_hr")
out_json = sr_dir.parent / "metrics.json"

def psnr(sr, gt):
    sr = sr.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = np.mean((sr - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

def ssim(sr, gt):
    if ssim_fn is None:
        return None
    if sr.ndim == 3 and sr.shape[2] == 3:
        return float(ssim_fn(sr, gt, channel_axis=2, data_range=255))
    return float(ssim_fn(sr, gt, data_range=255))

sr_files = sorted(sr_dir.glob("*_x4_SR.png"))
results = {}
missing = []
psnr_vals = []
ssim_vals = []

for sr_path in sr_files:
    base = sr_path.name.replace("_x4_SR.png", "")
    gt_path = gt_dir / f"{base}.png"
    if not gt_path.exists():
        gt_jpg = gt_dir / f"{base}.jpg"
        gt_path = gt_jpg if gt_jpg.exists() else None
    if gt_path is None or not gt_path.exists():
        missing.append(sr_path.name)
        continue

    sr = imageio.imread(sr_path)
    gt = imageio.imread(gt_path)
    p = float(psnr(sr, gt))
    s = ssim(sr, gt)

    item = {"psnr": p}
    if s is not None:
        item["ssim"] = s
        ssim_vals.append(s)
    results[base] = item
    psnr_vals.append(p)

summary = {
    "count": len(results),
    "missing_gt": missing,
    "mean": {
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
    },
    "per_image": results,
}

out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved metrics to {out_json}")
PY
