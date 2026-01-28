#!/bin/bash

# ==============================================================================
# PTQ (Post-Training Quantization) Script
# 对预训练模型进行后训练量化
# ==============================================================================
#
# 使用方法:
#   1. 修改下面的配置参数（数据集路径、预训练模型路径等）
#   2. 运行: bash ptq.sh
#
# 注意:
#   - PTQ 需要一个已训练好的 FP32 模型（--pre_train）
#   - 需要校准数据集（使用 data_train 进行校准）
#   - 量化模型在 CPU 上运行与保存
#
# ==============================================================================

# ==============================================================================
# 配置参数 - 请根据实际情况修改
# ==============================================================================

# 数据集路径（请修改为你的数据集路径）
# 注意：从 src 目录出发，使用 ../dataset
DATASET_PATH="../dataset"

# 训练/校准数据集名称（用于 PTQ 校准）
DATA_TRAIN="DIV2K"

# 测试数据集名称
DATA_TEST="DIV2K"

# 训练/测试数据范围 (DIV2K: 1-800训练, 801-900测试)
DATA_RANGE="1-800/801-810"

# 数据格式 (sep_reset: 生成bin, sep: 使用bin, img: 直接读png)
DATA_EXT="sep"

# 是否自动下载官方 EDSR Baseline x4 预训练模型
# 1 = 自动下载, 0 = 使用本地路径
AUTO_DOWNLOAD_PRETRAIN=1

# 预训练 FP32 模型路径（当 AUTO_DOWNLOAD_PRETRAIN=0 时使用）
PRETRAIN_PATH="../experiment/edsr_baseline_x4/model_best.pt"

# 量化后端 (fbgemm: CPU, qnnpack: 移动端)
QUANTIZE_BACKEND="fbgemm"

# 校准样本数量
CALIBRATION_SAMPLES=100

# ==============================================================================
# EDSR Baseline x4 PTQ
# ==============================================================================

if [ "${AUTO_DOWNLOAD_PRETRAIN}" -eq 1 ]; then
    PRETRAIN_PATH="download"
    RESUME_OPTION=0
else
    RESUME_OPTION=0
fi

python main.py \
    --model EDSR \
    --scale 4 \
    --patch_size 192 \
    --save edsr_baseline_x4_ptq \
    --dir_data ${DATASET_PATH} \
    --data_train ${DATA_TRAIN} \
    --data_test ${DATA_TEST} \
    --data_range ${DATA_RANGE} \
    --ext ${DATA_EXT} \
    --quantize ptq \
    --quantize_backend ${QUANTIZE_BACKEND} \
    --calibration_samples ${CALIBRATION_SAMPLES} \
    --test_only \
    --pre_train ${PRETRAIN_PATH} \
    --resume ${RESUME_OPTION} \
    --save_quantized \
    --reset

echo "PTQ 完成！量化模型保存在: ../experiment/edsr_baseline_x4_ptq/model/"

python - <<'PY'
import json
import os
from pathlib import Path

save_dir = Path("../experiment/edsr_baseline_x4_ptq").resolve()
model_dir = save_dir / "model"
quantized_path = model_dir / "model_ptq_quantized.pt"
config_path = save_dir / "config.txt"

pretrain_path = os.environ.get("PRETRAIN_PATH", "download")
if pretrain_path == "download":
    models_dir = Path("../models").resolve()
    candidates = sorted(models_dir.glob("edsr_baseline_x4-*.pt"))
    pretrain_file = candidates[-1] if candidates else None
else:
    pretrain_file = Path(pretrain_path).resolve()

def file_size_bytes(path):
    try:
        return path.stat().st_size
    except Exception:
        return None

def bytes_to_mb(size_bytes):
    if size_bytes is None:
        return None
    return size_bytes / (1024 * 1024)

orig_size_b = file_size_bytes(pretrain_file) if pretrain_file else None
quant_size_b = file_size_bytes(quantized_path)
reduction_b = orig_size_b - quant_size_b if (orig_size_b is not None and quant_size_b is not None) else None
reduction_ratio = None
if orig_size_b and quant_size_b is not None:
    reduction_ratio = 1 - (quant_size_b / orig_size_b)

params = {}
if config_path.exists():
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            params[key.strip()] = value.strip()

summary = {
    "paths": {
        "pretrain_fp32": str(pretrain_file) if pretrain_file else None,
        "quantized_ptq": str(quantized_path) if quantized_path.exists() else None,
        "config": str(config_path) if config_path.exists() else None,
    },
    "sizes": {
        "fp32_bytes": orig_size_b,
        "fp32_mb": bytes_to_mb(orig_size_b),
        "quantized_bytes": quant_size_b,
        "quantized_mb": bytes_to_mb(quant_size_b),
        "reduced_bytes": reduction_b,
        "reduced_mb": bytes_to_mb(reduction_b),
        "reduced_ratio": reduction_ratio,
    },
    "params": params,
}

summary_path = save_dir / "summary.md"
summary_path.parent.mkdir(parents=True, exist_ok=True)
lines = []
lines.append("# PTQ Summary")
lines.append("")
lines.append("## Paths")
lines.append(f"- fp32: `{summary['paths']['pretrain_fp32']}`")
lines.append(f"- quantized_ptq: `{summary['paths']['quantized_ptq']}`")
lines.append(f"- config: `{summary['paths']['config']}`")
lines.append("")
lines.append("## Sizes")
lines.append(f"- fp32_bytes: `{summary['sizes']['fp32_bytes']}`")
lines.append(f"- fp32_mb: `{summary['sizes']['fp32_mb']}`")
lines.append(f"- quantized_bytes: `{summary['sizes']['quantized_bytes']}`")
lines.append(f"- quantized_mb: `{summary['sizes']['quantized_mb']}`")
lines.append(f"- reduced_bytes: `{summary['sizes']['reduced_bytes']}`")
lines.append(f"- reduced_mb: `{summary['sizes']['reduced_mb']}`")
lines.append(f"- reduced_ratio: `{summary['sizes']['reduced_ratio']}`")
lines.append("")
lines.append("## Params")
for key in sorted(summary["params"].keys()):
    lines.append(f"- {key}: `{summary['params'][key]}`")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Summary saved to {summary_path}")
PY
