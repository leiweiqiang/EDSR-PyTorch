#!/bin/bash

# ==============================================================================
# PTQ INT4 Script (Weight-Only)
# ==============================================================================
#
# 使用方法:
#   1. 修改下面的配置参数（数据集路径、模型路径等）
#   2. 运行: bash ptq_int4.sh
#
# 注意:
#   - INT4 仅对权重量化，激活保持浮点
#   - 不需要校准数据集
#
# ==============================================================================

# ==============================================================================
# 配置参数 - 请根据实际情况修改
# ==============================================================================

# 数据集路径（请修改为你的数据集路径）
# 注意：从 src 目录出发，使用 ../dataset
DATASET_PATH="../dataset"

# 测试数据集名称
DATA_TEST="DIV2K"

# 测试数据范围
DATA_RANGE="801-810"

# 数据格式
DATA_EXT="sep"

# 预训练 FP32 模型路径
PRETRAIN_PATH="../models/edsr_baseline_x4-6b446fab.pt"

# ==============================================================================
# EDSR Baseline x4 PTQ INT4 (weight-only)
# ==============================================================================

python main.py \
    --model EDSR \
    --scale 4 \
    --patch_size 192 \
    --save edsr_baseline_x4_ptq_int4 \
    --dir_data ${DATASET_PATH} \
    --data_test ${DATA_TEST} \
    --data_range ${DATA_RANGE} \
    --ext ${DATA_EXT} \
    --quantize ptq_int4 \
    --test_only \
    --pre_train ${PRETRAIN_PATH} \
    --save_quantized \
    --reset

echo "PTQ INT4 完成！量化模型保存在: ../experiment/edsr_baseline_x4_ptq_int4/model/"
