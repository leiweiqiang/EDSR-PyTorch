#!/bin/bash

# ==============================================================================
# 简化版 QAT 从零训练脚本
# 快速开始训练 EDSR Baseline x4 模型
# ==============================================================================

# 配置参数 - 请修改为你的数据集路径
# 注意：从 src 目录出发，使用 ../dataset
DATASET_PATH="../dataset"

# 运行训练
python main.py \
    --model EDSR \
    --scale 4 \
    --patch_size 192 \
    --save edsr_baseline_x4_qat_scratch \
    --dir_data ${DATASET_PATH} \
    --data_train DIV2K \
    --data_test DIV2K \
    --data_range 1-800/801-810 \
    --ext sep \
    --quantize qat \
    --quantize_backend fbgemm \
    --epochs 300 \
    --batch_size 16 \
    --lr 1e-4 \
    --test_every 1000 \
    --reset \
    --save_quantized

echo "训练完成！模型保存在: ../experiment/edsr_baseline_x4_qat_scratch/"
