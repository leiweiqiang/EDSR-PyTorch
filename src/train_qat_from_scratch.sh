#!/bin/bash

# ==============================================================================
# QAT (Quantization-Aware Training) Training Script from Scratch
# 从零开始的量化感知训练脚本
# ==============================================================================
# 
# 使用方法:
#   1. 修改下面的配置参数（数据集路径、模型参数等）
#   2. 取消注释要训练的模型配置
#   3. 运行: bash train_qat_from_scratch.sh
#
# 注意:
#   - 此脚本不使用预训练模型，完全从零开始训练
#   - QAT 训练会在训练过程中模拟量化误差
#   - 训练完成后会自动转换为 INT8 量化模型
#   - 确保已准备好 DIV2K 数据集
#
# ==============================================================================

# ==============================================================================
# 配置参数 - 请根据实际情况修改
# ==============================================================================

# 数据集路径（请修改为你的数据集路径）
# 注意：从 src 目录出发，使用 ../dataset 而不是 ../../../dataset
DATASET_PATH="../dataset"

# 训练数据集名称
DATA_TRAIN="DIV2K"

# 测试数据集名称
DATA_TEST="DIV2K"

# 训练/测试数据范围 (DIV2K: 1-800训练, 801-900测试)
DATA_RANGE="1-800/801-810"

# 数据格式 (sep_reset: 生成bin, sep: 使用bin, img: 直接读png)
# 若测试集为空，先用 PREPARE_BIN=1 生成完整 bin，再用 sep
DATA_EXT="sep"

# 预处理开关：1 = 先生成 bin，0 = 直接训练
PREPARE_BIN=0

if [ "${PREPARE_BIN}" -eq 1 ]; then
    DATA_EXT="sep_reset"
fi

# 量化后端 (fbgemm: CPU, qnnpack: 移动端)
QUANTIZE_BACKEND="fbgemm"

# 训练轮数（QAT 通常需要较少的轮数）
EPOCHS=300

# 批次大小（根据GPU内存调整）
BATCH_SIZE=16

# 学习率
LEARNING_RATE=1e-4

# 测试频率（每N个batch测试一次）
TEST_EVERY=1000

# ==============================================================================
# EDSR Baseline 模型配置
# ==============================================================================

# EDSR Baseline x2 - 从零开始 QAT 训练
#python main.py \
#    --model EDSR \
#    --scale 2 \
#    --patch_size 96 \
#    --save edsr_baseline_x2_qat_scratch \
#    --dir_data ${DATASET_PATH} \
#    --data_train ${DATA_TRAIN} \
#    --data_test ${DATA_TEST} \
#    --data_range ${DATA_RANGE} \
#    --ext ${DATA_EXT} \
#    --quantize qat \
#    --quantize_backend ${QUANTIZE_BACKEND} \
#    --epochs ${EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --lr ${LEARNING_RATE} \
#    --test_every ${TEST_EVERY} \
#    --reset \
#    --save_quantized

# EDSR Baseline x3 - 从零开始 QAT 训练
#python main.py \
#    --model EDSR \
#    --scale 3 \
#    --patch_size 144 \
#    --save edsr_baseline_x3_qat_scratch \
#    --dir_data ${DATASET_PATH} \
#    --data_train ${DATA_TRAIN} \
#    --data_test ${DATA_TEST} \
#    --data_range ${DATA_RANGE} \
#    --ext ${DATA_EXT} \
#    --quantize qat \
#    --quantize_backend ${QUANTIZE_BACKEND} \
#    --epochs ${EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --lr ${LEARNING_RATE} \
#    --test_every ${TEST_EVERY} \
#    --reset \
#    --save_quantized

# EDSR Baseline x4 - 从零开始 QAT 训练（推荐）
python main.py \
    --model EDSR \
    --scale 4 \
    --patch_size 192 \
    --save edsr_baseline_x4_qat_scratch \
    --dir_data ${DATASET_PATH} \
    --data_train ${DATA_TRAIN} \
    --data_test ${DATA_TEST} \
    --data_range ${DATA_RANGE} \
    --ext ${DATA_EXT} \
    --quantize qat \
    --quantize_backend ${QUANTIZE_BACKEND} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --test_every ${TEST_EVERY} \
    --reset \
    --save_quantized

# ==============================================================================
# EDSR Paper 模型配置（更大模型，需要更多内存）
# ==============================================================================

# EDSR Paper x2 - 从零开始 QAT 训练
#python main.py \
#    --model EDSR \
#    --scale 2 \
#    --patch_size 96 \
#    --save edsr_x2_qat_scratch \
#    --n_resblocks 32 \
#    --n_feats 256 \
#    --res_scale 0.1 \
#    --dir_data ${DATASET_PATH} \
#    --data_train ${DATA_TRAIN} \
#    --data_test ${DATA_TEST} \
#    --data_range ${DATA_RANGE} \
#    --ext ${DATA_EXT} \
#    --quantize qat \
#    --quantize_backend ${QUANTIZE_BACKEND} \
#    --epochs ${EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --lr ${LEARNING_RATE} \
#    --test_every ${TEST_EVERY} \
#    --reset \
#    --save_quantized

# EDSR Paper x3 - 从零开始 QAT 训练
#python main.py \
#    --model EDSR \
#    --scale 3 \
#    --patch_size 144 \
#    --save edsr_x3_qat_scratch \
#    --n_resblocks 32 \
#    --n_feats 256 \
#    --res_scale 0.1 \
#    --dir_data ${DATASET_PATH} \
#    --data_train ${DATA_TRAIN} \
#    --data_test ${DATA_TEST} \
#    --data_range ${DATA_RANGE} \
#    --ext ${DATA_EXT} \
#    --quantize qat \
#    --quantize_backend ${QUANTIZE_BACKEND} \
#    --epochs ${EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --lr ${LEARNING_RATE} \
#    --test_every ${TEST_EVERY} \
#    --reset \
#    --save_quantized

# EDSR Paper x4 - 从零开始 QAT 训练
#python main.py \
#    --model EDSR \
#    --scale 4 \
#    --patch_size 192 \
#    --save edsr_x4_qat_scratch \
#    --n_resblocks 32 \
#    --n_feats 256 \
#    --res_scale 0.1 \
#    --dir_data ${DATASET_PATH} \
#    --data_train ${DATA_TRAIN} \
#    --data_test ${DATA_TEST} \
#    --data_range ${DATA_RANGE} \
#    --ext ${DATA_EXT} \
#    --quantize qat \
#    --quantize_backend ${QUANTIZE_BACKEND} \
#    --epochs ${EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --lr ${LEARNING_RATE} \
#    --test_every ${TEST_EVERY} \
#    --reset \
#    --save_quantized

# ==============================================================================
# 其他模型配置示例
# ==============================================================================

# RDN BI x4 - 从零开始 QAT 训练
#python main.py \
#    --model RDN \
#    --scale 4 \
#    --patch_size 128 \
#    --save rdn_bix4_qat_scratch \
#    --dir_data ${DATASET_PATH} \
#    --data_train ${DATA_TRAIN} \
#    --data_test ${DATA_TEST} \
#    --data_range ${DATA_RANGE} \
#    --ext ${DATA_EXT} \
#    --quantize qat \
#    --quantize_backend ${QUANTIZE_BACKEND} \
#    --epochs ${EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --lr ${LEARNING_RATE} \
#    --test_every ${TEST_EVERY} \
#    --reset \
#    --save_quantized

# RCAN x4 - 从零开始 QAT 训练
#python main.py \
#    --template RCAN \
#    --model RCAN \
#    --scale 4 \
#    --patch_size 192 \
#    --save rcan_bix4_qat_scratch \
#    --dir_data ${DATASET_PATH} \
#    --data_train ${DATA_TRAIN} \
#    --data_test ${DATA_TEST} \
#    --data_range ${DATA_RANGE} \
#    --ext ${DATA_EXT} \
#    --quantize qat \
#    --quantize_backend ${QUANTIZE_BACKEND} \
#    --epochs ${EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --lr ${LEARNING_RATE} \
#    --test_every ${TEST_EVERY} \
#    --reset \
#    --save_quantized \
#    --chop

# ==============================================================================
# 训练完成后的测试命令（取消注释以测试训练好的模型）
# ==============================================================================

# 测试训练好的 QAT 模型
#python main.py \
#    --model EDSR \
#    --scale 4 \
#    --data_test Set5+Set14+B100+Urban100+DIV2K \
#    --data_range 801-900 \
#    --pre_train ../experiment/edsr_baseline_x4_qat_scratch/model_best.pt \
#    --test_only \
#    --save_results

# ==============================================================================
# 说明
# ==============================================================================
#
# 1. 首次运行前，请确保：
#    - 已下载并准备好 DIV2K 数据集
#    - 修改 DATASET_PATH 为正确的数据集路径
#    - 首次运行使用 --ext sep_reset 预处理数据
#
# 2. QAT 训练特点：
#    - 训练过程中会模拟量化误差
#    - 模型会学习适应量化带来的精度损失
#    - 训练完成后自动转换为 INT8 量化模型
#
# 3. 训练参数建议：
#    - epochs: 300 (baseline) 或更多 (paper版本)
#    - batch_size: 根据GPU内存调整 (16, 32, 64)
#    - lr: 1e-4 (baseline) 或 5e-5 (paper版本)
#    - 学习率衰减: 默认在 epoch 200 时衰减
#
# 4. 输出文件：
#    - 模型保存在: ../experiment/[save_name]/model/
#    - 量化模型: model_best_quantized.pt (如果使用 --save_quantized)
#    - 训练日志: ../experiment/[save_name]/log.txt
#
# ==============================================================================
