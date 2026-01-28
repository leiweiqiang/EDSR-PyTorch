#!/bin/bash

# ==============================================================================
# 数据集检查脚本
# ==============================================================================

echo "=========================================="
echo "检查数据集配置..."
echo "=========================================="

DATASET_PATH="../dataset"
DIV2K_PATH="${DATASET_PATH}/DIV2K"

echo "数据集路径: ${DATASET_PATH}"
echo "DIV2K路径: ${DIV2K_PATH}"
echo ""

# 检查目录是否存在
if [ ! -d "${DATASET_PATH}" ]; then
    echo "❌ 错误: 数据集目录不存在: ${DATASET_PATH}"
    echo "请创建目录或修改脚本中的 DATASET_PATH"
    exit 1
fi

if [ ! -d "${DIV2K_PATH}" ]; then
    echo "❌ 错误: DIV2K 目录不存在: ${DIV2K_PATH}"
    echo "请下载并解压 DIV2K 数据集"
    exit 1
fi

echo "✓ 数据集目录存在"
echo ""

# 检查期望的目录结构
HR_DIR="${DIV2K_PATH}/DIV2K_train_HR"
LR_DIR="${DIV2K_PATH}/DIV2K_train_LR_bicubic"

echo "检查期望的目录结构:"
echo "  HR目录: ${HR_DIR}"
echo "  LR目录: ${LR_DIR}"
echo ""

if [ ! -d "${HR_DIR}" ]; then
    echo "❌ 错误: HR目录不存在: ${HR_DIR}"
    echo ""
    echo "请确保数据集结构如下:"
    echo "  ${DIV2K_PATH}/"
    echo "    ├── DIV2K_train_HR/          (高分辨率图像)"
    echo "    └── DIV2K_train_LR_bicubic/  (低分辨率图像)"
    echo "        ├── X2/"
    echo "        ├── X3/"
    echo "        └── X4/"
    exit 1
fi

if [ ! -d "${LR_DIR}" ]; then
    echo "❌ 错误: LR目录不存在: ${LR_DIR}"
    exit 1
fi

echo "✓ 目录结构正确"
echo ""

# 检查图像文件
HR_COUNT=$(find "${HR_DIR}" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
LR_COUNT=$(find "${LR_DIR}" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')

echo "检查图像文件:"
echo "  HR图像数量: ${HR_COUNT}"
echo "  LR图像数量: ${LR_COUNT}"
echo ""

if [ "${HR_COUNT}" -eq 0 ]; then
    echo "❌ 错误: HR目录中没有找到PNG图像文件"
    echo ""
    echo "数据集下载链接:"
    echo "  https://cv.snu.ac.kr/research/EDSR/DIV2K.tar"
    echo ""
    echo "下载后解压命令:"
    echo "  tar -xf DIV2K.tar -C ${DATASET_PATH}/"
    exit 1
fi

if [ "${LR_COUNT}" -eq 0 ]; then
    echo "⚠️  警告: LR目录中没有找到PNG图像文件"
    echo "如果使用 --ext sep，程序会自动生成二进制文件"
fi

echo "✓ 数据集检查完成"
echo ""
echo "数据集准备就绪，可以开始训练！"
