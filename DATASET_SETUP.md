# DIV2K 数据集准备指南

## 问题诊断

当前错误：`ValueError: num_samples should be a positive integer value, but got num_samples=0`

**原因**：数据集目录结构不正确或数据集未正确下载。

## 解决方案

### 步骤 1: 下载 DIV2K 数据集

```bash
# 下载数据集 (7.1GB)
cd /Users/leiweiqiang/Documents/GitHub/EDSR-PyTorch
wget https://cv.snu.ac.kr/research/EDSR/DIV2K.tar

# 或者使用 curl
curl -O https://cv.snu.ac.kr/research/EDSR/DIV2K.tar
```

### 步骤 2: 解压数据集到正确位置

```bash
# 确保数据集目录存在
mkdir -p dataset

# 解压数据集
tar -xf DIV2K.tar -C dataset/

# 验证目录结构
ls -la dataset/DIV2K/
```

### 步骤 3: 验证数据集结构

正确的目录结构应该是：

```
dataset/
└── DIV2K/
    ├── DIV2K_train_HR/              # 高分辨率图像 (800张)
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── DIV2K_train_LR_bicubic/     # 低分辨率图像
    │   ├── X2/                      # 2倍下采样
    │   │   ├── 0001x2.png
    │   │   └── ...
    │   ├── X3/                      # 3倍下采样
    │   │   ├── 0001x3.png
    │   │   └── ...
    │   └── X4/                      # 4倍下采样
    │       ├── 0001x4.png
    │       └── ...
    ├── DIV2K_valid_HR/              # 验证集高分辨率 (100张)
    └── DIV2K_valid_LR_bicubic/     # 验证集低分辨率
```

### 步骤 4: 运行检查脚本

```bash
cd src
bash check_dataset.sh
```

### 步骤 5: 首次运行（预处理数据）

首次训练时，使用 `--ext sep_reset` 预处理图像：

```bash
# 修改训练脚本，将 DATA_EXT 改为 "sep_reset"
# 或者直接运行：
python main.py \
    --model EDSR \
    --scale 4 \
    --patch_size 192 \
    --save edsr_baseline_x4_qat_scratch \
    --dir_data ../../../dataset \
    --data_train DIV2K \
    --data_test DIV2K \
    --data_range 1-800/801-810 \
    --ext sep_reset \
    --quantize qat \
    --quantize_backend fbgemm \
    --epochs 300 \
    --batch_size 16 \
    --lr 1e-4 \
    --test_every 1000 \
    --reset \
    --save_quantized
```

## 快速修复命令

如果数据集已经下载但目录结构不对：

```bash
# 1. 检查当前数据集位置
cd /Users/leiweiqiang/Documents/GitHub/EDSR-PyTorch
find . -name "DIV2K_train_HR" -type d

# 2. 如果数据集在其他位置，创建符号链接或移动
# 例如，如果数据集在 ~/Downloads/DIV2K/
mkdir -p dataset
ln -s ~/Downloads/DIV2K dataset/DIV2K
# 或者
cp -r ~/Downloads/DIV2K dataset/
```

## 验证数据集

运行检查脚本验证：

```bash
cd src
bash check_dataset.sh
```

如果看到 "✓ 数据集检查完成"，说明数据集准备就绪！

## 常见问题

### Q: 数据集太大，下载慢怎么办？
A: 可以使用下载工具（如 aria2）加速：
```bash
aria2c -x 16 -s 16 https://cv.snu.ac.kr/research/EDSR/DIV2K.tar
```

### Q: 只有部分数据集可以训练吗？
A: 可以，修改 `--data_range` 参数：
```bash
--data_range 1-100/801-810  # 只用前100张训练
```

### Q: 可以使用其他数据集吗？
A: 可以，但需要按照 DIV2K 的目录结构组织数据。
