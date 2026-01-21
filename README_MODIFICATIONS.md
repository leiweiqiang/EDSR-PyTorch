# EDSR Modifications: Pre-Upsampling with Canny Edge Enhancement

## Overview

This document describes the modifications made to the EDSR-PyTorch codebase to implement a pre-upsampling approach with Canny edge map enhancement. Instead of learning upsampling within the model, we:

1. **Pre-upsample** low-resolution (LQ) images to high-resolution (HQ) size using bicubic interpolation
2. **Extract Canny edge maps** from the high-resolution ground truth images
3. **Add edge maps** to the upscaled LQ images (element-wise addition)
4. **Train the model** to enhance the already-upscaled images (input and output are the same size)

## Key Changes

### 1. Model Architecture (`src/model/edsr.py`)

- **Removed upsampler module** from the tail
- Model now expects input and output to be the same spatial size
- Architecture simplified: `Head → Body → Tail` (no upsampling layer)

**Before:**
```python
m_tail = [
    common.Upsampler(conv, scale, n_feats, act=False),  # Removed
    conv(n_feats, args.n_colors, kernel_size)
]
```

**After:**
```python
m_tail = [
    conv(n_feats, args.n_colors, kernel_size)  # Direct output
]
```

### 2. Data Processing (`src/data/common.py`)

Added three new helper functions:

#### `bicubic_upsample(lr, scale)`
- Upsamples LR images to HQ size using bicubic interpolation
- Uses `cv2.resize()` with `INTER_CUBIC` mode
- Input/Output: numpy arrays in [0-255] range

#### `compute_canny_edge(img)`
- Computes Canny edge map from HR ground truth images
- Applies Gaussian blur `(5, 5)` with sigma `1.4` before edge detection
- Uses hardcoded thresholds: `low=100`, `high=200`
- Returns edge map as `(H, W, 1)` numpy array [0-255]

#### `add_edge_to_image(rgb, edge)`
- Adds edge map to RGB image element-wise
- Broadcasting: `(H, W, 3) + (H, W, 1) → (H, W, 3)`
- Clips values to [0, 255] maximum
- Returns enhanced RGB image

### 3. Training Data Pipeline (`src/data/srdata.py`)

Modified `get_patch()` method for training:

1. Extract LR and HR patches
2. **Upscale LR** to HR size using bicubic interpolation
3. **Compute Canny edge** from HR ground truth
4. **Add edge** to upscaled LR (element-wise addition)
5. Apply augmentation (if enabled)
6. Return enhanced LR and HR (both same size)

**Training Flow (x8, patch_size=512):**
```
LR patch (64×64×3) 
  → Bicubic x8 → (512×512×3)
  → Add edge from HR → (512×512×3)
  → Model → Enhanced (512×512×3)
  → Loss vs HR (512×512×3)
```

### 4. Inference (`src/data/demo.py`)

- Upscales input LR images using bicubic
- Computes edge from upscaled LR (placeholder - will be replaced with provided HQ edge map)
- Adds edge to upscaled LR before passing to model

## Training Instructions

### Basic Training Command

```bash
python main.py --scale 8 --patch_size 512 --save edsr_bicubic_edge_x8 --reset
```

### Parameters

- `--scale 8`: Upscaling factor (can be 2, 3, 4, 8, etc.)
- `--patch_size 512`: High-resolution patch size (LR patch will be `patch_size / scale`)
- `--save`: Experiment name for saving models and logs
- `--reset`: Start training from scratch

### Example: Training x8 Model

```bash
# For x8 scale with patch_size=512:
# - LR patches: 64×64
# - HR patches: 512×512
# - After upsampling: 64×64 → 512×512 (bicubic)
# - Model processes: 512×512 → 512×512

python main.py \
    --scale 8 \
    --patch_size 512 \
    --batch_size 4 \
    --n_resblocks 16 \
    --n_feats 64 \
    --save edsr_bicubic_edge_x8 \
    --reset
```

**Note:** With `patch_size=512`, use smaller `batch_size` (4-8) due to higher memory requirements from pre-upsampled inputs.

### Example: Training x4 Model

```bash
# For x4 scale with patch_size=256:
# - LR patches: 64×64
# - HR patches: 256×256

python main.py \
    --scale 4 \
    --patch_size 256 \
    --batch_size 8 \
    --save edsr_bicubic_edge_x4 \
    --reset
```

## Implementation Details

### Edge Detection Parameters

- **Gaussian Blur**: Kernel size `(5, 5)`, sigma `1.4`
- **Canny Thresholds**: Low `100`, High `200` (hardcoded)
- **Edge Range**: [0, 255] (binary-like after Canny)

### Normalization

- Images are processed in numpy space [0-255]
- Edge maps are added in numpy space [0-255]
- Values are clipped to [0, 255] after addition
- Conversion to tensors uses existing `rgb_range` parameter (default: 255)

### Memory Considerations

- **Input size**: Larger than original (already upscaled)
- **Memory usage**: ~16× more for x8 scale (compared to original LR input)
- **Recommendation**: May need to reduce `batch_size` if running out of memory

### Model Capacity

- **Parameters reduced**: ~295K fewer parameters (removed upsampler)
- **Focus**: Model now focuses solely on enhancement/refinement
- **Opportunity**: Could add more residual blocks to compensate

## Advantages

1. **Simpler architecture**: No learnable upsampling module
2. **Faster inference**: Bicubic upsampling is very fast
3. **Edge guidance**: Canny edges provide structural information
4. **Flexible scales**: Easy to switch scales via `--scale` parameter

## Limitations

1. **Bicubic artifacts**: Model must correct bicubic upsampling artifacts
2. **Edge quality**: Depends on Canny edge detection quality
3. **Memory**: Higher memory usage due to larger inputs
4. **Pre-trained models**: Existing EDSR checkpoints are incompatible

## File Structure

```
src/
├── data/
│   ├── common.py          # Added: bicubic_upsample, compute_canny_edge, add_edge_to_image
│   ├── srdata.py          # Modified: get_patch() for training
│   └── demo.py            # Modified: __getitem__() for inference
└── model/
    └── edsr.py            # Modified: Removed upsampler from tail
```

## Testing

To test the implementation:

```bash
# Test with a single image
python main.py \
    --data_test Demo \
    --scale 8 \
    --pre_train <path_to_model> \
    --test_only \
    --save_results
```

## Notes

- Edge maps are computed from HR ground truth during training
- For inference, HQ edge maps should be provided separately (see `demo.py` for placeholder)
- The implementation maintains compatibility with existing data loading infrastructure
- All modifications preserve the original normalization scheme

## Future Work

- [ ] Implement HQ edge map loading for inference
- [ ] Experiment with different edge detection parameters
- [ ] Compare performance with standard EDSR
- [ ] Explore multi-scale training with this approach

