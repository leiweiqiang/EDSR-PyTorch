import random

import numpy as np
import skimage.color as sc
import cv2

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

def bicubic_upsample(lr, scale):
    """
    Upsample LR image by scale factor using bicubic interpolation
    
    Args:
        lr: numpy array (H, W, 3) RGB image [0-255]
        scale: upsampling factor (e.g., 8)
    
    Returns:
        upsampled: numpy array (H*scale, W*scale, 3) RGB image [0-255]
    """
    h, w = lr.shape[:2]
    target_h, target_w = h * scale, w * scale
    
    # Use cv2.resize with bicubic interpolation
    upsampled = cv2.resize(lr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    return upsampled

def compute_canny_edge(img):
    """
    Compute Canny edge map from HR ground truth
    
    Args:
        img: numpy array (H, W, 3) RGB image [0-255]
    
    Returns:
        edge_map: numpy array (H, W, 1) edge map [0-255]
    """
    # Convert RGB to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur to reduce noise (standard practice before Canny)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Apply Canny edge detection with hardcoded thresholds
    low_threshold = 100
    high_threshold = 200
    edge = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Expand dimensions to (H, W, 1)
    edge = np.expand_dims(edge, axis=2)
    
    return edge

def add_edge_to_image(rgb, edge):
    """
    Add edge map to RGB image element-wise in numpy space
    
    Args:
        rgb: numpy array (H, W, 3) RGB image [0-255]
        edge: numpy array (H, W, 1) edge map [0-255]
    
    Returns:
        enhanced: numpy array (H, W, 3) RGB + edge, clipped to [0, 255]
    """
    # Broadcasting: (H, W, 3) + (H, W, 1) → (H, W, 3)
    result = rgb.astype(np.float32) + edge.astype(np.float32)
    
    # Clip to [0, 255] and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

