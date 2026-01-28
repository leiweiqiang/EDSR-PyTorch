"""
Model Quantization Module for INT8 Quantization
Supports both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from tqdm import tqdm

def get_qconfig(backend='fbgemm'):
    """
    Get quantization configuration for the specified backend.
    
    Args:
        backend: Quantization backend ('fbgemm' for CPU, 'qnnpack' for mobile)
    
    Returns:
        QConfig object
    """
    if backend == 'fbgemm':
        return quantization.get_default_qconfig('fbgemm')
    elif backend == 'qnnpack':
        return quantization.get_default_qconfig('qnnpack')
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'fbgemm' or 'qnnpack'")

def fuse_modules(model):
    """
    Fuse modules in the model for better quantization performance.
    Fuses Conv+BN+ReLU and Conv+ReLU patterns.
    
    Args:
        model: The model to fuse
    
    Returns:
        Fused model
    """
    # For EDSR, we need to handle the structure carefully
    # The model has head, body (with ResBlocks), and tail
    # EDSR typically doesn't use BatchNorm, so we focus on Conv+ReLU fusion
    
    # Fuse head modules if possible (Conv+ReLU pattern)
    if hasattr(model, 'head') and isinstance(model.head, nn.Sequential):
        modules_list = list(model.head)
        if len(modules_list) >= 2:
            try:
                # Check if we have Conv+ReLU pattern
                if isinstance(modules_list[0], nn.Conv2d) and isinstance(modules_list[1], nn.ReLU):
                    model.head = quantization.fuse_modules(model.head, [['0', '1']])
            except Exception:
                # If fusion fails, continue without fusing
                pass
    
    # Fuse body modules (ResBlocks)
    if hasattr(model, 'body') and isinstance(model.body, nn.Sequential):
        # ResBlocks have a body Sequential with Conv+ReLU+Conv pattern
        # We can fuse Conv+ReLU in each ResBlock
        for i, module in enumerate(model.body):
            if hasattr(module, 'body') and isinstance(module.body, nn.Sequential):
                modules_list = list(module.body)
                if len(modules_list) >= 2:
                    try:
                        # Check if first two are Conv and ReLU
                        if isinstance(modules_list[0], nn.Conv2d) and isinstance(modules_list[1], nn.ReLU):
                            # Fuse first Conv+ReLU
                            module.body = quantization.fuse_modules(module.body, [['0', '1']])
                    except Exception:
                        # If fusion fails, continue without fusing
                        pass
    
    # Fuse tail modules if possible
    # Tail has Upsampler and Conv
    # Upsampler contains Conv+PixelShuffle patterns which are harder to fuse
    # We'll leave tail as is for now
    
    return model

def prepare_model_for_quantization(model, backend='fbgemm'):
    """
    Prepare model for quantization by setting qconfig and fusing modules.
    
    Args:
        model: The model to prepare
        backend: Quantization backend ('fbgemm' or 'qnnpack')
    
    Returns:
        Prepared model
    """
    # Set quantization config
    qconfig = get_qconfig(backend)
    
    # Set qconfig for all modules
    model.qconfig = qconfig
    
    # Set qconfig for submodules
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.qconfig = qconfig
    
    # Fuse modules for better performance
    model = fuse_modules(model)
    
    return model

def calibrate_model(model, calibration_loader, num_samples=100):
    """
    Calibrate model for post-training quantization using calibration dataset.
    
    Args:
        model: The prepared model
        calibration_loader: DataLoader for calibration data
        num_samples: Number of samples to use for calibration
    
    Returns:
        Calibrated model
    """
    model.eval()
    
    # Prepare model for calibration
    model = quantization.prepare(model, inplace=False)
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Run calibration
    count = 0
    with torch.no_grad():
        for batch_idx, (lr, hr, _) in enumerate(calibration_loader):
            if count >= num_samples:
                break
            
            # Move to same device as model
            lr = lr.to(device)
            
            # Forward pass for calibration
            _ = model(lr)
            
            count += lr.size(0)
    
    return model

def convert_to_quantized(model):
    """
    Convert prepared/calibrated model to quantized INT8 model.
    
    Args:
        model: The prepared or calibrated model
    
    Returns:
        Quantized model
    """
    model.eval()
    quantized_model = quantization.convert(model, inplace=False)
    return quantized_model

def quantize_model(model, calibration_loader=None, num_samples=100, backend='fbgemm'):
    """
    Complete post-training quantization workflow.
    
    Args:
        model: The model to quantize
        calibration_loader: DataLoader for calibration (required for PTQ)
        num_samples: Number of calibration samples
        backend: Quantization backend
    
    Returns:
        Quantized model
    """
    # Step 1: Prepare model
    model = prepare_model_for_quantization(model, backend)
    
    # Step 2: Calibrate if calibration data provided
    if calibration_loader is not None:
        model = calibrate_model(model, calibration_loader, num_samples)
    else:
        # If no calibration data, just prepare (will use default observers)
        model = quantization.prepare(model, inplace=False)
        # Run a dummy forward pass to initialize observers
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, 48, 48).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Step 3: Convert to quantized model
    quantized_model = convert_to_quantized(model)
    
    return quantized_model

def prepare_qat_model(model, backend='fbgemm'):
    """
    Prepare model for Quantization-Aware Training (QAT).
    
    Args:
        model: The model to prepare for QAT
        backend: Quantization backend
    
    Returns:
        QAT-prepared model
    """
    # Set quantization config
    qconfig = get_qconfig(backend)
    
    # Set qconfig for all modules
    model.qconfig = qconfig
    
    # Set qconfig for submodules
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.qconfig = qconfig
    
    # Fuse modules
    model = fuse_modules(model)
    
    # Prepare for QAT
    model.train()
    qat_model = quantization.prepare_qat(model, inplace=False)
    
    return qat_model

def save_quantized_model(model, save_path, use_torchscript=True):
    """
    Save quantized model.
    
    Args:
        model: The quantized model
        save_path: Path to save the model
        use_torchscript: Whether to save as TorchScript (recommended for quantized models)
    """
    model.eval()
    
    if use_torchscript:
        # Create a dummy input for tracing
        dummy_input = torch.randn(1, 3, 48, 48)
        
        # Trace the model
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(traced_model, save_path)
        except Exception as e:
            print(f"Warning: Failed to save as TorchScript: {e}")
            print("Falling back to state_dict save...")
            torch.save(model.state_dict(), save_path)
    else:
        # Save state dict
        torch.save(model.state_dict(), save_path)

def load_quantized_model(model, load_path, use_torchscript=True):
    """
    Load quantized model.
    
    Args:
        model: The model architecture (for state_dict loading)
        load_path: Path to load the model from
        use_torchscript: Whether the saved model is TorchScript
    
    Returns:
        Loaded model
    """
    if use_torchscript:
        try:
            loaded_model = torch.jit.load(load_path)
            return loaded_model
        except Exception as e:
            print(f"Warning: Failed to load TorchScript model: {e}")
            print("Trying to load as state_dict...")
            model.load_state_dict(torch.load(load_path, map_location='cpu'))
            return model
    else:
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
        return model

def get_model_size(model, quantized=False):
    """
    Get model size in MB.
    
    Args:
        model: The model
        quantized: Whether the model is quantized
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        if quantized:
            # Quantized models use INT8 (1 byte per parameter)
            param_size += param.numel() * 1
        else:
            # FP32 models use 4 bytes per parameter
            param_size += param.numel() * 4
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * 4
    
    total_size = (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
    return total_size
