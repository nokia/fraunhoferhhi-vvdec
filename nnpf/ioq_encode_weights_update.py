"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

"""
IOQ (Inference-based QP Optimization) for weight update encoding.

This script extends the weight update encoding workflow with per-tensor QP
optimization based on rate-distortion cost. Instead of using a single QP for
all tensors, it searches for optimal per-tensor QPs that minimize the
Lagrangian cost: cost = -dPSNR + lambda * rate

Key features:
- Uses nnc.compress() API which supports qp_per_tensor parameter
- Evaluates quality using PyTorch model inference on the dataset
- Greedy per-tensor QP search with configurable search range
- Lambda estimation from coarse/fine quantization (QP±1)
"""

import argparse
import copy
import json
import os
import sys
import io
import random
import tempfile
import numpy as np
import onnx
import onnx.numpy_helper
import torch
from copy import deepcopy

# nncodec imports
from nncodec import nnc
from nncodec.framework.pytorch_model import model_diff, model_add, np_to_torch, torch_to_numpy

# Local imports
from dataset import Dataset
from model import FilterWithMultipliersPyTorch

# Reuse utilities from encode_weights_update.py
from encode_weights_update import (
    create_short_name_mapping,
    rename_weight_dict,
    rename_onnx_model,
    compute_weight_statistics,
)


def load_model_weights_for_ioq(model_path):
    """
    Load model weights from an ONNX file, including multiplier weights.
    
    Unlike encode_weights_update.load_model_weights, this includes 'view' tensors
    which contain the multiplier weights that are trained during overfitting.
    
    Args:
        model_path: Path to the ONNX model file (.onnx)
        
    Returns:
        weights_dict: Dictionary mapping parameter names to numpy arrays
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load ONNX model
    model = onnx.load(model_path)
    
    # Extract weights from initializers
    # Include view tensors (multipliers) but exclude val/reshape/transpose artifacts
    weights_dict = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        array = onnx.numpy_helper.to_array(initializer)
        
        # Include tensors with size > 1
        # - Include 'view' tensors (these are multiplier weights)
        # - Exclude 'val', 'reshape', 'transpose' artifacts
        if array.size > 1:
            if name.startswith(('val_', 'reshape', 'transpose')):
                continue
            weights_dict[name] = array
    
    if not weights_dict:
        raise ValueError(f"No trainable weights found in ONNX model: {model_path}")
    
    return weights_dict


def de_interleave_luma(blocks):
    """
    De-interleave four image partitions back into a single image.
    Args:
        blocks: torch tensor (N, 4, H, W) - 4 channels representing tl, tr, bl, br
    Returns:
        De-interleaved image (N, 1, 2*H, 2*W)
    """
    n, _, h, w = blocks.shape
    output = torch.zeros(n, 1, h * 2, w * 2, device=blocks.device)
    output[:, :, 0::2, 0::2] = blocks[:, 0:1, :, :]  # tl
    output[:, :, 0::2, 1::2] = blocks[:, 1:2, :, :]  # tr
    output[:, :, 1::2, 0::2] = blocks[:, 2:3, :, :]  # bl
    output[:, :, 1::2, 1::2] = blocks[:, 3:4, :, :]  # br
    return output


def de_interleave_chroma(chroma):
    """
    Chroma channels are not interleaved (already at half resolution).
    Args:
        chroma: torch tensor (N, 1, H, W) - single chroma channel
    Returns:
        chroma: torch tensor (H, W)
    """
    return chroma[0, 0, :, :]


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images (assumed to be in [0, 1] range).
    Args:
        img1, img2: torch tensors
    Returns:
        PSNR in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.item()


class ModelEvaluator:
    """
    Evaluates model quality by running inference on dataset frames.
    Computes average Y-PSNR across all frames.
    """
    
    def __init__(self, model_path, input_yuv, recon_yuv, log_enc, 
                 width, height, bit_depth, block_size, pad_size,
                 num_frames=None, device=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to base PyTorch model (.pt) for architecture
            input_yuv: Path to original YUV file
            recon_yuv: Path to reconstructed YUV file  
            log_enc: Path to encoder log file
            width, height: Frame dimensions
            bit_depth: Bit depth of YUV
            block_size: Block size without padding
            pad_size: Padding size
            num_frames: Optional limit on frames to evaluate
            device: torch device (auto-detect if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block_size = block_size
        self.pad_size = pad_size
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.input_yuv = input_yuv
        self.recon_yuv = recon_yuv
        self.log_enc = log_enc
        
        # Load model architecture
        self.model = FilterWithMultipliersPyTorch()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Store original state dict keys for mapping
        self.model_state_keys = list(self.model.state_dict().keys())
        
        # Determine total frames available
        temp_dataset = Dataset(
            input_yuv=input_yuv,
            recon_yuv=recon_yuv,
            log_enc=log_enc,
            width=width,
            height=height,
            block_size=block_size,
            pad_size=pad_size,
            bit_depth=bit_depth,
            frames=[0],
            device=self.device
        )
        total_frames = len(temp_dataset.frames_info)
        
        # Limit frames if requested
        self.num_frames = min(num_frames, total_frames) if num_frames else total_frames
        
        print(f"ModelEvaluator initialized: {self.num_frames} frames, device={self.device}")
    
    def _apply_weights_to_model(self, weights_dict):
        """
        Apply numpy weight dictionary to the PyTorch model.
        
        Args:
            weights_dict: Dictionary of numpy arrays (can use short names)
        """
        # Convert numpy to torch and load
        state_dict = {}
        for name, value in weights_dict.items():
            if isinstance(value, np.ndarray):
                # Make a copy to avoid non-writable tensor warning
                state_dict[name] = torch.from_numpy(value.copy()).to(self.device)
            else:
                state_dict[name] = value.to(self.device)
        
        self.model.load_state_dict(state_dict, strict=False)
    
    def _process_frame(self, frame_idx):
        """
        Process a single frame through the model and return Y-PSNR.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            (psnr_without, psnr_with) - Y-PSNR without and with filter
        """
        # Create dataset for this specific frame
        dataset = Dataset(
            input_yuv=self.input_yuv,
            recon_yuv=self.recon_yuv,
            log_enc=self.log_enc,
            width=self.width,
            height=self.height,
            block_size=self.block_size,
            pad_size=self.pad_size,
            bit_depth=self.bit_depth,
            frames=[frame_idx],
            device=self.device
        )
        
        # Get metadata from first patch
        first_patch = dataset[0]
        num_v_blocks = first_patch['num_v_blocks']
        num_h_blocks = first_patch['num_h_blocks']
        height = first_patch['height']
        width = first_patch['width']
        poc = first_patch['poc']
        
        # Load original frame for Y-PSNR computation
        orig_yuv = dataset._read_yuv_frame(
            dataset.orig_yuv_path, poc + dataset.frames_to_skip,
            dataset.width, dataset.height, dataset.bit_depth
        )
        orig_y = torch.from_numpy(orig_yuv[0]).to(self.device)
        
        # Reconstruction for baseline PSNR
        reco_yuv = dataset._read_yuv_frame(
            dataset.reco_yuv_path, poc,
            dataset.width, dataset.height, dataset.bit_depth
        )
        reco_y = torch.from_numpy(reco_yuv[0]).to(self.device)
        
        # Baseline PSNR (without filter)
        psnr_without = compute_psnr(orig_y, reco_y)
        
        # Process all patches through the model
        output_patches_list = []
        num_patches = len(dataset)
        
        with torch.no_grad():
            for idx in range(num_patches):
                patch_data = dataset[idx]
                input_patch = patch_data['input'].unsqueeze(0).to(self.device)
                output_patch = self.model(input_patch)
                output_patches_list.append(output_patch[0])
        
        # Stack and extract Y channel (4 interleaved channels)
        output_patches = torch.stack(output_patches_list, dim=0)
        output_patches_y = output_patches[:, :4, :, :]
        
        # Pad if needed
        total_expected = num_v_blocks * num_h_blocks
        if output_patches_y.shape[0] < total_expected:
            padding = torch.zeros(
                total_expected - output_patches_y.shape[0],
                4, self.block_size, self.block_size,
                device=self.device
            )
            output_patches_y = torch.cat([output_patches_y, padding], dim=0)
        
        # Reshape and de-interleave
        output_grid = output_patches_y[:total_expected].view(
            num_v_blocks, num_h_blocks, 4, self.block_size, self.block_size
        )
        output_grid = output_grid.permute(2, 0, 3, 1, 4)
        output_grid = output_grid.reshape(
            4, num_v_blocks * self.block_size, num_h_blocks * self.block_size
        )
        filtered_y = de_interleave_luma(output_grid.unsqueeze(0))
        filtered_y = filtered_y[0, 0, :height, :width]
        
        # PSNR with filter
        psnr_with = compute_psnr(orig_y, filtered_y)
        
        return psnr_without, psnr_with
    
    def eval_model(self, weights_dict):
        """
        Evaluate model with given weights on the dataset.
        
        Args:
            weights_dict: Dictionary of weights (numpy arrays or torch tensors)
            
        Returns:
            Average Y-PSNR across all evaluated frames (with filter)
        """
        # Apply weights
        self._apply_weights_to_model(weights_dict)
        self.model.eval()
        
        y_psnr_sum = 0.0
        
        for frame_idx in range(self.num_frames):
            _, psnr_with = self._process_frame(frame_idx)
            y_psnr_sum += psnr_with
        
        return y_psnr_sum / self.num_frames if self.num_frames > 0 else 0.0
    
    def eval_baseline(self):
        """
        Evaluate baseline PSNR (reconstruction without filter).
        
        Returns:
            Average Y-PSNR without filter across all frames
        """
        y_psnr_sum = 0.0
        
        for frame_idx in range(self.num_frames):
            psnr_without, _ = self._process_frame(frame_idx)
            y_psnr_sum += psnr_without
        
        return y_psnr_sum / self.num_frames if self.num_frames > 0 else 0.0


def compress_and_eval(weights_diff, base_weights, evaluator, qp_per_tensor, args, suppress_output=True):
    """
    Compress weight differences with given per-tensor QPs, decompress, and evaluate.
    
    Args:
        weights_diff: Dictionary of weight differences (numpy)
        base_weights: Dictionary of base model weights (numpy)  
        evaluator: ModelEvaluator instance
        qp_per_tensor: Dictionary mapping tensor names to QP values
        args: Encoding arguments
        suppress_output: Whether to suppress nncodec verbose output
        
    Returns:
        (bitstream_bytes, psnr): Compressed bitstream and quality metric
    """
    # Always provide approx_param_base (required by nnc.compress)
    approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}}
    
    # Suppress nncodec output
    if suppress_output:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.nnc', delete=True) as tmp:
            # Compress with per-tensor QPs
            bitstream = nnc.compress(
                weights_diff,
                bitstream_path=tmp.name,
                codebook_mode=2 if args.approx_method == 'codebook' else 0,
                qp=args.qp,
                nonweight_qp=args.nonweight_qp,
                qp_per_tensor=qp_per_tensor,
                use_dq=args.use_dq,
                opt_qp=False,  # We're doing our own optimization
                row_skipping=True,
                tca=args.tca,
                approx_param_base=approx_param_base,
                compress_differences=True,
                bnf=args.bnf,
                lsa=args.lsa,
                return_bitstream=True,
                verbose=False,
            )
            
            # Decompress
            decoded_diff = nnc.decompress(tmp.name, approx_param_base=approx_param_base)
    finally:
        if suppress_output:
            sys.stdout = old_stdout
    
    # Reconstruct model weights
    reconstructed = model_add(base_weights, decoded_diff)
    
    # Evaluate
    psnr = evaluator.eval_model(reconstructed)
    
    return bitstream, psnr


def run_ioq_optimization(weights_diff, base_weights, evaluator, args):
    """
    Run IOQ optimization to find optimal per-tensor QPs.
    
    Args:
        weights_diff: Dictionary of weight differences
        base_weights: Dictionary of base model weights
        evaluator: ModelEvaluator instance
        args: Encoding arguments
        
    Returns:
        (best_qp_per_tensor, best_bitstream, optimization_log)
    """
    base_qp = args.qp
    tensor_names = list(weights_diff.keys())
    
    print("\n" + "=" * 80)
    print("IOQ: INFERENCE-BASED QP OPTIMIZATION")
    print("=" * 80)
    
    # Initialize per-tensor QP dict with base QP
    qp_per_tensor = {name: base_qp for name in tensor_names}
    
    # Step 1: Get baseline with uniform QP
    print("\n[IOQ] Step 1: Computing baseline (uniform QP)...")
    bitstream_ref, ref_psnr = compress_and_eval(
        weights_diff, base_weights, evaluator, qp_per_tensor, args
    )
    ref_size = len(bitstream_ref)
    
    # Compute base model PSNR (without weight update)
    base_model_psnr = evaluator.eval_model(base_weights)
    ref_dpsnr = ref_psnr - base_model_psnr
    
    print(f"[IOQ] Base model PSNR: {base_model_psnr:.4f} dB")
    print(f"[IOQ] Baseline: size={ref_size} bytes, PSNR={ref_psnr:.4f} dB, dPSNR={ref_dpsnr:.4f} dB")
    
    # Step 2: Estimate lambda from QP-1 and QP+1
    print("\n[IOQ] Step 2: Estimating lambda...")
    
    # QP-1 for all tensors (finer quantization)
    qp_m1 = {name: base_qp - 1 for name in tensor_names}
    bitstream_m1, psnr_m1 = compress_and_eval(
        weights_diff, base_weights, evaluator, qp_m1, args
    )
    size_m1 = len(bitstream_m1)
    dpsnr_m1 = psnr_m1 - base_model_psnr
    
    diff_rate_m1 = size_m1 - ref_size
    diff_acc_m1 = ref_dpsnr - dpsnr_m1
    lambda_m1 = -diff_acc_m1 / diff_rate_m1 if diff_rate_m1 != 0 else 0.0
    print(f"[IOQ] QP-1: size={size_m1} bytes, dPSNR={dpsnr_m1:.4f} dB, lambda={lambda_m1:.8f}")
    
    # QP+1 for all tensors (coarser quantization)
    qp_p1 = {name: base_qp + 1 for name in tensor_names}
    bitstream_p1, psnr_p1 = compress_and_eval(
        weights_diff, base_weights, evaluator, qp_p1, args
    )
    size_p1 = len(bitstream_p1)
    dpsnr_p1 = psnr_p1 - base_model_psnr
    
    diff_rate_p1 = size_p1 - ref_size
    diff_acc_p1 = ref_dpsnr - dpsnr_p1
    lambda_p1 = -diff_acc_p1 / diff_rate_p1 if diff_rate_p1 != 0 else 0.0
    print(f"[IOQ] QP+1: size={size_p1} bytes, dPSNR={dpsnr_p1:.4f} dB, lambda={lambda_p1:.8f}")
    
    # Combined lambda (always non-negative)
    lamb = max((lambda_m1 + lambda_p1) / 2, 0.0)
    if args.lambda_override is not None:
        lamb = args.lambda_override
        print(f"[IOQ] Using override lambda={lamb:.8f}")
    else:
        print(f"[IOQ] Using estimated lambda={lamb:.8f}")
    
    # Step 3: Sort tensors by size (largest first, skip zero-diff tensors)
    tensor_sizes = []
    for name in tensor_names:
        tensor = weights_diff[name]
        if np.sum(np.abs(tensor)) > 0:  # Only non-zero tensors
            tensor_sizes.append((name, np.size(tensor)))
    tensor_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n[IOQ] Step 3: Optimizing QP for {len(tensor_sizes)} non-zero tensors...")
    
    # Step 4: Greedy per-tensor optimization
    best_qp_per_tensor = copy.deepcopy(qp_per_tensor)
    best_cost = sys.float_info.max
    best_bitstream = bitstream_ref
    best_dpsnr = ref_dpsnr
    
    qp_range = args.qp_search_range
    qp_offsets_neg = list(range(-qp_range, 0))  # e.g., [-4, -3, -2, -1]
    qp_offsets_pos = list(range(1, qp_range + 1))  # e.g., [1, 2, 3, 4]
    qp_offset_sets = [qp_offsets_neg, qp_offsets_pos]
    
    optimization_log = []
    
    # Skip largest tensor (typically has minimal impact)
    for i_param, (tensor_name, tensor_size) in enumerate(tensor_sizes[1:]):
        tensor_changed = False
        
        for i_qp_set, qp_set in enumerate(qp_offset_sets):
            for i_qp_off, qp_offset in enumerate(qp_set):
                # Progress indicator
                total_offsets = len(qp_offsets_neg) + len(qp_offsets_pos)
                offset_idx = i_qp_set * len(qp_offsets_neg) + i_qp_off + 1
                print(f"\r[IOQ] Tensor {i_param + 1}/{len(tensor_sizes) - 1}, "
                      f"QP offset {offset_idx}/{total_offsets}: {tensor_name[:30]:<30}", end="")
                
                # Try this QP offset
                qp_test = copy.deepcopy(best_qp_per_tensor)
                qp_test[tensor_name] = base_qp + qp_offset
                
                bitstream_test, psnr_test = compress_and_eval(
                    weights_diff, base_weights, evaluator, qp_test, args
                )
                size_test = len(bitstream_test)
                dpsnr_test = psnr_test - base_model_psnr
                
                # Skip if quality drops below base model
                if dpsnr_test < 0:
                    continue
                
                # Compute Lagrangian cost
                diff_rate = size_test - ref_size
                diff_acc = ref_dpsnr - dpsnr_test
                cost = diff_acc + lamb * diff_rate
                
                if cost < best_cost:
                    best_qp_per_tensor = copy.deepcopy(qp_test)
                    best_cost = cost
                    best_bitstream = bitstream_test
                    best_dpsnr = dpsnr_test
                    tensor_changed = True
                    
                    optimization_log.append({
                        'tensor': tensor_name,
                        'qp_offset': qp_offset,
                        'size': size_test,
                        'dpsnr': dpsnr_test,
                        'cost': cost
                    })
                    print(f" -> New best! cost={cost:.6f}, dPSNR={dpsnr_test:.4f}")
        
        if not tensor_changed:
            print()  # Newline after tensor with no improvement
    
    print(f"\n\n[IOQ] Optimization complete!")
    print(f"[IOQ] Best cost: {best_cost:.6f}")
    
    return best_qp_per_tensor, best_bitstream, optimization_log, base_model_psnr


def main():
    parser = argparse.ArgumentParser(
        description='IOQ (Inference-based QP Optimization) for weight update encoding'
    )
    
    # Model arguments
    parser.add_argument('--base_model', type=str, default='model3.onnx',
                        help='Path to base ONNX model (default: model3.onnx)')
    parser.add_argument('--updated_model', type=str, default='model3_overfitted_best.onnx',
                        help='Path to updated/overfitted ONNX model (default: model3_overfitted_best.onnx)')
    parser.add_argument('--model_path', type=str, default='model3.pt',
                        help='Path to PyTorch model for evaluation (default: model3.pt)')
    parser.add_argument('--output_bitstream', type=str, default='weights_update_ioq.bin',
                        help='Path to save the compressed bitstream (default: weights_update_ioq.bin)')
    parser.add_argument('--output_base_model', type=str, default='base_model.onnx',
                        help='Path to save modified base ONNX model with short names (default: base_model.onnx)')
    parser.add_argument('--output_qp_map', type=str, default='qp_per_tensor.json',
                        help='Path to save optimized QP map (default: qp_per_tensor.json)')
    
    # Dataset arguments (same as overfit.py)
    parser.add_argument('--input_yuv', type=str, default='input.yuv',
                        help='Path to input (original) YUV file')
    parser.add_argument('--recon_yuv', type=str, default='enc_rec.yuv',
                        help='Path to reconstructed YUV file')
    parser.add_argument('--log_enc', type=str, default='log_enc.txt',
                        help='Path to encoder log file')
    parser.add_argument('--width', type=int, default=1920,
                        help='Frame width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Frame height in pixels (default: 1080)')
    parser.add_argument('--bit_depth', type=int, default=10,
                        help='Bit depth of images (default: 10)')
    parser.add_argument('--block_size', type=int, default=64,
                        help='Block size without padding (default: 64)')
    parser.add_argument('--pad_size', type=int, default=8,
                        help='Padding size (default: 8)')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to use for evaluation (None for all)')
    
    # nncodec encoding parameters
    parser.add_argument('--qp', type=int, default=-32,
                        help='Base quantization parameter (default: -32)')
    parser.add_argument('--nonweight_qp', type=int, default=-75,
                        help='QP for non-weights (default: -75)')
    parser.add_argument('--approx_method', type=str, default='uniform', choices=['uniform', 'codebook'],
                        help='Approximation method (default: uniform)')
    parser.add_argument('--use_dq', action='store_true',
                        help='Enable dependent scalar / Trellis-coded quantization')
    parser.add_argument('--lsa', action='store_true',
                        help='Enable Local Scaling Adaptation')
    parser.add_argument('--bnf', action='store_true',
                        help='Enable BatchNorm Folding')
    parser.add_argument('--tca', action='store_true',
                        help='Enable Temporal Context Adaptation')
    
    # IOQ-specific parameters
    parser.add_argument('--qp_search_range', type=int, default=4,
                        help='QP search range for per-tensor optimization (default: 4, searches ±4)')
    parser.add_argument('--lambda_override', type=float, default=None,
                        help='Override estimated lambda with fixed value')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    # Set random seeds
    print(f"Setting random seed to {args.seed}...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print()
    
    # Validate inputs
    for path, name in [(args.base_model, 'base_model'), 
                       (args.updated_model, 'updated_model'),
                       (args.model_path, 'model_path'),
                       (args.input_yuv, 'input_yuv'),
                       (args.recon_yuv, 'recon_yuv'),
                       (args.log_enc, 'log_enc')]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            sys.exit(1)
    
    print("=" * 80)
    print("IOQ WEIGHT UPDATE ENCODING")
    print("=" * 80)
    print(f"Base ONNX model: {args.base_model}")
    print(f"Updated ONNX model: {args.updated_model}")
    print(f"PyTorch model (eval): {args.model_path}")
    print(f"Dataset: {args.input_yuv}, {args.recon_yuv}")
    print(f"Resolution: {args.width}x{args.height}, bit_depth={args.bit_depth}")
    print(f"Base QP: {args.qp}, search range: ±{args.qp_search_range}")
    print("-" * 80)
    
    # Load ONNX models and extract weights (including multiplier weights)
    print("\nLoading ONNX models...")
    base_numpy_all = load_model_weights_for_ioq(args.base_model)
    updated_numpy_all = load_model_weights_for_ioq(args.updated_model)
    
    # Find common layers
    common_keys = set(base_numpy_all.keys()) & set(updated_numpy_all.keys())
    if not common_keys:
        print("Error: No common layers found between models!")
        sys.exit(1)
    print(f"✓ Found {len(common_keys)} common layers")
    
    # Filter to common layers (keep original names for IOQ evaluation)
    base_numpy = {k: base_numpy_all[k] for k in common_keys}
    updated_numpy = {k: updated_numpy_all[k] for k in common_keys}
    
    # Create short name mapping (for final encoding)
    print("\nCreating short name mapping...")
    name_mapping = create_short_name_mapping(list(common_keys))
    reverse_mapping = {v: k for k, v in name_mapping.items()}
    
    # Compute weight differences (with original names for PyTorch compatibility)
    print("Computing weight differences...")
    weights_diff = model_diff(updated_numpy, base_numpy)
    
    # Statistics
    stats, total_params, total_nonzero, overall_sparsity = compute_weight_statistics(weights_diff)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-zero differences: {total_nonzero:,}")
    print(f"  Sparsity: {overall_sparsity*100:.2f}%")
    
    # Initialize evaluator
    print("\nInitializing model evaluator...")
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        input_yuv=args.input_yuv,
        recon_yuv=args.recon_yuv,
        log_enc=args.log_enc,
        width=args.width,
        height=args.height,
        bit_depth=args.bit_depth,
        block_size=args.block_size,
        pad_size=args.pad_size,
        num_frames=args.num_frames,
    )
    
    # Run IOQ optimization (using original names)
    best_qp_per_tensor, best_bitstream_orig_names, opt_log, base_psnr = run_ioq_optimization(
        weights_diff, base_numpy, evaluator, args
    )
    
    # Now re-encode with short names for final bitstream
    print("\n" + "=" * 80)
    print("FINAL ENCODING WITH SHORT NAMES")
    print("=" * 80)
    
    # Apply short name mapping
    base_numpy_short = rename_weight_dict(base_numpy, name_mapping)
    weights_diff_short = rename_weight_dict(weights_diff, name_mapping)
    
    # Map QPs to short names
    qp_per_tensor_short = {}
    for orig_name, qp in best_qp_per_tensor.items():
        short_name = name_mapping.get(orig_name, orig_name)
        qp_per_tensor_short[short_name] = qp
    
    # Encode with short names
    approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}}
    
    # Suppress nncodec output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.nnc', delete=True) as tmp:
            best_bitstream = nnc.compress(
                weights_diff_short,
                bitstream_path=tmp.name,
                codebook_mode=2 if args.approx_method == 'codebook' else 0,
                qp=args.qp,
                nonweight_qp=args.nonweight_qp,
                qp_per_tensor=qp_per_tensor_short,
                use_dq=args.use_dq,
                opt_qp=False,
                row_skipping=True,
                tca=args.tca,
                approx_param_base=approx_param_base,
                compress_differences=True,
                bnf=args.bnf,
                lsa=args.lsa,
                return_bitstream=True,
                verbose=False,
            )
    finally:
        sys.stdout = old_stdout
    
    # Save modified base ONNX model with short names
    print(f"Saving modified base ONNX model: {args.output_base_model}")
    base_model = onnx.load(args.base_model)
    modified_base_model = rename_onnx_model(base_model, name_mapping)
    onnx.save(modified_base_model, args.output_base_model)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save optimized bitstream
    with open(args.output_bitstream, 'wb') as f:
        f.write(best_bitstream)
    print(f"✓ Optimized bitstream: {args.output_bitstream} ({len(best_bitstream):,} bytes)")
    
    # Save QP map (with both name mappings for reference)
    qp_map_output = {
        'qp_per_tensor': qp_per_tensor_short,
        'qp_per_tensor_original_names': best_qp_per_tensor,
        'name_mapping': name_mapping,
        'reverse_mapping': reverse_mapping,
        'base_qp': args.qp,
        'optimization_log': opt_log
    }
    with open(args.output_qp_map, 'w') as f:
        json.dump(qp_map_output, f, indent=2)
    print(f"✓ QP map: {args.output_qp_map}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Compare with uniform QP baseline (using original names for evaluation)
    print("\nComputing baseline with uniform QP for comparison...")
    qp_uniform = {name: args.qp for name in weights_diff.keys()}
    baseline_bitstream, baseline_psnr = compress_and_eval(
        weights_diff, base_numpy, evaluator, qp_uniform, args
    )
    baseline_dpsnr = baseline_psnr - base_psnr
    
    # Get final metrics from IOQ optimization
    # (we already computed these during optimization)
    _, final_psnr = compress_and_eval(
        weights_diff, base_numpy, evaluator, best_qp_per_tensor, args
    )
    final_dpsnr = final_psnr - base_psnr
    
    print(f"\nBase model PSNR:     {base_psnr:.4f} dB")
    print(f"\nUniform QP={args.qp}:")
    print(f"  Size:  {len(baseline_bitstream):,} bytes")
    print(f"  PSNR:  {baseline_psnr:.4f} dB (dPSNR: {baseline_dpsnr:+.4f} dB)")
    print(f"\nIOQ Optimized:")
    print(f"  Size:  {len(best_bitstream):,} bytes ({len(best_bitstream) - len(baseline_bitstream):+,} bytes)")
    print(f"  PSNR:  {final_psnr:.4f} dB (dPSNR: {final_dpsnr:+.4f} dB)")
    
    # List tensors with non-default QP
    non_default_qps = {k: v for k, v in best_qp_per_tensor.items() if v != args.qp}
    if non_default_qps:
        print(f"\nTensors with optimized QP ({len(non_default_qps)} tensors):")
        for name, qp in sorted(non_default_qps.items()):
            print(f"  {name}: QP={qp} (offset={qp - args.qp:+d})")
    else:
        print("\nNo tensors were optimized (uniform QP was optimal)")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
