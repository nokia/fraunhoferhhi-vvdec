#!/usr/bin/env python3
"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

"""
Script to encode the weight differences between two ONNX models
using nncodec compression.

Based on the incremental differential neural networks encoding example from:
https://github.com/d-becking/nncodec2/blob/3f69d6528f7b923c1f3efce042adf1c8633e1b88/example/nn_coding.py#L295
"""

import argparse
import os
import sys
import io
import random
import numpy as np
import onnx
import onnx.numpy_helper
from copy import deepcopy
from nncodec.nn import encode, decode
from nncodec.framework.pytorch_model import model_diff, model_add

# Try to import torch for seeding (nncodec may use it internally)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_short_name_mapping(layer_names):
    """
    Create a mapping from original layer names to short names.
    Uses single characters for maximum compression:
    - a-z (26 layers)
    - A-Z (26 more = 52 total)
    - 0-9 (10 more = 62 total)
    - aa, ab, ac... for layers beyond 62
    
    Args:
        layer_names: List of original layer names
        
    Returns:
        Dictionary mapping original names to short names (e.g., 'a', 'b', 'c', ...)
    """
    # Create character pool: lowercase, uppercase, digits
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    name_mapping = {}
    for idx, name in enumerate(sorted(layer_names)):
        if idx < len(chars):
            # Single character
            short_name = chars[idx]
        else:
            # For models with > 62 layers, use two characters (aa, ab, ac, ...)
            first_idx = (idx - len(chars)) // 26
            second_idx = (idx - len(chars)) % 26
            short_name = chars[first_idx] + chars[second_idx]
        name_mapping[name] = short_name
    return name_mapping


def rename_onnx_model(model, name_mapping):
    """
    Create a copy of an ONNX model with renamed initializers.
    
    Args:
        model: Original ONNX model
        name_mapping: Dictionary mapping original names to new names
        
    Returns:
        Modified ONNX model with renamed initializers
    """
    # Create a deep copy to avoid modifying the original
    model_copy = deepcopy(model)
    
    # Rename initializers
    for initializer in model_copy.graph.initializer:
        if initializer.name in name_mapping:
            initializer.name = name_mapping[initializer.name]
    
    return model_copy


def rename_weight_dict(weights_dict, name_mapping):
    """
    Rename keys in a weights dictionary according to name mapping.
    
    Args:
        weights_dict: Dictionary with original layer names as keys
        name_mapping: Dictionary mapping original names to new names
        
    Returns:
        New dictionary with renamed keys
    """
    renamed_dict = {}
    for orig_name, weights in weights_dict.items():
        new_name = name_mapping.get(orig_name, orig_name)
        renamed_dict[new_name] = weights
    return renamed_dict


def load_model_weights(model_path):
    """
    Load model weights from an ONNX file, filtering to only include
    multi-dimensional weight and bias parameters.
    
    Args:
        model_path: Path to the ONNX model file (.onnx)
        
    Returns:
        weights_dict: Dictionary mapping parameter names to numpy arrays
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.onnx'):
        raise ValueError(f"Only ONNX files (.onnx) are supported. Got: {model_path}")
    
    # Load ONNX model
    model = onnx.load(model_path)
    
    # Extract weights from initializers, filtering out scalars and non-trainable parameters
    weights_dict = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        # Convert ONNX tensor to numpy array
        array = onnx.numpy_helper.to_array(initializer)
        
        # Only include actual weight/bias tensors:
        # - Must have at least 1 dimension with size > 1 (not scalars)
        # - Skip common non-weight parameter names (ONNX export artifacts)
        if array.size > 1 and not name.startswith(('val', 'view', 'reshape', 'transpose')):
            weights_dict[name] = array
    
    if not weights_dict:
        raise ValueError(f"No trainable weights found in ONNX model: {model_path}")
    
    return weights_dict


def compute_weight_statistics(weights_diff):
    """
    Compute statistics about the weight differences.
    
    Args:
        weights_diff: Dictionary of weight differences
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    total_params = 0
    total_nonzero = 0
    
    for name, param in weights_diff.items():
        param_array = param if isinstance(param, np.ndarray) else param.cpu().numpy()
        num_params = param_array.size
        num_nonzero = np.count_nonzero(param_array)
        
        total_params += num_params
        total_nonzero += num_nonzero
        
        stats[name] = {
            'shape': param_array.shape,
            'num_params': num_params,
            'num_nonzero': num_nonzero,
            'sparsity': 1.0 - (num_nonzero / num_params) if num_params > 0 else 0.0,
            'mean_abs': np.mean(np.abs(param_array)),
            'max_abs': np.max(np.abs(param_array)),
            'std': np.std(param_array)
        }
    
    overall_sparsity = 1.0 - (total_nonzero / total_params) if total_params > 0 else 0.0
    
    return stats, total_params, total_nonzero, overall_sparsity


def main():
    parser = argparse.ArgumentParser(
        description='Encode weight differences between two model checkpoints using nncodec'
    )
    
    # Model arguments
    parser.add_argument(
        '--base_model',
        type=str,
        default='model3.onnx',
        help='Path to the base ONNX model (default: model3.onnx)'
    )
    parser.add_argument(
        '--updated_model',
        type=str,
        default='model3_overfitted_best.onnx',
        help='Path to the updated/overfitted ONNX model (default: model3_overfitted_best.onnx)'
    )
    parser.add_argument(
        '--output_bitstream',
        type=str,
        default='weights_update.bin',
        help='Path to save the compressed bitstream (default: weights_update.bin)'
    )
    parser.add_argument(
        '--output_base_model',
        type=str,
        default='base_model.onnx',
        help='Path to save the modified base ONNX model with shortened names (default: base_model.onnx)'
    )
    parser.add_argument(
        '--output_decoded',
        type=str,
        default='weights_diff_decoded.npz',
        help='Path to save the decoded weight differences for verification (default: weights_diff_decoded.npz)'
    )
    
    # nncodec encoding parameters
    parser.add_argument(
        '--qp',
        type=int,
        default=-32,
        help='Quantization parameter for weights (default: -32)'
    )
    parser.add_argument(
        '--qp_density',
        type=int,
        default=2,
        help='Quantization density parameter (default: 2)'
    )
    parser.add_argument(
        '--nonweight_qp',
        type=int,
        default=-75,
        help='QP for non-weights, e.g., 1D or BatchNorm params (default: -75)'
    )
    parser.add_argument(
        '--opt_qp',
        action='store_true',
        help='Modify QP layer-wise based on relative layer size within NN'
    )
    parser.add_argument(
        '--use_dq',
        action='store_true',
        help='Enable dependent scalar / Trellis-coded quantization'
    )
    parser.add_argument(
        '--approx_method',
        type=str,
        default='uniform',
        choices=['uniform', 'codebook'],
        help='Approximation method (default: uniform)'
    )
    parser.add_argument(
        '--lsa',
        action='store_true',
        help='Enable Local Scaling Adaptation'
    )
    parser.add_argument(
        '--bnf',
        action='store_true',
        help='Enable BatchNorm Folding'
    )
    parser.add_argument(
        '--sparsity',
        type=float,
        default=0.0,
        help='Sparsity rate (default: 0.0)'
    )
    parser.add_argument(
        '--tca',
        action='store_true',
        help='Enable Temporal Context Adaptation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information during encoding/decoding'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the decoded weights by decoding and comparing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    print(f"Setting random seed to {args.seed} for reproducibility...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("✓ Set seeds for: Python random, NumPy, PyTorch (CPU/CUDA)")
    else:
        print("✓ Set seeds for: Python random, NumPy")
    print()
    
    # Load both ONNX models
    print(f"Loading base model from: {args.base_model}")
    base_numpy_all = load_model_weights(args.base_model)
    
    print(f"Loading updated model from: {args.updated_model}")
    updated_numpy_all = load_model_weights(args.updated_model)
    
    # Find common layers (intersection of both models)
    base_keys = set(base_numpy_all.keys())
    updated_keys = set(updated_numpy_all.keys())
    common_keys = base_keys & updated_keys
    
    if not common_keys:
        raise ValueError("No common layers found between the two models!")
    
    # Report any differences
    missing_in_updated = base_keys - updated_keys
    missing_in_base = updated_keys - base_keys
    
    if missing_in_updated or missing_in_base:
        if args.verbose:
            print(f"\nWarning: Layer name differences detected")
            if missing_in_updated:
                print(f"  Layers only in base model: {len(missing_in_updated)}")
                for name in sorted(missing_in_updated):
                    print(f"    - {name}")
            if missing_in_base:
                print(f"  Layers only in updated model: {len(missing_in_base)}")
                for name in sorted(missing_in_base):
                    print(f"    - {name}")
        print(f"Using {len(common_keys)} common layers for comparison")
    else:
        print(f"✓ Both models have identical layer names ({len(common_keys)} layers)")
    
    # Filter to only include common layers
    base_numpy = {k: base_numpy_all[k] for k in common_keys}
    updated_numpy = {k: updated_numpy_all[k] for k in common_keys}
    
    # Create short name mapping to minimize bitstream overhead
    print("\nCreating short name mapping...")
    name_mapping = create_short_name_mapping(list(common_keys))
    
    if args.verbose:
        print("Name mapping:")
        for orig, short in sorted(name_mapping.items()):
            print(f"  {orig} -> {short}")
    
    print(f"✓ Created mapping for {len(name_mapping)} layers")
    
    # Apply name mapping to weight dictionaries
    print("Applying short names to weight dictionaries...")
    base_numpy = rename_weight_dict(base_numpy, name_mapping)
    updated_numpy = rename_weight_dict(updated_numpy, name_mapping)
    
    # Create and save modified base ONNX model with shortened names
    print("\nCreating modified base ONNX model with shortened names...")
    base_model = onnx.load(args.base_model)
    modified_base_model = rename_onnx_model(base_model, name_mapping)
    onnx.save(modified_base_model, args.output_base_model)
    print(f"✓ Modified base model saved to: {args.output_base_model}")
    
    # Compute the difference between models
    print("\nComputing weight differences...")
    weights_diff = model_diff(updated_numpy, base_numpy)
    
    # Compute and display statistics
    print("\n" + "="*80)
    print("WEIGHT DIFFERENCE STATISTICS")
    print("="*80)
    layer_stats, total_params, total_nonzero, overall_sparsity = compute_weight_statistics(weights_diff)
    
    print(f"\nOverall Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-zero differences: {total_nonzero:,}")
    print(f"  Overall sparsity: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)")
    
    if args.verbose:
        print(f"\nPer-layer Statistics:")
        for name, stats in layer_stats.items():
            print(f"\n  Layer: {name}")
            print(f"    Shape: {stats['shape']}")
            print(f"    Parameters: {stats['num_params']:,}")
            print(f"    Non-zero: {stats['num_nonzero']:,}")
            print(f"    Sparsity: {stats['sparsity']:.4f}")
            print(f"    Mean |Δ|: {stats['mean_abs']:.6e}")
            print(f"    Max |Δ|: {stats['max_abs']:.6e}")
            print(f"    Std: {stats['std']:.6e}")
    
    # Prepare encoding parameters
    encoding_args = vars(args)
    use_case_name = 'NNR_PYT'  # Generic PyTorch use case
    
    # Encode the weight differences
    print("\n" + "="*80)
    print("ENCODING WEIGHT DIFFERENCES")
    print("="*80)
    print(f"Encoding parameters:")
    print(f"  QP: {args.qp}")
    print(f"  QP Density: {args.qp_density}")
    print(f"  Non-weight QP: {args.nonweight_qp}")
    print(f"  Approximation method: {args.approx_method}")
    print(f"  Optimize QP: {args.opt_qp}")
    print(f"  Use DQ: {args.use_dq}")
    print(f"  LSA: {args.lsa}")
    print(f"  BNF: {args.bnf}")
    print(f"  TCA: {args.tca}")
    
    # Set up TCA parameters if enabled
    approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}} if args.tca else None
    
    print("\nEncoding...")
    
    # Suppress nncodec's verbose output unless --verbose is enabled
    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        bitstream = encode(weights_diff, encoding_args, use_case_name, incremental=True, epoch=0, 
                          approx_param_base=approx_param_base)
    finally:
        if not args.verbose:
            sys.stdout = old_stdout
    
    # Save the bitstream
    with open(args.output_bitstream, 'wb') as f:
        f.write(bitstream)
    
    bitstream_size = len(bitstream)
    print(f"\n✓ Encoding complete!")
    print(f"  Bitstream size: {bitstream_size:,} bytes ({bitstream_size/1024:.2f} KB)")
    print(f"  Saved to: {args.output_bitstream}")
    
    # Calculate compression metrics
    uncompressed_size = sum(param.nbytes for param in weights_diff.values())
    compression_ratio = uncompressed_size / bitstream_size if bitstream_size > 0 else 0
    bits_per_param = (bitstream_size * 8) / total_params if total_params > 0 else 0
    
    # Estimate metadata overhead (layer names)
    import json
    layer_names = list(weights_diff.keys())
    layer_names_json = json.dumps(layer_names)
    layer_names_size_estimate = len(layer_names_json.encode('utf-8'))
    
    # Count actual occurrences in bitstream
    layer_names_actual_size = 0
    for name in layer_names:
        # Count how many times the name appears in bitstream
        count = bitstream.count(name.encode('utf-8'))
        layer_names_actual_size += len(name.encode('utf-8')) * count
    
    metadata_percentage = (layer_names_actual_size / bitstream_size * 100) if bitstream_size > 0 else 0
    data_payload_size = bitstream_size - layer_names_actual_size
    
    print(f"\nCompression metrics:")
    print(f"  Uncompressed size: {uncompressed_size:,} bytes ({uncompressed_size/1024:.2f} KB)")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Bits per parameter: {bits_per_param:.4f}")
    
    print(f"\nBitstream breakdown:")
    print(f"  Total bitstream: {bitstream_size:,} bytes (100.0%)")
    print(f"  Compressed data payload: {data_payload_size:,} bytes ({100-metadata_percentage:.1f}%)")
    print(f"  Number of layers: {len(layer_names)}")
    print(f"  Layer names (estimated): {layer_names_actual_size:,} bytes ({metadata_percentage:.1f}%)")
    print(f"  Avg bytes per layer name (estimated): {layer_names_actual_size/len(layer_names) if len(layer_names) > 0 else 0:.1f}")
    
    # Verify by decoding if requested
    if args.verify:
        print("\n" + "="*80)
        print("VERIFICATION - DECODING")
        print("="*80)
        print("Decoding bitstream...")
        
        # Suppress nncodec's verbose output unless --verbose is enabled
        if not args.verbose:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        try:
            decoded_weights_diff = decode(bitstream, encoding_args, approx_param_base=approx_param_base)
        finally:
            if not args.verbose:
                sys.stdout = old_stdout
        
        # Save decoded weights as numpy archive
        np.savez(args.output_decoded, **decoded_weights_diff)
        print(f"✓ Decoded weights saved to: {args.output_decoded}")
        
        # Compare original and decoded differences
        print("\nComparing original vs decoded differences...")
        max_error = 0.0
        mean_error = 0.0
        total_elements = 0
        
        for name in weights_diff.keys():
            orig = weights_diff[name]
            dec = decoded_weights_diff[name]
            
            error = np.abs(orig - dec)
            layer_max_error = np.max(error)
            layer_mean_error = np.mean(error)
            
            if layer_max_error > max_error:
                max_error = layer_max_error
            
            mean_error += np.sum(error)
            total_elements += orig.size
            
            if args.verbose:
                print(f"  {name}: max_error={layer_max_error:.6e}, mean_error={layer_mean_error:.6e}")
        
        mean_error /= total_elements if total_elements > 0 else 1
        
        print(f"\nReconstruction error:")
        print(f"  Max absolute error: {max_error:.6e}")
        print(f"  Mean absolute error: {mean_error:.6e}")
        
        # Reconstruct the updated model from base + decoded differences
        print("\nReconstructing updated model from base + decoded differences...")
        reconstructed_numpy = model_add(base_numpy, decoded_weights_diff)
        
        # Compute reconstruction error relative to original updated model
        print("Computing reconstruction error relative to original updated model...")
        max_model_error = 0.0
        mean_model_error = 0.0
        total_model_elements = 0
        
        for name, orig_param in updated_numpy.items():
            if name in reconstructed_numpy:
                rec_param = reconstructed_numpy[name]
                error = np.abs(orig_param - rec_param)
                layer_max_error = np.max(error)
                layer_mean_error = np.mean(error)
                
                if layer_max_error > max_model_error:
                    max_model_error = layer_max_error
                
                mean_model_error += np.sum(error)
                total_model_elements += orig_param.size
                
                if args.verbose:
                    print(f"  {name}: max_error={layer_max_error:.6e}, mean_error={layer_mean_error:.6e}")
        
        mean_model_error /= total_model_elements if total_model_elements > 0 else 1
        
        print(f"\nModel reconstruction error:")
        print(f"  Max absolute error: {max_model_error:.6e}")
        print(f"  Mean absolute error: {mean_model_error:.6e}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print("\nOutput files:")
    print(f"  1. Compressed bitstream: {args.output_bitstream}")
    print(f"  2. Modified base model: {args.output_base_model}")
    if args.verify:
        print(f"  3. Decoded weights (verification): {args.output_decoded}")


if __name__ == '__main__':
    main()
