#!/usr/bin/env python3
"""
Script to decode weight differences from a compressed bitstream
and reconstruct the updated ONNX model.

This script reverses the encoding process performed by encode_weights_update.py:
1. Loads the modified base model (with short names)
2. Decodes the compressed weight differences bitstream
3. Applies the differences to the base model
4. Saves the reconstructed model as an ONNX file
"""

import argparse
import os
import sys
import io
import tempfile
import urllib.request
import urllib.parse
import numpy as np
import onnx
import onnx.numpy_helper
from copy import deepcopy
from nncodec.nn import decode
from nncodec.framework.pytorch_model import model_add


def load_model_weights(model_path):
    """
    Load model weights from an ONNX file.
    
    Args:
        model_path: Path to the ONNX model file (.onnx)
        
    Returns:
        Tuple of (weights_dict, onnx_model)
        - weights_dict: Dictionary mapping parameter names to numpy arrays
        - onnx_model: The loaded ONNX model object
    """
    if not os.path.exists(model_path):
        abs_path = os.path.abspath(model_path)
        cwd = os.getcwd()
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"  Absolute path: {abs_path}\n"
            f"  Current directory: {cwd}"
        )
    
    if not model_path.endswith('.onnx'):
        raise ValueError(f"Only ONNX files (.onnx) are supported. Got: {model_path}")
    
    # Load ONNX model
    model = onnx.load(model_path)
    
    # Extract all weights from initializers
    weights_dict = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        # Convert ONNX tensor to numpy array
        array = onnx.numpy_helper.to_array(initializer)
        # Include all parameters that were in the encoding (filter out ONNX export artifacts)
        if array.size > 1 and not name.startswith(('val', 'view', 'reshape', 'transpose')):
            weights_dict[name] = array
    
    if not weights_dict:
        raise ValueError(f"No trainable weights found in ONNX model: {model_path}")
    
    return weights_dict, model


def apply_weights_to_model(model, weights_dict):
    """
    Apply weight values to an ONNX model.
    
    Args:
        model: ONNX model to modify
        weights_dict: Dictionary of weights to apply
        
    Returns:
        Modified ONNX model with updated weights
    """
    # Create a deep copy to avoid modifying the original
    model_copy = deepcopy(model)
    
    # Update initializers with new weights
    for initializer in model_copy.graph.initializer:
        if initializer.name in weights_dict:
            # Convert numpy array to ONNX tensor
            new_tensor = onnx.numpy_helper.from_array(
                weights_dict[initializer.name],
                name=initializer.name
            )
            # Replace the initializer
            initializer.CopyFrom(new_tensor)
    
    return model_copy


def fetch_model_from_url(url, temp_dir=None):
    """
    Fetch a model file from a URL (file:// or https://) and save to a temporary file.
    
    Args:
        url: URL to fetch the model from (file:// or https://)
        temp_dir: Optional temporary directory to use
        
    Returns:
        Path to the downloaded temporary file
    """
    print(f"Fetching model from URL: {url}")
    
    # Create a temporary file (with .onnx suffix for clarity)
    temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.onnx', delete=False, dir=temp_dir)
    temp_path = temp_file.name
    
    try:
        # Fetch the URL
        with urllib.request.urlopen(url) as response:
            data = response.read()
            temp_file.write(data)
        temp_file.close()
        
        file_size = os.path.getsize(temp_path)
        print(f"✓ Downloaded model: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
        return temp_path
    except Exception as e:
        temp_file.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(f"Failed to fetch model from URL: {e}")


def load_bitstream(bitstream_path):
    """
    Load a compressed bitstream from file.
    
    Args:
        bitstream_path: Path to the bitstream file
        
    Returns:
        Bytes object containing the bitstream
    """
    if not os.path.exists(bitstream_path):
        raise FileNotFoundError(f"Bitstream file not found: {bitstream_path}")
    
    with open(bitstream_path, 'rb') as f:
        bitstream = f.read()
    
    return bitstream


def compute_model_statistics(weights_dict):
    """
    Compute statistics about model weights.
    
    Args:
        weights_dict: Dictionary of model weights
        
    Returns:
        Dictionary with statistics
    """
    total_params = 0
    total_size_bytes = 0
    layer_count = len(weights_dict)
    
    for name, param in weights_dict.items():
        param_array = param if isinstance(param, np.ndarray) else param
        total_params += param_array.size
        total_size_bytes += param_array.nbytes
    
    return {
        'num_layers': layer_count,
        'total_params': total_params,
        'total_size_bytes': total_size_bytes
    }


def main():
    parser = argparse.ArgumentParser(
        description='Decode compressed weight differences and reconstruct the updated ONNX model'
    )
    
    # Input arguments
    parser.add_argument(
        '--base_model',
        type=str,
        default='base_model_short_names.onnx',
        help='Path to the modified base ONNX model with short names (default: base_model_short_names.onnx)'
    )
    parser.add_argument(
        '--base_model_url',
        type=str,
        help='URL to fetch the base ONNX model from (file:// or https://). If provided, this takes precedence over --base_model'
    )
    parser.add_argument(
        '--bitstream',
        type=str,
        default='weights_update.bin',
        help='Path to the compressed bitstream file (default: weights_update.bin)'
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default='model_reconstructed.onnx',
        help='Path to save the reconstructed ONNX model (default: model_reconstructed.onnx)'
    )
    
    # nncodec decoding parameters (must match encoding parameters)
    parser.add_argument(
        '--qp',
        type=int,
        default=-32,
        help='Quantization parameter used during encoding (default: -32)'
    )
    parser.add_argument(
        '--qp_density',
        type=int,
        default=2,
        help='Quantization density parameter used during encoding (default: 2)'
    )
    parser.add_argument(
        '--nonweight_qp',
        type=int,
        default=-75,
        help='QP for non-weights used during encoding (default: -75)'
    )
    parser.add_argument(
        '--opt_qp',
        action='store_true',
        help='QP optimization was used during encoding'
    )
    parser.add_argument(
        '--use_dq',
        action='store_true',
        help='Dependent quantization was used during encoding'
    )
    parser.add_argument(
        '--approx_method',
        type=str,
        default='uniform',
        choices=['uniform', 'codebook'],
        help='Approximation method used during encoding (default: uniform)'
    )
    parser.add_argument(
        '--lsa',
        action='store_true',
        help='Local Scaling Adaptation was used during encoding'
    )
    parser.add_argument(
        '--bnf',
        action='store_true',
        help='BatchNorm Folding was used during encoding'
    )
    parser.add_argument(
        '--sparsity',
        type=float,
        default=0.0,
        help='Sparsity rate used during encoding (default: 0.0)'
    )
    parser.add_argument(
        '--tca',
        action='store_true',
        help='Temporal Context Adaptation was used during encoding'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information during decoding'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Compare with reference model if available (requires original base model for name mapping)'
    )
    parser.add_argument(
        '--reference_model',
        type=str,
        help='Path to reference updated model for verification (e.g., model3_overfitted_best.onnx)'
    )
    parser.add_argument(
        '--original_base_model',
        type=str,
        help='Path to original base model (needed for name mapping in verification, e.g., model3.onnx)'
    )
    
    args = parser.parse_args()
    
    # Determine base model path - fetch from URL if provided
    temp_model_path = None
    if args.base_model_url:
        base_model_path = fetch_model_from_url(args.base_model_url)
        temp_model_path = base_model_path  # Track for cleanup
    else:
        base_model_path = args.base_model
    
    # Load the modified base model
    print(f"Loading base model from: {base_model_path}")
    base_weights, base_model = load_model_weights(base_model_path)
    
    base_stats = compute_model_statistics(base_weights)
    print(f"✓ Loaded base model:")
    print(f"  Layers: {base_stats['num_layers']}")
    print(f"  Parameters: {base_stats['total_params']:,}")
    print(f"  Size: {base_stats['total_size_bytes']:,} bytes ({base_stats['total_size_bytes']/1024:.2f} KB)")
    
    # Get the bitstream file path (nncodec decode expects a path, not bytes)
    print(f"\nUsing compressed bitstream from: {args.bitstream}")
    if not os.path.exists(args.bitstream):
        raise FileNotFoundError(f"Bitstream file not found: {args.bitstream}")
    
    bitstream_path = args.bitstream
    bitstream_size = os.path.getsize(bitstream_path)
    print(f"✓ Bitstream file:")
    print(f"  Size: {bitstream_size:,} bytes ({bitstream_size/1024:.2f} KB)")
    
    # Prepare decoding parameters
    decoding_args = vars(args)
    
    # Set up TCA parameters if enabled
    approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}} if args.tca else None
    
    print("\n" + "="*80)
    print("DECODING WEIGHT DIFFERENCES")
    print("="*80)
    print(f"Decoding parameters:")
    print(f"  QP: {args.qp}")
    print(f"  QP Density: {args.qp_density}")
    print(f"  Non-weight QP: {args.nonweight_qp}")
    print(f"  Approximation method: {args.approx_method}")
    print(f"  Optimize QP: {args.opt_qp}")
    print(f"  Use DQ: {args.use_dq}")
    print(f"  LSA: {args.lsa}")
    print(f"  BNF: {args.bnf}")
    print(f"  TCA: {args.tca}")
    
    print("\nDecoding...")
    
    # Suppress nncodec's verbose output unless --verbose is enabled
    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        decoded_weights_diff = decode(bitstream_path, decoding_args, approx_param_base=approx_param_base)
    finally:
        if not args.verbose:
            sys.stdout = old_stdout
    
    print(f"✓ Decoding complete!")
    print(f"  Decoded {len(decoded_weights_diff)} weight difference tensors")
    
    # Compute statistics on decoded differences
    diff_stats = compute_model_statistics(decoded_weights_diff)
    print(f"\nDecoded differences:")
    print(f"  Layers: {diff_stats['num_layers']}")
    print(f"  Parameters: {diff_stats['total_params']:,}")
    
    # Compute statistics on the differences
    total_nonzero = 0
    mean_abs_diff = 0.0
    max_abs_diff = 0.0
    
    for name, diff in decoded_weights_diff.items():
        total_nonzero += np.count_nonzero(diff)
        abs_diff = np.abs(diff)
        mean_abs_diff += np.sum(abs_diff)
        max_abs_diff = max(max_abs_diff, np.max(abs_diff))
    
    if diff_stats['total_params'] > 0:
        mean_abs_diff /= diff_stats['total_params']
        sparsity = 1.0 - (total_nonzero / diff_stats['total_params'])
    else:
        sparsity = 0.0
    
    print(f"  Non-zero differences: {total_nonzero:,}")
    print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"  Mean |Δ|: {mean_abs_diff:.6e}")
    print(f"  Max |Δ|: {max_abs_diff:.6e}")
    
    # Reconstruct the updated model
    print("\n" + "="*80)
    print("RECONSTRUCTING UPDATED MODEL")
    print("="*80)
    print("Applying weight differences to base model...")
    
    reconstructed_weights = model_add(base_weights, decoded_weights_diff)
    
    print(f"✓ Reconstructed {len(reconstructed_weights)} weight tensors")
    
    # Create the reconstructed ONNX model
    print("\nCreating ONNX model...")
    reconstructed_model = apply_weights_to_model(base_model, reconstructed_weights)
    
    # Save the reconstructed model
    print(f"Saving reconstructed model to: {args.output_model}")
    onnx.save(reconstructed_model, args.output_model)
    
    # Get file size
    output_size = os.path.getsize(args.output_model)
    print(f"✓ Model saved successfully!")
    print(f"  Size: {output_size:,} bytes ({output_size/1024:.2f} KB)")
    
    # Verify against reference model if provided
    if args.verify and args.reference_model:
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        if not args.original_base_model:
            print("Warning: --original_base_model not provided.")
            print("Verification requires the original base model to map short names back to original names.")
            print("Skipping verification.")
        else:
            print(f"Loading original base model from: {args.original_base_model}")
            original_base_weights, _ = load_model_weights(args.original_base_model)
            print(f"✓ Loaded original base model with {len(original_base_weights)} layers")
            
            print(f"Loading reference model from: {args.reference_model}")
            reference_weights, _ = load_model_weights(args.reference_model)
            print(f"✓ Loaded reference model with {len(reference_weights)} layers")
            
            # Create mapping from short names to original names by sorting
            # Both base models should have the same layers, just different names
            original_base_sorted = sorted(original_base_weights.keys())
            short_names_sorted = sorted(base_weights.keys())
            
            if len(original_base_sorted) != len(short_names_sorted):
                print(f"Error: Mismatch in number of layers: {len(original_base_sorted)} vs {len(short_names_sorted)}")
            else:
                # Create reverse mapping: short name -> original name
                short_to_original = {short: orig for short, orig in zip(short_names_sorted, original_base_sorted)}
                
                print(f"\nComparing reconstructed vs reference model...")
                print(f"Using name mapping from {args.original_base_model}")
                
                max_error = 0.0
                mean_error = 0.0
                total_elements = 0
                layers_compared = 0
                layers_missing = 0
                
                for short_name, rec_weights in reconstructed_weights.items():
                    orig_name = short_to_original.get(short_name)
                    
                    if orig_name not in reference_weights:
                        layers_missing += 1
                        if args.verbose:
                            print(f"  Warning: Layer {short_name} ({orig_name}) not found in reference")
                        continue
                    
                    ref_weights = reference_weights[orig_name]
                    
                    if rec_weights.shape != ref_weights.shape:
                        if args.verbose:
                            print(f"  Warning: Shape mismatch for {short_name} ({orig_name}): {rec_weights.shape} vs {ref_weights.shape}")
                        continue
                    
                    error = np.abs(rec_weights - ref_weights)
                    layer_max_error = np.max(error)
                    layer_mean_error = np.mean(error)
                    
                    if layer_max_error > max_error:
                        max_error = layer_max_error
                    
                    mean_error += np.sum(error)
                    total_elements += rec_weights.size
                    layers_compared += 1
                    
                    if args.verbose:
                        print(f"  {short_name} ({orig_name}): max_error={layer_max_error:.6e}, mean_error={layer_mean_error:.6e}")
                
                mean_error /= total_elements if total_elements > 0 else 1
                
                print(f"\nReconstruction error vs reference:")
                print(f"  Layers compared: {layers_compared}/{len(reconstructed_weights)}")
                if layers_missing > 0:
                    print(f"  Layers missing in reference: {layers_missing}")
                print(f"  Max absolute error: {max_error:.6e}")
                print(f"  Mean absolute error: {mean_error:.6e}")
                
                # Calculate relative error
                ref_values = [reference_weights[short_to_original[k]] for k in reconstructed_weights.keys() 
                             if short_to_original.get(k) in reference_weights]
                if ref_values:
                    ref_mean_abs = np.mean([np.mean(np.abs(v)) for v in ref_values])
                    rel_error = mean_error / ref_mean_abs if ref_mean_abs > 0 else 0
                    print(f"  Relative error: {rel_error:.6e} ({rel_error*100:.4f}%)")
                    
                    if max_error < 1e-4:
                        print(f"\n✓ Verification PASSED: Reconstruction is very accurate (max error < 1e-4)")
                    elif max_error < 1e-2:
                        print(f"\n⚠ Verification WARNING: Some reconstruction error present (max error < 1e-2)")
                    else:
                        print(f"\n✗ Verification FAILED: Significant reconstruction error (max error >= 1e-2)")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print(f"\nReconstructed model saved to: {args.output_model}")
    
    # Print compression summary
    print("\nCompression summary:")
    print(f"  Original model size: {base_stats['total_size_bytes']:,} bytes (base)")
    print(f"  Bitstream size: {bitstream_size:,} bytes (update)")
    print(f"  Reconstructed model size: {output_size:,} bytes")
    if base_stats['total_size_bytes'] > 0:
        update_ratio = (bitstream_size / base_stats['total_size_bytes']) * 100
        print(f"  Update size: {update_ratio:.2f}% of base model")
    
    # Clean up temporary file if we downloaded from URL
    if temp_model_path and os.path.exists(temp_model_path):
        os.unlink(temp_model_path)


if __name__ == '__main__':
    main()
