#!/usr/bin/env python3

"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

"""
Convert PyTorch model checkpoint to ONNX format for inference.

This script loads a trained FilterWithMultipliersPyTorch model from a .pt checkpoint
and exports it to ONNX format for deployment and inference.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.onnx

try:
    import onnx
    import onnx.checker
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx package not found. Validation will be skipped.")
    print("Install with: pip install onnx")

from model import FilterWithMultipliersPyTorch


def load_model(checkpoint_path, device='cpu'):
    """
    Load the PyTorch model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False needed for full checkpoint dicts)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both checkpoint dict and direct state dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint with metadata:")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'y_psnr_gain' in checkpoint:
            print(f"  Y-PSNR Gain: {checkpoint['y_psnr_gain']:.4f} dB")
    else:
        state_dict = checkpoint
        print("Loaded direct state dict")
    
    # Instantiate model and load weights
    model = FilterWithMultipliersPyTorch()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def export_to_onnx(model, output_path, opset_version=18):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model in eval mode
        output_path: Path for output .onnx file
        opset_version: ONNX opset version to use
    """
    print(f"\nExporting to ONNX format...")
    print(f"Output path: {output_path}")
    print(f"Opset version: {opset_version}")
    
    # Create dummy input with correct shape: (batch, channels, height, width)
    # Input: 7 channels (4 luma interleaved + 2 chroma + 1 qp), 72x72 padded block
    dummy_input = torch.randn(1, 7, 72, 72, requires_grad=False)
    
    # Get actual output shape by running forward pass
    with torch.no_grad():
        dummy_output = model(dummy_input)
        output_shape = tuple(dummy_output.shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # Fixed input/output shapes for optimal performance
        verbose=False
    )
    
    # Convert external data to internal storage (single file)
    if ONNX_AVAILABLE:
        try:
            onnx_model = onnx.load(output_path)
            # Convert all external tensors to internal storage
            onnx.save(onnx_model, output_path, save_as_external_data=False)
            # Remove the external data file if it was created
            external_data_path = str(output_path) + '.data'
            if Path(external_data_path).exists():
                Path(external_data_path).unlink()
                print(f"  Consolidated into single file (removed {Path(external_data_path).name})")
        except Exception as e:
            print(f"  Warning: Could not consolidate model: {e}")
    
    print(f"✓ Export successful!")
    print(f"  Input shape: {tuple(dummy_input.shape)}")
    print(f"  Output shape: {output_shape}")


def validate_onnx_model(onnx_path):
    """
    Validate the exported ONNX model.
    
    Args:
        onnx_path: Path to the .onnx file to validate
    
    Returns:
        True if validation passes, False otherwise
    """
    if not ONNX_AVAILABLE:
        print("\nSkipping validation (onnx package not installed)")
        return False
    
    print(f"\nValidating ONNX model...")
    
    try:
        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print("✓ ONNX model is valid!")
        
        # Print model info
        graph = onnx_model.graph
        print(f"  Model name: {graph.name if graph.name else 'FilterWithMultipliersPyTorch'}")
        print(f"  Inputs: {len(graph.input)}")
        print(f"  Outputs: {len(graph.output)}")
        print(f"  Nodes: {len(graph.node)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model checkpoint to ONNX format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        default='model_overfitted_best.pt',
        help='Input PyTorch checkpoint file (.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model.onnx',
        help='Output ONNX model file (.onnx)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=18,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for model loading'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip ONNX model validation'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'
    
    try:
        # Load model
        model = load_model(args.input, device=args.device)
        
        # Export to ONNX
        export_to_onnx(model, args.output, opset_version=args.opset)
        
        # Validate
        if not args.no_validate:
            validate_onnx_model(args.output)
        
        print(f"\n✓ Conversion complete!")
        print(f"ONNX model saved to: {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
