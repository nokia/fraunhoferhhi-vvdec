import torch
import torch.nn as nn


class PassthroughFilter(nn.Module):
    """
    Passthrough filter model - simply crops 72x72 to 64x64 without any processing.
    Same input/output format as FilterWithMultipliersPyTorch.
    
    Architecture:
    - Input: (N, 7, 72, 72) in PyTorch format (NCHW)
      - Channels 0-3: Y (luma) - 4 channels
      - Channel 4: U (chroma)
      - Channel 5: V (chroma)
      - Channel 6: Strength (ignored in passthrough)
    - Output: (N, 6, 64, 64) - 4 Y channels, 1 U, 1 V
    - Simply extracts the center 64x64 region and adds residual connection
    """

    def __init__(self):
        super(PassthroughFilter, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - passthrough with cropping.
        
        Args:
            x: Input tensor of shape (N, 7, 72, 72) in NCHW format
            
        Returns:
            Output tensor of shape (N, 6, 64, 64)
        """
        # Extract first 6 channels and crop to 64x64 center region
        input_slice = x[:, :6, 4:68, 4:68]  # (N, 6, 64, 64)
        
        return input_slice


def export_to_onnx(model, output_path='passthrough.onnx', batch_size=1, opset_version=18):
    """
    Export the model to ONNX format with 72x72 input and 64x64 output.
    
    Args:
        model: The PassthroughFilter model instance
        output_path: Path to save the ONNX file
        batch_size: Batch size for the model
        opset_version: ONNX opset version to use
    """
    model.eval()
    
    # Create dummy input: (batch_size, 7, 72, 72)
    dummy_input = torch.randn(batch_size, 7, 72, 72, requires_grad=False)
    
    # Get actual output shape by running forward pass
    with torch.no_grad():
        dummy_output = model(dummy_input)
        output_shape = tuple(dummy_output.shape)
    
    # Export to ONNX
    try:
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
    except Exception as e:
        print(f"Standard export failed: {e}")
        print("Trying with dynamo=False...")
        # Try with older exporter
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,
            verbose=False,
            dynamo=False
        )
    
    print(f"Model exported to {output_path}")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {output_shape}")


def main():
    """
    Main function to create and export the passthrough filter model.
    """
    # Create model
    model = PassthroughFilter()
    
    # Test with dummy data
    print("Testing passthrough model...")
    test_input = torch.randn(1, 7, 72, 72)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Input center region (64x64) range: [{test_input[:, :6, 4:68, 4:68].min():.3f}, {test_input[:, :6, 4:68, 4:68].max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verify passthrough is working (output should equal input center region)
    input_center = test_input[:, :6, 4:68, 4:68]
    difference = torch.abs(output - input_center).max()
    print(f"Max difference between input and output: {difference:.6f} (should be ~0 for passthrough)")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    export_to_onnx(model, 'passthrough.onnx', batch_size=1, opset_version=18)
    
    print("\nDone! Passthrough filter created - simply crops 72x72 to 64x64.")


if __name__ == '__main__':
    main()


__all__ = ['PassthroughFilter', 'export_to_onnx']
