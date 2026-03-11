import torch
import torch.nn as nn
import numpy as np


class QPColorBlender(nn.Module):
    """
    QP-aware color blending model with the same input/output as FilterWithMultipliersPyTorch.
    
    Architecture:
    - Input: (N, 7, 72, 72) in PyTorch format (NCHW)
      - Channels 0-3: Y (luma) - 4 channels
      - Channel 4: U (chroma)
      - Channel 5: V (chroma)
      - Channel 6: Strength/QP value (0-1 normalized)
    - Output: (N, 6, 64, 64) - 4 Y channels, 1 U, 1 V
    - Uses the 7th "strength" channel to directly modulate chroma
    - Simple chroma shift based on strength for colorful effect
    """

    def __init__(self):
        super(QPColorBlender, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with direct chroma modulation.
        
        Args:
            x: Input tensor of shape (N, 7, 72, 72) in NCHW format
            
        Returns:
            Output tensor of shape (N, 6, 64, 64)
        """
        # Extract the input slice for residual connection
        input_slice = x[:, :6, 4:68, 4:68]  # (N, 6, 64, 64)
        
        # Extract strength channel (7th channel, index 6)
        strength = x[:, 6:7, :, :]  # (N, 1, 72, 72)
        
        # Clamp strength to [0, 1] range
        strength = torch.clamp(strength, 0.0, 1.0)
        
        # Extract Y, U, V channels from input
        # Channels 0-3: Y (luma) - 4 channels
        # Channel 4: U (chroma)
        # Channel 5: V (chroma)
        y1 = x[:, 0:1, :, :]
        y2 = x[:, 1:2, :, :]
        y3 = x[:, 2:3, :, :]
        y4 = x[:, 3:4, :, :]
        u = x[:, 4:5, :, :]
        v = x[:, 5:6, :, :]
        
        # Apply direct chroma modulation based on strength
        # Scale down for a subtler effect (0.3x instead of 1.0x)
        scale = 0.5
        u_delta = (0.25 - torch.abs(strength - 0.5)) * scale
        v_delta = (-0.5 + strength) * scale
        
        # Apply modulation and clamp to valid range [0, 1]
        u_output = torch.clamp(0.5 + u_delta, 0.0, 1.0)
        v_output = torch.clamp(0.5 + v_delta, 0.0, 1.0)
        
        # Concatenate output channels (Y unchanged, U/V modulated)
        output = torch.cat([y1, y2, y3, y4, u_output, v_output], dim=1)
        
        # Crop to output size (64x64)
        output = output[:, :, 4:68, 4:68]
        
        return output


def export_to_onnx(model, output_path='qp.onnx', batch_size=1, opset_version=18):
    """
    Export the model to ONNX format with 72x72 chroma resolution input 
    and 64x64 resolution output.
    
    Args:
        model: The QPColorBlender model instance
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
    
    # Export to ONNX using dynamo_export (newer PyTorch method)
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
    Main function to create and export the QP color blending model.
    """
    # Create model
    model = QPColorBlender()
    
    # Test with dummy data
    print("Testing model...")
    test_input = torch.rand(1, 7, 72, 72)  # Uniform distribution [0, 1]
    # Set a specific strength pattern for testing
    test_input[:, 6, :, :] = torch.linspace(0, 1, 72).unsqueeze(0).unsqueeze(-1).expand(1, 72, 72)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Strength channel range: [{test_input[:, 6, :, :].min():.3f}, {test_input[:, 6, :, :].max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    export_to_onnx(model, 'qp.onnx', batch_size=1, opset_version=18)
    
    print("\nDone!")


if __name__ == '__main__':
    main()


__all__ = ['QPColorBlender', 'export_to_onnx']
