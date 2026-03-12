"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

import torch
import torch.nn as nn
import numpy as np
import onnx


class BlueFilter(nn.Module):
    """
    Blue filter model with blue color effect, same input/output as FilterWithMultipliersPyTorch.
    
    Architecture:
    - Input: (N, 7, 72, 72) in PyTorch format (NCHW)
      - Channels 0-3: Y (luma) - 4 channels
      - Channel 4: U (chroma)
      - Channel 5: V (chroma)
      - Channel 6: Strength/effect intensity (0-1 normalized)
    - Output: (N, 6, 64, 64) - 4 Y channels, 1 U, 1 V
    - Applies blue tone by setting chroma values
    - Uses the 7th "strength" channel to control effect intensity
    """

    def __init__(self):
        super(BlueFilter, self).__init__()
        
    def apply_blue_tone_yuv(self, u, v, strength):
        """
        Apply blue tone directly in YUV color space.
        Blue tones: shift U and V from neutral (0.5).
        
        Args:
            u, v: Chroma tensors of shape (N, 1, H, W) in range [0, 1] where 0.5 is neutral
            strength: Effect strength tensor (scalar or tensor)
            
        Returns:
            u_blue, v_blue: Blue-toned chroma values in range [0, 1]
        """
        # NNPF standard: chroma range is 0..1 where 0.5 is neutral (zero chroma)
        # For blue tone:
        # - U > 0.5: more blue
        # - V < 0.5: less red (moving toward cyan/blue)
        
        # Target blue chroma values
        u_blue_target = 0.70  # Above 0.5 for more blue
        v_blue_target = 0.35  # Below 0.5 for less red
        
        # Blend toward blue chroma based on strength
        blue_strength = 1 
        u_blue = (1.0 - blue_strength) * u + blue_strength * u_blue_target
        v_blue = (1.0 - blue_strength) * v + blue_strength * v_blue_target
        
        return u_blue, v_blue
    
    def apply_tone_mapping(self, y_channel, strength):
        """
        Apply subtle tone mapping for effect.
        - Keep luma mostly unchanged for natural look
        
        Args:
            y_channel: Luma channel tensor (N, 1, H, W)
            strength: Effect strength (N, 1, H, W) in [0, 1]
            
        Returns:
            Tone-mapped luma channel
        """
        # Normalize to [0, 1] range for tone mapping (assuming input is roughly in this range)
        y_norm = torch.clamp(y_channel, 0.0, 1.0)
        
        # Minimal adjustments to luma for natural blue tint
        return y_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with blue filter.
        
        Args:
            x: Input tensor of shape (N, 7, 72, 72) in NCHW format
            
        Returns:
            Output tensor of shape (N, 6, 64, 64)
        """
        # Extract the input slice for residual connection
        input_slice = x[:, :6, 4:68, 4:68]  # (N, 6, 64, 64)
        
        strength = 0.5
        
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
        
        # Apply blue tone directly in YUV color space
        u_blue, v_blue = self.apply_blue_tone_yuv(u, v, strength)
                
        # Concatenate output channels (4 Y, 1 U, 1 V)
        output = torch.cat([y1, y2, y3, y4, u_blue, v_blue], dim=1)
        
        # Crop to output size (64x64)
        output = output[:, :, 4:68, 4:68]
        
        return output


def export_to_onnx(model, output_path='blue.onnx', batch_size=1, opset_version=18):
    """
    Export the model to ONNX format with 72x72 chroma resolution input 
    and 64x64 resolution output.
    
    Args:
        model: The BlueFilter model instance
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
    
    # Load the model and save it as a self-contained file (no external data)
    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    
    print(f"Model exported to {output_path}")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {output_shape}")


def main():
    """
    Main function to create and export the blue filter model.
    """
    # Create model
    model = BlueFilter()
    
    # Test with dummy data
    print("Testing model...")
    test_input = torch.randn(1, 7, 72, 72)
    # Set a gradient of strength values for testing
    test_input[:, 6, :, :] = torch.linspace(0, 1, 72).unsqueeze(0).unsqueeze(-1).expand(1, 72, 72)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Strength channel range: [{test_input[:, 6, :, :].min():.3f}, {test_input[:, 6, :, :].max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    export_to_onnx(model, 'blue.onnx', batch_size=1, opset_version=18)
    
    print("\nDone! Blue filter created.")


if __name__ == '__main__':
    main()


__all__ = ['BlueFilter', 'export_to_onnx']
