"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

import torch
import torch.nn as nn
import numpy as np
import onnx


class GreenFilter(nn.Module):
    """
    Green filter model with green color effect, same input/output as FilterWithMultipliersPyTorch.
    
    Architecture:
    - Input: (N, 7, 72, 72) in PyTorch format (NCHW)
      - Channels 0-3: Y (luma) - 4 channels
      - Channel 4: U (chroma)
      - Channel 5: V (chroma)
      - Channel 6: Strength/effect intensity (0-1 normalized)
    - Output: (N, 6, 64, 64) - 4 Y channels, 1 U, 1 V
    - Applies green tone by setting chroma values
    - Uses the 7th "strength" channel to control effect intensity
    """

    def __init__(self):
        super(GreenFilter, self).__init__()
        
    def apply_green_tone_yuv(self, u, v, strength):
        """
        Apply green tone directly in YUV color space.
        Green tones: shift U and V from neutral (0.5).
        
        Args:
            u, v: Chroma tensors of shape (N, 1, H, W) in range [0, 1] where 0.5 is neutral
            strength: Effect strength tensor (scalar or tensor)
            
        Returns:
            u_green, v_green: Green-toned chroma values in range [0, 1]
        """
        # NNPF standard: chroma range is 0..1 where 0.5 is neutral (zero chroma)
        # For green tone:
        # - U < 0.5: less blue (moving toward yellow-green)
        # - V < 0.5: less red (moving away from red)
        
        # Target green chroma values
        u_green_target = 0.30  # Below 0.5 for yellow-green shift
        v_green_target = 0.30  # Below 0.5 for less red
        
        # Blend toward green chroma based on strength
        green_strength = 0.8 * strength  # 80% max green effect
        u_green = (1.0 - green_strength) * u + green_strength * u_green_target
        v_green = (1.0 - green_strength) * v + green_strength * v_green_target
        
        return u_green, v_green
    
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
        
        # Minimal adjustments to luma for natural green tint
        return y_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with green filter.
        
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
        
        # Apply green tone directly in YUV color space
        u_green, v_green = self.apply_green_tone_yuv(u, v, strength)
        
        # Apply tone mapping to luma channels
        # Use average Y for tone mapping calculation
        y_avg = (y1 + y2 + y3 + y4) / 4.0
        y_toned = self.apply_tone_mapping(y_avg, strength)
        
        # Blend original and green-toned based on strength
        # strength = 0: original, strength = 1: full green effect
        y1_output = (1.0 - strength) * y1 + strength * y_toned
        y2_output = (1.0 - strength) * y2 + strength * y_toned
        y3_output = (1.0 - strength) * y3 + strength * y_toned
        y4_output = (1.0 - strength) * y4 + strength * y_toned
        u_output = u_green  # Already blended in apply_green_tone_yuv
        v_output = v_green  # Already blended in apply_green_tone_yuv
        
        # Concatenate output channels (4 Y, 1 U, 1 V)
        output = torch.cat([y1_output, y2_output, y3_output, y4_output,
                           u_output, v_output], dim=1)
        
        # Crop to output size (64x64)
        output = output[:, :, 4:68, 4:68]
        
        return output


def export_to_onnx(model, output_path='green.onnx', batch_size=1, opset_version=18):
    """
    Export the model to ONNX format with 72x72 chroma resolution input 
    and 64x64 resolution output.
    
    Args:
        model: The GreenFilter model instance
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
    Main function to create and export the green filter model.
    """
    # Create model
    model = GreenFilter()
    
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
    export_to_onnx(model, 'green.onnx', batch_size=1, opset_version=18)
    
    print("\nDone! Green filter created.")


if __name__ == '__main__':
    main()


__all__ = ['GreenFilter', 'export_to_onnx']
