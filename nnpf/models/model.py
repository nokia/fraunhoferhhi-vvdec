"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

import torch
import torch.nn as nn


class Multiplier(nn.Module):
    """
    Multiplier layer equivalent to TensorFlow's custom Multiplier.
    Applies element-wise multiplication with a learnable parameter.
    """

    def __init__(self, units: int):
        super(Multiplier, self).__init__()
        self.units = units
        # Initialize with ones, matching TensorFlow implementation
        self._multiplier = nn.Parameter(torch.ones(units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, H, W) - multiply along channel dimension
        # _multiplier shape: (C,) -> reshape to (1, C, 1, 1) for broadcasting
        return x * self._multiplier.view(1, -1, 1, 1)


class FilterWithMultipliersPyTorch(nn.Module):
    """
    PyTorch equivalent of the TensorFlow FilterWithMultipliers model.
    
    Architecture:
    - Input: (N, 7, 72, 72) in PyTorch format (NCHW)
    - Output: (N, 6, 64, 64)
    - 35 Conv2D layers with Multiplier layers
    - LeakyReLU activations (alpha=0.2) after certain layers
    - Residual connection: input[:, :6, 4:68, 4:68] added to output
    """

    def __init__(self):
        super(FilterWithMultipliersPyTorch, self).__init__()

        # Layer definitions matching TensorFlow model
        # Conv1: 7 -> 72 channels, 3x3 kernel
        self._conv1 = nn.Conv2d(7, 72, kernel_size=3, stride=1, padding=1)
        self._multiplier1 = Multiplier(72)
        self._leaky1 = nn.LeakyReLU(negative_slope=0.2)

        # Conv2: 72 -> 72 channels, 1x1 kernel
        self._conv2 = nn.Conv2d(72, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier2 = Multiplier(72)
        self._leaky2 = nn.LeakyReLU(negative_slope=0.2)

        # Conv3: 72 -> 24 channels, 1x1 kernel
        self._conv3 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier3 = Multiplier(24)

        # Conv4: 24 -> 24 channels, 3x3 kernel
        self._conv4 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier4 = Multiplier(24)

        # Conv5: 24 -> 72 channels, 1x1 kernel
        self._conv5 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier5 = Multiplier(72)
        self._leaky3 = nn.LeakyReLU(negative_slope=0.2)

        # Conv6: 72 -> 24 channels, 1x1 kernel
        self._conv6 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier6 = Multiplier(24)

        # Conv7: 24 -> 24 channels, 3x3 kernel
        self._conv7 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier7 = Multiplier(24)

        # Conv8: 24 -> 72 channels, 1x1 kernel
        self._conv8 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier8 = Multiplier(72)
        self._leaky4 = nn.LeakyReLU(negative_slope=0.2)

        # Conv9: 72 -> 24 channels, 1x1 kernel
        self._conv9 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier9 = Multiplier(24)

        # Conv10: 24 -> 24 channels, 3x3 kernel
        self._conv10 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier10 = Multiplier(24)

        # Conv11: 24 -> 72 channels, 1x1 kernel
        self._conv11 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier11 = Multiplier(72)
        self._leaky5 = nn.LeakyReLU(negative_slope=0.2)

        # Conv12: 72 -> 24 channels, 1x1 kernel
        self._conv12 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier12 = Multiplier(24)

        # Conv13: 24 -> 24 channels, 3x3 kernel
        self._conv13 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier13 = Multiplier(24)

        # Conv14: 24 -> 72 channels, 1x1 kernel
        self._conv14 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier14 = Multiplier(72)
        self._leaky6 = nn.LeakyReLU(negative_slope=0.2)

        # Conv15: 72 -> 24 channels, 1x1 kernel
        self._conv15 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier15 = Multiplier(24)

        # Conv16: 24 -> 24 channels, 3x3 kernel
        self._conv16 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier16 = Multiplier(24)

        # Conv17: 24 -> 72 channels, 1x1 kernel
        self._conv17 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier17 = Multiplier(72)
        self._leaky7 = nn.LeakyReLU(negative_slope=0.2)

        # Conv18: 72 -> 24 channels, 1x1 kernel
        self._conv18 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier18 = Multiplier(24)

        # Conv19: 24 -> 24 channels, 3x3 kernel
        self._conv19 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier19 = Multiplier(24)

        # Conv20: 24 -> 72 channels, 1x1 kernel
        self._conv20 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier20 = Multiplier(72)
        self._leaky8 = nn.LeakyReLU(negative_slope=0.2)

        # Conv21: 72 -> 24 channels, 1x1 kernel
        self._conv21 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier21 = Multiplier(24)

        # Conv22: 24 -> 24 channels, 3x3 kernel
        self._conv22 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier22 = Multiplier(24)

        # Conv23: 24 -> 72 channels, 1x1 kernel
        self._conv23 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier23 = Multiplier(72)
        self._leaky9 = nn.LeakyReLU(negative_slope=0.2)

        # Conv24: 72 -> 24 channels, 1x1 kernel
        self._conv24 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier24 = Multiplier(24)

        # Conv25: 24 -> 24 channels, 3x3 kernel
        self._conv25 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier25 = Multiplier(24)

        # Conv26: 24 -> 72 channels, 1x1 kernel
        self._conv26 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier26 = Multiplier(72)
        self._leaky10 = nn.LeakyReLU(negative_slope=0.2)

        # Conv27: 72 -> 24 channels, 1x1 kernel
        self._conv27 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier27 = Multiplier(24)

        # Conv28: 24 -> 24 channels, 3x3 kernel
        self._conv28 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier28 = Multiplier(24)

        # Conv29: 24 -> 72 channels, 1x1 kernel
        self._conv29 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier29 = Multiplier(72)
        self._leaky11 = nn.LeakyReLU(negative_slope=0.2)

        # Conv30: 72 -> 24 channels, 1x1 kernel
        self._conv30 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier30 = Multiplier(24)

        # Conv31: 24 -> 24 channels, 3x3 kernel
        self._conv31 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier31 = Multiplier(24)

        # Conv32: 24 -> 72 channels, 1x1 kernel
        self._conv32 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self._multiplier32 = Multiplier(72)
        self._leaky12 = nn.LeakyReLU(negative_slope=0.2)

        # Conv33: 72 -> 24 channels, 1x1 kernel
        self._conv33 = nn.Conv2d(72, 24, kernel_size=1, stride=1, padding=0)
        self._multiplier33 = Multiplier(24)

        # Conv34: 24 -> 24 channels, 3x3 kernel
        self._conv34 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self._multiplier34 = Multiplier(24)

        # Conv35: 24 -> 6 channels, 3x3 kernel (final output)
        self._conv35 = nn.Conv2d(24, 6, kernel_size=3, stride=1, padding=1)
        self._multiplier35 = Multiplier(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 10, 72, 72) in NCHW format
            
        Returns:
            Output tensor of shape (N, 6, 64, 64)
        """
        # Store input for residual connection
        input_slice = x[:, :6, 4:68, 4:68]  # (N, 6, 64, 64)

        y = self._conv1(x)
        y = self._multiplier1(y)
        y = self._leaky1(y)

        y = self._conv2(y)
        y = self._multiplier2(y)
        y = self._leaky2(y)

        y = self._conv3(y)
        y = self._multiplier3(y)

        y = self._conv4(y)
        y = self._multiplier4(y)

        y = self._conv5(y)
        y = self._multiplier5(y)
        y = self._leaky3(y)

        y = self._conv6(y)
        y = self._multiplier6(y)

        y = self._conv7(y)
        y = self._multiplier7(y)

        y = self._conv8(y)
        y = self._multiplier8(y)
        y = self._leaky4(y)

        y = self._conv9(y)
        y = self._multiplier9(y)

        y = self._conv10(y)
        y = self._multiplier10(y)

        y = self._conv11(y)
        y = self._multiplier11(y)
        y = self._leaky5(y)

        y = self._conv12(y)
        y = self._multiplier12(y)

        y = self._conv13(y)
        y = self._multiplier13(y)

        y = self._conv14(y)
        y = self._multiplier14(y)
        y = self._leaky6(y)

        y = self._conv15(y)
        y = self._multiplier15(y)

        y = self._conv16(y)
        y = self._multiplier16(y)

        y = self._conv17(y)
        y = self._multiplier17(y)
        y = self._leaky7(y)

        y = self._conv18(y)
        y = self._multiplier18(y)

        y = self._conv19(y)
        y = self._multiplier19(y)

        y = self._conv20(y)
        y = self._multiplier20(y)
        y = self._leaky8(y)

        y = self._conv21(y)
        y = self._multiplier21(y)

        y = self._conv22(y)
        y = self._multiplier22(y)

        y = self._conv23(y)
        y = self._multiplier23(y)
        y = self._leaky9(y)

        y = self._conv24(y)
        y = self._multiplier24(y)

        y = self._conv25(y)
        y = self._multiplier25(y)

        y = self._conv26(y)
        y = self._multiplier26(y)
        y = self._leaky10(y)

        y = self._conv27(y)
        y = self._multiplier27(y)

        y = self._conv28(y)
        y = self._multiplier28(y)

        y = self._conv29(y)
        y = self._multiplier29(y)
        y = self._leaky11(y)

        y = self._conv30(y)
        y = self._multiplier30(y)

        y = self._conv31(y)
        y = self._multiplier31(y)

        y = self._conv32(y)
        y = self._multiplier32(y)
        y = self._leaky12(y)

        y = self._conv33(y)
        y = self._multiplier33(y)

        y = self._conv34(y)
        y = self._multiplier34(y)

        y = self._conv35(y)
        y = self._multiplier35(y)

        # Slice and add residual connection
        y = y[:, :, 4:68, 4:68]  # (N, 6, 64, 64)
        y = y + input_slice

        return y


__all__ = ['FilterWithMultipliersPyTorch', 'Multiplier']

