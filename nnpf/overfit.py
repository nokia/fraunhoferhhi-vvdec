
"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

"""
Training script to overfit multiplier layers of the PyTorch model.
Freezes all convolutional layers and trains only the multiplier parameters.
"""

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataset import Dataset
from model import FilterWithMultipliersPyTorch


def de_interleave_luma(blocks):
    """
    De-interleave four image partitions back into a single image
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
    Chroma channels are not interleaved (already at half resolution)
    Args:
        chroma: torch tensor (N, 1, H, W) - single chroma channel
    Returns:
        chroma: torch tensor (1, H, W) - same as input, just reshaped
    """
    return chroma[0, 0, :, :]


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images (assumed to be in [0, 1] range)
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


def compute_mse_loss(img1, img2):
    """
    Compute MSE loss between two images
    Args:
        img1, img2: torch tensors
    Returns:
        MSE loss
    """
    return torch.mean((img1 - img2) ** 2)


def freeze_conv_layers(model):
    """
    Freeze all convolutional layers and enable training for multiplier parameters only.
    
    Args:
        model: FilterWithMultipliersPyTorch model instance
    """
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        if '_multiplier' in name and param.requires_grad:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"Trainable parameters (multipliers): {trainable_params:,}")
    print(f"Frozen parameters (conv layers): {frozen_params:,}")
    print(f"Total parameters: {trainable_params + frozen_params:,}")
    
    return trainable_params


def process_frame_forward(dataset, model, block_size, device):
    """
    Process frame data through the model and reconstruct the filtered images (Y, U, V).
    
    Args:
        dataset: Dataset instance that iterates over patches
        model: PyTorch model
        block_size: Block size without padding
        device: torch device
        
    Returns:
        (filtered_y, filtered_u, filtered_v), (orig_y, orig_u, orig_v), (reco_y, reco_u, reco_v), metadata
    """
    # Get metadata from first patch
    first_patch_data = dataset[0]
    height = first_patch_data['height']
    width = first_patch_data['width']
    num_v_blocks = first_patch_data['num_v_blocks']
    num_h_blocks = first_patch_data['num_h_blocks']
    poc = first_patch_data['poc']
    frame_type = first_patch_data['frame_type']
    frame_qp = first_patch_data['frame_qp']
    
    # Load full frame images (Y, U, V) from dataset
    frame_number = first_patch_data['frame_number']
    orig_yuv = dataset._read_yuv_frame(
        dataset.orig_yuv_path, poc + dataset.frames_to_skip,
        dataset.width, dataset.height, dataset.bit_depth
    )
    reco_yuv = dataset._read_yuv_frame(
        dataset.reco_yuv_path, poc,
        dataset.width, dataset.height, dataset.bit_depth
    )
    
    # Convert to tensors (arrays of 3 components: Y, U, V)
    orig = [torch.from_numpy(orig_yuv[i]).to(device) for i in range(3)]
    reco = [torch.from_numpy(reco_yuv[i]).to(device) for i in range(3)]
    
    # Process all patches through the model
    output_patches_list = []
    num_patches = len(dataset)
    
    for i in range(num_patches):
        patch_data = dataset[i]
        input_patch = patch_data['input'].unsqueeze(0)  # Add batch dimension (1, 10, H, W)
        output_patch = model(input_patch)  # (1, 6, 64, 64)
        output_patches_list.append(output_patch[0])  # Remove batch dimension
    
    # Stack all patches: (num_patches, 6, 64, 64)
    output_patches = torch.stack(output_patches_list, dim=0)
    
    # Extract luma (4 channels) and chroma (2 channels) from output
    output_patches_yuv = [
        output_patches[:, :4, :, :],   # Y: (num_patches, 4, block_size, block_size)
        output_patches[:, 4:5, :, :],  # U: (num_patches, 1, block_size, block_size)
        output_patches[:, 5:6, :, :]   # V: (num_patches, 1, block_size, block_size)
    ]
    
    total_expected_patches = num_v_blocks * num_h_blocks
    num_patches_actual = output_patches_yuv[0].shape[0]
    
    if num_patches_actual != total_expected_patches:
        # Pad with zeros if needed
        if num_patches_actual < total_expected_patches:
            padding_needed = total_expected_patches - num_patches_actual
            padding_y = torch.zeros(padding_needed, 4, block_size, block_size, device=device)
            padding_uv = torch.zeros(padding_needed, 1, block_size, block_size, device=device)
            output_patches_yuv[0] = torch.cat([output_patches_yuv[0], padding_y], dim=0)
            output_patches_yuv[1] = torch.cat([output_patches_yuv[1], padding_uv], dim=0)
            output_patches_yuv[2] = torch.cat([output_patches_yuv[2], padding_uv], dim=0)
    
    # Process luma: Reshape and de-interleave
    output_patches_grid_y = output_patches_yuv[0][:total_expected_patches].view(
        num_v_blocks, num_h_blocks, 4, block_size, block_size
    )
    output_patches_grid_y = output_patches_grid_y.permute(2, 0, 3, 1, 4)
    output_patches_grid_y = output_patches_grid_y.reshape(
        4, num_v_blocks * block_size, num_h_blocks * block_size
    )
    filtered_y = de_interleave_luma(output_patches_grid_y.unsqueeze(0))
    filtered_y = filtered_y[0, 0, :height, :width]
    
    # Process chroma: Reshape (for U and V)
    filtered = [filtered_y]  # Start with Y
    for i in [1, 2]:  # U and V
        output_patches_grid = output_patches_yuv[i][:total_expected_patches].view(
            num_v_blocks, num_h_blocks, 1, block_size, block_size
        )
        output_patches_grid = output_patches_grid.permute(2, 0, 3, 1, 4)
        output_patches_grid = output_patches_grid.reshape(
            1, num_v_blocks * block_size, num_h_blocks * block_size
        )
        filtered_component = de_interleave_chroma(output_patches_grid.unsqueeze(0))
        filtered_component = filtered_component[:height // 2, :width // 2]
        filtered.append(filtered_component)
    
    metadata = {
        'poc': poc,
        'frame_type': frame_type,
        'frame_qp': frame_qp,
        'height': height,
        'width': width
    }
    
    return filtered, orig, reco, metadata


@click.command()
@click.option(
    "--model_path",
    default="model3.pt",
    type=click.Path(exists=True),
    help="Path to the initial PyTorch model weights file",
)
@click.option(
    "--output_path",
    default="model_overfitted.pt",
    type=str,
    help="Path to save the trained model",
)
@click.option(
    "--input_yuv",
    default="input.yuv",
    type=click.Path(exists=True),
    help="Path to input (original) YUV file",
)
@click.option(
    "--recon_yuv",
    default="enc_rec.yuv",
    type=click.Path(exists=True),
    help="Path to reconstructed YUV file",
)
@click.option(
    "--log_enc",
    default="log_enc.txt",
    type=click.Path(exists=True),
    help="Path to encoder log file (for deriving frame QP from POC)",
)
@click.option(
    "--width",
    default=1920,
    type=int,
    help="Frame width in pixels",
)
@click.option(
    "--height",
    default=1080,
    type=int,
    help="Frame height in pixels",
)
@click.option(
    "--bit_depth",
    default=10,
    type=int,
    help="Bit depth of images",
)
@click.option(
    "--block_size",
    default=64,
    type=int,
    help="Block size (without padding)",
)
@click.option(
    "--pad_size",
    default=8,
    type=int,
    help="Padding size",
)
@click.option(
    "--num_frames",
    default=None,
    type=int,
    help="Number of frames to train on (None for all)",
)
@click.option(
    "--epochs",
    default=20,
    type=int,
    help="Number of training epochs",
)
@click.option(
    "--learning_rate",
    default=0.001,
    type=float,
    help="Learning rate for optimizer",
)
@click.option(
    "--save_interval",
    default=10,
    type=int,
    help="Save model checkpoint every N epochs",
)
@click.option(
    "--max_patches",
    default=None,
    type=int,
    help="Maximum number of patches to use for training (randomly selected). None for all patches.",
)
def train_multipliers(
    model_path,
    output_path,
    input_yuv,
    recon_yuv,
    log_enc,
    width,
    height,
    bit_depth,
    block_size,
    pad_size,
    num_frames,
    epochs,
    learning_rate,
    save_interval,
    max_patches,
):
    """
    Train only the multiplier layers of the model on a video sequence
    """
    print("=" * 80)
    print("MULTIPLIER LAYER TRAINING (OVERFITTING)")
    print("=" * 80)
    print(f"Initial model: {model_path}")
    print(f"Output model: {output_path}")
    print(f"Input YUV: {input_yuv}")
    print(f"Recon YUV: {recon_yuv}")
    print(f"Log file: {log_enc}")
    print(f"Resolution: {width}x{height}")
    print(f"Block size: {block_size}, Pad size: {pad_size}, Bit depth: {bit_depth}")
    print(f"Training epochs: {epochs}, Learning rate: {learning_rate}")
    if max_patches is not None:
        print(f"Max patches: {max_patches} (randomly selected)")
    print("-" * 80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        # Instantiate the model
        model = FilterWithMultipliersPyTorch()
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Freeze convolutional layers
    print("\nFreezing convolutional layers...")
    num_trainable = freeze_conv_layers(model)
    
    if num_trainable == 0:
        print("ERROR: No trainable parameters found!")
        return
    
    # Setup optimizer (only for trainable parameters)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Determine frames to process by creating a temporary Dataset to get frame count
    try:
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
            device=device
        )
        total_frames = len(temp_dataset.frames_info)
        print(f"\nTotal frames available: {total_frames}")
        
        if num_frames is None:
            frames_to_process = total_frames
        else:
            frames_to_process = min(num_frames, total_frames)
        
        print(f"Will train on {frames_to_process} frames")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load all frames in a single dataset
    print("\nLoading dataset with all frames...")
    frame_list = list(range(frames_to_process))
    try:
        dataset = Dataset(
            input_yuv=input_yuv,
            recon_yuv=recon_yuv,
            log_enc=log_enc,
            width=width,
            height=height,
            block_size=block_size,
            pad_size=pad_size,
            bit_depth=bit_depth,
            frames=frame_list,
            device=device
        )
        print(f"Successfully loaded dataset with {len(dataset)} patches from {frames_to_process} frames")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Select random subset of patches if max_patches is specified
    if max_patches is not None and max_patches < len(dataset):
        print(f"\nRandomly selecting {max_patches} patches out of {len(dataset)} total patches...")
        np.random.seed(42)  # For reproducibility
        selected_patch_indices = np.random.choice(len(dataset), size=max_patches, replace=False)
        selected_patch_indices = sorted(selected_patch_indices.tolist())
        print(f"Selected patch indices: {selected_patch_indices[:10]}{'...' if len(selected_patch_indices) > 10 else ''}")
    else:
        selected_patch_indices = list(range(len(dataset)))
        if max_patches is not None:
            print(f"\nNote: max_patches ({max_patches}) >= total patches ({len(dataset)}), using all patches")
    
    print(f"Training will use {len(selected_patch_indices)} patches")
    
    # Compute initial PSNR per frame
    print("\n" + "=" * 80)
    print("INITIAL EVALUATION (Before Training)")
    print("=" * 80)
    model.eval()
    initial_psnr_list = [[], [], []]  # Y, U, V
    baseline_psnr_list = [[], [], []]  # Y, U, V
    
    with torch.no_grad():
        # Get unique frames from dataset metadata
        unique_frames = sorted(set(meta['frame_number'] for meta in dataset.frame_metadata))
        for frame_num in unique_frames:
            # Create a temporary single-frame dataset for evaluation
            temp_dataset = Dataset(
                input_yuv=input_yuv,
                recon_yuv=recon_yuv,
                log_enc=log_enc,
                width=width,
                height=height,
                block_size=block_size,
                pad_size=pad_size,
                bit_depth=bit_depth,
                frames=[frame_num],
                device=device
            )
            filtered, orig, reco, metadata = process_frame_forward(
                temp_dataset, model, block_size, device
            )
            
            # Compute PSNR for each component (Y, U, V)
            for i in range(3):
                psnr_filtered = compute_psnr(orig[i], filtered[i])
                psnr_baseline = compute_psnr(orig[i], reco[i])
                initial_psnr_list[i].append(psnr_filtered)
                baseline_psnr_list[i].append(psnr_baseline)
    
    avg_baseline = np.array([np.mean(baseline_psnr_list[i]) for i in range(3)])
    avg_initial = np.array([np.mean(initial_psnr_list[i]) for i in range(3)])
    avg_initial_gain = avg_initial - avg_baseline
    
    print(f"Average baseline PSNR (no filter): [{avg_baseline[0]:6.3f} {avg_baseline[1]:6.3f} {avg_baseline[2]:6.3f}] dB")
    print(f"Average initial PSNR (with filter): [{avg_initial[0]:6.3f} {avg_initial[1]:6.3f} {avg_initial[2]:6.3f}] dB")
    print(f"Average initial gain:                [{avg_initial_gain[0]:+6.3f} {avg_initial_gain[1]:+6.3f} {avg_initial_gain[2]:+6.3f}] dB")
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_psnr_y = avg_initial[0]  # Track best Y PSNR
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Train on selected patches only
        for patch_idx in selected_patch_indices:
            optimizer.zero_grad()
            
            # Get patch data
            patch_data = dataset[patch_idx]
            input_patch = patch_data['input'].unsqueeze(0)  # Add batch dimension (1, 7, H, W)
            
            # Get frame information
            frame_number = patch_data['frame_number']
            poc = patch_data['poc']
            
            # Read the original frame for ground truth
            orig_yuv = dataset._read_yuv_frame(
                dataset.orig_yuv_path, poc + dataset.frames_to_skip,
                dataset.width, dataset.height, dataset.bit_depth
            )
            
            # Convert to tensors (array of 3 components: Y, U, V)
            orig_full = [torch.from_numpy(orig_yuv[i]).to(device) for i in range(3)]
            
            # Forward pass
            output_patch = model(input_patch)  # (1, 6, 64, 64)
            
            # De-interleave the output to get filtered patch
            output_yuv = [
                output_patch[0, :4, :, :],   # Y: (4, 64, 64)
                output_patch[0, 4:5, :, :],  # U: (1, 64, 64)
                output_patch[0, 5:6, :, :]   # V: (1, 64, 64)
            ]
            
            # De-interleave luma: (4, H, W) -> (1, 1, 2*H, 2*W)
            filtered_patch = [
                de_interleave_luma(output_yuv[0].unsqueeze(0))[0, 0, :, :],  # Y: (128, 128) max
                output_yuv[1][0, :, :],  # U: (64, 64)
                output_yuv[2][0, :, :]   # V: (64, 64)
            ]
            
            # For patch-level training, we need to determine which region of the original
            # image this patch corresponds to. Since we don't have explicit patch coordinates,
            # we'll use a simpler approach: compute loss on the entire output patch size
            # against the center region of the original image (simplified assumption)
            
            # Alternative: Extract corresponding region based on patch index
            # Calculate patch position from index within the frame
            patch_idx_in_frame = patch_data['patch_idx_in_frame']
            num_h_blocks = patch_data['num_h_blocks']
            patch_v_idx = patch_idx_in_frame // num_h_blocks
            patch_h_idx = patch_idx_in_frame % num_h_blocks
            
            # Calculate pixel coordinates
            y_start = patch_v_idx * block_size * 2
            y_end = min(y_start + block_size * 2, dataset.height)
            x_start = patch_h_idx * block_size * 2
            x_end = min(x_start + block_size * 2, dataset.width)
            
            # Extract target regions from original (Y at full res, U/V at half res)
            target = [
                orig_full[0][y_start:y_end, x_start:x_end],
                orig_full[1][y_start//2:y_end//2, x_start//2:x_end//2],
                orig_full[2][y_start//2:y_end//2, x_start//2:x_end//2]
            ]
            
            # Crop filtered patches to match target size
            for i in range(3):
                filtered_patch[i] = filtered_patch[i][:target[i].shape[0], :target[i].shape[1]]
            
            # Compute loss for all components
            loss = sum(compute_mse_loss(filtered_patch[i], target[i]) for i in range(3))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Compute average loss
        avg_loss = epoch_loss / len(selected_patch_indices)
        
        # Evaluate after each epoch
        model.eval()
        epoch_psnr_list = [[], [], []]  # Y, U, V
        
        with torch.no_grad():
            for frame_num in unique_frames:
                temp_dataset = Dataset(
                    input_yuv=input_yuv,
                    recon_yuv=recon_yuv,
                    log_enc=log_enc,
                    width=width,
                    height=height,
                    block_size=block_size,
                    pad_size=pad_size,
                    bit_depth=bit_depth,
                    frames=[frame_num],
                    device=device
                )
                filtered, orig, reco, metadata = process_frame_forward(
                    temp_dataset, model, block_size, device
                )
                
                # Compute PSNR for each component
                for i in range(3):
                    psnr = compute_psnr(orig[i], filtered[i])
                    epoch_psnr_list[i].append(psnr)
        
        avg_psnr = np.array([np.mean(epoch_psnr_list[i]) for i in range(3)])
        avg_gain = avg_psnr - avg_baseline
        
        print(f"Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.6f}")
        print(f"  PSNR: [{avg_psnr[0]:6.3f} {avg_psnr[1]:6.3f} {avg_psnr[2]:6.3f}] dB")
        print(f"  Gain: [{avg_gain[0]:+6.3f} {avg_gain[1]:+6.3f} {avg_gain[2]:+6.3f}] dB")
        
        # Save checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
            checkpoint_path = output_path.replace('.pt', f'_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr_y': avg_psnr[0],
                'psnr_u': avg_psnr[1],
                'psnr_v': avg_psnr[2],
                'gain_y': avg_gain[0],
                'gain_u': avg_gain[1],
                'gain_v': avg_gain[2],
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_psnr[0] > best_psnr_y:
            best_psnr_y = avg_psnr[0]
            best_path = output_path.replace('.pt', '_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr_y': avg_psnr[0],
                'psnr_u': avg_psnr[1],
                'psnr_v': avg_psnr[2],
                'gain_y': avg_gain[0],
                'gain_u': avg_gain[1],
                'gain_v': avg_gain[2],
            }, best_path)
            print(f"  New best model saved: {best_path} (Y PSNR: {best_psnr_y:.3f} dB)")
    
    # Save final model
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'psnr_y': avg_psnr[0],
        'psnr_u': avg_psnr[1],
        'psnr_v': avg_psnr[2],
        'gain_y': avg_gain[0],
        'gain_u': avg_gain[1],
        'gain_v': avg_gain[2],
    }, output_path)
    print(f"Final model saved: {output_path}")
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    model.eval()
    
    with torch.no_grad():
        for frame_idx, frame_num in enumerate(unique_frames):
            temp_dataset = Dataset(
                input_yuv=input_yuv,
                recon_yuv=recon_yuv,
                log_enc=log_enc,
                width=width,
                height=height,
                block_size=block_size,
                pad_size=pad_size,
                bit_depth=bit_depth,
                frames=[frame_num],
                device=device
            )
            filtered, orig, reco, metadata = process_frame_forward(
                temp_dataset, model, block_size, device
            )
            
            # Compute PSNR for each component
            psnr_filtered = np.array([compute_psnr(orig[i], filtered[i]) for i in range(3)])
            psnr_baseline = np.array([compute_psnr(orig[i], reco[i]) for i in range(3)])
            gain = psnr_filtered - psnr_baseline
            
            print(f"Frame {frame_idx+1:3d} (POC {metadata['poc']:3d}):")
            print(f"  Baseline: [{psnr_baseline[0]:6.3f} {psnr_baseline[1]:6.3f} {psnr_baseline[2]:6.3f}] dB")
            print(f"  Filtered: [{psnr_filtered[0]:6.3f} {psnr_filtered[1]:6.3f} {psnr_filtered[2]:6.3f}] dB")
            print(f"  Gain:     [{gain[0]:+6.3f} {gain[1]:+6.3f} {gain[2]:+6.3f}] dB")
    
    print("\nSummary:")
    print(f"  Initial average PSNR: [{avg_initial[0]:6.3f} {avg_initial[1]:6.3f} {avg_initial[2]:6.3f}] dB")
    print(f"  Initial gain:         [{avg_initial_gain[0]:+6.3f} {avg_initial_gain[1]:+6.3f} {avg_initial_gain[2]:+6.3f}] dB")
    print(f"  Final average PSNR:   [{avg_psnr[0]:6.3f} {avg_psnr[1]:6.3f} {avg_psnr[2]:6.3f}] dB")
    print(f"  Final gain:           [{avg_gain[0]:+6.3f} {avg_gain[1]:+6.3f} {avg_gain[2]:+6.3f}] dB")
    print(f"  Best Y PSNR achieved: {best_psnr_y:.3f} dB")
    print("=" * 80)


if __name__ == "__main__":
    train_multipliers()
