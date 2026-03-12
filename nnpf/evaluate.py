"""
 © 2026 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
"""

"""
Evaluation script to load a PyTorch or ONNX model and compute PSNR metrics
against the dataset used in training.py
Uses the FilterWithMultipliersPyTorch model class for PyTorch models,
and onnxruntime for ONNX models.
"""

import click
import numpy as np
import torch
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


def process_frame_data(dataset, model, block_size, device, is_onnx=False):
    """
    Process frame data through the model and compute PSNR for Y, U, V channels.
    
    Args:
        dataset: Dataset instance that iterates over patches
        model: PyTorch model or ONNX InferenceSession (or None to skip filtering)
        block_size: Block size without padding
        device: torch device
        is_onnx: Whether the model is an ONNX model
        
    Returns:
        psnr_without_filter (Y,U,V), psnr_with_filter (Y,U,V) if model provided, frame_metadata
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
    
    # Convert to tensors (orig_yuv and reco_yuv are tuples of 3 numpy arrays)
    orig = [torch.from_numpy(orig_yuv[i]).to(device) for i in range(3)]
    reco = [torch.from_numpy(reco_yuv[i]).to(device) for i in range(3)]
    
    # Compute PSNR without filter (baseline)
    psnr_without = np.array([compute_psnr(orig[i], reco[i]) for i in range(3)])
    
    if model is None:
        return psnr_without, None, (poc, frame_type, frame_qp)
    
    # Process all patches through the model
    output_patches_list = []
    num_patches = len(dataset)
    
    with torch.no_grad():
        for i in range(num_patches):
            patch_data = dataset[i]
            input_patch = patch_data['input'].unsqueeze(0)  # Add batch dimension (1, 7, H, W)
            
            if is_onnx:
                # ONNX inference
                input_np = input_patch.cpu().numpy()
                ort_inputs = {model.get_inputs()[0].name: input_np}
                ort_outputs = model.run(None, ort_inputs)
                output_patch = torch.from_numpy(ort_outputs[0]).to(device)  # (1, 6, 64, 64)
            else:
                # PyTorch inference
                output_patch = model(input_patch)  # (1, 6, 64, 64)
            
            output_patches_list.append(output_patch[0])  # Remove batch dimension
    
    # Stack all patches: (num_patches, 6, 64, 64)
    output_patches = torch.stack(output_patches_list, dim=0)
    
    # Extract luma (4 channels) and chroma (2 channels) from output
    # output_patches_yuv[0] = Y (4 channels), output_patches_yuv[1] = U (1 channel), output_patches_yuv[2] = V (1 channel)
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
    
    # Process chroma U: Reshape
    output_patches_grid_u = output_patches_yuv[1][:total_expected_patches].view(
        num_v_blocks, num_h_blocks, 1, block_size, block_size
    )
    output_patches_grid_u = output_patches_grid_u.permute(2, 0, 3, 1, 4)
    output_patches_grid_u = output_patches_grid_u.reshape(
        1, num_v_blocks * block_size, num_h_blocks * block_size
    )
    filtered_u = de_interleave_chroma(output_patches_grid_u.unsqueeze(0))
    filtered_u = filtered_u[:height // 2, :width // 2]
    
    # Process chroma V: Reshape
    output_patches_grid_v = output_patches_yuv[2][:total_expected_patches].view(
        num_v_blocks, num_h_blocks, 1, block_size, block_size
    )
    output_patches_grid_v = output_patches_grid_v.permute(2, 0, 3, 1, 4)
    output_patches_grid_v = output_patches_grid_v.reshape(
        1, num_v_blocks * block_size, num_h_blocks * block_size
    )
    filtered_v = de_interleave_chroma(output_patches_grid_v.unsqueeze(0))
    filtered_v = filtered_v[:height // 2, :width // 2]
    
    # Compute PSNR with filter for all channels
    filtered = [filtered_y, filtered_u, filtered_v]
    psnr_with = np.array([compute_psnr(orig[i], filtered[i]) for i in range(3)])
    
    return psnr_without, psnr_with, (poc, frame_type, frame_qp)


@click.command()
@click.option(
    "--model_path",
    default="model.pt",
    type=click.Path(exists=True),
    help="Path to the PyTorch (.pt) or ONNX (.onnx) model file",
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
    help="Path to encoder log file (for frame QP info, uses POC to handle reordering)",
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
    help="Number of frames to process (None for all)",
)
def evaluate_model(
    model_path,
    input_yuv,
    recon_yuv,
    log_enc,
    width,
    height,
    bit_depth,
    block_size,
    pad_size,
    num_frames,
):
    """
    Evaluate a PyTorch or ONNX model on a video sequence and report PSNR metrics
    """
    print(f"Loading model from: {model_path}")
    print(f"Input YUV: {input_yuv}")
    print(f"Recon YUV: {recon_yuv}")
    print(f"Log file: {log_enc}")
    print(f"Resolution: {width}x{height}")
    print(f"Block size: {block_size}, Pad size: {pad_size}, Bit depth: {bit_depth}")
    print("-" * 80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine if model is ONNX or PyTorch
    is_onnx = model_path.endswith('.onnx')
    
    # Load model
    try:
        if is_onnx:
            # Load ONNX model
            import onnxruntime as ort
            print("Loading ONNX model...")
            
            # Set up ONNX Runtime session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Choose execution provider based on device
            if device.type == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            model = ort.InferenceSession(model_path, sess_options, providers=providers)
            print(f"ONNX model loaded successfully (providers: {model.get_providers()})")
        else:
            # Load PyTorch model
            print("Loading PyTorch model...")
            # Instantiate the model
            model = FilterWithMultipliersPyTorch()
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Handle different save formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint contains metadata
                state_dict = checkpoint['model_state_dict']
            else:
                # Direct state_dict
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            model.to(device)
            model.eval()
            print("PyTorch model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing without model (will only compute baseline PSNR)")
        model = None
        is_onnx = False
    
    # Determine the number of frames to process
    # Create a temporary dataset just to count frames
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
        # Get the total number of frames available
        total_frames = len(temp_dataset.frames_info)
        print(f"Total frames available: {total_frames}")
        
        if num_frames is None:
            frames_to_process = total_frames
        else:
            frames_to_process = min(num_frames, total_frames)
        
        print(f"Will process {frames_to_process} frames")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    
    # Process frames
    psnr_without_list = []  # Will store arrays of shape (3,) for [Y, U, V]
    psnr_with_list = []     # Will store arrays of shape (3,) for [Y, U, V]
    
    print("\nProcessing frames...")
    for idx in range(frames_to_process):
        # Create dataset for this specific frame
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
                frames=[idx],
                device=device
            )
        except Exception as e:
            print(f"Error loading frame {idx}: {e}")
            continue
        
        # Process frame
        psnr_without, psnr_with, (poc, frame_type, frame_qp) = process_frame_data(
            dataset, model, block_size, device, is_onnx
        )
        
        psnr_without_list.append(psnr_without)
        
        if psnr_with is not None:
            psnr_with_list.append(psnr_with)
        
        # Print progress
        if model is not None:
            gain = psnr_with - psnr_without
            print(f"Frame {idx+1:3d} (POC {poc:3d}, {frame_type}, QP {frame_qp}):")
            print(f"  Without filter: [{psnr_without[0]:6.3f} {psnr_without[1]:6.3f} {psnr_without[2]:6.3f}] dB")
            print(f"  With filter:    [{psnr_with[0]:6.3f} {psnr_with[1]:6.3f} {psnr_with[2]:6.3f}] dB")
            print(f"  Gain:           [{gain[0]:+6.3f} {gain[1]:+6.3f} {gain[2]:+6.3f}] dB")
        else:
            print(f"Frame {idx+1:3d} (POC {poc:3d}, {frame_type}, QP {frame_qp}): "
                  f"PSNR: [{psnr_without[0]:6.3f} {psnr_without[1]:6.3f} {psnr_without[2]:6.3f}] dB")
    
    # Compute and print average metrics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if psnr_without_list:
        avg_without = np.mean(psnr_without_list, axis=0)  # Average across frames, shape (3,)
        print(f"Average PSNR without post-filter: [{avg_without[0]:6.3f} {avg_without[1]:6.3f} {avg_without[2]:6.3f}] dB")
    
    if psnr_with_list:
        avg_with = np.mean(psnr_with_list, axis=0)  # Average across frames, shape (3,)
        avg_gain = avg_with - avg_without
        print(f"Average PSNR with post-filter:    [{avg_with[0]:6.3f} {avg_with[1]:6.3f} {avg_with[2]:6.3f}] dB")
        print(f"Average PSNR gain:                 [{avg_gain[0]:+6.3f} {avg_gain[1]:+6.3f} {avg_gain[2]:+6.3f}] dB")
    
    print("=" * 80)


if __name__ == "__main__":
    evaluate_model()
