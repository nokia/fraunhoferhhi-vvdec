"""Dataset module for neural network filter training."""

import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import torch.nn.functional as F


class Dataset(TorchDataset):
    """
    Custom dataset class for neural network filter data.
    Each dataset instance represents one frame of video.
    """
    
    def __init__(
        self,
        input_yuv,
        recon_yuv,
        log_enc=None,
        width=None,
        height=None,
        block_size=64,
        pad_size=8,
        bit_depth=10,
        frames=None,
        frames_to_skip=0,
        device='cpu'
    ):
        """
        Initialize the dataset for a single video sequence.
        
        Args:
            input_yuv: Path to input (original) YUV file
            recon_yuv: Path to reconstructed YUV file
            log_enc: Path to encoder log file (for deriving frame QP from POC). Optional.
            width: Frame width in pixels
            height: Frame height in pixels
            block_size: Block size without padding (default: 64)
            pad_size: Padding size (default: 8)
            bit_depth: Bit depth of images (default: 10)
            frames: List of frame indices to load (0-based), e.g., [0, 1, 2] or [9]
            frames_to_skip: Number of frames to skip in the input YUV (default: 0)
            device: torch device to use
        """
        self.orig_yuv_path = Path(input_yuv)
        self.reco_yuv_path = Path(recon_yuv)
        self.log_enc_path = Path(log_enc) if log_enc is not None else None
        self.width = width
        self.height = height
        self.block_size = block_size
        self.pad_size = pad_size
        self.bit_depth = bit_depth
        self.device = device
        self.frames_to_skip = frames_to_skip
        self.frames = frames if frames is not None else [0]
        
        if not self.orig_yuv_path.exists():
            raise FileNotFoundError(f"Original YUV file not found: {self.orig_yuv_path}")
        
        if not self.reco_yuv_path.exists():
            raise FileNotFoundError(f"Reconstructed YUV file not found: {self.reco_yuv_path}")
        
        # Parse log file to get frame information (if provided)
        if self.log_enc_path is not None:
            if not self.log_enc_path.exists():
                raise FileNotFoundError(f"Encoder log file not found: {self.log_enc_path}")
            self.frames_info = self._parse_log_enc(self.log_enc_path)
        else:
            # No log file provided - create dummy frame info without QP/type
            # Count frames from YUV file size
            num_frames = self._count_frames_in_yuv(self.reco_yuv_path, width, height, bit_depth)
            self.frames_info = [
                {'poc': i, 'frame_type': 'U', 'qp': 0}
                for i in range(num_frames)
            ]
        
        # Validate frame numbers
        for frame_num in self.frames:
            if frame_num >= len(self.frames_info):
                raise IndexError(f"Frame number {frame_num} is out of range (0-{len(self.frames_info)-1})")
        
        # Pre-load frame data and extract patches for all requested frames
        self._load_all_frames_data()
    
    def _load_all_frames_data(self):
        """Load and process data for all requested frames, extracting all patches."""
        all_input_patches = []
        self.frame_metadata = []  # Store metadata for each frame's patches
        
        for frame_number in self.frames:
            # Get frame info from parsed log
            frame_info = self.frames_info[frame_number]
            curr_poc = frame_info['poc']
            frame_type = frame_info['frame_type']
            frame_qp = frame_info['qp']
        
            # Calculate original POC
            orig_poc = curr_poc + self.frames_to_skip
            
            # Read YUV frames (luma and chroma)
            orig_img, orig_u, orig_v = self._read_yuv_frame(
                self.orig_yuv_path, orig_poc, self.width, self.height, self.bit_depth
            )
            reco_img, reco_u, reco_v = self._read_yuv_frame(
                self.reco_yuv_path, curr_poc, self.width, self.height, self.bit_depth
            )
            
            # Convert to tensors
            orig_tensor = torch.from_numpy(orig_img).to(self.device)
            reco_tensor = torch.from_numpy(reco_img).to(self.device)
        
            # Apply padding to luma
            reco_padded = self._pad_image(reco_img, self.block_size * 2, self.pad_size).to(self.device)
            
            # Apply padding to chroma (half resolution)
            reco_u_padded = self._pad_image(reco_u, self.block_size, self.pad_size // 2).to(self.device)
            reco_v_padded = self._pad_image(reco_v, self.block_size, self.pad_size // 2).to(self.device)
            
            # Interleave luma
            reco_tl, reco_tr, reco_bl, reco_br = self._interleave_image(reco_padded)
            
            # Extract patches from luma
            patch_size = self.block_size + self.pad_size
            stride = self.block_size
            
            patches_tl = self._extract_patches(reco_tl, patch_size, stride)
            patches_tr = self._extract_patches(reco_tr, patch_size, stride)
            patches_bl = self._extract_patches(reco_bl, patch_size, stride)
            patches_br = self._extract_patches(reco_br, patch_size, stride)
            
            # Extract patches from chroma
            patches_u = self._extract_patches(reco_u_padded, patch_size, stride)
            patches_v = self._extract_patches(reco_v_padded, patch_size, stride)
            
            # Normalize QP to 0...1 range (VVC QP range: 0-63)
            qp_normalized = frame_qp / 63.0
            
            # Create QP channel
            num_patches = patches_tl.shape[0]
            qp_channel = torch.full((num_patches, 1, patch_size, patch_size), 
                                    qp_normalized, device=self.device)
            
            # Stack into (N, 7, H, W) format: 4 luma + 2 chroma + 1 qp
            input_patches = torch.cat([patches_tl, patches_tr, patches_bl, patches_br, 
                                       patches_u, patches_v, qp_channel], dim=1)
            
            # Calculate number of blocks
            num_h_blocks = self.width // (self.block_size * 2)
            if self.width % (self.block_size * 2) > 0:
                num_h_blocks += 1
            num_v_blocks = self.height // (self.block_size * 2)
            if self.height % (self.block_size * 2) > 0:
                num_v_blocks += 1
            
            # Store patches and metadata for this frame
            all_input_patches.append(input_patches)
            
            # Store metadata for each patch in this frame
            for patch_idx in range(num_patches):
                self.frame_metadata.append({
                    'orig_img': orig_tensor,
                    'reco_img': reco_tensor,
                    'poc': curr_poc,
                    'frame_type': frame_type,
                    'frame_qp': frame_qp,
                    'height': self.height,
                    'width': self.width,
                    'num_v_blocks': num_v_blocks,
                    'num_h_blocks': num_h_blocks,
                    'frame_number': frame_number,
                    'patch_idx_in_frame': patch_idx,
                })
        
        # Concatenate all patches from all frames
        self.input_patches = torch.cat(all_input_patches, dim=0)
    
    def __len__(self):
        """Return the number of patches in the dataset."""
        return self.input_patches.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a single patch from the dataset.
        
        Args:
            idx: Patch index (across all frames)
            
        Returns:
            Dictionary containing:
                - 'input': Input tensor (10, patch_size, patch_size) - single patch
                - 'patch_idx': Global index of this patch across all frames
                - 'patch_idx_in_frame': Index of this patch within its frame
                - 'frame_number': Frame number this patch belongs to
                - 'orig_img': Original image tensor (H, W) - full frame
                - 'reco_img': Reconstructed image tensor (H, W) - full frame
                - 'poc': Picture Order Count
                - 'frame_type': Frame type (I, P, B, etc.)
                - 'frame_qp': Frame QP value
                - 'height': Original image height
                - 'width': Original image width
                - 'num_v_blocks': Number of vertical blocks
                - 'num_h_blocks': Number of horizontal blocks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        metadata = self.frame_metadata[idx]
        
        return {
            'input': self.input_patches[idx],  # Single patch (7, H, W)
            'patch_idx': idx,
            'patch_idx_in_frame': metadata['patch_idx_in_frame'],
            'frame_number': metadata['frame_number'],
            'orig_img': metadata['orig_img'],
            'reco_img': metadata['reco_img'],
            'poc': metadata['poc'],
            'frame_type': metadata['frame_type'],
            'frame_qp': metadata['frame_qp'],
            'height': metadata['height'],
            'width': metadata['width'],
            'num_v_blocks': metadata['num_v_blocks'],
            'num_h_blocks': metadata['num_h_blocks']
        }
    
    @staticmethod
    def _parse_log_enc(log_path):
        """
        Parse log_enc.txt (encoder log) to extract frame information.
        The encoder log shows frames in bitstream order, but we use POC numbers
        to get the frame information in display order.
        
        Args:
            log_path: Path to log_enc.txt file
            
        Returns:
            List of dicts with 'poc', 'frame_type', and 'qp' for each frame (sorted by POC)
        """
        frames = []
        # Pattern to match encoder log lines like:
        # POC    0 LId:  0 TId: 0 ( IDR_N_LP, I-SLICE, QP 29 )...
        # POC   16 LId:  0 TId: 1 ( STSA, B-SLICE, QP 34 )...
        pattern = r'POC\s+(\d+).*?([IB])-SLICE.*?QP\s+(\d+)'
        
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    poc = int(match.group(1))
                    frame_type = match.group(2)
                    qp = int(match.group(3))
                    frames.append({
                        'poc': poc,
                        'frame_type': frame_type,
                        'qp': qp
                    })
        
        # Sort by POC to get display order
        frames.sort(key=lambda x: x['poc'])
        return frames
    
    @staticmethod
    def _count_frames_in_yuv(yuv_path, width, height, bit_depth):
        """
        Count the number of frames in a YUV file based on file size.
        
        Args:
            yuv_path: Path to YUV file
            width: Frame width
            height: Frame height
            bit_depth: Bit depth (8 or 10)
            
        Returns:
            Number of frames in the file
        """
        import os
        file_size = os.path.getsize(yuv_path)
        
        # Calculate frame size
        # YUV 4:2:0 format: Y plane is width*height, U and V are each (width/2)*(height/2)
        # Total pixels per frame = width*height*1.5
        bytes_per_pixel = 1 if bit_depth == 8 else 2
        frame_size = int(width * height * 1.5 * bytes_per_pixel)
        
        num_frames = file_size // frame_size
        return num_frames
    
    @staticmethod
    def _parse_log_dec(log_path):
        """
        Parse log_dec.txt (decoder log) to extract frame information.
        
        Args:
            log_path: Path to log_dec.txt file
            
        Returns:
            List of dicts with 'poc', 'frame_type', and 'qp' for each frame
        """
        frames = []
        pattern = r'POC\s+(\d+).*?([IB])-SLICE.*?QP\s+(\d+)'
        
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    poc = int(match.group(1))
                    frame_type = match.group(2)
                    qp = int(match.group(3))
                    frames.append({
                        'poc': poc,
                        'frame_type': frame_type,
                        'qp': qp
                    })
        
        # Sort by POC to ensure correct order
        frames.sort(key=lambda x: x['poc'])
        return frames
    
    @staticmethod
    def _read_yuv_frame(yuv_path, frame_idx, width, height, bit_depth=10):
        """
        Read a single frame from a YUV 4:2:0 file.
        
        Args:
            yuv_path: Path to YUV file
            frame_idx: Frame index (0-based)
            width: Frame width
            height: Frame height
            bit_depth: Bit depth (default: 10)
            
        Returns:
            Tuple of (luma, chroma_u, chroma_v) as float32 numpy arrays normalized to [0, 1]
        """
        # Calculate frame size
        chroma_width = width // 2
        chroma_height = height // 2
        
        # For 10-bit, each sample is 2 bytes
        bytes_per_sample = 2 if bit_depth > 8 else 1
        luma_size = width * height * bytes_per_sample
        chroma_size = chroma_width * chroma_height * bytes_per_sample
        frame_size = luma_size + 2 * chroma_size
        
        # Seek to frame position
        with open(yuv_path, 'rb') as f:
            f.seek(frame_idx * frame_size)
            
            # Read luma
            if bit_depth > 8:
                luma_bytes = f.read(luma_size)
                luma = np.frombuffer(luma_bytes, dtype=np.uint16).reshape(height, width)
            else:
                luma_bytes = f.read(luma_size)
                luma = np.frombuffer(luma_bytes, dtype=np.uint8).reshape(height, width).astype(np.uint16)
            
            # Read chroma U
            if bit_depth > 8:
                u_bytes = f.read(chroma_size)
                chroma_u = np.frombuffer(u_bytes, dtype=np.uint16).reshape(chroma_height, chroma_width)
            else:
                u_bytes = f.read(chroma_size)
                chroma_u = np.frombuffer(u_bytes, dtype=np.uint8).reshape(chroma_height, chroma_width).astype(np.uint16)
            
            # Read chroma V
            if bit_depth > 8:
                v_bytes = f.read(chroma_size)
                chroma_v = np.frombuffer(v_bytes, dtype=np.uint16).reshape(chroma_height, chroma_width)
            else:
                v_bytes = f.read(chroma_size)
                chroma_v = np.frombuffer(v_bytes, dtype=np.uint8).reshape(chroma_height, chroma_width).astype(np.uint16)
        
        # Normalize to [0, 1]
        max_val = 2**bit_depth - 1
        luma = luma.astype(np.float32) / max_val
        chroma_u = chroma_u.astype(np.float32) / max_val
        chroma_v = chroma_v.astype(np.float32) / max_val
        
        return luma, chroma_u, chroma_v
    
    @staticmethod
    def _pad_image(image, block_size, pad_size):
        """
        Apply reflection padding to image edges
        Args:
            image: numpy array (H, W)
            block_size: size of blocks (without padding)
            pad_size: padding to add on each side
        Returns:
            padded image as torch tensor (1, 1, H', W')
        """
        h, w = image.shape
        
        # Pad left and right
        left_pad = image[:, :1].repeat(pad_size, axis=1)
        right_excess = w % block_size
        if right_excess > 0:
            right_pad_size = pad_size + block_size - right_excess
        else:
            right_pad_size = pad_size
        right_pad = image[:, -1:].repeat(right_pad_size, axis=1)
        image = np.concatenate([left_pad, image, right_pad], axis=1)
        
        # Pad top and bottom
        top_pad = image[:1, :].repeat(pad_size, axis=0)
        bottom_excess = h % block_size
        if bottom_excess > 0:
            bottom_pad_size = pad_size + block_size - bottom_excess
        else:
            bottom_pad_size = pad_size
        bottom_pad = image[-1:, :].repeat(bottom_pad_size, axis=0)
        image = np.concatenate([top_pad, image, bottom_pad], axis=0)
        
        return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    @staticmethod
    def _interleave_image(image):
        """
        Interleave an image into four partitions (checkerboard pattern)
        Args:
            image: torch tensor (1, 1, H, W)
        Returns:
            Four image partitions: tl, tr, bl, br each (1, 1, H/2, W/2)
        """
        tl = image[:, :, 0::2, 0::2]  # top-left
        tr = image[:, :, 0::2, 1::2]  # top-right
        bl = image[:, :, 1::2, 0::2]  # bottom-left
        br = image[:, :, 1::2, 1::2]  # bottom-right
        return tl, tr, bl, br
    
    @staticmethod
    def _extract_patches(image, patch_size, stride):
        """
        Extract non-overlapping patches from an image
        Args:
            image: torch tensor (1, 1, H, W)
            patch_size: size of each patch
            stride: stride between patches
        Returns:
            patches: torch tensor (N, 1, patch_size, patch_size)
        """
        patches = F.unfold(image, kernel_size=patch_size, stride=stride)
        # patches shape: (1, patch_size*patch_size, num_patches)
        num_patches = patches.shape[2]
        patches = patches.transpose(1, 2).reshape(num_patches, 1, patch_size, patch_size)
        return patches
    
    @staticmethod
    def _add_zeros_to_image(input_data, num_channels=3):
        """
        Add zero channels to input
        Args:
            input_data: torch tensor (N, C, H, W)
            num_channels: number of zero channels to add
        Returns:
            tensor with zeros: (N, C+num_channels, H, W)
        """
        n, c, h, w = input_data.shape
        zeros = torch.zeros(n, num_channels, h, w, device=input_data.device)
        return torch.cat([input_data, zeros], dim=1)
