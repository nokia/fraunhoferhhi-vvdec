# NNPF Content Encode and Decode Guide

## Table of Contents

- [Introduction](#introduction)
- [Components](#components)
- [Prerequisites](#prerequisites)
- [Encode VVC](#encode-vvc)
- [Overfit the NNPF model](#overfit-the-nnpf-model)
- [Generate a compressed weights update](#generate-a-compressed-weights-update)
- [IOQ: Inference-based QP Optimization (Optional)](#ioq-inference-based-qp-optimization-optional)
- [Generate a video bitstream with NNPF SEI messages](#generate-a-video-bitstream-with-nnpf-sei-messages)
- [Decoding this content with VVdeC](#decoding-this-content-with-vvdec)
- [Realtime decoding and rendering](#realtime-decoding-and-rendering)
- [More obvious NNPF example](#more-obvious-nnpf-example)
- [Security considerations](#security-considerations)
- [References](#references)

## Introduction

This document describes how one can author and decode bitstreams incorporating NNPF (Neural Network Post Filter) data. A NNPF can serve multiple purposes and its parameters can be sent in SEI messages alongside the VVC bitstream. The examples here describe a fidelity improvement filter that is overfitted to the specific content in use.

There are four main steps to the authoring process: encoding, overfitting, coding a weights update, and generating a bitstream with NNPF SEI messages. The decoding process is more straightforward with the weight updating step performed automatically by VVdeC.

The inference process in this decoder uses OpenVINO to run the NNPF model. This allows for real-time inference on supported hardware such as Intel NPUs. Real time filtering of 1080p24 video with the model provided has previously been demonstrated [1].

## Components

The media workflow described in this guide involves the following components:

| Component | Description |
|--|--|
| VTM | The VVC Test Model reference encoder, used to encode video and insert NNPF SEI messages |
| VVdeC | The VVC decoder that supports NNPF SEI messages and performs NNPF inference during decoding |
| PyTorch | Used for overfitting the NNPF model to the content and for evaluating the model's performance |
| nncodec [2]  | A Python library used for encoding and decoding the weights updates for the NNPF model |
| OpenVINO | Used for running the NNPF model in the decoder for real-time inference on supported hardware |
| Python scripts | Custom scripts provided in this repository for overfitting the model, encoding weights updates, and decoding weights updates | 

Multiple Python scripts are provided in this folder to facilitate the overfitting and weights update processes. 

| Script | Description |
|--|--|
| [`overfit.py`](./overfit.py) | Adjusts the weights of the base model to optimize performance on the particular content that was encoded |
| [`evaluate.py`](./evaluate.py) | Evaluates the performance of the overfitted model on the encoded content |
| [`encode_weights_update.py`](./encode_weights_update.py) | Generates a compressed weights update payload for an NNPFC SEI message based on the difference between the base model and the overfitted model |
| [`ioq_encode_weights_update.py`](./ioq_encode_weights_update.py) | Similar to `encode_weights_update.py` but performs an IOQ (Inference-based QP Optimization) process to find optimal QP values for each tensor |
| [`decode_weights_update.py`](./decode_weights_update.py) | Decodes the weights update SEI message and reconstructs the overfitted model during decoding in VVdeC |
| [`to_onnx.py`](./to_onnx.py) | Converts the PyTorch model to ONNX format for use in SEI messages and the decoder |
| [`red.py`](./red.py), [`green.py`](./green.py), [`blue.py`](./blue.py) | Example scripts to generate simple color tint filter models for demonstration purposes |   


## Prerequisites

No special hardware is required to follow the steps in this guide. That said,
 * to run a useful amount of overfitting, appropriate hardware such as a GPU will be beneficial: use your hardware's pytorch support; and
 * to run the overfitted model in real-time in the decoder, an Intel NPU is recommended but the OpenVINO CPU offers a fallback for other platforms.

The OpenVino runtime will be required to build VVdeC with NNPF inference support.

Multiple Python modules are required to run the scripts in this guide, including PyTorch and OpenVINO. It is assumed the user is familiar with the installation of Python modules and the use of virtual environments.  Note that the `nncodec` module is required to run the scripts in this guide. This module is available on PyPI and can be installed with `pip install nncodec`


## Encode VVC

For encoding, use the VTM reference encoder to generate a VVC bitstream. To obtain and build the encoder, run the following commands.

```
git clone --depth 1 --branch VTM-23.14 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ../..
```

To encode video, use a command such as this, adjusted according to the actual source file that you are using. It is important to capture the command's output to `log_enc.txt` as data in this file will be used by the overfitting process.

```
./VVCSoftware_VTM/bin/EncoderAppStatic \
    -c VVCSoftware_VTM/cfg/encoder_randomaccess_vtm.cfg \
    --InputFile=input.yuv \
    --ReconFile=enc_rec.yuv \
    --BitstreamFile=stream_42.266 \
    --SourceWidth=1920 \
    --SourceHeight=1080 \
    --InputBitDepth=10 \
    --FrameRate=25 \
    --FramesToBeEncoded=25 \
    --QP=42 | tee log_enc.txt
```

## Overfit the NNPF model

An example base model is provided in [`./models/model.py`](./models//model.py).  It implements the convolutional filter of [3] and was converted from TensorFlow to PyTorch. Additional multiplier layers were added to the model to facilitate overfitting as described in [4].

Sample model weights are provided in `./models/model3.pt`. These were derived through training as described in [3] and then further fine tuning using the BVI-DVC [5] and DIV2K [6] datasets. These base model weights are not overfitted to the particular content and thus provide a starting point for overfitting.

The base model is also provided in ONNX format as `./models/model3.onnx`. This is the format required for the SEI messages and is also used by the decoder. The ONNX file was generated from the PyTorch model using the [`to_onnx.py`](./to_onnx.py) script.

To overfit this base model so that it performs optimally on the VVC bitstream just encoded, use the [`./overfit.py`](./overfit.py) script.:

```
python ./overfit.py \
    --model_path models/model3.pt \
    --output_path models/model_overfitted.pt \
    --input_yuv input.yuv \
    --recon_yuv enc_rec.yuv \
    --log_enc log_enc.txt \
    --width 1920 \
    --height 1080 \
    --bit_depth 10 \
    --block_size 64 \
    --pad_size 8 \
    --epochs 20 \
    --learning_rate 0.0003 \
    --save_interval 10
```

This command adjusts a subset of model weights such the model performs optimally on the particular distortions in the video previously encoded. To check the performance of the overfitted model, run:

```
python ./evaluate.py \
    --model_path models/model_overfitted.pt \
    --input_yuv input.yuv \
    --recon_yuv enc_rec.yuv \
    --log_enc log_enc.txt \
    --width 1920 \
    --height 1080 \
    --bit_depth 10 \
    --block_size 64 \
    --pad_size 8
```


## Generate a compressed weights update

It is assumed that decoders that will decode our content already have a copy of the base model. So, by sending updated weights as an SEI message, we can enable decoders to reconstruct the overfitted model. To do so we need to prepare a payload for an NNPFC "Neural Network Post Filter Characteristics" SEI message. This payload will contain the updated weights in a compressed format. To generate it, run:

```
python ./to_onnx.py \
    --input models/model_overfitted.pt \
    --output models/model_overfitted.onnx

python ./encode_weights_update.py \
    --base_model models/model3.onnx \
    --updated_model models/model_overfitted.onnx \
    --output_bitstream models/weights_update.bin \
    --output_base_model models/base_model.onnx \
    --output_decoded models/weights_diff_decoded.npz \
    --qp -32 \
    --qp_density 2 \
    --nonweight_qp -75 \
    --approx_method uniform \
    --sparsity 0.0 \
    --seed 42
```

### IOQ: Inference-based QP Optimization (Optional)

For improved rate-distortion performance, you can use IOQ (Inference-based QP Optimization) to find optimal per-tensor QP values. Instead of using a uniform QP for all weight tensors, IOQ searches for the best QP per tensor by minimizing a Lagrangian cost function:

$$\text{cost} = -\Delta\text{PSNR} + \lambda \cdot \Delta\text{rate}$$

where $\Delta\text{PSNR}$ is the quality improvement over the base model, and $\lambda$ is estimated from quantization trials.



To use IOQ instead of standard encoding:

```
python ioq_encode_weights_update.py \
    --base_model models/model3.onnx \
    --updated_model models/model3_overfitted_best.onnx \
    --model_path models/model3.pt \
    --output_bitstream models/weights_update_ioq.bin \
    --output_base_model models/base_model.onnx \
    --output_qp_map models/qp_per_tensor.json \
    --input_yuv input.yuv \
    --recon_yuv enc_rec.yuv \
    --log_enc log_enc.txt \
    --width 1920 \
    --height 1080 \
    --bit_depth 10 \
    --block_size 64 \
    --pad_size 8 \
    --qp -32 \
    --qp_search_range 4 \
    --seed 42
```

The script outputs:
- `models/weights_update_ioq.bin`: Optimized bitstream (use this in SEI messages)
- `models/base_model.onnx`: Modified base model with short names
- `models/qp_per_tensor.json`: Optimized QP values per tensor (for analysis/reproducibility)

**Key parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--qp` | Base QP (starting point for optimization) | -32 |
| `--qp_search_range` | Search range around base QP (±N) | 4 |
| `--lambda_override` | Override estimated λ with fixed value | auto |
| `--num_frames` | Limit frames for evaluation (faster but less accurate) | all |

## Generate a video bitstream with NNPF SEI messages

Finally, we can generate a new VVC bitstream that incorporates the NNPF SEI messages containing the overfitted model weights. In principle we could do this by inserting the SEI messages into the bitstream generated in the first step. However, with the VTM encoder one must re-encode the video and insert the SEI messages during encoding. To do so, run:

```
./VVCSoftware_VTM/bin/EncoderAppStatic \
    -c VVCSoftware_VTM/cfg/encoder_randomaccess_vtm.cfg \
    --InputFile=input.yuv \
    --ReconFile=/dev/null \
    --BitstreamFile=./stream_42_with_sei.266 \
    --SourceWidth=1920 \
    --SourceHeight=1080 \
    --InputBitDepth=10 \
    --FrameRate=25 \
    --FramesToBeEncoded=25 \
    --QP=42 \
\
    --SEINNPFCEnabled=1 \
    --SEINNPFCNumFilters=2 \
\
    --SEINNPFCId0=0 \
    --SEINNPFCBaseFlag0=1 \
    --SEINNPFCModeIdc0=1 \
    --SEINNPFCPurpose0=0 \
    --SEINNPFCUriTag0="tag:localhost,2017:onnx" \
    --SEINNPFCUri0="file://./models/base_model.onnx" \
    --SEINNPFCPropertyPresentFlag0=1 \
    --SEINNPFCInpOrderIdc0=3 \
    --SEINNPFCOutOrderIdc0=3 \
    --SEINNPFCAuxInpIdc0=1 \
    --SEINNPFCInpFormatIdc0=0 \
    --SEINNPFCOutFormatIdc0=0 \
\
    --SEINNPFCId1=0 \
    --SEINNPFCBaseFlag1=0 \
    --SEINNPFCModeIdc1=0 \
    --SEINNPFCPurpose1=0 \
    --SEINNPFCPayloadFilename1=weights_update.bin \
    --SEINNPFCPropertyPresentFlag1=0 \
\
    --SEINNPostFilterActivationEnabled=1 \
    --SEINNPostFilterActivationTargetId=0 \
    --SEINNPostFilterActivationTargetBaseFlag=0 \
    --SEINNPostFilterActivationPersistenceFlag=1 \
\
    | tee log_enc_with_sei.txt
```

>Note that setting `SEINNPFCPurpose0=1` (announcing that this is a filter for fidelity enhancement) causes an encoder failure with this version of VTM. This has no effect on the decoder which should apply the NNPF regardless of the value of the purpose field. 

>Note also that the example URI given here is for a local file. The reader is invited to test `https://` web URIs here too. To do so, the file must be hosted on a server and the URI must point to the file's location on that server. The decoder will need to have access to the server to download the file during decoding.

>Note that the Tag URI `tag:localhost,2017:onnx` shown here is a placeholder and has no effect on the decoder. 

## Decoding this content with VVdeC

Follow the instructions in the [VVdeC README](../README.md) to build the VVdeC decoder. Then, to decode the bitstream with SEI messages, run:

```
vvdecapp -i stream_42_with_sei.266 -o output.yuv
```

The decoder will parse the SEI messages, reconstruct the overfitted model, and apply the NNPF to the reconstructed video frames. The resulting output video should have improved fidelity compared to a version of the video that has not been processed by the NNPF.

The decoder invokes python script `decode_weights_update.py` to decode the weights update SEI message and reconstruct the overfitted model. The OpenVINO library is used to run the model on the decoded video frames. This permits real-time inference on supported hardware, for example Intel NPUs. 

NNPF inference behaviour may be influenced by setting one or more environment variables as detailed below.

|Environment Variable|Description|Default|
|--|--|--|
|NNPF_MODEL|Override the model used | Use model specified by NNPFC and NNPFA SEI messages|
|NNPF_OPENVINO_DEVICE|Specify the OpenVINO device to use|CPU|
|NNPF_SYNCHRONOUS|Enable synchronous inference (1=synchronous, 0=asynchronous)|1|
|NNPF_GAIN|Gain multiplier for the filter output - does not work with asynchronous inference|1.0|



## Realtime decoding and rendering

For realtime decoding and rendering, it is necessary to integrate VVdeC to a video player for example by using its FFmpeg integration to `ffplay` or to `mpv`.

For full performance on suitable hardware, it may be advantageous to set `NNPF_OPENVINO_DEVICE=NPU` and `NNPF_SYNCHRONOUS=0` to enable asynchronous inference on an Intel NPU. This allows the NNPF inference to be performed in parallel with decoding and rendering.


## More obvious NNPF example

It is not always obvious that the NNPF is working when looking at the decoded video, especially if the overfitted model only provides a subtle improvement in fidelity. To more clearly demonstrate the effect of the NNPF, we test with some simple filter models that apply strong color tints to the video. 

Here is an example that uses the provided 'blue.onnx' model to apply a strong blue tint to the video. The SEI messages for this example are configured to activate the filter from the start of the video and persist it throughout. To test this, run:

```
./VVCSoftware_VTM/bin/EncoderAppStatic \
    -c VVCSoftware_VTM/cfg/encoder_randomaccess_vtm.cfg \
    --InputFile=input.yuv \
    --ReconFile=/dev/null \
    --BitstreamFile=./stream_blue.266 \
    --SourceWidth=1920 \
    --SourceHeight=1080 \
    --InputBitDepth=10 \
    --FrameRate=25 \
    --FramesToBeEncoded=25 \
    --QP=42 \
\
    --SEINNPFCEnabled=1 \
    --SEINNPFCNumFilters=1 \
\
    --SEINNPFCId0=0 \
    --SEINNPFCBaseFlag0=1 \
    --SEINNPFCModeIdc0=1 \
    --SEINNPFCPurpose0=0 \
    --SEINNPFCUriTag0="tag:onnx.ai,2017:onnx" \
    --SEINNPFCUri0="file://./models/blue.onnx" \
    --SEINNPFCPropertyPresentFlag0=1 \
    --SEINNPFCInpOrderIdc0=3 \
    --SEINNPFCOutOrderIdc0=3 \
    --SEINNPFCAuxInpIdc0=1 \
    --SEINNPFCInpFormatIdc0=0 \
    --SEINNPFCOutFormatIdc0=0 \
\
    --SEINNPostFilterActivationEnabled=1 \
    --SEINNPostFilterActivationTargetId=0 \
    --SEINNPostFilterActivationTargetBaseFlag=1 \
    --SEINNPostFilterActivationPersistenceFlag=1
```

>The filter models `models/red.onnx`, `models/green.onnx`, and `models/blue.onnx` are simple color tint filters that modify the U and V chroma components to produce red, green, and blue color effects, respectively.


## Security considerations

NNPF SEI messages can contain arbitrary URIs and arbritrary model data that is not subject to any constraints. This means that a malicious content provider could craft NNPF SEI messages in such a way as to cause harm to decoders that process them. For example, the model data could be maliciously designed to exploit vulnerabilities in the decoder's model parsing or inference code, or to cause excessive resource consumption.

This implementation processes weights update using a `std::system` call to invoke a Python script. It is entirely possible that a maliciously crafted bitstream exploits command injection vulnerabilities in our code.


## References


[1] J. Funnell, M. Santamaria Gomez, R. Yang, F. Cricri, M. M. Hannuksela, S. Schwarz, "AHG9/AHG11: Demo of real-time NNPF inference," Joint Video Experts Team (JVET) of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, 37th Meeting, Geneva, CH, Doc. JVET-AK00258-v1, Jan. 2025.

[2] D. Becking, P. Haase, H. Kirchhoffer, K. Müller, W. Samek, and D. Marpe, "{NNC}odec: An Open Source Software Implementation of the Neural Network Coding {ISO}/{IEC} Standard," ICML 2023 Workshop Neural Compression: From Information Theory to Applications, 2023. 

[3] H. Wang, J. Chen, K. Reuze, A. M. Kotra, and M. Karczewicz, "EE1-related: Neural Network-based in-loop filter with constrained computational complexity," Joint Video Experts Team (JVET), ITU‑T SG16 WP3 and ISO/IEC JTC1/SC29, Doc. JVET‑W0131‑v2, 23rd Meeting, Jul. 2021.

[4] R. Yang et al., "AHG11: Content-adaptive neural network loop-filter," Joint Video Experts Team (JVET) of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29, 31st Meeting, Geneva, CH, Document JVET-AE0093-v2, July 2023.

[5] D. Ma, F. Zhang, and D. R. Bull, "BVI-DVC: A Training Database for Deep Video Compression," IEEE Transactions on Multimedia, vol. 24, pp. 3847-3858, 2022.

[6] E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study," in Proc. IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1122-1131, Jul. 2017.
