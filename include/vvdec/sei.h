/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2018-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVdeC Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */

#ifndef VVDEC_SEI_H
#define VVDEC_SEI_H

#include <stdio.h>
#include <stdint.h>

typedef enum
{
  VVDEC_BUFFERING_PERIOD                     = 0,
  VVDEC_PICTURE_TIMING                       = 1,
  VVDEC_FILLER_PAYLOAD                       = 3,
  VVDEC_USER_DATA_REGISTERED_ITU_T_T35       = 4,
  VVDEC_USER_DATA_UNREGISTERED               = 5,
  VVDEC_FILM_GRAIN_CHARACTERISTICS           = 19,
  VVDEC_FRAME_PACKING                        = 45,
  VVDEC_PARAMETER_SETS_INCLUSION_INDICATION  = 129,
  VVDEC_DECODING_UNIT_INFO                   = 130,
  VVDEC_DECODED_PICTURE_HASH                 = 132,
  VVDEC_SCALABLE_NESTING                     = 133,
  VVDEC_MASTERING_DISPLAY_COLOUR_VOLUME      = 137,
  VVDEC_DEPENDENT_RAP_INDICATION             = 145,
  VVDEC_EQUIRECTANGULAR_PROJECTION           = 150,
  VVDEC_SPHERE_ROTATION                      = 154,
  VVDEC_REGION_WISE_PACKING                  = 155,
  VVDEC_OMNI_VIEWPORT                        = 156,
  VVDEC_GENERALIZED_CUBEMAP_PROJECTION       = 153,
  VVDEC_FRAME_FIELD_INFO                     = 168,
  VVDEC_SUBPICTURE_LEVEL_INFO                = 203,
  VVDEC_SAMPLE_ASPECT_RATIO_INFO             = 204,
  VVDEC_CONTENT_LIGHT_LEVEL_INFO             = 144,
  VVDEC_ALTERNATIVE_TRANSFER_CHARACTERISTICS = 147,
  VVDEC_AMBIENT_VIEWING_ENVIRONMENT          = 148,
  VVDEC_CONTENT_COLOUR_VOLUME                = 149,
  VVDEC_NEURAL_NETWORK_POST_FILTER_CHARACTERISTICS = 210,
  VVDEC_NEURAL_NETWORK_POST_FILTER_ACTIVATION      = 211,
    
  VVDEC_SEI_UNKNOWN                          = -1,
}vvdecSEIPayloadType;

typedef enum
{
  VVDEC_LEVEL_NONE = 0,
  VVDEC_LEVEL1   = 16,
  VVDEC_LEVEL2   = 32,
  VVDEC_LEVEL2_1 = 35,
  VVDEC_LEVEL3   = 48,
  VVDEC_LEVEL3_1 = 51,
  VVDEC_LEVEL4   = 64,
  VVDEC_LEVEL4_1 = 67,
  VVDEC_LEVEL5   = 80,
  VVDEC_LEVEL5_1 = 83,
  VVDEC_LEVEL5_2 = 86,
  VVDEC_LEVEL6   = 96,
  VVDEC_LEVEL6_1 = 99,
  VVDEC_LEVEL6_2 = 102,
  VVDEC_LEVEL15_5 = 255,
}vvdecLevel;


typedef enum
{
  VVDEC_HASHTYPE_MD5             = 0,
  VVDEC_HASHTYPE_CRC             = 1,
  VVDEC_HASHTYPE_CHECKSUM        = 2,
  VVDEC_HASHTYPE_NONE            = 3,
  VVDEC_NUMBER_OF_HASHTYPES      = 4
}vvdecHashType;

/* vvdecSEI
  The struct vvdecSEI contains the payload of a SEI message.
  To get the data from the payload the SEI has to be type casted into target type
  e.g.
  vvdecSEIBufferingPeriod* s = reinterpret_cast<vvdecSEIBufferingPeriod *>(sei->payload);
*/
typedef struct vvdecSEI
{
  vvdecSEIPayloadType  payloadType;     /* payload type as defined in sei.h */
  unsigned int         size;            /* size of payload in bytes */
  void                *payload;         /* payload structure as defined in sei.h */
}vvdecSEI;

typedef struct vvdecSEIBufferingPeriod
{
  bool        bpNalCpbParamsPresentFlag;
  bool        bpVclCpbParamsPresentFlag;
  uint32_t    initialCpbRemovalDelayLength;
  uint32_t    cpbRemovalDelayLength;
  uint32_t    dpbOutputDelayLength;
  int         bpCpbCnt;
  uint32_t    duCpbRemovalDelayIncrementLength;
  uint32_t    dpbOutputDelayDuLength;
  uint32_t    initialCpbRemovalDelay [7][32][2];
  uint32_t    initialCpbRemovalOffset[7][32][2];
  bool        concatenationFlag;
  uint32_t    auCpbRemovalDelayDelta;
  bool        cpbRemovalDelayDeltasPresentFlag;
  int         numCpbRemovalDelayDeltas;
  int         bpMaxSubLayers;
  uint32_t    cpbRemovalDelayDelta[15];
  bool        bpDecodingUnitHrdParamsPresentFlag;
  bool        decodingUnitCpbParamsInPicTimingSeiFlag;
  bool        decodingUnitDpbDuParamsInPicTimingSeiFlag;
  bool        sublayerInitialCpbRemovalDelayPresentFlag;
  bool        additionalConcatenationInfoPresentFlag;
  uint32_t    maxInitialRemovalDelayForConcatenation;
  bool        sublayerDpbOutputOffsetsPresentFlag;
  uint32_t    dpbOutputTidOffset[7];
  bool        altCpbParamsPresentFlag;
  bool        useAltCpbParamsFlag;
} vvdecSEIBufferingPeriod;


typedef struct vvdecSEIPictureTiming
{
  bool        ptSubLayerDelaysPresentFlag[7];
  bool        cpbRemovalDelayDeltaEnabledFlag[7];
  uint32_t    cpbRemovalDelayDeltaIdx[7];
  uint32_t    auCpbRemovalDelay[7];
  uint32_t    picDpbOutputDelay;
  uint32_t    picDpbOutputDuDelay;
  uint32_t    numDecodingUnits;
  bool        duCommonCpbRemovalDelayFlag;
  uint32_t    duCommonCpbRemovalDelay[7];
  uint32_t    numNalusInDu[32];
  uint32_t    duCpbRemovalDelay[32*7+7];
  bool        cpbAltTimingInfoPresentFlag;
  uint32_t    nalCpbAltInitialRemovalDelayDelta[7][32];
  uint32_t    nalCpbAltInitialRemovalOffsetDelta[7][32];
  uint32_t    nalCpbDelayOffset[7];
  uint32_t    nalDpbDelayOffset[7];
  uint32_t    vclCpbAltInitialRemovalDelayDelta[7][32];
  uint32_t    vclCpbAltInitialRemovalOffsetDelta[7][32];
  uint32_t    vclCpbDelayOffset[7];
  uint32_t    vclDpbDelayOffset[7];
  int         ptDisplayElementalPeriods;
} vvdecSEIPictureTiming;


typedef struct vvdecSEIUserDataRegistered
{
  uint16_t    ituCountryCode;
  uint32_t    userDataLength;
  uint8_t    *userData;
} vvdecSEIUserDataRegistered;

typedef struct vvdecSEIUserDataUnregistered
{
  uint8_t     uuid_iso_iec_11578[16];
  uint32_t    userDataLength;
  uint8_t    *userData;
} vvdecSEIUserDataUnregistered;


typedef struct vvdecCompModelIntensityValues
{
  uint8_t     intensityIntervalLowerBound;
  uint8_t     intensityIntervalUpperBound;
  int         compModelValue[6];
}vvdecCompModelIntensityValues;

typedef struct vvdecCompModel
{
  bool                          presentFlag;
  uint8_t                       numModelValues;
  uint16_t                      numIntensityIntervals;
  vvdecCompModelIntensityValues intensityValues[256];
}vvdecCompModel;

typedef struct vvdecSEIFilmGrainCharacteristics
{
  bool             filmGrainCharacteristicsCancelFlag;
  uint8_t          filmGrainModelId;
  bool             separateColourDescriptionPresentFlag;
  uint8_t          filmGrainBitDepthLuma;
  uint8_t          filmGrainBitDepthChroma;
  bool             filmGrainFullRangeFlag;
  uint8_t          filmGrainColourPrimaries;
  uint8_t          filmGrainTransferCharacteristics;
  uint8_t          filmGrainMatrixCoeffs;
  uint8_t          blendingModeId;
  uint8_t          log2ScaleFactor;
  vvdecCompModel   compModel[3];
  bool             filmGrainCharacteristicsPersistenceFlag;
}vvdecSEIFilmGrainCharacteristics;

typedef struct vvdecSEIFramePacking
{
  int         arrangementId;
  bool        arrangementCancelFlag;
  int         arrangementType;
  bool        quincunxSamplingFlag;
  int         contentInterpretationType;
  bool        spatialFlippingFlag;
  bool        frame0FlippedFlag;
  bool        fieldViewsFlag;
  bool        currentFrameIsFrame0Flag;
  bool        frame0SelfContainedFlag;
  bool        frame1SelfContainedFlag;
  int         frame0GridPositionX;
  int         frame0GridPositionY;
  int         frame1GridPositionX;
  int         frame1GridPositionY;
  int         arrangementReservedByte;
  bool        arrangementPersistenceFlag;
  bool        upsampledAspectRatio;
}vvdecSEIFramePacking;

typedef struct vvdecSEIParameterSetsInclusionIndication
{
  int         selfContainedClvsFlag;
}vvdecSEIParameterSetsInclusionIndication;

typedef struct vvdecSEIDecodingUnitInfo
{
  int         decodingUnitIdx;
  bool        duiSubLayerDelaysPresentFlag[7];
  int         duSptCpbRemovalDelayIncrement[7];
  bool        dpbOutputDuDelayPresentFlag;
  int         picSptDpbOutputDuDelay;
}vvdecSEIDecodingUnitInfo;


typedef struct vvdecSEIDecodedPictureHash
{
  vvdecHashType method;
  bool          singleCompFlag;
  int           digest_length;
  unsigned char digest[16*3];
}vvdecSEIDecodedPictureHash;


typedef struct vvdecSEIScalableNesting
{
  bool        snOlsFlag;
  bool        snSubpicFlag;
  uint32_t    snNumOlss;
  uint32_t    snOlsIdxDelta[64];
  uint32_t    snOlsIdx[64];
  bool        snAllLayersFlag;
  uint32_t    snNumLayers;
  uint8_t     snLayerId[64];
  uint32_t    snNumSubpics;
  uint8_t     snSubpicIdLen;
  uint16_t    snSubpicId[64];
  uint32_t    snNumSEIs;

  vvdecSEI* nestedSEIs[64];
}vvdecSEIScalableNesting;

typedef struct vvdecSEIMasteringDisplayColourVolume
{
  uint32_t    maxLuminance;
  uint32_t    minLuminance;
  uint16_t    primaries[3][2];
  uint16_t    whitePoint[2];
}vvdecSEIMasteringDisplayColourVolume;

typedef struct vvdecSEIDependentRapIndication
{
} vvdecSEIDependentRapIndication;

typedef struct vvdecSEIEquirectangularProjection
{
  bool        erpCancelFlag;
  bool        erpPersistenceFlag;
  bool        erpGuardBandFlag;
  uint8_t     erpGuardBandType;
  uint8_t     erpLeftGuardBandWidth;
  uint8_t     erpRightGuardBandWidth;
}vvdecSEIEquirectangularProjection;

typedef struct vvdecSEISphereRotation
{
  bool        sphereRotationCancelFlag;
  bool        sphereRotationPersistenceFlag;
  int         sphereRotationYaw;
  int         sphereRotationPitch;
  int         sphereRotationRoll;
}vvdecSEISphereRotation;

typedef struct vvdecSEIRegionWisePacking
{
  bool        rwpCancelFlag;
  bool        rwpPersistenceFlag;
  bool        constituentPictureMatchingFlag;
  int         numPackedRegions;
  int         projPictureWidth;
  int         projPictureHeight;
  int         packedPictureWidth;
  int         packedPictureHeight;
  uint8_t     rwpTransformType[256];
  bool        rwpGuardBandFlag[256];
  uint32_t    projRegionWidth[256];
  uint32_t    projRegionHeight[256];
  uint32_t    rwpProjRegionTop[256];
  uint32_t    projRegionLeft[256];
  uint16_t    packedRegionWidth[256];
  uint16_t    packedRegionHeight[256];
  uint16_t    packedRegionTop[256];
  uint16_t    packedRegionLeft[256];
  uint8_t     rwpLeftGuardBandWidth[256];
  uint8_t     rwpRightGuardBandWidth[256];
  uint8_t     rwpTopGuardBandHeight[256];
  uint8_t     rwpBottomGuardBandHeight[256];
  bool        rwpGuardBandNotUsedForPredFlag[256];
  uint8_t     rwpGuardBandType[4*256];
}vvdecSEIRegionWisePacking;

typedef struct vvdecOmniViewportRegion
{
  int         azimuthCentre;
  int         elevationCentre;
  int         tiltCentre;
  uint32_t    horRange;
  uint32_t    verRange;
}vvdecOmniViewportRegion;

typedef struct vvdecSEIOmniViewport
{
  uint32_t                omniViewportId;
  bool                    omniViewportCancelFlag;
  bool                    omniViewportPersistenceFlag;
  uint8_t                 omniViewportCnt;
  vvdecOmniViewportRegion omniViewportRegions[16];
}vvdecSEIOmniViewport;


typedef struct vvdecSEIGeneralizedCubemapProjection
{
  bool        gcmpCancelFlag;
  bool        gcmpPersistenceFlag;
  uint8_t     gcmpPackingType;
  uint8_t     gcmpMappingFunctionType;
  uint8_t     gcmpFaceIndex[6];
  uint8_t     gcmpFaceRotation[6];
  uint8_t     gcmpFunctionCoeffU[6];
  bool        gcmpFunctionUAffectedByVFlag[6];
  uint8_t     gcmpFunctionCoeffV[6];
  bool        gcmpFunctionVAffectedByUFlag[6];
  bool        gcmpGuardBandFlag;
  uint8_t     gcmpGuardBandType;
  bool        gcmpGuardBandBoundaryExteriorFlag;
  uint8_t     gcmpGuardBandSamples;
}vvdecSEIGeneralizedCubemapProjection;

typedef struct vvdecSEIFrameFieldInfo
{
  bool        fieldPicFlag;
  bool        bottomFieldFlag;
  bool        pairingIndicatedFlag;
  bool        pairedWithNextFieldFlag;
  bool        displayFieldsFromFrameFlag;
  bool        topFieldFirstFlag;
  int         displayElementalPeriods;
  int         sourceScanType;
  bool        duplicateFlag;
}vvdecSEIFrameFieldInfo;

typedef struct vvdecSEISubpictureLevelInfo
{
  int         numRefLevels;
  bool        explicitFractionPresentFlag;
  bool        cbrConstraintFlag;
  int         numSubpics;
  int         sliMaxSublayers;
  bool        sliSublayerInfoPresentFlag;
  int         nonSubpicLayersFraction[6][6];
  vvdecLevel  refLevelIdc[6][6];
  int         refLevelFraction[6][64][6];
}vvdecSEISubpictureLevelInfo;

typedef struct vvdecSEISampleAspectRatioInfo
{
  bool        sariCancelFlag;
  bool        sariPersistenceFlag;
  int         sariAspectRatioIdc;
  int         sariSarWidth;
  int         sariSarHeight;
}vvdecSEISampleAspectRatioInfo;

typedef struct vvdecSEIContentLightLevelInfo
{
  uint16_t    maxContentLightLevel;
  uint16_t    maxPicAverageLightLevel;
}vvdecSEIContentLightLevelInfo;

typedef struct vvdecSEIAlternativeTransferCharacteristics
{
  uint8_t     preferred_transfer_characteristics;
}vvdecSEIAlternativeTransferCharacteristics;

typedef struct vvdecSEIAmbientViewingEnvironment
{
  uint32_t    ambientIlluminance;
  uint16_t    ambientLightX;
  uint16_t    ambientLightY;
}vvdecSEIAmbientViewingEnvironment;

typedef struct vvdecSEIContentColourVolume
{
  bool        ccvCancelFlag;
  bool        ccvPersistenceFlag;
  bool        ccvPrimariesPresentFlag;
  bool        ccvMinLuminanceValuePresentFlag;
  bool        ccvMaxLuminanceValuePresentFlag;
  bool        ccvAvgLuminanceValuePresentFlag;
  int         ccvPrimariesX[3];
  int         ccvPrimariesY[3];
  uint32_t    ccvMinLuminanceValue;
  uint32_t    ccvMaxLuminanceValue;
  uint32_t    ccvAvgLuminanceValue;
}vvdecSEIContentColourVolume;

/**
 * \brief SEI message containing characteristics of a neural network post-filter.
 *
 * This message specifies the properties and configuration of a neural network
 * that can be used as a post-processing filter on the decoded video.
 * It can define a new filter or update an existing one.
 */
typedef struct vvdecSEINeuralNetworkPostFilterCharacteristics
{
  // --- NNPFC Top-Level Information ---
  uint16_t m_purpose;                                    ///< (nnpfc_purpose) u(16): A bitmask indicating the intended use(s) of the filter (e.g., super-resolution, denoising, etc.).
  uint32_t m_id;                                         ///< (nnpfc_id) ue(v): Unique identifier for the post-processing filter.
  bool     m_baseFlag;                                   ///< (nnpfc_base_flag) u(1): 1 indicates this is a base filter definition. 0 indicates an update to an existing filter.
  uint32_t m_modeIdc;                                    ///< (nnpfc_mode_idc) ue(v): Specifies how the filter is defined (e.g., via URI or embedded payload).
  char*    m_uri;                                        ///< (nnpfc_uri) st(v): URI pointing to the neural network definition if m_modeIdc is 1.
  char*    m_tagUri;                                     ///< (nnpfc_tag_uri) st(v): URI identifying the format of the data at m_uri.
  // Presence flags for optional top-level fields
  bool     m_uriParsedFlag;                              ///< Parser-only (internal): true if `m_uri` was parsed from the bitstream; not part of the SEI spec and not emitted in JSON
  bool     m_tagUriParsedFlag;                           ///< Parser-only (internal): true if `m_tagUri` was parsed from the bitstream; not part of the SEI spec and not emitted in JSON
  
  
  // --- Property Flag and Conditional Block ---
  bool     m_propertyPresentFlag;                        ///< (nnpfc_property_present_flag) u(1): If 1, formatting, purpose, and complexity info are present.

  // --- Input and Output Formatting Properties (Present if m_propertyPresentFlag is true) ---
  uint32_t m_numInputPicsMinus1;                         ///< (nnpfc_num_input_pics_minus1) ue(v): Specifies the number of decoded pictures used as input to the filter (Value is N-1).
  bool*    m_inputPicFilteringFlag;                      ///< (nnpfc_input_pic_filtering_flag[i]) u(1): Array of flags. 1 indicates the i-th input picture is filtered.
  // Presence flags for property sub-fields
  bool     m_numInputPicsMinus1ParsedFlag;               ///< Parser-only (internal): true if `m_numInputPicsMinus1` was parsed from the bitstream
  bool     m_inputPicFilteringParsedFlag;                ///< Parser-only (internal): true if `m_inputPicFilteringFlag` array was allocated and parsed
  bool     m_absentInputPicZeroParsedFlag;               ///< Parser-only (internal): true if `m_absentInputPicZeroFlag` was parsed from the bitstream
  
  bool     m_absentInputPicZeroFlag;                     ///< (nnpfc_absent_input_pic_zero_flag) u(1): Specifies how to handle input pictures not present in the bitstream.
  
  // --- Purpose-Dependent Formatting ---
  bool     m_outSubCFlag;                                ///< (nnpfc_out_sub_c_flag) u(1): Specifies chroma subsampling for the output (present for chroma upsampling purpose).
  uint32_t m_outColourFormatIdc;                         ///< (nnpfc_out_colour_format_idc) u(2): Specifies output color format (present for colourization purpose).
  uint32_t m_picWidthNumMinus1;                          ///< (nnpfc_pic_width_num_minus1) ue(v): Numerator for output picture width resampling ratio.
  uint32_t m_picWidthDenomMinus1;                        ///< (nnpfc_pic_width_denom_minus1) ue(v): Denominator for output picture width resampling ratio.
  uint32_t m_picHeightNumMinus1;                         ///< (nnpfc_pic_height_num_minus1) ue(v): Numerator for output picture height resampling ratio.
  uint32_t m_picHeightDenomMinus1;                       ///< (nnpfc_pic_height_denom_minus1) ue(v): Denominator for output picture height resampling ratio.
  uint32_t* m_interpolatedPics;                          ///< (nnpfc_interpolated_pics[i]) ue(v): Number of pictures to interpolate between input pictures.
  // Presence flags for purpose-dependent fields
  bool     m_outSubCParsedFlag;                          ///< Parser-only (internal): true if `m_outSubCFlag` was parsed from the bitstream
  bool     m_outColourFormatIdcParsedFlag;               ///< Parser-only (internal): true if `m_outColourFormatIdc` was parsed from the bitstream
  bool     m_resolutionResamplingParsedFlag;             ///< Parser-only (internal): true if resolution resampling fields were parsed
  bool     m_interpolatedPicsParsedFlag;                 ///< Parser-only (internal): true if `m_interpolatedPics` array was allocated and parsed
  
  
  // --- Tensor and Color Formatting ---
  bool     m_componentLastFlag;                          ///< (nnpfc_component_last_flag) u(1): Specifies tensor dimension order (e.g., CHW vs HWC).
  uint32_t m_inpFormatIdc;                               ///< (nnpfc_inp_format_idc) ue(v): Specifies the data type and range of input tensor values.
  uint32_t m_auxiliaryInpIdc;                            ///< (nnpfc_auxiliary_inp_idc) ue(v): Indicates if auxiliary input data is used.
  uint32_t m_inpOrderIdc;                                ///< (nnpfc_inp_order_idc) ue(v): Specifies the ordering of color components in the input tensor.
  uint32_t m_inpTensorLumaBitdepthMinus8;                ///< (nnpfc_inp_tensor_luma_bitdepth_minus8) ue(v): Bit depth of input luma samples for integer tensors (Value is D-8).
  uint32_t m_inpTensorChromaBitdepthMinus8;              ///< (nnpfc_inp_tensor_chroma_bitdepth_minus8) ue(v): Bit depth of input chroma samples for integer tensors (Value is D-8).
  uint32_t m_outFormatIdc;                               ///< (nnpfc_out_format_idc) ue(v): Specifies the data type and range of output tensor values.
  uint32_t m_outOrderIdc;                                ///< (nnpfc_out_order_idc) ue(v): Specifies the ordering of color components in the output tensor.
  uint32_t m_outTensorLumaBitdepthMinus8;                ///< (nnpfc_out_tensor_luma_bitdepth_minus8) ue(v): Bit depth of output luma samples for integer tensors (Value is D-8).
  uint32_t m_outTensorChromaBitdepthMinus8;              ///< (nnpfc_out_tensor_chroma_bitdepth_minus8) ue(v): Bit depth of output chroma samples for integer tensors (Value is D-8).
  // Presence flags for tensor bitdepths
  bool     m_inpTensorLumaBitdepthParsedFlag;            ///< Parser-only (internal): true if input tensor luma bitdepth was parsed
  bool     m_inpTensorChromaBitdepthParsedFlag;          ///< Parser-only (internal): true if input tensor chroma bitdepth was parsed
  bool     m_outTensorLumaBitdepthParsedFlag;            ///< Parser-only (internal): true if output tensor luma bitdepth was parsed
  bool     m_outTensorChromaBitdepthParsedFlag;          ///< Parser-only (internal): true if output tensor chroma bitdepth was parsed

  // --- Separate Color Description ---
  bool     m_separateColourDescriptionPresentFlag;       ///< (nnpfc_separate_colour_description_present_flag) u(1): If 1, custom color space info for the output is present.
  uint8_t  m_colourPrimaries;                            ///< (nnpfc_colour_primaries) u(8): Output colour primaries.
  uint8_t  m_transferCharacteristics;                    ///< (nnpfc_transfer_characteristics) u(8): Output transfer characteristics.
  uint8_t  m_matrixCoeffs;                               ///< (nnpfc_matrix_coeffs) u(8): Output matrix coefficients.
  bool     m_fullRangeFlag;                              ///< (nnpfc_full_range_flag) u(1): Output full range flag.
  // Presence flags for colour description fields
  bool     m_colourPrimariesParsedFlag;                  ///< Parser-only (internal): true if colour_primaries was parsed
  bool     m_transferCharacteristicsParsedFlag;          ///< Parser-only (internal): true if transfer_characteristics was parsed
  bool     m_matrixCoeffsParsedFlag;                     ///< Parser-only (internal): true if matrix_coeffs was parsed
  bool     m_fullRangeFlagParsedFlag;                    ///< Parser-only (internal): true if full_range_flag was parsed
  
  // --- Chroma Location and Patch Information ---
  bool     m_chromaLocInfoPresentFlag;                   ///< (nnpfc_chroma_loc_info_present_flag) u(1): If 1, chroma sample location type is present.
  uint32_t m_chromaSampleLocTypeFrame;                   ///< (nnpfc_chroma_sample_loc_type_frame) ue(v): Specifies the location of chroma samples.
  uint32_t m_overlap;                                    ///< (nnpfc_overlap) ue(v): Number of overlapping samples between adjacent patches.
  bool     m_constantPatchSizeFlag;                      ///< (nnpfc_constant_patch_size_flag) u(1): If 1, the filter uses a fixed patch size.
  uint32_t m_patchWidthMinus1;                           ///< (nnpfc_patch_width_minus1) ue(v): Horizontal size of processing patches (Value is S-1).
  uint32_t m_patchHeightMinus1;                          ///< (nnpfc_patch_height_minus1) ue(v): Vertical size of processing patches (Value is S-1).
  uint32_t m_extendedPatchWidthCdDeltaMinus1;            ///< (nnpfc_extended_patch_width_cd_delta_minus1) ue(v): Extended horizontal patch size for chroma.
  uint32_t m_extendedPatchHeightCdDeltaMinus1;           ///< (nnpfc_extended_patch_height_cd_delta_minus1) ue(v): Extended vertical patch size for chroma.
  uint32_t m_paddingType;                                ///< (nnpfc_padding_type) ue(v): Specifies the type of padding to use at picture boundaries.
  uint32_t m_lumaPaddingVal;                             ///< (nnpfc_luma_padding_val) ue(v): Luma value for constant padding.
  uint32_t m_cbPaddingVal;                               ///< (nnpfc_cb_padding_val) ue(v): Cb value for constant padding.
  uint32_t m_crPaddingVal;                               ///< (nnpfc_cr_padding_val) ue(v): Cr value for constant padding.

  // --- Complexity Information ---
  bool     m_complexityInfoPresentFlag;                  ///< (nnpfc_complexity_info_present_flag) u(1): If 1, computational complexity information is present.
  uint8_t  m_parameterTypeIdc;                           ///< (nnpfc_parameter_type_idc) u(2): Specifies how model parameters are represented (e.g., float, integer).
  uint8_t  m_log2ParameterBitLengthMinus3;               ///< (nnpfc_log2_parameter_bit_length_minus3) u(2): Bit length of quantized parameters.
  uint8_t  m_numParametersIdc;                           ///< (nnpfc_num_parameters_idc) u(6): Maximum number of filter parameters.
  uint32_t m_numKmacOperationsIdc;                       ///< (nnpfc_num_kmac_operations_idc) ue(v): Maximum number of kilo multiply-accumulate operations per sample.
  uint32_t m_totalKilobyteSize;                          ///< (nnpfc_total_kilobyte_size) ue(v): Size in kilobytes required to store uncompressed filter parameters.
  // Presence flags for complexity info sub-fields
  bool     m_parameterTypeIdcParsedFlag;                 ///< Parser-only (internal): true if parameter_type_idc was parsed
  bool     m_log2ParameterBitLengthMinus3ParsedFlag;     ///< Parser-only (internal): true if log2_parameter_bit_length_minus3 was parsed
  bool     m_numParametersIdcParsedFlag;                 ///< Parser-only (internal): true if num_parameters_idc was parsed
  bool     m_numKmacOperationsIdcParsedFlag;            ///< Parser-only (internal): true if num_kmac_operations_idc was parsed
  bool     m_totalKilobyteSizeParsedFlag;                ///< Parser-only (internal): true if total_kilobyte_size was parsed

  // --- Metadata and Payload ---
  uint32_t m_numMetadataExtensionBits;                   ///< (nnpfc_num_metadata_extension_bits) ue(v): Number of bits for metadata extension.
  uint8_t* m_reservedMetadataExtension;                  ///< (nnpfc_reserved_metadata_extension) u(v): Reserved bits for future metadata.
  uint8_t* m_payload;                                    ///< (nnpfc_payload_byte) b(8): Raw payload containing the filter definition if not specified by URI.
  uint32_t m_payloadSize;                                ///< Size of the raw payload in bytes.
  // Presence flags for metadata/payload
  bool     m_reservedMetadataExtensionParsedFlag;        ///< Parser-only (internal): true if reserved metadata extension bits were parsed
  bool     m_payloadParsedFlag;                          ///< Parser-only (internal): true if `m_payload` and `m_payloadSize` were parsed
  
  // --- Derived Flags (not parsed from bitstream) ---
  bool m_purposeChromaUpsampling;
  bool m_purposeColourization;
  bool m_purposeResolutionResampling;
  bool m_purposePictureRateUpsampling;

} vvdecSEINeuralNetworkPostFilterCharacteristics;

/**
 * \brief SEI message providing activation instructions for a neural network post-filter.
 *
 * This message is used to activate, deactivate, or control the persistence of a
 * neural network post-filter that has been previously defined by a
 * vvdecSEINeuralNetworkPostFilterCharacteristics message. It links to a specific
 * filter definition via its ID.
 */
typedef struct vvdecSEINeuralNetworkPostFilterActivation
{
  bool     m_present;            ///< Indicates if the SEI message is present in the current access unit.
  uint32_t m_id;                 ///< (nnpfa_target_id) The ID of the NNPF characteristics message that defines the filter to be activated/deactivated.
  bool     m_cancelFlag;         ///< (nnpfa_cancel_flag) A value of 1 cancels the persistence of the filter, deactivating it for subsequent pictures unless re-activated.
  bool     m_persistenceFlag;    ///< (nnpfa_persistence_flag) A value of 1 specifies that the filter activation persists across pictures until explicitly cancelled or a new CLVS begins.
  bool     m_targetBaseFlag;     ///< (nnpfa_target_base_flag) When 1, indicates that the target filter is a base version. When 0, it's an updated version.
  bool     m_noPrevClvsFlag;     ///< (nnpfa_no_prev_clvs_flag) When 1, indicates the filter should not be applied to pictures from a previous CLVS that are used as reference.
  bool     m_noFollClvsFlag;     ///< (nnpfa_no_foll_clvs_flag) When 1, indicates the filter should not be applied to pictures from a following CLVS that are used for temporal processing.
  uint32_t m_numOutputEntries;   ///< (nnpfa_num_output_entries) Specifies the number of output entries. This value determines the size of the m_outputFlag array.
  bool*    m_outputFlag;         ///< (nnpfa_output_flag[i]) Array of flags associated with each output entry. The semantics of the flag depend on the specific filter definition.
  // Presence flags for activation sub-fields
  bool     m_numOutputEntriesPresentFlag;                ///< True if `m_numOutputEntries` was parsed
  bool     m_outputFlagPresentFlag;                      ///< True if `m_outputFlag` array was present and parsed
} vvdecSEINeuralNetworkPostFilterActivation;

#endif /*VVDEC_SEI_H*/
