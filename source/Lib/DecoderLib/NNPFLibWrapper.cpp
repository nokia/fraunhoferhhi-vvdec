#include "NNPFLibWrapper.h"
#include "NNPF/nnpf.h"
#include "CommonLib/Picture.h"
#include "CommonLib/SEI_internal.h"
#include "CommonLib/dtrace_next.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace vvdec
{
  // Define static member variable
  bool NnpfLibWrapper::s_backupAttempted = false;

  NnpfLibWrapper::NnpfFilterState::NnpfFilterState(const char *model_path, int width, int height, const char *weights_update_path)
    : modelPath(model_path ? model_path : "model3.onnx"),
      weightsUpdatePath(weights_update_path ? weights_update_path : ""),
      width(width),
      height(height),
      hasConfiguration(true)
  {
    // Don't create instance yet - wait until NNPFA activation
    if (!weightsUpdatePath.empty())
    {
      msg(VERBOSE, "NNPFC configuration stored (model=%s, weights=%s, %dx%d), deferring instance creation until NNPFA activation.\n",
          this->modelPath.c_str(), this->weightsUpdatePath.c_str(), width, height);
    }
    else
    {
      msg(VERBOSE, "NNPFC configuration stored (model=%s, %dx%d), deferring instance creation until NNPFA activation.\n",
          this->modelPath.c_str(), width, height);
    }
  }

  NnpfLibWrapper::NnpfFilterState::~NnpfFilterState()
  {
    if (instance)
    {
      nnpf_destroy(instance);
      instance = nullptr;
    }
  }

  void NnpfLibWrapper::NnpfFilterState::ensureInstanceCreated()
  {
    if (instance || !hasConfiguration)
    {
      return; // Already created or no configuration available
    }

    const char* weightsUpdate = weightsUpdatePath.empty() ? nullptr : weightsUpdatePath.c_str();
    
    try
    {
      instance = nnpf_create(width, height, modelPath.c_str(), weightsUpdate);
    }
    catch (...)
    {
      instance = nullptr;
    }

    if (!instance)
    {
      if (weightsUpdate)
      {
        msg(WARNING, "nnpf_create failed for model %s with update %s.\n", modelPath.c_str(), weightsUpdate);
      }
      else
      {
        msg(WARNING, "nnpf_create failed for model %s.\n", modelPath.c_str());
      }
    }
    else
    {
      if (weightsUpdate)
      {
        msg(INFO, "NNPF instance created successfully (id=%d, model=%s, update=%s).\n", id, modelPath.c_str(), weightsUpdate);
      }
      else
      {
        msg(INFO, "NNPF instance created successfully (id=%d, model=%s).\n", id, modelPath.c_str());
      }
    }
  }

  void NnpfLibWrapper::updateFilterStatesFromPic(Picture *pic, bool isNewClvs)
  {
    if (!pic)
    {
      return;
    }

    if (isNewClvs)
    {
      msg(VERBOSE, "New CLVS, resetting persistent NNPFA activations.\n");
      for (auto &ptr : m_filterStates)
      {
        if (!ptr)
          continue;
        ptr->isPersistent = false;
        ptr->isActive = false;
        ptr->pocOfActivation = -1;
      }
    }

    updateFilterStates(pic);
  }

  void NnpfLibWrapper::applyFilterToPic(Picture *pic)
  {
    std::vector<NnpfFilterState *> filtersToApply;
    for (auto &ptr : m_filterStates)
    {
      if (!ptr)
        continue;
      NnpfFilterState &filterState = *ptr;
      bool shouldBeActive = filterState.isActive;

      if (filterState.isActive)
      {
        if (!filterState.isPersistent && filterState.pocOfActivation != pic->poc)
        {
          // Non-persistent activation is only for the picture that contains the SEI
          shouldBeActive = false;
          filterState.isActive = false; // Deactivate it for subsequent pictures
        }
      }

      if (shouldBeActive)
      {
        filtersToApply.push_back(&filterState);
      }
    }

    // Now apply the filters in sequence
    for (auto *filterState : filtersToApply)
    {
      if (!filterState)
      {
        msg(WARNING, "NNPFA requests application of filter (id:%d), but no NNPFC has been received yet!\n", filterState->id);
        continue;
      }

      if (filterState->instance)
      {
        applyFilter(pic, *filterState);
        msg(VERBOSE, "Filter[id:%d; p:%p] applied on frame:%d.\n", filterState->id, filterState->instance, pic->poc);
      }
    }
  }

  void NnpfLibWrapper::updateFilterStates(Picture *pic)
  {
    for (const auto &sei : pic->seiMessageList)
    {
      if (sei->payloadType == VVDEC_NEURAL_NETWORK_POST_FILTER_CHARACTERISTICS)
      {
        const vvdecSEINeuralNetworkPostFilterCharacteristics *nnpfc = static_cast<const vvdecSEINeuralNetworkPostFilterCharacteristics *>(sei->payload);
        msg(VERBOSE, "Found NNPFC SEI message with id %d.\n", nnpfc->m_id);


        // Write binary payload and reinitialize filter if payload exists
        if (nnpfc->m_payloadSize > 0 && nnpfc->m_modeIdc != 1 && !nnpfc->m_baseFlag)
        {
          writeBinaryPayloadAndReinitializeFilter(nnpfc, pic);
        }

        else if (nnpfc->m_modeIdc == 1 && nnpfc->m_uri && nnpfc->m_baseFlag)
        {
          loadModelFromUriAndReinitializeFilter(nnpfc, pic);
        }
      }
      else if (sei->payloadType == VVDEC_NEURAL_NETWORK_POST_FILTER_ACTIVATION)
      {
        const vvdecSEINeuralNetworkPostFilterActivation *nnpfa = static_cast<const vvdecSEINeuralNetworkPostFilterActivation *>(sei->payload);
        msg(VERBOSE, "Found NNPFA SEI message for target id %d (cancel=%d, persist=%d).\n", nnpfa->m_id, nnpfa->m_cancelFlag, nnpfa->m_persistenceFlag);

        if (!m_filterStates[nnpfa->m_id])
        {
          msg(ERROR, "NNPFA SEI message received for filter id %d, but no characteristics have been received for this filter yet!\n", nnpfa->m_id);
          continue;
        }

        auto &state = *m_filterStates[nnpfa->m_id];
        if (nnpfa->m_cancelFlag)
        {
          state.isActive = false;
          state.isPersistent = false;
          state.pocOfActivation = -1; // Reset pocOfActivation on cancel
        }
        else
        {
          // Create the filter instance now upon activation (if not already created)
          state.ensureInstanceCreated();
          
          state.isActive = true;
          state.isPersistent = nnpfa->m_persistenceFlag;
          // Only update pocOfActivation if the current pic's POC is greater than or equal to the stored pocOfActivation
          // or if it's a non-persistent filter (which should always update for the current frame)
          if (state.pocOfActivation == -1 || pic->poc >= state.pocOfActivation || !state.isPersistent)
          {
            state.pocOfActivation = pic->poc;
          }
          state.targetBaseFilter = nnpfa->m_targetBaseFlag;
        }
      }
    }
  }

  void NnpfLibWrapper::loadModelFromUriAndReinitializeFilter(const vvdecSEINeuralNetworkPostFilterCharacteristics *nnpfc, Picture *pic)
  {
    m_filterStates[nnpfc->m_id].reset();

    if (!nnpfc->m_tagUriParsedFlag || !nnpfc->m_tagUri || std::string("tag:onnx.ai,2017:onnx") != nnpfc->m_tagUri)
    {
      msg(WARNING, "Unable to use NNPF %d: unsupported or missing tag URI\n", nnpfc->m_id);
      return;
    }

    if (!nnpfc->m_uriParsedFlag || !nnpfc->m_uri)
    {
      msg(WARNING, "Unable to use NNPF %d: missing URI\n", nnpfc->m_id);
      return;
    }

    static const std::string fileScheme = "file://";
    if (std::string(nnpfc->m_uri).rfind(fileScheme, 0) != 0)
    {
      msg(WARNING, "Unable to use NNPF %d: URI is not a file URI\n", nnpfc->m_id);
      //return;
    }

    if (nnpfc->m_componentLastFlag)
    {
      msg(WARNING, "Unable to use NNPF %d: nnpfc_component_last_flag is 1\n", nnpfc->m_id);
      return;
    }

    if (nnpfc->m_inpOrderIdc != 3)
    {
      msg(WARNING, "Unable to use NNPF %d: unsupported input order (nnpfc_inp_order_idc=%d)\n", nnpfc->m_id, nnpfc->m_inpOrderIdc);
      return;
    }

    if (nnpfc->m_outOrderIdc != 3)
    {
      msg(WARNING, "Unable to use NNPF %d: unsupported output order (nnpfc_out_order_idc=%d)\n", nnpfc->m_id, nnpfc->m_outOrderIdc);
      return;
    }

    if (nnpfc->m_auxiliaryInpIdc != 1)
    {
      msg(WARNING, "Unable to use NNPF %d: unsupported auxiliary input (nnpfc_auxiliary_inp_idc=%d)\n", nnpfc->m_id, nnpfc->m_auxiliaryInpIdc);
      return;
    }

    if (nnpfc->m_inpFormatIdc != 0)
    {
      msg(WARNING, "Unable to use NNPF %d: unsupported input format (nnpfc_inp_format_idc=%d)\n", nnpfc->m_id, nnpfc->m_inpFormatIdc);
      return;
    }

    if (nnpfc->m_outFormatIdc != 0)
    {
      msg(WARNING, "Unable to use NNPF %d: unsupported output format (nnpfc_out_format_idc=%d)\n", nnpfc->m_id, nnpfc->m_outFormatIdc);
      return;
    }

    if (nnpfc->m_purpose & ~(0x01))
    {
      msg(WARNING, "Unable to use NNPF %d: unsupported purpose (nnpfc_purpose=0x%04x)\n", nnpfc->m_id, nnpfc->m_purpose);
      return;
    }

    const std::string filePath = std::string(nnpfc->m_uri).substr(fileScheme.length());


    int width = pic->getRecoBuf().get(COMPONENT_Y).width;
    int height = pic->getRecoBuf().get(COMPONENT_Y).height;

    // For URI-based models, use the URI as the model path
    // URI-based models are typically base models, so no weights update path
    m_filterStates[nnpfc->m_id] = std::make_shared<NnpfFilterState>(filePath.c_str(), width, height, nullptr);
  }

  void NnpfLibWrapper::writeBinaryPayloadAndReinitializeFilter(const vvdecSEINeuralNetworkPostFilterCharacteristics *nnpfc, Picture *pic)
  {
    int width = pic->getRecoBuf().get(COMPONENT_Y).width;
    int height = pic->getRecoBuf().get(COMPONENT_Y).height;

    // Check if this is a base filter
    if (nnpfc->m_baseFlag)
    {
      msg(VERBOSE, "Reinitializing NNPF instance for base filter ID %d with dimensions %dx%d.\n", nnpfc->m_id, width, height);
      m_filterStates[nnpfc->m_id] = std::make_shared<NnpfFilterState>(nnpfc->m_uri, width, height, nullptr);
    }
    else
    {
      // For non-base filters (weights updates), write binary payload to weights update file
      std::ofstream binary_file("wu.bin", std::ios::binary);
      if (!binary_file.is_open())
      {
        std::cerr << "ERROR: Cannot open weights update file for writing: " << "wu.bin" << "." << std::endl;
        return;
      }

      binary_file.write(reinterpret_cast<const char *>(nnpfc->m_payload), nnpfc->m_payloadSize);
      if (binary_file.good())
      {
        msg(INFO, "Writing weights update to: %s (size: %u bytes) for filter ID %d.\n", "wu.bin", nnpfc->m_payloadSize, nnpfc->m_id);
      }
      else
      {
        std::cerr << "ERROR: Failed to write weights update to: " << "wu.bin" << "." << std::endl;
        return;
      }
      
      binary_file.close();

      // Create new instance with base model and weights update
      msg(VERBOSE, "Reinitializing NNPF instance for filter ID %d with dimensions %dx%d (base model + weights update).\n", nnpfc->m_id, width, height);
      m_filterStates[nnpfc->m_id] = std::make_shared<NnpfFilterState>(nnpfc->m_uri, width, height, "wu.bin");
    }
  }

  void NnpfLibWrapper::applyFilter(Picture *pcPic, NnpfFilterState &filterState)
  {
    if (!filterState.instance)
    {
      msg(WARNING, "Cannot apply filter: NNPF instance is null.\n");
      return;
    }

    PelUnitBuf srcBuf = pcPic->getIsFiltered() ? pcPic->getBuf(PIC_FILTERED) : pcPic->getRecoBuf();

    // Y Plane
    auto yPicPlane = srcBuf.Y();
    uint16_t *ySrcPic = reinterpret_cast<uint16_t *>(yPicPlane.buf);
    const int yPicWidth = yPicPlane.width;
    const int yPicHeight = yPicPlane.height;
    const int yNumOfRowBytes = yPicWidth * sizeof(uint16_t);

    // u Plane
    auto uPicPlane = srcBuf.Cb();
    uint16_t *uSrcPic = reinterpret_cast<uint16_t *>(uPicPlane.buf);
    const int uPicWidth = uPicPlane.width;
    const int uPicHeight = uPicPlane.height;
    const int uNumOfRowBytes = uPicWidth * sizeof(uint16_t);

    // v Plane
    auto vPicPlane = srcBuf.Cr();
    uint16_t *vSrcPic = reinterpret_cast<uint16_t *>(vPicPlane.buf);
    const int vPicWidth = vPicPlane.width;
    const int vPicHeight = vPicPlane.height;
    const int vNumOfRowBytes = vPicWidth * sizeof(uint16_t);

    // msg(INFO, "[applyNeuralNetPostfilter] Y: Width: %d. Height: %d, Stride: %d, yPicDstStrideElements:%d \n",yPicWidth,yPicHeight,yPicPlane.stride, yPicDstStrideElements);
    // msg(INFO, "[applyNeuralNetPostfilter] u: Width: %d. Height: %d, Stride: %d, yPicDstStrideElements:%d \n",uPicWidth,uPicHeight,uPicPlane.stride, uPicDstStrideElements);
    // msg(INFO, "[applyNeuralNetPostfilter] v: Width: %d. Height: %d, Stride: %d, yPicDstStrideElements:%d \n",vPicWidth,vPicHeight,vPicPlane.stride, vPicDstStrideElements);

    std::vector<uint16_t> input[3];
    input[0].resize(yPicPlane.stride * yPicHeight);
    input[1].resize(uPicPlane.stride * uPicHeight);
    input[2].resize(vPicPlane.stride * vPicHeight);

    // Init buffers with source picture
    for (int r = 0; r < yPicHeight; r++)
    {
      memcpy(input[0].data() + r * yPicPlane.stride, ySrcPic + r * yPicPlane.stride, yNumOfRowBytes);
    }
    for (int r = 0; r < uPicHeight; r++)
    {
      memcpy(input[1].data() + r * uPicPlane.stride, uSrcPic + r * uPicPlane.stride, uNumOfRowBytes);
    }
    for (int r = 0; r < vPicHeight; r++)
    {
      memcpy(input[2].data() + r * vPicPlane.stride, vSrcPic + r * vPicPlane.stride, vNumOfRowBytes);
    }

    std::vector<uint16_t> output[3];
    output[0].resize(yPicPlane.stride * yPicHeight);
    output[1].resize(uPicPlane.stride * uPicHeight);
    output[2].resize(vPicPlane.stride * vPicHeight);

    uint16_t *in[] = {
        input[0].data(),
        input[1].data(),
        input[2].data(),
    };

    uint16_t *out[] = {
        output[0].data(),
        output[1].data(),
        output[2].data(),
    };

    int inStride[3] = {static_cast<int>(yPicPlane.stride), static_cast<int>(uPicPlane.stride), static_cast<int>(vPicPlane.stride)};
    int outStride[3] = {static_cast<int>(yPicPlane.stride), static_cast<int>(uPicPlane.stride), static_cast<int>(vPicPlane.stride)};

    nnpf_process_picture(filterState.instance, in, inStride, out, outStride, 1.f);

    // Get the filtered buffer as the destination for the copy
    PelUnitBuf filteredBuf = pcPic->getBuf(PIC_FILTERED);

    // Copy the filtered data from the temporary output vectors to the filtered buffer
    for (int r = 0; r < yPicHeight; r++)
    {
      memcpy(filteredBuf.Y().buf + r * filteredBuf.Y().stride, output[0].data() + r * yPicPlane.stride, yNumOfRowBytes);
    }
    for (int r = 0; r < uPicHeight; r++)
    {
      memcpy(filteredBuf.Cb().buf + r * filteredBuf.Cb().stride, output[1].data() + r * uPicPlane.stride, uNumOfRowBytes);
    }
    for (int r = 0; r < vPicHeight; r++)
    {
      memcpy(filteredBuf.Cr().buf + r * filteredBuf.Cr().stride, output[2].data() + r * vPicPlane.stride, vNumOfRowBytes);
    }

    pcPic->setIsFiltered(true);

    // msg( VERBOSE, "Filter (id:%d) applied to frame:%d.\n" , filterState.id, pcPic->poc);
  }

} // namespace vvdec
