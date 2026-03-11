#pragma once

#include "NNPF/nnpf.h"
#include <vector>
#include <array>
#include <memory>
#include <string>
#include "vvdec/sei.h"

namespace vvdec
{
class Picture;

struct NnpfLibWrapper
{
  void updateFilterStatesFromPic( Picture* pic, bool isNewClvs );
  void applyFilterToPic( Picture* pic );

  struct NnpfFilterState
  {
    void*           instance = nullptr;
    int             id = -1;
    bool            isActive = false;
    bool            isPersistent = false;
    bool            targetBaseFilter = false;
    int             pocOfActivation = -1;

    // Configuration stored from NNPFC, used to create instance on NNPFA activation
    std::string     modelPath;        // Base model file path
    std::string     weightsUpdatePath; // Weights update file path (optional)
    int             width = 0;
    int             height = 0;
    bool            hasConfiguration = false;

    NnpfFilterState( const char* model_path, int width, int height, const char* weights_update_path = nullptr );
    ~NnpfFilterState();
    
    // Create the instance if not already created
    void ensureInstanceCreated();

    // Non-copyable and non-movable
    NnpfFilterState( const NnpfFilterState& ) = delete;
    NnpfFilterState& operator=( const NnpfFilterState& ) = delete;
    NnpfFilterState( NnpfFilterState&& ) = delete;
    NnpfFilterState& operator=( NnpfFilterState&& ) = delete;
  };

  void create( NnpfFilterState& filterState, const char *model_path, int width, int height );
  void destroy( NnpfFilterState& filterState );

  void updateFilterStates( Picture* pic );
  void applyFilter( Picture* pic, NnpfFilterState& filterState );
  void writeBinaryPayloadAndReinitializeFilter( const vvdecSEINeuralNetworkPostFilterCharacteristics* nnpfc, Picture* pic );
  void loadModelFromUriAndReinitializeFilter( const vvdecSEINeuralNetworkPostFilterCharacteristics* nnpfc, Picture* pic );

  std::array<std::shared_ptr<NnpfFilterState>, 256> m_filterStates;
    
  // Static flag to track if backup has been attempted
  static bool s_backupAttempted;
};

}   // namespace vvdec