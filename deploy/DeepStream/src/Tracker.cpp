#include "Tracker.h"
#include <iostream>

NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize,
                        char *pCustomConfigFilePath,
                        NvMOTQuery *pQuery) {
    /**
     * Users can parse the low-level config file in pCustomConfigFilePath to check
     * the low-level tracker's requirements
     */

    pQuery->computeConfig = NVMOTCOMP_CPU;       // among {NVMOTCOMP_GPU, NVMOTCOMP_CPU}
    pQuery->numTransforms = 0;                   // 0 for IOU tracker, 1 for NvDCF or DeepSORT tracker as they require the video frames
    pQuery->colorFormats[0] = NVBUF_COLOR_FORMAT_NV12; // among {NVBUF_COLOR_FORMAT_NV12, NVBUF_COLOR_FORMAT_RGBA}

    // among {NVBUF_MEM_DEFAULT, NVBUF_MEM_CUDA_DEVICE, NVBUF_MEM_CUDA_UNIFIED, NVBUF_MEM_CUDA_PINNED, ... }
#ifdef __aarch64__
    pQuery->memType = NVBUF_MEM_DEFAULT;
#else
    pQuery->memType = NVBUF_MEM_CUDA_DEVICE;
#endif

    pQuery->supportBatchProcessing = false;    // set NvMOTBatchMode_Batch if the low-level tracker supports batch processing mode. Otherwise, NvMOTBatchMode_NonBatch
    pQuery->supportPastFrame       = false;    // set true if the low-level tracker supports the past-frame data or not

    /**
     * return NvMOTStatus_Error if something is wrong
     * return NvMOTStatus_OK if everything went well
     */
    std::cout << "[BYTETrack Initialized]" << std::endl;
    return NvMOTStatus_OK;
}


NvMOTStatus NvMOT_Init(NvMOTConfig *pConfigIn,
                       NvMOTContextHandle *pContextHandle,
                       NvMOTConfigResponse *pConfigResponse) {
    if (pContextHandle != nullptr) {
        NvMOT_DeInit(*pContextHandle);
    }

    /// User-defined class for the context
    NvMOTContext *pContext = nullptr;

    /// Instantiate the user-defined context
    pContext = new NvMOTContext(*pConfigIn, *pConfigResponse);

    /// Pass the pointer as the context handle
    *pContextHandle = pContext;

    /**
     * return NvMOTStatus_Error if something is wrong
     * return NvMOTStatus_OK if everything went well
     */
    return NvMOTStatus_OK;
}

void NvMOT_DeInit(NvMOTContextHandle contextHandle) {
    /// Destroy the context handle
    delete contextHandle;
}

NvMOTStatus NvMOT_Process(NvMOTContextHandle contextHandle,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackedObjBatch *pTrackedObjectsBatch) {
    /// Process the given video frame using the user-defined method in the context, and generate outputs
    contextHandle->processFrame(pParams, pTrackedObjectsBatch);
    return NvMOTStatus_OK;
}

void NvMOT_RemoveStreams(NvMOTContextHandle contextHandle, NvMOTStreamId streamIdMask) {

}
