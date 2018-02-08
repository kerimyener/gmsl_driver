/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2017 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef SAMPLES_LANEDETECTIONCOMMON_HPP__
#define SAMPLES_LANEDETECTIONCOMMON_HPP__

// Sample
#include "DataPath.hpp"
#include "ProgramArguments.hpp"
#include "SampleFramework.hpp"

// SAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

// Raw
#include <dw/raw/RawPipeline.h>

// Renderer
#include <dw/renderer/Renderer.h>

// LaneDetector
#include <dw/lanes/LaneDetector.h>

// IMAGE
#include <dw/image/FormatConverter.h>
#include <dw/image/ImageStreamer.h>

#include <string>

class LaneNet
{
public:
    LaneNet(uint32_t windowWidth, uint32_t windowHeight, const std::string &inputType);

    virtual bool initializeModules();
    virtual void releaseModules();

    virtual dwStatus runSingleCameraPipeline();

protected:
    bool initDriveworks();
    bool initCameras();
    bool initRenderer();
    bool initPipeline();
    bool initDNN();

    void setupRenderer(dwRendererHandle_t &renderer, const dwRect &screenRect, dwContextHandle_t dwSdk);
    void setupLineBuffer(dwRenderBufferHandle_t &lineBuffer, unsigned int maxLines,
                         dwContextHandle_t dwSdk);

    void renderCameraTexture(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer);
    void drawLaneMarkings(const dwLaneDetection &lanes, float32_t laneWidth,
                          dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);
    void drawLaneDetectionROI(dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer);

    dwStatus runSingleCameraPipelineH264();
    dwStatus runSingleCameraPipelineRaw();

    void runDetector(dwImageCUDA* frame);
    bool createVideoReplay(dwSensorHandle_t &salSensor, float32_t &cameraFrameRate, dwSALHandle_t sal);

    // SDK
    dwContextHandle_t m_sdk;

    // Cameras
    std::string m_inputType;
    dwSALHandle_t m_sal;
    dwSensorHandle_t m_cameraSensor;

    // Renderers
    dwRendererHandle_t m_renderer;
    dwRenderBufferHandle_t m_renderBuffer;

    // Lane detector
    dwLaneDetectorHandle_t m_laneDetector;

    // Streamers and converters
    bool m_isRaw;
    dwImageCUDA m_frameCUDArgba;
    dwImageCUDA m_frameCUDArcb;
    dwImageProperties m_cameraImageProperties;
    dwCameraProperties m_cameraProperties;
    dwRawPipelineHandle_t m_rawPipeline;
    dwImageStreamerHandle_t m_streamerInput2CUDA;
    dwImageFormatConverterHandle_t m_converterInput2Rgba;
    dwImageStreamerHandle_t m_streamerCamera2GL;

    // Screen geometry
    dwRect m_screenRectangle;
    uint32_t m_windowWidth;
    uint32_t m_windowHeight;

    uint32_t m_cameraWidth;
    uint32_t m_cameraHeight;

    float32_t m_threshold;

    //dwRect m_roi; //default full frame, customize with x,y,width,height in video resolution
    //float32_t m_temporalSmoothFactor; //smoothed point = factor*previous + (1.0-factor)*current

    // Cuda stream
    cudaStream_t m_cudaStream;

    #ifdef DW_USE_NVMEDIA
    dwImageFormatConverterHandle_t m_converterNvMYuv2rgba;
    dwImageStreamerHandle_t m_streamerNvMedia2CUDA;
    dwImageNvMedia m_frameNVMrgba;
    #endif

};


#endif // SAMPLES_LANEDETECTIONCOMMON_HPP__
