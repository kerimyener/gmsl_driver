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

#include "common/LaneDetectionCommon.hpp"

#include <iostream>

//#######################################################################################
LaneNet::LaneNet(uint32_t windowWidth, uint32_t windowHeight, const std::string &inputType)
    : m_sdk(DW_NULL_HANDLE)
    , m_inputType(inputType)
    , m_sal(DW_NULL_HANDLE)
    , m_cameraSensor(DW_NULL_HANDLE)
    , m_renderer(DW_NULL_HANDLE)
    , m_renderBuffer(DW_NULL_HANDLE)
    , m_laneDetector(DW_NULL_HANDLE)
    , m_isRaw(false)
    , m_frameCUDArgba{}
    , m_frameCUDArcb{}
    , m_cameraImageProperties{}
    , m_cameraProperties{}
    , m_rawPipeline(DW_NULL_HANDLE)
    , m_streamerInput2CUDA(DW_NULL_HANDLE)
    , m_converterInput2Rgba(DW_NULL_HANDLE)
    , m_streamerCamera2GL(DW_NULL_HANDLE)
    , m_screenRectangle{}
    , m_windowWidth(windowWidth)
    , m_windowHeight(windowHeight)
    , m_cameraWidth(0)
    , m_cameraHeight(0)
    , m_threshold(0.0f)
    , m_cudaStream(0)
    #ifdef DW_USE_NVMEDIA
    , m_converterNvMYuv2rgba(DW_NULL_HANDLE)
    , m_streamerNvMedia2CUDA(DW_NULL_HANDLE)
    , m_frameNVMrgba{}
    #endif
{

}

//#######################################################################################
bool LaneNet::initializeModules()
{
    if (!initDriveworks()) return false;
    if (!initCameras()) return false;
    if (!initRenderer()) return false;
    if (!initPipeline()) return false;
    if (!initDNN()) return false;

    // start cameras as late as possible, so that all initialization routines are finished before
    gRun = gRun && dwSensor_start(m_cameraSensor) == DW_SUCCESS;
    return true;
}

//#######################################################################################
bool LaneNet::initDriveworks()
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_DEBUG);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams{};

    std::string path = DataPath::get();
    sdkParams.dataPath = path.c_str();

#ifdef VIBRANTE
    sdkParams.eglDisplay = gWindow->getEGLDisplay();
#endif

    return dwInitialize(&m_sdk, DW_VERSION, &sdkParams) == DW_SUCCESS;
}

//#######################################################################################
bool LaneNet::initCameras()
{
    dwStatus result = DW_FAILURE;

    // create sensor abstraction layer
    result = dwSAL_initialize(&m_sal, m_sdk);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot init sal: " << dwGetStatusName(result) << std::endl;
        return false;
    }
    // create GMSL Camera interface
    float32_t cameraFramerate = 0.0f;

    if(!createVideoReplay(m_cameraSensor, cameraFramerate, m_sal))
        return false;

    std::cout << "Camera image with " << m_cameraWidth << "x" << m_cameraHeight << " at "
              << cameraFramerate << " FPS" << std::endl;

    return true;
}

//#######################################################################################
bool LaneNet::initRenderer()
{
    // init renderer
    m_screenRectangle.height = gWindow->height();
    m_screenRectangle.width = gWindow->width();
    m_screenRectangle.x = 0;
    m_screenRectangle.y = 0;

    unsigned int maxLines = 20000;
    setupRenderer(m_renderer, m_screenRectangle, m_sdk);
    setupLineBuffer(m_renderBuffer, maxLines, m_sdk);

    return true;
}

//#######################################################################################
bool LaneNet::initPipeline()
{
    dwStatus status = DW_FAILURE;

#ifdef DW_USE_NVMEDIA
    // NvMedia yuv -> rgba format converter
    dwImageProperties cameraImageProperties = m_cameraImageProperties;
    dwImageProperties displayImageProperties = cameraImageProperties;
    cameraImageProperties.type = DW_IMAGE_NVMEDIA;
    displayImageProperties.type = DW_IMAGE_NVMEDIA;
    displayImageProperties.pxlFormat = DW_IMAGE_RGBA;
    displayImageProperties.planeCount = 1;
    dwImageFormatConverter_initialize(&m_converterNvMYuv2rgba, &cameraImageProperties,
                                      &displayImageProperties, m_sdk);

    // NvMedia -> CUDA image streamer
    status = dwImageStreamer_initialize(&m_streamerNvMedia2CUDA, &cameraImageProperties,
                                        DW_IMAGE_CUDA, m_sdk);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot init image streamer: " << dwGetStatusName(status) << std::endl;
        return false;
    }
#endif

    if(m_isRaw){
        dwImageProperties cameraImageProperties = m_cameraImageProperties;
        dwCameraProperties cameraProperties = m_cameraProperties;

        // Raw pipeline
        dwImageProperties rccbImageProperties;
        status = dwRawPipeline_initialize(&m_rawPipeline, cameraImageProperties, cameraProperties, m_sdk);
        status = status != DW_SUCCESS ? status : dwRawPipeline_setCUDAStream(m_cudaStream, m_rawPipeline);
        status = status != DW_SUCCESS ? status : dwRawPipeline_setDemosaicMethod(DW_DEMOSAIC_INTERPOLATION, m_rawPipeline);
        status = status != DW_SUCCESS ? status : dwRawPipeline_getDemosaicImageProperties(&rccbImageProperties, m_rawPipeline);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot initialize raw pipeline: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Input -> CUDA streamer
        dwImageStreamer_initialize(&m_streamerInput2CUDA, &cameraImageProperties, DW_IMAGE_CUDA, m_sdk);

        // Input -> RGBA format converter
        dwImageProperties displayImageProperties = rccbImageProperties;
        displayImageProperties.pxlFormat = DW_IMAGE_RGBA;
        displayImageProperties.pxlType = DW_TYPE_UINT8;
        displayImageProperties.planeCount = 1;
        status = dwImageFormatConverter_initialize(&m_converterInput2Rgba, &rccbImageProperties, &displayImageProperties, m_sdk);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot initialize input -> rgba format converter: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Setup RCB image
        m_frameCUDArcb.prop = rccbImageProperties;
        m_frameCUDArcb.layout = DW_IMAGE_CUDA_PITCH;
        cudaMallocPitch(&m_frameCUDArcb.dptr[0], &m_frameCUDArcb.pitch[0], rccbImageProperties.width * dwSizeOf(rccbImageProperties.pxlType),
                rccbImageProperties.height * rccbImageProperties.planeCount);
        m_frameCUDArcb.pitch[1] = m_frameCUDArcb.pitch[2] = m_frameCUDArcb.pitch[0];
        m_frameCUDArcb.dptr[1] = reinterpret_cast<uint8_t*>(m_frameCUDArcb.dptr[0]) + rccbImageProperties.height * m_frameCUDArcb.pitch[0];
        m_frameCUDArcb.dptr[2] = reinterpret_cast<uint8_t*>(m_frameCUDArcb.dptr[1]) + rccbImageProperties.height * m_frameCUDArcb.pitch[1];

        // Camera -> GL image streamer
        status = dwImageStreamer_initialize(&m_streamerCamera2GL, &displayImageProperties, DW_IMAGE_GL, m_sdk);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot init GL streamer: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Setup RGBA CUDA image
        {
            void *dptr = nullptr;
            size_t pitch = 0;
            cudaMallocPitch(&dptr, &pitch, rccbImageProperties.width * 4, rccbImageProperties.height);
            dwImageCUDA_setFromPitch(&m_frameCUDArgba, dptr, rccbImageProperties.width, rccbImageProperties.height,
                                     pitch, DW_IMAGE_RGBA);
        }
#ifdef DW_USE_NVMEDIA
        // Setup RGBA NvMedia image
        {
            NvMediaDevice *nvmDevice = nullptr;
            dwContext_getNvMediaDevice(&nvmDevice, m_sdk);

            NvMediaImageAdvancedConfig advConfig{};
            NvMediaImage *rgbaNvMediaImage = NvMediaImageCreate(nvmDevice,
                                                                NvMediaSurfaceType_Image_RGBA,
                                                                NVMEDIA_IMAGE_CLASS_SINGLE_IMAGE, 1,
                                                                rccbImageProperties.width, rccbImageProperties.height,
                                                                NVMEDIA_IMAGE_ATTRIBUTE_UNMAPPED,
                                                                &advConfig);
            dwImageNvMedia_setFromImage(&m_frameNVMrgba, rgbaNvMediaImage);
        }
#endif
        // Set camera width and height for DNN
        m_cameraWidth = rccbImageProperties.width;
        m_cameraHeight = rccbImageProperties.height;
    }
    else{
        dwImageProperties cameraImageProperties = m_cameraImageProperties;

        // Input -> RGBA format converter
        dwImageProperties displayImageProperties = cameraImageProperties;
        displayImageProperties.pxlFormat = DW_IMAGE_RGBA;
        displayImageProperties.pxlType = DW_TYPE_UINT8;
        displayImageProperties.planeCount = 1;
        displayImageProperties.type = DW_IMAGE_CUDA;
        cameraImageProperties.type = DW_IMAGE_CUDA;
        status = dwImageFormatConverter_initialize(&m_converterInput2Rgba, &cameraImageProperties, &displayImageProperties, m_sdk);
        if (status != DW_SUCCESS) {
            std::cerr << "Cannot initialize input -> rgba format converter: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Camera -> GL image streamer
#ifdef DW_USE_NVMEDIA
        displayImageProperties.type = DW_IMAGE_NVMEDIA;
#endif
        status = dwImageStreamer_initialize(&m_streamerCamera2GL, &displayImageProperties, DW_IMAGE_GL, m_sdk);

        if (status != DW_SUCCESS) {
            std::cerr << "Cannot init GL streamer: " << dwGetStatusName(status) << std::endl;
            return false;
        }

        // Setup RGBA CUDA image
        {
            void *dptr = nullptr;
            size_t pitch = 0;
            cudaMallocPitch(&dptr, &pitch, m_cameraWidth * 4, m_cameraHeight);
            dwImageCUDA_setFromPitch(&m_frameCUDArgba, dptr, m_cameraWidth, m_cameraHeight,
                                     pitch, DW_IMAGE_RGBA);
        }

#ifdef DW_USE_NVMEDIA
        // Setup RGBA NvMedia image
        {
            NvMediaDevice *nvmDevice = nullptr;
            dwContext_getNvMediaDevice(&nvmDevice, m_sdk);

            NvMediaImageAdvancedConfig advConfig{};
            NvMediaImage *rgbaNvMediaImage = NvMediaImageCreate(nvmDevice,
                                                                NvMediaSurfaceType_Image_RGBA,
                                                                NVMEDIA_IMAGE_CLASS_SINGLE_IMAGE, 1,
                                                                cameraImageProperties.width, cameraImageProperties.height,
                                                                NVMEDIA_IMAGE_ATTRIBUTE_UNMAPPED,
                                                                &advConfig);
            dwImageNvMedia_setFromImage(&m_frameNVMrgba, rgbaNvMediaImage);
        }
#endif
        // Set camera width and height for DNN
        m_cameraWidth = cameraImageProperties.width;
        m_cameraHeight = cameraImageProperties.height;
    }

    return true;
}

//#######################################################################################
bool LaneNet::initDNN()
{
    dwStatus res = DW_FAILURE;
    //dwLaneDector_initializeLaneNet in LaneDetector.h in dw folder
    res = dwLaneDetector_initializeLaneNet(&m_laneDetector,
                                           m_cameraWidth, m_cameraHeight,
                                           m_cudaStream, m_sdk);

    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot initialize LaneNet: " << dwGetStatusName(res) << std::endl;
        return false;
    }

    m_threshold = 0.3f;
    std::string inputThreshold = gArguments.get("threshold");
    if(inputThreshold!="0.3"){
        try{
            m_threshold = std::stof(inputThreshold);
        } catch(...) {
            std::cerr << "Given threshold can't be parsed" << std::endl;
            return false;
        }
    }

    res = dwLaneDetectorLaneNet_setDetectionThreshold(m_threshold, m_laneDetector);
    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot set LaneNet threshold: " << dwGetStatusName(res) << std::endl;
        return false;
    }

    //Default to 60 FOV input video setup, FOV is only set when "120" is parsed
    std::string videoFOV = gArguments.get("fov");
    if (videoFOV.compare("120") == 0) {
        res = dwLaneDetectorLaneNet_setVideoFOV(DW_LANEMARK_VIDEO_FOV_120, m_laneDetector);
        if (res != DW_SUCCESS)
        {
            std::cerr << "Cannot set LaneNet input video FOV: " << dwGetStatusName(res) << std::endl;
            return false;
        }
    }
    else if(videoFOV.compare("60") != 0) {
        std::cerr << "Given video FOV is not valid" << std::endl;
        return false;
    }

    //Default to 0.25, uncomment this block to customize lane temporal smoothing factor
    /*
    res = dwLaneDetectorLaneNet_setTemporalSmoothFactor(m_temporalSmoothFactor, m_laneDetector);

    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot set LaneNet temporal smooth factor: " << dwGetStatusName(res) << std::endl;
        return false;
    }
    */


    //Default to full frame, uncomment this block to customize lane detection ROI
    /*
    res = dwLaneDetector_setDetectionROI(&m_roi, m_laneDetector);

    if (res != DW_SUCCESS)
    {
        std::cerr << "Cannot set LaneNet detection ROI: " << dwGetStatusName(res) << std::endl;
        return false;
    }
    */

    return true;
}

//#######################################################################################
void LaneNet::drawLaneDetectionROI(dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    dwRect roi{};
    dwLaneDetector_getDetectionROI(&roi, m_laneDetector);
    float32_t x_start = static_cast<float32_t>(roi.x);
    float32_t x_end   = static_cast<float32_t>(roi.x + roi.width);
    float32_t y_start = static_cast<float32_t>(roi.y);
    float32_t y_end   = static_cast<float32_t>(roi.y + roi.height);
    float32_t *coords     = nullptr;
    uint32_t maxVertices  = 0;
    uint32_t vertexStride = 0;
    dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
    coords[0]  = x_start;
    coords[1]  = y_start;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_start;
    coords[1] = y_start;
    dwRenderBuffer_unmap(8, renderBuffer);
    dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
    dwRenderer_setLineWidth(2, renderer);
    dwRenderer_renderBuffer(renderBuffer, renderer);
}

//#######################################################################################
void LaneNet::drawLaneMarkings(const dwLaneDetection &lanes, float32_t laneWidth,
                                     dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    drawLaneDetectionROI(renderBuffer, renderer);

    /*data.array.layout.dim.push_back(std_msgs::MultiArrayDimension());
    data.array.layout.dim.push_back(std_msgs::MultiArrayDimension());
    data.array.layout.dim[0].label = "x";
    data.array.layout.dim[0].label = "y";*/

    for (uint32_t i = 0; i < lanes.numLaneMarkings; ++i) {

        const dwLaneMarking& laneMarking = lanes.laneMarkings[i];

        dwLanePositionType category = laneMarking.positionType;

        if(category==DW_LANEMARK_POSITION_ADJACENT_LEFT){
            dwRenderer_setColor(DW_RENDERER_COLOR_YELLOW, renderer);
            data.header.frame_id="yellow";}
        else if(category==DW_LANEMARK_POSITION_EGO_LEFT){
            dwRenderer_setColor(DW_RENDERER_COLOR_RED, renderer);
            data.header.frame_id="red";}
        else if(category==DW_LANEMARK_POSITION_EGO_RIGHT){
            dwRenderer_setColor(DW_RENDERER_COLOR_GREEN, renderer);
            data.header.frame_id="green";}
        else if(category==DW_LANEMARK_POSITION_ADJACENT_RIGHT){
            dwRenderer_setColor(DW_RENDERER_COLOR_BLUE, renderer);
            data.header.frame_id="blue";}
        dwRenderer_setLineWidth(laneWidth, renderer);

        /*data.array.layout.dim[0].size = laneMarking.numPoints;
        data.array.layout.dim[1].size = laneMarking.numPoints;
        data.array.layout.dim[0].stride = 2*laneMarking.numPoints;
        data.array.layout.dim[1].stride = laneMarking.numPoints;*/

        float32_t* coords = nullptr;
        uint32_t maxVertices = 0;
        uint32_t vertexStride = 0;
        dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);

        uint32_t n_verts = 0;
        dwVector2f previousP{};
        bool firstPoint = true;

        for (uint32_t j = 0; j < laneMarking.numPoints; ++j) {

            dwVector2f center;
            center.x = laneMarking.imagePoints[j].x;
            center.y = laneMarking.imagePoints[j].y;
            data.array.data.push_back(center.x) ;
            data.array.data.push_back(center.y) ;

            if (firstPoint) { // Special case for the first point
                previousP = center;
                firstPoint = false;
            }
            else {
                n_verts += 2;
                if(n_verts > maxVertices)
                    break;

                coords[0] = static_cast<float32_t>(previousP.x);
                coords[1] = static_cast<float32_t>(previousP.y);
                coords += vertexStride;

                coords[0] = static_cast<float32_t>(center.x);
                coords[1] = static_cast<float32_t>(center.y);
                coords += vertexStride;

                previousP = center;
            }
        }

        dwRenderBuffer_unmap(n_verts, renderBuffer);
        dwRenderer_renderBuffer(renderBuffer, renderer);
        //std::cout << "laneMarking.imagePoints["<< i << "].x: " << laneMarking.imagePoints[i].x<< std::endl;
        //std::cout << "laneMarking.imagePoints["<< i << "].y: " << laneMarking.imagePoints[i].y<< std::endl;



    }
}

//#######################################################################################
void LaneNet::renderCameraTexture(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer)
{
    dwImageGL *frameGL = nullptr;

    if (dwImageStreamer_receiveGL(&frameGL, 30000, streamer) != DW_SUCCESS) {
        std::cerr << "did not received GL frame within 30ms" << std::endl;
    } else {
        // render received texture
        dwRenderer_renderTexture(frameGL->tex, frameGL->target, renderer);
        dwImageStreamer_returnReceivedGL(frameGL, streamer);
    }
}

//#######################################################################################
bool LaneNet::createVideoReplay(dwSensorHandle_t &salSensor,
                                      float32_t &cameraFrameRate,
                                      dwSALHandle_t sal)
{
    dwSensorParams params;
    dwStatus result;

    if (m_inputType.compare("camera") == 0) {
        std::string cameraType = gArguments.get("camera-type");
        std::string parameterString = "camera-type=" + cameraType;
        parameterString += ",csi-port=" + gArguments.get("csi-port");
        parameterString += ",slave=" + gArguments.get("slave");
        parameterString += ",serialize=false,camera-count=4";
        if(cameraType.compare("c-ov10640-b1") == 0 ||
                cameraType.compare("ov10640-svc210") == 0 ||
                cameraType.compare("ov10640-svc212") == 0)
        {
            parameterString += ",output-format=yuv";
            m_isRaw = false;
        }
        else{
            parameterString += ",output-format=raw";
            m_isRaw = true;
        }
        std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
        uint32_t cameraIdx = std::stoi(gArguments.get("camera-index"));
        if(cameraIdx < 0 || cameraIdx > 3){
            std::cerr << "Error: camera index must be 0, 1, 2 or 3" << std::endl;
            return false;
        }
        parameterString += ",camera-mask=" + cameraMask[cameraIdx];

        params.parameters           = parameterString.c_str();
        params.protocol             = "camera.gmsl";

        result                      = dwSAL_createSensor(&salSensor, params, sal);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot create driver: camera.gmsl with params: " << params.parameters << std::endl
                      << "Error: " << dwGetStatusName(result) << std::endl;
            return false;
        }
    }
    else{
        std::string parameterString = gArguments.parameterString();
        params.parameters           = parameterString.c_str();
        params.protocol             = "camera.virtual";
        result                      = dwSAL_createSensor(&salSensor, params, sal);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot create driver: camera.virtual with params: " << params.parameters << std::endl
                      << "Error: " << dwGetStatusName(result) << std::endl;
            return false;
        }
        std::string videoFormat = gArguments.get("video");
        std::size_t found = videoFormat.find_last_of(".");
        m_isRaw = videoFormat.substr(found+1).compare("raw") == 0 ? true : false;
    }

    dwSensorCamera_getSensorProperties(&m_cameraProperties, salSensor);
    cameraFrameRate = m_cameraProperties.framerate;

    if(m_isRaw)
        dwSensorCamera_getImageProperties(&m_cameraImageProperties, DW_CAMERA_RAW_IMAGE, salSensor);
    else
        dwSensorCamera_getImageProperties(&m_cameraImageProperties, DW_CAMERA_PROCESSED_IMAGE, salSensor);

    if(m_isRaw && m_inputType.compare("camera") == 0)
        m_cameraImageProperties.height = m_cameraProperties.resolution.y;
    m_cameraHeight = m_cameraImageProperties.height;
    m_cameraWidth = m_cameraImageProperties.width;

    return true;
}

//#######################################################################################
void LaneNet::setupRenderer(dwRendererHandle_t &renderer, const dwRect &screenRect, dwContextHandle_t dwSdk)
{
    dwRenderer_initialize(&renderer, dwSdk);

    float32_t boxColor[4] = {0.0f,1.0f,0.0f,1.0f};
    dwRenderer_setColor(boxColor, renderer);
    dwRenderer_setLineWidth(2.0f, renderer);
    dwRenderer_setRect(screenRect, renderer);
}

//#######################################################################################
void LaneNet::setupLineBuffer(dwRenderBufferHandle_t &lineBuffer, unsigned int maxLines, dwContextHandle_t dwSdk)
{
    dwRenderBufferVertexLayout layout;
    layout.posFormat   = DW_RENDER_FORMAT_R32G32_FLOAT;
    layout.posSemantic = DW_RENDER_SEMANTIC_POS_XY;
    layout.colFormat   = DW_RENDER_FORMAT_NULL;
    layout.colSemantic = DW_RENDER_SEMANTIC_COL_NULL;
    layout.texFormat   = DW_RENDER_FORMAT_NULL;
    layout.texSemantic = DW_RENDER_SEMANTIC_TEX_NULL;
    dwRenderBuffer_initialize(&lineBuffer, layout, DW_RENDER_PRIM_LINELIST, maxLines, dwSdk);
    dwRenderBuffer_set2DCoordNormalizationFactors((float32_t)m_cameraWidth,
                                                  (float32_t)m_cameraHeight, lineBuffer);
}

//#######################################################################################
void LaneNet::runDetector(dwImageCUDA* frame)
{
    // Run inference if the model is valid
    if (m_laneDetector)
    {
        dwLaneDetection lanes{};
        dwStatus res = dwLaneDetector_processDeviceAsync(frame, m_laneDetector);
        res = res == DW_SUCCESS ? dwLaneDetector_interpretHost(m_laneDetector) : res;
        if (res != DW_SUCCESS)
        {
            std::cerr << "runDetector failed with: " << dwGetStatusName(res) << std::endl;
        }

        dwLaneDetector_getLaneDetections(&lanes, m_laneDetector);
        drawLaneMarkings(lanes, 6.0f, m_renderBuffer, m_renderer);
        //std::cout << "inside runDetector"<< std::endl;
        //std::cout << "laneMarkings: " << &lanes.laneMarkings<< std::endl;
       // std::cout << "m_laneDectector: "<< m_laneDetector << std::endl;
    }
}

//#######################################################################################
dwStatus LaneNet::runSingleCameraPipeline()
{
    dwStatus status = DW_SUCCESS;
    if (m_isRaw) {
        status = runSingleCameraPipelineRaw();
    } else {
        status = runSingleCameraPipelineH264();
    }

    if (status == DW_END_OF_STREAM) {
        std::cout << "Camera reached end of stream" << std::endl;
        dwSensor_reset(m_cameraSensor);
    }
    else if (status != DW_SUCCESS) {
        gRun = false;
    }

    return status;
}

//#######################################################################################
dwStatus LaneNet::runSingleCameraPipelineRaw()
{
    dwStatus result                 = DW_FAILURE;
    dwCameraFrameHandle_t frame     = nullptr;
    dwImageCUDA* frameCUDARaw       = nullptr;
    dwImageCPU *frameCPURaw         = nullptr;
    dwImageCUDA* retimg             = nullptr;
    const dwCameraDataLines* dataLines;
#ifdef DW_USE_NVMEDIA
    dwImageNvMedia *frameNvMediaRaw = nullptr;
#endif

    result = dwSensorCamera_readFrame(&frame, 0, 1000000, m_cameraSensor);
    if (result == DW_END_OF_STREAM)
        return result;
    if (result != DW_SUCCESS && result != DW_END_OF_STREAM) {
        std::cerr << "readFrameNvMedia: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    if (m_inputType.compare("camera") == 0) {
#ifdef DW_USE_NVMEDIA
        result = dwSensorCamera_getImageNvMedia(&frameNvMediaRaw, DW_CAMERA_RAW_IMAGE, frame);
#endif
    }
    else{
        result = dwSensorCamera_getImageCPU(&frameCPURaw, DW_CAMERA_RAW_IMAGE, frame);
    }
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot get raw image: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    result = dwSensorCamera_getDataLines(&dataLines, frame);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot get data lines: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    if (m_inputType.compare("camera") == 0) {
#ifdef DW_USE_NVMEDIA
        result = dwImageStreamer_postNvMedia(frameNvMediaRaw, m_streamerInput2CUDA);
#endif
    }
    else{
        result = dwImageStreamer_postCPU(frameCPURaw, m_streamerInput2CUDA);
    }
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot post image: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    result = dwImageStreamer_receiveCUDA(&frameCUDARaw, 10000, m_streamerInput2CUDA);
    dwImageCUDA cudaFrame{};
    {
        dwRect roi;
        dwSensorCamera_getImageROI(&roi, m_cameraSensor);
        dwImageCUDA_mapToROI(&cudaFrame, frameCUDARaw, roi);
    }

    if (result != DW_SUCCESS) {
        std::cerr << "Cannot reiceve CUDA: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    // Raw -> RCB
    result = dwRawPipeline_convertRawToDemosaic(&m_frameCUDArcb, &cudaFrame, dataLines, m_rawPipeline);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot run rccb pipeline: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    // RCB -> RGBA
    dwImageFormatConverter_copyConvertCUDA(&m_frameCUDArgba, &m_frameCUDArcb, m_converterInput2Rgba, m_cudaStream);

    // frame -> GL (rgba) - for rendering
    {
        result = dwImageStreamer_postCUDA(&m_frameCUDArgba, m_streamerCamera2GL);
        if (result != DW_SUCCESS) {
            std::cerr << "cannot post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }

        renderCameraTexture(m_streamerCamera2GL, m_renderer);

        result = dwImageStreamer_waitPostedCUDA(&retimg, 60000, m_streamerCamera2GL);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot wait post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }
    }

    dwImageStreamer_returnReceivedCUDA(frameCUDARaw, m_streamerInput2CUDA);

    runDetector(&m_frameCUDArcb);

    if (m_inputType.compare("camera") == 0) {
#ifdef DW_USE_NVMEDIA
        dwImageStreamer_waitPostedNvMedia(&frameNvMediaRaw, 10000, m_streamerInput2CUDA);
#endif
    }
    else{
        dwImageStreamer_waitPostedCPU(&frameCPURaw, 10000, m_streamerInput2CUDA);
    }

    dwSensorCamera_returnFrame(&frame);

    return DW_SUCCESS;
}

//#######################################################################################
dwStatus LaneNet::runSingleCameraPipelineH264()
{
    dwStatus result             = DW_FAILURE;
    dwCameraFrameHandle_t frame     = nullptr;
#ifdef DW_USE_NVMEDIA
    dwImageNvMedia *frameNvMediaYuv = nullptr;
    dwImageCUDA *imgCUDA            = nullptr;
    dwImageNvMedia *retimg          = nullptr;
#else
    dwImageCUDA *frameCUDAyuv       = nullptr;
    dwImageCUDA *retimg             = nullptr;
#endif

    result = dwSensorCamera_readFrame(&frame, 0, 50000, m_cameraSensor);
    if (result == DW_END_OF_STREAM)
        return result;
    if (result != DW_SUCCESS) {
        std::cout << "readFrameCUDA: " << dwGetStatusName(result) << std::endl;
        return result;
    }

#ifdef DW_USE_NVMEDIA
    result = dwSensorCamera_getImageNvMedia(&frameNvMediaYuv, DW_CAMERA_PROCESSED_IMAGE, frame);
#else
    result = dwSensorCamera_getImageCUDA(&frameCUDAyuv, DW_CAMERA_PROCESSED_IMAGE, frame);
#endif
    if (result != DW_SUCCESS) {
        std::cout << "getImage: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    // YUV->RGBA
#ifdef DW_USE_NVMEDIA
    result = dwImageFormatConverter_copyConvertNvMedia(&m_frameNVMrgba, frameNvMediaYuv, m_converterNvMYuv2rgba);
#else
    result = dwImageFormatConverter_copyConvertCUDA(&m_frameCUDArgba, frameCUDAyuv, m_converterInput2Rgba, 0);
#endif
    if (result != DW_SUCCESS) {
        std::cout << "Cannot convert to RGBA: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    // we can return the frame already now, we are working with a copy from now on
    dwSensorCamera_returnFrame(&frame);

    // frame -> GL (rgba) - for rendering
    {
#ifdef DW_USE_NVMEDIA
        result = dwImageStreamer_postNvMedia(&m_frameNVMrgba, m_streamerCamera2GL);
#else
        result = dwImageStreamer_postCUDA(&m_frameCUDArgba, m_streamerCamera2GL);
#endif
        if (result != DW_SUCCESS) {
            std::cerr << "cannot post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }

        renderCameraTexture(m_streamerCamera2GL, m_renderer);

#ifdef DW_USE_NVMEDIA
        result = dwImageStreamer_waitPostedNvMedia(&retimg, 60000, m_streamerCamera2GL);
#else
        result = dwImageStreamer_waitPostedCUDA(&retimg, 60000, m_streamerCamera2GL);
#endif
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot wait post RGBA image" << dwGetStatusName(result) << std::endl;
            return result;
        }
    }

#ifdef DW_USE_NVMEDIA
    // (nvmedia) NVMEDIA -> CUDA (rgba) - for processing
    // since DNN expects pitch linear cuda memory we cannot just post gFrameNVMrgba through the streamer
    // cause the outcome of the streamer would have block layout, but we need pitch
    // hence we perform one more extra YUV2RGBA conversion using CUDA
    {
        result = dwImageStreamer_postNvMedia(frameNvMediaYuv, m_streamerNvMedia2CUDA);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot post NvMedia frame " << dwGetStatusName(result) << std::endl;
            return result;
        }

        result = dwImageStreamer_receiveCUDA(&imgCUDA, 60000, m_streamerNvMedia2CUDA);
        if (result != DW_SUCCESS || imgCUDA == 0) {
            std::cerr << "did not received CUDA frame within 60ms" << std::endl;
            return result;
        }

        // copy convert into RGBA
        result = dwImageFormatConverter_copyConvertCUDA(&m_frameCUDArgba, imgCUDA, m_converterInput2Rgba, 0);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot convert to RGBA" << std::endl;
            return result;
        }

    }
#endif

    runDetector(&m_frameCUDArgba);

#ifdef DW_USE_NVMEDIA
    dwImageStreamer_returnReceivedCUDA(imgCUDA, m_streamerNvMedia2CUDA);
    dwImageStreamer_waitPostedNvMedia(&retimg, 60000, m_streamerNvMedia2CUDA);
#endif

    return DW_SUCCESS;
}

//#######################################################################################
void LaneNet::releaseModules()
{
    if (m_converterInput2Rgba != DW_NULL_HANDLE)
        dwImageFormatConverter_release(&m_converterInput2Rgba);

    if (m_frameCUDArgba.dptr[0])
        cudaFree(m_frameCUDArgba.dptr[0]);

    if (m_streamerCamera2GL != DW_NULL_HANDLE)
        dwImageStreamer_release(&m_streamerCamera2GL);

    if(m_isRaw){
        if (m_streamerInput2CUDA != DW_NULL_HANDLE)
            dwImageStreamer_release(&m_streamerInput2CUDA);
        if (m_rawPipeline != DW_NULL_HANDLE)
            dwRawPipeline_release(&m_rawPipeline);
        if (m_frameCUDArcb.dptr[0])
            cudaFree(m_frameCUDArcb.dptr[0]);
    }

#ifdef DW_USE_NVMEDIA
    if (m_converterNvMYuv2rgba != DW_NULL_HANDLE)
        dwImageFormatConverter_release(&m_converterNvMYuv2rgba);

    if (m_streamerNvMedia2CUDA != DW_NULL_HANDLE)
        dwImageStreamer_release(&m_streamerNvMedia2CUDA);

    if (m_frameNVMrgba.img != nullptr)
        NvMediaImageDestroy(m_frameNVMrgba.img);
#endif

    dwSensor_stop(m_cameraSensor);
    dwSAL_releaseSensor(&m_cameraSensor);

    dwRenderBuffer_release(&m_renderBuffer);
    dwRenderer_release(&m_renderer);

    // release used objects in correct order
    dwSAL_release(&m_sal);
    dwLaneDetector_release(&m_laneDetector);
    dwRelease(&m_sdk);
    dwLogger_release();
}

void LaneNet::lanePub (ros::Publisher *publisher){

  publisher->publish(data);
}
