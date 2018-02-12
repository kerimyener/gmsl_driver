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
// Copyright (c) 2015-2016 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////
#define _CRT_SECURE_NO_WARNINGS

#include <thread>

// Sample
#include <common/LaneDetectionCommon.hpp>

//#######################################################################################
int main(int argc, const char **argv)
{   
         ros::init(argc,(char**)argv,"mai");
         ros::NodeHandle n;
	 ros::Publisher pub = n.advertise<std_msgs::UInt32MultiArray>("array", 100);


	 const ProgramArguments arguments = ProgramArguments({
#ifdef DW_USE_NVMEDIA
            ProgramArguments::Option_t("camera-type", "ar0231-rccb-ssc"),
            ProgramArguments::Option_t("csi-port", "ab"),
            ProgramArguments::Option_t("camera-index", "0"),
            ProgramArguments::Option_t("slave", "0"),
            ProgramArguments::Option_t("input-type", "video"),
#endif
            ProgramArguments::Option_t("offscreen", "0"),
            ProgramArguments::Option_t("video",
                                       (DataPath::get() +
                    std::string{"/samples/laneDetection/video_lane.h264"}).c_str()),
            ProgramArguments::Option_t("threshold", "0.3"),
            ProgramArguments::Option_t("width", "960"),
            ProgramArguments::Option_t("height", "576"),
            ProgramArguments::Option_t("fov", "60"),
        });

    // Default window width and height
    uint32_t windowWidth = 960;
    uint32_t windowHeight = 576;

    // init framework
    initSampleApp(argc, argv, &arguments, NULL, windowWidth, windowHeight);

    std::string inputType = "video";
#ifdef DW_USE_NVMEDIA
    inputType = gArguments.get("input-type");
#endif

    LaneNet laneNet(windowWidth, windowHeight, inputType);

    // init driveworks
    if (!laneNet.initializeModules())
    {
        std::cerr << "Cannot initialize DW subsystems" << std::endl;
        gRun = false;
    }

    typedef std::chrono::high_resolution_clock myclock_t;
    typedef std::chrono::time_point<myclock_t> timepoint_t;
    timepoint_t lastUpdateTime = myclock_t::now();

    // main loop
    // grun and gWindow defined in SampleFramework.hpp.
    //gRun is Boolean and gWindow is object derived from WindowBase in WindowGLFW
    while (gRun && !gWindow->shouldClose() && ros::ok()) {
    	pub.publish(laneNet.array);
        std::this_thread::yield();

        bool processImage = true;

        // run with at most 30FPS
        std::chrono::milliseconds timeSinceUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(myclock_t::now() - lastUpdateTime);
        if (timeSinceUpdate < std::chrono::milliseconds(33)) //33
            processImage = false;

        if (processImage) {

            lastUpdateTime = myclock_t::now();

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            laneNet.runSingleCameraPipeline();

            gWindow->swapBuffers();
 	    ros::spinOnce();
        }
    }

    // Release modules and driveworks.
    laneNet.releaseModules();

    // release framework
    releaseSampleApp();

    return 0;
}
