#ifndef PUBLISHER_LANE
#define PUBLISHER_LANE

#include <stdio.h>
#include <stdlib.h>
#include "ros/ros.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"

class Publisher_lane {

public:
   Publisher_lane(std::string topic_name);

   //void WriteToOpenCV(unsigned char*, int, int);


   ros::NodeHandle nh;
   ros::Publisher pub;
   std::string topic_name;

   unsigned int counter;

};



#endif

