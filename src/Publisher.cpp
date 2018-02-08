#include "common/Publisher.hpp"

#include <vector>
#include <string>



Publisher_lane::Publisher_lane(std::string topic_name) : it(nh), counter(0)	{
   pub = it.advertise(topic_name, 1);
    
}


