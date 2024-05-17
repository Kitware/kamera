#ifndef __KWIVER_ROS_PARAM_INTERFACE__
#define __KWIVER_ROS_PARAM_INTERFACE__

#include "ros/ros.h"
#include "std_msgs/Int32.h"
#include "sensor_msgs/JointState.h"

#define WITH_ROS


template <typename DATA_T>
class KwiverRosParamInterface{
#ifdef WITH_ROS
    ros::NodeHandle ros_node_handle_;
    ros::Publisher pan_tilt_pub_;
#endif
    std::string key_;

  public:
    KwiverRosParamInterface(std::string key) : key_(key){}
    ~KwiverRosParamInterface();

    bool SetParam(DATA_T data){
#ifdef WITH_ROS
      return ros::param::set(key_, data);
#endif
    }
};

#endif // __KWIVER_ROS_PARAM_INTERFACE__

