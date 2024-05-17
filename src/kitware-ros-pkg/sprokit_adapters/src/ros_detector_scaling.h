/*ckwg +5
 * Copyright 2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "ros_detector_scaling_export.h"

#include "ros/ros.h"
#include "std_msgs/Float64.h"

#include <vital/algo/dynamic_configuration.h>

// ------------------------------------------------------------------
/** Algorithm instance to support detector scaling.
 *
 * This algorithm registers as a ROS subscriber, listening to a topic
 * that will supply a double value to be supplied as a scaling factor.
 *
 * Typical config
 * :scaling:type ros_detector_scaling
 * :scaling:ros_detector_scaling:topic spinner1
 *
 * Note that this class is a special case of ros_dynamic_config class.
 */
class ROS_DETECTOR_SCALING_EXPORT ros_detector_scaling
: public kwiver::vital::algo::dynamic_configuration
{
public:
  ros_detector_scaling();
  virtual ~ros_detector_scaling();

  virtual kwiver::vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Return dynamic configuration values
  /**
   * This method returns dynamic configuration values. a valid config
   * block is returned even if there are not values being returned.
   */
  virtual kwiver::vital::config_block_sptr get_dynamic_configuration();

private:
  void callbackEvent(const std_msgs::Float64& msg );

  double m_scaleFactor; // scale factor (0-1)
  ros::Subscriber m_sub; // handle to subscriber

  // Persistent config block that is used to hold the scaling value.
  kwiver::vital::config_block_sptr m_config;
  std::string m_topic;

  std::shared_ptr<ros::NodeHandle> m_node;
};
