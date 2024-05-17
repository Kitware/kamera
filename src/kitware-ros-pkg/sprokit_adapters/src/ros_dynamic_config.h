/*ckwg +5
 * Copyright 2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "ros_dynamic_config_export.h"

#include "ros/ros.h"
#include "diagnostic_msgs/DiagnosticStatus.h"

#include <vital/algo/dynamic_configuration.h>
#include <atomic>
#include <mutex>

// ------------------------------------------------------------------
/** Algorithm instance to dynamic configuration in a ROS environment
 *
 * This algorithm registers as a ROS subscriber, listening to a topic
 * that will supply a diagnostic_msgs::DiagnosticStatus message. This
 * message is not a great fit for this application, but it does have a
 * key/value vector.
 *
 * The key/value vector is transferred to the config block and made
 * available in the get_dynamic_configuration() call.
 *
 * Typical config
 * :dynamic_config:type ros_dynamic_config
 * :dynamic_config:ros_dynamic_config:topic display_config
 */
class ROS_DYNAMIC_CONFIG_EXPORT ros_dynamic_config
: public kwiver::vital::algo::dynamic_configuration
{
public:
  ros_dynamic_config();
  virtual ~ros_dynamic_config();

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
  /// Callback from subscriber.
  /// This may be misusing the DiagnosticStatus message, but it has a key/value array.
  void callbackEvent(const diagnostic_msgs::DiagnosticStatus& msg );

  ros::Subscriber m_sub; // handle to subscriber

  // config block that is used to hold the scaling value.
  kwiver::vital::config_block_sptr m_config;

  // Subscribe to this topic
  std::string m_topic;

  std::shared_ptr<ros::NodeHandle> m_node;

  // Lock for pointer to config block.
  // Tried std::atomic, but did not work well with smart pointers.
  std::mutex m_config_lock;
};
