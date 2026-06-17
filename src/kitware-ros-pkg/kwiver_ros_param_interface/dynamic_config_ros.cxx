
/**
 * \file
 * \brief Header defining the implementation to dynamic_config_ros
 */

#include "dynamic_config_ros.h"
#include "ros/ros.h"

namespace kwiver {
namespace arrows {
namespace core {


// ------------------------------------------------------------------
dynamic_config_ros::
dynamic_config_ros()
{ }


// ------------------------------------------------------------------
void
dynamic_config_ros::
set_configuration( kwiver::vital::config_block_sptr config )
{ }


// ------------------------------------------------------------------
bool
dynamic_config_ros::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
dynamic_config_ros::
get_dynamic_configuration()
{
  auto config_block = kwiver::vital::config_block::empty_config();
  int target_size;
  ros::param::getCached("person_detect_target_size", target_size);
  config_block->set_value("person_detect_target_size", target_size);
  return config_block;
}

} } } // end namespace

