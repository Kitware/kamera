/*ckwg +5
 * Copyright 2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "ros_detector_scaling.h"

#include <vital/algo/dynamic_configuration.h>
#include <vital/algo/algorithm_factory.h>

// ------------------------------------------------------------------
ros_detector_scaling::
ros_detector_scaling()
  : m_scaleFactor(1.0)
{
  // Allocate config block
  m_config = kwiver::vital::config_block::empty_config();
}


// ------------------------------------------------------------------
ros_detector_scaling::
~ros_detector_scaling()
{}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
ros_detector_scaling::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  // Could configure the topic to listen on
  config->set_value( "topic", "scale_factor",
		     "ROS Topic name to subscribe. This topic will supply the float64 scaling value "
		     "from (0 - 1)." );

  return config;
}


// ------------------------------------------------------------------
void
ros_detector_scaling::
set_configuration( kwiver::vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  kwiver::vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  this->m_topic = config->get_value<std::string>( "topic", "scale_factor" );

  // Need to delay initializing the ROS interface until we have a good
  // config.  Can not be done in CTOR because objects are created for
  // introspection without valid ROS environment.
  ros::init( ros::M_string(), "ros_detector_scaling" );
  m_node = std::make_shared<ros::NodeHandle>("~");

  // Set up callback for input topic
  m_sub = m_node->subscribe( this->m_topic, 1, &ros_detector_scaling::callbackEvent, this );

  // Start ROS spinner
  ros::AsyncSpinner spinner( 1 );
  spinner.start();
}


// ------------------------------------------------------------------
bool
ros_detector_scaling::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
ros_detector_scaling::
get_dynamic_configuration()
{
  m_config->set_value("scale_factor", this->m_scaleFactor );

  return m_config;
}


// ------------------------------------------------------------------
// accepts a message of expected type.
void
ros_detector_scaling::
callbackEvent(const std_msgs::Float64& msg )
{
  double factor = msg.data;

  // validate stane factor
  if (factor < 0)
    {
      factor = 0;
      ROS_WARN( "Scaling value less than 0. Set to 0." );
    }
  else if (factor > 1)
    {
      factor = 1.0 ;
      ROS_WARN( "Scaling factor greater than 1. Set to 1." );
    }

  // save value in local storage;
  this->m_scaleFactor = factor;
}


// ==================================================================
// Register this as a plugin
extern "C"
ROS_DETECTOR_SCALING_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "kamera.ros.ros_detector_scaling" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory                  implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "ros_detector_scaling", ros_detector_scaling );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
		       "Proivides dynamic scale factor.\n\n"
		       "Listens on topic \"scale_factor\" for ros::Float64 message. "
		       "Supplies value as \"scale_factor\" in the config block." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}
