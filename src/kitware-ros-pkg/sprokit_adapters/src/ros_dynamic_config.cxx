/*ckwg +5
 * Copyright 2017 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "ros_dynamic_config.h"

#include <vital/logger/logger.h>
#include <vital/algo/dynamic_configuration.h>
#include <vital/algo/algorithm_factory.h>

// ------------------------------------------------------------------
ros_dynamic_config::
ros_dynamic_config()
{
  // Allocate config block
  m_config = kwiver::vital::config_block::empty_config();
}


// ------------------------------------------------------------------
ros_dynamic_config::
~ros_dynamic_config()
{}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
ros_dynamic_config::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  // Could configure the topic to listen on
  config->set_value( "topic", "dyn_config",
		     "ROS Topic name to subscribe. This topic will supply "
                     "a diagnostic_msgs::DiagnosticStatus message that contains the key/value "
                     "pairs that are copied to the config block." );

  return config;
}


// ------------------------------------------------------------------
void
ros_dynamic_config::
set_configuration( kwiver::vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  kwiver::vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  m_topic = config->get_value<std::string>( "topic", "dyn_config" );

  // Need to delay initializing the ROS interface until we have a good
  // config.  Can not be done in CTOR because objects are created for
  // introspection without valid ROS environment.
  ros::init( ros::M_string(), "ros_dynamic_config" );
  m_node = std::make_shared<ros::NodeHandle>("~");

  // Set up callback for input topic
  m_sub = m_node->subscribe( this->m_topic, 1, &ros_dynamic_config::callbackEvent, this );

  // Start ROS spinner
  ros::AsyncSpinner spinner( 1 );
  spinner.start();
}


// ------------------------------------------------------------------
bool
ros_dynamic_config::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
ros_dynamic_config::
get_dynamic_configuration()
{
  std::lock_guard<std::mutex> lock(m_config_lock);

  return m_config;
}


// ------------------------------------------------------------------
// accepts a message of expected type.
void
ros_dynamic_config::
callbackEvent(const diagnostic_msgs::DiagnosticStatus& msg )
{
  // Start with a new config block so we can add entries with out
  // colliding with the get_dynamic_configuration() method since we
  // are running in separate threads.
  auto config = kwiver::vital::config_block::empty_config();

  for ( auto kv : msg.values )
  {
    LOG_DEBUG( logger(), "Adding config entry to dynamic set - "
               << kv.key << " = " << kv.value );

    config->set_value( kv.key, kv.value );
  }

  // Do an atomic store so we do not get into trouble with the
  // asynchronous client
  std::lock_guard<std::mutex> lock(m_config_lock);

  this->m_config = config;
}


// ==================================================================
// Register this as a plugin
extern "C"
ROS_DYNAMIC_CONFIG_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "kamera.ros.ros_dynamic_config" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory                  implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "ros_dynamic_config", ros_dynamic_config );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
		       "Proivides dynamic configuration values.\n\n"
		       "Listens on topic \"dyn_config\" by default for ROS "
                       "diagnostic_msgs::DiagnosticStatus message. "
		       "Puts all key/value pairs in the config block." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}
