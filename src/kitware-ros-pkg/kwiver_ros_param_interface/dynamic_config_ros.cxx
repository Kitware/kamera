/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

