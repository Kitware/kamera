
#include <arrows/ros/kwiver_algo_dynamic_config_ros_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/ros/dynamic_config_ros.h>

namespace kwiver {
namespace arrows {
namespace darknet {

extern "C"
KWIVER_ALGO_DARKNET_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.dynamic_config_ros" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "dynamic_config_ros", kwiver::arrows::darknet::dynamic_config_ros );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Dynamic Configuration from ROS parameter server" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} } } // end namespace

