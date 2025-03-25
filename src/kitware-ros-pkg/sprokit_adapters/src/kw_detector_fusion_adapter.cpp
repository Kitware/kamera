/*ckwg +5
 * Copyright 2019 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/String.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <custom_msgs/ImageSpaceDetection.h>
#include <custom_msgs/ImageSpaceDetectionList.h>
#include <custom_msgs/SynchronizedImages.h>

#include <sprokit/processes/adapters/embedded_pipeline.h>
#include <arrows/ocv/image_container.h>
#include <sprokit/pipeline_util/literal_pipeline.h>
#include <vital/types/vector.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <roskv/envoy.h>

#include <sys/stat.h>
#include <fstream>
#include <signal.h>
#include <string>
#include <ostream>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <boost/filesystem.hpp>

// GLOBAL pointer to embedded pipeline
kwiver::embedded_pipeline* g_pep;


// ===============================================================
struct InputMetadata
{
  // input message header
  std_msgs::Header m_header;

  // Input image size
  int m_height;
  int m_width;
};

// check if file exists for sync node
inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


// For sorting detections in decreasing order with std::sort
static
bool
greater_detection_score ( kwiver::vital::detected_object_sptr a,
                          kwiver::vital::detected_object_sptr b)
{
  auto ob_class_a = a->type();
  auto ob_class_b = b->type();

  if( ob_class_a->size() == 0 )
  {
    return false;
  }
  else if( ob_class_b->size() == 0 )
  {
    return true;
  }

  std::string max_name;
  double max_score_a;
  double max_score_b;
  ob_class_a->get_most_likely( max_name, max_score_a );
  ob_class_b->get_most_likely( max_name, max_score_b );
  return max_score_b < max_score_a;
}


// Random string generator.
// TODO: This is a naive generator. Replace with appropriate KWIVER version.
class random_string_generator
{
private:
  const char charset[63] =
  "0123456789"
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz";
  const size_t max_index = (sizeof(charset) - 1);

  char
  randchar()
  {
    return charset[ std::rand() % max_index ];
  }

public:
  std::string
  random_string( size_t length )
  {
    std::string rand_str( length, 0 );
    for (int i = 0; i < length; ++i) {
        rand_str[i] = randchar();
    }
    return rand_str;
  }
};


// ===============================================================
/**
 * @brief Subscriber handler class.
 *
 * This class represents a subscriber callback for an image.
 * The image is converted to acceptable format and pushed into the pipeline.
 */
class AdapterCallback
{
public:
  AdapterCallback( ros::NodeHandle &nh,
                   kwiver::embedded_pipeline* pipeline_ptr,
                   std::string topic, int sync_q_size, int rgb_port_ind,
		   int ir_port_ind, int uv_port_ind)
    : m_pep( pipeline_ptr ),
      m_topic( topic ),
      m_rgb_port_ind( rgb_port_ind ),
      m_ir_port_ind( ir_port_ind ),
      m_uv_port_ind( uv_port_ind )
  {
    // Set up callback for input topic depending on the image message type
    ROS_INFO_STREAM( "Subscribing to SynchronizedImages topic: " << topic );
    m_synchronized_images_sub = nh.subscribe( topic, sync_q_size,
                                              &AdapterCallback::synchronizedImagesCallback, this );
  }

  // ROS callback for SynchronizedImages
  void synchronizedImagesCallback( const custom_msgs::SynchronizedImagesConstPtr& msg )
  {
    // Create dataset for input
    auto ds = kwiver::adapter::adapter_data_set::create();

    if ( m_rgb_port_ind != 0 )
    {
      // Process RGB image, reading from disk if data is not present in sync msg
      cv::Mat cv_image;
      std::string file_name = msg->file_path_rgb;
      if ( msg->image_rgb.height > 0 || msg->image_rgb.width > 0 ) {
        cv_image = cv_bridge::toCvCopy( msg->image_rgb, "rgb8" )->image;
      } else if ( msg->image_rgb.data.empty() ) {
          if ( file_exists(file_name) ) {
	          cv_image = cv::imread(file_name, cv::IMREAD_COLOR);
	          if ( cv_image.empty() ) {
	              ROS_ERROR_STREAM("Could not read image from disk: " << file_name.c_str());
                  return;
	          } else {
	              ROS_INFO("Successfully read rgb image from disk.");
	          }
	      } else {
	            ROS_ERROR_STREAM("RGB file name does not exist " << file_name.c_str());
            return ;
	      }
      } else {
           ROS_ERROR("RGB image is null-ish ");
           return ;
      }

      // Put OCV image into vital container
      ROS_INFO_STREAM( "Pushing RGB image into pipeline" );
      kwiver::vital::image_container_sptr img( new kwiver::arrows::ocv::image_container
        ( cv_image, kwiver::arrows::ocv::image_container::ColorMode::RGB_COLOR ) );
      ds->add_value( "image", img );

      ROS_INFO_STREAM( "Pushing RGB file_name '" << file_name << "' to pipeline" );

      if ( m_rgb_port_ind == 1 )
      {
        ds->add_value( "file_name", file_name );
      }
      else if ( m_rgb_port_ind == 2 )
      {
        ds->add_value( "file_name2", file_name );
      }
      else if ( m_rgb_port_ind == 3 )
      {
        ds->add_value( "file_name2", file_name );
      }
      else
      {
        ROS_ERROR("Invalid port index for RGB");
      }
    }

    if ( m_ir_port_ind != 0 )
    {
      // Process IR image, reading from disk if data is not present in sync msg
      cv::Mat cv_image2;
      std::string file_name = msg->file_path_ir;
      if ( msg->image_ir.height > 0 || msg->image_ir.width > 0 ) {
          // data exists
          cv_image2 = cv_bridge::toCvCopy( msg->image_ir, "mono16" )->image;
          // find if NUCing or not
          std::string frame_id = msg->image_ir.header.frame_id;
          std::string delimiter = "nucing=";
          int ind = frame_id.find(delimiter);
          std::string token = frame_id.substr(ind+7, ind+8);
          int is_nucing = std::stoi(token);
          if (is_nucing == 1) {
              // overload filename
              file_name = "nucing";
              // return;
          }
      } else if ( msg->image_ir.data.empty() ) {
          if ( file_exists(file_name) ) {
	          cv_image2 = cv::imread(file_name, cv::IMREAD_ANYDEPTH);
	          if ( cv_image2.empty() ) {
	              ROS_ERROR_STREAM("Could not read image from disk: " << file_name.c_str());
                  return;
	          } else {
	              ROS_INFO("Successfully read IR image from disk.");
	          }
	      } else {
	            ROS_ERROR_STREAM("IR file name does not exist " << file_name.c_str());
            return ;
	      }
      } else {
           ROS_ERROR("IR image is null-ish ");
           return ;
      }

      // Put OCV image into vital container
      ROS_INFO_STREAM( "Pushing IR image into pipeline" );
      kwiver::vital::image_container_sptr img2( new kwiver::arrows::ocv::image_container
        ( cv_image2, kwiver::arrows::ocv::image_container::ColorMode::RGB_COLOR ) );
      ds->add_value( "image2", img2 );

      ROS_INFO_STREAM( "Pushing IR file_name '" << file_name << "' to pipeline" );

      if ( m_ir_port_ind == 1 )
      {
        ds->add_value( "file_name", file_name );
      }
      else if ( m_ir_port_ind == 2 )
      {
        ds->add_value( "file_name2", file_name );
      }
      else if ( m_ir_port_ind == 3 )
      {
        ds->add_value( "file_name3", file_name );
      }
      else
      {
        ROS_ERROR("Invalid port index for IR");
      }
    }

    m_pep->send( ds );
  }

  std::string get_frame_id() const
  {
    return m_frame_id;
  }


private:
  kwiver::embedded_pipeline* m_pep;
  ros::Subscriber m_synchronized_images_sub;
  std::string m_frame_id;
  std::string m_topic;
  int m_rgb_port_ind;
  int m_ir_port_ind;
  int m_uv_port_ind;
};


// ==================================================================
void
sigint_handler( int sig )
{
  g_pep->send_end_of_input();
  ros::shutdown();
}


// ===============================================================
/**
 * @brief Launches the ROS detector adapter node.
 *
 * See README.rst for more details.
 */
int
main( int argc, char** argv )
{
  // Seed with product of time and pointer. Since time is integer seconds,
  // starting two of these processes within the same second would otherwise
  // cause duplicate seeds.
  int time_int = time( NULL );
  int random_int = reinterpret_cast<long>(&time_int);
  std::cout << "Seeding random number generation using product of time " <<
            std::to_string(time_int) << " and pointer " <<
            std::to_string(random_int) << std::endl;
  srand( time_int*random_int );

  random_string_generator string_generator;

  std::ofstream debug_out("/root/kamera_ws/image_id.txt");
  ros::init( argc, argv, "sprokit_detector_fusion_adapter" );
  ros::NodeHandle nh_pub;
  ros::NodeHandle nh_priv("~");

  std::string redis_uri;
  if ( ! nh_priv.getParam("redis_uri", redis_uri) ) {
    ROS_ERROR( "'redis_uri' not found in parameters." );
    return -1;
  }

  RedisEnvoyOpts envoy_opts = RedisEnvoyOpts{"detector_envoy", redis_uri, "/agent/default"};
  std::shared_ptr<RedisEnvoy> envoy = std::make_shared<RedisEnvoy>(envoy_opts);

  // Find pipeline file name from parameters (see README)
  // get hostname from param namespace
  std::string ns = nh_priv.getNamespace();
  std::ostringstream oss;
  std::ostringstream healthss;
  healthss << ns << "/health";
  oss << "/sys" << ns << "/pipefile";
  std::string health_param = healthss.str();
  std::string redis_pipefile = oss.str();
  // Try and get pipefile from redis, if fails, get from rosparam
  std::string pipe_file;
  try {
    ROS_INFO("Trying to get Redis pipefile from: %s.", redis_pipefile.c_str());
    pipe_file = envoy->get(redis_pipefile);
  } catch( std::invalid_argument e ) {
      ROS_WARN("No Redis failed for pipefile, falling back to rosparam.");
      if ( ! nh_priv.getParam("pipe_file", pipe_file) ) {
        ROS_ERROR( "'pipe_file' not found in parameter path <<." );
        return -1;
      } else {
        ROS_INFO( "Setting Redis pipefile based off ros param." );
        envoy->put(redis_pipefile, pipe_file);
      }
    }

  std::string pipeline_dir;
  if ( ! nh_priv.param<std::string>("pipeline_dir", pipeline_dir, "")) {
    ROS_WARN( "'pipeline_dir' file not found in parameters" );
  }
  // Open pipeline description
  std::ifstream pipe_str;
  pipe_str.open( pipe_file, std::ifstream::in );
  if ( ! pipe_str )
  {
    // issue error message
    ROS_ERROR( "Could not open pipeline file %s", pipe_file.c_str() );
    return -1;
  }
  else
  {
    ROS_INFO( "Opening pipeline file %s", pipe_file.c_str() );
  }

  if (pipeline_dir.empty()) {
    ROS_WARN( "'pipeline_dir' file not found in parameters. Defaulting to '`dirname pipe_file`" );
    boost::filesystem::path pipefile_path(pipe_file);
    pipefile_path.remove_filename();
    pipeline_dir = pipefile_path.string();
  }
  ROS_INFO("pipeline_dir=%s", pipeline_dir.c_str());

  int rgb_port_ind;
  if ( ! nh_priv.getParam("rgb_port_ind", rgb_port_ind) )
  {
    // Entry not found, use default name
    ROS_WARN( "'rgb_port_ind' not found, RGB image will not be sent to pipeline." );
    rgb_port_ind = 0;
  }

  int ir_port_ind;
  if ( ! nh_priv.getParam("ir_port_ind", ir_port_ind) )
  {
    // Entry not found, use default name
    ROS_WARN( "'ir_port_ind' not found, RGB image will not be sent to pipeline." );
    ir_port_ind = 0;
  }

  int uv_port_ind;
  if ( ! nh_priv.getParam("uv_port_ind", uv_port_ind) )
  {
    // Entry not found, use default name
    ROS_WARN( "'uv_port_ind' not found, RGB image will not be sent to pipeline." );
    uv_port_ind = 0;
  }

  // Get detector ID string, which identifies the detector used (see README).
  std::string detector_id_string;
  if( ! nh_priv.getParam( "detector_id_string", detector_id_string ) )
  {
    // Entry not found, use default name
    ROS_WARN( "'detector_id_string' not defined, defaulting to 'unspecified'." );
    detector_id_string = "unspecified";
  }
  else
  {
    ROS_INFO( "'detector_id_string' set to '%s'", detector_id_string.c_str() );
  }

  // OpenCV Threading Value
  int ocv_num_threads;
  if( ! nh_priv.getParam( "ocv_num_threads", ocv_num_threads ) )
  {
    ROS_WARN( "'ocv_num_threads' not defined, defaulting to -1 (serial execution)." );
    ocv_num_threads = -1;
  }

  // ROS sync queue value
  int sync_q_size;
  if( ! nh_priv.getParam( "sync_q_size", sync_q_size ) )
  {
    ROS_WARN( "'sync_q_size' not defined, defaulting to 5,000." );
    sync_q_size = 5000;
  }

  // 0 means "OpenCV will disable threading optimizations and run all its functions sequentially"
  // <0 means default allocation.
  ROS_INFO( "Asking OpenCV to use %d thread.", ocv_num_threads );
  cv::setNumThreads( ocv_num_threads );

  // Set up pipeline
  ROS_INFO( "Setting up pipeline" );
  kwiver::embedded_pipeline pipeline;
  pipeline.build_pipeline(pipe_str, pipeline_dir);
  g_pep = &pipeline; // copy to GLOBAL space

  pipeline.start(); // start pipeline
  signal( SIGINT, sigint_handler );

  ROS_INFO( "Finished setting up pipeline" );

  // There are an, as of yet, unknown number of image topics that are to be
  // multiplexed through the detector pipeline. So, we incrementally seek
  // parameter "synchronized_images_in_topic#" until we find it not populated or populated
  // with "unused".
  std::vector<AdapterCallback*> input_cbs;

  int i = 1;
  // The annotated image base will be concatenated with the integer image number.
  std::string topic_name;

  // Incrementally seek parameter "synchronized_images_in_topic#" until we find
  // it not populated or populated with "unused".
  while( true )
  {
    std::string topic_param;
    topic_param = std::string( "synchronized_images_in" ) + std::to_string( i );
    if ( nh_priv.getParam( topic_param, topic_name ) )
    {
      if( topic_name == "unused" )
      {
        break;
      }
      // Set callbacks

      // Create instance of image callback for current image topic and add to
      // vector of callback instances.
      ROS_INFO( "Found SynchronizedImages topic %s", topic_name.c_str() );
      input_cbs.push_back( new AdapterCallback( nh_pub, &pipeline, topic_name,
			      			sync_q_size, rgb_port_ind,
					       	ir_port_ind, uv_port_ind ) );
    }
    else if( i == 1)
    {
      ROS_ERROR( "Must at least provide synchronized_images_in1." );
      return -1;
    }
    else
    {
      break;
    }
    ++i;
  }

  ros::Publisher detection_pub;
  detection_pub = nh_priv.advertise< custom_msgs::ImageSpaceDetectionList > ( "detections_out", 20 );

  // Start ROS spinner
  ros::AsyncSpinner spinner( 1 );
  spinner.start();

  json::json health;

  int frame = 0;

  while ( ros::ok() )
  {
    ROS_INFO( "OpenCV thread number: %d", cv::getNumThreads() );
    auto ods = pipeline.receive(); // blocks until data ready

    ROS_INFO( "Pipeline finished" );

    // check for end of data marker
    if ( ods->is_end_of_data() )
    {
      ROS_INFO( "End of data found by node. Waiting for scheduler to complete." );
      pipeline.wait(); // wait for pipeline scheduler to complete
      ROS_INFO( "Waiting for scheduler to ROS shutdown." );
      ros::shutdown();
      return 0;
    }

    // -------------------
    // Process detected objects
    auto ix = ods->find( "detected_object_set" );
    if ( ix == ods->end() )
    {
      ROS_ERROR( "Required \"detected_object_set\" datum not present in output set" );
      continue;
    }
    ROS_INFO( "Received 'detected_object_set' from pipeline" );

    auto det_set = ix->second->get_datum< kwiver::vital::detected_object_set_sptr > ();


    auto iy = ods->find( "file_name" );
    if ( iy == ods->end() )
    {
      ROS_ERROR( "Required \"file_name\" datum not present in output set" );
      continue;
    }
    
    auto ir = ods->find( "file_name2" );
    if ( ir == ods->end() )
    {
      ROS_WARN( "Optional \"file_name2\" datum not present in output set" );
    }

    // File name associated with the image in which the detection bounding boxes
    // are defined.
    auto src_img_fname = iy->second->get_datum< std::string > ();
    auto ir_img_fname = ir->second->get_datum< std::string > ();

    ROS_INFO( "Received 'file_name' %s from pipeline", src_img_fname.c_str() );

    if (ir_img_fname == "nucing") {
      ROS_WARN_STREAM("IR camera is NUCing, skipping detection.");
      double time = ros::Time::now().toSec();
      auto int_dets = 0;
      // Convert things to string for json
      std::string str_dets = std::to_string(int_dets);
      std::string str_time = std::to_string(time);
      health["num_dets"] = str_dets;
      health["time"] = str_time;
      health["src_img_fname"] = src_img_fname;
      health["frame"] = std::to_string( frame );
      health["pipefile"] = pipe_file;
      // Add health check publishing here
      envoy->put_dict(health_param, health);
      frame++;
      // skip publishing detections
      continue;
    }

    custom_msgs::ImageSpaceDetectionList det_list;

    // Get values from the metadata "header" item.
    det_list.header.frame_id = src_img_fname;

    std::vector< kwiver::vital::detected_object_sptr> det_v;
    for ( auto adet : *det_set )
    {
      det_v.push_back( adet );
    }

    // loop over det_v - adding to list
    for ( auto det : det_v )
    {
      custom_msgs::ImageSpaceDetection one_det;
      one_det.header = det_list.header;
      one_det.camera_of_origin = src_img_fname;
      one_det.uid = string_generator.random_string( 20 );
      one_det.detector = detector_id_string;

      auto bbox = det->bounding_box();

      kwiver::vital::vector_2d corner;
      corner = bbox.upper_left();
      one_det.top = corner.y();
      one_det.left = corner.x();

      corner = bbox.lower_right();
      one_det.bottom = corner.y();
      one_det.right = corner.x();

      auto ob_class = det->type();
      if ( ! ob_class )
      {
        // If there are no class names/scores attached to this detection
        continue;
      }

      if ( ob_class->size() > 0 )
      {
        ob_class->get_most_likely( one_det.category, one_det.confidence );
        det_list.detections.push_back( one_det );
      }
    } // end loop

    detection_pub.publish( det_list );

    double time = ros::Time::now().toSec();
    auto int_dets = det_v.size();

    // Convert things to string for json
    std::string str_dets = std::to_string(int_dets);
    std::string str_time = std::to_string(time);
    health["num_dets"] = str_dets;
    health["time"] = str_time;
    health["src_img_fname"] = src_img_fname;
    health["frame"] = std::to_string( frame );
    health["pipefile"] = pipe_file;
    // Add health check publishing here
    envoy->put_dict(health_param, health);
    frame++;

  } // end big while

} // main
