#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/image_encodings.h>

// Genicam stuff
#include "GenApi/GenApi.h"    //!< GenApi lib definitions.
#include "gevapi.h"       //!< GEV lib definitions.

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <sstream>
#include <signal.h>

#include "decode_error.h"

#define CAMERA_INDEX 0
#define PUB_NAME  "raw_image"

/*

ros configuration items:

ip_addr - Camera address in dotted decilal "xx.xx.xx.xx"
          Selects camera based on IP address

camera_index - camera index number. A camera can be selected using its
               SDK index. This does not work if ip_addr ias been
               selected.

num_buffers - number of application level buffers to allocate. This
              value has some latency implications and interacts with
              device buffering in an, as yet, unknown manner.

frame_id - String to use to identify the image stream source. Default
           value is "/cueing/right".

frame_rate - frame rate for the camera.

output_frame_rate - rate of published images.

*/

// Number of application buffers
#define NUM_BUF 32 // 8, 32

// Number of frames buffered internally
#define INT_FR_BUF 16 // 4, 16

#define MAX_NETIF         2
#define MAX_CAMERAS_PER_NETIF 32
#define MAX_CAMERAS   ( MAX_NETIF * MAX_CAMERAS_PER_NETIF )


// --- global data areas --
bool G_interrupt_seen(false);
GEV_CAMERA_HANDLE G_camera_handle = NULL;
GEV_DEVICE_INTERFACE  G_pCamera[MAX_CAMERAS] = { 0 };
unsigned long G_ip_addr;
UINT32 G_height = 0;
UINT32 G_width = 0;
UINT32 G_x_offset = 0;
UINT32 G_y_offset = 0;
UINT32 G_format = 0;
UINT32 G_buffer_size = 0;
PUINT8 G_buf_address[NUM_BUF];
int G_num_buffers(NUM_BUF);
float G_frame_rate(21.0); // frame rate requested from camera
float G_output_frame_rate(10.0); // frame rate published


// ----------------------------------------------------------------------------
void display_camera_info()
{
  GEV_CAMERA_INFO *info = GevGetCameraInfo(G_camera_handle);

#define IP_ADDR(IP) (((IP) >> 24) & 0xff) << "." << (((IP) >> 16) & 0xff) << "." << (((IP) >> 8) & 0xff) << "." << (((IP) >> 0) & 0xff)
  std::cout << "fIPv6: " <<  info->fIPv6 << std::endl
            << "ipAddr: " << IP_ADDR(info->ipAddr) << std::endl
            << "ipAddrLow: " << IP_ADDR(info->ipAddrLow) << std::endl
            << "ipAddrHIGH: " << IP_ADDR(info->ipAddrHigh) << std::endl
            << "Mac addr: " << IP_ADDR(info->macLow) << ":" << IP_ADDR(info->macHigh) << std::endl
            << "Mfgr: " << info->manufacturer << std::endl
            << "Model: " << info->model << std::endl
            << "Serial: " << info->serial << std::endl
            << "Version: " << info->version << std::endl
            << "Username: " << info->username << std::endl
    ;

#undef IP_ADDR
}


// ----------------------------------------------------------------------------
void display_buffer( GEV_BUFFER_OBJECT* imbuf )
{
  std::cout << "state: " << imbuf->state << std::endl
            << "status: " << imbuf->status << std::endl
            << "timestamp_hi: " << imbuf->timestamp_hi << std::endl
            << "timestamp_lo: " << imbuf->timestamp_lo << std::endl
            << "recv_size: " << imbuf->recv_size << std::endl
            << "id: " << imbuf->id << std::endl
            << "height: " << imbuf->h << std::endl
            << "width: " << imbuf->w << std::endl
            << "x_offset: " << imbuf->x_offset << std::endl
            << "y_offset: " << imbuf->y_offset << std::endl
            << "x_padding: " << imbuf->x_padding << std::endl
            << "y_padding: " << imbuf->y_padding << std::endl
            << "bytes per pixel: " << imbuf->d << std::endl
            << "format: " << decode_pixel_format( imbuf->format ) << std::endl
            << "address: 0x" << std::hex << static_cast<void*>( imbuf->address) << std::endl
            << std::dec
    ;

}


// ----------------------------------------------------------------------------
void display_camera_options( const GEV_CAMERA_OPTIONS& opt )
{
#define P( F ) #F << " :" << opt.F << std::endl

  std::cout << P(numRetries)
            << P(command_timeout_ms)
            << P(heartbeat_timeout_ms)
            << P(streamPktSize)
            << P( streamPktDelay)
            << P( streamNumFramesBuffered)
            << P( streamMemoryLimitMax)
            << P( streamMaxPacketResends)
            << P( streamFrame_timeout_ms)
            << P( streamThreadAffinity)
            << P( serverThreadAffinity)
            << P( msgChannel_timeout_ms)
    ;

#undef P
}


// ---------------------------------------------------------------------------------
UINT32 parseIPV4string(const std::string& ipAddress)
{
  unsigned char ipbytes[4];
  if ( sscanf(ipAddress.c_str(), "%hhu.%hhu.%hhu.%hhu",
              &ipbytes[3], &ipbytes[2], &ipbytes[1], &ipbytes[0]) != 4)
  {
    std::cerr << "Error parsing IP address: \"" << ipAddress << "\"\n";
    return 0;
  }

  return ((UINT32)ipbytes[0]) | ((UINT32)ipbytes[1]) << 8 | ((UINT32)ipbytes[2]) << 16 | ((UINT32)ipbytes[3]) << 24;
}


// ----------------------------------------------------------------------------
GEV_STATUS report_if_error( GEV_STATUS status, const std::string& msg )
{
  if (status != 0)
  {
    std::cerr <<  "Error " << msg << " - " << decode_sdk_status(status) << "\n";
  }

  return status;
}


// ==================================================================
void
sigint_handler( int sig )
{
  G_interrupt_seen = true;
}


// ----------------------------------------------------------------------------
bool open_camera( ros::NodeHandle pnh )
{
  ROS_INFO( "Opening a camera..." );
  UINT16 status(0);

  // Open camera given IP address
  if ( pnh.hasParam("ip_addr") )
  {
    ROS_INFO( "... by IPv4 address ..." );

    std::string ip_string;
    pnh.getParam( "ip_addr", ip_string);
    ROS_INFO( "    - ip_addr: %s", ip_string.c_str() );

    G_ip_addr = parseIPV4string(ip_string);
    if (G_ip_addr == 0)
    {
      return false;
    }

    status = report_if_error( GevOpenCameraByAddress( G_ip_addr, // i: ip addreess of camera
                                                      GevExclusiveMode,  // i: open mode
                                                      &G_camera_handle), // o: camera handle
                              "opening camera by IP address" );
  }
  else
  {
    ROS_INFO( "... by discovered camera index ..." );
    // No IP address given. Open the first camera found;
    int numCamera = 0;

    // Select the first camera found if no other is specified
    status = report_if_error( GevGetCameraList( G_pCamera, MAX_CAMERAS, &numCamera ), "get camera list" );

    printf( "%d camera(s) on the network\n", numCamera );

    if ( numCamera == 0 )
    {
      return false;
    }

    int camIndex = CAMERA_INDEX;
    if ( pnh.hasParam("camera_index"))
    {
      pnh.getParam( "camera_index", camIndex);
    }

    if ( camIndex >= (int)numCamera )
    {
      printf( "Camera index out of range - only %d camera(s) are present\n", numCamera );
      return false;
    }

    std::cout << "Selecting camera " << camIndex << std::endl;

    // Open the camera.
    status = report_if_error( GevOpenCamera( &G_pCamera[camIndex],
                                             GevExclusiveMode,  // i: open mode
                                             &G_camera_handle ),
                              "opening camera" );
  }

  if (status != 0)
  {
    return false;
  }

  return true;
}


// ----------------------------------------------------------------------------
bool allocate_buffers(ros::NodeHandle pnh, UINT32 format)
{
  // number of image buffers. If there are too few, images will be
  // dropped.  If there are too many, it increases latency.
  if ( pnh.hasParam("num_buffers"))
  {
    // pnh.getParam( "num_buffers", G_num_buffers);
  }

  // Allocate and format image buffers
  UINT32 maxDepth = GetPixelSizeInBytes( format );

  // Allocate image buffers
  G_buffer_size = maxDepth * G_width * G_height;
  for ( int i = 0; i < G_num_buffers; i++ )
  {
    G_buf_address[i] = (PUINT8) malloc( G_buffer_size );
    memset( G_buf_address[i], 0, G_buffer_size );
  }

  return true;
}



// ----------------------------------------------------------------------------
bool get_camera_info()
{
  UINT16 status(0);

  //=====================================================================
  // Get the GenICam FeatureNodeMap object and access the camera features.
  static GenApi::CNodeMapRef* Camera = static_cast< GenApi::CNodeMapRef* >( GevGetFeatureNodeMap( G_camera_handle ) );

  if ( Camera )
  {
    // Access some features using the bare GenApi interface methods
    try
    {
      //Mandatory features....
      GenApi::CIntegerPtr ptrIntNode = Camera->_GetNode( "Width" );
      G_width = (UINT32) ptrIntNode->GetValue();

      ptrIntNode = Camera->_GetNode( "Height" );
      G_height = (UINT32) ptrIntNode->GetValue();

      GenApi::CEnumerationPtr ptrEnumNode = Camera->_GetNode( "PixelFormat" );
      G_format = (UINT32)ptrEnumNode->GetIntValue();


      GenApi::CFloatPtr ptrFloatNode = Camera->_GetNode( "AcquisitionFrameRate" );
      if (ptrFloatNode.IsValid())
      {
        double rate = (float) ptrFloatNode->GetValue();
        std::cout << "Acquisition rate: " << rate << "\n";
        ptrFloatNode->SetValue(G_frame_rate);
        rate = (float) ptrFloatNode->GetValue();
        std::cout << "Acquisition new rate: " << rate << "\n";
      }
      else
      {
        std::cout << "Acquisition frame rate not available\n";
      }
    }
    // Catch all possible exceptions from a node access.
    CATCH_GENAPI_ERROR( status );
  }

  if (status != 0)
  {
    std::cerr << "Caught exception\n";
    return false;
  }

  return true;
}


// ----------------------------------------------------------------------------
bool get_camera_XML()
{
  UINT16 status(0);

  // Initiliaze access to GenICam features via Camera XML File
  // Not sure if all this XML stuff is needed
  status = GevInitGenICamXMLFeatures( G_camera_handle, // i: the handle
                                      false ); // i: TRUE updates XML file
  if ( status == GEVLIB_OK )
  {
    // Get the name of XML file name back (example only - in case you need it somewhere).
    char xmlFileName[MAX_PATH] = { 0 };
    status = GevGetGenICamXML_FileName( G_camera_handle, (int)sizeof( xmlFileName ), xmlFileName );
    if ( status == GEVLIB_OK )
    {
      std::cout << "XML stored as - " << xmlFileName << std::endl;
    }

    // can use GevGetFeatureValue() and GevSetFeatureValue() to work with features.

    /*
    // Code to get device temp to do high temp warning
    int type;
    float val;
    status = GevGetFeatureValue( G_camera_handle, "DeviceTemperature", &type, sizeof( val ), &val );
    if (status != GEVLIB_OK )
    {
      // print error message
    }
    // If temp is above threshold, print message.
    */
  }

  if ( status != GEVLIB_OK )
  {
    std::cerr <<  "Error getting XML features - " << decode_sdk_status(status) << "\n";
    return false;
  }

  { // limit scope of camera options
    // Adjust camera options
    GEV_CAMERA_OPTIONS camOptions = { 0 };

    // Adjust the camera interface options if desired (see the manual)
    GevGetCameraInterfaceOptions( G_camera_handle, &camOptions );

    camOptions.heartbeat_timeout_ms = 3000; // 10000 initially

    // Some tuning can be done here. (see the manual)
    camOptions.streamNumFramesBuffered = INT_FR_BUF;       // Buffer frames internally. (4)
    camOptions.numRetries = 10;

    int sfto = (camOptions.streamNumFramesBuffered -1) * (1/G_frame_rate) * 1000;
    camOptions.streamFrame_timeout_ms = 3001;       // Internal timeout for frame reception. (1001)

    camOptions.streamMemoryLimitMax = 64 * 1024 * 1024;   // Adjust packet memory buffering limit. (64m)
    camOptions.streamPktSize = 9180;            // Adjust the GVSP packet size. (9180)
    camOptions.streamPktDelay = 10;             // Add usecs between packets to pace arrival at NIC.

    // Assign specific CPUs to threads (affinity) - if required for better performance.
    if (0)
    {
      int numCpus = _GetNumCpus();
      if ( numCpus > 1 )
      {
        camOptions.streamThreadAffinity = numCpus - 1;
        camOptions.serverThreadAffinity = numCpus - 2;
      }
    }

    // Write the adjusted interface options back.
    GevSetCameraInterfaceOptions( G_camera_handle, &camOptions );

    std::cout << "-- From camera --\n";
    display_camera_options( camOptions );

  } // end camera options

  return true;
}


// ============================================================================
int main(int argc, char**argv)
{
  ros::init(argc, argv, "kw_genicam_driver");
  ros::NodeHandle pnh("~");

  // camera specific data areas
  UINT16 status(0);

  report_if_error( GevApiInitialize(), "API initialize" );

  // Set default options for the library.
  {
    GEVLIB_CONFIG_OPTIONS options = { 0 };

    GevGetLibraryConfigOptions( &options );
    // options.logLevel = GEV_LOG_LEVEL_OFF;
    options.logLevel = GEV_LOG_LEVEL_TRACE;
    // options.logLevel = GEV_LOG_LEVEL_NORMAL;
    GevSetLibraryConfigOptions( &options );
  }

  // Start with default value
  std::string C_frame_id( "/cueing/right");
  if ( pnh.hasParam("frame_id"))
  {
    pnh.getParam( "frame_id", C_frame_id);
  }

  if ( pnh.hasParam("frame_rate"))
  {
     pnh.getParam( "frame_rate", G_frame_rate);
  }

  if ( pnh.hasParam("output_frame_rate"))
  {
     pnh.getParam( "output_frame_rate", G_output_frame_rate);
  }

  std::string C_output_topic_name_color( PUB_NAME );
  if ( pnh.hasParam( "output_topic_name_color" ) )
  {
    pnh.getParam( "output_topic_name_color", C_output_topic_name_color );
  }

  std::string C_output_topic_name_bayer( "bayer_" PUB_NAME );
  if ( pnh.hasParam( "output_topic_name_bayer" ) )
  {
    pnh.getParam( "output_topic_name_bayer", C_output_topic_name_bayer );
  }

  // Set up our outbound image transport
  image_transport::ImageTransport it(pnh);
  image_transport::Publisher it_pub = it.advertise(C_output_topic_name_color, 1);

  image_transport::ImageTransport bayer_it(pnh);
  image_transport::Publisher bayer_it_pub = it.advertise(C_output_topic_name_bayer, 1);

  // counters used for reducing the output data rate
  const int frame_rate_divisor( G_frame_rate / G_output_frame_rate );
  int frame_counter(0);

  ROS_INFO( "Overriding SIGINT signal handler" );
  signal( SIGINT, sigint_handler );

  // find a camera
  if ( ! open_camera( pnh ) )
  {
    ROS_ERROR( "Failed to open camera" );
    goto exit_spot;
  }

  // Initialize camera. Quit if init fails.
  if ( ! get_camera_XML() )
  {
    goto exit_spot;
  }

  if ( ! get_camera_info() )
  {
    goto exit_spot;
  }

  std::cout << "Output frame rate: " << G_output_frame_rate << std::endl;
  std::cout << "Frame rate divisor: " << frame_rate_divisor << std::endl;

  display_camera_info(); //+ temp ----------------

  allocate_buffers(pnh, G_format);

  // Initialize a transfer with synchronous buffer handling.
  status = GevInitImageTransfer( G_camera_handle, // i: camera handle
                                 SynchronousNextEmpty, // i: buffer handling mode
                                 G_num_buffers,        // i: number of buffers
                                 G_buf_address );      // i: buffer address list

  if (status != GEVLIB_OK)
  {
    std::cerr << "Error initializing image transfer - " << decode_sdk_status(status) << std::endl;
    goto exit_spot;
  }

  // clear the buffers - may not be needed since they are cleared when created.
  for ( int i = 0; i < G_num_buffers; i++ )
  {
    memset( G_buf_address[i], 0, G_buffer_size );
  }

  status = GevStartImageTransfer( G_camera_handle, -1 );
  if ( status != 0 )
  {
    std::cerr << "Error starting grab - " << decode_sdk_status(status) << std::endl;
    goto exit_spot;
  }

  // -------- Read image - publish image - repeat --------
  while(ros::ok())
  {
    if (G_interrupt_seen) break;

    static sensor_msgs::Image ros_image; // the message
    GEV_BUFFER_OBJECT* image_object_ptr = NULL;

    status = GevWaitForNextImage (G_camera_handle, // i: camera handle
                                  &image_object_ptr, // o: the image object
                                  2000); // i: timeout in msec

    if ( ( image_object_ptr == NULL ) || ( status != GEVLIB_OK ) )
    {
      // log this event and continue
      std::cout << "Timeout waiting for image - " << decode_sdk_status(status) << std::endl;
      continue;
    }

    if ( image_object_ptr->status != 0 )
    {
      // Image had an error (incomplete (timeout/overflow/lost)).
      // Do any handling of this condition necessary.
      std::cout << "Got image with error: " << image_object_ptr->status
                << " - " << decode_sdk_status(image_object_ptr->status) <<"\n";

      // Release the buffer back to the image transfer process.
      GevReleaseImage( G_camera_handle, image_object_ptr );

      continue;
    }

    std::cout << "-- Got new image " << image_object_ptr->id << std::endl;

    // display_buffer( image_object_ptr ); //+ ------- temp

    // See if we are to publish the image
    if (++frame_counter >= frame_rate_divisor)
    {
      frame_counter = 0;

      //
      // Publish colour image
      //
      if ( it_pub.getNumSubscribers() > 0 )
      {
        // Copy the data into an OpenCV Mat structure
        cv::Mat raw_image(G_height, G_width, CV_8UC1, image_object_ptr->address );

        // Convert Bayer to BGR8 format
        cv::Mat bgr_image(G_height, G_width, CV_8UC3);
        cv::cvtColor(raw_image,            // i: input image
                     bgr_image,            // o: converted image
                     cv::COLOR_BayerBG2BGR, 3 ); // i: conversion specification

        // Publish the image.
        cv_bridge::CvImagePtr cv_ptr( new cv_bridge::CvImage );
        cv_ptr->image = bgr_image;
        cv_ptr->encoding = "bgr8";

        cv_ptr->header.stamp = ros::Time::now();
        cv_ptr->header.frame_id = C_frame_id;
        it_pub.publish( cv_ptr->toImageMsg() );

        std::cout << "Publish id: " << image_object_ptr->id << "\n";
      }

      // Publish the unconverted original smokey-the-Baer image
      if ( bayer_it_pub.getNumSubscribers() > 0 )
      {
        // Copy the data into an OpenCV Mat structure
        cv::Mat raw_image(G_height, G_width, CV_8UC1, image_object_ptr->address );

        // Publish the image.
        cv_bridge::CvImagePtr cv_ptr( new cv_bridge::CvImage );
        cv_ptr->image = raw_image;
        cv_ptr->encoding = "mono8";

        cv_ptr->header.stamp = ros::Time::now();
        cv_ptr->header.frame_id = C_frame_id;
        bayer_it_pub.publish( cv_ptr->toImageMsg() );
      }
    }

    // Release the buffer back to the image transfer process.
    GevReleaseImage( G_camera_handle, image_object_ptr );

  } // end while

exit_spot:
  if (G_camera_handle != NULL)
  {
    GevStopImageTransfer( G_camera_handle );

    GevAbortImageTransfer( G_camera_handle );
    GevFreeImageTransfer( G_camera_handle );
    GevCloseCamera( &G_camera_handle );
  }

  // Close down the API.
  GevApiUninitialize();

  // Close socket API
  _CloseSocketAPI();  // must close API even on error

  return 0;
} // main
