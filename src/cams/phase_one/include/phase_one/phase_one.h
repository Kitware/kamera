#pragma once
#ifndef PHASE_ONE_H
#define PHASE_ONE_H


#include <iostream>
#include <cstdio>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include <boost/filesystem.hpp>

#include <P1Camera.hpp>
#include <P1Image.hpp>
#include <P1ImageJpegWriter.hpp>
#include <P1ImageTiffWriter.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Header.h>
#include <diagnostic_updater/diagnostic_updater.h>
#include <diagnostic_updater/publisher.h>

#include <roskv/envoy.h>
#include <roskv/archiver.h>
#include <custom_msgs/GSOF_EVT.h>
#include <phase_one/phase_one_utils.h>
#include <phase_one/GetPhaseOneParameter.h>
#include <phase_one/SetPhaseOneParameter.h>
#include <phase_one/GetCompressedImageView.h>
#include <phase_one/GetImageView.h>


namespace phase_one
{
    class PhaseOne {
        public:
            // global thread tracker for a clean exit
            bool running_ = false;

            PhaseOne();

            // Shutdown threads and camera safely
            ~PhaseOne();

            // Init call if thread-based
            void init();

            // Definition of init call, connect to camera, instantiate threads
            virtual void onInit();

            // Given an ip to connect to a PhaseOne camera, enter connection / retry loop
            int connectToIPCamera(std::string ip);

            // Thread defined for capturing camera images
            int capture();

            // Thread defined for demosaicing / debayering images asynchronously from capture
            int demosaic();
            
            // Given PhaseOne raw image and bayered image, write image to disk as jpeg or tiff
            bool dumpImage(P1::ImageSdk::RawImage rawImage,
                           P1::ImageSdk::BitmapImage bitmap,
                           const std::string &filename,
                           std::string format);

            // Fill in the entries of the maps 'property_to_id_' and 'property_to_type_' given
            // a 'camera' handle. These maps define the values used for setting/getting the
            // parameters in the ROS service calls 
            void getPropertyMaps(const P1::CameraSdk::Camera& camera,
                                 std::map<std::string, int>& property_to_id_, 
                                 std::map<std::string, P1::CameraSdk::PropertyValue>& property_to_type);

            // ROS service call, grabs a parameter or lists of parameters from the camera
            // and returns the string values
            bool getPhaseOneParameter(phase_one::GetPhaseOneParameter::Request& req,
                                      phase_one::GetPhaseOneParameter::Response& resp);

            // ROS service call, sets the list of param=value calls requested on
            // the camera
            bool setPhaseOneParameter(phase_one::SetPhaseOneParameter::Request& req,
                                      phase_one::SetPhaseOneParameter::Response& resp);

            // ROS service call, given a homography, return the compressed image chip
            // of that warp
            bool getCompressedImageView(phase_one::GetCompressedImageView::Request& req,
                                        phase_one::GetCompressedImageView::Response& resp);

            // ROS service call, given a homography, return the raw image chip of that 
            // warp
            bool getImageView(custom_msgs::RequestImageView::Request& req,
                              custom_msgs::RequestImageView::Response& resp);

            // ROS subscriber, listens for "event" messages published from the INS, and 
            // when received, adds those to the current EventCache
            void eventCallback (const boost::shared_ptr<custom_msgs::GSOF_EVT const>& msg);
        private:
            // Phase One
            P1::CameraSdk::Camera camera;
            P1::ImageSdk::DecodeConfig decodeConfig;
            P1::ImageSdk::ConvertConfig convertConfig;
            P1::ImageSdk::JpegConfig jpegConfig;
            P1::CameraSdk::Listener listener;
            // ROS
            ros::ServiceServer image_view_service_;
            ros::ServiceServer compressed_image_view_service_;
            ros::ServiceServer get_param_service_;
            ros::ServiceServer set_param_service_;
            ros::Subscriber event_sub_;
            image_transport::Publisher image_pub;
            ros::Publisher stat_pub_;
            cv_bridge::CvImage img_bridge;
            ros::Time frame_recv_time_;
            // ROS params
            std::string ip_address_;
            std::string trigger_mode_;
            std::string cam_channel_;
            std::string cam_fov_;
            std::string hostname;
            double      auto_trigger_rate_;
            int         num_threads_;
            // Internal data structures / sync structures
            int processed_counter = 0;
            int total_counter = 0;
            std::thread capture_thread_;
            std::thread demosaic_thread_;
            // Lock between capture/demosaic threads
            mutable std::mutex mtx;
            // Lock for ImageSDK thread pool
            mutable std::mutex debayer_mtx;
            // Holds the current queue of images to demosaic
            std::queue<P1::ImageSdk::RawImage> image_q_;
            // Keeps track of the view requests
            std::vector<double> lastH;
            bool new_image = true;
            bool last_show_sat = false;
            // Holds the current queue of files to process
            std::queue<std::string> filename_q_;
            // Class maps for camera parameters
            std::map<std::string, int> property_to_id_;
            std::map<std::string, P1::CameraSdk::PropertyValue> property_to_type_;
            // Output files of images that have been demosaiced vs. those that have
            // been captured but not yet demosaiced
            std::ofstream to_process_out;
            std::ofstream processed_out;
            // custom
            ArchiverOpts arch_opts_ = ArchiverOpts::from_env();
            std::shared_ptr<RedisEnvoy> envoy_;
            custom_msgs::GSOF_EVT event_; // store the last received event
            // Holds events from the INS in a map to be searched for and matched
            // to incoming images
            EventCache event_cache;
            // Debugger output
            diagnostic_updater::Updater updater;
    };
}


#endif //PHASE_ONE_UTILS_H