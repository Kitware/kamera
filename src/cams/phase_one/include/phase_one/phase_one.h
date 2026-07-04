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
#include <queue>
#include <algorithm>
#include <filesystem>

#include <P1Camera.hpp>
#include <P1Image.hpp>
#include <P1ImageJpegWriter.hpp>
#include <P1ImageTiffWriter.hpp>

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/header.hpp>

#include <roskv/envoy.h>
#include <roskv/archiver.h>
#include <custom_msgs/msg/gsof_evt.hpp>
#include <custom_msgs/msg/image_space_detection_list.hpp>
#include <custom_msgs/msg/stat.hpp>
#include <custom_msgs/srv/request_image_view.hpp>
#include <cam_utils/event_cache.hpp>
#include <phase_one/srv/get_phase_one_parameter.hpp>
#include <phase_one/srv/set_phase_one_parameter.hpp>
#include <phase_one/srv/get_compressed_image_view.hpp>
#include <phase_one/srv/get_image_view.hpp>

// rclcpp logging shims to keep ROS1-style call sites
#define ROS_INFO(...) RCLCPP_INFO(rclcpp::get_logger("phase_one"), __VA_ARGS__)
#define ROS_WARN(...) RCLCPP_WARN(rclcpp::get_logger("phase_one"), __VA_ARGS__)
#define ROS_ERROR(...) RCLCPP_ERROR(rclcpp::get_logger("phase_one"), __VA_ARGS__)
#define ROS_INFO_STREAM(args) RCLCPP_INFO_STREAM(rclcpp::get_logger("phase_one"), args)
#define ROS_WARN_STREAM(args) RCLCPP_WARN_STREAM(rclcpp::get_logger("phase_one"), args)
#define ROS_ERROR_STREAM(args) RCLCPP_ERROR_STREAM(rclcpp::get_logger("phase_one"), args)


namespace phase_one
{
    class PhaseOne {
        public:
            // global thread tracker for a clean exit
            bool running_ = false;

            PhaseOne();

            // Shutdown threads and camera safely
            ~PhaseOne();

            // Init call; the node owns all ROS interfaces
            void init(rclcpp::Node::SharedPtr node);

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

            // Compress JPEG using nvjpeg (GPU-accelerated)
            bool compressJpegNvjpeg(const cv::Mat& bgr_image,
                                    std::vector<unsigned char>& output,
                                    int quality = 90);

            // Fill in the entries of the maps 'property_to_id_' and 'property_to_type_' given
            // a 'camera' handle. These maps define the values used for setting/getting the
            // parameters in the ROS service calls
            void getPropertyMaps(const P1::CameraSdk::Camera& camera,
                                 std::map<std::string, int>& property_to_id_,
                                 std::map<std::string, P1::CameraSdk::PropertyValue>& property_to_type);

            // ROS service call, grabs a parameter or lists of parameters from the camera
            // and returns the string values
            void getPhaseOneParameter(const std::shared_ptr<phase_one::srv::GetPhaseOneParameter::Request> req,
                                      std::shared_ptr<phase_one::srv::GetPhaseOneParameter::Response> resp);

            // ROS service call, sets the list of param=value calls requested on
            // the camera
            void setPhaseOneParameter(const std::shared_ptr<phase_one::srv::SetPhaseOneParameter::Request> req,
                                      std::shared_ptr<phase_one::srv::SetPhaseOneParameter::Response> resp);

            // ROS service call, given a homography, return the compressed image chip
            // of that warp
            void getCompressedImageView(const std::shared_ptr<phase_one::srv::GetCompressedImageView::Request> req,
                                        std::shared_ptr<phase_one::srv::GetCompressedImageView::Response> resp);

            // ROS service call, given a homography, return the raw image chip of that
            // warp
            void getImageView(const std::shared_ptr<custom_msgs::srv::RequestImageView::Request> req,
                              std::shared_ptr<custom_msgs::srv::RequestImageView::Response> resp);

            // ROS subscriber, listens for "event" messages published from the INS, and
            // when received, adds those to the current EventCache
            void eventCallback (const custom_msgs::msg::GsofEvt::ConstSharedPtr& msg);

            // ROS subscriber, listens for "detection list" messages published from the detector,
            // and when received, adds these to the detection cache
            void detectionListCallback (const custom_msgs::msg::ImageSpaceDetectionList::ConstSharedPtr& msg);
        private:
            // Phase One
            P1::CameraSdk::Camera camera;
            P1::ImageSdk::DecodeConfig decodeConfig;
            P1::ImageSdk::ConvertConfig convertConfig;
            P1::ImageSdk::JpegConfig jpegConfig;
            P1::CameraSdk::Listener listener;
            // ROS
            rclcpp::Node::SharedPtr node_;
            rclcpp::Service<custom_msgs::srv::RequestImageView>::SharedPtr image_view_service_;
            rclcpp::Service<phase_one::srv::GetCompressedImageView>::SharedPtr compressed_image_view_service_;
            rclcpp::Service<phase_one::srv::GetPhaseOneParameter>::SharedPtr get_param_service_;
            rclcpp::Service<phase_one::srv::SetPhaseOneParameter>::SharedPtr set_param_service_;
            rclcpp::Subscription<custom_msgs::msg::GsofEvt>::SharedPtr event_sub_;
            rclcpp::Subscription<custom_msgs::msg::ImageSpaceDetectionList>::SharedPtr detection_sub_;
            image_transport::Publisher image_pub;
            rclcpp::Publisher<custom_msgs::msg::Stat>::SharedPtr stat_pub_;
            cv_bridge::CvImage img_bridge;
            rclcpp::Time frame_recv_time_;
            // ROS params
            std::string ip_address_;
            std::string trigger_mode_;
            std::string cam_channel_;
            std::string cam_fov_;
            std::string hostname;
            std::string effort;
            std::string project;
            std::string base_dir;
            std::string to_process_filename_;
            std::string processed_filename_;
            double      auto_trigger_rate_;
            int         num_threads_;
            // Internal data structures / sync structures
            int processed_counter = 0;
            int total_counter = 0;
            int save_every_x = 1;
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
            std::map<std::string, int> filename_to_seq_map_;
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
            custom_msgs::msg::GsofEvt event_; // store the last received event
            // Holds events from the INS in a map to be searched for and matched
            // to incoming images
            EventCache event_cache;
    };
}


#endif //PHASE_ONE_H
