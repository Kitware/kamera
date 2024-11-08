#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <queue>

#include <P1Camera.hpp>
#include <P1Image.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include <phase_one/GetImageView.h>

using namespace P1::CameraSdk;
using namespace P1::ImageSdk;

using std::cout;
using std::endl;

namespace phase_one
{
    class ViewServerNodelet : public nodelet::Nodelet
    {
        public:
            ViewServerNodelet()
            {
            }

        private:
            std::queue<sensor_msgs::ImageConstPtr> image_q_;
            cv_bridge::CvImage img_bridge;
            P1::CameraSdk::Camera camera;
            P1::ImageSdk::DecodeConfig decodeConfig;
            P1::ImageSdk::ConvertConfig config;
            ros::ServiceServer image_view_service_;

        virtual void onInit() {
            ROS_INFO("Image View Server: Initialization");
            // ROS initialization
            ros::NodeHandle& nh = getNodeHandle();
            ros::NodeHandle& pnh = getPrivateNodeHandle();
            image_transport::ImageTransport it(nh);
            image_transport::Subscriber sub = it.subscribe("/image_raw", 1,
                                        &ViewServerNodelet::image_callback, this);
            image_view_service_ = pnh.advertiseService("get_image_view",
                                      &ViewServerNodelet::getImageView, this);
            ROS_INFO("Image View Server: Finished");
            //ros::spin();

            // Phase one Initialization
            //P1::ImageSdk::Initialize();
            //decodeConfig = P1::ImageSdk::DecodeConfig::Defaults;
            //P1::ImageSdk::SetSensorProfilesLocation(
            //        "/home/squadx/noaa/phaseOne/build/ImageSDK/SensorProfiles");
        };

        void image_callback(const sensor_msgs::ImageConstPtr& msg) {
            // Keep a running most-recent queue of size 1
            auto tic = std::chrono::high_resolution_clock::now();
            if ( !image_q_.empty() ) {
                image_q_.pop();
                image_q_.push(msg);
            } else {
                image_q_.push(msg);
            }
            ROS_INFO_STREAM("Size of image queue is: " << image_q_.size());
            auto toc = std::chrono::high_resolution_clock::now();
            auto dt = toc - tic;
            ROS_INFO_STREAM("View Server: Time to receive image camera was: " << dt.count() / 1e9 << "s\n");
        };

        bool getImageView(phase_one::GetImageView::Request& req,
                          phase_one::GetImageView::Response& resp) {
            auto tic = std::chrono::high_resolution_clock::now();
            ROS_INFO("Received Request for image view.");
            sensor_msgs::Image msg;
            if ( !image_q_.empty() ) {
                msg = *image_q_.front();
            } else {
                resp.success = false;
                return false;
            }

            cv_bridge::CvImagePtr cv_ptr;
            try
            {
              cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            }
            catch (cv_bridge::Exception& e)
            {
              ROS_ERROR("cv_bridge exception: %s", e.what());
              resp.success = false;
              return false;
            }
            cv::Mat cv_image = cv_ptr->image.clone();
            ROS_INFO_STREAM("View Server: Received image of width: " << cv_image.size().width <<
                            " and height: " << cv_image.size().height << std::endl);

            try {
                std::vector<double> H = req.homography;
                int h = req.output_height;
                int w = req.output_width;
                int interp = req.interpolation;

                cv::Mat imgWarp = cv::Mat(cv::Size(w, h), CV_8UC3);
                ROS_INFO("Warping image.");
                cv::warpPerspective(cv_image, imgWarp, H, cv::Point(w, h), interp);
                ROS_INFO("Finished warping image.");

                img_bridge = cv_bridge::CvImage(msg.header,
                        sensor_msgs::image_encodings::RGB8, imgWarp);
                sensor_msgs::Image output_msg;
                img_bridge.toImageMsg(output_msg);

                resp.success = true;
                resp.image = output_msg;
            } catch (...) {
              ROS_ERROR("CV Warp exception.");
              resp.success = false;
              return false;
            }

            auto toc = std::chrono::high_resolution_clock::now();
            auto dt = toc - tic;
            ROS_INFO_STREAM("View Server: Time to process image request was: " << dt.count() / 1e9 << "s\n");
            return true;
        };
    };


PLUGINLIB_EXPORT_CLASS(phase_one::ViewServerNodelet, nodelet::Nodelet);
}
