#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <queue>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include <custom_msgs/RequestImageView.h>

using std::cout;
using std::endl;

namespace prosilica_camera
{
    class ViewServerNodelet : public nodelet::Nodelet
    {
        public:
            ViewServerNodelet()
            {
            }

        private:
            // Cache image, its header, and its encoding
            std::deque<std::tuple<cv::Mat, std_msgs::Header, std::string>> image_q_;
            std::map<std::string, std::vector<double>> frame2H;
            std::map<std::string, bool> frame2newimg;
            cv_bridge::CvImage img_bridge;
            ros::ServiceServer image_view_service_;
            image_transport::Subscriber sub;

        virtual void onInit() {
            ROS_INFO("Image View Server: Initialization");
            // ROS initialization
            ros::NodeHandle& nh = getNodeHandle();
            ros::NodeHandle& pnh = getPrivateNodeHandle();
            image_transport::ImageTransport it(nh);
            std::string ns = ros::this_node::getNamespace();
            std::string topic = ns + "/image_raw";
            std::cout << topic << "\n\n\n";
            sub = it.subscribe(topic, 1,
                               &ViewServerNodelet::image_callback, this);
            image_view_service_ = pnh.advertiseService("get_image_view",
                                      &ViewServerNodelet::getImageView, this);
            ROS_INFO("Image View Server: Finished");
        };

        void image_callback(const sensor_msgs::ImageConstPtr& msg) {
            ROS_INFO("Received Image!");
            // Keep a running most-recent queue of size 1
            // Decompress message into raw image, and store that in q
            auto tic = std::chrono::high_resolution_clock::now();
            cv_bridge::CvImagePtr cv_ptr;
            std::string encoding = msg->encoding;
            if (encoding == "bayer_grbg8")
                encoding = sensor_msgs::image_encodings::RGB8;
            try
            {
              cv_ptr = cv_bridge::toCvCopy(msg, encoding);
            }
            catch (cv_bridge::Exception& e)
            {
              ROS_ERROR("cv_bridge exception: %s", e.what());
              return;
            }
            cv::Mat cv_image = cv_ptr->image.clone();
            std_msgs::Header header = msg->header;
            std::tuple<cv::Mat, std_msgs::Header, std::string> T = std::make_tuple(
                                                   cv_image, header, encoding);
            if ( !image_q_.empty() ) {
                image_q_.pop_front();
                image_q_.push_back(T);
            } else {
                image_q_.push_back(T);
            }
            // Reset cache
            for (auto const& x : frame2newimg)
            {
                frame2newimg[x.first] = true;
            }
            ROS_INFO_STREAM("Size of image queue is: " << image_q_.size());
            auto toc = std::chrono::high_resolution_clock::now();
            auto dt = toc - tic;
            ROS_INFO_STREAM("View Server: Time to process image was: " << dt.count() / 1e9 << "s\n");
        };

        bool getImageView(custom_msgs::RequestImageView::Request& req,
                          custom_msgs::RequestImageView::Response& resp) {
            auto tic = std::chrono::high_resolution_clock::now();
            ROS_INFO_STREAM("Size of image queue is: " << image_q_.size());
            std::tuple<cv::Mat, std_msgs::Header, std::string> T;
            cv::Mat cv_image;
            std_msgs::Header header;
            std::string encoding;
            if ( !image_q_.empty() ) {
                T = image_q_[0];
                cv_image = std::get<0>(T);
                header = std::get<1>(T);
                encoding = std::get<2>(T);
            } else {
                resp.success = false;
                resp.image = sensor_msgs::Image();
                return false;
            }

            std::vector<double> H = req.homography;
            // If the last requested hasn't changed, don't bother returning
            bool stale_H;
            try {
                stale_H = H == frame2H[req.frame];
            } catch (...) {
                stale_H = false;
            }
            // If homography is the same as last frequest for the same frame,
            // and no new image has arrived, return a null frame.
            if (stale_H && !frame2newimg[req.frame]) {
                resp.success = false;
                resp.image = sensor_msgs::Image();
                return true;
            }
            // Cache new values
            frame2H[req.frame] = H;
            // Global variable holding state if a new image has been received
            frame2newimg[req.frame] = false;

            ROS_INFO_STREAM("View Server: Received image of width: " << cv_image.size().width <<
                            " and height: " << cv_image.size().height <<
                           " channels: " << cv_image.channels() <<  std::endl);
            int h = req.output_height;
            int w = req.output_width;
            int interp = req.interpolation;
            try {

                cv::Mat imgWarp = cv::Mat(cv::Size(w, h), cv_image.type());

                cv::Mat warp_matrix;
                warp_matrix = cv::Mat::eye(3, 3, CV_32F);
                int row = 0;
                int col = 0;
                for (auto &hom : H)
                {
                    // stuff the values into the matrix
                    warp_matrix.at<float>(row, col) = hom;
                    //ROS_INFO_STREAM(" " << hom);
                    col++;
                    if (col > 2)
                    {
                        col = 0;
                        row++;
                    }
                }
                cv::warpPerspective(cv_image, imgWarp, warp_matrix.inv(), cv::Size(w, h), interp);
                cv::Mat claheWarp;
                if(req.apply_clahe) {
                    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                    clahe->setClipLimit(req.contrast_strength);
                    clahe->setTilesGridSize(cv::Size(8, 8));
                    clahe->apply(imgWarp, claheWarp);
                } else {
                    claheWarp = imgWarp;
                }

                // convert UV mono images to color
                cv::Mat finalWarp;
                if (claheWarp.channels() < 3) {
                    cv::cvtColor(claheWarp, finalWarp, cv::COLOR_GRAY2RGB);
                } else {
                    finalWarp = claheWarp;
                }
                if (req.show_saturated_pixels) {
                    int maxval = 255;
                    cv::Scalar sat_pix = cv::Scalar(maxval, maxval, maxval);
                    cv::Mat mask;
                    cv::inRange(claheWarp, sat_pix, sat_pix, mask);
                    // Set white pixels to red
                    finalWarp.setTo(cv::Scalar(255, 0, 0), mask);
                }

                // Debayer raw rgb
                // Always return "color" image
                img_bridge = cv_bridge::CvImage(header,
                        sensor_msgs::image_encodings::RGB8, finalWarp);
                sensor_msgs::Image output_msg;
                img_bridge.toImageMsg(output_msg);

                resp.success = true;
                resp.image = output_msg;
            } catch (...) {
              ROS_INFO("CV Warp exception.");
              resp.success = false;
              return false;
            }

            auto toc = std::chrono::high_resolution_clock::now();
            auto dt = toc - tic;
            ROS_INFO_STREAM("View Server: Time to process image request for " << req.frame << " was: " << dt.count() / 1e9 << "s\n");
            return true;
        };
    };


PLUGINLIB_EXPORT_CLASS(prosilica_camera::ViewServerNodelet, nodelet::Nodelet);
}  // end namespace
