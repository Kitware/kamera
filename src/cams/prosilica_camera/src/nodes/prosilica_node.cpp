/*********************************************************************
* Software License Agreement (BSD License)
*
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above
* copyright notice, this list of conditions and the following
* disclaimer in the documentation and/or other materials provided
* with the distribution.
* * Neither the name of the Willow Garage nor the names of its
* contributors may be used to endorse or promote products derived
* from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
* ROS2 port of the KAMERA prosilica driver (formerly a ROS1 nodelet).
* dynamic_reconfigure is replaced by node parameters applied at startup;
* the unused polled_camera / diagnostics plumbing is gone.
*********************************************************************/

#include <string>
#include <csignal>
#include <map>
#include <sstream>

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <camera_calibration_parsers/parse_ini.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/fill_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/srv/set_camera_info.hpp>

#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <filesystem>

#include "prosilica/prosilica.h"
#include "prosilica/rolling_sum.h"

#include <roskv/envoy.h>
#include <roskv/archiver.h>

#include <cam_utils/event_cache.hpp>

// custom messages from KAMERA
#include <custom_msgs/srv/cam_get_attr.hpp>
#include <custom_msgs/srv/cam_set_attr.hpp>
#include <custom_msgs/srv/str_list.hpp>
#include <custom_msgs/msg/gsof_evt.hpp>
#include <custom_msgs/msg/stat.hpp>


bool prosilica_inited = false;
int num_cameras = 0;

namespace prosilica_camera {
    class ProsilicaDriver;
}

// Container so we can reference the driver from the signal handler
std::map<int, prosilica_camera::ProsilicaDriver *> active_drivers;

void driver_shutdown();

void signalHandler( int signum ) {
    ROS_WARN("<!> Interrupt signal (%d)\n", signum);
    driver_shutdown();
    rclcpp::shutdown();
}

std::string dumpImageMessage(const sensor_msgs::msg::Image::ConstSharedPtr &received_image, const std::string filename, bool debayer)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    cv_bridge::CvImagePtr cvPtr;
    if (debayer) {
        cvPtr = cv_bridge::toCvCopy(received_image, sensor_msgs::image_encodings::BGR8);
    } else {
        cvPtr = cv_bridge::toCvCopy(received_image, received_image->encoding);
    }

    std::filesystem::path path_filename{filename};
    try {
        std::filesystem::create_directories(path_filename.parent_path());
        cv::imwrite(filename, cvPtr->image, compression_params);
    } catch (std::filesystem::filesystem_error &e) {
        ROS_ERROR("Archive Failed [%d]: %s", e.code().value(), e.what());
        return "";
    }
    if (!std::filesystem::exists(path_filename)) {
        ROS_ERROR("Failed to create file");
    }
    return filename;
}

/** === === === === === === === === === === === ===  */

static const char* camera_channels[] = {"rgb", "ir", "uv"};
static std::map<std::string, double> camera_delays{{"rgb", 0.423}, {"ir", 0.003}, {"uv", 0.289}};

namespace prosilica_camera
{
    std::map<int, const char *> pv_error_codes = {
            {0, "ePvErrSuccess, No error"},
            {1, "ePvErrCameraFault, Unexpected camera fault"},
            {2, "ePvErrInternalFault, Unexpected fault in PvApi or driver"},
            {3, "ePvErrBadHandle, Camera handle is invalid"},
            {4, "ePvErrBadParameter, Bad parameter to API call"},
            {5, "ePvErrBadSequence, Sequence of API calls is incorrect"},
            {6, "ePvErrNotFound, Camera or attribute not found"},
            {7, "ePvErrAccessDenied, Camera cannot be opened in the specified mode"},
            {8, "ePvErrUnplugged, Camera was unplugged"},
            {9, "ePvErrInvalidSetup, Setup is invalid (an attribute is invalid)"},
            {10, "ePvErrResources, System/network resources or memory not available"},
            {11, "ePvErrBandwidth, 1394 bandwidth not available"},
            {12, "ePvErrQueueFull, Too many frames on queue"},
            {13, "ePvErrBufferTooSmall, Frame buffer is too small"},
            {14, "ePvErrCancelled, Frame cancelled by user"},
            {15, "ePvErrDataLost, The data for the frame was lost"},
            {16, "ePvErrDataMissing, Some data in the frame is missing"},
            {17, "ePvErrTimeout, Timeout during wait"},
            {18, "ePvErrOutOfRange, Attribute value is out of the expected range"},
            {19, "ePvErrWrongType, Attribute is not this type (wrong access function)"},
            {20, "ePvErrForbidden, Attribute write forbidden at this time"},
            {21, "ePvErrUnavailable, Attribute is not available at this time"},
            {22, "ePvErrFirewall, A firewall is blocking the traffic (Windows only)"},
    };

    /// Static camera configuration, formerly the dynamic_reconfigure config.
    struct DriverConfig {
        std::string trigger_mode = "fixedrate";
        double trig_rate = 1.0;
        bool auto_exposure = true;
        double exposure = 0.025;
        bool auto_gain = true;
        int gain = 0;
        bool auto_whitebalance = true;
        int whitebalance_red = 100;
        int whitebalance_blue = 100;
        int binning_x = 1;
        int binning_y = 1;
        int x_offset = 0;
        int y_offset = 0;
        int width = 0;
        int height = 0;
        std::string frame_id = "";
        bool auto_adjust_stream_bytes_per_second = true;
        int stream_bytes_per_second = 45000000;
        double exposure_auto_max = 0.5;
        int exposure_auto_target = 50;
        int gain_auto_max = 24;
        int gain_auto_target = 50;
    };


    class ProsilicaDriver
{

public:

    ProsilicaDriver(rclcpp::Node::SharedPtr node)
      : node_{node},
        auto_adjust_stream_bytes_per_second_(false),
        auto_adjust_binning_(false),
        count_(0),
        frames_dropped_total_(0), frames_completed_total_(0),
        frames_dropped_acc_(WINDOW_SIZE),
        frames_completed_acc_(WINDOW_SIZE),
        packets_missed_total_(0), packets_received_total_(0),
        packets_missed_acc_(WINDOW_SIZE),
        packets_received_acc_(WINDOW_SIZE)
    {
        cam_index = num_cameras;
        active_drivers[cam_index] = this;

        ++num_cameras;
        printf("<> Hi, I am the Prosilica driver\n");
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        onInitImpl();
    }

    ~ProsilicaDriver()
    {
        //! Make sure we interrupt initialization (if it happened to still execute).
        init_thread_.interrupt();
        init_thread_.join();

        if(camera_)
        {
            camera_->stop();
            camera_.reset(); // must destroy Camera before calling prosilica::fini
        }

        active_drivers.erase(cam_index);
        --num_cameras;
        if(num_cameras<=0)
        {
            prosilica::fini();
            prosilica_inited = false;
            num_cameras = 0;
        }

        ROS_WARN("Unloaded prosilica camera with guid %s", hw_id_.c_str());
    }

    void public_stop() {
        stop();
    }

private:
    rclcpp::Node::SharedPtr node_;
    std::string cam_fov;
    std::string cam_channel;
    boost::shared_ptr<prosilica::Camera> camera_;
    boost::thread init_thread_;
    rclcpp::TimerBase::SharedPtr update_timer_;
    int cam_index;

    image_transport::CameraPublisher    image_publisher_;
    rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr missed_frames_pub_;
    rclcpp::Publisher<custom_msgs::msg::Stat>::SharedPtr stat_pub_;
    rclcpp::Publisher<custom_msgs::msg::Stat>::SharedPtr errstat_pub_;
    rclcpp::Service<sensor_msgs::srv::SetCameraInfo>::SharedPtr set_camera_info_srv_;
    rclcpp::Service<custom_msgs::srv::CamGetAttr>::SharedPtr get_camera_attr_srv_;
    rclcpp::Service<custom_msgs::srv::CamSetAttr>::SharedPtr set_camera_attr_srv_;
    rclcpp::Service<custom_msgs::srv::StrList>::SharedPtr get_attr_list_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr health_srv_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr trigger_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr exposure_sub;
    rclcpp::Subscription<custom_msgs::msg::GsofEvt>::SharedPtr event_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr dumper_sub_;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr shutdown_sub_;

    EventCache                          event_cache;
    Watchdog                            watchdog;
    rclcpp::Duration                    clock_offset{0, 0}; // offset between system time and camera internal time
    std::shared_ptr<RedisEnvoy>         envoy_;
    ArchiverOpts                        arch_opts_ = ArchiverOpts::from_env();

    sensor_msgs::msg::Image img_;
    sensor_msgs::msg::Image broken_img_;
    sensor_msgs::msg::CameraInfo cam_info_;

    custom_msgs::msg::GsofEvt event_;     // store the last received event
    uint64_t last_published_event_num_ = 0;

    std::string     frame_id_;
    unsigned long   guid_;
    std::string     hw_id_;
    std::string     ip_address_;
    double          open_camera_retry_period_;
    std::string     trig_timestamp_topic_;
    // time last frame was received. Putting this here because frameDone is static
    rclcpp::Time    frame_recv_time_;
    int             gvspRetries_;
    float           gvspResendPercent_;

    double        update_rate_;
    int           trigger_mode_;
    bool          auto_adjust_stream_bytes_per_second_;
    bool          auto_adjust_binning_; // allow binning to be requested, otherwise set to 1

    tPvUint32 sensor_width_, sensor_height_;
    tPvUint32 max_binning_x, max_binning_y, dummy;
    int count_;

    DriverConfig last_config_;
    boost::recursive_mutex config_mutex_;

    // State updater
    enum CameraState
    {
        OPENING,
        CAMERA_NOT_FOUND,
        FORMAT_ERROR,
        ERROR,
        OK
    }camera_state_;
    std::string state_info_;
    std::string intrinsics_;
    static const int WINDOW_SIZE = 100; // remember previous 5s
    unsigned long frames_dropped_total_, frames_completed_total_;
    RollingSum<unsigned long> frames_dropped_acc_, frames_completed_acc_;
    unsigned long packets_missed_total_, packets_received_total_;
    RollingSum<unsigned long> packets_missed_acc_, packets_received_acc_;

    std::string getName() {
        return std::string(node_->get_fully_qualified_name());
    }

    void onInitImpl()
    {
        //! initialize prosilica if necessary
        if(!prosilica_inited)
        {
            ROS_INFO("Initializing prosilica GIGE API");
            prosilica::init();
            prosilica_inited = true;
        }

        //! Retrieve parameters
        count_ = 0;
        update_rate_=30;
        frame_id_ = node_->declare_parameter("frame_id", std::string("/camera_optical_frame"));
        ROS_INFO("Loaded param frame_id: %s", frame_id_.c_str());

        hw_id_ = node_->declare_parameter("guid", std::string(""));
        if(hw_id_ == "")
        {
            guid_ = 0;
        }
        else
        {
            guid_ = boost::lexical_cast<unsigned long>(hw_id_);
            ROS_INFO("Loaded param guid: %lu lu", guid_);
        }

        ip_address_ = node_->declare_parameter("ip_address", std::string(""));
        ROS_INFO("Loaded ip address: %s", ip_address_.c_str());

        open_camera_retry_period_ = node_->declare_parameter("open_camera_retry_period", 1.);
        ROS_INFO("Retry period: %f", open_camera_retry_period_);

        // load static config (formerly dynamic_reconfigure)
        loadConfig();

        // Setup periodic callback to get new data from the camera (software mode only)
        // created on demand in start()

        // Open camera
        openCamera();

        cam_channel = node_->declare_parameter("cam_chan", std::string(""));
        cam_fov = node_->declare_parameter("cam_fov", std::string(""));
        ROS_INFO("Cameratype: %s/%s", cam_fov.c_str(), cam_channel.c_str());

        RedisEnvoyOpts envoy_opts = RedisEnvoyOpts::from_env("driver_" + cam_fov + "_" + cam_channel );
        /// Connect with redis param server
        std::cout << envoy_opts << " | " << RedisHelper::get_redis_uri() <<  std::endl;
        envoy_ = std::make_shared<RedisEnvoy>(envoy_opts);
        ROS_WARN("echo: %s", envoy_->echo("Redis connected").c_str());
        std::string ns = node_->get_namespace();
        std::string image_read = ns + "/image_raw";
        ROS_WARN("read topic: %s", image_read.c_str());


        // Advertise topics
        auto expected_delay = camera_delays[cam_channel];
        event_cache.set_delay(expected_delay);
        event_cache.set_tolerance(rclcpp::Duration::from_seconds(0.49));
        image_publisher_     = image_transport::create_camera_publisher(node_.get(), image_read);
        missed_frames_pub_   = node_->create_publisher<std_msgs::msg::Header>("/missed_frames", 3);
        stat_pub_            = node_->create_publisher<custom_msgs::msg::Stat>("/stat", 3);
        errstat_pub_         = node_->create_publisher<custom_msgs::msg::Stat>("/errstat", 3);
        set_camera_info_srv_ = node_->create_service<sensor_msgs::srv::SetCameraInfo>(
            "set_camera_info",
            [this](const std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Request> req,
                   std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Response> rsp) { setCameraInfo(req, rsp); });
        get_camera_attr_srv_ = node_->create_service<custom_msgs::srv::CamGetAttr>(
            "get_camera_attr",
            [this](const std::shared_ptr<custom_msgs::srv::CamGetAttr::Request> req,
                   std::shared_ptr<custom_msgs::srv::CamGetAttr::Response> rsp) { getCameraAttrSrv(req, rsp); });
        set_camera_attr_srv_ = node_->create_service<custom_msgs::srv::CamSetAttr>(
            "set_camera_attr",
            [this](const std::shared_ptr<custom_msgs::srv::CamSetAttr::Request> req,
                   std::shared_ptr<custom_msgs::srv::CamSetAttr::Response> rsp) { setCameraAttr(req, rsp); });
        get_attr_list_srv_ = node_->create_service<custom_msgs::srv::StrList>(
            "get_attr_list",
            [this](const std::shared_ptr<custom_msgs::srv::StrList::Request> req,
                   std::shared_ptr<custom_msgs::srv::StrList::Response> rsp) { getAttrList(req, rsp); });
        health_srv_ = node_->create_service<std_srvs::srv::Trigger>(
            "health",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> rsp) { health(req, rsp); });
        trigger_sub_ = node_->create_subscription<std_msgs::msg::Header>(
            "trigger", 1,
            [this](const std_msgs::msg::Header::ConstSharedPtr msg) { syncInCallback(msg); });
        exposure_sub = node_->create_subscription<std_msgs::msg::Int32>(
            "exposure", 1,
            [this](const std_msgs::msg::Int32::ConstSharedPtr msg) { exposureCallback(msg); });
        event_sub = node_->create_subscription<custom_msgs::msg::GsofEvt>(
            "/event", 1,
            [this](const custom_msgs::msg::GsofEvt::ConstSharedPtr msg) { eventCallback(msg); });
        dumper_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
            image_read, 1,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { dumperCallback(msg); });
        shutdown_sub_ = node_->create_subscription<std_msgs::msg::Int8>(
            "/shutdown", 1,
            [](const std_msgs::msg::Int8::ConstSharedPtr msg) {
                ROS_INFO("Requesting clean shutdown: %d", msg->data);
                rclcpp::shutdown();
            });

        applyConfig(last_config_, true);

        /**
         * These parameters are a double-edged sword. Increasing GvspRetries can decrease risk of dropped frames if
         * the system is not under heavy load. Under heavy load and unpredictable conditions, this can cause a
         * resend storm that causes no frames to make it. Tune with caution.
         */
        gvspRetries_ = node_->declare_parameter("GvspRetries", 3);
        gvspResendPercent_ = (float) node_->declare_parameter("GvspResendPercent", 2.0);

        camera_->setAttribute("GvspRetries", (tPvUint32) gvspRetries_); // default: 3.0, too high, and you risk a UDP storm
        camera_->setAttribute("GvspResendPercent", (tPvFloat32) gvspResendPercent_); // default: 1.0

        sensor_msgs::clearImage(broken_img_);

        ROS_GREEN("<> <> <> Cam init complete 1");

        /// Ignore the first few seconds of frames health-wise since there is a purge
        watchdog.DelayedStart(node_, 6.0);

    }

    /** Fill the static config from node parameters (formerly dynamic_reconfigure) */
    void loadConfig() {
        DriverConfig c;
        c.trigger_mode = node_->declare_parameter("trigger_mode", c.trigger_mode);
        c.trig_rate = node_->declare_parameter("trig_rate", c.trig_rate);
        c.auto_exposure = node_->declare_parameter("auto_exposure", c.auto_exposure);
        c.exposure = node_->declare_parameter("exposure", c.exposure);
        // GainMode/GainValue come from the KAMERA config; map onto gain settings
        std::string gain_mode = node_->declare_parameter("GainMode", std::string("Auto"));
        int gain_value = node_->declare_parameter("GainValue", 0);
        c.auto_gain = (gain_mode != "Manual");
        c.gain = gain_value;
        c.auto_whitebalance = node_->declare_parameter("auto_whitebalance", c.auto_whitebalance);
        c.whitebalance_red = node_->declare_parameter("whitebalance_red", c.whitebalance_red);
        c.whitebalance_blue = node_->declare_parameter("whitebalance_blue", c.whitebalance_blue);
        c.binning_x = node_->declare_parameter("binning_x", c.binning_x);
        c.binning_y = node_->declare_parameter("binning_y", c.binning_y);
        c.x_offset = node_->declare_parameter("x_offset", c.x_offset);
        c.y_offset = node_->declare_parameter("y_offset", c.y_offset);
        c.width = node_->declare_parameter("width", c.width);
        c.height = node_->declare_parameter("height", c.height);
        c.frame_id = frame_id_;
        c.auto_adjust_stream_bytes_per_second = node_->declare_parameter(
            "auto_adjust_stream_bytes_per_second", c.auto_adjust_stream_bytes_per_second);
        c.stream_bytes_per_second = node_->declare_parameter("stream_bytes_per_second", c.stream_bytes_per_second);
        c.exposure_auto_max = node_->declare_parameter("exposure_auto_max", c.exposure_auto_max);
        c.exposure_auto_target = node_->declare_parameter("exposure_auto_target", c.exposure_auto_target);
        c.gain_auto_max = node_->declare_parameter("gain_auto_max", c.gain_auto_max);
        c.gain_auto_target = node_->declare_parameter("gain_auto_target", c.gain_auto_target);
        last_config_ = c;
    }

    void openCamera()
    {
        int loop_count = 0;
        while (!camera_ && rclcpp::ok())
        {
            // For some reason, this doesn't see any on the first try.
            // The startup sequence in general is a hot mess but it eventually works
            ROS_INFO("== %d Available cameras: %ld ==\n%s", loop_count++,
                    prosilica::numCameras(), getAvailableCameras().c_str());
            ROS_INFO("==   __________________   ==");

            boost::lock_guard<boost::recursive_mutex> scoped_lock(config_mutex_);
            camera_state_ = OPENING;
            try
            {
                if(guid_ != 0)
                {
                    state_info_ = "Trying to load camera with guid " + hw_id_;
                    ROS_INFO("%s", state_info_.c_str());
                    camera_ = boost::make_shared<prosilica::Camera>((unsigned long)guid_);
                    ROS_INFO("Started Prosilica camera with guid \"%lu\"", guid_);

                }
                else if(!ip_address_.empty())
                {
                    state_info_ = "Trying to load camera with ipaddress: " + ip_address_;
                    ROS_INFO("%s", state_info_.c_str());
                    camera_ = boost::make_shared<prosilica::Camera>(ip_address_.c_str());
                    guid_ = camera_->guid();
                    hw_id_ = boost::lexical_cast<std::string>(guid_);

                    ROS_INFO("Started Prosilica camera with guid \"%d\"", (int)camera_->guid());
                }
                else
                {
                    if(prosilica::numCameras()>0)
                    {
                        state_info_ = "Trying to load first camera found";
                        ROS_INFO("%s", state_info_.c_str());
                        guid_ = prosilica::getGuid(0);
                        camera_ = boost::make_shared<prosilica::Camera>((unsigned long)guid_);
                        hw_id_ = boost::lexical_cast<std::string>(guid_);
                        ROS_INFO("Started Prosilica camera with guid \"%d\"", (int)guid_);
                    }
                    else
                    {
                        throw std::runtime_error("ERR: Found no cameras on local subnet");
                    }
                }

            }
            catch (std::exception& e)
            {
                camera_state_ = CAMERA_NOT_FOUND;
                std::stringstream err;
                if (prosilica::numCameras() == 0)
                {
                    err << "Hm. Found no cameras on local subnet";
                }
                else if (guid_ != 0)
                {
                    err << "Unable to open prosilica camera with guid " << guid_ <<": "<<e.what();
                }
                else if (ip_address_ != "")
                {
                    err << "Unable to open prosilica camera with ip address " << ip_address_ <<": "<<e.what();
                }

                state_info_ = err.str();
                ROS_WARN("%s", state_info_.c_str());

                camera_.reset();

            }
            long milliseconds = (long) (1000 * open_camera_retry_period_);
            boost::this_thread::sleep(boost::posix_time::milliseconds(milliseconds));
        }
        ROS_GREEN("Camera opened. Loading intrinsics. Synching clock");
        loadIntrinsics();
        syncCamToSysClock();
        ROS_BLUE("Camera starting");
        start();
    }

    std::string getAvailableCameras()
    {
        std::vector<prosilica::CameraInfo> cameras = prosilica::listCameras();
        std::stringstream list;
        for (unsigned int i = 0; i < cameras.size(); ++i)
        {
            list << cameras[i].serial << " - " <<cameras[i].name<< " - GUID = "<<cameras[i].guid<<" IP = "<<cameras[i].ip_address<<std::endl;
        }
        return list.str();
    }

    void setSpeed() {
        std::string actualStreamBps;
        if(camera_->hasAttribute("StreamBytesPerSecond")) {
            camera_->setAttribute("StreamBytesPerSecond", (tPvUint32)(camera_->max_data_rate / num_cameras));
            camera_->getAttribute("StreamBytesPerSecond", actualStreamBps);
            ROS_INFO("Max data rate: %lu current set: %s", camera_->max_data_rate, actualStreamBps.c_str());
        } else {
            ROS_WARN("Cannot set StreamBytesPerSecond");
        }
    }

    void loadIntrinsics()
    {
        try
        {
            camera_->setKillCallback(boost::bind(&ProsilicaDriver::kill, this, boost::placeholders::_1));

            if(auto_adjust_stream_bytes_per_second_ && camera_->hasAttribute("StreamBytesPerSecond")) {
                setSpeed();
            }


            // Retrieve contents of user memory
            std::string buffer(prosilica::Camera::USER_MEMORY_SIZE, '\0');
            camera_->readUserMemory(&buffer[0], prosilica::Camera::USER_MEMORY_SIZE);

            PvAttrRangeUint32(camera_->handle(), "BinningX", &dummy, &max_binning_x);
            PvAttrRangeUint32(camera_->handle(), "BinningY", &dummy, &max_binning_y);
            PvAttrRangeUint32(camera_->handle(), "Width",    &dummy, &sensor_width_);
            PvAttrRangeUint32(camera_->handle(), "Height",   &dummy, &sensor_height_);


            // Parse calibration file
            std::string camera_name;
            if (camera_calibration_parsers::parseCalibrationIni(buffer, camera_name, cam_info_))
            {
                intrinsics_ = "Loaded calibration";
                ROS_INFO("Loaded calibration for camera '%s'", camera_name.c_str());
            }
            else
            {
                intrinsics_ = "Failed to load intrinsics from camera";
                ROS_WARN("Failed to load intrinsics from camera");
            }
        }
        catch(std::exception &e)
        {
            camera_state_ = CAMERA_NOT_FOUND;
            state_info_ = e.what();
        }
    }

    void start()
    {
        try
        {
            switch(trigger_mode_)
            {
                case prosilica::Software:
                    ROS_INFO("starting camera %s in software trigger mode", hw_id_.c_str());
                    camera_->start(prosilica::Software, 1., prosilica::Continuous);
                    if(update_rate_ > 0)
                    {
                        update_timer_ = node_->create_wall_timer(
                            std::chrono::duration<double>(1.0 / update_rate_),
                            [this]() { updateCallback(node_->now()); });
                    }
                    break;
                case prosilica::Freerun:
                    ROS_INFO("starting camera %s in freerun trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaDriver::publishImage, this, boost::placeholders::_1));
                    camera_->start(prosilica::Freerun, 1., prosilica::Continuous);
                    break;
                case prosilica::FixedRate:
                    ROS_INFO("starting camera %s in fixedrate trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaDriver::publishImage, this, boost::placeholders::_1));
                    camera_->start(prosilica::FixedRate, update_rate_, prosilica::Continuous);
                    break;
                case prosilica::SyncIn1:
                    ROS_INFO("starting camera %s in sync1 trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaDriver::publishImage, this, boost::placeholders::_1));
                    camera_->start(prosilica::SyncIn1, update_rate_, prosilica::Continuous);
                    break;
                case prosilica::SyncIn2:
                    ROS_INFO("starting camera %s in sync2 trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaDriver::publishImage, this, boost::placeholders::_1));
                    camera_->start(prosilica::SyncIn2, update_rate_, prosilica::Continuous);
                    break;
                default:
                    break;
            }
        }
        catch(std::exception &e)
        {
            camera_state_ = CAMERA_NOT_FOUND;
            state_info_ = e.what();
        }

        try {
            ROS_INFO("exposure = %s", getCameraAttr("ExposureValue").value.c_str());
        }
        catch(std::exception &e) {
            camera_state_ = CAMERA_NOT_FOUND;
            state_info_ = e.what();
        }
        ROS_GREEN("start() complete");
    }

    void stop()
    {
        if (update_timer_) {
            update_timer_->cancel();
        }
        if(!camera_)
            return;
        camera_->removeEvents();
        camera_->stop();

    }

    void kill(unsigned long guid)
    {
        if(guid == guid_)
        {
            ROS_WARN("[%s] got Camera::kill() request for prosilica camera %lu",getName().c_str(), guid);
            //! Make sure we interrupt initialization (if it happened to still execute).
            init_thread_.interrupt();
            init_thread_.join();

            camera_state_ = CAMERA_NOT_FOUND;
            state_info_ = "Prosilica camera " + hw_id_ + " disconnected";
            ROS_ERROR("%s", state_info_.c_str());
            boost::lock_guard<boost::recursive_mutex> scoped_lock(config_mutex_);
            stop();
            camera_.reset();
            init_thread_ = boost::thread(boost::bind(&ProsilicaDriver::openCamera, this));
            return;
        }
    }


    int syncCamToSysClock() {
        ROS_INFO("call syncCamToSysClock() ");
        auto err = PvCommandRun(camera_->handle(), "TimeStampValueLatch");
        if (err != ePvErrSuccess) {
            ROS_ERROR("Could not sync clock");
            return (int) err;
        }
        rclcpp::Time after = node_->now();
        tPvUint32 timelo, timehi, freq;
        camera_->getAttribute("TimeStampValueHi", timehi);
        camera_->getAttribute("TimeStampValueLo", timelo);
        camera_->getAttribute("TimeStampFrequency", freq);
        rclcpp::Time tsframe = prosilica::CvtPvTimestamp(timehi, timelo, freq);
        clock_offset = after - tsframe;
        ROS_INFO("Clock synced, offset = %lf", clock_offset.seconds());
        return 0;
    }

    /// todo: variably disable archiving and/or publishing. totally remove it and profile ePvWhatevr
    void publishImage(tPvFrame* frame)
    {   bool ok = false;
        try {
            auto recv_time = node_->now();
            this->publishImageOld(frame, recv_time);
            ok = true;

        } catch (std::exception &e) {
            ROS_ERROR("publishImage failed: %s", e.what());
            ok = false;
        }
        if (ok) {
            watchdog.pet();
        } else {
            watchdog.kick();
        }
    }

    void publishImageOld(tPvFrame* frame, rclcpp::Time time)
    {
        frame_recv_time_ = node_->now();

        camera_state_ = OK;
        state_info_ = "Camera operating normally";

        /** allow most recent event to be received.
         * Events arrive asynchronously via the executor; check a few times
         * for a newer event than the last published one. */
        int64_t seq_dt = 0;
        int loop_count = 0;
        do {
            seq_dt = (int64_t) event_.event_num - (int64_t) last_published_event_num_;
        } while (seq_dt < 1 && loop_count++ < 3);

        std_msgs::msg::Header gps_header;
        uint64_t gps_event_num = 0;
        /// todo: null check here or use context manager
        prosilica::MetaFrame* meta_frame = (prosilica::MetaFrame*) frame->Context[0];
        if (!meta_frame) {
            ROS_ERROR("tPvFrame context is null");
            return;
        }
        ROS_INFO("FrameDone %p #%ld @ %ld %ld", (void*) meta_frame, (long int) meta_frame->idx, (long int) frame->TimestampHi, (long int) frame->TimestampLo);

        bool success = event_cache.search(frame_recv_time_, gps_header, gps_event_num);

        rclcpp::Time tsframe = prosilica::CvtPvTimestamp(frame->TimestampHi, frame->TimestampLo);
        rclcpp::Time corrFrameTime = tsframe + clock_offset;
        ROS_INFO("frameTime: %16.4f GPS: %16.4f DT: %7.4f", corrFrameTime.seconds(),
                 rclcpp::Time(gps_header.stamp).seconds(),
                 (corrFrameTime - rclcpp::Time(gps_header.stamp)).seconds());
        std::stringstream this_frame_id;
        this_frame_id << frame_id_;

        // convey the status of the event binding process
        if (success) {
            this_frame_id << "?lock=1&eventNum=" << gps_event_num << "&eventTime" << rclcpp::Time(gps_header.stamp).seconds() ;
        } else {
            this_frame_id << "?lock=0";
        }

        if (image_publisher_.getNumSubscribers() > 0)
        {
            auto nodeName = getName();
            std::stringstream link;
            custom_msgs::msg::Stat stat_msg;
            stat_msg.header.stamp = node_->now();
            stat_msg.trace_topic = nodeName + "/publishImage";
            stat_msg.node = nodeName;
            stat_msg.trace_header = std_msgs::msg::Header(img_.header);
            link << nodeName << "/event/" << event_.event_num; // link this trace to the event trace
            stat_msg.link = link.str();
            meta_frame->img_.header.stamp = event_.gps_time;
            if (seq_dt > 1) {
                ROS_ERROR("[%lu] Missed %ld frames, based on event seq ", (unsigned long) event_.event_num, (long) (seq_dt - 1));
                for (auto i = 0; i < 4 && i < seq_dt - 1; i++) {
                    watchdog.kick();
                }
            }
            sensor_msgs::msg::Image::SharedPtr p_img = std::make_shared<sensor_msgs::msg::Image>(meta_frame->img_);

            if (processFrame(frame, *p_img, cam_info_))  // this will memcpy frame's buffer into img_
            {
                // Set the image timestamp to match the event that actually triggered it
                if (success) {
                    stat_msg.note = "success";
                    img_.header = gps_header;
                }
                p_img->header.frame_id = this_frame_id.str();
                cam_info_.header = p_img->header;
                stat_pub_->publish(stat_msg);
                image_publisher_.publish(*p_img, cam_info_);
                frames_dropped_acc_.add(0);

            }
            else
            {
                ROS_ERROR("[?][3] Frame parse failed, checking status");
                auto status = frame->Status;
                ROS_ERROR("[%lu][3] Frame parse failed, frame status: %d %s", (unsigned long) event_.event_num, status, pv_error_codes[status]);
                camera_state_ = FORMAT_ERROR;
                state_info_ = "Unable to process frame";
                this_frame_id << "&status=" << status << "&error=" << pv_error_codes[status];
                std_msgs::msg::Header msg = std_msgs::msg::Header(img_.header);
                broken_img_.header.stamp = img_.header.stamp;
                broken_img_.header.frame_id = this_frame_id.str();
                ++frames_dropped_total_;
                missed_frames_pub_->publish(msg);
                stat_msg.note = pv_error_codes[status];
                stat_pub_->publish(stat_msg);
                errstat_pub_->publish(stat_msg);
                image_publisher_.publish(broken_img_, cam_info_);
                frames_dropped_acc_.add(1);
            }
            last_published_event_num_ = event_.event_num;


            ++frames_completed_total_;
            frames_completed_acc_.add(1);
        }
        auto end = node_->now();
        ROS_INFO("publishImage1 in %.4f seconds", (end - frame_recv_time_).seconds());
    }

    void updateCallback(rclcpp::Time current_real)
    {
        // Download the most recent data from the device
        camera_state_ = OK;
        state_info_ = "Camera operating normally";
        if(image_publisher_.getNumSubscribers() > 0)
        {
            boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
            try
            {
                tPvFrame* frame = NULL;
                frame = camera_->grab(1000);
                publishImageOld(frame, current_real);
            }
            catch(std::exception &e)
            {
                camera_state_ = ERROR;
                state_info_ = e.what();
                ROS_ERROR("Unable to read from camera: %s", e.what());
                ++frames_dropped_total_;
                frames_dropped_acc_.add(1);
                return;
            }
        }
    }

    void syncInCallback (const std_msgs::msg::Header::ConstSharedPtr& msg)
    {
        printf("\n <> syncInCallback <> \n");
        if (trigger_mode_ != prosilica::Software)
        {
            camera_state_ = ERROR;
            state_info_ = "Can not sync from topic trigger unless in Software Trigger mode";
            ROS_ERROR("%s", state_info_.c_str());
            return;
        }
        updateCallback(rclcpp::Time(msg->stamp));
    }

    void dumperCallback (const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        int is_archiving = ArchiverHelper::get_is_archiving(envoy_, "/sys/arch/is_archiving");
        ROS_INFO("dumper: is archiving: %d", is_archiving);
        if(is_archiving) {
            long int sec  = msg->header.stamp.sec;
            long int nsec = msg->header.stamp.nanosec;
            std::string filename = ArchiverHelper::generateFilename(envoy_, arch_opts_, sec, nsec);
            try {
                bool debayer{false};

                if ("rgb" == cam_channel) {
                    debayer = true;
                }
                auto filename_written = dumpImageMessage(msg, filename, debayer );
                ROS_INFO("[%s] dumped %s", cam_channel.c_str(), filename_written.c_str());
            } catch (cv_bridge::Exception &e) {
                ROS_ERROR("%s", e.what());
            }

        }
    }

    void eventCallback (const custom_msgs::msg::GsofEvt::ConstSharedPtr& msg)
    {
        ROS_INFO("[%lu]<1> eventCallback <>         %2.2f", (unsigned long) msg->event_num, rclcpp::Time(msg->gps_time).seconds());
        event_ = *msg;
        event_cache.push_back(rclcpp::Time(msg->sys_time), msg);
        auto nodeName = getName();
        custom_msgs::msg::Stat stat_msg;
        std::stringstream link;
        stat_msg.header.stamp = node_->now();
        stat_msg.trace_header = (*msg).header;
        stat_msg.trace_topic = nodeName + "/eventCallback";
        stat_msg.node = nodeName;
        link << nodeName << "/event/" << event_.event_num; // link this trace to the event trace
        stat_msg.link = link.str();
        stat_pub_->publish(stat_msg);
        event_cache.purge();
        watchdog.check();
    }

    /** Exposure is in microseconds (microsecs). Minimum varies by camera I think. */
    void exposureCallback (const std_msgs::msg::Int32::ConstSharedPtr &msg)
    {
        printf("\n <> exposureCallback <> \n");
        tPvUint32 microsecs_min = 30;
        int32_t microsecs = msg->data;
        if (microsecs < 0) {
            printf("WARNING: Exposure set to less that zero. Setting auto exposure. This is not a recommended feature");
            camera_->setExposure(microsecs, prosilica::Auto);
            return;
        }
        tPvUint32 umicrosecs = msg->data;

        if (umicrosecs < microsecs_min) {
            printf("WARNING: Exposure set to less than max allowed. Clipping to %lu", microsecs_min);
            umicrosecs = microsecs_min;
        }
        printf("INFO: Exposure set to: %lu microseconds", umicrosecs);
        camera_->setExposure(umicrosecs, prosilica::Manual);

    }

    void health(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                std::shared_ptr<std_srvs::srv::Trigger::Response> rsp) {
        (void) req;
        auto healthy = watchdog.Ok();
        rsp->success = healthy;
        if (!healthy) {
            rsp->message = "Watchdog timed out";
        }
    }


    /** this is a pretty gross api. It's stringly-typed, so be careful
     * also currently does not work with certain types.
     * Note: use "1"/"0" for pushing bools. They are pretty rare though*/
    void setCameraAttr(const std::shared_ptr<custom_msgs::srv::CamSetAttr::Request> req,
                       std::shared_ptr<custom_msgs::srv::CamSetAttr::Response> rsp) {
        ROS_INFO("<API> setCameraAttr(%s, %s)", req->name.c_str(), req->value.c_str());
        tPvHandle handle = camera_->handle();
        rsp->pv_err = ePvErrUnknown;
        rsp->value = "error";
        const char *c_name = req->name.c_str();

        if (req->name == "SyncClock") {
            rsp->pv_err = syncCamToSysClock();
            rsp->dtype = "time";
            rsp->value = std::to_string(clock_offset.seconds());
            return;
        }

        /** On failure, pass error to message. This is more descriptive than
         * return false*/
        rsp->pv_err = PvAttrIsAvailable(handle, c_name);
        if (rsp->pv_err != 0) { return; }

        tPvAttributeInfo info;

        rsp->pv_err = PvAttrInfo(handle, c_name, &info);
        if (rsp->pv_err != 0) { return; }


        switch (info.Datatype) {
            case ePvDatatypeEnum: {
                rsp->pv_err = PvAttrEnumSet(
                        handle, c_name, req->value.c_str());
                rsp->dtype = "enum";
                break;
            }
            case ePvDatatypeString: {
                rsp->pv_err = PvAttrStringSet(
                        handle, c_name, req->value.c_str());
                rsp->dtype = "string";
                break;
            }
            case ePvDatatypeUint32: {
                rsp->pv_err = PvAttrUint32Set(
                        handle, c_name, std::stoul(req->value));
                rsp->dtype = "uint32";
                break;
            }
            case ePvDatatypeInt64: {
                rsp->pv_err = PvAttrInt64Set(
                        handle, c_name, std::stol(req->value));
                rsp->dtype = "int64";
                break;
            }
            case ePvDatatypeFloat32: {
                rsp->pv_err = PvAttrFloat32Set(
                        handle, c_name, std::stof(req->value));
                rsp->dtype = "float32";
                break;
            }
            case ePvDatatypeBoolean: {
                rsp->pv_err = PvAttrBooleanSet(
                        handle, c_name, std::stoi(req->value));
                rsp->dtype = "bool";
                break;
            }
            case ePvDatatypeCommand: {
                rsp->pv_err = PvCommandRun(
                        handle, c_name);
                rsp->dtype = "PvCommandRun";
                rsp->value = "ok";
                return;
            }
            default: {
                rsp->pv_err = ePvErrBadParameter;
                break;
            }
        }
        if (rsp->pv_err != 0) { return; }
        custom_msgs::srv::CamGetAttr::Response new_rsp = getCameraAttr(req->name);
        rsp->value = new_rsp.value;
    }

    void getCameraAttrSrv(const std::shared_ptr<custom_msgs::srv::CamGetAttr::Request> req,
                          std::shared_ptr<custom_msgs::srv::CamGetAttr::Response> rsp) {
        ROS_INFO("<API> getCameraAttr(%s)", req->name.c_str());

        try {
            *rsp = getCameraAttr(req->name);
        }
        catch (prosilica::ProsilicaException &) {
            rsp->value = "error";
        }
    }

    custom_msgs::srv::CamGetAttr::Response getCameraAttr(std::string name) {
        tPvHandle handle = camera_->handle();
        custom_msgs::srv::CamGetAttr::Response rsp;
        rsp.pv_err = ePvErrUnknown;
        rsp.value = "error";
        const char *c_name = name.c_str();

        rsp.pv_err = PvAttrIsAvailable(handle, c_name);
        if (rsp.pv_err != 0) { return rsp; }

        tPvAttributeInfo info;

        rsp.pv_err = PvAttrInfo(handle, c_name, &info);
        if (rsp.pv_err != 0) { return rsp; }


        switch (info.Datatype) {
            case ePvDatatypeEnum:
                camera_->getAttributeEnum(name, rsp.value);
                rsp.pv_err = ePvErrSuccess;
                rsp.dtype = "enum";
                break;
            case ePvDatatypeString:
                rsp.pv_err = prosilica::getAttribute(handle, c_name, rsp.value);
                rsp.dtype = "string";
                break;
            case ePvDatatypeUint32:
                tPvUint32 value;
                rsp.pv_err = PvAttrUint32Get(handle, c_name, &value);
                rsp.dtype = "uint32";
                rsp.value = std::to_string((unsigned long) value);
                break;
            case ePvDatatypeInt64:
                tPvInt64 l_value;
                rsp.pv_err = PvAttrInt64Get(handle, c_name, &l_value);
                rsp.dtype = "int64";
                rsp.value = std::to_string((int64_t) l_value);
                break;
            case ePvDatatypeFloat32:
                tPvFloat32 f_value;
                rsp.pv_err = PvAttrFloat32Get(handle, c_name, &f_value);
                rsp.dtype = "float32";
                rsp.value = std::to_string((float) f_value);
                break;
            case ePvDatatypeBoolean:
                tPvBoolean b_value;
                rsp.pv_err = PvAttrBooleanGet(handle, c_name, &b_value);
                rsp.dtype = "bool";
                rsp.value = std::to_string((bool) b_value);
                break;
            default:
                rsp.pv_err = ePvErrBadParameter;
                break;
        }
        return rsp;

    }

    void getAttrList(const std::shared_ptr<custom_msgs::srv::StrList::Request> req,
                     std::shared_ptr<custom_msgs::srv::StrList::Response> rsp) {
        (void) req;
        tPvAttrListPtr pListPtr;
        unsigned long sz;
        // The attribute list is contained in memory allocated by the PvApi module.
        tPvErr err = PvAttrList(camera_->handle(), &pListPtr, &sz);
        ROS_WARN("sz: %ld ", sz);
        if (err != 0) {
            rsp->pv_err = err;
            rsp->values.push_back("error");
            return;
        }
        for (unsigned long i = 0; i < sz; i++) {
            rsp->values.push_back(pListPtr[i]);
        }
    }

    // this calls frameToImage which calls fillImage which calls memcpy
    bool processFrame(tPvFrame* frame, sensor_msgs::msg::Image &img, sensor_msgs::msg::CameraInfo &cam_info)
    {
        /// @todo Match time stamp from frame to ROS time?
        if (frame==NULL ) {
            return false;
        }
        // we want to deliberately allow some missing-date frames through
        // for debugging
        if (frame->Status == ePvErrSuccess) {
            // pass
        } else if (frame->Status == ePvErrDataMissing) {
            // pass
            ROS_WARN("Data Missing from Frame. This may fail");
        } else {
            return false;  // you shall not pass
        }
        try
        {
            /// @todo Binning values retrieved here may differ from the ones used to actually
            /// capture the frame! Maybe need to clear queue when changing binning and/or
            /// stuff binning values into context?
            tPvUint32 binning_x = 1, binning_y = 1;
            if (auto_adjust_binning_) {
                if (camera_->hasAttribute("BinningX")) {
                    camera_->getAttribute("BinningX", binning_x);
                    camera_->getAttribute("BinningY", binning_y);
                }
            }

            // Binning averages bayer samples, so just call it mono8 in that case
            if (frame->Format == ePvFmtBayer8 && (binning_x > 1 || binning_y > 1))
                frame->Format = ePvFmtMono8;

            if (!frameToImage(frame, img)) {
                return false;
            }
            // Set the operational parameters in CameraInfo (binning, ROI)
            cam_info.binning_x = binning_x;
            cam_info.binning_y = binning_y;
            // ROI in CameraInfo is in unbinned coordinates, need to scale up
            cam_info.roi.x_offset = frame->RegionX * binning_x;
            cam_info.roi.y_offset = frame->RegionY * binning_y;
            cam_info.roi.height = frame->Height * binning_y;
            cam_info.roi.width = frame->Width * binning_x;
            cam_info.roi.do_rectify = (frame->Height != sensor_height_ / binning_y) ||
                                       (frame->Width  != sensor_width_  / binning_x);
        }
        catch(std::exception &e)
        {
            return false;
        }

        count_++;
        return true;
    }

    bool frameToImage(tPvFrame* frame, sensor_msgs::msg::Image &image)
    {
        // NOTE: 16-bit and Yuv formats not supported
        static const char* BAYER_ENCODINGS[] = { "bayer_rggb8", "bayer_gbrg8", "bayer_grbg8", "bayer_bggr8" };

        std::string encoding;
        if (frame->Format == ePvFmtMono8)       encoding = sensor_msgs::image_encodings::MONO8;
        else if (frame->Format == ePvFmtBayer8) encoding = BAYER_ENCODINGS[frame->BayerPattern];
        else if (frame->Format == ePvFmtRgb24)  encoding = sensor_msgs::image_encodings::RGB8;
        else if (frame->Format == ePvFmtBgr24)  encoding = sensor_msgs::image_encodings::BGR8;
        else if (frame->Format == ePvFmtRgba32) encoding = sensor_msgs::image_encodings::RGBA8;
        else if (frame->Format == ePvFmtBgra32) encoding = sensor_msgs::image_encodings::BGRA8;
        else {
            ROS_WARN("Received frame with unsupported pixel format %d", frame->Format);
            return false;
        }


        if(frame->ImageSize == 0) {
            ROS_WARN("Image size is zero");
            return false;
        }
        if(frame->Height == 0) {
            ROS_WARN("Image height is zero");
            return false;
        }

        uint32_t step = frame->ImageSize / frame->Height;
        // fillImage calls memcpy
        return sensor_msgs::fillImage(image, encoding, frame->Height, frame->Width, step, frame->ImageBuffer);
    }

    void setCameraInfo(const std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Request> req,
                       std::shared_ptr<sensor_msgs::srv::SetCameraInfo::Response> rsp)
    {
        ROS_INFO("<API> New camera info received");
        sensor_msgs::msg::CameraInfo &info = req->camera_info;

        // Sanity check: the image dimensions should match the max resolution of the sensor.
        if (info.width != sensor_width_ || info.height != sensor_height_)
        {
            rsp->success = false;
            std::stringstream err;
            err << "Camera_info resolution " << info.width << "x" << info.height
                << " does not match current video setting, camera running at resolution "
                << sensor_width_ << "x" << sensor_height_ << ".";
            rsp->status_message = err.str();
            ROS_ERROR("%s", rsp->status_message.c_str());
            return;
        }

        stop();

        std::string cam_name = "prosilica";
        cam_name += hw_id_;
        std::stringstream ini_stream;
        if (!camera_calibration_parsers::writeCalibrationIni(ini_stream, cam_name, info))
        {
            rsp->status_message = "Error formatting camera_info for storage.";
            rsp->success = false;
        }
        else
        {
            std::string ini = ini_stream.str();
            if (ini.size() > prosilica::Camera::USER_MEMORY_SIZE)
            {
                rsp->success = false;
                rsp->status_message = "Unable to write camera_info to camera memory, exceeded storage capacity.";
            }
            else
            {
                try
                {
                    camera_->writeUserMemory(ini.c_str(), ini.size());
                    cam_info_ = info;
                    rsp->success = true;
                }
                catch (prosilica::ProsilicaException &e)
                {
                    rsp->success = false;
                    rsp->status_message = e.what();
                }
            }
        }
        if (!rsp->success)
            ROS_ERROR("%s", rsp->status_message.c_str());

        start();
    }

    /** Apply the static configuration to the camera.
     * ROS2 port of the old dynamic_reconfigure callback; runs once at startup. */
    void applyConfig(DriverConfig &config, bool restart)
    {
        printf("\n<> Apply config \n");

        if (restart)
            stop();

        //! Trigger mode
        if (config.trigger_mode == "streaming")
        {
            trigger_mode_ = prosilica::Freerun;
            update_rate_ = 1.; // make sure we get _something_
        }
        else if (config.trigger_mode == "syncin1")
        {
            trigger_mode_ = prosilica::SyncIn1;
            update_rate_ = config.trig_rate;
        }
        else if (config.trigger_mode == "syncin2")
        {
            trigger_mode_ = prosilica::SyncIn2;
            update_rate_ = config.trig_rate;
        }
        else if (config.trigger_mode == "fixedrate")
        {
            trigger_mode_ = prosilica::FixedRate;
            update_rate_ = config.trig_rate;
        }
        else if (config.trigger_mode == "software")
        {
            trigger_mode_ = prosilica::Software;
            update_rate_ = config.trig_rate;
        }

        else if (config.trigger_mode == "polled")
        {
            trigger_mode_ = prosilica::Software;
            update_rate_ = 0;
        }
        else if (config.trigger_mode == "triggered")
        {
            trigger_mode_ = prosilica::Software;
            update_rate_ = 0;
        }
        else
        {
            ROS_ERROR("Invalid trigger mode '%s' in reconfigure request", config.trigger_mode.c_str());
        }

        // Exposure
        if (config.auto_exposure)
        {
            camera_->setExposure(0, prosilica::Auto);
            if (camera_->hasAttribute("ExposureAutoMax"))
            {
                tPvUint32 us = config.exposure_auto_max*1000000. + 0.5;
                camera_->setAttribute("ExposureAutoMax", us);
            }
            if (camera_->hasAttribute("ExposureAutoTarget"))
                camera_->setAttribute("ExposureAutoTarget", (tPvUint32)config.exposure_auto_target);
        }
        else
        {
            unsigned us = config.exposure*1000000. + 0.5;
            camera_->setExposure(us, prosilica::Manual);
            camera_->setAttribute("ExposureValue", (tPvUint32)us);
        }

        // Gain
        if (config.auto_gain)
        {
            if (camera_->hasAttribute("GainAutoMax"))
            {
                camera_->setGain(0, prosilica::Auto);
                camera_->setAttribute("GainAutoMax", (tPvUint32)config.gain_auto_max);
                camera_->setAttribute("GainAutoTarget", (tPvUint32)config.gain_auto_target);
            }
            else
            {
                tPvUint32 major, minor;
                camera_->getAttribute("FirmwareVerMajor", major);
                camera_->getAttribute("FirmwareVerMinor", minor);
                ROS_WARN("Auto gain not available for this camera. Auto gain is available "
                "on firmware versions 1.36 and above. You are running version %u.%u.",
                (unsigned)major, (unsigned)minor);
                config.auto_gain = false;
            }
        }
        else
        {
            camera_->setGain(config.gain, prosilica::Manual);
            camera_->setAttribute("GainValue", (tPvUint32)config.gain);
        }

        // White balance
        if (config.auto_whitebalance)
        {
            if (camera_->hasAttribute("WhitebalMode"))
                camera_->setWhiteBalance(0, 0, prosilica::Auto);
            else
            {
                ROS_WARN("Auto white balance not available for this camera.");
                config.auto_whitebalance = false;
            }
        }
        else
        {
            camera_->setWhiteBalance(config.whitebalance_blue, config.whitebalance_red, prosilica::Manual);
            if (camera_->hasAttribute("WhitebalValueRed"))
                camera_->setAttribute("WhitebalValueRed", (tPvUint32)config.whitebalance_red);
            if (camera_->hasAttribute("WhitebalValueBlue"))
                camera_->setAttribute("WhitebalValueBlue", (tPvUint32)config.whitebalance_blue);
        }

        // Binning configuration
        if (camera_->hasAttribute("BinningX"))
        {
            config.binning_x = std::min(config.binning_x, (int)max_binning_x);
            config.binning_y = std::min(config.binning_y, (int)max_binning_y);

            camera_->setBinning(config.binning_x, config.binning_y);
        }
        else if (config.binning_x > 1 || config.binning_y > 1)
        {
            ROS_WARN("Binning not available for this camera.");
            config.binning_x = config.binning_y = 1;
        }

        // Region of interest configuration
        // Make sure ROI fits in image
        config.x_offset = std::min(config.x_offset, (int)sensor_width_ - 1);
        config.y_offset = std::min(config.y_offset, (int)sensor_height_ - 1);
        config.width  = std::min(config.width, (int)sensor_width_ - config.x_offset);
        config.height = std::min(config.height, (int)sensor_height_ - config.y_offset);
        // If width or height is 0, set it as large as possible
        int width  = config.width  ? config.width  : sensor_width_  - config.x_offset;
        int height = config.height ? config.height : sensor_height_ - config.y_offset;

        // Adjust full-res ROI to binning ROI
        int x_offset = config.x_offset / config.binning_x;
        int y_offset = config.y_offset / config.binning_y;
        unsigned int right_x  = (config.x_offset + width  + config.binning_x - 1) / config.binning_x;
        unsigned int bottom_y = (config.y_offset + height + config.binning_y - 1) / config.binning_y;
        // Rounding up is bad when at max resolution which is not divisible by the amount of binning
        right_x = std::min(right_x, (unsigned)(sensor_width_ / config.binning_x));
        bottom_y = std::min(bottom_y, (unsigned)(sensor_height_ / config.binning_y));
        width = right_x - x_offset;
        height = bottom_y - y_offset;

        camera_->setRoi(x_offset, y_offset, width, height);

        // TF frame
        img_.header.frame_id = cam_info_.header.frame_id = config.frame_id;

        // Normally the node adjusts the bandwidth used by the camera during diagnostics, to use as
        // much as possible without dropping packets. But this can create interference if two
        // cameras are on the same switch, e.g. for stereo. So we allow the user to set the bandwidth
        // directly.
        auto_adjust_stream_bytes_per_second_ = config.auto_adjust_stream_bytes_per_second;
        if (!auto_adjust_stream_bytes_per_second_)
            camera_->setAttribute("StreamBytesPerSecond", (tPvUint32)config.stream_bytes_per_second);
        else
            camera_->setAttribute("StreamBytesPerSecond", (tPvUint32)(camera_->max_data_rate/num_cameras));

        //! If exception thrown due to bad settings, it will fail to start camera
        if (restart)
        {
            try
            {
                start();
            }
            catch(std::exception &e)
            {
                ROS_ERROR("Invalid settings: %s", e.what());
            }
        }

        last_config_ = config;
    }
};



} // end namespace

/** === === === === === === === === === === === ===  */
void driver_shutdown() {
    for ( auto const& it: active_drivers)
    {
        ROS_WARN("Stopping driver %d", it.first);
        it.second->public_stop();
    }
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("prosilica_driver");
    prosilica_camera::ProsilicaDriver driver(node);
    // Use a multithreaded executor to handle the numerous callbacks
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
