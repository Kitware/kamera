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
*********************************************************************/

#include <string>
#include <csignal>

#include <ros/ros.h>
#include <ros/console.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>
#include <dynamic_reconfigure/SensorLevels.h>
#include <diagnostic_updater/diagnostic_updater.h>
#include <diagnostic_updater/publisher.h>
#include <camera_calibration_parsers/parse_ini.h>
#include <polled_camera/publication_server.h>
#include <sensor_msgs/CameraInfo.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/SetCameraInfo.h>

#include <std_msgs/Int32.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Float64.h>
#include <std_srvs/Trigger.h>

#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <fmt/core.h>

#include <prosilica_camera/ProsilicaCameraConfig.h>
#include "prosilica/prosilica.h"
#include "prosilica/rolling_sum.h"

#include <roskv/envoy.h>
#include <roskv/archiver.h>

#include <phase_one/phase_one_utils.h>

// custom messages from KAMERA
#include <custom_msgs/CamGetAttr.h>
#include <custom_msgs/CamSetAttr.h>
#include <custom_msgs/StrList.h>
#include <custom_msgs/GSOF_EVT.h>
#include <custom_msgs/Stat.h>


bool prosilica_inited = false;//for if the nodelet is loaded multiple times in the same manager
int num_cameras = 0;

namespace prosilica_camera {
    class ProsilicaNodelet ;
}

// Container so we can actually reference the ProsilicaNodelet without ROS magic
std::map<int, prosilica_camera::ProsilicaNodelet *> active_nodelets;

void cb_request_shutdown(std_msgs::Int8 const &msg) {
    ROS_INFO("Requesting clean shutdown: %d", msg.data);
    ros::requestShutdown();
}

void driver_shutdown();

bool build_broken_img(sensor_msgs::Image& img);

void signalHandler( int signum ) {
    ROS_WARN("<!> Interrupt signal (%d)\n", signum);
    driver_shutdown();
    ros::shutdown();
}

// sensor_msgs::ImageConstPtr
void resizeImageMessage(const sensor_msgs::ImageConstPtr& received_image)
{
    cv_bridge::CvImagePtr cvPtr;
    cvPtr = cv_bridge::toCvCopy(received_image, sensor_msgs::image_encodings::BGR8);

    cv::Mat undist = cvPtr->image;
//    NODELET_WARN("Warning: hack here, deliberately shrinking image");

    cv::resize(undist, cvPtr->image, cv::Size(64, 48),
               0, 0, cv::INTER_LINEAR);

//    return cvPtr->toImageMsg();
}

// sensor_msgs::ImageConstPtr
std::string dumpImageMessage(const sensor_msgs::Image & received_image, const std::string filename) {
    ROS_WARN("Deprecated! dumpImageMessage(Image&)");
    auto cvPtr = cv_bridge::toCvCopy(received_image, sensor_msgs::image_encodings::BGR8);

}
std::string dumpImageMessage(const sensor_msgs::ImageConstPtr &received_image, const std::string filename, bool debayer)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    auto start_db = ros::Time::now();
    cv_bridge::CvImagePtr cvPtr;
    if (debayer) {
        cvPtr = cv_bridge::toCvCopy(received_image, sensor_msgs::image_encodings::BGR8);
    } else {
        cvPtr = cv_bridge::toCvCopy(received_image, received_image->encoding);
    }
    ROS_INFO("debayered? %d %ld in %2.3f ", debayer, (long int)(cvPtr->image.total() * cvPtr->image.elemSize()), (ros::Time::now()-start_db).toSec());
    cv::Mat undist = cvPtr->image;
    auto nodeName = ros::this_node::getName();

    start_db = ros::Time::now();

    boost::filesystem::path path_filename{filename};
    try {
        boost::filesystem::create_directories(path_filename.parent_path());
        cv::imwrite(filename, cvPtr->image, compression_params);
    } catch (boost::filesystem::filesystem_error &e) {
        ROS_ERROR("Archive Failed [%d]: %s", e.code().value(), e.what());
        return "";
    }
//    cv::imwrite(filename, cvPtr->image);
    ROS_INFO("dumped   %ld in %2.3f ", (long int)(cvPtr->image.total() * cvPtr->image.elemSize()), (ros::Time::now()-start_db).toSec());
    start_db = ros::Time::now();
//    auto filename2 = std::string("/mnt/ram/miketest/driverdump/") + nodeName + "/" + std::to_string(now.toSec()) + ".jpg";
//    cv::imwrite(filename2, cvPtr->image, compression_params);
//    auto out = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (!boost::filesystem::exists(path_filename)) {
        ROS_ERROR("Failed to create file");
    }
//    ROS_INFO("read     %ld in %2.3f ", (long int)(out.total() * out.elemSize()), (ros::Time::now()-start_db).toSec());
    return filename;

//    return cvPtr->toImageMsg();
}

/** === === === === === === === === === === === ===  */

static const char* camera_channels[] = {"rgb", "ir", "uv"};
static std::map<std::string, double> camera_delays{{"rgb", 0.423}, {"ir", 0.003}, {"uv", 0.289}};

/// Look for weird exceptions in time formats. Currently just checks validity, may check more if needed
/// isValid checks (!g_use_sim_time) || !g_sim_time.isZero();
bool timeIsBad(ros::Time const &t) {
    if (!t.isValid()) return false;
    return true;
}

/// Time should be within some reasonable distance of detected time
bool timeIsUnreasonable(ros::Time const &t, double tolerance) {

}


std::string get_cam_chan() {
    auto nodeName = ros::this_node::getName();
    for (auto i=0; i<3; i++) {
        auto s = std::string(camera_channels[i]);
        std::size_t found = nodeName.find("/" + s + "/");
        if (found != std::string::npos) {
            return s;
        }
    }
    ROS_ERROR("Unable to auto-detect camera channel");
    return std::string("no_chan");
}

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

    std::map<const std::string, tPvDatatype> tPv_dtypes = {
            {"unknown", ePvDatatypeUnknown},
            {"string", ePvDatatypeString},
            {"ExposureMode", ePvDatatypeEnum},
            {"ExposureValue", ePvDatatypeUint32},
    };


    class ProsilicaNodelet : public nodelet::Nodelet
{

public:

    virtual ~ProsilicaNodelet()
    {

        //! Make sure we interrupt initialization (if it happened to still execute).
        init_thread_.interrupt();
        init_thread_.join();

        if(camera_)
        {
            camera_->stop();
            camera_.reset(); // must destroy Camera before calling prosilica::fini
        }

        trigger_sub_.shutdown();
        poll_srv_.shutdown();
        image_publisher_.shutdown();

        active_nodelets.erase(cam_index);
        --num_cameras;
        if(num_cameras<=0)
        {
            prosilica::fini();
            prosilica_inited = false;
            num_cameras = 0;
        }

        NODELET_WARN("Unloaded prosilica camera with guid %s", hw_id_.c_str());
    }

    /// todo: may want to disable auto_adj
    ProsilicaNodelet()
      : auto_adjust_stream_bytes_per_second_(false),
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
        active_nodelets[cam_index] = this;

        ++num_cameras;
        printf("<> Hi, I am Prosilica Nodelet 1\n");
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);

    }

    void public_stop() {
        stop();
    }

private:
    std::string cam_fov;
    std::string cam_channel = get_cam_chan();
    boost::shared_ptr<prosilica::Camera> camera_;
    boost::thread init_thread_;
    ros::Timer update_timer_;
    int cam_index;
    ros::NodeHandle nh;
    ros::NodeHandlePtr nhp = boost::make_shared<ros::NodeHandle>(nh);

    image_transport::CameraPublisher    image_publisher_;
    polled_camera::PublicationServer    poll_srv_;
    ros::Publisher                      missed_frames_pub_;
    ros::Publisher                      stat_pub_;
    ros::Publisher                      errstat_pub_;
    ros::ServiceServer                  set_camera_info_srv_;
    ros::ServiceServer                  get_camera_attr_srv_;
    ros::ServiceServer                  set_camera_attr_srv_;
    ros::ServiceServer                  get_attr_list_srv_;
    ros::ServiceServer                  health_srv_;
    ros::Subscriber                     trigger_sub_;
    ros::Subscriber                     trigger_sub2_;
    ros::Subscriber                     exposure_sub;
    ros::Subscriber                     event_sub;
    ros::Subscriber                     dumper_sub_;
    ros::Subscriber                     shutdown_sub_;

    EventCache                          event_cache;
    Watchdog                            watchdog;
    ros::Duration                       clock_offset; // offset between system time and camera internal time
    std::shared_ptr<RedisEnvoy>         envoy_;
    ArchiverOpts                        arch_opts_ = ArchiverOpts::from_env();
    prosilica::OneShotManager           oneShotManager{};
//    std::unique_ptr<ArchiverFormatter>  archiver_;
//    ImageSync fsm;

    sensor_msgs::Image img_;
    sensor_msgs::Image broken_img_;
    sensor_msgs::CameraInfo cam_info_;

    custom_msgs::GSOF_EVT event_;     // store the last received event
    std_msgs::Header last_published_; // Last header which was successfully published

        std::string     frame_id_;
    unsigned long   guid_;
    std::string     hw_id_;
    std::string     ip_address_;
    double          open_camera_retry_period_;
    std::string     trig_timestamp_topic_;
    ros::Time       trig_time_;
    // time last frame was received. Putting this here because frameDone is static
    ros::Time       frame_recv_time_;
    std::string     gainMode;
    int             gainValue;
    int             gvspRetries_;
    float           gvspResendPercent_;

    // Dynamic reconfigure parameters
    double        update_rate_;
    int           trigger_mode_;
    bool          auto_adjust_stream_bytes_per_second_;
    bool          auto_adjust_binning_; // allow binning to be requested, otherwise set to 1

    tPvUint32 sensor_width_, sensor_height_;
    tPvUint32 max_binning_x, max_binning_y, dummy;
    int count_;

    // Dynamic Reconfigure
    prosilica_camera::ProsilicaCameraConfig last_config_;
    boost::recursive_mutex config_mutex_;
    typedef dynamic_reconfigure::Server<prosilica_camera::ProsilicaCameraConfig> ReconfigureServer;
    boost::shared_ptr<ReconfigureServer> reconfigure_server_;

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

    diagnostic_updater::Updater updater;


    virtual void onInit()
    {
        //! We will be retrying to open camera until it is open, which may block the
        //! thread. Nodelet::onInit() should not block, hence spawning a new thread
        //! to do initialization.
        init_thread_ = boost::thread(boost::bind(&ProsilicaNodelet::onInitImpl, this));

    }

    void onInitImpl()
    {
        nh = getNodeHandle();
//        ros::NodeHandle& nh = getNodeHandle();
//        nh2 = getNodeHandle();
        ros::NodeHandle& pn = getPrivateNodeHandle();

        //! initialize prosilica if necessary
        if(!prosilica_inited)
        {
            NODELET_INFO("Initializing prosilica GIGE API");
            prosilica::init();
            prosilica_inited = true;
        }

        //! Retrieve parameters from server
        count_ = 0;
        update_rate_=30;
        NODELET_INFO("namespace: %s", pn.getNamespace().c_str());
        pn.param<std::string>("frame_id", frame_id_, "/camera_optical_frame");
        NODELET_INFO("Loaded param frame_id: %s", frame_id_.c_str());

        pn.param<std::string>("guid", hw_id_, "");
        if(hw_id_ == "")
        {
            guid_ = 0;
        }
        else
        {
            guid_ = boost::lexical_cast<unsigned long>(hw_id_);
            NODELET_INFO("Loaded param guid: %lu lu", guid_);
        }

        pn.param<std::string>("ip_address", ip_address_, "");
        NODELET_INFO("Loaded ip address: %s", ip_address_.c_str());

        pn.param<double>("open_camera_retry_period", open_camera_retry_period_, 1.);
        NODELET_INFO("Retry period: %f", open_camera_retry_period_);

        // Setup updater
        updater.add(getName().c_str(), this, &ProsilicaNodelet::getCurrentState);
        NODELET_INFO("updated state");
        // Setup periodic callback to get new data from the camera
        update_timer_ = nh.createTimer(ros::Rate(update_rate_).expectedCycleTime(), &ProsilicaNodelet::updateCallback, this, false ,false);
        update_timer_.stop();
        NODELET_INFO("created update timer");
        // Open camera
        openCamera();

        pn.param<std::string>("cam_chan", cam_channel, "");
        pn.param<std::string>("cam_fov", cam_fov, "");
        NODELET_INFO("Cameratype: %s/%s", cam_fov.c_str(), cam_channel.c_str());

        RedisEnvoyOpts envoy_opts = RedisEnvoyOpts::from_env("driver_" + cam_fov + "_" + cam_channel );
        /// Connect with redis param server
        NODELET_WARN("gonna initialize");
        std::cout << envoy_opts << " | " << RedisHelper::get_redis_uri() <<  std::endl;
        envoy_ = std::make_shared<RedisEnvoy>(envoy_opts);
        NODELET_WARN("echo: %s", envoy_->echo("Redis connected").c_str());
        std::string ns = ros::this_node::getNamespace();
        std::string image_read = ns + "/image_raw";
        ROS_WARN("read topic: %s", image_read.c_str());


        // Advertise topics
        auto expected_delay = camera_delays[cam_channel];
        event_cache.set_delay(expected_delay);
        event_cache.set_tolerance(ros::Duration(0.49));
        ros::NodeHandle image_nh(nh);
        image_transport::ImageTransport image_it(image_nh);
        image_publisher_     = image_it.advertiseCamera("image_raw", 1);
        poll_srv_            = polled_camera::advertise(nh, "request_image", &ProsilicaNodelet::pollCallback, this);
        missed_frames_pub_   = nh.advertise<std_msgs::Header>("/missed_frames", 3);
        stat_pub_            = nh.advertise<custom_msgs::Stat>("/stat", 3);
        errstat_pub_            = nh.advertise<custom_msgs::Stat>("/errstat", 3);
        set_camera_info_srv_ = pn.advertiseService("set_camera_info", &ProsilicaNodelet::setCameraInfo, this);
        get_camera_attr_srv_ = pn.advertiseService("get_camera_attr", &ProsilicaNodelet::getCameraAttr, this);
        set_camera_attr_srv_ = pn.advertiseService("set_camera_attr", &ProsilicaNodelet::setCameraAttr, this);
        get_attr_list_srv_   = pn.advertiseService("get_attr_list", &ProsilicaNodelet::getAttrList, this);
        health_srv_          = pn.advertiseService("health", &ProsilicaNodelet::health, this);
        trigger_sub_         = pn.subscribe(trig_timestamp_topic_, 1, &ProsilicaNodelet::syncInCallback, this);
        trigger_sub2_        = pn.subscribe("trigger", 1, &ProsilicaNodelet::syncInCallback, this);
        exposure_sub         = pn.subscribe("exposure", 1, &ProsilicaNodelet::exposureCallback, this);
        event_sub            = pn.subscribe("/event", 1, &ProsilicaNodelet::eventCallback, this);
        dumper_sub_          = nh.subscribe( image_read, 1, &ProsilicaNodelet::dumperCallback, this);
        shutdown_sub_        = nh.subscribe("/shutdown", 1, cb_request_shutdown);

        /**Setup dynamic reconfigure server
        This for some goofy reason needs to run in order to apply launch settings
         */

        printf("\n <> Setup dynamic reconfigure server <> \n");
        reconfigure_server_.reset(new ReconfigureServer(config_mutex_, pn));
        ReconfigureServer::CallbackType f = boost::bind(&ProsilicaNodelet::reconfigureCallback, this, _1, _2);
        reconfigure_server_->setCallback(f);
        printf("END Setup dynamic reconfigure server <1> \n");


        /**
         * These parameters are a double-edged sword. Increasing GvspRetries can decrease risk of dropped frames if
         * the system is not under heavy load. Under heavy load and unpredictable conditions, this can cause a
         * resend storm that causes no frames to make it. Tune with caution.
         */
        nhp->param<int >("/cfg/prosilica/GvspRetries", gvspRetries_, 3);
        nhp->param<float >("/cfg/prosilica/GvspResendPercent", gvspResendPercent_, 2.0);

        camera_->setAttribute("GvspRetries", (tPvUint32) gvspRetries_); // default: 3.0, too high, and you risk a UDP storm
        camera_->setAttribute("GvspResendPercent", (tPvFloat32) gvspResendPercent_); // default: 1.0

        build_broken_img(broken_img_);


//        archiver_ = std::make_unique<ArchiverFormatter>(ArchiverHelper::from_env());
        ROS_GREEN("<> <> <> Cam init complete 1");

        /// Ignore the first few seconds of frames health-wise since there is a purge
        watchdog.DelayedStart(nhp, 6.0);

    }

    void openCamera()
    {
        int loop_count = 0;
        while (!camera_)
        {
            // For some reason, this doesn't see any on the first try.
            // The startup sequence in general is a hot mess but it eventually works
            NODELET_INFO("== %d Available cameras: %ld ==\n%s", loop_count++,
                    prosilica::numCameras(), getAvailableCameras().c_str());
            NODELET_INFO("==   __________________   ==");

            boost::lock_guard<boost::recursive_mutex> scoped_lock(config_mutex_);
            camera_state_ = OPENING;
            try
            {
                if(guid_ != 0)
                {
                    state_info_ = "Trying to load camera with guid " + hw_id_;
                    NODELET_INFO("%s", state_info_.c_str());
                    camera_ = boost::make_shared<prosilica::Camera>((unsigned long)guid_);
                    updater.setHardwareIDf("%d", guid_);
                    ROS_INFO("Started Prosilica camera with guid \"%lu\"", guid_);

                }
                else if(!ip_address_.empty())
                {
                    state_info_ = "Trying to load camera with ipaddress: " + ip_address_;
                    NODELET_INFO("%s", state_info_.c_str());
                    camera_ = boost::make_shared<prosilica::Camera>(ip_address_.c_str());
                    guid_ = camera_->guid();
                    hw_id_ = boost::lexical_cast<std::string>(guid_);
                    updater.setHardwareIDf("%d", guid_);

                    ROS_INFO("Started Prosilica camera with guid \"%d\"", (int)camera_->guid());
                }
                else
                {
                    updater.setHardwareID("unknown");
                    if(prosilica::numCameras()>0)
                    {
                        state_info_ = "Trying to load first camera found";
                        NODELET_INFO("%s", state_info_.c_str());
                        guid_ = prosilica::getGuid(0);
                        camera_ = boost::make_shared<prosilica::Camera>((unsigned long)guid_);
                        hw_id_ = boost::lexical_cast<std::string>(guid_);
                        updater.setHardwareIDf("%d", guid_);
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
                NODELET_WARN("%s", state_info_.c_str());

                camera_.reset();

            }
            updater.update();
            // this fixes an api change in boost ~1.6-1.7.1
            long milliseconds = (long) (1000 * open_camera_retry_period_);
            boost::this_thread::sleep(boost::posix_time::milliseconds(milliseconds));
        }
        ROS_GREEN("Camera opened. Loading intrinsics. Synching clock");
        loadIntrinsics();
//        prosilica::dumpAttributeList(camera_->handle());
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
            camera_->setKillCallback(boost::bind(&ProsilicaNodelet::kill, this, _1));

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
                NODELET_INFO("Loaded calibration for camera '%s'", camera_name.c_str());
            }
            else
            {
                intrinsics_ = "Failed to load intrinsics from camera";
                NODELET_WARN("Failed to load intrinsics from camera");
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
                    NODELET_INFO("starting camera %s in software trigger mode", hw_id_.c_str());
                    camera_->start(prosilica::Software, 1., prosilica::Continuous);
                    if(update_rate_ > 0)
                    {
                        update_timer_.setPeriod(ros::Rate(update_rate_).expectedCycleTime());
                        update_timer_.start();
                    }
                    break;
                case prosilica::Freerun:
                    NODELET_INFO("starting camera %s in freerun trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaNodelet::publishImage, this, _1));
                    camera_->start(prosilica::Freerun, 1., prosilica::Continuous);
                    break;
                case prosilica::FixedRate:
                    NODELET_INFO("starting camera %s in fixedrate trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaNodelet::publishImage, this, _1));
                    camera_->start(prosilica::FixedRate, update_rate_, prosilica::Continuous);
                    break;
                case prosilica::SyncIn1:
                    NODELET_INFO("starting camera %s in sync1 trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaNodelet::publishImage, this, _1));
                    camera_->start(prosilica::SyncIn1, update_rate_, prosilica::Continuous);
                    break;
                case prosilica::SyncIn2:
                    NODELET_INFO("starting camera %s in sync2 trigger mode", hw_id_.c_str());
                    camera_->setFrameCallback(boost::bind(&ProsilicaNodelet::publishImage, this, _1));
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
            NODELET_INFO("exposure = %s", getCameraAttr("ExposureValue").value.c_str());
        }
        catch(std::exception &e) {
            camera_state_ = CAMERA_NOT_FOUND;
            state_info_ = e.what();
        }
        ROS_GREEN("start() complete");
    }

    void stop()
    {
        update_timer_.stop();
        if(!camera_)
            return;
        camera_->removeEvents();
        camera_->stop();

    }

    void kill(unsigned long guid)
    {
        if(guid == guid_)
        {
            NODELET_WARN("[%s] got Camera::kill() request for prosilica camera %lu",getName().c_str(), guid);
            //! Make sure we interrupt initialization (if it happened to still execute).
            init_thread_.interrupt();
            init_thread_.join();

            camera_state_ = CAMERA_NOT_FOUND;
            state_info_ = "Prosilica camera " + hw_id_ + " disconnected";
            NODELET_ERROR("%s", state_info_.c_str());
            updater.update();
            boost::lock_guard<boost::recursive_mutex> scoped_lock(config_mutex_);
            stop();
            camera_.reset();
            init_thread_ = boost::thread(boost::bind(&ProsilicaNodelet::openCamera, this));
            return;
        }
    }


    int syncCamToSysClock() {
        ros::Time before = ros::Time::now();
        NODELET_INFO("call syncCamToSysClock() ");
        auto err = PvCommandRun(camera_->handle(), "TimeStampValueLatch");
        if (err != ePvErrSuccess) {
            ROS_ERROR("Could not sync clock");
            return (int) err;
        }
        ros::Time after = ros::Time::now();
        tPvUint32 timelo, timehi, freq;
        camera_->getAttribute("TimeStampValueHi", timehi);
        camera_->getAttribute("TimeStampValueLo", timelo);
        camera_->getAttribute("TimeStampFrequency", freq);
        ros::Time tsframe = prosilica::CvtPvTimestamp(timehi, timelo, freq);
        clock_offset = after - tsframe;
        NODELET_INFO("Clock synced, offset = %lf", clock_offset.toSec());
        return 0;
    }

    void publishImageProfile(tPvFrame* frame)
    {
        auto start_time = ros::Time::now();
//        publishImage(frame, ros::Time::now());
        if (frame->ImageBufferSize > 0) {
            auto start_resize = ros::Time::now();
            ROS_WARN("Deprecated! publishImageProfile)");

//            meta_frame->broken.data.resize(frame->ImageBufferSize);
//            ROS_INFO("resizedImage in %ld/%.8f seconds", frame->ImageBufferSize, (ros::Time::now() - start_resize).toSec());
//            auto start_memcpy = ros::Time::now();
//            memcpy(&(meta_frame->broken.data)[0], frame->ImageBuffer, frame->ImageBufferSize);
//            ROS_INFO("memcpy in %ld/%.8f seconds", frame->ImageBufferSize, (ros::Time::now() - start_memcpy).toSec());

        }
        auto end = ros::Time::now();
        ROS_INFO("publishImage0 in %.4f seconds", (end - start_time).toSec());
    }

    /// this will definitely take some finess to figure out. seems like there is a race condition which occasionally starts
    /// on the 2-th (3rd) callback thunk. with the mutex, it freezes, without the mutex, it segs.
    /// buffer initializes in order. error occurs roughly 1 in 5.
    /// seems like when it segs, the meta_frame pointer is bad
    /// I wonder if I should simplify the meta_frame struct in some way
    void postProcessImage(prosilica::MetaFrame *meta_frame, int is_archiving) {
        ROS_WARN("Deprecated! postPRocessImage(MetaFrame*)");

//        ROS_INFO("postProcImage #? entry");
//        ROS_INFO("postProcImage %p #%d entry", meta_frame, meta_frame->idx);
        auto start_time = ros::Time::now();
//        boost::lock_guard<boost::mutex> guard(meta_frame->frameMutex_); // this thing isn't releasing correctly, but without it, there's asegfault
//       meta_frame->frameMutex_.lock(); // deliberate lock for testing
//        ROS_INFO("is_archiving: %d", is_archiving);
        if(is_archiving) {
//            long int sec  = meta_frame->img_.header.stamp.sec;
//            long int nsec = meta_frame->img_.header.stamp.nsec;
//            std::string filename = ArchiverHelper::generateFilename(envoy_, arch_opts_, sec, nsec);
//            auto filename_written = dumpImageMessage(meta_frame->img_, filename);
//            ROS_INFO("dumped #%d %s",meta_frame->idx, filename_written.c_str());
        }
        auto end = ros::Time::now();
        ROS_WARN("postProcImage #%d in %.4f seconds",meta_frame->idx, (end - start_time).toSec());
    }

    void nop() {
        ROS_WARN("nop");
    }

    /// todo: variably disable archiving and/or publishing. totally remove it and profile ePvWhatevr
    void publishImage(tPvFrame* frame)
    {   bool ok = false;
        try {
            auto recv_time = ros::Time::now();
            ROS_INFO("yeeting frame buffer into new thread callback");
            if (false ) {
                prosilica::PvFrameWrapperPtr pframe = prosilica::PvFrameWrapper::make_shared(frame);
                /// ProsilicaNodelet *this, shared<PvFrameWrap>, ros::Time
                ros::TimerCallback cb = [this, pframe, recv_time](ros::TimerEvent const &e) {
                    this->publishImage(pframe, recv_time);
                };
                oneShotManager.addOneShot(nhp, ALMOST_INSTANT, cb);
            } else {
                this->publishImageOld(frame, recv_time);
            }
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

    void publishImage(prosilica::PvFrameWrapperPtr pframe, ros::Time time) {
        tPvFrame *frameptr = &pframe->frame_;
        publishImageOld(frameptr, time);
    }

    void publishImageOld(tPvFrame* frame, ros::Time time)
    {
        frame_recv_time_ = ros::Time::now();
        int is_archiving = 0;
//        ROS_WARN("about to get");

//        std::string msg = envoy_->get("foo");
//        ROS_WARN(msg.c_str());

//        envoy_->get()
//        nh.getParam("/sys/arch/is_archiving", is_archiving);

        camera_state_ = OK;
        state_info_ = "Camera operating normally";

        /** allow most recent event to be received
         * There is a lot of async going on here, I'm sure there are better ways.
         * Really the best way would be to use the PvCaptureWaitForFrameDone api
         * call and do things more sync-like. But I am out of time for this problem.
         */
        int seq_dt = 0;
        int loop_count = 0;
        do {

            seq_dt = event_.header.seq - last_published_.seq;
        } while (seq_dt < 1 && loop_count++ < 3);

        std_msgs::Header gps_header;
        /// todo: null check here or use context manager
        prosilica::MetaFrame* meta_frame = (prosilica::MetaFrame*) frame->Context[0];
        if (!meta_frame) {
            ROS_ERROR("tPvFrame context is null");
            return;
        }
        ROS_INFO("FrameDone %p #%ld @ %ld %ld", meta_frame, (long int) meta_frame->idx, (long int) frame->TimestampHi, (long int) frame->TimestampLo);
        auto frame_stamp_time = prosilica::CvtPvTimestamp(frame->TimestampHi, frame->TimestampLo) + clock_offset;

        bool success = event_cache.search(frame_recv_time_, gps_header);

        ros::Time tsframe = prosilica::CvtPvTimestamp(frame->TimestampHi, frame->TimestampLo);
        ros::Time corrFrameTime = tsframe + clock_offset;
        NODELET_INFO("frameTime: %16.4f GPS: %16.4f DT: %7.4f", corrFrameTime.toSec(), gps_header.stamp.toSec(), (corrFrameTime-gps_header.stamp).toSec());
//        std::cout << "frameTime: " << corrFrameTime << ", GPS: " << gps_header.stamp << ", DT" <<  std::endl;
//        std::cout << "ROS: "<< ros::Time::now() << ", frameTime: " << tsframe + clock_offset << ", TSF: " << tsframe << ", GPS: " << gps_header.stamp<< std::endl;
//        std::cout << "ROS: "<< ros::Time::now() << ", frameTime: " << tsframe + clock_offset << ", TSF: " << tsframe << ", GPS: " << gps_header.stamp<< std::endl;
        std::stringstream this_frame_id;
        this_frame_id << frame_id_;

        // convey the status of the event binding process
        if (success) {
            this_frame_id << "?lock=1&eventNum=" << gps_header.seq << "&eventTime" << gps_header.stamp.toSec() ;
        } else {
            this_frame_id << "?lock=0";
        }


        ros::Duration delta = frame_recv_time_ - event_.gps_time;
//        auto seq = event_.header.seq;
//        ROS_DEBUG("[%d] evt seq ", seq);
        if (image_publisher_.getNumSubscribers() > 0)
        {
            auto nodeName = ros::this_node::getName();
            std::stringstream link;
            custom_msgs::Stat stat_msg;
            stat_msg.header.stamp = ros::Time::now();
            stat_msg.trace_topic = nodeName + "/publishImage";
            stat_msg.node = nodeName;
            stat_msg.trace_header = std_msgs::Header(img_.header);
            stat_msg.trace_header.seq = count_;
            link << nodeName << "/event/" << event_.header.seq; // link this trace to the event trace
            stat_msg.link = link.str();
            meta_frame->img_.header.stamp = event_.gps_time;
            if (seq_dt > 1) {
                ROS_ERROR("[%d] Missed %d frames, based on event seq ", event_.header.seq, seq_dt - 1);
                for (auto i = 0; i < 4 && i < seq_dt - 1; i++) {
                    watchdog.kick();
                }
            }
            sensor_msgs::ImagePtr p_img = boost::make_shared<sensor_msgs::Image>(meta_frame->img_);

            if (processFrame(frame, *p_img, cam_info_))  // this will memcpy frame's buffer into img_
            {
                // time taken to buffer image from cam
//                ROS_DEBUG("[%d][3] Pub'ing! === Elapsed: %2.3f", seq, delta.toSec());
//                ROS_INFO("Publishing! === ===        : %2.3f", img_.header.stamp.toSec());

//                 Set the image timestamp to match the event that actually triggered it
//              ROS image transport does its own thing with header sequence here, so we can't actually rely on that
//              to group the events together. Fortunately, the time stamp itself is hashable.
                if (success) {
                    stat_msg.note = "success";
                    img_.header = gps_header;
                }
                sensor_msgs::CameraInfoConstPtr pc_cam_info = boost::make_shared<sensor_msgs::CameraInfo>(cam_info_);
                meta_frame->img_.header.frame_id = this_frame_id.str();
                stat_pub_.publish(stat_msg);
                image_publisher_.publish(p_img, pc_cam_info);
//                ROS_WARN("<d> %d %d", event_.header.seq, img_.header.seq);
                frames_dropped_acc_.add(0);

            }
            else
            {
                ROS_ERROR("[?][3] Frame parse failed, checking status");
                auto status = frame->Status;
                ROS_ERROR("[%d][3] Frame parse failed, frame status: %d %s", event_.header.seq, status, pv_error_codes[status]);
                camera_state_ = FORMAT_ERROR;
                state_info_ = "Unable to process frame";
                this_frame_id << "&status=" << status << "&error=" << pv_error_codes[status];
                std_msgs::Header msg = std_msgs::Header(img_.header);
                broken_img_.header.stamp = img_.header.stamp;
                broken_img_.header.frame_id = this_frame_id.str();
                msg.seq = (unsigned int) ++frames_dropped_total_;
                missed_frames_pub_.publish(msg);
                stat_msg.note = pv_error_codes[status];
                stat_pub_.publish(stat_msg);
                errstat_pub_.publish(stat_msg);
                image_publisher_.publish(broken_img_, cam_info_);
                frames_dropped_acc_.add(1);
            }
            last_published_ = event_.header;


            ++frames_completed_total_;
            frames_completed_acc_.add(1);
        }
        updater.update();
        auto end = ros::Time::now();
        ROS_INFO("publishImage1 in %.4f seconds", (end - frame_recv_time_).toSec());
    }

    void updateCallback(const ros::TimerEvent &event)
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
                ROS_WARN("is this enabled?");
                publishImageOld(frame, event.current_real);
            }
            catch(std::exception &e)
            {
                camera_state_ = ERROR;
                state_info_ = e.what();
                NODELET_ERROR("Unable to read from camera: %s", e.what());
                ++frames_dropped_total_;
                frames_dropped_acc_.add(1);
                updater.update();
                return;
            }
        }
    }

    void pollCallback(polled_camera::GetPolledImage::Request& req,
                      polled_camera::GetPolledImage::Response& rsp,
                      sensor_msgs::Image& image, sensor_msgs::CameraInfo& info)
    {
        if (trigger_mode_ != prosilica::Software)
        {
            rsp.success = false;
            rsp.status_message = "Camera is not in software triggered mode";
            return;
        }

        last_config_.binning_x = req.binning_x;
        last_config_.binning_y = req.binning_y;
        last_config_.x_offset = req.roi.x_offset;
        last_config_.y_offset = req.roi.y_offset;
        last_config_.height   = req.roi.height;
        last_config_.width    = req.roi.width;

        reconfigureCallback(last_config_, dynamic_reconfigure::SensorLevels::RECONFIGURE_RUNNING);

        try
        {
            tPvFrame* frame = NULL;
            frame = camera_->grab(req.timeout.toSec()*100);
            if (processFrame(frame, image, info))
            {
                image.header.stamp = info.header.stamp =rsp.stamp = ros::Time::now();
                rsp.status_message = "Success";
                rsp.success = true;
            }
            else
            {
                rsp.success = false;
                rsp.status_message = "Failed to process image";
                return;
            }
        }
        catch(std::exception &e)
        {
            rsp.success = false;
            std::stringstream err;
            err<< "Failed to grab frame: "<<e.what();
            rsp.status_message = err.str();
            return;
        }
    }

    void syncInCallback (const std_msgs::HeaderConstPtr& msg)
    {
        printf("\n <> syncInCallback <> \n");
        if (trigger_mode_ != prosilica::Software)
        {
            camera_state_ = ERROR;
            state_info_ = "Can not sync from topic trigger unless in Software Trigger mode";
            NODELET_ERROR_ONCE("%s", state_info_.c_str());
            return;
        }
        ros::TimerEvent e;
        e.current_real = msg->stamp;
        updateCallback(e);
    }

    void dumperCallback (const sensor_msgs::ImageConstPtr &msg) {
        int is_archiving = ArchiverHelper::get_is_archiving(envoy_, "/sys/arch/is_archiving");
        NODELET_INFO("dumper: is archiving: %d", is_archiving);
        if(is_archiving) {
            long int sec  = msg->header.stamp.sec;
            long int nsec = msg->header.stamp.nsec;
            std::string filename = ArchiverHelper::generateFilename(envoy_, arch_opts_, sec, nsec);
            try {
                bool debayer{false};

                if ("rgb" == cam_channel) {
                    debayer = true;
                }
                auto filename_written = dumpImageMessage(msg, filename, debayer );
                ROS_INFO("[%s] dumped #%d %s", cam_channel.c_str(), msg->header.seq, filename_written.c_str());
            } catch (cv_bridge::Exception &e) {
                ROS_ERROR("%s", e.what());
            }

        }
    }

        ///  void (ProsilicaNodelet::*)(const custom_msgs::GSOF_WVTConstPtr&) = ProsilicaNodelet::eventCallback
    void eventCallback (const boost::shared_ptr<custom_msgs::GSOF_EVT const>& msg)
    {
        ROS_INFO("[%d]<1> eventCallback <>         %2.2f", msg->header.seq, msg->gps_time.toSec());
        // I am pretty sure this does not reference leak but this is not my wheelhouse
        event_ = *msg;
        event_cache.push_back(msg->sys_time, msg);
        auto nodeName = getName();
        custom_msgs::Stat stat_msg;
        std::stringstream link;
        stat_msg.header.stamp = ros::Time::now();
        stat_msg.trace_header = (*msg).header;
        stat_msg.trace_topic = nodeName + "/eventCallback";
        stat_msg.node = nodeName;
        link << nodeName << "/event/" << event_.header.seq; // link this trace to the event trace
        stat_msg.link = link.str();
        stat_pub_.publish(stat_msg);
        event_cache.purge();
        watchdog.check();
//        event_cache.show();


//        std::cout << stat_msg << "\n---" << std::endl;
        }

    /** Exposure is in microseconds (microsecs). Minimum varies by camera I think. */
    void exposureCallback (const std_msgs::Int32Ptr &msg)
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
//        camera_->setAttribute("ExposureValue", (tPvUint32)microsecs); // redundant?

    }

    void gainCallback (const std_msgs::Int32Ptr &msg)
    {
        printf("\n <> gainCallback <> \n");
        const tPvUint32 gain_max = 24;
        int32_t gain = msg->data;
        if (gain < 0) {
            printf("WARNING: Gain set to less that zero. Setting auto_gain once. This is not a recommended feature");
            camera_->setGain(gain, prosilica::AutoOnce);
            return;
        }
        tPvUint32 ugain = (tPvUint32) gain;
        if (ugain > gain_max) {
            printf("WARNING: Gain set to greater than max allowed. Clipping to %lu/n", gain_max);
            ugain = gain_max;
        }
        printf("INFO: Gain set to: %lu", ugain);
        camera_->setGain( ugain, prosilica::Manual);

    }
    bool health(std_srvs::TriggerRequest &req, std_srvs::TriggerResponse &rsp) {
        auto healthy = watchdog.Ok();
        rsp.success = healthy;
        if (!healthy) {
            rsp.message = "Watchdog timed out";
        }
        return healthy;
    }


            // todo: check if I want return false, or return true with error code on srv
    /** this is a pretty gross api. It's stringly-typed, so be careful
     * also currently does not work with certain types.
     * Note: use "1"/"0" for pushing bools. They are pretty rare though*/
    bool setCameraAttr(custom_msgs::CamSetAttrRequest &req, custom_msgs::CamSetAttrResponse &rsp) {
        NODELET_INFO("<API> setCameraAttr(%s, %s)", req.name.c_str(), req.value.c_str());
        tPvHandle handle = camera_->handle();
        rsp.tPvErr = ePvErrUnknown;
        rsp.value = "error";
        const char *c_name = req.name.c_str();

        if (req.name == "SyncClock") {
            rsp.tPvErr = syncCamToSysClock();
            rsp.dtype = "time";
            rsp.value = std::to_string(clock_offset.toSec());
            return (rsp.tPvErr != 0);
        }

        /** On failure, pass error to message. This is more descriptive than
         * return false*/
        rsp.tPvErr = PvAttrIsAvailable(handle, c_name);
        if (rsp.tPvErr != 0) { return true; }

        tPvAttributeInfo info;

        rsp.tPvErr = PvAttrInfo(handle, c_name, &info);
        if (rsp.tPvErr != 0) { return true; }


        switch (info.Datatype) {
            case ePvDatatypeEnum: {
                rsp.tPvErr = PvAttrEnumSet(
                        handle, c_name, req.value.c_str());
                rsp.dtype = "enum";
                break;
            }
            case ePvDatatypeString: {
                rsp.tPvErr = PvAttrStringSet(
                        handle, c_name, req.value.c_str());
                rsp.dtype = "string";
                break;
            }
            case ePvDatatypeUint32: {
                rsp.tPvErr = PvAttrUint32Set(
                        handle, c_name, std::stoul(req.value));
                rsp.dtype = "uint32";
                break;
            }
            case ePvDatatypeInt64: {
                rsp.tPvErr = PvAttrInt64Set(
                        handle, c_name, std::stol(req.value));
                rsp.dtype = "int64";
                break;
            }
            case ePvDatatypeFloat32: {
                rsp.tPvErr = PvAttrFloat32Set(
                        handle, c_name, std::stof(req.value));
                rsp.dtype = "float32";
                break;
            }
            case ePvDatatypeBoolean: {
                rsp.tPvErr = PvAttrBooleanSet(
                        handle, c_name, std::stoi(req.value));
                rsp.dtype = "bool";
                break;
            }
            case ePvDatatypeCommand: {
                rsp.tPvErr = PvCommandRun(
                        handle, c_name);
                rsp.dtype = "PvCommandRun";
                rsp.value = "ok";
                return true;
            }
            default: {
                rsp.tPvErr = ePvErrBadParameter;
                break;
            }
        }
        if (rsp.tPvErr != 0) { return true; }
        custom_msgs::CamGetAttrResponse new_rsp = getCameraAttr(req.name);
        rsp.value = new_rsp.value;

        return true;
    }

    bool getCameraAttr(custom_msgs::CamGetAttrRequest &req, custom_msgs::CamGetAttrResponse &rsp) {
        NODELET_INFO("<API> getCameraAttr(%s)", req.name.c_str());

        tPvUint32 value;
        try {
            rsp = getCameraAttr(req.name);
        }
        catch (prosilica::ProsilicaException) {
            rsp.value = "error";
        }
        return true;
    }

    custom_msgs::CamGetAttrResponse getCameraAttr(std::string name) {
        tPvHandle handle = camera_->handle();
        custom_msgs::CamGetAttrResponse rsp;
        rsp.tPvErr = ePvErrUnknown;
        rsp.value = "error";
        const char *c_name = name.c_str();

        rsp.tPvErr = PvAttrIsAvailable(handle, c_name);
        if (rsp.tPvErr != 0) { return rsp; }

        tPvAttributeInfo info;

        rsp.tPvErr = PvAttrInfo(handle, c_name, &info);
        if (rsp.tPvErr != 0) { return rsp; }


        switch (info.Datatype) {
            case ePvDatatypeEnum:
                camera_->getAttributeEnum(name, rsp.value);
                rsp.tPvErr = ePvErrSuccess;
                rsp.dtype = "enum";
                break;
            case ePvDatatypeString:
                rsp.tPvErr = prosilica::getAttribute(handle, c_name, rsp.value);
                rsp.dtype = "string";
                break;
            case ePvDatatypeUint32:
                tPvUint32 value;
                rsp.tPvErr = PvAttrUint32Get(handle, c_name, &value);
                rsp.dtype = "uint32";
                rsp.value = std::to_string((unsigned long) value);
                break;
            case ePvDatatypeInt64:
                tPvInt64 l_value;
                rsp.tPvErr = PvAttrInt64Get(handle, c_name, &l_value);
                rsp.dtype = "int64";
                rsp.value = std::to_string((int64_t) l_value);
                break;
            case ePvDatatypeFloat32:
                tPvFloat32 f_value;
                rsp.tPvErr = PvAttrFloat32Get(handle, c_name, &f_value);
                rsp.dtype = "float32";
                rsp.value = std::to_string((float) f_value);
                break;
            case ePvDatatypeBoolean:
                tPvBoolean b_value;
                rsp.tPvErr = PvAttrBooleanGet(handle, c_name, &b_value);
                rsp.dtype = "bool";
                rsp.value = std::to_string((bool) b_value);
                break;
            default:
                rsp.tPvErr = ePvErrBadParameter;
                break;
        }
        return rsp;

    }

        bool getAttrList(custom_msgs::StrListRequest &req, custom_msgs::StrListResponse &rsp) {
            tPvAttrListPtr pListPtr;
            unsigned long sz;
            //typedef const char* const* tPvAttrListPtr;
            // The attribute list is contained in memory allocated by the PvApi module.
            tPvErr err = PvAttrList(camera_->handle(), &pListPtr, &sz);
            NODELET_WARN("sz: %ld ", sz);
            if (err != 0) {
                rsp.tPvErr = err;
                rsp.values[0] = "error";
                return false;
            }
            for (int i = 0; i < sz; i++) {
                rsp.values.push_back(pListPtr[i]);
            }
            return true;
        }
        // this calls frameToImage which calls fillImage which calls memcpy
    bool processFrame(tPvFrame* frame, sensor_msgs::Image &img, sensor_msgs::CameraInfo &cam_info)
    {
        auto start_time = ros::Time::now();
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
            NODELET_WARN("Data Missing from Frame. This may fail");
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
//                NODELET_WARN("Failed to parse frame to image message");
                return false;
            } /// this suceeds === === === === === ===  VVV
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

            /// this doesn't seem like it's doing anything, but chesterton's fence
            if (false) {
                if (auto_adjust_stream_bytes_per_second_ && camera_->hasAttribute("StreamBytesPerSecond"))
                    camera_->setAttribute("StreamBytesPerSecond", (tPvUint32)(camera_->max_data_rate / num_cameras));
            }
        }
        catch(std::exception &e)
        {
            return false;
        }
        auto end = ros::Time::now();
        /// === === === === === ===  ^^^
        ROS_INFO("processFrame in %.4f seconds", (end - start_time).toSec());

        count_++;
        return true;
    }

    bool frameToImage(tPvFrame* frame, sensor_msgs::Image &image)
    {
        auto start_time = ros::Time::now();

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
        NODELET_WARN("Received frame with unsupported pixel format %d", frame->Format);
        return false;
      }


      if(frame->ImageSize == 0) {
          /** image size for GT6600 is 28829184. You can try to spoof the buffer
           * size check but you will get stale data
           */

//          NODELET_WARN("Image size is zero but will try to recover");
//          frame->ImageSize = ; // hack
          NODELET_WARN("Image size is zero");
          return false;
      }
      if(frame->Height == 0) {
          NODELET_WARN("Image height is zero");
          return false;
      }

      uint32_t step = frame->ImageSize / frame->Height;
      // fillImage calls memcpy
      auto out = sensor_msgs::fillImage(image, encoding, frame->Height, frame->Width, step, frame->ImageBuffer);
        auto end = ros::Time::now();
        /// makes it to here before freezing
        ROS_INFO("frameToImage in %.4f seconds, format: %ld", (end - start_time).toSec(), (long int) frame->Format);

        return out;
    }

    bool setCameraInfo(sensor_msgs::SetCameraInfoRequest &req, sensor_msgs::SetCameraInfoResponse &rsp)
    {
        NODELET_INFO("<API> New camera info received");
        sensor_msgs::CameraInfo &info = req.camera_info;

        // Sanity check: the image dimensions should match the max resolution of the sensor.
        if (info.width != sensor_width_ || info.height != sensor_height_)
        {
            rsp.success = false;
            rsp.status_message = (boost::format("Camera_info resolution %ix%i does not match current video "
                                                "setting, camera running at resolution %ix%i.")
                                                 % info.width % info.height % sensor_width_ % sensor_height_).str();
            NODELET_ERROR("%s", rsp.status_message.c_str());
            return true;
        }

        stop();

        std::string cam_name = "prosilica";
        cam_name += hw_id_;
        std::stringstream ini_stream;
        if (!camera_calibration_parsers::writeCalibrationIni(ini_stream, cam_name, info))
        {
            rsp.status_message = "Error formatting camera_info for storage.";
            rsp.success = false;
        }
        else
        {
            std::string ini = ini_stream.str();
            if (ini.size() > prosilica::Camera::USER_MEMORY_SIZE)
            {
                rsp.success = false;
                rsp.status_message = "Unable to write camera_info to camera memory, exceeded storage capacity.";
            }
            else
            {
                try
                {
                    camera_->writeUserMemory(ini.c_str(), ini.size());
                    cam_info_ = info;
                    rsp.success = true;
                }
                catch (prosilica::ProsilicaException &e)
                {
                    rsp.success = false;
                    rsp.status_message = e.what();
                }
            }
        }
        if (!rsp.success)
        NODELET_ERROR("%s", rsp.status_message.c_str());

        start();

        return true;
    }

    void reconfigureCallback(prosilica_camera::ProsilicaCameraConfig &config, uint32_t level)
    {
        printf("\n<> Reconf callback \n");
        NODELET_DEBUG("Reconfigure request received");

        if (level >= (uint32_t)dynamic_reconfigure::SensorLevels::RECONFIGURE_STOP)
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
            NODELET_ERROR("Invalid trigger mode '%s' in reconfigure request", config.trigger_mode.c_str());
        }

        if(config.trig_timestamp_topic != last_config_.trig_timestamp_topic)
        {
            trigger_sub_.shutdown();
            trig_timestamp_topic_ = config.trig_timestamp_topic;
        }

        if(!trigger_sub_ && config.trigger_mode == "triggered")
        {
            trigger_sub_ = ros::NodeHandle().subscribe(trig_timestamp_topic_, 1, &ProsilicaNodelet::syncInCallback, this);
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
                NODELET_WARN("Auto gain not available for this camera. Auto gain is available "
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
                NODELET_WARN("Auto white balance not available for this camera.");
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
            NODELET_WARN("Binning not available for this camera.");
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
        /// @todo Replicating logic from polledCallback
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
        //! Reload last good config
        if (level >= (uint32_t)dynamic_reconfigure::SensorLevels::RECONFIGURE_STOP)
        {
            try
            {
                start();
            }
            catch(std::exception &e)
            {
                NODELET_ERROR("Invalid settings: %s", e.what());
                config = last_config_;
            }
        }

        last_config_ = config;
    }

    void getCurrentState(diagnostic_updater::DiagnosticStatusWrapper &stat)
    {
        stat.add("Serial", guid_);
        stat.add("Info", state_info_);
        stat.add("Intrinsics", intrinsics_);
        stat.add("Total frames dropped", frames_dropped_total_);
        stat.add("Total frames", frames_completed_total_);

        if(frames_completed_total_>0)
        {
            stat.add("Total % frames dropped", 100.*(double)frames_dropped_total_/frames_completed_total_);
        }
        if(frames_completed_acc_.sum()>0)
        {
            stat.add("Recent % frames dropped", 100.*frames_dropped_acc_.sum()/frames_completed_acc_.sum());
        }

        switch (camera_state_)
        {
            case OPENING:
                stat.summary(diagnostic_msgs::DiagnosticStatus::WARN, "Opening camera");
                break;
            case OK:
                stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "Camera operating normally");
                break;
            case CAMERA_NOT_FOUND:
                stat.summaryf(diagnostic_msgs::DiagnosticStatus::ERROR, "Can not find camera %d", guid_ );
                stat.add("Available Cameras", getAvailableCameras());
                break;
            case FORMAT_ERROR:
                stat.summary(diagnostic_msgs::DiagnosticStatus::ERROR, "Problem retrieving frame");
                break;
            case ERROR:
                stat.summary(diagnostic_msgs::DiagnosticStatus::ERROR, "Camera has encountered an error");
                break;
            default:
                break;
        }
    }
};



} // end namespace

/** === === === === === === === === === === === ===  */
void driver_shutdown() {
    for ( auto const& it: active_nodelets)
    {
        ROS_WARN("Stopping nodelet %d", it.first);
        it.second->public_stop();
    }
}

/** Makes a placeholder image for broken frames
 * @param img - image message to populate
 * @return success
 */
bool build_broken_img(sensor_msgs::Image& img) {
    sensor_msgs::clearImage(img);
}


#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(prosilica_camera::ProsilicaNodelet, nodelet::Nodelet);

