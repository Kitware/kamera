#include <chrono>
#include <csignal>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <string>
#include <mutex>
#include <deque>
#include <atomic>
#include <thread>

// ROS stuff
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/u_int8.hpp"
#include "std_msgs/msg/int8.hpp"
#include "std_msgs/msg/header.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <roskv/envoy.h>
#include <roskv/archiver.h>

// Includes from: /opt/genicam_v3_0/library/CPP/include/
#include <GenApi/GenApi.h>
// Includes from: /usr/dalsa/GigeV/include/
#include <gevapi.h>

#include <custom_msgs/msg/gsof_evt.hpp>
#include <custom_msgs/msg/stat.hpp>
#include <custom_msgs/srv/cam_get_attr.hpp>
#include <custom_msgs/srv/cam_set_attr.hpp>
#include <custom_msgs/srv/str_list.hpp>
#include <cam_utils/event_cache.hpp>


#include "utils.h"
#include "macros.h"
#include "decode_error.h"
#include "spec_a6750.h"

using namespace std::chrono;

enum FirmwareMode { Bayer=0, Color=1, Mono=2, Mono8=3, Mono16=4};

/// Forward declarations

class Transporter {
public:
    Transporter(rclcpp::Node::SharedPtr nhp, const std::string &out_topic) :
            it_raw(nhp),
            it_raw_pub(it_raw.advertise(out_topic, 1)) {}

    image_transport::ImageTransport it_raw;
    image_transport::Publisher it_raw_pub;
};


int purge_stale(std::map<rclcpp::Time, sensor_msgs::msg::Image::SharedPtr> &image_map, rclcpp::Time t)
{
    int stale = 0;
    for (const auto pair: image_map) {
        if (pair.first < t)
        {
            image_map.erase(pair.first);
            stale++;
        }
    }
    return stale;
};


std::string dumpImageMessage(const sensor_msgs::msg::Image::SharedPtr & received_image, const std::string &filename)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    cv_bridge::CvImagePtr cvPtr;
    auto start_db = steady_clock::now();
    cvPtr = cv_bridge::toCvCopy(received_image, received_image->encoding);
    ROS_INFO("debayered %ld in %2.3f ", (long int)(cvPtr->image.total() * cvPtr->image.elemSize()),
             duration<double>(steady_clock::now() - start_db).count());

    start_db = steady_clock::now();

    std::filesystem::path path_filename{filename};
    std::filesystem::create_directories(path_filename.parent_path());
    cv::imwrite(filename, cvPtr->image, compression_params);
    ROS_INFO("dumped   %ld in %2.3f ", (long int)(cvPtr->image.total() * cvPtr->image.elemSize()),
             duration<double>(steady_clock::now() - start_db).count());
    return filename;
}


class CameraTimeSync {
public:
    static rclcpp::Time timestampToRos(uint32_t timehi, uint32_t timelo) {
        return timestampToRos(timehi, timelo, 1000000000);
    }
    static rclcpp::Time timestampToRos(uint32_t timehi, uint32_t timelo, uint32_t freq) {
        uint64_t utime = ((uint64_t )timehi) << 32;
        utime = utime + (uint64_t )timelo;
        double dtime = double (utime) / (double) freq;
        return rclcpp::Time{(int64_t)(dtime * 1e9), RCL_ROS_TIME};
    }
};


// ----------------------------------------------------------------------------
// Global Variables
volatile bool G_SIGINT_TRIGGERED = false;
bool G_NEW_BRIGHTNESS = false;
bool G_TIMING_VERBOSE = false;
uint8_t G_INFO_VERBOSITY = 0;
Trigger trigger{};

void signalHandler( int signum ) {
    ROS_WARN("<!> Interrupt signal (%d)\n", signum);
    std::cerr << "<!!> Interrupt signal " << signum << std::endl;
    G_SIGINT_TRIGGERED = true;
}

void cb_request_shutdown(std_msgs::msg::Int8 const &msg) {
    ROS_INFO("Requesting clean shutdown: %d", msg.data);
    G_SIGINT_TRIGGERED = true;
    rclcpp::shutdown();
}


/**
 * Encapsulation of this ROS node's settings.
 *
 * Contains various helper methods to fill in node settings into various
 * GEV structures as appropriate.
 */
struct NodeSettings {
    FirmwareMode firmware_mode;     // enum for pixel output format/bitrate

    std::string camType;
    std::string firmwareMode;
    std::string triggerSource;
    std::string frameSyncSource;
    std::string triggerNodeName;
    std::string irFormat;

    bool triggeredBySoftware;
    bool renormalize = FALSE;        // Rescale all pixel values to the min/max of array

    //+ Camera connection parameters
    uint32_t uint_cam_ip_addr = 0;
    CameraIdentifier camera_id;

    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<RedisEnvoy>         envoy_;

    /** XML Feature settings parameters
     * Optional path to the XML feature settings file to load.
     * If no settings file is to be used, this should be an empty string. */
    std::string xmlFeatures_filepath;
    bool xmlFeatures_autoBrightness;        // If we should attempt to manually turn on the auto brightness feature.
    std::atomic_int xmlFeatures_autoBrightnessTarget;
    int xmlFeatures_BalanceWhiteAuto;

     /** Number of image buffers to allocate for image acquisition.

     We use the SynchronousNextEmpty mode, which fills buffers in the order
     they are released back to the acquisition process. If there are no more
     buffers available, subsequent images are dropped on the floor. */
    UINT32 imageTransfer_numImageBuffers;
    UINT32 nextImage_timeout;               // Timeout in milliseconds to wait for the next image frame from the camera.

    /** Output parameters
     * Optional output image cropping pixel row specifications for the top and
     * bottom of the crop.  Negative values disable dropping the top or bottom. */
    int output_image_crop_top_row,
            output_image_crop_bot_row;

    std::string frame_id;                       // Frame ID string to set in output messages.
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr brightness_sub;   // handle to subscriber
    rclcpp::Subscription<custom_msgs::msg::GsofEvt>::SharedPtr event_sub;   // handle to gps events
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr shutdown_sub;      // sub for clean shutdown
    rclcpp::Publisher<custom_msgs::msg::Stat>::SharedPtr stat_pub;          // handle to tracing stat
    rclcpp::Publisher<custom_msgs::msg::Stat>::SharedPtr errstat_pub;       // handle to error tracing
    std::string output_topic_raw;               // Name of topic to output raw image to.
    float output_frame_rate;                    // Output frame-rate (hz).
    std::string output_topic_debayer;           // Topic to output debayered image to.
    /**   If output_topic_debayer is an empty string, debayering does not occur. */

    std::string rawImageCVEncoding;
    int rawImageCVMatType;

    std_msgs::msg::Header last_published_; // Last header which was successfully published
    custom_msgs::msg::GsofEvt event_;     // store the last received event

    /**
     * Construct settings from the node.
     *
     * \throws ConfigurationError Failed extracting setting values appropriately.
     */
    NodeSettings(rclcpp::Node::SharedPtr node) : node_{node} {
        bool failed(false);
        rclcpp::Node &nh = *node;

        G_INFO_VERBOSITY = static_cast<UINT8>( parse_pos_int(nh, "info_verbosity", failed, 0) );
        ROS_INFO("Info Verbosity set to %d", G_INFO_VERBOSITY);

        camType = parse_validate_arg(nh, "cam_type", CamTypeMap, failed);
        firmwareMode = parse_validate_arg(nh, "firmware_mode", FirmModeMap, failed);
        triggerSource = parse_validate_arg(nh, "trigger_source", TrigSrcMap, failed);
        frameSyncSource = parse_validate_arg(nh, "frame_sync_source", TrigSrcMap, failed);
        irFormat = parse_validate_arg(nh, "ir_format", IrFormatMap, failed);
        firmware_mode = static_cast<FirmwareMode > (arg2enum(firmwareMode, FirmModeMap));
        triggeredBySoftware = (triggerSource == "Software") ? TRUE : FALSE;
        trigger.set_fire_state(triggeredBySoftware);

        RedisEnvoyOpts envoy_opts = RedisEnvoyOpts::from_env("driver_" + camType + "_" + "nofovset" );
        envoy_ = std::make_shared<RedisEnvoy>(envoy_opts);
        ROS_WARN("echo: %s", envoy_->echo("Redis connected").c_str());

        /** Cache the values for data encoding to pass to cvBridge.
         * In this application, we don't expect it to change after initialization.
         */
        rawImageCVEncoding = FirmModeToCVEnc[firmwareMode];
        rawImageCVMatType = FirmModeToCVMat[firmwareMode];

        camera_id.username = nh.declare_parameter("camera_username", std::string());
        camera_id.manufacturer = nh.declare_parameter("camera_manufacturer", std::string());
        camera_id.ip_addr = nh.declare_parameter("camera_ip_addr", std::string());
        camera_id.serial = nh.declare_parameter("camera_serial", std::string());
        camera_id.mac = nh.declare_parameter("camera_mac", std::string());

        if (G_INFO_VERBOSITY >= 2) { print_camera_id(camera_id); }

        if (!camera_id.ip_addr.empty()){
            uint_cam_ip_addr = ip_string_to_uint(camera_id.ip_addr);
        } else {
            // This takes some time, so avoid if we aready know IP
            uint_cam_ip_addr = locate_camera(camera_id);
        }

        if (uint_cam_ip_addr == 0) {
            ROS_ERROR("Unable to locate a valid IP address for Genicam Camera");
            failed = true;
        } else {
            ROS_GREEN("SUCCESS! located genicam at");
            ROS_INFO(" %ud", uint_cam_ip_addr);
        }

        // Optional path to an XML settings file to load and use.
        xmlFeatures_filepath = nh.declare_parameter("xmlFeatures_filepath", std::string());

        xmlFeatures_autoBrightness = nh.declare_parameter("xmlFeatures_autoBrightness", false);
        xmlFeatures_autoBrightnessTarget = nh.declare_parameter("xmlFeatures_autoBrightnessTarget", 128);
        // 0 - Off, 1 - On Demand, 2 - Periodic
        xmlFeatures_BalanceWhiteAuto = nh.declare_parameter("xmlFeatures_BalanceWhiteAuto", 0);

        imageTransfer_numImageBuffers = static_cast<UINT32>( parse_pos_int(nh, "imageTransfer_numImageBuffers", failed) );
        nextImage_timeout = static_cast<UINT32>( parse_pos_int(nh, "nextImage_timeout", failed) );
        output_frame_rate = parse_pos_float(nh, "output_frame_rate", failed);

        output_image_crop_top_row = nh.declare_parameter("output_image_crop_top_row", -1);
        output_image_crop_bot_row = nh.declare_parameter("output_image_crop_bot_row", -1);
        if (output_image_crop_bot_row >= 0 &&
            output_image_crop_bot_row <= output_image_crop_top_row) {
            failed = true;
            ROS_ERROR("Bottom crop row must be greater than top crop row (%d !> %d).",
                      output_image_crop_bot_row, output_image_crop_top_row);
        }

        frame_id = nh.declare_parameter("frame_id", std::string());
        if (frame_id.empty()) {
            failed = true;
            ROS_ERROR("No frame ID provided!");
        }
        output_topic_raw = nh.declare_parameter("output_topic_raw", std::string());
        if (output_topic_raw.empty()) {
            failed = true;
            ROS_ERROR("No output topic string provided");
        }
        // Debayer output is optional
        // - Debayering is undefined if the firmware is not set to bayer mode.
        output_topic_debayer = nh.declare_parameter("output_topic_debayer", std::string());
        if (output_topic_debayer.size() > 0
            && firmware_mode != FirmwareMode::Bayer) {
            failed = true;
            ROS_ERROR("Cannot output debayered imagery if the firmware is not "
                      "outputting bayered imagery.");
        }

        if (failed) {
            // Nothing has really happened thus far, so a "normal" exit is fine here.
            ROS_ERROR("One or more parameter errors. Exiting.");
            throw ConfigurationError();
        }
    }

     /* ==========================================================================
     Conversion and derived value methods

     Methods to get values derrived from node settings. */

    /** If debayered output is enabled.
     * This checks if the debayer output topic string is empty or not. */
    bool debayer_enabled() const {
        return output_topic_debayer.size() > 0;
    }

    /** Get the pixel format for the raw image firmware mode.*/
    enumGevPixelFormat get_firmware_pixel_format() const {
        switch (firmware_mode) {
            case FirmwareMode::Bayer:
                return enumGevPixelFormat::fMtBayerRG8;
            case FirmwareMode::Color:
                return enumGevPixelFormat::fmt_PFNC_YUV422_8;
            case FirmwareMode::Mono:
                return enumGevPixelFormat::fmtMono8; // default to mono8, not sure if this is the best
            case FirmwareMode::Mono8:
                return enumGevPixelFormat::fmtMono8;
            case FirmwareMode::Mono16:
                return enumGevPixelFormat::fmtMono16;
            default:
                throw std::invalid_argument("Invalid firmware mode set! [get_firmware_pixel_format()]");
        }
    }

    /** Raw image cv::Mat type based on set firmware mode. */
    int raw_image_cvmat_type() const {
        switch (firmware_mode) {
            case FirmwareMode::Bayer:
            case FirmwareMode::Mono:
            case FirmwareMode::Mono8:
                return CV_8UC1;
            case FirmwareMode::Color:
                return CV_8UC2;
            case FirmwareMode::Mono16:
                return CV_16UC1; // this is an experiment, may break later down in the pipeline
            default:
                throw std::invalid_argument("Invalid firmware mode set! [raw_image_cvmat_type]");
        }
    }

    /** Create a cv::Rect for cropping based on input image width and height.
     *
     * @param height Pixel height of the image to crop.
     * @param width Pixel width of the image to crop.
     * @roi Output cv::Rect to set the crop ROI to.
     */
    void makeRoi(int height, int width, cv::Rect &roi) {
        roi.x = 0;
        roi.y = 0;
        roi.width = width;
        roi.height = height;

        if (output_image_crop_top_row > 0) {
            roi.y = output_image_crop_top_row;
        }
        if (output_image_crop_bot_row > 0) {
            roi.height = output_image_crop_bot_row - roi.y;
        }
    }

    /**
     * Encoding string for cv_bridge::CvImage object.
     */
    const char *raw_image_cvimage_encoding() const {
        switch (firmware_mode) {
            case FirmwareMode::Bayer:
            case FirmwareMode::Mono:
            case FirmwareMode::Mono8:
                return "mono8";
            case FirmwareMode::Color:
                return "bgr8";
            case FirmwareMode::Mono16:
                ROS_WARN("Mono16 currently not fully tested [raw_image_cvimage_encoding]");
                return "mono16";
            default:
                throw std::invalid_argument("Invalid firmware mode set!");
        }
    }

    // ==========================================================================
    // Settings Hook methods
    //
    // Methods to set various options and settings objects based on this node's
    // input setting values.

    /** Set the library config options from current settings. */
    void set_library_config_options(GEVLIB_CONFIG_OPTIONS &opts) {
        // Currently nothing to set.
        // TODO: Add node parameters for options here, e.g. log level
    }

    /**
     * Set camera options structure values from current settings.
     *
     * @param[out] cam_opts Camera options structure to set values to.
     */
    void set_camera_options(GEV_CAMERA_OPTIONS &cam_opts) {
        /** Transferring values from example/previous driver.
         * The following states there is 32MB of onboard memory for acquisitions:
         * http://info.teledynedalsa.com/acton/attachment/14932/f-054e/1/-/-/l-0042/l-0042/Genie%20Nano%20Series%20User%20Manual.pdf
         */
        cam_opts.numRetries = 3;                           // default is 3
        cam_opts.heartbeat_timeout_ms = 3000;               // default is 10000
        cam_opts.streamNumFramesBuffered = 4;               // default is 4

        cam_opts.streamMemoryLimitMax = 1024 * 1024 * 32;   // default: 32MB
        cam_opts.streamFrame_timeout_ms = 1000;             // default is 1000
    }


    /** Set brightness */
    bool set_brightness_target(GenApi::CNodeMapRef *feature_node_map_ptr) {
        UINT16 genapi_exception_status(0);

        try {
            GenApi::CIntegerPtr int_node_ptr;

            // Auto-brightness target
            ROS_INFO1("-- \"autoBrightnessTarget\"");
            int_node_ptr = feature_node_map_ptr->_GetNode("autoBrightnessTarget");
            if (int_node_ptr.IsValid()) {
                ROS_INFO1("   Prev value: %d", (int) int_node_ptr->GetValue());
                // Range between 0-255
                int_node_ptr->SetValue(((int) xmlFeatures_autoBrightnessTarget) & 0xff);
                ROS_INFO1("   New value : %d", (int) int_node_ptr->GetValue());
            } else {
                ROS_WARN("   !! Feature \"autoBrightnessTarget\" not available.");
            }
        }
        CATCH_GENAPI_ERROR(genapi_exception_status);

        if (genapi_exception_status != 0) {
            ROS_ERROR_STREAM("Caught GenApi exception (status = " << genapi_exception_status << ")");
            return false;
        }
        return true;
    }

    /** Set feature values to the given GenICam FeatureNodeMap. */
    bool set_featuremap_options(GenApi::CNodeMapRef *feature_node_map_ptr) {
        ROS_INFO1("Setting GenApi::CNodeMapRef features:");

        // Global GenApi exception handling
        UINT16 genapi_exception_status(0);
        try {
            /** Set the GenICam camera parameters. See `docs/devices/genicam.rst` for more info
             * TriggerMode = { FreeRun, TriggeredFreeRun, TriggeredSequence, TriggeredPresetAdvance }
             * TriggerSelector not used by A6750
             * TriggerSource = { Internal, External, Software, IRIG }
             * IRFormat = { Radiometric, TemperatureLinear100mK, TemperatureLinear10mK }
             * FrameSyncSource = { Internal, External, Video }
             * */
            if (camType == "6750") {
                push_node_ptr(feature_node_map_ptr, "TriggerMode", "FreeRun");
                push_node_ptr(feature_node_map_ptr, "TriggerSource", triggerSource);

                push_node_ptr(feature_node_map_ptr, "FrameSyncSource", frameSyncSource);
                triggerNodeName = "TriggerSoftware";
            } else if (camType == "6xx") {
                /** Axx does not support any kind of hardware triggering
                 * but we can fake it with singe frame */
                push_node_ptr(feature_node_map_ptr, "AcquisitionMode", "SingleFrame");
                triggerNodeName = "AcquisitionStart";

            }
            /** Common params */
            push_node_ptr(feature_node_map_ptr, "IRFormat", irFormat);

        }
        CATCH_GENAPI_ERROR(genapi_exception_status);

        if (genapi_exception_status != 0) {
            ROS_ERROR_STREAM("Caught GenApi exception (status = " << genapi_exception_status << ")");
            return false;
        }
        return true;
    }

    // ==================================================================
    // ROS Subscriber callback methods.

    /** Accepts a message of expected type. */
    void update_autobrightness(const std_msgs::msg::UInt8 &msg) {
        xmlFeatures_autoBrightnessTarget = msg.data;
        ROS_INFO_STREAM("Received new brightness target: " << xmlFeatures_autoBrightnessTarget);


        G_NEW_BRIGHTNESS = true;                    // signal to main loop that new brightness value received.
    }

    /** Set listener callbacks for this node. */
    void set_callback(rclcpp::Node::SharedPtr nh) {
        // Setup listener for camera_brightness topic
        brightness_sub = nh->create_subscription<std_msgs::msg::UInt8>(
            "camera_brightness", 1,
            [this](const std_msgs::msg::UInt8::ConstSharedPtr msg) { update_autobrightness(*msg); });
        event_sub = nh->create_subscription<custom_msgs::msg::GsofEvt>(
            "/event", 5,
            [this](const custom_msgs::msg::GsofEvt::ConstSharedPtr msg) { eventCallback(msg); });
        shutdown_sub = nh->create_subscription<std_msgs::msg::Int8>(
            "/shutdown", 1,
            [](const std_msgs::msg::Int8::ConstSharedPtr msg) { cb_request_shutdown(*msg); });
        stat_pub = nh->create_publisher<custom_msgs::msg::Stat>("/stat", 5);
        errstat_pub = nh->create_publisher<custom_msgs::msg::Stat>("/errstat", 5);

    }


    void eventCallback (const custom_msgs::msg::GsofEvt::ConstSharedPtr& msg)
    {
        ROS_INFO("<^> eventCallback <>         %2.2f", rclcpp::Time(msg->gps_time).seconds());
        event_ = *msg;
        auto nodeName = std::string(node_->get_name());
        custom_msgs::msg::Stat stat_msg;
        std::stringstream link;
        stat_msg.header.stamp = node_->now();
        stat_msg.trace_header = (*msg).header;
        stat_msg.trace_topic = nodeName + "/eventCallback";
        stat_msg.node = nodeName;
        link << nodeName << "/event/" << event_.event_num; // link this trace to the event trace
        stat_msg.link = link.str();
        stat_pub->publish(stat_msg);
        trigger.fire_cond();
    }

}; // end NodeSettings   =.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=

class CamParamHandler {
public:
    CamParamHandler(rclcpp::Node::SharedPtr nhp, GEV_CAMERA_HANDLE camera_handle, const std::shared_ptr<NodeSettings> settings)
    :
    settings{settings},
    nhp_{nhp},
    camera_handle_{camera_handle} {
        genapi_ = std::make_shared<GenApiConnector>(camera_handle_);
        get_camera_attr_srv_ = nhp_->create_service<custom_msgs::srv::CamGetAttr>(
            "get_camera_attr",
            [this](const std::shared_ptr<custom_msgs::srv::CamGetAttr::Request> req,
                   std::shared_ptr<custom_msgs::srv::CamGetAttr::Response> rsp) { getCameraAttr(req, rsp); });
        set_camera_attr_srv_ = nhp_->create_service<custom_msgs::srv::CamSetAttr>(
            "set_camera_attr",
            [this](const std::shared_ptr<custom_msgs::srv::CamSetAttr::Request> req,
                   std::shared_ptr<custom_msgs::srv::CamSetAttr::Response> rsp) { setCameraAttr(req, rsp); });
        get_attr_list_srv_ = nhp_->create_service<custom_msgs::srv::StrList>(
            "get_attr_list",
            [this](const std::shared_ptr<custom_msgs::srv::StrList::Request> req,
                   std::shared_ptr<custom_msgs::srv::StrList::Response> rsp) { getAttrList(req, rsp); });
        nuc_srv_ = nhp_->create_service<custom_msgs::srv::CamSetAttr>(
            "nuc",
            [this](const std::shared_ptr<custom_msgs::srv::CamSetAttr::Request> req,
                   std::shared_ptr<custom_msgs::srv::CamSetAttr::Response> rsp) { nuc(req, rsp); });
    }

    void getCameraAttr(const std::shared_ptr<custom_msgs::srv::CamGetAttr::Request> req,
                       std::shared_ptr<custom_msgs::srv::CamGetAttr::Response> rsp) {
        ROS_INFO("<API> getCameraAttr(%s)", req->name.c_str());
        int feature_type = 0;
        try {
            genapi_->getCamAttr(req->name, rsp->value, &feature_type);
        }
        catch (std::exception &e) {
            rsp->value = "error:" + std::string(e.what());
        }
        ROS_INFO("<API ON GET> %s[%d]: %s.", req->name.c_str(), feature_type, rsp->value.c_str());
    }

    void setCameraAttr(const std::shared_ptr<custom_msgs::srv::CamSetAttr::Request> req,
                       std::shared_ptr<custom_msgs::srv::CamSetAttr::Response> rsp) {
        ROS_INFO("<API> setCameraAttr(%s)", req->name.c_str());
        ROS_INFO("<API> setCameraVal(%s)", req->value.c_str());
        if (! is_number(req->value) ) {
            std::string error = "Invalid value for attribute, must be a number.";
            ROS_ERROR("%s", error.c_str());
            rsp->value = error;
            return;
        }
        std::string tmp;
        int feature_type = 0;
        bool success = genapi_->getCamAttr(req->name, tmp, &feature_type);
        if (!success) {
            ROS_ERROR("Camera Attribute %s does not exist.", req->name.c_str());
            return;
        }
        ROS_INFO("<API BEFORE SET> %s[%d]: %s.", req->name.c_str(), feature_type, tmp.c_str());
        genapi_->setCamAttr(req->name, req->value, tmp);
        genapi_->getCamAttr(req->name, rsp->value, &feature_type);
        ROS_INFO("<API AFTER SET> %s[%d]: %s.", req->name.c_str(), feature_type, rsp->value.c_str());
    }

    void getAttrList(const std::shared_ptr<custom_msgs::srv::StrList::Request> req,
                     std::shared_ptr<custom_msgs::srv::StrList::Response> rsp) {
        (void) req;
        std::vector<std::string> paramList;
        get_attr_list(camera_handle_, paramList);
        if (!paramList.size()) {
            ROS_ERROR("failed to populate param list");
            return;
        }
        for (size_t i = 0; i < paramList.size(); i++) {
            rsp->values.push_back(paramList[i]);
        }
    }

    void nuc(const std::shared_ptr<custom_msgs::srv::CamSetAttr::Request> req,
             std::shared_ptr<custom_msgs::srv::CamSetAttr::Response> rsp) {
        (void) req;
        ROS_INFO("Initiating a camera NUC via service call.");
        rsp->value = tryNucCam(camera_handle_) ? "OK" : "ERROR";
    }
        private:
    const std::shared_ptr<NodeSettings> settings;
    rclcpp::Node::SharedPtr nhp_;
    GEV_CAMERA_HANDLE camera_handle_ = NULL;  // void* type
    std::shared_ptr<GenApiConnector> genapi_;

    rclcpp::Service<custom_msgs::srv::CamGetAttr>::SharedPtr get_camera_attr_srv_;
    rclcpp::Service<custom_msgs::srv::CamSetAttr>::SharedPtr set_camera_attr_srv_;
    rclcpp::Service<custom_msgs::srv::StrList>::SharedPtr    get_attr_list_srv_;
    rclcpp::Service<custom_msgs::srv::CamSetAttr>::SharedPtr nuc_srv_;
};

/**
 * */
class EventHandler {
public:
    EventHandler(rclcpp::Node::SharedPtr nhp, GEV_CAMERA_HANDLE camera_handle, CameraImageInfo cam_image_info,
                 const std::shared_ptr<NodeSettings> settings, std::shared_ptr<GenApiConnector> genapi) :
            camera_handle{camera_handle}, cam_image_info{cam_image_info},
            genapi_{genapi},
            settings{settings},
            nhp_{nhp},
            xport{nhp, settings->output_topic_raw} {
        postprocSub = nhp->create_subscription<sensor_msgs::msg::Image>(
            settings->output_topic_raw, 10,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) { postProcessImage(msg); });
        watchdog.setFailCallback([camera_handle]() {
            ROS_ERROR("Failed health check, attempting to safely shut down camera");
            safe_exit(13, camera_handle);
        });
        watchdog.pet(); // initial pet to give it a head start to avoid crib death.
        init();
    }


    /**
     * Since we expect a 1:1 relation between events and images, and the system goes sideways
     * if events aren't being received, we ought to be able to trigger off of the event itself
     * @param msg Event message received
     */
    void eventCallback(const custom_msgs::msg::GsofEvt::ConstSharedPtr& msg) {
        {
            std::lock_guard<std::mutex> lck(event_mutex);
            t_event_received_ = nhp_->now();
            event_ = *msg;
            event_cache.push_back(rclcpp::Time(msg->sys_time), msg);
            event_cache.purge();
            event_cache.show();
        }
        ROS_INFO("%2.4f <> eventCallback <> ", rclcpp::Time(msg->gps_time).seconds());
        processImages();

    }


    void fetchLoop() {
        ROS_INFO("fetch loop");
        while (running && rclcpp::ok() && !G_SIGINT_TRIGGERED) {
            fetchImage();
        }
    }

    void fetchImage() {
        GEV_BUFFER_OBJECT *img_buff_obj_ptr;                    // Also the same stuct as GEVBUF_ENTRY and GEVBUF_HEADER

        /// timeout is ms
        std::lock_guard<std::mutex> lck(buffer_mutex);

        GEV_STATUS call_status = GevWaitForNextImage(camera_handle, &img_buff_obj_ptr, 10000);
        rclcpp::Time t_image_received = nhp_->now(); // maybe

        auto nodeName = std::string(nhp_->get_name());
        custom_msgs::msg::Stat stat_msg;
        stat_msg.trace_header = std_msgs::msg::Header();
        stat_msg.trace_topic = nodeName + "/publishImage";
        stat_msg.node = nodeName;
        stat_msg.header.stamp = t_image_received;
        stat_msg.trace_header.stamp = t_image_received;

        if (call_status == GEV_FRAME_STATUS_TIMEOUT) {
            ///
            return;
        } else {
            WARN_ON_FAILURE(GevWaitForNextImage, call_status, GEVLIB_OK);
        }

        if (img_buff_obj_ptr) {
            t_image_received_ = t_image_received;
            ROS_INFO("%2.4f Got image", t_image_received.seconds());
        } else {
            ROS_WARN("null image pointer");
            return endOfTurn(img_buff_obj_ptr);
        }

        if (validate_image(img_buff_obj_ptr, &cam_image_info)) {

            rclcpp::Time camTime = CameraTimeSync::timestampToRos(img_buff_obj_ptr->timestamp_hi,
                                                                  img_buff_obj_ptr->timestamp_lo);
            ROS_INFO2("Received image with ID %d (%d x %d) ", img_buff_obj_ptr->id, img_buff_obj_ptr->w, img_buff_obj_ptr->h );

            // Check if we're NUCing, and flag as such
            int feature_type = 0;
            std::string rsp;
            try {
                genapi_->getCamAttr("CorrectionAutoInProgress", rsp, &feature_type);
            }
            catch (std::exception &e) {
                rsp = "error:" + std::string(e.what());
            }
            ROS_INFO("<DRIVER GET> %s[%d]: %s.", "CorrectionAutoInProgress", feature_type, rsp.c_str());


            std::string msg;
            msg.resize(128);
            snprintf(&msg[0], msg.size(), R"({"cam": %f, "recv": %f, "hi": %u, "lo": %u})",
                    camTime.seconds(), t_image_received.seconds(), img_buff_obj_ptr->timestamp_hi, img_buff_obj_ptr->timestamp_lo);
            std::cout << msg << "," << std::endl;
            {
                cv::Mat raw_image(img_buff_obj_ptr->h, img_buff_obj_ptr->w,
                                  settings->rawImageCVMatType,
                                  img_buff_obj_ptr->address);
                cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
                cv_ptr->encoding = settings->rawImageCVEncoding;
                cv_ptr->image = raw_image;
                cv_ptr->header.frame_id = "?nucing=" + std::string(rsp.c_str());

                /// IR-specific processing to remove in-band data field in the top row
                if (cv_ptr->image.rows % 2 == 1) {
                    cv::Rect roi;
                    roi.x = 0;
                    roi.y = 1;
                    roi.width = cv_ptr->image.cols;
                    roi.height = cv_ptr->image.rows - 1;
                    cv_ptr->image = cv_ptr->image(roi);
                    ROS_INFO("Cropped top row, new: (%d x %d)", cv_ptr->image.cols, cv_ptr->image.rows);
                }

                auto imgp = cv_ptr->toImageMsg();
                ROS_INFO("success at making image message");

                std::string link = "/" + nodeName + "/event/NA"; // link this trace to the event trace
                stat_msg.link = link;
                stat_msg.note = "success";

                image_map.emplace(t_image_received, imgp);
            }

            processImages();
        } else {
            stat_msg.note = "failure";
        }
        // Publish status msg, FPS is tracked from this in UI
        settings->stat_pub->publish(stat_msg);

        endOfTurn(img_buff_obj_ptr);
    }
    void start() {
        running = true;
        fetch_thread_ = std::thread([this]() { fetchLoop(); });
    }

    void stop() {
        running = false;
        if (fetch_thread_.joinable()) {
            fetch_thread_.join();
        }
    }

    void shutdown() {
        ROS_WARN("Shutting down the event handler");
        stop();
        event_sub.reset();
    }

    void processImages() {
        std::lock_guard<std::mutex> lck(event_mutex);
        rclcpp::Time img_key;
        bool hit = false;
        std_msgs::msg::Header gps_header;
        uint64_t event_num = 0;
        for ( const auto pair: image_map ) {
            // iterate through every image in the map to find
            // nearest event to image received
            bool success = event_cache.search(pair.first, gps_header, event_num);
            if (success) {
                // found a hit
                img_key = pair.first;
                hit = true;
                break;
            }
        }
        if (!hit) {
            return;
        }

        std::stringstream this_frame_id;
        this_frame_id << settings->frame_id;
        sensor_msgs::msg::Image::SharedPtr img = image_map[img_key];

        this_frame_id << "?lock=1&eventNum=" << event_num << "&eventTime"
                      << rclcpp::Time(gps_header.stamp).seconds() << img->header.frame_id ;
        ROS_INFO_STREAM("Timestamp: " << rclcpp::Time(gps_header.stamp).seconds());
        img->header = gps_header;
        img->header.frame_id = this_frame_id.str();
        ROS_INFO(" !!! Found matching %2.4f %2.4f !!! ", rclcpp::Time(gps_header.stamp).seconds(), img_key.seconds());
        /// remove the image so we don't get confused later
        image_map.erase(img_key);
        ROS_INFO("Remaining evt %u img %lu ", event_cache.size(), image_map.size());
        xport.it_raw_pub.publish(img);
        watchdog.pet();
    } // end process_img

    void postProcessImage(const sensor_msgs::msg::Image::SharedPtr &msg) {
        auto is_archiving = ArchiverHelper::get_is_archiving(settings->envoy_, "/sys/arch/is_archiving");
        if (!is_archiving) {
            return;
        }
        long int sec = msg->header.stamp.sec;
        long int nsec = msg->header.stamp.nanosec;
        std::string filename = ArchiverHelper::generateFilename(settings->envoy_, arch_opts_, sec, nsec);
        auto filename_written = dumpImageMessage(msg, filename);
        ROS_INFO("dumped %s", filename_written.c_str());
    }

    rclcpp::Time timeLastEventReceived() {
        rclcpp::Time last;
        {
            std::lock_guard<std::mutex> lck(event_mutex);
            last = t_event_received_;
        }
        return last;
    }
    rclcpp::Time timeLastImageReceived() {
        std::lock_guard<std::mutex> lck(image_mutex);
        return t_image_received_;
    }

    /// fields
    int rawImageCVMatType_;
    GEV_CAMERA_HANDLE camera_handle = NULL;  // void* type
    CameraImageInfo cam_image_info;
    rclcpp::Subscription<custom_msgs::msg::GsofEvt>::SharedPtr event_sub;
    std::shared_ptr<GenApiConnector> genapi_;


private:
    bool synced = false;
    std::atomic<bool> running{false};
    const std::shared_ptr<NodeSettings> settings;
    custom_msgs::msg::GsofEvt event_; /// todo: deprecated?
    cv::Mat raw_image_;
    EventCache event_cache;
    std::map<rclcpp::Time, sensor_msgs::msg::Image::SharedPtr> image_map;
    rclcpp::Time t_event_received_;
    rclcpp::Time t_image_received_;
    rclcpp::Node::SharedPtr nhp_;
    Transporter xport;
    std::mutex event_mutex;
    std::mutex image_mutex;
    std::mutex buffer_mutex;
    Watchdog watchdog;
    ArchiverOpts                        arch_opts_ = ArchiverOpts::from_env();
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr postprocSub;     // handle to subscriber
    std::thread fetch_thread_;    /// runs blocking image fetch loop

    void init() {
        static double min_image_delay = 0.05;
        event_cache.set_delay(min_image_delay);
        event_cache.set_tolerance(rclcpp::Duration::from_seconds(0.49));
    }
    void endOfTurn() {

        /// event/image messages are kept around for this amount of time, after that the expire
        /// which means they didn't get paired in the grace period
        rclcpp::Duration lookback_period{10, 0};
        rclcpp::Time old = nhp_->now() - lookback_period;
        int losses = 0;
        losses += purge_stale(image_map, old);
        for (auto i = 0; i < losses; i++ ) {
            watchdog.kick();
        }
        if ( event_cache.size() > (int) image_map.size() ) {
            int diff = event_cache.size() - image_map.size();
            for (int i = 0; i < diff; ++i) {
                watchdog.kick();
            }
        }
        watchdog.check();
    }
    void endOfTurn(GEV_BUFFER_OBJECT *img_buff_obj_ptr) {
        GevReleaseImage(camera_handle, img_buff_obj_ptr);
        endOfTurn();
    }

}; /// end EventHandler


/// === === === === === === === === MAIN === === === === === === === ===
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
    /** Setup signal handler to kick out of run loop for a clean shutdown.
 -- Initializing before GEV stuff in order to be sure we don't interrupt
    communication to the hardware in case something goes wrong. */
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // ROS Node initialization + params
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("a6750_driver_node");

    // ROS/GEV Input Parameters
    std::shared_ptr<NodeSettings> node_settings;
    try {
        node_settings = std::make_shared<NodeSettings>(node);
    } catch (ConfigurationError const &) {
        return 1;
    }

    CameraImageInfo cam_image_info;
    GEV_CAMERA_HANDLE camera_handle = NULL;  // void* type


    node_settings->set_callback(node);

    // Initialize GEV API
    {
        GEV_STATUS s(GevApiInitialize());           // GEV_STATUS is a UINT16 type
        RETURN_ON_FAILURE(GevApiInitialize, s, GEVLIB_OK, 1, NULL);
    }

     /** Set default options for GEV library
      * These options apply globally to the operation of the GigE-V Framework API
      * library within the current application. */
    {
        GEVLIB_CONFIG_OPTIONS options;

        GevGetLibraryConfigOptions(&options);

        // Pass through option values from NodeSettings as appropriate.
        node_settings->set_library_config_options(options);
        log_config_options("-- ", options);

        GEV_STATUS s(GevSetLibraryConfigOptions( &options ));
        RETURN_ON_FAILURE(GevSetLibraryConfigOptions, s, GEVLIB_OK, 1, NULL);
    }

    /** Acquire camera handle. If we fail to do so, log available interfaces and exit. */
    try {
        // Open camera by IP address, either already specified, or located in startup
        gev_open_by_ip_addr_wrapper(node_settings->uint_cam_ip_addr, camera_handle);

        // Log information of camera we just connected to.
        {
            ROS_GREEN("SUCCESS! Connected to camera:");
            GEV_CAMERA_INFO *ci = GevGetCameraInfo(camera_handle);
            log_camera_interface(*ci, "-- ");
        }


        GEV_CAMERA_OPTIONS cam_opts;                             // Camera interface options.
        {
            GEV_STATUS s(GevGetCameraInterfaceOptions(camera_handle, &cam_opts));
            RETURN_ON_FAILURE(GevGetCameraInterfaceOptions,
                              s, GEVLIB_OK, 1, camera_handle);
        }

        node_settings->set_camera_options(cam_opts);
        ROS_INFO1("Setting camera options:");
        log_camera_options("-- ", cam_opts);
        {
            GEV_STATUS s(GevSetCameraInterfaceOptions(camera_handle, &cam_opts));
            RETURN_ON_FAILURE(GevSetCameraInterfaceOptions,
                              s, GEVLIB_OK, 1, camera_handle);
        }
    }
    catch (std::invalid_argument const &) { return safe_exit(1, camera_handle); }
    catch (CameraConnectionError const &) { return safe_exit(1, camera_handle); }
    catch (CameraInUseError const &) { return safe_exit(1, camera_handle); }

    /** Set up feature access using the XML retrieved from the camera. */
    GenApi::CNodeMapRef *cam_node_map_ptr = NULL;
    auto genapi = std::make_shared<GenApiConnector>(camera_handle);
    genapi->initNodeMap();
    std::string tmp;
    if (genapi->getVal("TriggerMode", tmp)) {
        ROS_INFO("got: %s", tmp.c_str());
    }

    {
        if (node_settings->xmlFeatures_filepath.size() > 0) {
            ROS_INFO_STREAM("Attempting setting XML params from file: "
                                    << node_settings->xmlFeatures_filepath);

            const char *c_filepath_str = node_settings->xmlFeatures_filepath.c_str();
            GEV_STATUS s(GevInitGenICamXMLFeatures_FromFile( camera_handle, (char*) c_filepath_str));
            RETURN_ON_FAILURE(GevInitGenICamXMLFeatures_FromFile,
                              s, GEVLIB_OK, 1, camera_handle);
        } else {
            // true flags saving the XML to disk in the directory:
            //     "$GIGEV_XML_DOWNLOAD/xml/download/"
            ROS_GREEN("Loading XML from camera");
            GEV_STATUS s(GevInitGenICamXMLFeatures(camera_handle, true));
            RETURN_ON_FAILURE(GevInitGenICamXMLFeatures,
                              s, GEVLIB_OK, 1, camera_handle);
        }

        // Set explicit feature values based on node settings via hook.
        cam_node_map_ptr = static_cast< GenApi::CNodeMapRef * >(
                GevGetFeatureNodeMap(camera_handle)
        );
        if (!node_settings->set_featuremap_options(cam_node_map_ptr)) {
            return safe_exit(1, camera_handle);
        }
        tellFeatureValue(camera_handle, "TriggerMode");
        get_node_val(cam_node_map_ptr, "TriggerMode");
        tellFeatureValue(camera_handle, "GevTimestampTickFrequency");
        tellFeatureValue(camera_handle, "FlagState");
        tellFeatureValue(camera_handle, "CorrectionAutoEnabled");
        tellFeatureValue(camera_handle, "CorrectionAutoUseDeltaTemp");
        tellFeatureValue(camera_handle, "CorrectionAutoUseDeltaTime");
        tellFeatureValue(camera_handle, "CorrectionAutoDeltaTemp");
        tellFeatureValue(camera_handle, "CorrectionAutoDeltaTime");

    } // cam pointer stuff


    {
        ROS_INFO1("Getting camera image output metadata parameters...");
        GEV_STATUS s(GevGetImageParameters(camera_handle,
                                           &cam_image_info.width, &cam_image_info.height,
                                           &cam_image_info.x_offset, &cam_image_info.y_offset,
                                           &cam_image_info.pixel_format));
        RETURN_ON_FAILURE(GevGetImageParameters,
                          s, GEVLIB_OK, 1, camera_handle);

         /** Set the pixel format for the camera to output based on the configured
         * firmware mode.
         * NOTE: An error status of GEVLIB_ERROR_ACCESS_DENIED means that we
         *       attempted to set a pixel format that is not allowed with the
         *       actual firmware on the camera. This means that the incorrect
         *       firmware mode was set in this driver's configuration. */
        cam_image_info.pixel_format = node_settings->get_firmware_pixel_format();
        ROS_INFO1("Setting pixel format based on firmware: %s",
                 decode_pixel_format(cam_image_info.pixel_format));
        s = GevSetImageParameters(camera_handle,
                                  cam_image_info.width, cam_image_info.height,
                                  cam_image_info.x_offset, cam_image_info.y_offset,
                                  cam_image_info.pixel_format);
        RETURN_ON_FAILURE(GevSetImageParameters,
                          s, GEVLIB_OK, 1, camera_handle);

        ROS_INFO1("Camera output image:");
        ROS_INFO1("--        width: %d", cam_image_info.width);
        ROS_INFO1("--       height: %d", cam_image_info.height);
        ROS_INFO1("--     x_offset: %d", cam_image_info.x_offset);
        ROS_INFO1("--     y_offset: %d", cam_image_info.y_offset);
        ROS_INFO1("-- pixel_format: %s", decode_pixel_format(cam_image_info.pixel_format));
        ROS_INFO1("                 (depth: %d )", cam_image_info.depth());
    }

    // Initialize and start image transfer
    PUINT8 image_buffer_array[node_settings->imageTransfer_numImageBuffers];
    {
        // Allocate memory for each image buffer, 0`ed initial values.
        UINT32 buffer_size =
                cam_image_info.width * cam_image_info.height * cam_image_info.depth();
        ROS_INFO1("Allocating %d image buffers if size %d (%d x %d x %d)",
                 node_settings->imageTransfer_numImageBuffers, buffer_size,
                 cam_image_info.width, cam_image_info.height,
                 cam_image_info.depth());
        for (unsigned int i = 0; i < node_settings->imageTransfer_numImageBuffers; ++i) {
            ROS_INFO1("-- buffer %d", i);
            image_buffer_array[i] = (PUINT8) calloc(buffer_size, sizeof(UINT8));
        }

        // Initialize transfer with allocated buffer space.
        GEV_STATUS s(GevInitImageTransfer(camera_handle,
                                          SynchronousNextEmpty,
                                          node_settings->imageTransfer_numImageBuffers,
                                          image_buffer_array));
        RETURN_ON_FAILURE(GevInitImageTransfer,
                          s, GEVLIB_OK, 1, camera_handle);

        /** Start the image transfer.
         * -- "-1" signifies continuous transfer.
         * -- Alternative is to start transfer at the top of every loop step in
         *    order to acquire a single frame. */
        s = GevStartImageTransfer(camera_handle, -1);
        RETURN_ON_FAILURE(GevStartImageTransfer,
                          s, GEVLIB_OK, 1, camera_handle);
    }

    // Command node reference to manually trigger image acquisition.
    trigger.bind_node_action(cam_node_map_ptr, node_settings->triggerNodeName.c_str());

    EventHandler handler{node, camera_handle, cam_image_info, node_settings, genapi};
    CamParamHandler paramHandler{node, camera_handle, node_settings};
    handler.event_sub = node->create_subscription<custom_msgs::msg::GsofEvt>(
        "/event", 5,
        [&handler](const custom_msgs::msg::GsofEvt::ConstSharedPtr msg) { handler.eventCallback(msg); });

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
    executor.add_node(node);
    handler.start();

    auto terminator = node->create_wall_timer(std::chrono::milliseconds(250), [&]() {
        if (G_SIGINT_TRIGGERED) {
            ROS_WARN("shutting down spinner");
            handler.shutdown();
            rclcpp::shutdown();
        }
    });
    executor.spin();
    handler.shutdown();
    return safe_exit(0, camera_handle);
}
