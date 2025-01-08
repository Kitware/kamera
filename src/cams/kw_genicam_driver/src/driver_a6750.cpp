#include <chrono>
#include <csignal>
#include <cstring>
#include <cstdio>
#include <string>
#include <mutex>
#include <deque>
#include <atomic>

// ROS stuff
#include <ros/ros.h>
#include "std_msgs/UInt8.h"
#include "std_msgs/Int8.h"

#include <boost/filesystem.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <roskv/envoy.h>
#include <roskv/archiver.h>

// Includes from: /opt/genicam_v3_0/library/CPP/include/
#include <GenApi/GenApi.h>
// Includes from: /usr/dalsa/GigeV/include/
#include <gevapi.h>

#include <custom_msgs/GSOF_EVT.h>
#include <custom_msgs/Stat.h>
#include <custom_msgs/CamGetAttr.h>
#include <custom_msgs/CamSetAttr.h>
#include <custom_msgs/StrList.h>
#include <phase_one/phase_one_utils.h>


#include "utils.h"
#include "macros.h"
#include "decode_error.h"
#include "spec_a6750.h"

using namespace std::chrono;

enum FirmwareMode { Bayer=0, Color=1, Mono=2, Mono8=3, Mono16=4};

/// Forward declarations

void cb_request_shutdown(std_msgs::Int8 const &msg) {
    ROS_INFO("Requesting clean shutdown: %d", msg.data);
    ros::requestShutdown();
}

class Transporter {
public:
    Transporter(ros::NodeHandlePtr nhp, const std::string &out_topic) :
            nhp_{nhp},
            it_raw(*nhp),
            it_raw_pub(it_raw.advertise(out_topic, 1)) {}

    image_transport::ImageTransport it_raw;
    image_transport::Publisher it_raw_pub;
private:
    ros::NodeHandlePtr nhp_;
};

typedef std::pair<ros::Time, cv::Mat> TimeMat;


int purge_stale(std::map<ros::Time, sensor_msgs::ImagePtr> &image_map, ros::Time t)
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


std::string dumpImageMessage(const sensor_msgs::ImagePtr & received_image, const std::string &filename)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    cv_bridge::CvImagePtr cvPtr;
    auto start_db = ros::Time::now();
    cvPtr = cv_bridge::toCvCopy(received_image, received_image->encoding);
    ROS_INFO("debayered %ld in %2.3f ", (long int)(cvPtr->image.total() * cvPtr->image.elemSize()), (ros::Time::now()-start_db).toSec());
    cv::Mat undist = cvPtr->image;
    auto now = ros::WallTime::now();
    auto nodeName = ros::this_node::getName();

    start_db = ros::Time::now();

    boost::filesystem::path path_filename{filename};
    boost::filesystem::create_directories(path_filename.parent_path());
    cv::imwrite(filename, cvPtr->image, compression_params);
//    cv::imwrite(filename, cvPtr->image);
    ROS_INFO("dumped   %ld in %2.3f ", (long int)(cvPtr->image.total() * cvPtr->image.elemSize()), (ros::Time::now()-start_db).toSec());
    start_db = ros::Time::now();
//    auto filename2 = std::string("/mnt/ram/miketest/driverdump/") + nodeName + "/" + std::to_string(now.toSec()) + ".jpg";
//    cv::imwrite(filename2, cvPtr->image, compression_params);
//    auto out = cv::imread(filename, cv::IMREAD_UNCHANGED);
//    ROS_INFO("read     %ld in %2.3f ", (long int)(out.total() * out.elemSize()), (ros::Time::now()-start_db).toSec());
    return filename;

//    return cvPtr->toImageMsg();
}


class CameraTimeSync {
public:
    static ros::Time timestampToRos(uint32_t timehi, uint32_t timelo) {
        return timestampToRos(timehi, timelo, 1000000000);
    }
    static ros::Time timestampToRos(uint32_t timehi, uint32_t timelo, uint32_t freq) {
        uint64_t utime = ((uint64_t )timehi) << 32;
        utime = utime + (uint64_t )timelo;
        double dtime = double (utime) / (double) freq;
//        double dtime = (double)(((uint64_t )timehi) << 32);
        return ros::Time{dtime};
    }
};

class DriverHandlerOpts {
public:
    DriverHandlerOpts(int rawImageCVMatType)
    :
    rawImageCVMatType{rawImageCVMatType}
    {}
    int rawImageCVMatType;
    std::string output_topic_raw;
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


/** Rescale the image pixel values according to a specified temperature range.
 * Will set the output range approximately such that
 * minTempK ~= min(dst) = -N-sigma and maxTempK ~= max(dst) = +N-sigma
 * Note: the A6750 in mono16/temp10mK mode has min/max values of 0/65535, even though the average is ~30000.
 * This prevents naive approach to min-max-scaling. We want the lower / bounds on the histogram (+/- N-sigma).
 *  # in progress! #
 *
 * @param src       - input matrix
 * @param dest      - destination matrix
 * @param minTempK  - expected min temperature, in Kelvin
 * @param maxTempK  - expected max temperature, in Kelvin
 * @param rtype     - output matrix data type
 * @param nSigma     - lower/upper threshold.
 */
void rerange_temp(cv::Mat src, cv::Mat dest, double minTempK=273, double maxTempK=303, int rtype=CV_16UC1,
                    double nSigma=3.0) {
    double imax;
    if (rtype == CV_16UC1) {
        imax = 65535;
    } else if (rtype == CV_8UC1) {
        imax = 255;
    } else {
        ROS_WARN("Unsupported matrix data type. Falling back to mono8");
        imax = 255;
        rtype = CV_8UC1;
    }
    double mean, minVal, maxVal;
    double countRange = (maxTempK - minTempK) / 0.01; // adjust for 10mK/count
    double scaler = 4; //imax / countRange / 2.0f;
    cv::Scalar tmp;
    cv::minMaxLoc(src, &minVal, &maxVal);
    tmp = cv::mean(src);
    mean = tmp.val[0];
    ROS_WARN("Input Min/Max/Mean: %.1f %.1f %.1f", minVal, maxVal, mean);
    cv::Scalar tmpx = cv::mean(src);
    mean = (float) tmpx.val[0];
    cv::Mat srcD(src.rows, src.cols, CV_32FC1);
    src.convertTo(srcD, CV_32FC1);
//    srcD -= mean/2.0;
    srcD *= 0.01; // Convert to Kelvin
    srcD -= 273.15; // Convert to C
    cv::minMaxLoc(srcD, &minVal, &maxVal);
    tmp = cv::mean(srcD);
    mean = tmp.val[0];
    ROS_WARN("Celcius Min/Max/Mean: %.1f %.1f %.1f", minVal, maxVal, mean);
    srcD.convertTo(dest, CV_16UC1, 1500.); // rescale into 16 bit range
    cv::minMaxLoc(dest, &minVal, &maxVal);
    tmp = cv::mean(dest);
    mean = tmp.val[0];
    ROS_WARN("Output Min/Max/Mean: %.1f %.1f %.1f", minVal, maxVal, mean);
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
    ros::Subscriber brightness_sub;             // handle to subscriber
    ros::Subscriber event_sub;                  // handle to gps events
    ros::Subscriber shutdown_sub;                  // sub for clean shutdown
    ros::Publisher  stat_pub;                  // handle to tracing stat
    ros::Publisher  errstat_pub;                  // handle to error tracing
    std::string output_topic_raw;               // Name of topic to output raw image to.
    float output_frame_rate;                    // Output frame-rate (hz).
    std::string output_topic_debayer;           // Topic to output debayered image to.
    /**   If output_topic_debayer is an empty string, debayering does not occur. */

    std::string rawImageCVEncoding;
    int rawImageCVMatType;

    std_msgs::Header last_published_; // Last header which was successfully published
    custom_msgs::GSOF_EVT event_;     // store the last received event

    /**
     * Construct settings from a node handle (usually the private handle).
     *
     * \throws ConfigurationError Failed extracting setting values appropriately.
     */
    NodeSettings(ros::NodeHandle const &nh) {
        bool failed(false);
        int tmp_int;

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

        nh.param("camera_username", camera_id.username, std::string());
        nh.param("camera_manufacturer", camera_id.manufacturer, std::string());
        nh.param("camera_ip_addr", camera_id.ip_addr, std::string());
        nh.param("camera_serial", camera_id.serial, std::string());
        nh.param("camera_mac", camera_id.mac, std::string());

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
        nh.param("xmlFeatures_filepath", xmlFeatures_filepath, std::string());

        nh.param("xmlFeatures_autoBrightness", xmlFeatures_autoBrightness, false);
        nh.param("xmlFeatures_autoBrightnessTarget", tmp_int, 128);
        xmlFeatures_autoBrightnessTarget = tmp_int;
        // 0 - Off, 1 - On Demand, 2 - Periodic
        nh.param("xmlFeatures_BalanceWhiteAuto", xmlFeatures_BalanceWhiteAuto, 0);

        imageTransfer_numImageBuffers = static_cast<UINT32>( parse_pos_int(nh, "imageTransfer_numImageBuffers", failed) );
        nextImage_timeout = static_cast<UINT32>( parse_pos_int(nh, "nextImage_timeout", failed) );
        output_frame_rate = parse_pos_float(nh, "output_frame_rate", failed);

        nh.param("output_image_crop_top_row", output_image_crop_top_row, -1);
        nh.param("output_image_crop_bot_row", output_image_crop_bot_row, -1);
        if (output_image_crop_bot_row >= 0 &&
            output_image_crop_bot_row <= output_image_crop_top_row) {
            failed = true;
            ROS_ERROR("Bottom crop row must be greater than top crop row (%d !> %d).",
                      output_image_crop_bot_row, output_image_crop_top_row);
        }

        if (!nh.getParam("frame_id", frame_id)) {
            failed = true;
            ROS_ERROR("No frame ID provided!");
        }
        if (!nh.getParam("output_topic_raw", output_topic_raw)) {
            failed = true;
            ROS_ERROR("No output topic string provided");
        }
        // Debayer output is optional
        // - Debayering is undefined if the firmware is not set to bayer mode.
        nh.param("output_topic_debayer", output_topic_debayer, std::string());
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
    cv::Rect makeRoi(int height, int width, cv::Rect &roi) {
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
        /**
         typedef struct
         {                                      // Defaults:
               UINT32 numRetries;               // 3
               UINT32 command_timeout_ms;       // 2000
               UINT32 heartbeat_timeout_ms;     // 10000
               UINT32 streamPktSize;            // algorithmic
               UINT32 streamPktDelay;           // 0
               UINT32 streamNumFramesBuffered;  // 4
               UINT32 streamMemoryLimitMax;     // ???
               UINT32 streamMaxPacketResends;   // 100
               UINT32 streamFrame_timeout_ms;   // 1000
               INT32  streamThreadAffinity;     // -1
               INT32  serverThreadAffinity;     // -1
               UINT32 msgChannel_timeout_ms;    // 1000
         } GEV_CAMERA_OPTIONS, *PGEV_CAMERA_OPTIONS;
         */

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
            GenApi::CFloatPtr float_node_ptr;
            GenApi::CIntegerPtr int_node_ptr;
            GenApi::CEnumerationPtr enum_node_ptr;

            /** Auto-brightness, AB target, and AWB not currently used */
//            push_node_ptr(feature_node_map_ptr, "autoBrightnessMode", (int) xmlFeatures_autoBrightness);
//            push_node_ptr(feature_node_map_ptr, "autoBrightnessTarget", ((int) xmlFeatures_autoBrightnessTarget) & 0xff);
//            push_node_ptr(feature_node_map_ptr, "BalanceWhiteAuto", (int) xmlFeatures_BalanceWhiteAuto);
//            push_node_ptr(feature_node_map_ptr, "autoBrightnessAlgoConvergenceTime", (float) 15.0);


            /** Set the GenICam camera parameters. See `docs/devices/genicam.rst` for more info
             * TriggerMode = { FreeRun, TriggeredFreeRun, TriggeredSequence, TriggeredPresetAdvance }
             * TriggerSelector not used by A6750
             * TriggerSource = { Internal, External, Software, IRIG }
             * IRFormat = { Radiometric, TemperatureLinear100mK, TemperatureLinear10mK }
             * FrameSyncSource = { Internal, External, Video }
             * */
            if (camType == "6750") {
                push_node_ptr(feature_node_map_ptr, "TriggerMode", "FreeRun");
//            push_node_ptr(feature_node_map_ptr, "TriggerSelector", "FrameStart");
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
    void update_autobrightness(const std_msgs::UInt8 &msg) {
        xmlFeatures_autoBrightnessTarget = msg.data;
        ROS_INFO_STREAM("Received new brightness target: " << xmlFeatures_autoBrightnessTarget);


        G_NEW_BRIGHTNESS = true;                    // signal to main loop that new brightness value received.
    }

    /** Set listener callbacks for this node. */
    void set_callback(ros::NodeHandle &nh) {
        // Setup listener for camera_brightness topic
        brightness_sub  = nh.subscribe(std::string("camera_brightness"), 1, &NodeSettings::update_autobrightness, this);
        event_sub       = nh.subscribe(std::string("/event"), 5, &NodeSettings::eventCallback, this);
        shutdown_sub    = nh.subscribe("/shutdown", 1, cb_request_shutdown);
        stat_pub        = nh.advertise<custom_msgs::Stat>(std::string("/stat"), 5);
        errstat_pub        = nh.advertise<custom_msgs::Stat>(std::string("/errstat"), 5);

    }


    void eventCallback (const custom_msgs::GSOF_EVTConstPtr& msg)
    {
        ROS_INFO("<^> eventCallback <>         %2.2f", msg->gps_time.toSec());
        event_ = *msg;
        auto nodeName = ros::this_node::getName();
        custom_msgs::Stat stat_msg;
        std::stringstream link;
        stat_msg.header.stamp = ros::Time::now();
        stat_msg.trace_header = (*msg).header;
        stat_msg.trace_topic = nodeName + "/eventCallback";
        stat_msg.node = nodeName;
        link << nodeName << "/event/" << event_.header.seq; // link this trace to the event trace
        stat_msg.link = link.str();
        stat_pub.publish(stat_msg);
        trigger.fire_cond();
    }

}; // end NodeSettings   =.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=.=

class CamParamHandler {
public:
    CamParamHandler(ros::NodeHandlePtr nhp, GEV_CAMERA_HANDLE camera_handle, const boost::shared_ptr<NodeSettings> settings)
    :
    nhp_{nhp},
    camera_handle_{camera_handle},
    settings{settings} {
        genapi_ = std::make_shared<GenApiConnector>(camera_handle_);
        get_camera_attr_srv_ = nhp_->advertiseService("get_camera_attr", &CamParamHandler::getCameraAttr, this);
        set_camera_attr_srv_ = nhp_->advertiseService("set_camera_attr", &CamParamHandler::setCameraAttr, this);
        get_attr_list_srv_   = nhp->advertiseService("get_attr_list", &CamParamHandler::getAttrList, this);

        nuc_srv_             = nhp_->advertiseService("nuc", &CamParamHandler::nuc, this);
    }

    bool getCameraAttr(custom_msgs::CamGetAttrRequest &req, custom_msgs::CamGetAttrResponse &rsp) {
        ROS_INFO("<API> getCameraAttr(%s)", req.name.c_str());
        bool stat{false};
        int feature_type;
        try {
            stat = genapi_->getCamAttr(req.name, rsp.value, &feature_type);
        }
        catch (std::exception &e) {
            rsp.value = "error:" + std::string(e.what());
        }
        ROS_INFO("<API ON GET> %s[%d]: %s.", req.name.c_str(), feature_type, rsp.value.c_str());
        return stat;
    }

    bool setCameraAttr(custom_msgs::CamSetAttrRequest &req, custom_msgs::CamSetAttrResponse &rsp) {
        ROS_INFO("<API> setCameraAttr(%s)", req.name.c_str());
        ROS_INFO("<API> setCameraVal(%s)", req.value.c_str());
        if (! is_number(req.value) ) {
            std::string error = "Invalid value for attribute, must be a number.";
            ROS_ERROR(error.c_str());
            rsp.value = error;
            return false;
        }
	    std::string tmp;
	    int feature_type;
	    bool success = genapi_->getCamAttr(req.name, tmp, &feature_type);
	    if (!success) {
	        ROS_ERROR("Camera Attribute %s does not exist.", req.name.c_str());
	        return false;
	    }
	    ROS_INFO("<API BEFORE SET> %s[%d]: %s.", req.name.c_str(), feature_type, tmp.c_str());
	    bool stat = genapi_->setCamAttr(req.name, req.value, tmp);
	    genapi_->getCamAttr(req.name, rsp.value, &feature_type);
	    ROS_INFO("<API AFTER SET> %s[%d]: %s.", req.name.c_str(), feature_type, rsp.value.c_str());
	    return stat;
    }

    bool getAttrList(custom_msgs::StrListRequest &req, custom_msgs::StrListResponse &rsp) {
        std::vector<std::string> paramList;
        bool status = get_attr_list(camera_handle_, paramList);
        if (!paramList.size()) {
            ROS_ERROR("failed to populate param list");
            return false;
        }
        for (int i = 0; i < paramList.size(); i++) {
            rsp.values.push_back(paramList[i]);
        }
        return true;
    }

   bool nuc(custom_msgs::CamSetAttrRequest &req, custom_msgs::CamSetAttrResponse &rsp) {
//        return genapi_->tryNuc();
        /* This always seems to report 1, so can't use as a method
        std::string tmp;
        int feature_type;
        genapi_->getCamAttr("CorrectionAutoInProgress", tmp, &feature_type);
        int status = std::stoi(tmp);
        if ( status != 1 ) {
            ROS_ERROR("Correction already in progress, aborting.");
	    rsp.value = "ERROR";
            return false;
        }*/
        ROS_INFO("Initiating a camera NUC via service call.");
        rsp.value = tryNucCam(camera_handle_);
        return true;
    }
        private:
    const boost::shared_ptr<NodeSettings> settings;
    ros::NodeHandlePtr nhp_;
    GEV_CAMERA_HANDLE camera_handle_ = NULL;  // void* type
    std::shared_ptr<GenApiConnector> genapi_;

    ros::ServiceServer                  get_camera_attr_srv_;
    ros::ServiceServer                  set_camera_attr_srv_;
    ros::ServiceServer                  get_attr_list_srv_;
    ros::ServiceServer                  nuc_srv_;
};

/**
 * */
class EventHandler {
public:
    EventHandler(ros::NodeHandlePtr nhp, GEV_CAMERA_HANDLE camera_handle, CameraImageInfo cam_image_info,
                 const boost::shared_ptr<NodeSettings> settings) :
            nhp_{nhp},
            camera_handle{camera_handle}, cam_image_info{cam_image_info},
            settings{settings},
            xport{nhp, settings->output_topic_raw} {
        postprocSub = nhp->subscribe(settings->output_topic_raw, 10,
                                     &EventHandler::postProcessImage, this, ros::TransportHints().reliable());
        watchdog.setFailCallback([camera_handle](ros::TimerEvent const &e) {
            ROS_ERROR("Failed health check, attempting to safely shut down camera");
            safe_exit(13, camera_handle);
        });
        watchdog.pet(); // initial pet to give it a head start to avoid crib death.
    }


    /**
     * Since we expect a 1:1 relation between events and images, and the system goes sideways
     * if events aren't being received, we ought to be able to trigger off of the event itself
     * @param msg Event message received
     */
    void eventCallback(const custom_msgs::GSOF_EVTConstPtr& msg) {
        {
            std::lock_guard<std::mutex> lck(event_mutex);
            t_event_received_ = ros::Time::now();
            event_ = *msg;
            event_cache.push_back(msg->sys_time, msg);
            event_cache.purge();
            event_cache.show();
        }
        ROS_INFO("%2.4f <> eventCallback <> ", msg->gps_time.toSec());
//        if (synced) {
//            fetchImage(ros::Time{});
//        }
//        bool ok = checkSync();
        processImages();

    }


    void fetchImageRecurrent(const ros::TimerEvent &e) {
        ROS_INFO("fetch loop");
        fetchImage(ros::Time{});
//        checkSync();
        if (running && ros::ok()) {
            fetchTimer_ = nhp_->createTimer(ros::Duration(0.001),
                                            &EventHandler::fetchImageRecurrent, this, true, true);
        }

    }
    void fetchImage(const ros::Time &eventTime) {
        GEV_BUFFER_OBJECT *img_buff_obj_ptr;                    // Also the same stuct as GEVBUF_ENTRY and GEVBUF_HEADER        ROS_INFO("<> eventCallback <>         %2.2f", msg->header.stamp.toSec());

        /// timeout is ms
        std::lock_guard<std::mutex> lck(buffer_mutex);

        GEV_STATUS call_status = GevWaitForNextImage(camera_handle, &img_buff_obj_ptr, 10000);
        ros::Time t_image_received = ros::Time::now(); // maybe

    	auto nodeName = ros::this_node::getName();
    	custom_msgs::Stat stat_msg;
    	stat_msg.header.stamp = ros::Time::now();
    	stat_msg.trace_header = std_msgs::Header();
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
            ROS_INFO("%2.4f Got image", t_image_received.toSec());
        } else {
            ROS_WARN("null image pointer");
            return endOfTurn(img_buff_obj_ptr);
        }

        if (validate_image(img_buff_obj_ptr, &cam_image_info)) {

//            ROS_INFO("image good");
            ros::Time camTime = CameraTimeSync::timestampToRos(img_buff_obj_ptr->timestamp_hi,
                                                               img_buff_obj_ptr->timestamp_lo);
            ROS_INFO2("Received image with ID %d (%d x %d) ", img_buff_obj_ptr->id, img_buff_obj_ptr->w, img_buff_obj_ptr->h );


            std::string msg;
            msg.resize(128);
            sprintf(&msg[0], R"({"cam": %f, "recv": %f, "hi": %u, "lo": %u})",
                    camTime.toSec(), t_image_received.toSec(), img_buff_obj_ptr->timestamp_hi, img_buff_obj_ptr->timestamp_lo);
            std::cout << msg << "," << std::endl;
            {
                cv::Mat raw_image(img_buff_obj_ptr->h, img_buff_obj_ptr->w,
                                  settings->rawImageCVMatType,
                                  img_buff_obj_ptr->address);
                cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
//        cv_ptr->header.frame_id = this_frame_id.str();
                cv_ptr->encoding = settings->rawImageCVEncoding;
                cv_ptr->image = raw_image;

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

                std::cout << "!> cv_ptr: type: " << settings->rawImageCVMatType << cv_ptr->header << std::endl;
                std::cout << "!> img: dims:" << raw_image.dims << " size: "<<  raw_image.size
                          << " step: "<<  raw_image.step << " rows: "<<  raw_image.rows << " cols: "<<  raw_image.cols
                          << " type: "<<  raw_image.type() << std::endl;
                auto imgp = cv_ptr->toImageMsg();
                ROS_INFO("success at making image message");

                std::string link = "/" + nodeName + "/event/NA"; // link this trace to the event trace
                stat_msg.trace_header.seq = 0;
                stat_msg.link = link;
                stat_msg.note = "success";

                image_map.emplace(t_image_received, imgp);
            }
//            cv::Mat raw_image(img_buff_obj_ptr->h, img_buff_obj_ptr->w,
//                              opts.rawImageCVMatType,
//                              img_buff_obj_ptr->address);

            processImages();
        } else {
            stat_msg.note = "failure";
	    }
        // Publish status msg, FPS is tracked from this in UI
        settings->stat_pub.publish(stat_msg);

        endOfTurn(img_buff_obj_ptr);
    }
    void start() {
        running = true;
        fetchTimer_ = nhp_->createTimer(ros::Duration(0.001),
                                        &EventHandler::fetchImageRecurrent, this, true, true);
    }

    void stop() {
        running = false;
    }

    void shutdown() {
        ROS_WARN("Shutting down the event handler");
        stop();
        event_sub.shutdown();
    }

    void processImages() {
//        image_map.
//        auto test = cam_image_info;
        std::lock_guard<std::mutex> lck(event_mutex);
        ros::Time evt_key;
        ros::Time img_key;
        bool hit = false;
        std_msgs::Header gps_header;
        for ( const auto pair: image_map ) {
            // iterate through every image in the map to find
            // nearest event to image received
            bool success = event_cache.search(pair.first, gps_header);
            if (success == 1) {
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
        sensor_msgs::ImagePtr img = image_map[img_key];

        this_frame_id << "?lock=1&eventNum=" << gps_header.seq << "&eventTime" << gps_header.stamp.toSec() ;
        ROS_INFO_STREAM("Timestamp: " << gps_header.stamp.toSec());
        img->header = gps_header;
        img->header.frame_id = this_frame_id.str();
        ROS_INFO(" !!! Found matching %2.4f %2.4f !!! ", gps_header.stamp.toSec(), img_key.toSec());
        /// remove the image so we don't get confused later
        image_map.erase(img_key);
        ROS_INFO("Remaining evt %u img %lu ", event_cache.size(), image_map.size());
        xport.it_raw_pub.publish(img);
        watchdog.pet();
    } // end process_img

    void postProcessImage(const sensor_msgs::ImagePtr &msg) {
        auto is_archiving = ArchiverHelper::get_is_archiving(settings->envoy_, "/sys/arch/is_archiving");
        if (!is_archiving) {
            return;
        }
        long int sec = msg->header.stamp.sec;
        long int nsec = msg->header.stamp.nsec;
        std::string filename = ArchiverHelper::generateFilename(settings->envoy_, arch_opts_, sec, nsec);
        auto filename_written = dumpImageMessage(msg, filename);
        ROS_INFO("dumped #%d %s",msg->header.seq, filename_written.c_str());
    }

    ros::Time timeLastEventReceived() {
        ros::Time last;
        {
            std::lock_guard<std::mutex> lck(event_mutex);
//            last = event_.header.stamp();
            last = t_event_received_;
        }
        return last;
    }
    ros::Time timeLastImageReceived() {
        std::lock_guard<std::mutex> lck(image_mutex);
        return t_image_received_;
    }

    /// fields
    int rawImageCVMatType_;
    GEV_CAMERA_HANDLE camera_handle = NULL;  // void* type
    CameraImageInfo cam_image_info;
    ros::Subscriber event_sub;


private:
    bool synced = false;
    bool running = false;
    const boost::shared_ptr<NodeSettings> settings;
    custom_msgs::GSOF_EVT event_; /// todo: deprecated?
    cv::Mat raw_image_;
    EventCache event_cache;
    std::map<ros::Time, sensor_msgs::ImagePtr> image_map;
    ros::Time t_event_received_;
    ros::Time t_image_received_;
    ros::NodeHandlePtr nhp_;
    Transporter xport;
    std::mutex event_mutex;
    std::mutex image_mutex;
    std::mutex buffer_mutex;
    Watchdog watchdog;
    ArchiverOpts                        arch_opts_ = ArchiverOpts::from_env();
    ros::Subscriber postprocSub;             // handle to subscriber
    ros::Timer fetchTimer_; /// kicks off image fetch async events

    //    GEV_BUFFER_OBJECT *img_buff_obj_ptr;                    // Also the same stuct as GEVBUF_ENTRY and GEVBUF_HEADER
    void init() {
        static double min_image_delay = 0.05;
        event_cache.set_delay(min_image_delay);
        event_cache.set_tolerance(ros::Duration(0.49));
    }
    void endOfTurn() {

        /// event/image messages are kept around for this amount of time, after that the expire
        /// which means they didn't get paired in the grace period
        ros::Duration lookback_period{10.0};
        ros::Time old = ros::Time::now() - lookback_period;
        int losses = 0;
        losses += purge_stale(image_map, old);
        for (auto i = 0; i < losses; i++ ) {
            watchdog.kick();
        }
        if ( event_cache.size() > image_map.size() ) {
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
    communication to the hardware in case something goes wrong.
 -- We are not calling ros::shutdown() here so as to keep logging
    functional throughout the exit process.
     update: it seems with the new async stuff, roslaunch just isn't propagating signals properly
     */
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // ROS Node initialization + params
    ros::init(argc, argv, "a6750_driver_node");
    ros::NodeHandle nh,
            nhp("~");  // for parameters

    // ROS/GEV Input Parameters
    boost::shared_ptr<NodeSettings> node_settings = boost::make_shared<NodeSettings>(nhp);

    CameraImageInfo cam_image_info;
    GEV_CAMERA_HANDLE camera_handle = NULL;  // void* type


    node_settings->set_callback(nh);

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
        // Report        ROS_INFO1("Setting library config options:");
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
//            ROS_INFO("Connected to camera:");
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

//    tryBootstrap(camera_handle);

    /** Set up feature access using the XML retrieved from the camera. */
    GenApi::CNodeMapRef *cam_node_map_ptr = NULL;
    cam_node_map_ptr = static_cast< GenApi::CNodeMapRef * >(
            GevGetFeatureNodeMap(camera_handle)
    );
    std::shared_ptr<GenApi::CNodeMapRef> nodemap(new GenApi::CNodeMapRef);
//    cam_node_map_sptr = std::make_shared<GenApi::CNodeMapRef>(*cam_node_map_ptr);
    {

    }
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
            // TODO: Optionalize output of XML features file
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
//        tryBootstrap(camera_handle);
        tellFeatureValue(camera_handle, "GevTimestampTickFrequency");
        tellFeatureValue(camera_handle, "FlagState");
        std::string tmp;
        //genapi->setCamAttr("CorrectionAutoEnabled", "1", tmp);
        tellFeatureValue(camera_handle, "CorrectionAutoEnabled");
        tellFeatureValue(camera_handle, "CorrectionAutoUseDeltaTemp");
        tellFeatureValue(camera_handle, "CorrectionAutoUseDeltaTime");
        tellFeatureValue(camera_handle, "CorrectionAutoDeltaTemp");
        tellFeatureValue(camera_handle, "CorrectionAutoDeltaTime");

        //tellfeaturevalue(camera_handle, "correctionautoinprogress");
        //trynuc(cam_node_map_ptr);
        //tellfeaturevalue(camera_handle, "correctionautoinprogress");


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
        for (int i = 0; i < node_settings->imageTransfer_numImageBuffers; ++i) {
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

    /** ROS broadcast publishers */
    ROS_INFO1("Creating ROS image-transport publisher for raw image output...");
    image_transport::ImageTransport it_raw(nh);
    image_transport::Publisher it_raw_pub(
            it_raw.advertise(node_settings->output_topic_raw, 1)
    );
    ros::Publisher  missed_frames_pub_(
            nh.advertise<std_msgs::Header>("/missed_frames", 3)
    );

    image_transport::ImageTransport it_db(nh);
    image_transport::Publisher it_db_pub;
    if (node_settings->debayer_enabled()) {
        ROS_WARN("Creating ROS image-transport publisher for debayered image output...");
        it_db_pub = it_db.advertise(node_settings->output_topic_debayer, 1);
    }

    /** ROS Broadcast loop */
    ROS_INFO1("Starting run-loop");
    GEV_STATUS call_status(0);
    cv::Rect roi;

    // Command node reference to manually trigger image acquisition.
//    GenApi::CCommandPtr trigger_cmd_node_ptr = cam_node_map_ptr->_GetNode("TriggerSoftware");
    trigger.bind_node_action(cam_node_map_ptr, node_settings->triggerNodeName.c_str());
    GEV_BUFFER_OBJECT *img_buff_obj_ptr;                    // Also the same stuct as GEVBUF_ENTRY and GEVBUF_HEADER
    ros::Rate ros_rate(node_settings->output_frame_rate);    // Acquisition Rate
    int counter = 0;
    bool report_timings = false;                            // Enable print timing info
    // Loop component timers
    double deltaT;                                          // elapsed time
    double lpLoopTime = 1.0;                                // low-passed loop time
    double lpGamma  = 0.1;                                  // low pass coeff
    system_clock::time_point t_trigger_start, t_trigger_end, t_wait_return, t_after_error_checks,
            t_make_raw_image, t_yuv_bgr_conversion, t_raw_crop, t_published_raw_img, t_published_bayer_img,
            t_image_buf_released, t_ros_spin_once, t_brightness_update;

    unsigned int frames_dropped_total_ = 0;

    auto nodeName = ros::this_node::getName();
    custom_msgs::Stat stat_msg;
    stat_msg.header.stamp = ros::Time::now();
    stat_msg.trace_header = std_msgs::Header();
    stat_msg.trace_topic = nodeName + "/publishImage";
    stat_msg.node = nodeName;

    /** ======================================= alt loop ========================================================== */
    if (false) {
        ROS_WARN("exiting early for the sake of cam param testing (optional)");
        return safe_exit(0, camera_handle);
    }
    if (true) {
//    DriverHandlerOpts dopts{node_settings->rawImageCVMatType};
        /// !!! doing the nodehandleptr thing because I don't really know how to solve it
        ros::NodeHandlePtr nhptr = ros::NodeHandlePtr(new ros::NodeHandle);
        EventHandler handler{nhptr, camera_handle, cam_image_info, node_settings};
        CamParamHandler paramHandler{nhptr, camera_handle, node_settings};
//    handler.syncUp(nh, 0.0);
        handler.event_sub = nh.subscribe(std::string("/event"), 5, &EventHandler::eventCallback, &handler);
        ros::AsyncSpinner spinner{3};
        spinner.start();
        handler.start();

        auto terminator = nh.createTimer(ros::Duration(0.25), [&](const ros::TimerEvent &e) {
            if (G_SIGINT_TRIGGERED) {
                ROS_WARN("shutting down spinner");
                spinner.stop();
                handler.shutdown();
            }
        }, false, true);
        ros::waitForShutdown();
//    while (ros::ok() && !G_SIGINT_TRIGGERED) {
//    }
        return safe_exit(0, camera_handle);
    }

    /** ======================================= main loop ========================================================== */
    while (ros::ok() && !G_SIGINT_TRIGGERED) {
        std::stringstream link;
        std::stringstream this_frame_id;

        // Log message delineation
        ROS_INFO3("===============================================");
        img_buff_obj_ptr = NULL;                // Reset loop variables
        report_timings = false;
        ros::Time t_pre_trigger(ros::Time::now());          // Record image timestamp as time of acquisition request
        stat_msg.header.stamp = ros::Time::now();
        stat_msg.trace_header.seq = counter;

        t_trigger_start = system_clock::now();
        trigger.spin_until_trigger();
//        trigger.fire_cond(node_settings->triggeredBySoftware); // fires off software trigger, assuming that feature exists
        t_trigger_end = system_clock::now();

        // Get the next image
        ROS_INFO2("Waiting for next image...");
        // IR is not remotely rate-limiting, so we will spin a lot to ensure event is captured
        ros::spinOnce(); // allow most recent event to be received
        call_status = GevWaitForNextImage(camera_handle, &img_buff_obj_ptr,
                                          node_settings->nextImage_timeout);
        t_wait_return = system_clock::now();
        ros::Time t_image_received = ros::Time::now();
        stat_msg.trace_header.stamp = t_image_received;
        ROS_WARN("before");
        WARN_ON_FAILURE(GevWaitForNextImage, call_status, GEVLIB_OK);
        ROS_WARN("after");
        ros::spinOnce(); // allow most recent event to be received

        // bind image to most recent
        std_msgs::Header gps_header;
        bool success = false;



        if (validate_image(img_buff_obj_ptr, &cam_image_info)) {
            t_after_error_checks = system_clock::now();

            /** Successful image buffer retrieval. */
            ROS_INFO2("Received image with ID %d", img_buff_obj_ptr->id);
            cv::Mat raw_image(img_buff_obj_ptr->h, img_buff_obj_ptr->w,
                              node_settings->rawImageCVMatType,
                              img_buff_obj_ptr->address);
            t_make_raw_image = system_clock::now();
            t_yuv_bgr_conversion = system_clock::now();

            /** Crop image to configured output dims */
            node_settings->makeRoi(img_buff_obj_ptr->h, img_buff_obj_ptr->w, roi);
            ROS_INFO2("Cropping to ROI %dx%d+%d+%d", roi.width, roi.height, roi.x, roi.y);
            if (roi.area() == 0) {
                ROS_ERROR("Crop ROI has 0 area with raw image height of %d pixels.",
                          img_buff_obj_ptr->h);
                return safe_exit(1, camera_handle);
            }
            raw_image = raw_image(roi);
            t_raw_crop = system_clock::now();

            if (it_raw_pub.getNumSubscribers() > 0) {
                ros::spinOnce(); // allow most recent event to be received
                ROS_INFO2("Publishing raw image...");
                cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
                // Set the image timestamp to match the event that actually triggered it
                if (success) {
                    cv_ptr->header = gps_header;
                }
//                cv_ptr->header.frame_id = this_frame_id.str();
                cv_ptr->encoding = node_settings->rawImageCVEncoding;
                cv_ptr->image = raw_image;
                link << nodeName << "/event/" << node_settings->event_.header.seq; // link this trace to the event trace
                stat_msg.link = link.str();
                stat_msg.note = "success";

                std::cout << "!> cv_ptr: type: " << node_settings->rawImageCVMatType << cv_ptr->header << std::endl;
                std::cout << "!> img: dims:" << raw_image.dims << " size: "<<  raw_image.size
                          << " step: "<<  raw_image.step << " rows: "<<  raw_image.rows << " cols: "<<  raw_image.cols
                          << " type: "<<  raw_image.type() << std::endl;


                ROS_WARN("<D> evt: %.2f  %.2f", cv_ptr->header.stamp.toSec(), node_settings->event_.gps_time.toSec());
                it_raw_pub.publish(cv_ptr->toImageMsg());
            }
            t_published_raw_img = system_clock::now();

            t_published_bayer_img = system_clock::now();
            report_timings = true;          // All time_points have been successfully populated.
        } else {
            ROS_WARN("taking the nasty path, ptr: %p", img_buff_obj_ptr);
            std_msgs::Header msg = std_msgs::Header(node_settings->event_.header);
            std::stringstream status_int, status_name, note;

            if (img_buff_obj_ptr) {
                status_int <<     img_buff_obj_ptr->status;
                status_name << decode_sdk_status(img_buff_obj_ptr->status);
            } else {
                this_frame_id << "&status=" << 999 << "&error="
                              << "nil pointer in buffer";
            }
            this_frame_id << "&status=" << status_int.str() << "&error=" << status_name.str();
            note << "status " << status_int.str() << ": " << status_name.str();
            link << nodeName << "/event/" << node_settings->event_.header.seq; // link this trace to the event trace
            stat_msg.link = link.str();
            stat_msg.note = note.str();
            msg.seq = ++frames_dropped_total_;
            msg.frame_id = this_frame_id.str();
            ROS_WARN("about to publish");
            missed_frames_pub_.publish(msg);
            node_settings->errstat_pub.publish(stat_msg);
        }
        node_settings->stat_pub.publish(stat_msg);


        // Release used image buffer back to GEV acquisition process.
        GevReleaseImage(camera_handle, img_buff_obj_ptr);
        t_image_buf_released = system_clock::now();

        // Allow messages to be received
        ros::spinOnce();
        t_ros_spin_once = system_clock::now();

        // Check for updated brightness setting
        if (G_NEW_BRIGHTNESS) {
            // activate new brightness target
            ROS_INFO2("Setting new brightness value");
            if (!node_settings->set_brightness_target(cam_node_map_ptr)) {
                return safe_exit(1, camera_handle);
            }
            G_NEW_BRIGHTNESS = false;
        }

        t_brightness_update = system_clock::now();

        if (report_timings) {
            ROS_INFO2("Loop Timings:");
            ROS_INFO2("-- Before trigger -> After trigger     : %f ms",
                     milliseconds_between(t_trigger_start, t_trigger_end));
            ROS_INFO2("-- After trigger  -> Wait return       : %f ms (+%f ms)",
                     milliseconds_between(t_trigger_start, t_wait_return),
                     milliseconds_between(t_trigger_end, t_wait_return));
            if (G_TIMING_VERBOSE) {
                ROS_INFO2("-- Wait return    -> After checks      : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_after_error_checks),
                         milliseconds_between(t_wait_return, t_after_error_checks));
                ROS_INFO2("-- After checks   -> Make raw image    : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_make_raw_image),
                         milliseconds_between(t_after_error_checks, t_make_raw_image));
                ROS_INFO2("-- Make raw image -> YUV2BGR cvt       : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_yuv_bgr_conversion),
                         milliseconds_between(t_make_raw_image, t_yuv_bgr_conversion));
                ROS_INFO2("-- YUV2BGR cvt    -> Crop Image        : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_raw_crop),
                         milliseconds_between(t_yuv_bgr_conversion, t_raw_crop));
                ROS_INFO2("-- Crop Image     -> Publish raw       : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_published_raw_img),
                         milliseconds_between(t_raw_crop, t_published_raw_img));
                ROS_INFO2("-- Publish raw    -> Publish debay     : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_published_bayer_img),
                         milliseconds_between(t_published_raw_img, t_published_bayer_img));
                ROS_INFO2("-- Publish debay  -> Release image     : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_image_buf_released),
                         milliseconds_between(t_published_bayer_img, t_image_buf_released));
                ROS_INFO2("-- Release image  -> ROS Spin Once     : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_ros_spin_once),
                         milliseconds_between(t_image_buf_released, t_ros_spin_once));
                ROS_INFO2("-- ROS Spin Once  -> Update brightness : %f ms (+%f ms)",
                         milliseconds_between(t_trigger_start, t_brightness_update),
                         milliseconds_between(t_ros_spin_once, t_brightness_update));

            }
            deltaT = milliseconds_between(t_trigger_start, t_brightness_update);
            lpLoopTime = (lpLoopTime * (1-lpGamma)) + (deltaT * lpGamma);
            ROS_INFO2("-- Overall Cycle Time (rolling avg)    : %f ms (%f ms)", deltaT, lpLoopTime);

        }

        counter++;
        ros_rate.sleep();
    }

    return safe_exit(0, camera_handle);
}
