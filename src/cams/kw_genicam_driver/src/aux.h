// aux.h
/* Auxiliary code for genicam driver files */
#ifndef KW_GENICAM_DRIVER_AUX_H
#define KW_GENICAM_DRIVER_AUX_H

#include <cstdio>
#include <atomic>
#include <chrono>
#include <mutex>
#include <ros/ros.h>
// Local installs
#include <GenApi/GenApi.h>
#include <gevapi.h>

#ifndef _GEVAPI_H_
/// stupid shims for clion IDE
#include "../../../../../../../github.com/zcpp/DALSA/GigeV/include/gev_linux.h"
#include "../../../../../../../github.com/zcpp/DALSA/GigeV/include/gevapi.h"
#include "../../../../../../../github.com/ros/rc_genicam_api/genicam/library/CPP/include/GenApi/GenApi.h"
#include "../../../../src/core/roskv/include/roskv/envoy.h"

#endif

#include "decode_error.h"
#include "spec_a6750.h"
#include "macros.h"

/** Notes on gevapi.h since it's tricky to access:
 * typedef void* GEV_CAMERA_HANDLE;
 * GenApi::CNodeMapRef is GENAPI_NAMESPACE::CNodeMapRef
 * https://docs.ros.org/en/api/rc_genicam_api/html/classGENAPI__NAMESPACE_1_1CNodeMapRef.html
 */

using namespace std::chrono;

extern uint8_t G_INFO_VERBOSITY;

enum GeniAttributes { GA_MANUFACTURER, GA_MODEL, GA_SERIAL, GA_USERNAME, GA_MAC };

void cb_fail_shutdown(ros::TimerEvent e);

// ----------------------------------------------------------------------------
/** Structure containing the camera image output metadata. */
struct CameraImageInfo {
    UINT32 width, height, x_offset, y_offset, pixel_format;

    /**
     * Get the channel depth based on pixel format set.
     * Undefined if the pixel_format parameter.
     *
     * @returns Depth (channels) of the image.
     */
    UINT32 depth() {
        return GetPixelSizeInBytes(pixel_format);
    }
};

typedef struct {
    std::string manufacturer;
    std::string model;
    std::string serial;      // The serial number of the camera to connect to.
    std::string version;
    std::string username;    // The interface->username of the camera to connect to.
    std::string ip_addr;     // The IP address of the camera to connect to.
    std::string mac;         // The mac address (as raw string with colons) to connect to
} CameraIdentifier;

/// === === ===  new improved interface to GenApi - now with RAII!
class GenApiConnector {
public:
    GenApiConnector(GEV_CAMERA_HANDLE camera_handle);
    void assertCamHandle();
    void assertMapRef();
    int initNodeMap();

    bool getType(std::string const &param, std::string &type);
    bool getVal(std::string const &param, std::string &val);
    bool getCamAttr(std::string const &param, std::string &val, int *feature_type);
    bool setCamAttr(std::string const &param, std::string const &val, std::string &out);
    bool tryNuc();

private:
    GEV_CAMERA_HANDLE cam_handle_; /// aka void*, ick.
    GenApi::CNodeMapRef *node_map_;
};

class Watchdog {
public:
    Watchdog();
    Watchdog(double lookback_period, double health_threshold);


    void push_back(ros::Time const &t, double val);

    int size();

    void show();

    void purge();

    /// good puppy. positive health event
    void pet();
    /// bad puppy. negative health event
    void kick();
    /// check health status, and trigger any conditions
    void check();

    double computeHealth();

    void setFailCallback(ros::TimerCallback callback);
    void callFail();


private:

    /// lookback period in seconds  to consider events
    ros::Duration lookback_period_{30};
    /// 0 = balanced 50/50
    double health_threshold{0.0};
    std::mutex mutex_;
    std::vector<std::pair<ros::Time, double>> array;
    ros::TimerCallback failCallback_;

};

// ============== Function prototypes
double milliseconds_between(system_clock::time_point const &t1, system_clock::time_point const &t2);
std::string enum2str(int en, std::map<std::string, int> mymap, std::string errmsg);
std::string enum2str(int en, std::map<std::string, int> mymap);
std::string mac2str(uint32_t LOW, uint32_t HIGH);
std::string parse_validate_arg(ros::NodeHandle const &nh, std::string param, std::map<std::string, int> validmap, bool &failed);
int parse_pos_int(ros::NodeHandle const &nh, std::string param, bool &failed);
int parse_pos_int(ros::NodeHandle const &nh, std::string param, bool &failed, int minval);
float parse_pos_float(ros::NodeHandle const &nh, std::string param, bool &failed);
int arg2enum(std::string arg, std::map<std::string, int> validmap);
bool is_number(std::string entry);

UINT32 ip_string_to_uint(std::string camera_ip_addr);
uint32_t locate_camera(CameraIdentifier &camera);
void print_camera_id(CameraIdentifier &camera);

int safe_exit(int exit_code, GEV_CAMERA_HANDLE cam_handle);
uint32_t find_camera_by_attribute(std::string filter, GeniAttributes e_attr);

void log_config_options(std::string line_prefix, GEVLIB_CONFIG_OPTIONS const &options);
void log_camera_options(std::string line_prefix, GEV_CAMERA_OPTIONS const &opts);
void log_camera_interface(GEV_DEVICE_INTERFACE const &itf, std::string line_prefix);
void log_camera_interfaces(GEV_DEVICE_INTERFACE const *interfaces, int const &num_interfaces);
void discover_and_log_camera_interfaces();
void gev_open_by_serial_wrapper(std::string serial, GEV_CAMERA_HANDLE &cam_handle);
void gev_open_by_ip_addr_wrapper(UINT32 ip_addr, GEV_CAMERA_HANDLE &cam_handle);

bool push_node_ptr(GenApi::CNodeMapRef *feature_node_map_ptr, std::string param, std::string value);
bool push_node_ptr(GenApi::CNodeMapRef *feature_node_map_ptr, std::string param, int);
bool push_node_ptr(GenApi::CNodeMapRef *feature_node_map_ptr, std::string param, float);
bool get_node_val(GenApi::CNodeMapRef *feature_node_map_ptr, std::string param);

bool say_attr_info(GenApi::CNodeMapRef *feature_node_map_ptr, std::string param);
bool get_attr_list(GEV_CAMERA_HANDLE &camera_handle, std::vector<std::string> &paramList);

bool validate_image(GEV_BUFFER_OBJECT *img_buff_obj_ptr, CameraImageInfo *cam_image_info);

/// More GEV stuff

bool getFeatureValueAsString(GEV_CAMERA_HANDLE &handle, const std::string &name, std::string &out, int *feature_type);
bool getFeatureValueAsString(GEV_CAMERA_HANDLE &handle, const std::string &name, std::string &out);
bool tellFeatureValue(GEV_CAMERA_HANDLE &handle, const std::string &name);
bool tryExecute(GenApi::CNodeMapRef *feature_node_map_ptr, const std::string &name);
bool tryNucCam(GEV_CAMERA_HANDLE &handle);
bool tryNuc(GenApi::CNodeMapRef *feature_node_map_ptr);
bool tryBootstrap(GEV_CAMERA_HANDLE &handle);

// Set up factory and Function pointer for eventual currying, which binds GEV call to trigger_activate

void trigger_activate();

class Trigger {
private:
    GenApi::CCommandPtr trigger_cmd_node_ptr;
    bool setToFire = false;
    std::atomic<bool> ready = {false};
public:
    Trigger();
    void set_fire_state(bool state);
    void fire();
    void fire_cond();
    void fire_cond(bool condition);
    void bind_node_action(GenApi::CNodeMapRef *cam_node_map_ptr, const char* trigger_node_name);
    void spin_until_trigger();
};


// ----------------------------------------------------------------------------
// Exceptions used in this driver.

/**
 * Exception for a bad configuration.
 */
class ConfigurationError
        : public std::exception {
public:
    ConfigurationError() = default;

    virtual ~ConfigurationError() = default;

    virtual char const *what() const noexcept {
        return "Bad configuration";
    }
};


/**
 * Exception for an error during camera connection.
 */
class CameraConnectionError
        : public std::exception {
public:
    CameraConnectionError() = default;

    virtual ~CameraConnectionError() = default;

    virtual char const *what() const noexcept {
        return "Failed to connect to camera.";
    }
};

/**
 * Exception for an error during camera connection.
 */
class CameraInUseError
        : public std::exception {
public:
    CameraInUseError() = default;

    virtual ~CameraInUseError() = default;

    virtual char const *what() const noexcept {
        return "Failed to connect to camera.";
    }
};

class CameraLocateError
        : public std::exception {
public:
    CameraLocateError() = default;

    virtual ~CameraLocateError() = default;

    virtual char const *what() const noexcept {
        return "Failed to locate a specific camera on network. Either not found, or multiple matching cameras found";
    }
};

#define ROS_BLUE(mystr) ROS_INFO(BLU mystr NC)
#define ROS_GREEN(mystr) ROS_INFO(GRN mystr NC)
#define ROS_INFO1(...) if (G_INFO_VERBOSITY >= 1) {ROS_INFO(__VA_ARGS__);}
#define ROS_INFO2(...) if (G_INFO_VERBOSITY >= 2) {ROS_INFO(__VA_ARGS__);}
#define ROS_INFO3(...) if (G_INFO_VERBOSITY >= 3) {ROS_INFO(__VA_ARGS__);}
#define ROS_WARN1(...) if (G_INFO_VERBOSITY >= 1) {ROS_WARN(__VA_ARGS__);}



#endif //KW_GENICAM_DRIVER_AUX_H
