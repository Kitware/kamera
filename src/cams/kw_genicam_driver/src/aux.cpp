// aux.cpp
#include <cstdio>
#include <chrono>
#include <mutex>
#include <ros/ros.h>
// Local installs
#include <GenApi/GenApi.h>
#include <gevapi.h>

#include "decode_error.h"
#include "spec_a6750.h"
#include "aux.h"
#include "macros.h"

using std::string;
using std::map;


void cb_fail_shutdown(ros::TimerEvent e) {
    ROS_WARN("Shutting down due to unhealthy");
    ros::requestShutdown();
}

/// === === ===  new improved interface to GenApi - now with RAII!

GenApiConnector::GenApiConnector(GEV_CAMERA_HANDLE camera_handle)
: cam_handle_{camera_handle} {
    this->assertCamHandle();
}
void GenApiConnector::assertCamHandle() {
    if (cam_handle_ == nullptr) {
        throw std::runtime_error("Unable to initialize - camera handle is null");
    }
}
void GenApiConnector::assertMapRef() {
    if (node_map_ == nullptr) {
        throw std::runtime_error("Unable to initialize - camera handle is null");
    }
}
int GenApiConnector::initNodeMap() {
    ROS_BLUE("Loading XML from camera");
    GEV_STATUS s(GevInitGenICamXMLFeatures(cam_handle_, true));
    RETURN_ON_FAILURE(GevInitGenICamXMLFeatures,
                      s, GEVLIB_OK, 1, cam_handle_);
    node_map_ = static_cast< GenApi::CNodeMapRef * >(
            GevGetFeatureNodeMap(cam_handle_)
    );
    this->assertMapRef();
    ROS_GREEN("NodeMap OK ");
    return 0;
}

/// todo: wip
bool GenApiConnector::getType(std::string const &feature_name, std::string &type) {
    this->assertMapRef();
    GenApi::CNodePtr node = node_map_->_GetNode(feature_name.c_str());
//    auto node = node_map_->_GetNode(feature_name.c_str());

    if (!node) {
        ROS_WARN("   !! Feature \"%s\" not available.", feature_name.c_str());
        return false;
    }
    GenApi::EInterfaceType node_type = node->GetPrincipalInterfaceType();
    switch (node_type) {
        // The ICommand pattern.
        case GenApi::intfICommand:
            type = "Command";
            break;
        case GenApi::intfIInteger:
            type = "Integer";
            break;
        case GenApi::intfIBoolean:
            type = "Boolean";
            break;
        case GenApi::intfIFloat:
            type = "Float";
            break;
        case GenApi::intfIString:
            type = "String";
            break;
        case GenApi::intfIEnumeration:
            type = "Enumeration";
            break;
    }
    ROS_INFO("%s: [%d] - %s", feature_name.c_str(), node_type, type.c_str());
    return true;
}


bool GenApiConnector::getVal(std::string const &param, std::string &val) {
    this->assertMapRef();
    GenApi::CEnumerationPtr node_ptr;
    std::string value;
    node_ptr = node_map_->_GetNode(param.c_str());
    if (node_ptr.IsValid()) {
        ROS_INFO1("<API> %s: %s", param.c_str(), (**node_ptr).c_str());
    } else {
        ROS_WARN("   !! Feature \"%s\" not available.", param.c_str());
    }
    val = std::string(**node_ptr);
}


bool GenApiConnector::getCamAttr(std::string const &name, std::string &out, int *feature_type) {
/// GEVLIB_ERROR_NO_SPACE
    size_t sz = 32;
    GEV_STATUS status;
    while (true) {
        out.resize(sz);
        status = GevGetFeatureValueAsString(cam_handle_, name.c_str(), feature_type, (int) out.size(), &out[0]);
        if (status == GEVLIB_OK) {
            return true;
        } else if (status == GEVLIB_ERROR_NO_SPACE) {
            sz *= 2;
        } else {
            WARN_ON_FAILURE(GevGetFeatureValueAsString, status, GEVLIB_OK);
            return false;
        }
        if (sz > 65537) {
            ROS_ERROR("Tried to allocate too much memory for feature %s", name.c_str());
            return false;
        }
    }
}

bool GenApiConnector::setCamAttr(std::string const &feature_name,
	       		    	 std::string const &val, std::string &out) {
    GEV_STATUS stat = GevSetFeatureValueAsString(cam_handle_, feature_name.c_str(), val.c_str());
    RETURN_ON_FAILURE(GevSetFeatureValueAsString, stat, GEVLIB_OK, 1, cam_handle_);
    if (stat == GEVLIB_OK) {
        return true;
    }
    return false;
}


bool GenApiConnector::tryNuc() {
    const std::string name = "CorrectionAutoPerform";
    return tryExecute(node_map_, name);
}

Watchdog::Watchdog() : failCallback_{cb_fail_shutdown} {}
Watchdog::Watchdog(double lookback_period, double health_threshold)
:
    lookback_period_{lookback_period},
    health_threshold{health_threshold},
    failCallback_{cb_fail_shutdown} {}


void Watchdog::push_back(ros::Time const &t, double val) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!t.isValid()) {
        ROS_ERROR("zero/invalid time encountered in Watchdog::push_back()");
        return;
    }

    array.emplace_back(std::pair<ros::Time, double>(t, val));
    }

int Watchdog::size() {
    std::lock_guard<std::mutex> guard(mutex_);
    return (int) (array.size());
}

void Watchdog::show() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto pair : array) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    std::cout << "\n---" << std::endl;
}

void Watchdog::purge() {
    std::lock_guard<std::mutex> guard(mutex_);
    auto now = ros::Time::now();
    std::size_t index = 0;
    for (const auto pair : array) {
        auto age = now - pair.first ;
        if (age > lookback_period_) {
            auto it = array.begin() + index;
            array.erase(it);
        }
        index++;
    }
}

void Watchdog::pet() {
    ROS_INFO("Pet the watchdog");
    this->push_back(ros::Time::now(), 1.0);
}

void Watchdog::kick() {
    ROS_WARN("kicked the watchdog");
    this->push_back(ros::Time::now(), -1.0);
}

void Watchdog::check() {
    this->purge();
    double health = computeHealth();
    ROS_INFO("current health: %3.3f", health);
    if (health < health_threshold) {
        this->callFail();
    }
}

double Watchdog::computeHealth() {
    double total(array.size());
    double accu = 0;
    for (const auto pair : array) {
        accu += pair.second;
    }
    return accu / total;
}

void Watchdog::setFailCallback(ros::TimerCallback callback) {
    failCallback_ = callback;
}

void Watchdog::callFail() {
    ROS_ERROR("failed health check");
    ros::TimerEvent e;
    failCallback_(e);
}


// ----------------------------------------------------------------------------
// Time in milliseconds between two sequential time points (t2 > t1)
double milliseconds_between(system_clock::time_point const &t1, system_clock::time_point const &t2) {
    return ((duration < double, std::milli >)(t2 - t1)).count();
}


string enum2str(int en, map<string, int> mymap, string errmsg) {
    for (const auto &pair : mymap ) {
        if (pair.second == en) return const_cast<char*>(pair.first.c_str());
    }
    throw std::invalid_argument(errmsg);
}

string enum2str(int en, map<string, int> mymap) {
    return enum2str(en, mymap, "Casting enum to string failed.");
}

string mac2str(uint32_t LOW, uint32_t HIGH) {
    std::stringstream ss;
    ss <<std::hex << std::setfill('0') \
     << std::setw(2) << (((HIGH) >> 8) & 0xff) << ":" \
     << std::setw(2) << (((HIGH) >> 0) & 0xff) << ":" \
     << std::setw(2) << (((LOW) >> 24) & 0xff) << ":" \
     << std::setw(2) << (((LOW) >> 16) & 0xff) << ":" \
     << std::setw(2) << (((LOW) >> 8) & 0xff) << ":" \
     << std::setw(2) << (((LOW) >> 0) & 0xff);
    return ss.str();
}
// ----------------------------------------------------------------------------
Trigger::Trigger() {

}

void Trigger::set_fire_state(bool state) {
    ROS_INFO("SET FIRE STATE: %d", (int) state);
    setToFire = state;
};

void Trigger::fire_cond(bool condition) {
    if (condition) {
        ROS_INFO3("Software Triggering frame acquisition...");
        fire();
    }
}

void Trigger::fire_cond() {
    fire_cond(setToFire);
}

/** Send a command to the GenApi to issue a software trigger
 */
void Trigger::fire() {
    ROS_INFO1(GRN "[!] -- SOFTWARE TRIGGER -- [!]" NC);
    (*trigger_cmd_node_ptr)();
    ready = true;
}

/** Bind the node function pointer to the object to simplify the callback
 * @param cam_node_map_ptr
 * @param trigger_node_name
 */
void Trigger::bind_node_action(GenApi::CNodeMapRef *cam_node_map_ptr, const char* trigger_node_name) {
    // "TriggerSoftware"
    trigger_cmd_node_ptr = cam_node_map_ptr->_GetNode(trigger_node_name);
}

/** Hold up processing until trigger received */
void Trigger::spin_until_trigger() {
    if (!setToFire) return;
    while (!ready.load()) {
        ros::spinOnce();
    }
    ready = false;
}
// ----------------------------------------------------------------------------

string parse_validate_arg(ros::NodeHandle const &nh, string param, map<string, int> validmap, bool &failed) {
    string tmp_str;
    if( ! nh.getParam( param, tmp_str ) ) {
        failed = true;
        ROS_ERROR( "No param %s of type 'string' provided.", param.c_str() );
    }

    auto it = validmap.find(tmp_str);
    if (it != validmap.end()) {
        return tmp_str;
    } else {
        failed = true;
        ROS_ERROR("Unsupported value for %s : %s.\n", param.c_str(), tmp_str.c_str());
        return "";
    }
}

bool is_number(string entry) {
    int i = 0;
    int bad = 0;
    while (entry[i]) {
        if (! isdigit(entry[i])) {
	    bad++;
	}
	i++;
    }
    if (bad > 0) {
        return false;
    }
    return true;
}

// I have no idea why it's failing with with default parameters so for now just regular overloading
int parse_pos_int(ros::NodeHandle const &nh, string param, bool &failed) {
    return parse_pos_int(nh, param, failed, 1);
}

int parse_pos_int(ros::NodeHandle const &nh, string param, bool &failed, int minval) {
    int tmp_int;
    if( ! nh.getParam( param, tmp_int ) ) {
        failed = true;
        ROS_ERROR( "No param %s of type 'int' provided.", param.c_str() );
    } else if (tmp_int < minval) {
        failed = true;
        ROS_ERROR("%s=%d. It must be greater than or equal to %d.", param.c_str(), tmp_int, minval );
    } else {
        return tmp_int;
    }
}

float parse_pos_float(ros::NodeHandle const &nh, string param, bool &failed) {
    float tmp;
    if( ! nh.getParam( param, tmp ) ) {
        failed = true;
        ROS_ERROR( "No param %s provided.", param.c_str() );
    } else if (tmp <= 0) {
        failed = true;
        ROS_ERROR("%s must be a positive value.", param.c_str() );
    } else {
        return tmp;
    }
}


/** Casts a string argument to its numerical value, based on a map. This is a shim and should be replaced eventually.
 * Assumes already validated!
 * @param arg
 * @param validmap
 * @return
 */
int arg2enum(string arg, map<string, int> validmap) {
    auto it = validmap.find(arg);
    if (it != validmap.end()) {
        return it->second;
    }
    return 0;
}

/**
 * Get the IPv4 address as a UINT32 value for use with the GEV API.
 *
 * \throws std::invalid_argument We have no IPv4 address or failed to parse
 *                               it into the UINT32 type.
 *
 * \returns Unsigned 32-bit integer form of the IPv4 address. (i.e.
 *          "b1b2.b3b4.b5b6.b7b8" ends up represented in the integer bytes
 *          in the order of b8b7b6b5b4b3b2b1, where "b1" means "byte 1")
 */
UINT32 ip_string_to_uint(string camera_ip_addr) {
    unsigned char ipbytes[4];
    int parts = sscanf(camera_ip_addr.c_str(), "%hhu.%hhu.%hhu.%hhu",
                       &ipbytes[3], &ipbytes[2], &ipbytes[1], &ipbytes[0]);
    if (parts != 4) {
        ROS_ERROR_STREAM("Error parsing IP address: \"" << camera_ip_addr << "\".");
        ROS_ERROR_STREAM("  sscanf got " << parts << " parts.");
        throw std::invalid_argument("Could not parse IPv4 address");
    }
    return ipbytes[3] << 24 | ipbytes[2] << 16 | ipbytes[1] << 8 | ipbytes[0];

}

/** Find an IP camera based on arguments passed in from roslaunch at runtime
 *
 * @param camera - Container for search arguments, passed in from node handle or param server
 * @return IP address of successfully located camera (success) or 0 (failed to find
 */
uint32_t locate_camera(CameraIdentifier &camera) {
    if ( ((!camera.username.empty() + !camera.serial.empty() + !camera.ip_addr.empty()) > 1)
            ) {
        ROS_ERROR("Can only specify one kind of camera identifier");
        return 0;
    }

    if (!camera.ip_addr.empty()) {
        return ip_string_to_uint(camera.ip_addr);
    } else if (!camera.manufacturer.empty()) {
        return find_camera_by_attribute(camera.manufacturer, GA_MANUFACTURER);
    } else if (!camera.serial.empty()) {
        return find_camera_by_attribute(camera.serial, GA_SERIAL);
    } else if (!camera.username.empty()) {
        return find_camera_by_attribute(camera.username, GA_USERNAME);
    } else if (!camera.mac.empty()) {
        return find_camera_by_attribute(camera.mac, GA_MAC);
    }

    return 0;
}

void print_camera_id(CameraIdentifier &camera) {
    std::cout << "Camera args: \nUsername : " << camera.username << "\n"
                                "Mfgr     : " << camera.manufacturer << "\n"
                                "Serial   : " << camera.serial << "\n"
                                "IP Addr  : " << camera.ip_addr << "\n"
                                "MAC      : " << camera.mac << "\n";
}


/** Find a camera based on Genicam attributes
 *
 * @param filter String to search with
 * @param e_attr Enum for specifying which field to search against
 * @return
 */
uint32_t find_camera_by_attribute(std::string filter, GeniAttributes e_attr) {
    ROS_INFO("GevGetCameraList: Acquiring camera information...");
    GEV_DEVICE_INTERFACE camera_interfaces[MAX_CAMERAS_PER_NETIF];
    GEV_DEVICE_INTERFACE *match = camera_interfaces;
    int num_interfaces(0);
    int match_count = 0;
    std::string tmp;
    GevGetCameraList(camera_interfaces, MAX_CAMERAS, &num_interfaces);
    ROS_INFO("GevGetCameraList: found %d device interfaces", num_interfaces);

    for (int i = 0; i < num_interfaces; ++i) {
        switch (e_attr) {
            case GA_MANUFACTURER:
                tmp = camera_interfaces[i].manufacturer;
                break;
            case GA_MODEL:
                tmp = camera_interfaces[i].model;
                break;
            case GA_SERIAL:
                tmp = camera_interfaces[i].serial;
                break;
            case GA_USERNAME:
                tmp = camera_interfaces[i].username;
                break;
            case GA_MAC:
                tmp = mac2str(camera_interfaces[i].macLow, camera_interfaces[i].macHigh);
                ROS_INFO("Converted MAC: %s", tmp.c_str());
                break;
        }

        if (std::string::npos != tmp.find(filter)) {
            ROS_GREEN("Found matching camera");
            match = &camera_interfaces[i];
            match_count++;
            ROS_INFO("%ld %s", tmp.find(filter), match->username);
        }
    }
    if (match_count == 1) {
        return match->ipAddr;
    }
    if (match_count == 0) {
        ROS_ERROR("No matching cameras found.");
    } else {
        ROS_ERROR("Two or more matching cameras found, connection ambiguous.");
    }
    ROS_INFO("Available GEV camera interfaces:");
    log_camera_interfaces(camera_interfaces, num_interfaces);
    throw CameraLocateError();
    return 0;
}

// ----------------------------------------------------------------------------
// Perform safe shutdown operations before passing through error code.
int safe_exit(int exit_code, GEV_CAMERA_HANDLE cam_handle = NULL) {
    // Close camera connection if given a non-NULL handle.
    if (cam_handle != NULL) {
        // Stop/abort image transfer
        GEV_STATUS s(GevAbortImageTransfer(cam_handle));
        WARN_ON_FAILURE(GevAbortImageTransfer, s, GEVLIB_OK);

        s = GevFreeImageTransfer(cam_handle);
        WARN_ON_FAILURE(GevFreeImageTransfer, s, GEVLIB_OK);

        s = GevCloseCamera(&cam_handle);
        WARN_ON_FAILURE(GevCloseCamera, s, GEVLIB_OK);
    }

    // Uninitialize API
    {
        GEV_STATUS s(GevApiUninitialize());
        WARN_ON_FAILURE(GevApiUninitialize, s, GEVLIB_OK);

        // The example code also had this but didn't document why this is here
        // without the _InitSocketAPI() call.
        //_CloseSocketAPI();
    }

    ros::shutdown();

    return exit_code;
}

// ----------------------------------------------------------------------------
// ROS Info dump the options struct contents
void log_config_options(std::string line_prefix, GEVLIB_CONFIG_OPTIONS const &options) {
    std::string log_descr;
    switch (options.logLevel) {
        case GEV_LOG_LEVEL_OFF:
            log_descr = "OFF";
            break;
        case GEV_LOG_LEVEL_NORMAL:  // same as GEV_LOG_LEVEL_ERRORS
            log_descr = "NORMAL";
            break;
        case GEV_LOG_LEVEL_WARNINGS:
            log_descr = "WARNINGS";
            break;
        case GEV_LOG_LEVEL_DEBUG:
            log_descr = "DEBUG";
            break;
        case GEV_LOG_LEVEL_TRACE:
            log_descr = "TRACE";
            break;
        default:
            log_descr = "UNKNOWN";
            break;
    }

    if (G_INFO_VERBOSITY >= 1) {
        ROS_INFO_STREAM(line_prefix << "version ............. : " << options.version);
        ROS_INFO_STREAM(line_prefix << "GEV logLevel ........ : " << log_descr);
        ROS_INFO_STREAM(line_prefix << "numRetries .......... : " << options.numRetries);
        ROS_INFO_STREAM(line_prefix << "command_timeout_ms .. : " << options.command_timeout_ms);
        ROS_INFO_STREAM(line_prefix << "discovery_timeout_ms  : " << options.discovery_timeout_ms);
        ROS_INFO_STREAM(line_prefix << "enumeration_port .... : " << options.enumeration_port);
        ROS_INFO_STREAM(line_prefix << "gvcp_port_range_start : " << options.gvcp_port_range_start);
        ROS_INFO_STREAM(line_prefix << "gvcp_port_range_end . : " << options.gvcp_port_range_end);
    }
}

// ----------------------------------------------------------------------------
// ROS Info dump of camera options struct
void log_camera_options(std::string line_prefix, GEV_CAMERA_OPTIONS const &opts) {
    const char *prefix = line_prefix.c_str();
    if (G_INFO_VERBOSITY >= 1) {
        ROS_INFO("%snumRetries ............ : %u", prefix, opts.numRetries);
        ROS_INFO("%scommand_timeout_ms .... : %u", prefix, opts.command_timeout_ms);
        ROS_INFO("%sheartbeat_timeout_ms .. : %u", prefix, opts.heartbeat_timeout_ms);
        ROS_INFO("%sstreamPktSize ......... : %u", prefix, opts.streamPktSize);
        ROS_INFO("%sstreamPktDelay ........ : %u", prefix, opts.streamPktDelay);
        ROS_INFO("%sstreamNumFramesBuffered : %u", prefix, opts.streamNumFramesBuffered);
        ROS_INFO("%sstreamMemoryLimitMax .. : %u", prefix, opts.streamMemoryLimitMax);
        ROS_INFO("%sstreamMaxPacketResends  : %u", prefix, opts.streamMaxPacketResends);
        ROS_INFO("%sstreamFrame_timeout_ms  : %u", prefix, opts.streamFrame_timeout_ms);
        ROS_INFO("%sstreamThreadAffinity .. : %d", prefix, opts.streamThreadAffinity);
        ROS_INFO("%sserverThreadAffinity .. : %d", prefix, opts.serverThreadAffinity);
        ROS_INFO("%smsgChannel_timeout_ms . : %u", prefix, opts.msgChannel_timeout_ms);
    }
}

// ----------------------------------------------------------------------------
// Log single camera interface/info structure.
//
// NOTE: GEV_DEVICE_INTERFACE and GEV_CAMERA_INFO are the same structure.
void log_camera_interface(GEV_DEVICE_INTERFACE const &itf, std::string line_prefix = "") {
    ROS_INFO_STREAM(line_prefix << "manufacturer : " << itf.manufacturer);
    ROS_INFO_STREAM(line_prefix << "ipAddr ..... : " << IP_ADDR(itf.ipAddr));
    if (G_INFO_VERBOSITY >= 1) {
        ROS_INFO_STREAM(line_prefix << "model ...... : " << itf.model);
        ROS_INFO_STREAM(line_prefix << "serial ..... : " << itf.serial);
        ROS_INFO_STREAM(line_prefix << "version .... : " << itf.version);
        ROS_INFO_STREAM(line_prefix << "username ... : " << itf.username);
        ROS_INFO_STREAM(line_prefix << "mac ........ : " << MAC_ADDR(itf.macLow, itf.macHigh));
    }
    }

// ----------------------------------------------------------------------------
// Log to ROS_INFO the provided GEV camera interface structures.
//
// NOTE: GEV_DEVICE_INTERFACE and GEV_CAMERA_INFO are the same structure.
void log_camera_interfaces(GEV_DEVICE_INTERFACE const *interfaces, int const &num_interfaces) {
    ROS_INFO("----------------------------------------");
    GEV_DEVICE_INTERFACE const *itf;
    for (int i = 0; i < num_interfaces; ++i) {
        // See gevapi.h for struct definition.
        itf = &interfaces[i];

        ROS_INFO_STREAM("Interface Index " << i);
        log_camera_interface(*itf);
        ROS_INFO("----------------------------------------");
    }
}

// ----------------------------------------------------------------------------
// Discover available camera interfaces and log their information
void discover_and_log_camera_interfaces() {
    // Get the IP addresses of attached network cards.
    ROS_INFO("GevGetCameraList: Acquiring camera information...");
    GEV_DEVICE_INTERFACE camera_interfaces[MAX_CAMERAS_PER_NETIF];
    int num_interfaces(0);
    GevGetCameraList(camera_interfaces, MAX_CAMERAS, &num_interfaces);
    ROS_INFO("GevGetCameraList: found %d device interfaces", num_interfaces);

    ROS_INFO("Available GEV camera interfaces:");
    log_camera_interfaces(camera_interfaces, num_interfaces);
}

// ----------------------------------------------------------------------------
// Connect to a camera by input serial number, handling bad status logging.
// \throws CameraConnectionError Failed to connect to camera.
void gev_open_by_serial_wrapper(std::string serial, GEV_CAMERA_HANDLE &cam_handle) {
    const char* sn = serial.c_str();
    ROS_INFO_STREAM("Connecting to camera with serial \"" << sn << "\"...");
    GEV_STATUS conn_status(GevOpenCameraBySN((char*) sn, GevExclusiveMode,
                                             &cam_handle));

    if (conn_status == (GEV_STATUS) GEVLIB_OK) {
        ROS_INFO("GevOpenCameraBySN OK");
    } else {
        ROS_ERROR_STREAM("GevOpenCameraBySN " << decode_sdk_status(conn_status));
        throw CameraConnectionError();
    }
}

// ----------------------------------------------------------------------------
// Connect to a camera by IPv4 address, handling bad status logging.
void gev_open_by_ip_addr_wrapper(UINT32 ip_addr, GEV_CAMERA_HANDLE &cam_handle) {
    ROS_INFO_STREAM("Connecting to camera by IPv4 address: " << IP_ADDR(ip_addr));

    GEV_STATUS conn_status(GevOpenCameraByAddress(ip_addr, GevExclusiveMode,
                                                  &cam_handle));
    if (conn_status == (GEV_STATUS) GEVLIB_OK) {
        ROS_INFO("GevOpenCameraByAddress OK");
    } else {
        ROS_ERROR_STREAM("GevOpenCameraByAddress " << decode_sdk_status(conn_status));
        ROS_ERROR_STREAM("Tried to open by IP, but failed. Is the camera in use?");
        throw CameraInUseError();
    }
}

bool push_node_ptr(GenApi::CNodeMapRef *feature_node_map_ptr, string param, string value) {
    GenApi::CEnumerationPtr node_ptr;
    ROS_INFO1("-- \"%s\"", param.c_str());
    node_ptr = feature_node_map_ptr->_GetNode(param.c_str());
    if (node_ptr.IsValid()) {
        ROS_INFO1("   Prev value: %s", (**node_ptr).c_str());
        (*node_ptr) = value.c_str();
        ROS_INFO1("   New value : %s", (**node_ptr).c_str());
    } else {
        ROS_WARN("   !! Feature \"%s\" not available.", param.c_str());
    }
}

bool push_node_ptr(GenApi::CNodeMapRef *feature_node_map_ptr, string param, int value) {
    GenApi::CIntegerPtr node_ptr;
//    GenApi::CEnumerationPtr node_ptr;
    ROS_INFO1("-- \"%s\"", param.c_str());
    node_ptr = feature_node_map_ptr->_GetNode( param.c_str() );
    if( node_ptr.IsValid() ) {
        ROS_INFO1( "   Prev value: %d", (int)node_ptr->GetValue() );
        // Range between 0-255
        node_ptr->SetValue( value );
        ROS_INFO1( "   New value : %d", (int)node_ptr->GetValue() );
    } else {
        ROS_WARN("   !! Feature \"%s\" not available.", param.c_str());
    }
}

bool push_node_ptr(GenApi::CNodeMapRef *feature_node_map_ptr, string param, float value) {
    GenApi::CFloatPtr node_ptr;
    ROS_INFO1("-- \"%s\"", param.c_str());
    node_ptr = feature_node_map_ptr->_GetNode(param.c_str());
    if (node_ptr.IsValid()) {
        ROS_INFO1("   Prev value: %f", (float) node_ptr->GetValue());
        node_ptr->SetValue(value);
        ROS_INFO1("   New value : %f", (float) node_ptr->GetValue());
	return true;
    } else {
        ROS_WARN("   !! Feature \"%s\" not available.", param.c_str());
	return false;
    }
}
bool get_node_val(GenApi::CNodeMapRef *feature_node_map_ptr, string param) {
    GenApi::CEnumerationPtr node_ptr;
    std::string value;
    ROS_INFO1("-- \"%s\"", param.c_str());
    node_ptr = feature_node_map_ptr->_GetNode(param.c_str());
    if (node_ptr.IsValid()) {
        ROS_INFO1("   Curr value: %s", (**node_ptr).c_str());
    } else {
        ROS_WARN("   !! Feature \"%s\" not available.", param.c_str());
    }
}

bool say_attr_info(GenApi::CNodeMapRef *feature_node_map_ptr, std::string param) {
    GenApi::NodeList_t nodeList;
    feature_node_map_ptr->_GetNodes(nodeList);
    for (int i=0; i < nodeList.size(); i++) {
        std::cout << nodeList[i]->GetName() << std::endl ;
    }
}
bool get_attr_list(GEV_CAMERA_HANDLE &camera_handle, std::vector<std::string> &paramList) {
    GenApi::CNodeMapRef *cam_node_map_ptr = NULL;
    GenApi::NodeList_t nodeList;
    ROS_GREEN("Loading XML from camera");
    GEV_STATUS s(GevInitGenICamXMLFeatures(camera_handle, true));
    RETURN_ON_FAILURE(GevInitGenICamXMLFeatures,
                      s, GEVLIB_OK, 1, camera_handle);
    cam_node_map_ptr = static_cast< GenApi::CNodeMapRef * >(
            GevGetFeatureNodeMap(camera_handle)
    );
    cam_node_map_ptr->_GetNodes(nodeList);
    for (int i=0; i < nodeList.size(); i++) {
        std::string val{nodeList[i]->GetName()};
        std::cout << val<< std::endl ;
        paramList.emplace_back(val);
    }
}


bool validate_image(GEV_BUFFER_OBJECT *img_buff_obj_ptr, CameraImageInfo *cam_image_info) {

    if (img_buff_obj_ptr == NULL) {
        // img_buff_obj_ptr may be NULL, indicating a timeout in frame retrieval.
        ROS_WARN1("Continuing after frame timeout ...");
        return FALSE;
    } else if (img_buff_obj_ptr->status != 0) {
        // Something else went wrong with the image acquisition.
        ROS_WARN_STREAM("Bad image buffer status: "
                                << decode_sdk_status(img_buff_obj_ptr->status));
        return FALSE;
    }
        /** Checking that received image does not deviate from initially retrieved camera image parameters. */
    else if (img_buff_obj_ptr->w != cam_image_info->width
             || img_buff_obj_ptr->h != cam_image_info->height
             || img_buff_obj_ptr->format != cam_image_info->pixel_format
             || img_buff_obj_ptr->d != cam_image_info->depth()) {
        ROS_WARN("Received image dimensions/format differed from initial "
                 "settings.");
        ROS_WARN("Expected: H:%d x W:%d x D:%d (format: %s)",
                 cam_image_info->height, cam_image_info->width,
                 cam_image_info->depth(),
                 decode_pixel_format(cam_image_info->pixel_format));
        ROS_WARN("Actual  : H:%d x W:%d x D:%d (format: %s)",
                 img_buff_obj_ptr->h, img_buff_obj_ptr->w,
                 img_buff_obj_ptr->d,
                 decode_pixel_format(img_buff_obj_ptr->format));
        return FALSE;
    }
    return TRUE;
}

bool getFeatureValueAsString(GEV_CAMERA_HANDLE &handle, const std::string &name, std::string &out){
    int feature_type;
    return getFeatureValueAsString(handle, name, out, &feature_type);
}
bool getFeatureValueAsString(GEV_CAMERA_HANDLE &handle, const std::string &name, std::string &out, int *feature_type){
/// GEVLIB_ERROR_NO_SPACE
    size_t sz = 32;
    GEV_STATUS status;
    while (true) {
        out.resize(sz);
        status = GevGetFeatureValueAsString(handle, name.c_str(), feature_type, (int) out.size(), &out[0]);
        if (status == GEVLIB_OK) {
            return true;
        } else if (status == GEVLIB_ERROR_NO_SPACE) {
            sz *= 2;
        } else {
            WARN_ON_FAILURE(GevGetFeatureValueAsString, status, GEVLIB_OK);
            return false;
        }
        if (sz > 65537) {
            ROS_ERROR("Tried to allocate too much memory for feature %s", name.c_str());
            return false;
        }
    }
}

bool tellFeatureValue(GEV_CAMERA_HANDLE &handle, const std::string &name) {
    std::string tmp;
    int feature_type;
    bool success = getFeatureValueAsString(handle, name, tmp, &feature_type);
    if (success) {
        ROS_INFO("GEV> %s[%d]: %s", name.c_str(), feature_type, tmp.c_str());
    } else {
        ROS_ERROR("Failed to read string %s", name.c_str());
    }
    return success;
}


bool tryExecute(GenApi::CNodeMapRef *feature_node_map_ptr, const std::string &name) {
        GenApi::CCommandPtr node_ptr;
//    GenApi::CEnumerationPtr node_ptr;
        ROS_INFO1("-- Command \"%s\"", name.c_str());
        node_ptr = feature_node_map_ptr->_GetNode( name.c_str() );
        if( node_ptr.IsValid() ) {
            // Range between 0-255
            node_ptr->Execute();
            ROS_INFO1( "   Executed : %d", node_ptr->IsDone() );
	    return true;
        } else {
            ROS_WARN("   !! Feature \"%s\" not available.", name.c_str());
	    return false;
        }
	return false;
}


/**
 * this shows how to bootstrap a feature map pointer from a camera handle.
 * you have to set GevInitGenICamXMLFeatures with 'true' else it crashes
 * @param camera_handle
 * @return
 */
bool tryBootstrap(GEV_CAMERA_HANDLE &camera_handle) {
    ROS_WARN("trying to bootstrap");
    GenApi::CNodeMapRef *cam_node_map_ptr = NULL;
    ROS_GREEN("Loading XML from camera");
    GEV_STATUS s(GevInitGenICamXMLFeatures(camera_handle, true));
    RETURN_ON_FAILURE(GevInitGenICamXMLFeatures,
                      s, GEVLIB_OK, 1, camera_handle);
    cam_node_map_ptr = static_cast< GenApi::CNodeMapRef * >(
            GevGetFeatureNodeMap(camera_handle)
    );

    ROS_INFO("looks ok ");
    get_node_val(cam_node_map_ptr, "TriggerMode");
    ROS_WARN("success   bootstrap");
}


bool tryNucCam(GEV_CAMERA_HANDLE &camera_handle) {
    GenApi::CNodeMapRef *cam_node_map_ptr = NULL;
    ROS_GREEN("Loading XML from camera");
    //GEV_STATUS s(GevInitGenICamXMLFeatures(camera_handle, true));
    //RETURN_ON_FAILURE(GevInitGenICamXMLFeatures,
    //                  s, GEVLIB_OK, 1, camera_handle);
    cam_node_map_ptr = static_cast< GenApi::CNodeMapRef * >(
            GevGetFeatureNodeMap(camera_handle)
    );

    return tryNuc(cam_node_map_ptr);
}


bool tryNuc(GenApi::CNodeMapRef *feature_node_map_ptr) {
    get_node_val(feature_node_map_ptr, "FlagState");
    const std::string name = "CorrectionAutoPerform";
    bool success = tryExecute(feature_node_map_ptr, name);
    get_node_val(feature_node_map_ptr, "FlagState"); 
    //get_node_val(feature_node_map_ptr, "CorrectionAutoInProgress"); 
    return success;
}
