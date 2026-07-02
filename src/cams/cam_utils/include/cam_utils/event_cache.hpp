#pragma once
#ifndef CAM_UTILS_EVENT_CACHE_HPP
#define CAM_UTILS_EVENT_CACHE_HPP

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <custom_msgs/msg/gsof_evt.hpp>

const rclcpp::Duration ZERO_DURATION(0, 0);
// Used when you need something to trigger immediately,
// but where a truly zero duration may do weird things, e.g. div/0
const rclcpp::Duration ALMOST_INSTANT{0, 10};
const rclcpp::Duration MICROSECOND{0, 1000};

// Get the absolute value of a ROS duration
rclcpp::Duration rosabs(rclcpp::Duration dur);

/* This class holds the events published from the INS upon each trigger,
   indexed by the system time that event was published. When an image message
   is received, this cache is `searched` for the closest event matching the
   system time of the image received. These are then fused, and the header
   of the image is changed to match the GPS time of the event, and the image
   is saved under that GPS time.
*/
class EventCache {
public:
    // specifying some sane defaults
    EventCache();
    EventCache(rclcpp::Duration tol, rclcpp::Duration delay);

    // Set the maximum amount of time allowed between an event message
    // time received  and an image message time received to allow a match
    // between the 2. Allows for slop in estimate of delay
    void set_tolerance(rclcpp::Duration const &tolerance);

    // Set the expected delay between an image trigger and the time it is
    // actually received (including exposure, network transfer, etc.).
    // Close to 0 for small images (e.g. IR), up to a second for longer
    // exposure large imagery (e.g. Phase One)
    void set_delay(double delay);

    void set_delay(rclcpp::Duration const &delay);

    // Insert event message `msg` at time `t` into this map.
    void push_back(rclcpp::Time const &t, const custom_msgs::msg::GsofEvt::ConstSharedPtr& msg);

    // Return size of this cache
    int size();

    // Print out all headers in this cache
    void show();

    // Search this cache for an event closest to `image_time`, the system
    // time the image was received.
    // If found, change the header `head` to the GPS time of the event, set
    // `event_num` (header.seq no longer exists in ROS2) and return true.
    // If `remove_when_found`, delete cached event upon a successful find.
    bool search(rclcpp::Time image_time, std_msgs::msg::Header &head,
                uint64_t &event_num, bool remove_when_found);

    bool search(rclcpp::Time image_time, std_msgs::msg::Header &head,
                uint64_t &event_num) {
        return search(image_time, head, event_num, true);
    }

    // Remove all events older than `stale_time` from this cache.
    void purge();

    private:
        // Tolerance allowed in the expected delay
        rclcpp::Duration tol{ZERO_DURATION};
        // Expected delay from event to image
        rclcpp::Duration delay{ZERO_DURATION};
        // Set time to remove messages older than when `purge` is called
        rclcpp::Duration stale_time{5, 0};
        // Clock for purge aging; matches node clocks (RCL_ROS_TIME)
        rclcpp::Clock clock_{RCL_ROS_TIME};
        // Lock for thread safety
        std::mutex mutex_;
        // Data structure containing all event messages mapped to their
        // sys_time (time they were published)
        std::map<rclcpp::Time, custom_msgs::msg::GsofEvt> event_map;
};


// This function takes in a ROS standard list of params, delimited by:
// param1=val1,param2=val2,param3=val3,...,paramN=valN
// and returns a map of {param1: val1, param2: val2, etc.}
std::map<std::string, std::string> parseParams(std::string parameters);

// Takes in a filename, returns a vector of strings, each one mapping
// to a line in the given file.
std::vector<std::string> loadFile(std::string filename);


#endif //CAM_UTILS_EVENT_CACHE_HPP
