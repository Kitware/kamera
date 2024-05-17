#pragma once
#ifndef PHASE_ONE_UTILS_H
#define PHASE_ONE_UTILS_H
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
// ROS
#include <ros/ros.h>
// Custom
#include <custom_msgs/GSOF_EVT.h>


const ros::Duration ZERO_DURATION(0);
// Used when you need something to trigger immediately,
// but where a truly zero duration may do weird things, e.g. div/0
const ros::Duration ALMOST_INSTANT{0, 10};
const ros::Duration MICROSECOND{0, 1000};

// Get the absolute value of a ROS duration
ros::Duration rosabs(ros::Duration dur);

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
    EventCache(ros::Duration tol, ros::Duration delay);

    // Set the maximum amount of time allowed between an event message
    // time received  and an image message time received to allow a match
    // between the 2. Allows for slop in estimate of delay 
    void set_tolerance(ros::Duration const &tolerance);

    // Set the expected delay between an image trigger and the time it is
    // actually received (including exposure, network transfer, etc.).
    // Close to 0 for small images (e.g. IR), up to a second for longer
    // exposure large imagery (e.g. Phase One)
    void set_delay(double delay);

    void set_delay(ros::Duration const &delay); 

    // Insert event message `msg` at time `t` into this map.
    void push_back(ros::Time const &t, const boost::shared_ptr<custom_msgs::GSOF_EVT const>& msg);

    // Return size of this cache
    int size();

    // Print out all headers in this cache
    void show();

    // Search this cache for an event closest to `image_time`, the system
    // time the image was received.
    // If found, change the header `head` to the GPS time of the event and 
    // return true. If `remove_when_found`, delete cached event upon
    // a successful find.
    bool search(ros::Time image_time, std_msgs::Header &head, bool remove_when_found);

    bool search(ros::Time image_time, std_msgs::Header &head) {
        return search(image_time, head, true);
    }
    
    // Remove all events older than `stale_time` from this cache.
    void purge();
    
    private:
        // Tolerance allowed in the expected delay
        ros::Duration tol;
        // Expected delay from event to image
        ros::Duration delay;
        // Set time to remove messages older than when `purge` is called
        ros::Duration stale_time{5};
        // Lock for thread safety
        std::mutex mutex_;
        // Data structure containing all event messages mapped to their
        // sys_time (time they were published)
        std::map<ros::Time, custom_msgs::GSOF_EVT> event_map;
};
        

// This function takes in a ROS standard list of params, delimited by:
// param1=val1,param2=val2,param3=val3,...,paramN=valN
// and returns a map of {param1: val1, param2: val2, etc.}
std::map<std::string, std::string> parseParams(std::string parameters);
        
// Takes in a filename, returns a vector of strings, each one mapping
// to a line in the given file.
std::vector<std::string> loadFile(std::string filename);


#endif //PHASE_ONE_UTILS_H