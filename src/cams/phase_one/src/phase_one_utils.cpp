#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
// ROS
#include <ros/ros.h>
// Custom
#include <custom_msgs/GSOF_EVT.h>
#include <phase_one/phase_one_utils.h>


ros::Duration rosabs(ros::Duration dur) {
    if (dur < ZERO_DURATION) 
        return -dur;
    return dur;                                                           
}


EventCache::EventCache() {
    ros::Duration tol(0.499);
    ros::Duration delay(0);
    this->set_tolerance(tol);
    this->set_delay(delay);
}

EventCache::EventCache(ros::Duration tol, ros::Duration delay) {
    this->set_tolerance(tol);
    this->set_delay(delay);
}

void EventCache::set_tolerance(ros::Duration const &tolerance) {
    std::lock_guard<std::mutex> guard(mutex_);
    this->tol = tolerance;
}

void EventCache::set_delay(ros::Duration const &delay) {
    std::lock_guard<std::mutex> guard(mutex_);
    this->delay = delay;
}

void EventCache::set_delay(double delay) {
    std::lock_guard<std::mutex> guard(mutex_);
    this->delay = ros::Duration{delay};
}

void EventCache::push_back(ros::Time const &t, const boost::shared_ptr<custom_msgs::GSOF_EVT const>& msg) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!msg->sys_time.isValid()) {
        ROS_ERROR("Zero/invalid sys_time encountered in EventCache::push_back()");
        return;
    }
    if (!msg->gps_time.isValid()) {
        ROS_ERROR("Zero/invalid gps_time encountered in EventCache::push_back()");
        return;
    }
    event_map.emplace(t, custom_msgs::GSOF_EVT(*msg));
}

int EventCache::size() {
    std::lock_guard<std::mutex> guard(mutex_);
    return (int) event_map.size();
}

void EventCache::show() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto it = event_map.begin(); it != event_map.end(); ++it) {
        std::cout << it->second.header;
    }
    std::cout << "\n---" << std::endl;
}

bool EventCache::search(ros::Time image_time, std_msgs::Header &head, bool remove_when_found) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!image_time.isValid()) {
        ROS_ERROR("zero/invalid image_time encountered in EventCache::search()");
        return false;
    }
    int count = 0;
    // we want the lowest corrected time
    ros::Duration best_time{999,999};
    ros::Time best_sys_time; // best matching system time, aka key
    ROS_INFO_STREAM("Image time is: " << image_time.toSec() << std::endl);
    std::map<ros::Time, custom_msgs::GSOF_EVT> event_map_copy(event_map);
    for (const auto pair: event_map_copy) {
        auto actual_delay = rosabs(pair.second.sys_time - image_time);
        auto corrected_delay = rosabs(actual_delay - delay);
        if (corrected_delay < tol) {
            std::cout << "img: " << image_time << " sys: " << pair.second.sys_time << " cdt: " << corrected_delay << " ad: " << actual_delay;
            count++;
            if (corrected_delay < best_time ) {
                best_time = corrected_delay;
                head.stamp = pair.second.gps_time;
                head.seq = pair.second.header.seq;
                best_sys_time = pair.second.sys_time;
                std::cout << " *";
            }
            std::cout << std::endl;
        } else {
            // std::cout << count << ": " << it->first.toSec() << " dt: " << corrected_delay << " ad: " << actual_delay << std::endl;
        }
    }
//        std::cout << "\n---" << std::endl;
    ROS_INFO_STREAM("Matched " << count << "/" << event_map_copy.size() << " time headers" << std::endl);
    if (count >= 1) {
        if (remove_when_found && best_sys_time.isValid()) {
            event_map.erase(best_sys_time);
        }
        return true;
    }
    return false;
}

void EventCache::purge() {
    std::lock_guard<std::mutex> guard(mutex_);
    auto now = ros::Time::now();
    std::map<ros::Time, custom_msgs::GSOF_EVT> event_map_copy(event_map);
    for ( const auto pair : event_map_copy ) {
        auto age = now - pair.second.sys_time ;
        if (age > stale_time) {
            event_map.erase(pair.first);
        }
    }
}


std::map<std::string, std::string> parseParams(std::string parameters) {
    // Iterate through parameters organized by name=value, separated by
    // commas. Return a map of parameters to values.
    std::map<std::string, std::string> param_to_value;
    std::string delimiter1 = ",";
    std::string delimiter2 = "=";
    size_t pos = 0;
    std::string token;
    std::string name;
    std::string value;
    // Always run at least once even if there's no delimiter in request
    while ((pos = parameters.find(delimiter1)) != std::string::npos) {
        token = parameters.substr(0, pos);
        name = token.substr(0, token.find(delimiter2));
        token.erase(0, token.find(delimiter2) + delimiter2.length());
        value = token;
        param_to_value[name] = value;
        parameters.erase(0, pos + delimiter1.length());
    }
    token = parameters;
    name = token.substr(0, token.find(delimiter2));
    token.erase(0, token.find(delimiter2) + delimiter2.length());
    value = token;
    param_to_value[name] = value;
    return param_to_value;
};
        
        
std::vector<std::string> loadFile(std::string filename) {
    std::vector<std::string> lines;
    std::ifstream inputFile(filename);
        // Check if the file exists and can be opened
    if (!inputFile.is_open()) {
        std::cout << "File " << filename << " does not exist or cannot be opened." << std::endl;
        return lines;
    } else {
        std::string line;
        while (std::getline(inputFile, line)) {
            lines.push_back(line);
        }
    }
    return lines;
};